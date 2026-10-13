package protocol

import (
	"encoding/json"
	"reflect"
	"strconv"
	"testing"
)

// This file pins the chat <-> responses REQUEST folds a LiteLLM v1.100.1 audit
// found diverging from the reference bridges. Each case was verified against
// litellm 1.100.1 directly (LiteLLMResponsesTransformationHandler.transform_request
// and LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request).

// A system turn with empty content still marks instructions as present, and
// the reference folds present-but-empty instructions into a system input item
// rather than an empty input the Responses API refuses. The turn's text rides
// verbatim, padding included.
func TestChatToResponsesEmptySystemFoldsIntoInput(t *testing.T) {
	for _, tc := range []struct{ content, wantText string }{
		{"", ""},
		{"   ", "   "},
	} {
		body := `{"model":"gpt-5","messages":[{"role":"system","content":"` + tc.content + `"}]}`
		out, err := ConvertRequest(Chat, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		input, ok := m["input"].([]any)
		if !ok || len(input) != 1 {
			t.Fatalf("content %q: input = %#v, want one system item", tc.content, m["input"])
		}
		item := input[0].(map[string]any)
		if item["role"] != "system" {
			t.Fatalf("content %q: item = %#v, want a system item", tc.content, item)
		}
		text := item["content"].([]any)[0].(map[string]any)["text"]
		if text != tc.wantText {
			t.Fatalf("content %q: text = %#v, want %#v", tc.content, text, tc.wantText)
		}
		if _, present := m["instructions"]; present {
			t.Fatalf("content %q: instructions should have folded into input: %#v", tc.content, m)
		}
	}

	// Non-empty text appends with a space whatever follows, so a trailing
	// EMPTY system turn leaves the separator on — the reference's own join.
	body := `{"model":"gpt-5","messages":[{"role":"system","content":"a"},{"role":"system","content":""}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	text := m["input"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	if text != "a " {
		t.Fatalf("two system turns folded to %#v, want \"a \"", text)
	}

	// With a user turn alongside, an empty system turn contributes no
	// instructions at all: the reference only sets instructions when the
	// accumulated string is truthy.
	body = `{"model":"gpt-5","messages":[{"role":"system","content":""},{"role":"user","content":"hi"}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["instructions"]; present {
		t.Fatalf("empty system + user: instructions = %#v, want none", m["instructions"])
	}
}

// The reference's chat -> responses bridge carries no mapping for a thinking
// budget: it drops the param rather than inventing a reasoning effort, so the
// budget rides nothing here either.
func TestChatToResponsesThinkingRidesNothing(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":4096}}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["reasoning"]; present {
		t.Fatalf("thinking must not become reasoning: %#v", m["reasoning"])
	}
}

// Only the effort vocabulary folds onto reasoning.effort; a string outside it
// is dropped rather than forwarded for the upstream to reject the request over.
func TestChatToResponsesReasoningEffortVocabulary(t *testing.T) {
	for _, effort := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		body := `{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"` + effort + `"}`
		out, err := ConvertRequest(Chat, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		reasoning, ok := m["reasoning"].(map[string]any)
		if !ok || reasoning["effort"] != effort {
			t.Fatalf("effort %q -> %#v, want it folded", effort, m["reasoning"])
		}
	}
	for _, effort := range []string{"very-high", ""} {
		body := `{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"` + effort + `"}`
		out, err := ConvertRequest(Chat, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if _, present := m["reasoning"]; present {
			t.Fatalf("effort %q must be dropped, got %#v", effort, m["reasoning"])
		}
	}
}

// The chat wire carries web search as a request param; the Responses wire
// carries it as a built-in member of tools, with the options folded into the
// tool's own fields and the tool joining any declared tools.
func TestChatToResponsesWebSearchOptionsBecomesTool(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"web_search_options":{"search_context_size":"low","user_location":{"type":"approximate","city":"SF"}}}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %#v, want the web_search tool", m["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "web_search" || tool["search_context_size"] != "low" {
		t.Fatalf("web_search tool = %#v", tool)
	}
	if loc, ok := tool["user_location"].(map[string]any); !ok || loc["city"] != "SF" {
		t.Fatalf("user_location = %#v, want it folded into the tool", tool["user_location"])
	}

	body = `{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"web_search_options":{"search_context_size":"medium"}}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools = m["tools"].([]any)
	if len(tools) != 2 || tools[1].(map[string]any)["type"] != "web_search" {
		t.Fatalf("tools = %#v, want the declared tool then web_search", m["tools"])
	}
}

// A tool_choice that names no tool is a pairing the chat wire rejects, so when
// conversion left no tools both fold away together — and an empty tools list
// must not survive as a `"tools": null` either.
func TestResponsesToChatDropsToolChoiceWithoutTools(t *testing.T) {
	for _, body := range []string{
		`{"model":"gpt-5","input":"x","tool_choice":"auto"}`,
		`{"model":"gpt-5","input":"x","tools":[],"tool_choice":"auto"}`,
	} {
		out, err := ConvertRequest(Responses, Chat, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if _, present := m["tool_choice"]; present {
			t.Fatalf("%s: tool_choice = %#v, want it folded away", body, m["tool_choice"])
		}
		if _, present := m["tools"]; present {
			t.Fatalf("%s: tools = %#v, want the key gone entirely", body, m["tools"])
		}
	}

	body := `{"model":"gpt-5","input":"x","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"tool_choice":{"type":"function","name":"f"}}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["tool_choice"]; !present {
		t.Fatalf("a request with a real tool keeps its tool_choice: %#v", m)
	}
	if tools := m["tools"].([]any); len(tools) != 1 {
		t.Fatalf("tools = %#v, want the declared function tool", m["tools"])
	}
}

// The reference's responses -> chat bridge takes metadata from its own call
// kwargs, never from the request body, so a body-level metadata rides nothing
// onto the chat wire.
func TestResponsesToChatDropsBodyMetadata(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","metadata":{"trace_id":"abc-123"}}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["metadata"]; present {
		t.Fatalf("metadata = %#v, want it not forwarded from the body", m["metadata"])
	}
}

// chatToolOutput runs a one-tool conversation through Chat -> Responses and
// returns the function_call_output's output the fold produced.
func chatToolOutput(t *testing.T, toolMsg string) any {
	t.Helper()
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"q"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},` + toolMsg + `]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	for _, item := range m["input"].([]any) {
		im := item.(map[string]any)
		if im["type"] == "function_call_output" {
			return im["output"]
		}
	}
	t.Fatalf("no function_call_output in %#v", m["input"])
	return nil
}

// A tool message's output is always a part list on the Responses wire: a bare
// string is one input_text part, an explicit null is an empty list, a MISSING
// content reads as the empty string, and an image or file part survives as its
// own part — flattening the content to text drops whatever the tool returned
// that was not text.
func TestChatToResponsesToolOutputIsAPartList(t *testing.T) {
	cases := []struct {
		name    string
		toolMsg string
		want    any
	}{
		{
			"string content",
			`{"role":"tool","tool_call_id":"c1","content":"42"}`,
			[]any{map[string]any{"type": "input_text", "text": "42"}},
		},
		{
			"missing content",
			`{"role":"tool","tool_call_id":"c1"}`,
			[]any{map[string]any{"type": "input_text", "text": ""}},
		},
		{
			"null content",
			`{"role":"tool","tool_call_id":"c1","content":null}`,
			[]any{},
		},
		{
			"text parts stay separate",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`,
			[]any{
				map[string]any{"type": "input_text", "text": "a"},
				map[string]any{"type": "input_text", "text": "b"},
			},
		},
		{
			"image part survives",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}`,
			[]any{map[string]any{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "auto"}},
		},
		{
			"image detail carries",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"image_url","image_url":{"url":"https://x/i.png","detail":"low"}}]}`,
			[]any{map[string]any{"type": "input_image", "image_url": "https://x/i.png", "detail": "low"}},
		},
		{
			"mixed text and image",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}`,
			[]any{
				map[string]any{"type": "input_text", "text": "see"},
				map[string]any{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "auto"},
			},
		},
		{
			"file part keeps its bytes",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBE"}}]}`,
			[]any{map[string]any{"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBE"}},
		},
		{
			"non-string content stringifies",
			`{"role":"tool","tool_call_id":"c1","content":42}`,
			[]any{map[string]any{"type": "input_text", "text": "42"}},
		},
		{
			"already-responses part passes through",
			`{"role":"tool","tool_call_id":"c1","content":[{"type":"input_text","text":"already"}]}`,
			[]any{map[string]any{"type": "input_text", "text": "already"}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := chatToolOutput(t, tc.toolMsg)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("output = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// A system turn whose content is a part list is not instruction text: it stays
// a system input item with its parts — flattening it to text would drop an
// image the prompt carried. A null content likewise becomes an item with an
// empty text part rather than instructions.
func TestChatToResponsesSystemArrayContentStaysStructured(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"system","content":[{"type":"text","text":"be terse"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]},{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["instructions"]; present {
		t.Fatalf("instructions = %#v, want none — the array content is not instruction text", m["instructions"])
	}
	item := m["input"].([]any)[0].(map[string]any)
	if item["role"] != "system" {
		t.Fatalf("first item = %#v, want the structured system item", item)
	}
	parts := item["content"].([]any)
	if parts[0].(map[string]any)["type"] != "input_text" || parts[0].(map[string]any)["text"] != "be terse" {
		t.Fatalf("text part = %#v", parts[0])
	}
	img := parts[1].(map[string]any)
	if img["type"] != "input_image" || img["image_url"] != "data:image/png;base64,AA" || img["detail"] != "auto" {
		t.Fatalf("image part = %#v", img)
	}

	body = `{"model":"gpt-5","messages":[{"role":"system","content":null},{"role":"user","content":"hi"}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	item = m["input"].([]any)[0].(map[string]any)
	if item["role"] != "system" || item["content"].([]any)[0].(map[string]any)["text"] != "" {
		t.Fatalf("null-content system item = %#v, want an item with an empty text part", item)
	}
}

// Only a system turn folds into instructions; a developer turn rides the input
// as its own developer-role item, its text reading as output_text.
func TestChatToResponsesDeveloperRidesItsOwnItem(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"u"},{"role":"system","content":"sys instr"},{"role":"developer","content":"dev instr"},{"role":"user","content":"v"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["instructions"] != "sys instr" {
		t.Fatalf("instructions = %#v, want only the system turn folded", m["instructions"])
	}
	input := m["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %#v, want user, developer item, user", input)
	}
	dev := input[1].(map[string]any)
	if dev["type"] != "message" || dev["role"] != "developer" {
		t.Fatalf("developer item = %#v", dev)
	}
	if part := dev["content"].([]any)[0].(map[string]any); part["type"] != "output_text" || part["text"] != "dev instr" {
		t.Fatalf("developer content = %#v", dev["content"])
	}
}

// Reasoning replays by id: stored reasoning_items carry the id the Responses
// API minted, and ride ahead of the turn. reasoning_content is free text the
// reference does not read — it rides nothing — and thinking_blocks stays the
// fallback for a turn that arrived over another surface.
func TestChatToResponsesReplaysReasoningItems(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"q"},{"role":"assistant","content":null,"reasoning_content":"free text","reasoning_items":[{"id":"rs_1","encrypted_content":"ENC","summary":[{"type":"summary_text","text":"s"}]}]},{"role":"user","content":"next"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %#v, want user, the reasoning replay, user", input)
	}
	ri := input[1].(map[string]any)
	if ri["type"] != "reasoning" || ri["id"] != "rs_1" || ri["encrypted_content"] != "ENC" {
		t.Fatalf("reasoning item = %#v", ri)
	}

	// reasoning_content alone rides nothing.
	body = `{"model":"gpt-5","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"answer","reasoning_content":"free"}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	for _, item := range m["input"].([]any) {
		if item.(map[string]any)["type"] == "reasoning" {
			t.Fatalf("reasoning_content must not become a reasoning item: %#v", item)
		}
	}

	// thinking_blocks remains the fallback.
	body = `{"model":"gpt-5","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","thinking_blocks":[{"type":"thinking","thinking":"tb","signature":"SIG"}]}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	first := m["input"].([]any)[1].(map[string]any)
	if first["type"] != "reasoning" {
		t.Fatalf("first item = %#v, want the thinking-block fallback", first)
	}
}

// An assistant turn's content keeps its shape: an image list is a message
// item with input_image parts (not dropped for having no text), and an empty
// string without calls still states an empty output_text message. Alongside
// tool_calls, empty content contributes no message at all.
func TestChatToResponsesAssistantContentKeepsItsShape(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]},{"role":"assistant","content":""}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %#v, want user, image message, empty message", input)
	}
	imgMsg := input[1].(map[string]any)
	if imgMsg["role"] != "assistant" {
		t.Fatalf("image message = %#v", imgMsg)
	}
	if part := imgMsg["content"].([]any)[0].(map[string]any); part["type"] != "input_image" || part["detail"] != "auto" {
		t.Fatalf("image part = %#v", part)
	}
	if part := input[2].(map[string]any)["content"].([]any)[0].(map[string]any); part["type"] != "output_text" || part["text"] != "" {
		t.Fatalf("empty message part = %#v", part)
	}

	// With tool_calls, empty content emits no message: the truthiness gate
	// the reference reads.
	body = `{"model":"gpt-5","messages":[{"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"fn","arguments":"{}"}}]}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	for _, item := range m["input"].([]any) {
		if item.(map[string]any)["type"] == "message" {
			t.Fatalf("empty content alongside tool_calls must emit no message: %#v", item)
		}
	}
}

// A custom tool call keeps its freeform shape: custom_tool_call with the input
// verbatim, and the tool reply a custom_tool_call_output whose string content
// stays a string rather than being wrapped in parts.
func TestChatToResponsesCustomToolCallKeepsItsShape(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"cc1","custom":{"name":"get_weather","input":"boston"}}]},{"role":"tool","tool_call_id":"cc1","content":"72f sunny"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %#v, want user, custom_tool_call, custom_tool_call_output", input)
	}
	call := input[1].(map[string]any)
	if call["type"] != "custom_tool_call" || call["name"] != "get_weather" || call["input"] != "boston" {
		t.Fatalf("custom call = %#v", call)
	}
	reply := input[2].(map[string]any)
	if reply["type"] != "custom_tool_call_output" || reply["output"] != "72f sunny" {
		t.Fatalf("custom reply = %#v", reply)
	}
}

// A function call states only what it carries: a missing arguments key leaves
// the call without arguments (not "{}"), and an empty-string arguments stays
// the empty string.
func TestChatToResponsesFunctionCallStatesOnlyWhatItHas(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"fn"}}]}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	call := m["input"].([]any)[0].(map[string]any)
	if _, present := call["arguments"]; present {
		t.Fatalf("arguments = %#v, want the key absent", call["arguments"])
	}

	body = `{"model":"gpt-5","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"fn","arguments":""}}]}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	call = m["input"].([]any)[0].(map[string]any)
	if call["arguments"] != "" {
		t.Fatalf("arguments = %#v, want the empty string verbatim", call["arguments"])
	}
}

// An explicit null content on a user turn rides nothing — the reference's
// content-is-not-None gate — while a MISSING content key reads as the empty
// string and still states an empty input_text message.
func TestChatToResponsesUserNullContentRidesNothing(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"a"},{"role":"user","content":null},{"role":"user","content":"b"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input = %#v, want the null turn to contribute nothing", input)
	}

	body = `{"model":"gpt-5","messages":[{"role":"user"}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input = m["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %#v, want the missing content to read as empty text", input)
	}
	if text := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "" {
		t.Fatalf("text = %#v, want the empty string", text)
	}
}

// A system turn's text rides instructions verbatim — padding kept, no trim —
// the way the reference concatenates.
func TestChatToResponsesInstructionsKeepPadding(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"system","content":"  a  "},{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["instructions"] != "  a  " {
		t.Fatalf("instructions = %#v, want the padding kept verbatim", m["instructions"])
	}
}

// A user turn's parts keep their shape: a bare string item is input_text, a
// file part rides as input_file, an already-Responses part passes through, a
// typeless part passes through as it stands, and an image part carries the
// detail it stated (defaulting to auto). An empty part list stays empty
// rather than being padded with an empty text part.
func TestChatToResponsesUserPartsKeepTheirShape(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":[
		"raw",
		{"text":"typeless"},
		{"type":"text","text":"t"},
		{"type":"image_url","image_url":{"url":"https://x/i.png","detail":"low"}},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},
		{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBE"}},
		{"type":"input_text","text":"already"}
	]}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	parts := m["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(parts) != 7 {
		t.Fatalf("parts = %#v, want all seven kept", parts)
	}
	if p := parts[0].(map[string]any); p["type"] != "input_text" || p["text"] != "raw" {
		t.Fatalf("bare string item = %#v", p)
	}
	if _, hasType := parts[1].(map[string]any)["type"]; hasType {
		t.Fatalf("typeless part must pass through verbatim: %#v", parts[1])
	}
	if p := parts[2].(map[string]any); p["type"] != "input_text" || p["text"] != "t" {
		t.Fatalf("text part = %#v", p)
	}
	if p := parts[3].(map[string]any); p["type"] != "input_image" || p["detail"] != "low" {
		t.Fatalf("image with stated detail = %#v", p)
	}
	if p := parts[4].(map[string]any); p["type"] != "input_image" || p["detail"] != "auto" {
		t.Fatalf("image without detail = %#v, want detail auto", p)
	}
	if p := parts[5].(map[string]any); p["type"] != "input_file" || p["filename"] != "a.pdf" {
		t.Fatalf("file part = %#v", p)
	}
	if p := parts[6].(map[string]any); p["type"] != "input_text" || p["text"] != "already" {
		t.Fatalf("already-responses part = %#v", p)
	}

	body = `{"model":"gpt-5","messages":[{"role":"user","content":[]}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if parts := m["input"].([]any)[0].(map[string]any)["content"].([]any); len(parts) != 0 {
		t.Fatalf("empty part list = %#v, want it to stay empty", parts)
	}
}

// A custom tool call replays as an assistant tool_call whose arguments wrap
// the freeform input the way the reference does, and every output shape — a
// custom_tool_call_output, a web_search_call, a computer_call_output, an
// Anthropic-style tool_result — folds to the tool message answering its call.
// An output naming no call contributes nothing.
func TestResponsesToChatFoldsCustomAndSearchItems(t *testing.T) {
	body := `{"model":"gpt-5","input":[
		{"type":"custom_tool_call","call_id":"cc1","name":"get_weather","input":"boston"},
		{"type":"custom_tool_call_output","call_id":"cc1","output":"72f sunny"},
		{"type":"web_search_call","call_id":"ws1"},
		{"type":"computer_call_output","call_id":"co1","output":"clicked"},
		{"type":"tool_result","call_id":"tr1","output":"ok"},
		{"type":"web_search_call"}
	]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("messages = %#v, want assistant + four tool messages", msgs)
	}
	call := msgs[0].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	fn := call["function"].(map[string]any)
	if call["id"] != "cc1" || fn["name"] != "get_weather" || fn["arguments"] != `{"content": "boston"}` {
		t.Fatalf("custom call = %#v", call)
	}
	want := []struct{ id, content string }{
		{"cc1", "72f sunny"},
		{"ws1", ""},
		{"co1", "clicked"},
		{"tr1", "ok"},
	}
	for i, w := range want {
		msg := msgs[i+1].(map[string]any)
		if msg["role"] != "tool" || msg["tool_call_id"] != w.id || msg["content"] != w.content {
			t.Fatalf("tool message %d = %#v, want %s answering %s", i, msg, w.content, w.id)
		}
	}
}

// A developer input item keeps its developer role on the chat wire — the
// reference forwards the role verbatim rather than rewriting it to system.
func TestResponsesToChatDeveloperItemKeepsRole(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"message","role":"user","content":"hi"},{"type":"message","role":"developer","content":"dev"}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v", msgs)
	}
	if dev := msgs[1].(map[string]any); dev["role"] != "developer" || dev["content"] != "dev" {
		t.Fatalf("developer message = %#v", dev)
	}
}

// A function_call item that names itself only by id still names a call: the
// id falls back into the tool_call the way the reference's call_id-or-id does.
func TestResponsesToChatFunctionCallIdFallback(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"function_call","id":"fc_123","name":"f","arguments":"{}"}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	call := m["messages"].([]any)[0].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if call["id"] != "fc_123" {
		t.Fatalf("tool_call id = %#v, want the item's id", call["id"])
	}
}

// A reasoning item's block content reads as its text — joined, not rendered as
// Go syntax — and an encrypted_content the reference itself wrote decodes back
// into the signed thinking blocks a replay needs.
func TestResponsesToChatReasoningContentAndBlocks(t *testing.T) {
	blocks := `[{"type":"thinking","thinking":"t1","signature":"SIG"}]`
	body := `{"model":"gpt-5","input":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"think-1"},{"type":"reasoning_text","text":"think-2"}],"encrypted_content":` + strconv.Quote(blocks) + `}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msg := m["messages"].([]any)[0].(map[string]any)
	if msg["reasoning_content"] != "think-1\nthink-2" {
		t.Fatalf("reasoning_content = %#v, want the joined block text", msg["reasoning_content"])
	}
	tbs, ok := msg["thinking_blocks"].([]any)
	if !ok || len(tbs) != 1 {
		t.Fatalf("thinking_blocks = %#v, want the decoded signed block", msg["thinking_blocks"])
	}
	if tb := tbs[0].(map[string]any); tb["signature"] != "SIG" || tb["thinking"] != "t1" {
		t.Fatalf("decoded block = %#v", tb)
	}

	// A whitespace-only string content reads as absent; the summary answers.
	body = `{"model":"gpt-5","input":[{"type":"reasoning","content":"   ","summary":[{"type":"summary_text","text":"sum"}]}]}`
	out, err = ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["messages"].([]any)[0].(map[string]any)["reasoning_content"] != "sum" {
		t.Fatalf("want the summary text to answer for blank content: %#v", m["messages"])
	}
}

// A reasoning item landing after a turn that already spoke stays its own
// trailing assistant — the reference merges reasoning forward only.
func TestResponsesToChatLateReasoningStandsAlone(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"message","role":"assistant","content":"ans"},{"type":"reasoning","content":"late"}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v, want the turn and a trailing reasoning assistant", msgs)
	}
	late := msgs[1].(map[string]any)
	if late["reasoning_content"] != "late" || late["content"] != nil {
		t.Fatalf("late reasoning = %#v", late)
	}
}

// Item content folds part by part: text parts stay separate chat text blocks
// (never joined into one string), a bare string item rides verbatim, a null
// text contributes nothing, an encrypted_content part reads as text, and an
// image always states its detail (auto when unstated).
func TestResponsesToChatItemContentFoldsPartByPart(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[
		"bare",
		{"type":"input_text","text":"a"},
		{"type":"input_text","text":"b"},
		{"type":"input_text","text":null},
		{"type":"summary_text","text":"s"},
		{"type":"encrypted_content","encrypted_content":"blob"},
		{"type":"input_image","image_url":"https://x/i.png"},
		{"type":"input_image","image_url":"https://x/j.png","detail":"low"}
	]}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	want := []any{
		"bare",
		map[string]any{"type": "text", "text": "a"},
		map[string]any{"type": "text", "text": "b"},
		map[string]any{"type": "text", "text": "s"},
		map[string]any{"type": "text", "text": "blob"},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x/i.png", "detail": "auto"}},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x/j.png", "detail": "low"}},
	}
	if !reflect.DeepEqual(content, want) {
		t.Fatalf("content = %#v, want %#v", content, want)
	}

	// A plain string content stays a string.
	body = `{"model":"gpt-5","input":[{"type":"message","role":"user","content":"plain"}]}`
	out, err = ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["messages"].([]any)[0].(map[string]any)["content"] != "plain" {
		t.Fatalf("string content = %#v, want it verbatim", m["messages"])
	}
}

// A file part rides the chat file object — the url falling back into file_id,
// the bytes in file_data — rather than being disguised as an image data URL.
func TestResponsesToChatFilePartRidesTheFileObject(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[
		{"type":"input_file","file_id":"file-123"},
		{"type":"input_file","file_url":"https://x/a.pdf"},
		{"type":"input_file","file_data":"data:application/pdf;base64,JVBE"},
		{"type":"input_file","filename":"n.pdf","cache_control":{"type":"ephemeral"}}
	]}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	want := []any{
		map[string]any{"type": "file", "file": map[string]any{"file_id": "file-123"}},
		map[string]any{"type": "file", "file": map[string]any{"file_id": "https://x/a.pdf"}},
		map[string]any{"type": "file", "file": map[string]any{"file_data": "data:application/pdf;base64,JVBE"}},
		map[string]any{"type": "file", "file": map[string]any{}, "cache_control": map[string]any{"type": "ephemeral"}},
	}
	if !reflect.DeepEqual(content, want) {
		t.Fatalf("content = %#v, want %#v", content, want)
	}
}

// A system or developer item's content folds through the same per-part
// conversion any message takes: blocks stay blocks, never flattened into one
// joined string.
func TestResponsesToChatSystemItemContentStaysBlocks(t *testing.T) {
	body := `{"model":"gpt-5","input":[{"type":"message","role":"system","content":[{"type":"input_text","text":"be terse"},{"type":"input_text","text":"  "},{"type":"text","text":"and brief"}]}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v", msgs)
	}
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" {
		t.Fatalf("role = %v", sys["role"])
	}
	want := []any{
		map[string]any{"type": "text", "text": "be terse"},
		map[string]any{"type": "text", "text": "  "},
		map[string]any{"type": "text", "text": "and brief"},
	}
	if !reflect.DeepEqual(sys["content"], want) {
		t.Fatalf("content = %#v, want the blocks kept %#v", sys["content"], want)
	}
}

// A tool output's text parts join with no separator, a non-dict part is
// skipped rather than stringified, and a list nothing matched falls back to
// the JSON of the whole value — an empty list reads as "[]".
func TestResponsesToChatToolOutputJoinsLikeTheReference(t *testing.T) {
	cases := []struct {
		name   string
		output string
		want   any
	}{
		{"two texts", `[{"type":"input_text","text":"a"},{"type":"output_text","text":"b"}]`, "ab"},
		{"non-dict skipped", `["abc",{"type":"text","text":"def"}]`, "def"},
		{"empty list", `[]`, "[]"},
		{"non-dict only", `["abc"]`, `["abc"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"gpt-5","input":[{"type":"function_call_output","call_id":"c1","output":` + tc.output + `}]}`
			out, err := ConvertRequest(Responses, Chat, []byte(body))
			if err != nil {
				t.Fatal(err)
			}
			var m map[string]any
			if err := json.Unmarshal(out.Body, &m); err != nil {
				t.Fatal(err)
			}
			if got := m["messages"].([]any)[0].(map[string]any)["content"]; !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("content = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// A Responses custom tool folds into a function tool: the grammar rides the
// description as a Format block and the parameters fix the one string the
// format-shaped call takes.
func TestResponsesToChatCustomToolBecomesFunctionTool(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","tools":[{"type":"custom","name":"roll_dice","description":"Roll a die","format":{"type":"grammar","syntax":"lark","definition":"start: DIGIT"}}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" {
		t.Fatalf("tool = %#v, want a function tool", tool)
	}
	fn := tool["function"].(map[string]any)
	wantDesc := "Roll a die\n\nFormat:\n```lark\nstart: DIGIT\n```"
	if fn["name"] != "roll_dice" || fn["description"] != wantDesc {
		t.Fatalf("function = %#v, want the grammar in the description", fn)
	}
	params := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("parameters = %#v", params)
	}
	content := params["properties"].(map[string]any)["content"].(map[string]any)
	if content["type"] != "string" || content["description"] != "The roll_dice content following the specified format" {
		t.Fatalf("content property = %#v", content)
	}
	if req := params["required"].([]any); len(req) != 1 || req[0] != "content" {
		t.Fatalf("required = %#v", params["required"])
	}
}

// A web search tool contributes no tools entry — it derives the chat
// web_search_options the search stated, and a bare one states both options
// as absent.
func TestResponsesToChatWebSearchToolDerivesOptions(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","tools":[{"type":"web_search","search_context_size":"low","user_location":{"type":"approximate","city":"SF"}}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
		t.Fatalf("web search must not become a tool: %#v", tools)
	}
	wso := m["web_search_options"].(map[string]any)
	if wso["search_context_size"] != "low" {
		t.Fatalf("web_search_options = %#v", wso)
	}
	if loc, ok := wso["user_location"].(map[string]any); !ok || loc["city"] != "SF" {
		t.Fatalf("user_location = %#v", wso["user_location"])
	}

	body = `{"model":"gpt-5","input":"x","tools":[{"type":"web_search"}]}`
	out, err = ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	wso = m["web_search_options"].(map[string]any)
	if wso["search_context_size"] != nil || wso["user_location"] != nil {
		t.Fatalf("bare web_search_options = %#v, want both options stated as absent", wso)
	}
}

// mcp and preview tools pass through as they stand; only computer_use,
// image_generation and shell drop; and a function tool always states its
// description and strict (absent reads as "" and false).
func TestResponsesToChatToolListKeepsWhatChatReads(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","tools":[
		{"type":"mcp","server_label":"srv"},
		{"type":"computer_use_preview","display_height":768},
		{"type":"image_generation"},
		{"type":"shell"},
		{"type":"function","name":"f2"}
	]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("tools = %#v, want mcp, the preview, and the function tool", tools)
	}
	if tools[0].(map[string]any)["type"] != "mcp" || tools[1].(map[string]any)["type"] != "computer_use_preview" {
		t.Fatalf("passthrough tools = %#v", tools[:2])
	}
	fn := tools[2].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "f2" || fn["description"] != "" {
		t.Fatalf("function = %#v, want description always stated", fn)
	}
	if fn["strict"] != false {
		t.Fatalf("strict = %#v, want false stated", fn["strict"])
	}
	if fn["parameters"].(map[string]any)["type"] != "object" {
		t.Fatalf("parameters = %#v", fn["parameters"])
	}
}

// A Responses tool with no type passes through as it stands: the reference
// only folds typed function and custom tools, and drops none but its explicit
// list — a bare name is not a function tool.
func TestResponsesToChatTypelessToolPassesThrough(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","tools":[{"name":"typeless"}]}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %#v, want the typeless tool kept", tools)
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "typeless" {
		t.Fatalf("tool = %#v, want it verbatim", tool)
	}
	if _, isFunction := tool["function"]; isFunction {
		t.Fatalf("tool = %#v, a typeless tool must not become a function tool", tool)
	}
}

// A json_schema format that stated no schema reads as an empty schema, not a
// null one — the reference's .get(schema, {}) default.
func TestResponsesToChatMissingSchemaDefaultsEmpty(t *testing.T) {
	body := `{"model":"gpt-5","input":"x","text":{"format":{"type":"json_schema","name":"n"}}}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	js := m["response_format"].(map[string]any)["json_schema"].(map[string]any)
	if schema, ok := js["schema"].(map[string]any); !ok || len(schema) != 0 {
		t.Fatalf("schema = %#v, want an empty object rather than null", js["schema"])
	}
}

// A chat json_schema response_format fills its defaults whatever it was
// handed: no json_schema member still names response_schema with an empty
// schema and a stated false strict — never a nameless format the Responses
// upstream rejects — and a member without a schema reads as empty too.
func TestChatToResponsesJsonSchemaFillsDefaults(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema"}}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	format := m["text"].(map[string]any)["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "response_schema" {
		t.Fatalf("format = %#v, want the defaults filled", format)
	}
	if schema, ok := format["schema"].(map[string]any); !ok || len(schema) != 0 {
		t.Fatalf("schema = %#v, want an empty object rather than null", format["schema"])
	}
	if format["strict"] != false {
		t.Fatalf("strict = %#v, want a stated false", format["strict"])
	}

	body = `{"model":"gpt-5","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"n"}}}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	format = m["text"].(map[string]any)["format"].(map[string]any)
	if format["name"] != "n" {
		t.Fatalf("name = %#v", format["name"])
	}
	if schema, ok := format["schema"].(map[string]any); !ok || len(schema) != 0 {
		t.Fatalf("schema = %#v, want an empty object rather than null", format["schema"])
	}
}

// A chat function tool's parameters, strict and description always ride the
// Responses tool — the reference bridge carries the three keys, a tool that
// stated none sending them as null — and a stated value rides verbatim.
func TestChatToResponsesFunctionToolAlwaysCarriesItsKeys(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f2"}}]}`
	out, err := ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "f2" {
		t.Fatalf("tool = %#v", tool)
	}
	for _, k := range []string{"parameters", "strict", "description"} {
		v, present := tool[k]
		if !present || v != nil {
			t.Fatalf("%s = %#v (present=%v), want the key carried as null", k, v, present)
		}
	}

	body = `{"model":"gpt-5","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"},"strict":true}}]}`
	out, err = ConvertRequest(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool = m["tools"].([]any)[0].(map[string]any)
	if tool["description"] != "d" || tool["strict"] != true {
		t.Fatalf("tool = %#v, want stated values verbatim", tool)
	}
	if p, ok := tool["parameters"].(map[string]any); !ok || p["type"] != "object" {
		t.Fatalf("parameters = %#v", tool["parameters"])
	}
}
