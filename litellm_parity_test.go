package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// This file pins the folds that a LiteLLM v1.100.1 audit found diverging from
// the reference. Each case is a body or stream the codec used to render as
// something the target dialect would have refused, or as a silent loss of
// content the caller supplied.

// A system-only chat request must not become `"input": null`, which the
// Responses API rejects. The instructions ride a system input item instead.
func TestChatToResponsesSystemOnlyHasInput(t *testing.T) {
	body := `{"model":"gpt-5","messages":[{"role":"system","content":"be terse"}]}`
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
		t.Fatalf("input = %#v, want one system item", m["input"])
	}
	if _, present := m["instructions"]; present {
		t.Fatalf("instructions should have folded into input: %#v", m)
	}
	item := input[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "system" {
		t.Fatalf("item = %#v", item)
	}
}

// The assistant turn is reasoning -> message -> function_call. A call emitted
// before the message it was decided in reads as a call with no text turn.
func TestChatResponseToResponsesItemOrder(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"hi","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	var types []string
	for _, item := range m["output"].([]any) {
		types = append(types, item.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "message,function_call" {
		t.Fatalf("output order = %v, want message before function_call", types)
	}
}

// A reasoning delta on a Responses stream is model output the chat wire carries
// in its own field; dropping the frame leaves the client an answer with no
// reasoning.
func TestResponsesToChatStreamKeepsReasoning(t *testing.T) {
	src := strings.Join([]string{
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","delta":"think"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(dst.String(), `"reasoning_content":"think"`) {
		t.Fatalf("reasoning lost:\n%s", dst.String())
	}
}

// An incomplete turn that happened to contain a call is still a ceiling cut,
// not a turn where the model chose to call a tool.
func TestResponsesToChatIncompleteKeepsItsReason(t *testing.T) {
	src := strings.Join([]string{
		`event: response.incomplete`,
		`data: {"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(dst.String(), `"finish_reason":"length"`) {
		t.Fatalf("want length:\n%s", dst.String())
	}
}

// Usage details have to survive both folds, or a host loses the reasoning-token
// and cache breakdowns it bills on.
func TestUsageDetailsCrossChatAndResponses(t *testing.T) {
	chat := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":2}}}`
	out, err := ConvertResponse(Chat, Responses, []byte(chat))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if _, ok := usage["input_tokens_details"]; !ok {
		t.Fatalf("input_tokens_details dropped: %#v", usage)
	}
	if _, ok := usage["output_tokens_details"]; !ok {
		t.Fatalf("output_tokens_details dropped: %#v", usage)
	}

	back, err := ConvertResponse(Responses, Chat, out)
	if err != nil {
		t.Fatal(err)
	}
	var b map[string]any
	if err := json.Unmarshal(back, &b); err != nil {
		t.Fatal(err)
	}
	u := b["usage"].(map[string]any)
	if _, ok := u["prompt_tokens_details"]; !ok {
		t.Fatalf("prompt_tokens_details dropped: %#v", u)
	}
	if _, ok := u["completion_tokens_details"]; !ok {
		t.Fatalf("completion_tokens_details dropped: %#v", u)
	}
}

// Anthropic requires the wire to alternate roles, so two consecutive user
// turns are merged rather than sent as a request the upstream rejects.
func TestChatToMessagesMergesConsecutiveRoles(t *testing.T) {
	body := `{"model":"c","max_tokens":10,"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want one merged turn", msgs)
	}
	// Each merged turn keeps its own text block (litellm v1.100.1
	// factory.py:2453-2477 appends one text block per message); joining the
	// strings would read as one glued word the caller never wrote.
	blocks, ok := msgs[0].(map[string]any)["content"].([]any)
	if !ok || len(blocks) != 2 {
		t.Fatalf("content = %#v, want two text blocks", msgs[0].(map[string]any)["content"])
	}
	if blocks[0].(map[string]any)["text"] != "a" || blocks[1].(map[string]any)["text"] != "b" {
		t.Fatalf("content = %#v, want the two turns' texts kept apart", blocks)
	}
}

// Anthropic enforces [a-zA-Z0-9_-]{1,128} on a tool name, and a tool_use in the
// history must reference the name the request actually declares.
func TestChatToMessagesSanitizesToolNames(t *testing.T) {
	body := `{"model":"c","max_tokens":10,"messages":[{"role":"assistant","tool_calls":[{"id":"t1","type":"function","function":{"name":"a/b<c>","arguments":"{}"}}]},{"role":"tool","tool_call_id":"t1","content":"ok"}],"tools":[{"type":"function","function":{"name":"a/b<c>","parameters":{}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	declared := m["tools"].([]any)[0].(map[string]any)["name"]
	if declared != "a_b_c_" {
		t.Fatalf("declared name = %v, want a_b_c_", declared)
	}
	blocks := m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	used := blocks[0].(map[string]any)["name"]
	if used != declared {
		t.Fatalf("tool_use name = %v, but the request declares %v", used, declared)
	}
}

// An empty text block is a request Anthropic refuses outright, so an empty
// turn is rewritten rather than passed through.
func TestChatToMessagesRewritesEmptyText(t *testing.T) {
	body := `{"model":"c","max_tokens":10,"messages":[{"role":"assistant","content":""},{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[0].(map[string]any)["content"]
	if content == "" {
		t.Fatalf("empty assistant content passed through: %#v", content)
	}
}

// A cache_control breakpoint lives on a block, so a single text block that
// carries one cannot collapse to a bare string.
func TestChatToMessagesKeepsCacheControl(t *testing.T) {
	body := `{"model":"c","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[0].(map[string]any)["content"]
	blocks, ok := content.([]any)
	if !ok {
		t.Fatalf("content = %#v, want a block list carrying the breakpoint", content)
	}
	if _, ok := blocks[0].(map[string]any)["cache_control"]; !ok {
		t.Fatalf("cache_control dropped: %#v", blocks[0])
	}
}

// A refusal is not a clean stop; collapsing it to "stop" tells the client the
// model finished normally.
func TestMessagesToChatMapsRefusalStopReason(t *testing.T) {
	body := `{"id":"m1","model":"c","stop_reason":"refusal","content":[{"type":"text","text":"no"}],"usage":{"input_tokens":1,"output_tokens":1}}`
	out, err := ConvertResponse(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	fr := m["choices"].([]any)[0].(map[string]any)["finish_reason"]
	if fr != "content_filter" {
		t.Fatalf("finish_reason = %v, want content_filter", fr)
	}
}

// A chat content part list must be read part by part; rendering it with a
// generic formatter produces Go syntax the model cannot read.
func TestChatToMessagesReadsContentParts(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},"finish_reason":"stop"}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	block := m["content"].([]any)[0].(map[string]any)
	if block["text"] != "ab" {
		t.Fatalf("text = %#v, want the joined parts", block["text"])
	}
}

// An image a tool result returned cannot ride a function_call_output; it is
// re-attached in the same turn, behind a boundary line.
func TestMessagesToResponsesToolImageStaysInTurn(t *testing.T) {
	body := `{"model":"c","max_tokens":100,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]}]},{"role":"user","content":"next"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %#v, want the call, the images, then the next turn", input)
	}
	images := input[1].(map[string]any)
	if images["role"] != "user" {
		t.Fatalf("images item = %#v", images)
	}
	parts := images["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("image message parts = %#v, want a boundary and the image", parts)
	}
	if !strings.Contains(parts[0].(map[string]any)["text"].(string), "tool output") {
		t.Fatalf("boundary part = %#v", parts[0])
	}
}

// Responses reports the cache split under input_tokens_details; Anthropic
// reports it as separate fields and counts input_tokens as the remainder.
func TestResponsesToMessagesMapsCacheUsage(t *testing.T) {
	body := `{"id":"r1","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":80}}}`
	out, err := ConvertResponse(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if usage["cache_read_input_tokens"] != float64(80) {
		t.Fatalf("cache_read_input_tokens = %#v", usage)
	}
	if usage["input_tokens"] != float64(20) {
		t.Fatalf("input_tokens = %#v, want the uncached remainder", usage["input_tokens"])
	}
}

// Consecutive reasoning items belong to one assistant turn; two consecutive
// assistant messages are a request the target rejects.
func TestResponsesToMessagesMergesReasoningIntoTurn(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"a"}]},{"type":"reasoning","summary":[{"type":"summary_text","text":"b"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want one assistant turn", msgs)
	}
	blocks := msgs[0].(map[string]any)["content"].([]any)
	var types []string
	for _, b := range blocks {
		types = append(types, b.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "thinking,thinking,text" {
		t.Fatalf("blocks = %v", types)
	}
}

// A structured function_call_output must not be stringified with Go syntax.
func TestResponsesToMessagesRendersStructuredToolOutput(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["content"]
	if content != "a\nb" {
		t.Fatalf("tool result content = %#v", content)
	}
}

// A signed thinking block's signature has to reach the target, or the turn
// cannot be replayed. The reference carries it as litellm's array-encoded
// encrypted_content (v1.100.1
// responses/litellm_completion_transformation/transformation.py:2356
// _encode_thinking_blocks re-encodes every block with a signature as a compact
// JSON array), so the signature survives inside the array — not as the bare
// string this test used to pin.
func TestChatResponseToResponsesKeepsThinkingSignature(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi","thinking_blocks":[{"type":"thinking","thinking":"why","signature":"SIG"}]}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	for _, item := range m["output"].([]any) {
		im := item.(map[string]any)
		if im["type"] == "reasoning" {
			enc, _ := im["encrypted_content"].(string)
			var blocks []map[string]any
			if err := json.Unmarshal([]byte(enc), &blocks); err != nil {
				t.Fatalf("encrypted_content = %q, want the array-encoded blocks: %v", enc, err)
			}
			if len(blocks) != 1 || blocks[0]["type"] != "thinking" || blocks[0]["signature"] != "SIG" {
				t.Fatalf("encrypted blocks = %#v, want the signed thinking block", blocks)
			}
			return
		}
	}
	t.Fatalf("no reasoning item in %#v", m["output"])
}

// The Responses SSE protocol numbers its events in emission order, and a
// client that reassembles the stream reads the sequence.
func TestResponsesStreamSequenceNumbersAreContiguous(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"reasoning_content":"r"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var seqs []float64
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		var data string
		for _, line := range strings.Split(chunk, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		var m map[string]any
		if json.Unmarshal([]byte(data), &m) != nil {
			continue
		}
		n, ok := m["sequence_number"].(float64)
		if !ok {
			t.Fatalf("event without sequence_number: %s", data)
		}
		seqs = append(seqs, n)
	}
	if len(seqs) == 0 {
		t.Fatal("no events")
	}
	for i, n := range seqs {
		if n != float64(i) {
			t.Fatalf("sequence numbers not contiguous in emission order: %v", seqs)
		}
	}
}

// A custom tool call carries its freeform payload in `input`; the chat wire
// reads it as arguments, and dropping the item loses the call the model made.
func TestResponsesToChatKeepsCustomToolCall(t *testing.T) {
	body := `{"id":"r","status":"completed","output":[{"type":"custom_tool_call","call_id":"c1","name":"shell","input":"ls -la"}]}`
	out, err := ConvertResponse(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	choice := m["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v", choice["finish_reason"])
	}
	call := choice["message"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	fn := call["function"].(map[string]any)
	if fn["name"] != "shell" || fn["arguments"] != "ls -la" {
		t.Fatalf("tool call = %#v", fn)
	}
}

// Under stream_options.include_usage the usage rides a chunk of its own, after
// the finish_reason. A fold that closes on the finish chunk reports zero
// tokens for a turn that has a real count.
func TestChatUsageTailIsFoldedIntoTheTerminal(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"c","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}`,
		``,
	}, "\n")

	var toMessages bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &toMessages, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out := toMessages.String()
	if !strings.Contains(out, `"input_tokens":10`) || !strings.Contains(out, `"output_tokens":5`) {
		t.Fatalf("usage lost folding to Messages:\n%s", out)
	}
	if n := strings.Count(out, "event: message_stop"); n != 1 {
		t.Fatalf("message_stop events = %d, want 1\n%s", n, out)
	}

	var toResponses bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &toResponses, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(toResponses.String(), `"input_tokens":10`) {
		t.Fatalf("usage lost folding to Responses:\n%s", toResponses.String())
	}
}

// A chat tool_choice nests a named choice under function/custom, while the
// Responses API takes the name at the top level; and a type-only dict is the
// string the chat wire spells it as.
func TestToolChoiceShapeMatchesTheReference(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{"type":"auto"}`, `"auto"`},
		{`{"type":"required"}`, `"required"`},
		{`{"type":"function","name":"f"}`, `{"name":"f","type":"function"}`},
		{`{"type":"custom","custom":{"name":"c"}}`, `{"name":"c","type":"custom"}`},
	}
	for _, tc := range cases {
		body := `{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":` + tc.in + `}`
		out, err := ConvertRequest(Chat, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		got, _ := json.Marshal(m["tool_choice"])
		if string(got) != tc.want {
			t.Fatalf("tool_choice %s -> %s, want %s", tc.in, got, tc.want)
		}
	}
}

// A Responses choice that names no tool means "any tool", which the chat wire
// spells "required" — when the request declares a tool to pick. With no tools
// at all, the reference drops the pairing entirely rather than leaving a
// tool_choice that names nothing.
func TestResponsesToolChoiceWithoutNameBecomesRequired(t *testing.T) {
	for _, in := range []string{`{"type":"function"}`, `{"type":"tool"}`, `{"type":"any"}`} {
		body := `{"model":"m","input":"x","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"tool_choice":` + in + `}`
		out, err := ConvertRequest(Responses, Chat, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if m["tool_choice"] != "required" {
			t.Fatalf("tool_choice %s -> %v, want required", in, m["tool_choice"])
		}
	}
	// No tools declared: the key folds away altogether.
	body := `{"model":"m","input":"x","tool_choice":{"type":"function"}}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["tool_choice"]; present {
		t.Fatalf("tool_choice without tools = %#v, want the key gone", m["tool_choice"])
	}
}

// A Responses choice naming a custom tool still names a tool; folding it to
// "auto" would tell the model to pick any tool instead.
func TestResponsesToMessagesKeepsCustomToolChoiceName(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":"x","tool_choice":{"type":"custom","name":"c"}}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tc := m["tool_choice"].(map[string]any)
	if tc["type"] != "tool" || tc["name"] != "c" {
		t.Fatalf("tool_choice = %#v, want the named tool", tc)
	}
}

// A text.format folded to a chat response_format must not carry a null strict,
// and an explicit "text" format is a response_format the chat wire spells.
func TestTextFormatFoldsToResponseFormat(t *testing.T) {
	body := `{"model":"m","input":"x","text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object"}}}}`
	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	js := m["response_format"].(map[string]any)["json_schema"].(map[string]any)
	if st, present := js["strict"]; !present || st != false {
		t.Fatalf("strict = %#v, want a stated false", js["strict"])
	}

	body = `{"model":"m","input":"x","text":{"format":{"type":"text"}}}`
	out, err = ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	rf, ok := m["response_format"].(map[string]any)
	if !ok || rf["type"] != "text" {
		t.Fatalf("response_format = %#v, want a text format", m["response_format"])
	}
}

// A json_schema format folded to Messages must not carry a null strict.
func TestResponsesToMessagesStatesStrict(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":"x","text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object"}}}}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	of := m["output_format"].(map[string]any)
	if st, present := of["strict"]; !present || st != false {
		t.Fatalf("strict = %#v, want a stated false", of["strict"])
	}
}

// OpenAI's prompt_tokens counts the whole prompt, cache included, while
// Anthropic's input_tokens is the uncached remainder and reports the cache
// halves separately. Folding one onto the other must subtract.
func TestChatToMessagesSplitsCacheFromInputTokens(t *testing.T) {
	body := `{"id":"c","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":80,"cache_creation_tokens":10}}}`
	out, err := ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if usage["input_tokens"] != float64(10) {
		t.Fatalf("input_tokens = %#v, want the uncached remainder", usage["input_tokens"])
	}
	if usage["cache_read_input_tokens"] != float64(80) {
		t.Fatalf("cache_read_input_tokens = %#v", usage)
	}
	if usage["cache_creation_input_tokens"] != float64(10) {
		t.Fatalf("cache_creation_input_tokens = %#v", usage)
	}
}

// An Anthropic tool input_schema accepts a fixed key set and only `$defs` as a
// ref target, so a legacy definition has to be inlined before the block that
// carried it is dropped.
func TestChatToMessagesSanitizesToolSchema(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"properties":{"a":{"$ref":"#/definitions/T"}},"definitions":{"T":{"type":"string"}},"title":"drop me"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	schema := m["tools"].([]any)[0].(map[string]any)["input_schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("schema type = %#v, want object", schema["type"])
	}
	if _, present := schema["definitions"]; present {
		t.Fatalf("legacy definitions kept: %#v", schema)
	}
	if _, present := schema["title"]; present {
		t.Fatalf("a key outside the accepted set was kept: %#v", schema)
	}
	prop := schema["properties"].(map[string]any)["a"].(map[string]any)
	if prop["type"] != "string" {
		t.Fatalf("ref not inlined: %#v", prop)
	}
}

// A schema that is not already an object schema is coerced to one.
func TestChatToMessagesCoercesToolSchema(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	schema := m["tools"].([]any)[0].(map[string]any)["input_schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("schema = %#v, want an object schema", schema)
	}
	if _, ok := schema["properties"]; !ok {
		t.Fatalf("schema = %#v, want a properties object", schema)
	}
}

// An Anthropic-native tool owns a reserved name the provider defines; rewriting
// it into a function tool would drop the type that names it.
func TestMessagesToChatKeepsHostedTools(t *testing.T) {
	// web_search is the exception: litellm v1.100.1 (transformation.py:987-993)
	// folds it out of tools into the chat-native web_search_options, so it does
	// not ride the tools list; the other hosted tools stay as they stand.
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search_20260209","name":"t"}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var wm map[string]any
	if err := json.Unmarshal(out.Body, &wm); err != nil {
		t.Fatal(err)
	}
	if tools, ok := wm["tools"].([]any); ok && len(tools) > 0 {
		t.Fatalf("web_search tool stayed in tools: %#v", tools)
	}
	if wso, ok := wm["web_search_options"].(map[string]any); !ok || len(wso) != 0 {
		t.Fatalf("web_search_options = %#v, want {}", wm["web_search_options"])
	}
	for _, hosted := range []string{"bash_20250124", "text_editor_20250728"} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"` + hosted + `","name":"t"}]}`
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		tool := m["tools"].([]any)[0].(map[string]any)
		if tool["type"] != hosted {
			t.Fatalf("hosted tool %s became %#v", hosted, tool)
		}
	}
}

// The reference drops exactly the Responses tool types with no chat spelling
// — computer_use, image_generation, shell — and passes the rest (mcp,
// previews) through unchanged. The Messages fold still drops the dated hosted
// tools as Responses-only: each fold keeps what its own target reads.
func TestResponsesOnlyToolsAreDropped(t *testing.T) {
	body := `{"model":"m","input":"x","tools":[{"type":"computer_use_preview"},{"type":"image_generation"},{"type":"shell"}]}`

	out, err := ConvertRequest(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["type"] != "computer_use_preview" {
		t.Fatalf("folding to Chat = %#v, want the preview passed through", tools)
	}

	out, err = ConvertRequestWith(Responses, Messages, []byte(body), ConvertOptions{MaxTokensFallback: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
		t.Fatalf("folding to Messages kept %#v", tools)
	}
}

// A thinking block with no text and no signature holds nothing a client could
// need, and Anthropic rejects an empty one; a signature-only block is a real
// response and must be kept so reasoning can be replayed.
func TestChatToMessagesDropsEmptyUnsignedThinking(t *testing.T) {
	empty := `{"id":"c","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x","thinking_blocks":[{"type":"thinking","thinking":""}]}}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	for _, block := range m["content"].([]any) {
		if block.(map[string]any)["type"] == "thinking" {
			t.Fatalf("empty unsigned thinking block kept: %#v", block)
		}
	}

	signed := `{"id":"c","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x","thinking_blocks":[{"type":"thinking","thinking":"","signature":"SIG"}]}}]}`
	out, err = ConvertResponse(Chat, Messages, []byte(signed))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	kept := false
	for _, block := range m["content"].([]any) {
		b := block.(map[string]any)
		if b["type"] == "thinking" && b["signature"] == "SIG" {
			kept = true
		}
	}
	if !kept {
		t.Fatalf("signature-only thinking block dropped: %#v", m["content"])
	}
}

// Anthropic compiles a structured-output schema into a grammar and rejects the
// JSON Schema keywords that grammar cannot express, so each is dropped and
// stated in the schema's own description; oneOf becomes the anyOf it accepts.
func TestChatToMessagesFiltersOutputSchema(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object","properties":{"a":{"type":"array","maxItems":3,"items":{"type":"string"}}},"oneOf":[{"type":"object"}]}}}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	schema := m["output_format"].(map[string]any)["schema"].(map[string]any)
	if _, present := schema["oneOf"]; present {
		t.Fatalf("oneOf kept: %#v", schema)
	}
	if _, ok := schema["anyOf"]; !ok {
		t.Fatalf("oneOf not rewritten to anyOf: %#v", schema)
	}
	if schema["additionalProperties"] != false {
		t.Fatalf("an object schema must state additionalProperties=false: %#v", schema)
	}
	a := schema["properties"].(map[string]any)["a"].(map[string]any)
	if _, present := a["maxItems"]; present {
		t.Fatalf("maxItems kept: %#v", a)
	}
	if !strings.Contains(a["description"].(string), "maximum number of items: 3") {
		t.Fatalf("dropped constraint not stated: %#v", a)
	}
}

// A top-level cache_control marks the request for automatic prompt caching; it
// was in the accepted vocabulary but never copied to the output.
func TestChatToMessagesKeepsTopLevelCacheControl(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"cache_control":{"type":"ephemeral"}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	cc, ok := m["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control = %#v, want it carried through", m["cache_control"])
	}
}

// A Messages turn with thinking enabled needs room for the thinking plus the
// answer, so a request that named no ceiling gets one derived from the budget
// rather than none at all. Without thinking there is nothing to derive from.
func TestChatToMessagesDerivesMaxTokensFromThinking(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled","budget_tokens":2048}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["max_tokens"] != float64(2048+4096) {
		t.Fatalf("max_tokens = %#v, want the budget plus the default answer room", m["max_tokens"])
	}

	// No thinking budget to derive from: the host still supplies the value.
	if _, err := ConvertRequest(Chat, Messages, []byte(`{"model":"m","messages":[{"role":"user","content":"x"}]}`)); err == nil {
		t.Fatal("a fold with nothing to derive max_tokens from must report it")
	}
}

// The derivation applies wherever a fold into Messages produces thinking, not
// only the chat fold.
func TestResponsesToMessagesDerivesMaxTokensFromReasoning(t *testing.T) {
	body := `{"model":"m","input":"x","reasoning":{"effort":"high"}}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["max_tokens"] != float64(4096+4096) {
		t.Fatalf("max_tokens = %#v, want the budget plus the default answer room", m["max_tokens"])
	}
}
