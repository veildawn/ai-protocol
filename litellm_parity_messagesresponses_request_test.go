package protocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// This file pins the Messages -> Responses request folds a LiteLLM v1.100.1
// audit found diverging from the reference adapter
// (litellm/llms/anthropic/experimental_pass_through/responses_adapters).

// The assistant turn's text is one message appended after the turn's other
// items: litellm aggregates every text block of the message and emits it once,
// after the reasoning and function_call items derived from the same turn,
// rather than interleaving a message between the calls it sits around.
func TestMessagesToResponsesAggregatesAssistantTextAfterCalls(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"text","text":"T1"},{"type":"tool_use","id":"t1","name":"fn","input":{"a":1}},{"type":"text","text":"T2"}]}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input, ok := m["input"].([]any)
	if !ok || len(input) != 2 {
		t.Fatalf("input = %#v, want the call then one aggregated message", m["input"])
	}
	if first := input[0].(map[string]any); first["type"] != "function_call" {
		t.Fatalf("first item = %#v, want the function_call", first)
	}
	msg := input[1].(map[string]any)
	if msg["type"] != "message" || msg["role"] != "assistant" {
		t.Fatalf("second item = %#v, want the assistant message", msg)
	}
	parts := msg["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %#v, want T1 and T2 in one message", parts)
	}
	if parts[0].(map[string]any)["text"] != "T1" || parts[1].(map[string]any)["text"] != "T2" {
		t.Fatalf("parts = %#v, want both text blocks aggregated", parts)
	}
}

// Text surrounding both a thinking block and a call still lands in one
// message, behind the reasoning item and the call.
func TestMessagesToResponsesTextAfterThinkingAndCall(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"text","text":"A"},{"type":"thinking","thinking":"th"},{"type":"tool_use","id":"t2","name":"fn","input":{}},{"type":"text","text":"B"}]}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	var types []string
	for _, item := range m["input"].([]any) {
		types = append(types, item.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "reasoning,function_call,message" {
		t.Fatalf("input order = %v, want reasoning, call, then the message", types)
	}
}

// A run of consecutive thinking blocks is one reasoning item whose summary
// carries every block of the run; a later lone thinking block is its own item.
func TestMessagesToResponsesFoldsThinkingRuns(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"A"},{"type":"thinking","thinking":"B"},{"type":"text","text":"hi"},{"type":"thinking","thinking":"C"},{"type":"tool_use","id":"t3","name":"fn","input":{}}]}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	var types []string
	for _, item := range input {
		types = append(types, item.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "reasoning,reasoning,function_call,message" {
		t.Fatalf("input order = %v, want two reasoning items, the call, the message", types)
	}
	first := input[0].(map[string]any)["summary"].([]any)
	if len(first) != 2 {
		t.Fatalf("first run summary = %#v, want A and B folded together", first)
	}
	if first[0].(map[string]any)["text"] != "A" || first[1].(map[string]any)["text"] != "B" {
		t.Fatalf("first run summary = %#v, want A and B", first)
	}
	second := input[1].(map[string]any)["summary"].([]any)
	if len(second) != 1 || second[0].(map[string]any)["text"] != "C" {
		t.Fatalf("second run summary = %#v, want C alone", second)
	}
}

// The Responses reasoning item built from thinking blocks carries type and
// summary only — the reference drops the blocks' signatures rather than
// writing them as encrypted_content, so the fold drops them too.
func TestMessagesToResponsesDropsThinkingSignature(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"why","signature":"SigAbC123=="},{"type":"text","text":"hi"}]}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	item := m["input"].([]any)[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("first item = %#v, want a reasoning item", item)
	}
	if _, present := item["encrypted_content"]; present {
		t.Fatalf("reasoning item = %#v, want no encrypted_content", item)
	}
	if summary := item["summary"].([]any); len(summary) != 1 || summary[0].(map[string]any)["text"] != "why" {
		t.Fatalf("summary = %#v, want the thinking text", item["summary"])
	}
}

// A mid-turn system message's blocks are one input_text part each, so a
// block's prompt_cache_breakpoint stays on the part it belongs to instead of
// being lost in a joined string; non-text blocks carry nothing to fold.
func TestMessagesToResponsesSystemBlocksKeepBreakpoints(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"system","content":[{"type":"text","text":"S1","prompt_cache_breakpoint":true},{"type":"text","text":"S2"},{"type":"text","text":""},{"type":"image","source":{"type":"url","url":"https://x/i.png"}}]},{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	sys := input[0].(map[string]any)
	if sys["type"] != "message" || sys["role"] != "system" {
		t.Fatalf("first item = %#v, want a system message", sys)
	}
	parts := sys["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %#v, want one part per nonempty text block", parts)
	}
	if got := parts[0].(map[string]any)["prompt_cache_breakpoint"]; got != true {
		t.Fatalf("S1 part = %#v, want its prompt_cache_breakpoint kept", parts[0])
	}
	if _, present := parts[1].(map[string]any)["prompt_cache_breakpoint"]; present {
		t.Fatalf("S2 part = %#v, want no marker the block did not carry", parts[1])
	}
}

// A top-level system list carrying a prompt_cache_breakpoint cannot fold into
// instructions — the marker has nowhere to live there — so it rides a
// developer message ahead of the input, each part keeping its own marker.
func TestMessagesToResponsesSystemBreakpointBecomesDeveloper(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"system":[{"type":"text","text":"You are helpful."},{"type":"text","text":"More.","prompt_cache_breakpoint":true}],"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["instructions"]; present {
		t.Fatalf("instructions = %#v, want the system to ride a developer item instead", m["instructions"])
	}
	input := m["input"].([]any)
	dev := input[0].(map[string]any)
	if dev["type"] != "message" || dev["role"] != "developer" {
		t.Fatalf("input[0] = %#v, want a developer message", dev)
	}
	parts := dev["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("developer parts = %#v, want one per system text block", parts)
	}
	if got := parts[1].(map[string]any)["prompt_cache_breakpoint"]; got != true {
		t.Fatalf("marked part = %#v, want the breakpoint kept", parts[1])
	}

	// Without a marker the same list still folds into instructions.
	plain := `{"model":"m","max_tokens":10,"system":[{"type":"text","text":"You are helpful."},{"type":"text","text":"More."}],"messages":[{"role":"user","content":"hi"}]}`
	out, err = ConvertRequest(Messages, Responses, []byte(plain))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["instructions"] != "You are helpful.\nMore." {
		t.Fatalf("instructions = %#v, want the joined system texts", m["instructions"])
	}
	for _, item := range m["input"].([]any) {
		if item.(map[string]any)["role"] == "developer" {
			t.Fatalf("developer item = %#v, want none without a breakpoint", item)
		}
	}
}

// A custom_tool_call item has no Messages counterpart: the reference's item
// loop handles message, reasoning and function_call only, so the item is
// dropped — not folded into a text block, and without touching stop_reason.
func TestResponsesToMessagesDropsCustomToolCall(t *testing.T) {
	body := `{"id":"r1","status":"completed","output":[{"type":"custom_tool_call","call_id":"c1","name":"shell","input":"ls -la"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	out, err := ConvertResponse(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	content := m["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content = %#v, want only the message's text", content)
	}
	if block := content[0].(map[string]any); block["text"] != "hi" {
		t.Fatalf("block = %#v, want the surviving text block", block)
	}
	if m["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %#v, want end_turn untouched by the dropped call", m["stop_reason"])
	}
}

// A thinking block folded from a reasoning item carries signature null — only
// Anthropic can sign one, so the item's encrypted_content never rides along —
// and a summary with no readable text contributes nothing rather than a
// synthesized redacted block.
func TestResponsesToMessagesThinkingSignatureStaysNull(t *testing.T) {
	body := `{"id":"r1","status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"why"}],"encrypted_content":"ENC"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	out, err := ConvertResponse(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	block := m["content"].([]any)[0].(map[string]any)
	if block["type"] != "thinking" || block["thinking"] != "why" {
		t.Fatalf("block = %#v, want the thinking block", block)
	}
	if v, present := block["signature"]; !present || v != nil {
		t.Fatalf("signature = %#v (present=%v), want an explicit null", v, present)
	}

	empty := `{"id":"r1","status":"completed","output":[{"type":"reasoning","summary":[],"encrypted_content":"ENC"}]}`
	out, err = ConvertResponse(Responses, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if content := m["content"].([]any); len(content) != 0 {
		t.Fatalf("content = %#v, want nothing synthesized from the encrypted bytes", content)
	}
}

// An image source with no bytes produces no part — not a data URL with nothing
// in it — and in a tool_result it adds no placeholder either. A media_type the
// caller spelled out renders as itself (an explicit null included); the
// image/jpeg default applies only when the field is absent.
func TestMessagesToResponsesEmptyImageYieldsNoPart(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","data":""}},{"type":"image","source":{"type":"base64","media_type":null,"data":"AA"}},{"type":"image","source":{"type":"base64","media_type":"","data":"BB"}},{"type":"image","source":{"type":"base64","data":"CC"}}]}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	parts := m["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(parts) != 3 {
		t.Fatalf("parts = %#v, want the empty-data image gone", parts)
	}
	want := []string{"data:None;base64,AA", "data:;base64,BB", "data:image/jpeg;base64,CC"}
	for i, w := range want {
		if got := parts[i].(map[string]any)["image_url"]; got != w {
			t.Fatalf("part %d image_url = %#v, want %q", i, got, w)
		}
	}

	tool := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"see"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]}]}`
	out, err = ConvertRequest(Messages, Responses, []byte(tool))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %#v, want only the call output with no re-attached images", input)
	}
	if got := input[0].(map[string]any)["output"]; got != "see" {
		t.Fatalf("output = %#v, want the bare text with no placeholder", got)
	}
}

// The reference gates max_tokens, tools, tool_choice and stream on truthiness:
// a zero ceiling, empty list, empty object or false stream omit the key, while
// temperature rides on presence alone.
func TestMessagesToResponsesOmitsFalsyParams(t *testing.T) {
	body := `{"model":"m","max_tokens":0,"tools":[],"tool_choice":{},"stream":false,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"max_output_tokens", "tools", "tool_choice", "stream"} {
		if _, present := m[key]; present {
			t.Fatalf("%s = %#v, want the falsy value to omit the key", key, m[key])
		}
	}
	if m["temperature"] != float64(0) {
		t.Fatalf("temperature = %#v, want a present zero copied through", m["temperature"])
	}
}

// Empty system segments drop out of the instructions join rather than leaving
// blank lines between the texts that remain.
func TestMessagesToResponsesSystemJoinDropsEmptySegments(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"system":[{"type":"text","text":"A"},{"type":"text","text":""},{"type":"text","text":"B"}],"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["instructions"] != "A\nB" {
		t.Fatalf("instructions = %#v, want the empty segment dropped", m["instructions"])
	}
}

// An empty schema object is falsy to the reference: no text.format is set,
// rather than one wrapping a schema with nothing in it.
func TestMessagesToResponsesEmptySchemaSetsNoText(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"output_format":{"type":"json_schema","schema":{}},"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["text"]; present {
		t.Fatalf("text = %#v, want no format for an empty schema", m["text"])
	}
}

// metadata.user_id sets user on key presence alone, rendered the way the
// reference's str() renders it — null becomes "None" — truncated to 64
// characters, not bytes; the cache key is only pinned for a nonempty id.
func TestMessagesToResponsesUserIdEdgeCases(t *testing.T) {
	run := func(body string) map[string]any {
		out, err := ConvertRequest(Messages, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	m := run(`{"model":"m","max_tokens":10,"metadata":{"user_id":null},"messages":[{"role":"user","content":"hi"}]}`)
	if m["user"] != "None" {
		t.Fatalf("user = %#v, want the str() of null", m["user"])
	}
	if _, present := m["prompt_cache_key"]; present {
		t.Fatalf("prompt_cache_key = %#v, want none for a null id", m["prompt_cache_key"])
	}
	m = run(`{"model":"m","max_tokens":10,"metadata":{"user_id":""},"messages":[{"role":"user","content":"hi"}]}`)
	if v, present := m["user"]; !present || v != "" {
		t.Fatalf("user = %#v (present=%v), want an explicit empty string", v, present)
	}
	if _, present := m["prompt_cache_key"]; present {
		t.Fatalf("prompt_cache_key = %#v, want none for an empty id", m["prompt_cache_key"])
	}
	long := strings.Repeat("漢", 70)
	m = run(`{"model":"m","max_tokens":10,"metadata":{"user_id":"` + long + `"},"messages":[{"role":"user","content":"hi"}]}`)
	want := strings.Repeat("漢", 64)
	if m["user"] != want {
		t.Fatalf("user truncated to %d runes, want 64 whole characters", len([]rune(m["user"].(string))))
	}
	if m["prompt_cache_key"] != want {
		t.Fatalf("prompt_cache_key = %#v, want the same 64-character cut", m["prompt_cache_key"])
	}
}

// A tool's description is copied on key presence — an empty one the caller
// spelled out survives — and strict is read as truthiness, so any non-empty
// value (the string "true" included) switches strict on.
func TestMessagesToResponsesToolDescriptionPresenceAndStrictTruthiness(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"fn","description":"","input_schema":{"type":"object"}},{"name":"fn2","strict":"true","input_schema":{"type":"object"}}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	first := tools[0].(map[string]any)
	if v, present := first["description"]; !present || v != "" {
		t.Fatalf("description = %#v (present=%v), want an explicit empty string", v, present)
	}
	second := tools[1].(map[string]any)
	if second["strict"] != true {
		t.Fatalf("strict = %#v, want truthiness to read the string as true", second["strict"])
	}
}

// A custom_tool_call and its output are history the target must see: the call
// folds to a tool_use whose input wraps the freeform payload as {"content":
// ...} — the shape the reference's request chain normalizes it to — and the
// output folds to the same tool_result a function_call_output produces.
func TestResponsesToMessagesKeepsCustomToolCallAndOutput(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"custom_tool_call","call_id":"cc1","name":"shell","input":"ls -la"},{"type":"custom_tool_call_output","call_id":"cc1","output":"total 0"}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v, want the call and its result to survive", msgs)
	}
	call := msgs[0].(map[string]any)
	if call["role"] != "assistant" {
		t.Fatalf("call message = %#v, want an assistant turn", call)
	}
	block := call["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_use" || block["id"] != "cc1" || block["name"] != "shell" {
		t.Fatalf("tool_use block = %#v", block)
	}
	input, _ := block["input"].(map[string]any)
	if input["content"] != "ls -la" {
		t.Fatalf("tool_use input = %#v, want the payload wrapped as content", block["input"])
	}
	result := msgs[1].(map[string]any)
	if result["role"] != "user" {
		t.Fatalf("result message = %#v, want a user turn", result)
	}
	tr := result["content"].([]any)[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != "cc1" || tr["content"] != "total 0" {
		t.Fatalf("tool_result block = %#v", tr)
	}

	// An empty payload normalizes to an empty input object, as it does for a
	// function_call with no arguments.
	empty := `{"model":"m","max_output_tokens":10,"input":[{"type":"custom_tool_call","call_id":"cc2","name":"shell","input":""}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	block = m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	input, _ = block["input"].(map[string]any)
	if len(input) != 0 {
		t.Fatalf("input = %#v, want an empty object for an empty payload", block["input"])
	}
}

// A text.format schema rides output_format unpacked and filtered: $refs are
// inlined against their definitions, the keywords the target's grammar
// cannot compile fold into descriptions, and oneOf becomes anyOf — the one
// combinator it accepts.
func TestResponsesToMessagesFiltersOutputSchema(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object","properties":{"events":{"type":"array","items":{"$ref":"#/$defs/Event"},"uniqueItems":true,"minItems":1},"score":{"type":"number","minimum":0},"tag":{"oneOf":[{"type":"string","maxLength":4},{"type":"number"}]}},"required":["events"],"$defs":{"Event":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}},"additionalProperties":false}}}}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	of, ok := m["output_format"].(map[string]any)
	if !ok || of["type"] != "json_schema" {
		t.Fatalf("output_format = %#v", m["output_format"])
	}
	schema := of["schema"].(map[string]any)
	if _, present := schema["$defs"]; present {
		t.Fatalf("$defs not unpacked: %#v", schema["$defs"])
	}
	props := schema["properties"].(map[string]any)
	events := props["events"].(map[string]any)
	items := events["items"].(map[string]any)
	if items["type"] != "object" {
		t.Fatalf("inlined items = %#v, want the Event definition", items)
	}
	if _, present := items["properties"]; !present {
		t.Fatalf("inlined items = %#v, want the Event properties", items)
	}
	if desc := events["description"]; desc != "Note: minimum number of items: 1, all array items must be unique." {
		t.Fatalf("events description = %#v, want the folded constraints", desc)
	}
	score := props["score"].(map[string]any)
	if desc := score["description"]; desc != "Note: minimum value: 0." {
		t.Fatalf("score description = %#v", desc)
	}
	tag := props["tag"].(map[string]any)
	anyOf, ok := tag["anyOf"].([]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("tag = %#v, want oneOf folded into anyOf", tag)
	}
	if _, present := tag["oneOf"]; present {
		t.Fatalf("tag kept its oneOf: %#v", tag)
	}
	if desc := anyOf[0].(map[string]any)["description"]; desc != "Note: maximum length: 4." {
		t.Fatalf("anyOf member description = %#v", desc)
	}
}

// System-role items are lifted to the system parameter as text blocks —
// never folded into user content, never rendered as a bare string array —
// and adjacent user turns merge into one, as the target requires alternating
// roles.
func TestResponsesToMessagesLiftsSystemAndMergesUserTurns(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"instructions":"You are helpful.","input":[{"type":"message","role":"system","content":[{"type":"input_text","text":"be nice"},{"type":"input_text","text":""},{"type":"input_text","text":"and brief"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"q1"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"q2"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	system, ok := m["system"].([]any)
	if !ok || len(system) != 3 {
		t.Fatalf("system = %#v, want the instructions then the lifted blocks", m["system"])
	}
	if system[0].(map[string]any)["text"] != "You are helpful." {
		t.Fatalf("system[0] = %#v, want the instructions first", system[0])
	}
	if system[1].(map[string]any)["text"] != "be nice" || system[2].(map[string]any)["text"] != "and brief" {
		t.Fatalf("system blocks = %#v, want the nonempty text parts", system)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want the user turns merged", msgs)
	}
	user := msgs[0].(map[string]any)
	if user["role"] != "user" {
		t.Fatalf("merged turn = %#v", user)
	}
	parts := user["content"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["text"] != "q1" || parts[1].(map[string]any)["text"] != "q2" {
		t.Fatalf("merged content = %#v, want both turns' text blocks", parts)
	}
}

// The encrypted_content the reference's bridge writes is a JSON array of
// signed thinking blocks; replaying it verbatim is what lets the target
// verify the chain of thought. Unsigned blocks are not replayable, an opaque
// blob is skipped rather than forwarded, and summary text never rides along
// when blocks replay.
func TestResponsesToMessagesReplaysEncryptedThinkingBlocks(t *testing.T) {
	enc := `[{"type":"thinking","thinking":"why","signature":"SIG1"},{"type":"redacted_thinking","data":"OPQ"}]`
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"sum text"}],"encrypted_content":` + encodeJSONString(t, enc) + `},{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
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
	blocks := msgs[0].(map[string]any)["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("blocks = %#v, want the decoded thinking and redacted blocks", blocks)
	}
	th := blocks[0].(map[string]any)
	if th["type"] != "thinking" || th["thinking"] != "why" || th["signature"] != "SIG1" {
		t.Fatalf("thinking block = %#v", th)
	}
	rd := blocks[1].(map[string]any)
	if rd["type"] != "redacted_thinking" || rd["data"] != "OPQ" {
		t.Fatalf("redacted block = %#v", rd)
	}

	// Unsigned thinking and data-less redacted blocks do not replay.
	partial := encodeJSONString(t, `[{"type":"thinking","thinking":"unsigned"},{"type":"redacted_thinking"},{"type":"thinking","thinking":"ok","signature":"S"}]`)
	body = `{"model":"m","max_output_tokens":10,"input":[{"type":"reasoning","encrypted_content":` + partial + `},{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks = m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 1 || blocks[0].(map[string]any)["signature"] != "S" {
		t.Fatalf("blocks = %#v, want only the verifiable block", blocks)
	}

	// An opaque blob is not an array this fold wrote: skipped, not forwarded.
	opaque := encodeJSONString(t, "###notjson###")
	body = `{"model":"m","max_output_tokens":10,"input":[{"type":"reasoning","encrypted_content":` + opaque + `},{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs = m["messages"].([]any)
	if len(msgs) != 1 || msgs[0].(map[string]any)["role"] != "user" {
		t.Fatalf("messages = %#v, want the opaque blob dropped", msgs)
	}
}

// A reasoning item and the calls it precedes share one assistant turn, and a
// tool result joins the user turn that follows it — two consecutive messages
// of either role are a request the target rejects.
func TestResponsesToMessagesMergesCallsIntoOneTurn(t *testing.T) {
	enc := encodeJSONString(t, `[{"type":"thinking","thinking":"t","signature":"S"}]`)
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"reasoning","encrypted_content":` + enc + `},{"type":"function_call","call_id":"f1","name":"fn","arguments":"{\"a\":1}"},{"type":"function_call","call_id":"f2","name":"fn","arguments":"{}"},{"type":"function_call_output","call_id":"f1","output":"ok"},{"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v, want one assistant and one user turn", msgs)
	}
	asst := msgs[0].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("first turn = %#v", asst)
	}
	blocks := asst["content"].([]any)
	var types []string
	for _, b := range blocks {
		types = append(types, b.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "thinking,tool_use,tool_use" {
		t.Fatalf("assistant blocks = %v, want thinking then both calls", types)
	}
	user := msgs[1].(map[string]any)["content"].([]any)
	if len(user) != 2 || user[0].(map[string]any)["type"] != "tool_result" || user[1].(map[string]any)["text"] != "next" {
		t.Fatalf("user blocks = %#v, want the tool result then the turn's text", user)
	}
}

// encodeJSONString quotes s as a JSON string literal for embedding in a test
// body.
func encodeJSONString(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// A tool output part shaped as image_url rides an image block the way an
// input_image part does, and an output whose text parts are all empty falls
// back to serializing the whole part list rather than an empty string.
func TestResponsesToMessagesToolOutputEdgeParts(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"function_call_output","call_id":"c3","output":[{"type":"image_url","image_url":"https://x/i.png"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tr := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	content, ok := tr["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("tool_result content = %#v, want the image block", tr["content"])
	}
	img := content[0].(map[string]any)
	if img["type"] != "image" {
		t.Fatalf("block = %#v, want an image block", img)
	}
	if src, _ := img["source"].(map[string]any); src["type"] != "url" || src["url"] != "https://x/i.png" {
		t.Fatalf("source = %#v, want the url source", img["source"])
	}

	empty := `{"model":"m","max_output_tokens":10,"input":[{"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":""},{"type":"input_text","text":""}]}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tr = m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	got, ok := tr["content"].(string)
	if !ok || got == "" || !strings.Contains(got, "input_text") {
		t.Fatalf("tool_result content = %#v, want the serialized part list", tr["content"])
	}
}

// An email-shaped user is a legal Responses value the target's metadata
// rejects, so it drops the metadata entirely rather than failing the request;
// a plain id still rides user_id.
func TestResponsesToMessagesDropsEmailUserMetadata(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"user":"someone@example.com","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["metadata"]; present {
		t.Fatalf("metadata = %#v, want an email-shaped user dropped", m["metadata"])
	}

	plain := `{"model":"m","max_output_tokens":10,"user":"u-123","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(plain))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	md, _ := m["metadata"].(map[string]any)
	if md == nil || md["user_id"] != "u-123" {
		t.Fatalf("metadata = %#v, want the plain id kept", m["metadata"])
	}
}

// Anthropic's tool_use_id pattern is [a-zA-Z0-9_-]: a call id carrying
// anything else is rewritten — on the call and its result alike, so the pair
// still matches afterwards.
func TestResponsesToMessagesSanitizesToolUseIDs(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"function_call","call_id":"c:1","name":"fn","arguments":"{}"},{"type":"function_call_output","call_id":"c:1","output":"ok"}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	call := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if call["type"] != "tool_use" || call["id"] != "c_1" {
		t.Fatalf("tool_use = %#v, want the id sanitized to c_1", call)
	}
	result := msgs[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if result["type"] != "tool_result" || result["tool_use_id"] != "c_1" {
		t.Fatalf("tool_result = %#v, want the matching sanitized id", result)
	}
}

// A compaction threshold named as a numeric string still reaches the target —
// the reference runs the trigger value through int(), which accepts one.
func TestMessagesToResponsesNumericStringThreshold(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"context_management":{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":"150000"}}]},"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	cm := m["context_management"].([]any)[0].(map[string]any)
	if cm["compact_threshold"] != float64(150000) {
		t.Fatalf("compaction entry = %#v, want the string threshold parsed", cm)
	}
}

// A message that follows a call still reads ahead of it: within one assistant
// turn the thinking comes first, the turn's text, then the calls.
func TestResponsesToMessagesTextPrecedesCallsInTurn(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},{"type":"function_call","call_id":"f1","name":"fn","arguments":"{}"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks := m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	var types []string
	for _, b := range blocks {
		types = append(types, b.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "text,tool_use" {
		t.Fatalf("blocks = %v, want the text ahead of the call", types)
	}

	withThinking := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},{"type":"reasoning","encrypted_content":` + encodeJSONString(t, `[{"type":"thinking","thinking":"t","signature":"S"}]`) + `},{"type":"function_call","call_id":"f1","name":"fn","arguments":"{}"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(withThinking))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks = m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	types = nil
	for _, b := range blocks {
		types = append(types, b.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "thinking,text,tool_use" {
		t.Fatalf("blocks = %v, want thinking, text, then the call", types)
	}
}

// Messages requires input_schema on every tool, so a Responses function tool
// that named no parameters — or an empty or type-less one — still folds to a
// schema, and a custom tool carries the content-property schema the reference
// synthesizes, with its grammar embedded in the description.
func TestResponsesToMessagesToolsAlwaysCarrySchema(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"function","name":"noparams"},{"type":"function","name":"emptyobj","parameters":{}},{"type":"function","name":"notype","parameters":{"properties":{"a":{"type":"string"}}}},{"type":"custom","name":"sh","description":"run things","format":{"type":"grammar","syntax":"lark","definition":"root: hi"}}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 4 {
		t.Fatalf("tools = %#v, want all four folded", tools)
	}
	schemaOf := func(i int) map[string]any {
		return tools[i].(map[string]any)["input_schema"].(map[string]any)
	}
	if s := schemaOf(0); s["type"] != "object" || len(s["properties"].(map[string]any)) != 0 {
		t.Fatalf("noparams schema = %#v, want a bare object with empty properties", s)
	}
	if s := schemaOf(1); s["type"] != "object" || len(s["properties"].(map[string]any)) != 0 {
		t.Fatalf("emptyobj schema = %#v, want a bare object with empty properties", s)
	}
	if s := schemaOf(2); s["type"] != "object" {
		t.Fatalf("notype schema = %#v, want the type stated", s)
	}
	custom := tools[3].(map[string]any)
	if custom["name"] != "sh" {
		t.Fatalf("custom tool = %#v", custom)
	}
	if desc := custom["description"]; desc != "run things\n\nFormat:\n```lark\nroot: hi\n```" {
		t.Fatalf("custom description = %#v, want the grammar embedded", desc)
	}
	cs := custom["input_schema"].(map[string]any)
	if cs["type"] != "object" {
		t.Fatalf("custom schema = %#v", cs)
	}
	props := cs["properties"].(map[string]any)
	content := props["content"].(map[string]any)
	if content["type"] != "string" || content["description"] != "The sh content following the specified format" {
		t.Fatalf("content property = %#v", content)
	}
	if req, _ := cs["required"].([]any); len(req) != 1 || req[0] != "content" {
		t.Fatalf("required = %#v, want content", cs["required"])
	}
}

// An MCP server tool rides the mcp_servers list with the entry shape the
// reference emits — url, label, the allowed-tools configuration and a stripped
// bearer token — instead of being dropped as a Responses-only tool.
func TestResponsesToMessagesMapsMcpServerTool(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"mcp","server_label":"deepwiki","server_url":"https://mcp.example.com/s","allowed_tools":["search"],"headers":{"Authorization":"Bearer tok_123"}}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
		t.Fatalf("tools = %#v, want the mcp tool out of the tools list", tools)
	}
	servers := m["mcp_servers"].([]any)
	if len(servers) != 1 {
		t.Fatalf("mcp_servers = %#v, want one server entry", m["mcp_servers"])
	}
	server := servers[0].(map[string]any)
	if server["type"] != "url" || server["url"] != "https://mcp.example.com/s" || server["name"] != "deepwiki" {
		t.Fatalf("server = %#v", server)
	}
	tc := server["tool_configuration"].(map[string]any)
	if allowed := tc["allowed_tools"].([]any); len(allowed) != 1 || allowed[0] != "search" {
		t.Fatalf("tool_configuration = %#v", tc)
	}
	if server["authorization_token"] != "tok_123" {
		t.Fatalf("authorization_token = %#v, want the bearer prefix stripped", server["authorization_token"])
	}
}

// A web_search tool's tuning rides along: the context size buys a max_uses
// budget and a user location folds to the approximate shape the target reads.
func TestResponsesToMessagesWebSearchOptions(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"web_search_preview","search_context_size":"high","user_location":{"type":"approximate","approximate":{"city":"Paris","timezone":"Europe/Paris","precision":"dropped"}}}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "web_search_20250305" || tool["name"] != "web_search" {
		t.Fatalf("tool = %#v, want the reference's search tool shape", tool)
	}
	if tool["max_uses"] != float64(10) {
		t.Fatalf("max_uses = %#v, want the high budget", tool["max_uses"])
	}
	loc := tool["user_location"].(map[string]any)
	if loc["type"] != "approximate" || loc["city"] != "Paris" || loc["timezone"] != "Europe/Paris" {
		t.Fatalf("user_location = %#v, want the approximate fields", loc)
	}
	if _, present := loc["precision"]; present {
		t.Fatalf("user_location = %#v, want unrecognized fields dropped", loc)
	}

	low := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"web_search","search_context_size":"low"}]}`
	out, err = ConvertRequest(Responses, Messages, []byte(low))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool = m["tools"].([]any)[0].(map[string]any)
	if tool["max_uses"] != float64(1) {
		t.Fatalf("max_uses = %#v, want the low budget", tool["max_uses"])
	}
	if _, present := tool["user_location"]; present {
		t.Fatalf("user_location = %#v, want none when the tool named none", tool["user_location"])
	}
}

// A folded tool's schema is the shape the target accepts: the type stated as
// object, properties always present, legacy definitions inlined, and no keys
// outside the recognised set — $schema and friends drop.
func TestResponsesToMessagesSanitizesToolSchema(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"function","name":"fn","parameters":{"type":"object","$schema":"http://x","title":"T","additionalProperties":false,"definitions":{"T":{"type":"string"}},"properties":{"a":{"$ref":"#/definitions/T"}},"required":["a"]}}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	schema := m["tools"].([]any)[0].(map[string]any)["input_schema"].(map[string]any)
	for _, dropped := range []string{"$schema", "title", "definitions"} {
		if _, present := schema[dropped]; present {
			t.Fatalf("schema kept %s: %#v", dropped, schema)
		}
	}
	if schema["type"] != "object" || schema["additionalProperties"] != false {
		t.Fatalf("schema = %#v, want the type and additionalProperties kept", schema)
	}
	props := schema["properties"].(map[string]any)
	if a := props["a"].(map[string]any); a["type"] != "string" {
		t.Fatalf("inlined property = %#v, want the legacy ref resolved", props["a"])
	}
	if req := schema["required"].([]any); len(req) != 1 || req[0] != "a" {
		t.Fatalf("required = %#v", schema["required"])
	}
}

// Dict tool_choice shapes normalize the way the reference does: none stays
// none, tool/any/required force any tool, and a choice nothing maps to is
// omitted rather than spelled out as auto.
func TestResponsesToMessagesToolChoiceShapes(t *testing.T) {
	run := func(tc string) map[string]any {
		body := `{"model":"m","max_output_tokens":10,"tool_choice":` + tc + `,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"function","name":"fn","parameters":{"type":"object"}}]}`
		out, err := ConvertRequest(Responses, Messages, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	for shape, want := range map[string]string{
		`{"type":"none"}`:     "none",
		`{"type":"tool"}`:     "any",
		`{"type":"any"}`:      "any",
		`{"type":"required"}`: "any",
	} {
		choice := run(shape)["tool_choice"].(map[string]any)
		if choice["type"] != want {
			t.Fatalf("%s -> %#v, want %s", shape, choice, want)
		}
	}
	for _, unknown := range []string{`"bogus"`, `{"type":"file_search"}`} {
		if _, present := run(unknown)["tool_choice"]; present {
			t.Fatalf("%s folded to an explicit choice, want it omitted", unknown)
		}
	}
}

// An effort the reference cannot map is refused — an unmapped or adaptive
// string names no budget, and silently dropping it would change what the
// request asks for.
func TestResponsesToMessagesRejectsUnmappedEffort(t *testing.T) {
	for _, effort := range []string{`"extreme"`, `"adaptive"`} {
		body := `{"model":"m","max_output_tokens":10,"reasoning":{"effort":` + effort + `},"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
		_, err := ConvertRequest(Responses, Messages, []byte(body))
		var unsup UnsupportedParamError
		if !errors.As(err, &unsup) {
			t.Fatalf("effort %s: err = %v, want UnsupportedParamError", effort, err)
		}
		if unsup.Param != "reasoning.effort" {
			t.Fatalf("param = %q, want reasoning.effort", unsup.Param)
		}
	}
}

// A folded function tool carries the shape the reference emits — type custom,
// a description always stated, the bridge-defaulted empty string included —
// and a cache-control marker set on the tool survives the fold.
func TestResponsesToMessagesToolCarriesTypeAndCacheControl(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}],"tools":[{"type":"function","name":"fn","parameters":{"type":"object"},"cache_control":{"type":"ephemeral"}}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "custom" {
		t.Fatalf("tool = %#v, want the reference's custom tool type", tool)
	}
	if v, present := tool["description"]; !present || v != "" {
		t.Fatalf("description = %#v (present=%v), want an explicit empty string", v, present)
	}
	cc, _ := tool["cache_control"].(map[string]any)
	if cc == nil || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control = %#v, want the marker carried", tool["cache_control"])
	}
}

// A data URL this fold cannot parse is rejected rather than folded into a
// base64 block with nothing in it: no ;base64, separator or a second one is
// not a data URL the target reads. An empty media type, though, is the value
// the caller spelled out and rides as itself.
func TestResponsesToMessagesRejectsMalformedDataURL(t *testing.T) {
	for _, bad := range []string{
		`{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png,%41%42"}]}`,
		`{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,QQ;base64,BB"}]}`,
	} {
		body := `{"model":"m","max_output_tokens":10,"input":[` + bad + `]}`
		_, err := ConvertRequest(Responses, Messages, []byte(body))
		var conv ConversionError
		if !errors.As(err, &conv) {
			t.Fatalf("input %s: err = %v, want a ConversionError", bad, err)
		}
	}

	empty := `{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:;base64,QQ"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	img := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	src, _ := img["source"].(map[string]any)
	if src["type"] != "base64" || src["media_type"] != "" || src["data"] != "QQ" {
		t.Fatalf("source = %#v, want the empty media type carried as-is", img["source"])
	}
}

// Instructions alone ride the same one-block list — the reference's fold
// renders the system parameter as text blocks even when nothing was lifted
// from the input, never as a bare string.
func TestResponsesToMessagesInstructionsRideTextBlocks(t *testing.T) {
	body := `{"model":"m","max_output_tokens":10,"instructions":"top","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]}]}`
	out, err := ConvertRequest(Responses, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	system, ok := m["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system = %#v, want a one-block list", m["system"])
	}
	if b := system[0].(map[string]any); b["type"] != "text" || b["text"] != "top" {
		t.Fatalf("system block = %#v", b)
	}
}
