package protocol

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/internal/thinksig"
)

// This file pins the Messages→Responses folds a LiteLLM v1.100.1 audit found
// diverging from the reference (litellm/llms/anthropic/chat/transformation.py
// feeding litellm/responses/litellm_completion_transformation/transformation.py).
// Each case was reproduced against the reference before the fix.

// respJSON runs a non-streaming fold and decodes it.
func respJSON(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := ConvertResponse(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// respNum reads a JSON number the decoder delivered as float64.
func respNum(t *testing.T, v any) int {
	t.Helper()
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("not a number: %#v", v)
	}
	return int(f)
}

// respEvents splits a folded SSE stream into decoded data objects.
func respEvents(t *testing.T, raw string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, chunk := range strings.Split(raw, "\n\n") {
		var data string
		for _, line := range strings.Split(chunk, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(data), &obj); err != nil {
			continue
		}
		out = append(out, obj)
	}
	return out
}

// LiteLLM's calculate_usage folds cache_creation and cache_read into
// prompt_tokens before the responses usage is built, and the details object is
// always present — cached_tokens is simply 0 without a cache. The non-streaming
// fold used to read only Anthropic's input_tokens and to omit the details
// entirely without a cache, disagreeing with both litellm and the fold's own
// streaming path.
func TestMessagesToResponsesUsageFoldsCacheTokens(t *testing.T) {
	m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":100,"output_tokens":40,"cache_read_input_tokens":50,"cache_creation_input_tokens":20}}`)
	usage, ok := m["usage"].(map[string]any)
	if !ok {
		t.Fatalf("usage = %#v", m["usage"])
	}
	if got := respNum(t, usage["input_tokens"]); got != 170 {
		t.Fatalf("input_tokens = %d, want 170 (100+50+20)", got)
	}
	if got := respNum(t, usage["total_tokens"]); got != 210 {
		t.Fatalf("total_tokens = %d, want 210", got)
	}
	details, ok := usage["input_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("input_tokens_details = %#v", usage["input_tokens_details"])
	}
	if got := respNum(t, details["cached_tokens"]); got != 50 {
		t.Fatalf("cached_tokens = %d, want 50", got)
	}
	if got := respNum(t, details["text_tokens"]); got != 100 {
		t.Fatalf("text_tokens = %d, want 100", got)
	}
	if got := respNum(t, details["cache_write_tokens"]); got != 20 {
		t.Fatalf("cache_write_tokens = %d, want 20", got)
	}

	// Without a cache the details object still exists, with cached_tokens 0.
	m = respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"output_tokens":5}}`)
	usage = m["usage"].(map[string]any)
	details, ok = usage["input_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("input_tokens_details missing without a cache: %#v", usage)
	}
	if got := respNum(t, details["cached_tokens"]); got != 0 {
		t.Fatalf("cached_tokens = %d, want 0", got)
	}

	// The streaming terminal carries the same fold.
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":100,"cache_read_input_tokens":50,"cache_creation_input_tokens":20}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var completed map[string]any
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] == "response.completed" {
			completed, _ = e["response"].(map[string]any)
		}
	}
	if completed == nil {
		t.Fatalf("no response.completed:\n%s", dst.String())
	}
	usage, ok = completed["usage"].(map[string]any)
	if !ok {
		t.Fatalf("stream usage = %#v", completed["usage"])
	}
	if got := respNum(t, usage["input_tokens"]); got != 170 {
		t.Fatalf("stream input_tokens = %d, want 170", got)
	}
	details, ok = usage["input_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("stream input_tokens_details = %#v", usage["input_tokens_details"])
	}
	if got := respNum(t, details["cached_tokens"]); got != 50 {
		t.Fatalf("stream cached_tokens = %d, want 50", got)
	}
}

// LiteLLM merges every thinking and redacted_thinking block of the turn into
// ONE reasoning item: the concatenated thinking text rides an output_text part
// in content (not summary), and the blocks carrying opaque bytes are re-encoded
// as the compact JSON array encrypted_content is on the wire — the form
// litellm's replay decoder accepts.
func TestMessagesToResponsesMergesThinkingIntoOneReasoningItem(t *testing.T) {
	m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"end_turn","content":[
		{"type":"thinking","thinking":"Let me think.","signature":"sig1"},
		{"type":"text","text":"Hello"},
		{"type":"redacted_thinking","data":"REDDATA1"},
		{"type":"text","text":" world"},
		{"type":"thinking","thinking":" More thought.","signature":"sig2"}
	]}`)
	output, _ := m["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v, want reasoning+message", output)
	}
	item, _ := output[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("first item = %#v, want reasoning", output[0])
	}
	if _, has := item["summary"]; has {
		t.Fatalf("reasoning carries a summary: %#v", item)
	}
	parts, _ := item["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("content = %#v, want one output_text part", item["content"])
	}
	part, _ := parts[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "Let me think. More thought." {
		t.Fatalf("part = %#v", part)
	}
	if ann, ok := part["annotations"].([]any); !ok || len(ann) != 0 {
		t.Fatalf("annotations = %#v, want []", part["annotations"])
	}
	if item["role"] != "assistant" || item["status"] != "completed" {
		t.Fatalf("item = %#v", item)
	}
	if id, _ := item["id"].(string); !strings.HasPrefix(id, "rs_") {
		t.Fatalf("id = %#v, want rs_ prefix", item["id"])
	}
	var blocks []any
	if err := json.Unmarshal([]byte(item["encrypted_content"].(string)), &blocks); err != nil {
		t.Fatalf("encrypted_content is not a JSON array: %v (%#v)", err, item["encrypted_content"])
	}
	if len(blocks) != 3 {
		t.Fatalf("encrypted blocks = %#v, want the three opaque blocks", blocks)
	}
	b0, _ := blocks[0].(map[string]any)
	if b0["type"] != "thinking" || b0["thinking"] != "Let me think." || b0["signature"] != "sig1" {
		t.Fatalf("block 0 = %#v", b0)
	}
	b1, _ := blocks[1].(map[string]any)
	if b1["type"] != "redacted_thinking" || b1["data"] != "REDDATA1" {
		t.Fatalf("block 1 = %#v", b1)
	}
	b2, _ := blocks[2].(map[string]any)
	if b2["type"] != "thinking" || b2["thinking"] != " More thought." || b2["signature"] != "sig2" {
		t.Fatalf("block 2 = %#v", b2)
	}

	// A redacted-only turn keeps its reasoning item with empty content and the
	// array-encoded payload.
	m = respJSON(t, `{"id":"m2","type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"RD"}]}`)
	output, _ = m["output"].([]any)
	if len(output) != 2 || output[0].(map[string]any)["type"] != "reasoning" {
		t.Fatalf("output = %#v, want reasoning+message", output)
	}
	item, _ = output[0].(map[string]any)
	if parts, _ := item["content"].([]any); len(parts) != 0 {
		t.Fatalf("redacted-only content = %#v, want []", item["content"])
	}
	if err := json.Unmarshal([]byte(item["encrypted_content"].(string)), &blocks); err != nil {
		t.Fatalf("encrypted_content is not a JSON array: %v", err)
	}
	if len(blocks) != 1 || blocks[0].(map[string]any)["data"] != "RD" {
		t.Fatalf("encrypted blocks = %#v, want the redacted payload", blocks)
	}

	// A thinking block with neither text nor opaque bytes yields no item at all.
	m = respJSON(t, `{"id":"m3","type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"","signature":""},{"type":"text","text":"x"}]}`)
	output, _ = m["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["type"] != "message" {
		t.Fatalf("output = %#v, want the message item only", output)
	}
}

// LiteLLM folds server_tool_use through the same tool_call path as tool_use,
// so a server-side call arrives as a function_call item instead of being
// dropped. The streaming fold used to drop the block's events too.
func TestMessagesToResponsesServerToolUseBecomesFunctionCall(t *testing.T) {
	m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"end_turn","content":[
		{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"cats"}}
	]}`)
	output, _ := m["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v, want message+function_call", output)
	}
	fc, _ := output[1].(map[string]any)
	if fc["type"] != "function_call" {
		t.Fatalf("item = %#v, want function_call", fc)
	}
	if fc["call_id"] != "srv_1" || fc["id"] != "srv_1" || fc["name"] != "web_search" {
		t.Fatalf("item = %#v", fc)
	}
	if fc["arguments"] != `{"query":"cats"}` {
		t.Fatalf("arguments = %#v", fc["arguments"])
	}
	if fc["status"] != "completed" {
		t.Fatalf("status = %#v", fc["status"])
	}

	// The streaming fold relays the block's arguments deltas.
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":9}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \"cats\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var deltas []string
	var completed map[string]any
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] == "response.function_call_arguments.delta" {
			deltas = append(deltas, e["delta"].(string))
		}
		if e["type"] == "response.completed" {
			completed, _ = e["response"].(map[string]any)
		}
	}
	if strings.Join(deltas, "") != `{"query": "cats"}` {
		t.Fatalf("server tool deltas = %q, want the streamed arguments", deltas)
	}
	if completed == nil {
		t.Fatalf("no response.completed:\n%s", dst.String())
	}
	outItems, _ := completed["output"].([]any)
	var sawCall bool
	for _, it := range outItems {
		im, _ := it.(map[string]any)
		if im["type"] == "function_call" && im["call_id"] == "srv_1" {
			sawCall = true
		}
	}
	if !sawCall {
		t.Fatalf("completed output has no srv_1 function_call: %#v", completed["output"])
	}
}

// LiteLLM joins every text block into exactly one message item, and a turn
// with no text at all still gets one — its output_text part then carries no
// text.
func TestMessagesToResponsesSingleMessageItemJoinsText(t *testing.T) {
	m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"end_turn","content":[
		{"type":"text","text":"Hello"},
		{"type":"text","text":" world"}
	]}`)
	output, _ := m["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %#v, want exactly one message item", output)
	}
	item, _ := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item = %#v", item)
	}
	part, _ := item["content"].([]any)[0].(map[string]any)
	if part["text"] != "Hello world" {
		t.Fatalf("text = %#v, want joined text", part["text"])
	}

	// A tool-only turn still carries the (textless) message item.
	m = respJSON(t, `{"id":"m2","type":"message","role":"assistant","stop_reason":"tool_use","content":[
		{"type":"tool_use","id":"toolu_9","name":"get_weather","input":{"city":"SF"}}
	]}`)
	output, _ = m["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v, want message+function_call", output)
	}
	item, _ = output[0].(map[string]any)
	if item["type"] != "message" {
		t.Fatalf("first item = %#v, want the always-on message", item)
	}
	part, _ = item["content"].([]any)[0].(map[string]any)
	if part["type"] != "output_text" {
		t.Fatalf("part = %#v", part)
	}
	if txt, has := part["text"]; has && txt != nil {
		t.Fatalf("text = %#v, want absent on a textless turn", txt)
	}
}

// LiteLLM stamps the merged items with the fields the Responses schema
// expects: the message item carries id/status and its part annotations; the
// function_call item carries id (= call_id) and status. An incomplete turn
// carries the incomplete status on the reasoning and message items too.
//
// created_at is deliberately absent: litellm fills it from the wall clock
// (transformation.py sets model_response.created = int(time.time())) and an
// Anthropic Messages body carries no timestamp to derive it from, while this
// library's contract is offline with no clock.
func TestMessagesToResponsesItemsCarryStatusAndAnnotations(t *testing.T) {
	m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"tool_use","content":[
		{"type":"text","text":"hi"},
		{"type":"tool_use","id":"toolu_9","name":"get_weather","input":{"city":"SF"}}
	]}`)
	output, _ := m["output"].([]any)
	msg, _ := output[0].(map[string]any)
	if msg["status"] != "completed" {
		t.Fatalf("message status = %#v", msg["status"])
	}
	if id, _ := msg["id"].(string); !strings.HasPrefix(id, "msg_") {
		t.Fatalf("message id = %#v, want msg_ prefix", msg["id"])
	}
	part, _ := msg["content"].([]any)[0].(map[string]any)
	if ann, ok := part["annotations"].([]any); !ok || len(ann) != 0 {
		t.Fatalf("annotations = %#v, want []", part["annotations"])
	}
	fc, _ := output[1].(map[string]any)
	if fc["id"] != "toolu_9" || fc["call_id"] != "toolu_9" {
		t.Fatalf("function_call ids = %#v / %#v, want call_id on both", fc["id"], fc["call_id"])
	}
	if fc["status"] != "completed" {
		t.Fatalf("function_call status = %#v", fc["status"])
	}

	// An incomplete turn marks the reasoning and message items incomplete.
	m = respJSON(t, `{"id":"m2","type":"message","role":"assistant","stop_reason":"max_tokens","content":[
		{"type":"thinking","thinking":"half a thought","signature":"s"},
		{"type":"text","text":"partial"}
	]}`)
	output, _ = m["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v", output)
	}
	if got := output[0].(map[string]any)["status"]; got != "incomplete" {
		t.Fatalf("reasoning status = %#v, want incomplete", got)
	}
	if got := output[1].(map[string]any)["status"]; got != "incomplete" {
		t.Fatalf("message status = %#v, want incomplete", got)
	}
}

// On this chain litellm v1.100.1 leaves incomplete_details null — the
// Responses status alone says the turn did not finish. The fold used to
// synthesize a reason (max_output_tokens / content_filter) litellm does not
// emit, on both the non-streaming body and the streaming terminal.
func TestMessagesToResponsesIncompleteHasNoIncompleteDetails(t *testing.T) {
	for _, stop := range []string{"max_tokens", "refusal"} {
		m := respJSON(t, `{"id":"m1","type":"message","role":"assistant","stop_reason":"`+stop+`","content":[{"type":"text","text":"x"}]}`)
		if m["status"] != "incomplete" {
			t.Fatalf("stop_reason %s: status = %#v, want incomplete", stop, m["status"])
		}
		if _, has := m["incomplete_details"]; has {
			t.Fatalf("stop_reason %s: incomplete_details = %#v, want absent", stop, m["incomplete_details"])
		}
	}

	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":65536}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] != "response.completed" {
			continue
		}
		resp, _ := e["response"].(map[string]any)
		if resp["status"] != "incomplete" {
			t.Fatalf("stream status = %#v, want incomplete", resp["status"])
		}
		if _, has := resp["incomplete_details"]; has {
			t.Fatalf("stream incomplete_details = %#v, want absent", resp["incomplete_details"])
		}
	}
}

// messagesStream builds a minimal Messages SSE stream that stops with the
// given stop_reason.
func messagesStream(stop string) string {
	return strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"` + stop + `"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
}

// The reference's chat→responses streaming bridge closes EVERY turn with
// response.completed — a ceiling cut says so in the embedded response.status,
// never in the event type. The fold used to spell the terminal
// response.incomplete instead.
func TestMessagesToResponsesMaxTokensClosesAsResponseCompleted(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(messagesStream("max_tokens")), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(dst.String(), "response.incomplete") {
		t.Fatalf("ceiling cut closed as response.incomplete:\n%s", dst.String())
	}
	var terminal map[string]any
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] == "response.completed" {
			terminal, _ = e["response"].(map[string]any)
		}
	}
	if terminal == nil {
		t.Fatalf("no response.completed terminal:\n%s", dst.String())
	}
	if terminal["status"] != "incomplete" {
		t.Fatalf("terminal status = %#v, want incomplete", terminal["status"])
	}
	if _, has := terminal["incomplete_details"]; has {
		t.Fatalf("incomplete_details = %#v, want absent", terminal["incomplete_details"])
	}
}

// A safeguard refusal is not a clean stop on the streaming path either: the
// reference maps it refusal → content_filter → status=incomplete. The fold
// used to leave it status=completed, disagreeing with both the reference and
// its own non-streaming direction.
func TestMessagesToResponsesRefusalStopIsIncomplete(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(messagesStream("refusal")), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var terminal map[string]any
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] == "response.completed" {
			terminal, _ = e["response"].(map[string]any)
		}
	}
	if terminal == nil {
		t.Fatalf("no response.completed terminal:\n%s", dst.String())
	}
	if terminal["status"] != "incomplete" {
		t.Fatalf("refusal status = %#v, want incomplete", terminal["status"])
	}
}

// A typed response.failed still closed nothing for the Messages client — the
// fold swallowed it. The reference's adapter closes the turn exactly like a
// completed one (message_delta + message_stop) while the failure itself stays
// visible to the host as StreamResult.InBandErr.
func TestResponsesToMessagesFailedClosesTheTurn(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"partial"}`,
		``,
		`event: response.failed`,
		`data: {"type":"response.failed","response":{"id":"r1","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	res, err := PipeStreamWith(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Terminal || res.Truncated || res.InBandErr == nil {
		t.Fatalf("result=%+v, want Terminal with InBandErr", res)
	}
	if !strings.Contains(dst.String(), `"stop_reason":"end_turn"`) || !strings.Contains(dst.String(), "message_stop") {
		t.Fatalf("failed turn not closed for the messages client:\n%s", dst.String())
	}
}

// message_start carries the fields an Anthropic message has before they are
// known — null stop_reason/stop_sequence and an all-zero usage — and a minted
// msg id, the shape the reference's adapter opens with.
func TestResponsesToMessagesMessageStartShape(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"resp_src123","model":"m"}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_src123","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var start map[string]any
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] == "message_start" {
			start, _ = e["message"].(map[string]any)
		}
	}
	if start == nil {
		t.Fatalf("no message_start:\n%s", dst.String())
	}
	if id, _ := start["id"].(string); !strings.HasPrefix(id, "msg_") {
		t.Fatalf("message id = %#v, want a minted msg_ id", start["id"])
	}
	for _, k := range []string{"stop_reason", "stop_sequence"} {
		if v, has := start[k]; !has || v != nil {
			t.Fatalf("%s = %#v (present=%v), want present and null", k, v, has)
		}
	}
	usage, ok := start["usage"].(map[string]any)
	if !ok {
		t.Fatalf("usage = %#v", start["usage"])
	}
	for _, k := range []string{"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"} {
		if got := respNum(t, usage[k]); got != 0 {
			t.Fatalf("usage.%s = %d, want 0", k, got)
		}
	}
	if start["type"] != "message" || start["role"] != "assistant" || start["model"] != "m" {
		t.Fatalf("message_start = %#v", start)
	}
}

// The reference never emits a signature for reasoning replayed to a Messages
// client — its thinking blocks stay unsigned. The fold used to re-emit
// encrypted_content as a signature_delta; it no longer does.
func TestResponsesToMessagesNoSignatureDelta(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"thinking..."}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","item_id":"rs_1","item":{"type":"reasoning","id":"rs_1","encrypted_content":"OPAQUE"}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(dst.String(), "signature_delta") {
		t.Fatalf("signature_delta emitted:\n%s", dst.String())
	}
	for _, want := range []string{"thinking_delta", "content_block_stop", "message_stop"} {
		if !strings.Contains(dst.String(), want) {
			t.Fatalf("missing %s:\n%s", want, dst.String())
		}
	}
}

// A thinking block opens unsigned, with the empty signature the reference's
// adapter puts on the wire from the first frame.
func TestResponsesToMessagesThinkingBlockStartsWithStandInSignature(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"thinking..."}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] != "content_block_start" {
			continue
		}
		block, _ := e["content_block"].(map[string]any)
		if block["type"] != "thinking" {
			continue
		}
		if block["thinking"] != "" || block["signature"] != thinksig.Synthetic {
			t.Fatalf("thinking block = %#v, want empty thinking and the fold's stand-in signature", block)
		}
		return
	}
	t.Fatalf("no thinking content_block_start:\n%s", dst.String())
}

// The reference's Messages adapter opens the text block the moment a message
// item is announced, so an item with no deltas still gets its (empty) block,
// and the calls after it land on the right content-block index.
func TestResponsesToMessagesOpensTextBlockOnItemAdded(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m0","role":"assistant","content":[]}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"t1","call_id":"t1","name":"f","arguments":""}}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"t1","delta":"{}"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[{"type":"function_call","id":"t1","call_id":"t1","name":"f","arguments":"{}"}]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	var textStart, toolStart map[string]any
	for _, e := range evs {
		if e["type"] != "content_block_start" {
			continue
		}
		block, _ := e["content_block"].(map[string]any)
		if block["type"] == "text" && textStart == nil {
			textStart = e
		}
		if block["type"] == "tool_use" {
			toolStart = e
		}
	}
	if textStart == nil {
		t.Fatalf("empty message item opened no text block:\n%s", dst.String())
	}
	if textStart["index"] != float64(0) {
		t.Fatalf("text block index = %#v, want 0", textStart["index"])
	}
	if toolStart == nil || toolStart["index"] != float64(1) {
		t.Fatalf("tool block should land at index 1 after the empty message: %#v", toolStart)
	}
}

// The reference's Messages adapter knows only message and function_call
// items: a custom_tool_call produces no Anthropic events at all, and it does
// not turn the stop into tool_use either.
func TestResponsesToMessagesDropsCustomToolCalls(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ct_1","call_id":"ct_1","name":"draw","input":""}}`,
		``,
		`event: response.custom_tool_call_input.delta`,
		`data: {"type":"response.custom_tool_call_input.delta","item_id":"ct_1","delta":"a circle"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[{"type":"custom_tool_call","id":"ct_1","call_id":"ct_1","name":"draw","input":"a circle"}]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	for _, unwanted := range []string{"tool_use", "content_block_delta"} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("custom_tool_call folded into %s:\n%s", unwanted, out)
		}
	}
	if !strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Fatalf("custom_tool_call turned the stop into tool_use:\n%s", out)
	}
}

// On this fold the reference's chat→responses bridge announces the REASONING
// item off the first thinking delta (streaming_iterator.py:817
// _ensure_output_item_for_chunk — the message item of a reasoning-first stream
// is never announced), then closes the reasoning stretch with its done trio at
// the first following chunk (streaming_iterator.py:926: summary text done,
// summary part done, reasoning item done). What it never emits is the
// reasoning part's .added — the part is only ever closed — and the opaque
// payload only ever rides the completed snapshot.
func TestMessagesToResponsesReasoningHasNoLifecycleEvents(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	added := eventsOfTypes(t, evs, "response.output_item.added")
	if len(added) != 1 {
		t.Fatalf("output_item.added count=%d, want the single reasoning item:\n%s", len(added), dst.String())
	}
	if item, _ := added[0]["item"].(map[string]any); item["type"] != "reasoning" {
		t.Fatalf("added item = %#v, want reasoning", item)
	}
	// The done trio closes the reasoning stretch, mid-stream, right before the
	// text delta that ended it.
	for _, typ := range []string{"response.reasoning_summary_text.done", "response.reasoning_summary_part.done"} {
		if got := eventsOfTypes(t, evs, typ); len(got) != 1 {
			t.Fatalf("%s count=%d, want 1:\n%s", typ, len(got), dst.String())
		}
	}
	if got := eventsOfTypes(t, evs, "response.reasoning_summary_part.added"); len(got) != 0 {
		t.Fatalf("response.reasoning_summary_part.added must not be emitted on this fold:\n%s", dst.String())
	}
	rsDeltas := eventsOfTypes(t, evs, "response.reasoning_summary_text.delta")
	if len(rsDeltas) != 1 || rsDeltas[0]["delta"] != "let me " {
		t.Fatalf("reasoning deltas = %#v", rsDeltas)
	}
	rsID, _ := rsDeltas[0]["item_id"].(string)
	if !strings.HasPrefix(rsID, "rs_") {
		t.Fatalf("reasoning delta item_id = %#v", rsDeltas[0]["item_id"])
	}
	var completed map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			completed, _ = e["response"].(map[string]any)
		}
	}
	output, _ := completed["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("snapshot output = %#v", completed["output"])
	}
	rs, _ := output[0].(map[string]any)
	if rs["type"] != "reasoning" || rs["id"] != rsID {
		t.Fatalf("snapshot reasoning = %#v, want id %s", rs, rsID)
	}
}

// The prelude's content_part.added part carries the empty annotations list.
func TestMessagesToResponsesContentPartAddedHasAnnotations(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(messagesStream("end_turn")), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	for _, e := range respEvents(t, dst.String()) {
		if e["type"] != "response.content_part.added" {
			continue
		}
		part, _ := e["part"].(map[string]any)
		if part["type"] != "output_text" || part["text"] != "" {
			t.Fatalf("part = %#v", part)
		}
		if ann, ok := part["annotations"].([]any); !ok || len(ann) != 0 {
			t.Fatalf("annotations = %#v, want []", part["annotations"])
		}
		return
	}
	t.Fatalf("no content_part.added:\n%s", dst.String())
}

// A tool item is announced under its call id (never a self-made fc_ id), at
// output_index from 1 (0 is the merged message/reasoning item), and its
// arguments deltas are re-chunked to the ~10-character pieces the reference
// streams.
func TestMessagesToResponsesToolItemIdentityAndChunking(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"exec","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\": \"ls -la /tmp\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	var toolAdded map[string]any
	for _, e := range evs {
		if e["type"] == "response.output_item.added" {
			if item, _ := e["item"].(map[string]any); item["type"] == "function_call" {
				toolAdded = e
			}
		}
	}
	if toolAdded == nil {
		t.Fatalf("tool item never announced:\n%s", dst.String())
	}
	if toolAdded["output_index"] != float64(1) {
		t.Fatalf("tool output_index = %#v, want 1", toolAdded["output_index"])
	}
	item, _ := toolAdded["item"].(map[string]any)
	if item["id"] != "toolu_1" || item["call_id"] != "toolu_1" || item["name"] != "exec" || item["status"] != "in_progress" {
		t.Fatalf("tool item = %#v", item)
	}
	var pieces []string
	for _, e := range evs {
		if e["type"] == "response.function_call_arguments.delta" {
			if e["item_id"] != "toolu_1" || e["output_index"] != float64(1) {
				t.Fatalf("delta = %#v", e)
			}
			pieces = append(pieces, e["delta"].(string))
		}
	}
	want := []string{`{"cmd": "l`, `s -la /tmp`, `"}`}
	if len(pieces) != len(want) {
		t.Fatalf("delta pieces = %q", pieces)
	}
	for i := range want {
		if pieces[i] != want[i] {
			t.Fatalf("delta pieces = %q, want %q", pieces, want)
		}
	}
	var done map[string]any
	for _, e := range evs {
		if e["type"] == "response.function_call_arguments.done" {
			done = e
		}
	}
	if done == nil || done["arguments"] != `{"cmd": "ls -la /tmp"}` {
		t.Fatalf("arguments done = %#v", done)
	}
}

// Done events wait for the end of the stream, tools before the message
// regardless of source order, with one exception the reference makes: an
// active reasoning stretch closes with its done trio at the FIRST following
// chunk (streaming_iterator.py:926), so the trio lands mid-stream, before
// later deltas. A turn that reasoned folds its reasoning into the message
// part's done event as a reasoning_text part.
func TestMessagesToResponsesDoneEventsDeferredToolBeforeMessage(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_2","name":"exec","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	at := func(typ string) int {
		for i, e := range evs {
			if e["type"] == typ {
				return i
			}
		}
		return -1
	}
	lastDelta := -1
	for i, e := range evs {
		if strings.HasSuffix(e["type"].(string), ".delta") && i > lastDelta {
			lastDelta = i
		}
	}
	for _, typ := range []string{
		"response.function_call_arguments.done",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	} {
		if at(typ) < 0 {
			t.Fatalf("missing %s:\n%s", typ, dst.String())
		}
	}
	// The END-OF-STREAM done events follow every delta. The reasoning trio is
	// deliberately exempt — the reference fires it at the first non-reasoning
	// chunk, mid-stream (asserted by ReasoningHasNoLifecycleEvents above).
	if firstEndDone := at("response.function_call_arguments.done"); firstEndDone < lastDelta {
		t.Fatalf("end-of-stream done events must follow every delta:\n%s", dst.String())
	}
	if at("response.function_call_arguments.done") > at("response.output_text.done") {
		t.Fatalf("tool done must precede message done:\n%s", dst.String())
	}
	for _, e := range evs {
		if e["type"] != "response.content_part.done" {
			continue
		}
		part, _ := e["part"].(map[string]any)
		if part["type"] != "reasoning_text" || part["reasoning"] != "let me " {
			t.Fatalf("content_part.done part = %#v, want reasoning_text", part)
		}
	}
	itemsDone := eventsOfTypes(t, evs, "response.output_item.done")
	if len(itemsDone) != 3 {
		t.Fatalf("output_item.done count=%d, want reasoning+tool+message:\n%s", len(itemsDone), dst.String())
	}
	if it, _ := itemsDone[0]["item"].(map[string]any); it["type"] != "reasoning" {
		t.Fatalf("first done item = %#v, want the reasoning trio's item", itemsDone[0]["item"])
	}
	if it, _ := itemsDone[1]["item"].(map[string]any); it["type"] != "function_call" {
		t.Fatalf("second done item = %#v, want the tool", itemsDone[1]["item"])
	}
	if it, _ := itemsDone[2]["item"].(map[string]any); it["type"] != "message" {
		t.Fatalf("third done item = %#v, want the message", itemsDone[2]["item"])
	}
}

// eventsOfTypes collects the decoded events of one type.
func eventsOfTypes(t *testing.T, evs []map[string]any, typ string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, e := range evs {
		if e["type"] == typ {
			out = append(out, e)
		}
	}
	return out
}

// Done items and the completed snapshot carry the fields the reference does:
// the message part its (empty) annotations, the call item id = call_id, and
// the reasoning payload only ever inside the completed snapshot.
func TestMessagesToResponsesDoneAndSnapshotItemFields(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"f","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	var msgDone, fcDone map[string]any
	for _, e := range eventsOfTypes(t, evs, "response.output_item.done") {
		item, _ := e["item"].(map[string]any)
		switch item["type"] {
		case "message":
			msgDone = item
		case "function_call":
			fcDone = item
		}
	}
	if msgDone == nil || fcDone == nil {
		t.Fatalf("missing done items: %#v / %#v", msgDone, fcDone)
	}
	part, _ := msgDone["content"].([]any)[0].(map[string]any)
	if ann, ok := part["annotations"].([]any); !ok || len(ann) != 0 {
		t.Fatalf("done message part annotations = %#v, want []", part["annotations"])
	}
	if fcDone["id"] != "toolu_9" || fcDone["call_id"] != "toolu_9" {
		t.Fatalf("done call item ids = %#v / %#v", fcDone["id"], fcDone["call_id"])
	}
	var completed map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			completed, _ = e["response"].(map[string]any)
		}
	}
	output, _ := completed["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("snapshot output = %#v", completed["output"])
	}
	fcSnap, _ := output[2].(map[string]any)
	if fcSnap["id"] != "toolu_9" || fcSnap["call_id"] != "toolu_9" {
		t.Fatalf("snapshot call ids = %#v / %#v", fcSnap["id"], fcSnap["call_id"])
	}
	msgSnap, _ := output[1].(map[string]any)
	if ann, ok := msgSnap["content"].([]any)[0].(map[string]any)["annotations"].([]any); !ok || len(ann) != 0 {
		t.Fatalf("snapshot message part annotations missing")
	}
	// The opaque reasoning payload only ever rides the completed snapshot.
	for i, chunk := range strings.Split(dst.String(), "\n\n") {
		if strings.Contains(chunk, "encrypted_content") && !strings.Contains(chunk, "response.completed") {
			t.Fatalf("encrypted_content leaked outside the completed snapshot (chunk %d):\n%s", i, chunk)
		}
	}
}

// Anthropic sends no input_json_delta for an empty tool input; the reference's
// parser synthesizes "{}" so the arguments on the wire stay parseable JSON.
func TestMessagesToResponsesEmptyToolArgumentsAreEmptyJSON(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_e","name":"noop","input":{}}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	var synthDelta int
	for _, e := range eventsOfTypes(t, evs, "response.function_call_arguments.delta") {
		if e["delta"] == "{}" {
			synthDelta++
		}
	}
	if synthDelta != 1 {
		t.Fatalf("synthesized {} delta count = %d, want 1 (before the done event):\n%s", synthDelta, dst.String())
	}
	var done map[string]any
	for _, e := range eventsOfTypes(t, evs, "response.function_call_arguments.done") {
		done = e
	}
	if done == nil {
		t.Fatalf("no arguments done event:\n%s", dst.String())
	}
	if done["arguments"] != "{}" {
		t.Fatalf("empty-input arguments = %#v, want {}", done["arguments"])
	}
	var completed map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			completed, _ = e["response"].(map[string]any)
		}
	}
	for _, it := range completed["output"].([]any) {
		im, _ := it.(map[string]any)
		if im["type"] == "function_call" {
			if im["arguments"] != "{}" {
				t.Fatalf("snapshot empty-input arguments = %#v, want {}", im["arguments"])
			}
			if !json.Valid([]byte(im["arguments"].(string))) {
				t.Fatalf("snapshot arguments not JSON: %#v", im["arguments"])
			}
		}
	}
}

// The prelude shell carries the request-independent fields the reference's
// opening frames have, and the prelude fires on the first frame even when the
// source never sent message_start. created_at is the one field left out: the
// reference fills it from the wall clock, which this library has none of.
func TestMessagesToResponsesPreludeShell(t *testing.T) {
	src := strings.Join([]string{
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	if len(evs) == 0 || evs[0]["type"] != "response.created" {
		t.Fatalf("stream did not open with response.created:\n%s", dst.String())
	}
	shell, _ := evs[0]["response"].(map[string]any)
	if shell["object"] != "response" || shell["status"] != "in_progress" {
		t.Fatalf("shell = %#v", shell)
	}
	if out, ok := shell["output"].([]any); !ok || len(out) != 0 {
		t.Fatalf("shell output = %#v, want []", shell["output"])
	}
	if shell["parallel_tool_calls"] != true || shell["store"] != true {
		t.Fatalf("shell = %#v", shell)
	}
	if shell["tool_choice"] != "auto" {
		t.Fatalf("shell tool_choice = %#v", shell["tool_choice"])
	}
	if tools, ok := shell["tools"].([]any); !ok || len(tools) != 0 {
		t.Fatalf("shell tools = %#v, want []", shell["tools"])
	}
	if shell["top_p"] != float64(1) {
		t.Fatalf("shell top_p = %#v", shell["top_p"])
	}
	reasoning, ok := shell["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("shell reasoning = %#v", shell["reasoning"])
	}
	if v, has := reasoning["effort"]; !has || v != nil {
		t.Fatalf("shell reasoning.effort = %#v (present=%v), want null", v, has)
	}
	if v, has := reasoning["summary"]; !has || v != nil {
		t.Fatalf("shell reasoning.summary = %#v (present=%v), want null", v, has)
	}
	if _, has := shell["created_at"]; has {
		t.Fatalf("created_at present; the library has no clock to fill it from")
	}
	// The stream still opened without message_start and closed cleanly.
	if !strings.Contains(dst.String(), "response.completed") {
		t.Fatalf("no terminal:\n%s", dst.String())
	}

	// A legal message_start-first stream carries the id and model it names on
	// the opening shells — the reference read-aheads one chunk before opening.
	var dst2 bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst2, strings.NewReader(messagesStream("end_turn")), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	shells2 := respEvents(t, dst2.String())[:2]
	for i, want := range []string{"response.created", "response.in_progress"} {
		if shells2[i]["type"] != want {
			t.Fatalf("event %d = %v, want %s:\n%s", i, shells2[i]["type"], want, dst2.String())
		}
		sh, _ := shells2[i]["response"].(map[string]any)
		if sh["id"] != "msg_1" || sh["model"] != "m" {
			t.Fatalf("%s shell id/model = %#v / %#v, want the message_start values", want, sh["id"], sh["model"])
		}
	}
}

// A Responses source that skips response.created still folds into a legal
// Messages stream: message_start opens on the first frame whatever it is.
func TestResponsesToMessagesOpensWithoutResponseCreated(t *testing.T) {
	src := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"m0","delta":"Hi"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	if len(evs) == 0 || evs[0]["type"] != "message_start" {
		t.Fatalf("fold did not open with message_start:\n%s", dst.String())
	}
	msg, _ := evs[0]["message"].(map[string]any)
	if v, has := msg["stop_reason"]; !has || v != nil {
		t.Fatalf("message_start stop_reason = %#v (present=%v), want null", v, has)
	}
	if usage, ok := msg["usage"].(map[string]any); !ok || len(usage) != 4 {
		t.Fatalf("message_start usage = %#v", msg["usage"])
	}
	if !strings.Contains(dst.String(), `"text":"Hi"`) || !strings.Contains(dst.String(), "message_stop") {
		t.Fatalf("content or terminal lost:\n%s", dst.String())
	}
}

// The model on message_start is the one the host routed to (the constructor
// model), not the model the upstream response names.
func TestResponsesToMessagesMessageStartModelFromHost(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"upstream-model"}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "ctor-model"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(dst.String(), `"model":"ctor-model"`) {
		t.Fatalf("message_start model = upstream model instead of the host's:\n%s", dst.String())
	}
}

// An empty reasoning delta with no thinking block open produces nothing at
// all — the reference opens no block and emits no event; once a block is open,
// an empty delta still relays.
func TestResponsesToMessagesEmptyReasoningDeltaOpensNothing(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":""}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"x"}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":""}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := respEvents(t, dst.String())
	thinkingStarts := 0
	for _, e := range eventsOfTypes(t, evs, "content_block_start") {
		if b, _ := e["content_block"].(map[string]any); b["type"] == "thinking" {
			thinkingStarts++
		}
	}
	if thinkingStarts != 1 {
		t.Fatalf("thinking block starts = %d, want 1 (the leading empty delta opens nothing):\n%s", thinkingStarts, dst.String())
	}
	var deltas []string
	for _, e := range eventsOfTypes(t, evs, "content_block_delta") {
		if d, _ := e["delta"].(map[string]any); d["type"] == "thinking_delta" {
			deltas = append(deltas, d["thinking"].(string))
		}
	}
	if len(deltas) != 2 || deltas[0] != "x" || deltas[1] != "" {
		t.Fatalf("thinking deltas = %q, want [x ] (leading empty dropped, trailing empty relayed)", deltas)
	}
}

// The relay [DONE] sentinel is this library's source-close for a Responses
// stream, deliberately kept against the reference (whose iterator emits
// nothing further on it): the settlement facts count [DONE] as a terminal, so
// the fold must close the client's turn too — otherwise the host books a
// delivered turn (StreamResult.Terminal=true) while the Messages client's
// stream never ends. This pins the coherence, not litellm parity.
func TestResponsesToMessagesRelayDoneClosesAndMatchesSettlement(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"m0","delta":"Hi"}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	res, err := PipeStreamWith(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v, want the relay close counted as terminal", res)
	}
	if !strings.Contains(dst.String(), "message_stop") || !strings.Contains(dst.String(), `"stop_reason":"end_turn"`) {
		t.Fatalf("relay [DONE] did not close the client's turn:\n%s", dst.String())
	}
}

// Each message output item is its own Anthropic text block: the item's added
// opens one, its done closes it, and a second item's text never lands in the
// first one's block — even on a stream cut before the terminal.
func TestResponsesToMessagesEachMessageItemOwnsItsBlock(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m0","role":"assistant","content":[]}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"m0","output_index":0,"delta":"one"}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m0","role":"assistant","content":[{"type":"output_text","text":"one"}]}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m1","role":"assistant","content":[]}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"m1","output_index":1,"delta":"two"}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"two"}]}}`,
		``,
	}, "\n")

	// Truncated first: no terminal frame at all, and both blocks still closed.
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	assertPerItemBlocks(t, dst.String(), true)

	// With the terminal, the same block events precede the closing frames.
	var full bytes.Buffer
	if _, err := PipeStream(Responses, Messages, &full, strings.NewReader(src+"\n"+`event: response.completed`+"\n"+`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`+"\n\n"), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	assertPerItemBlocks(t, full.String(), false)
	for _, want := range []string{"message_stop"} {
		if !strings.Contains(full.String(), want) {
			t.Fatalf("missing %s:\n%s", want, full.String())
		}
	}
}

// assertPerItemBlocks checks the start0/delta0/stop0/start1/delta1/stop1 shape
// two message items must fold into. truncated marks a stream with no terminal,
// where no message_delta/message_stop may appear.
func assertPerItemBlocks(t *testing.T, raw string, truncated bool) {
	t.Helper()
	var kinds []string
	for _, e := range respEvents(t, raw) {
		switch e["type"] {
		case "content_block_start":
			if b, _ := e["content_block"].(map[string]any); b["type"] == "text" {
				kinds = append(kinds, fmtIndex("start", e["index"]))
			}
		case "content_block_delta":
			if d, _ := e["delta"].(map[string]any); d["type"] == "text_delta" {
				kinds = append(kinds, fmtIndex("delta"+d["text"].(string), e["index"]))
			}
		case "content_block_stop":
			kinds = append(kinds, fmtIndex("stop", e["index"]))
		case "message_delta", "message_stop":
			if truncated {
				t.Fatalf("terminal frame on a truncated stream:\n%s", raw)
			}
		}
	}
	got := strings.Join(kinds, ",")
	want := "start0,deltaone0,stop0,start1,deltatwo1,stop1"
	if got != want {
		t.Fatalf("block sequence = %s, want %s:\n%s", got, want, raw)
	}
}

func fmtIndex(kind string, idx any) string {
	return fmt.Sprintf("%s%d", kind, int(idx.(float64)))
}
