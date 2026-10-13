package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// This file pins the chat<->messages folds a LiteLLM v1.100.1 audit found
// diverging from the reference. Each case holds a fold to the shape the
// reference adapter emits for the same input — mostly content that used to be
// silently dropped (a thinking block's signature, a cache-capability probe)
// or mapped onto a target value with different semantics.

// chatMsgParityEvents splits an SSE byte stream into its parsed data payloads,
// comments and the [DONE] sentinel aside.
func chatMsgParityEvents(t *testing.T, out string) []map[string]any {
	t.Helper()
	var evs []map[string]any
	for _, block := range strings.Split(out, "\n\n") {
		var data string
		for _, line := range strings.Split(block, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("bad event data %q: %v", data, err)
		}
		evs = append(evs, m)
	}
	return evs
}

// A chat stream may carry its reasoning as delta.thinking_blocks — the shape
// this package's own Messages→chat fold spells its thinking in, and the shape
// litellm's chat dialects stream. The fold must open a thinking block carrying
// the chunk's text and signature and emit the signature as a signature_delta,
// not drop the block wholesale (litellm: adapters/transformation.py:1533-1599).
func TestChatToMessagesStreamFoldsThinkingBlocks(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"thinking_blocks":[{"type":"thinking","thinking":"why","signature":"SIG"}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	evs := chatMsgParityEvents(t, dst.String())
	started := false
	signed := false
	for _, e := range evs {
		if e["type"] == "content_block_start" {
			if cb, ok := e["content_block"].(map[string]any); ok && cb["type"] == "thinking" {
				started = true
				if cb["thinking"] != "why" || cb["signature"] != "SIG" {
					t.Fatalf("thinking content_block_start = %#v, want the text and signature", cb)
				}
			}
		}
		if e["type"] == "content_block_delta" {
			if d, ok := e["delta"].(map[string]any); ok && d["type"] == "signature_delta" && d["signature"] == "SIG" {
				signed = true
			}
		}
	}
	if !started || !signed {
		t.Fatalf("thinking block start=%v signature_delta=%v, want both:\n%s", started, signed, dst.String())
	}
	if !strings.Contains(dst.String(), `"text_delta"`) {
		t.Fatalf("text after the thinking block lost:\n%s", dst.String())
	}
	if n := strings.Count(dst.String(), `"type":"message_stop"`); n != 1 {
		t.Fatalf("message_stop events = %d, want 1\n%s", n, dst.String())
	}
}

// Replaying this package's own Messages→chat output back into Messages must
// keep the thinking signature: the chat terminal carries the signed block, and
// a fold that only reads reasoning_content loses it.
func TestChatToMessagesRoundTripKeepsThinkingSignature(t *testing.T) {
	msgSrc := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var chat bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &chat, strings.NewReader(msgSrc), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var back bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &back, strings.NewReader(chat.String()), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := back.String()
	if !strings.Contains(out, `"signature":"SIG"`) {
		t.Fatalf("signature lost on the replay:\n%s", out)
	}
	if !strings.Contains(out, `"thinking":"why"`) {
		t.Fatalf("thinking text lost on the replay:\n%s", out)
	}
}

// A chat content_filter cut has no Messages branch of its own; the reference
// folds it onto end_turn, not refusal — refusal says the model declined when
// the upstream said the turn was filtered (litellm:
// adapters/transformation.py:1340-1347).
func TestChatContentFilterFoldsToEndTurn(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"content_filter","message":{"role":"assistant","content":"x"}}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", m["stop_reason"])
	}

	// The streaming terminal folds through the same map.
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"x"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	for _, e := range chatMsgParityEvents(t, dst.String()) {
		if e["type"] == "message_delta" {
			if d, ok := e["delta"].(map[string]any); ok && d["stop_reason"] != "end_turn" {
				t.Fatalf("stream stop_reason = %v, want end_turn\n%s", d["stop_reason"], dst.String())
			}
		}
	}
}

// The message_start usage must state the cache halves as 0: clients like
// Claude Code probe those fields to decide whether the target supports prompt
// caching at all (litellm: adapters/streaming_iterator.py:474-493).
func TestChatToMessagesStreamStartUsageStatesCacheZeros(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	for _, e := range chatMsgParityEvents(t, dst.String()) {
		if e["type"] != "message_start" {
			continue
		}
		msg, ok := e["message"].(map[string]any)
		if !ok {
			t.Fatalf("message_start = %#v", e)
		}
		usage, ok := msg["usage"].(map[string]any)
		if !ok {
			t.Fatalf("message_start usage = %#v", msg)
		}
		for _, field := range []string{"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"} {
			if v, present := usage[field]; !present || v != float64(0) {
				t.Fatalf("usage[%s] = %#v, want a stated 0: %#v", field, v, usage)
			}
		}
		return
	}
	t.Fatalf("no message_start:\n%s", dst.String())
}

// A thinking block's fold to chat must not wait for the terminal: litellm
// emits the incremental text as reasoning_content plus an incremental
// thinking_block on every thinking_delta, and the whole signed block the
// moment the signature arrives, so a stream cut before message_delta still
// delivered the replayable block (litellm: anthropic/chat/handler.py:765-793).
func TestMessagesToChatStreamEmitsThinkingBlocksWhenTheyArrive(t *testing.T) {
	thinking := func(terminal string) string {
		return strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" now"}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
			``,
			`event: content_block_stop`,
			`data: {"type":"content_block_stop","index":0}`,
			``,
			terminal,
		}, "\n")
	}
	terminal := strings.Join([]string{
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(thinking(terminal)), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	chunks := chatMsgParityEvents(t, dst.String())
	signedAt, finishAt := -1, -1
	for i, c := range chunks {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue // the trailing usage-only chunk
		}
		delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
		if delta["reasoning_content"] == "why" {
			blocks, ok := delta["thinking_blocks"].([]any)
			if !ok || len(blocks) != 1 {
				t.Fatalf("incremental thinking chunk carries no thinking_blocks: %#v", delta)
			}
			b, _ := blocks[0].(map[string]any)
			if b["type"] != "thinking" || b["thinking"] != "why" {
				t.Fatalf("incremental thinking block = %#v", b)
			}
		}
		if blocks, ok := delta["thinking_blocks"].([]any); ok && len(blocks) == 1 {
			if b, _ := blocks[0].(map[string]any); b["signature"] == "SIG" {
				if b["thinking"] != "why now" {
					t.Fatalf("signed thinking block = %#v, want the accumulated text", b)
				}
				signedAt = i
			}
		}
		if fr := c["choices"].([]any)[0].(map[string]any)["finish_reason"]; fr != nil {
			finishAt = i
			if _, has := delta["thinking_blocks"]; has {
				t.Fatalf("terminal chunk re-sends thinking_blocks: %#v", delta)
			}
		}
	}
	if signedAt < 0 {
		t.Fatalf("no signed thinking block before the close:\n%s", dst.String())
	}
	if finishAt < 0 || signedAt > finishAt {
		t.Fatalf("signed block at %d must precede the finish at %d:\n%s", signedAt, finishAt, dst.String())
	}

	// The same stream cut after the signature, with no message_delta at all,
	// must already have delivered the signed block.
	var cut bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &cut, strings.NewReader(thinking("")), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cut.String(), `"signature":"SIG"`) {
		t.Fatalf("signed thinking block lost on a truncated stream:\n%s", cut.String())
	}
}

// A redacted thinking block's opaque bytes go out the moment the block opens —
// litellm emits them on content_block_start, not held for a terminal that a
// cut stream never reaches (litellm: anthropic/chat/handler.py:946-953).
func TestMessagesToChatStreamEmitsRedactedThinkingAtBlockStart(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"OPAQUE"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"type":"redacted_thinking"`) || !strings.Contains(out, `"data":"OPAQUE"`) {
		t.Fatalf("redacted thinking lost:\n%s", out)
	}
}

// A tool_use name in a Messages response is the name the caller declared; the
// chat wire's 64-char cap is a request-history concern, and truncating on the
// response path hands back a name no registered tool matches. The reference
// passes the original through (litellm: anthropic/chat/transformation.py:326).
func TestMessagesResponseToChatKeepsLongToolNames(t *testing.T) {
	name := strings.Repeat("n", 70)
	nameJSON, err := json.Marshal(name)
	if err != nil {
		t.Fatal(err)
	}
	body := `{"id":"m1","model":"c","stop_reason":"tool_use","content":[{"type":"tool_use","id":"t1","name":` + string(nameJSON) + `,"input":{"a":1}}],"usage":{"input_tokens":1,"output_tokens":1}}`
	out, err := ConvertResponse(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	call := m["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	fn := call["function"].(map[string]any)
	if fn["name"] != name {
		t.Fatalf("tool name = %q (%d chars), want the original %d-char name", fn["name"], len(fn["name"].(string)), len(name))
	}
}

// A present-but-empty content still folds to an empty text block: the
// reference emits one whenever content is not null, and a client that
// distinguishes "" from absent reads the difference (litellm:
// adapters/transformation.py:1300-1303).
func TestChatToMessagesEmptyContentFoldsEmptyTextBlock(t *testing.T) {
	empty := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":""}}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	content := m["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content = %#v, want the empty text block", content)
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "" {
		t.Fatalf("block = %#v, want {type:text, text:\"\"}", block)
	}

	absent := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null}}]}`
	out, err = ConvertResponse(Chat, Messages, []byte(absent))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if content := m["content"].([]any); len(content) != 0 {
		t.Fatalf("null content = %#v, want no block", content)
	}
}

// The Messages block shapes are fixed: a thinking block always states its
// signature (null on the reasoning_content fallback, "" when the chat block
// came unsigned), a redacted block keeps only its data, extra chat-side keys
// do not ride along, and a tool_use states provider_specific_fields even when
// nothing rides it (litellm: adapters/transformation.py:1259-1338).
func TestChatToMessagesFoldsFixedBlockShapes(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"reasoning_content":"why"}}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	block := m["content"].([]any)[0].(map[string]any)
	sig, present := block["signature"]
	if !present || sig != nil {
		t.Fatalf("reasoning_content fallback block = %#v, want signature:null stated", block)
	}

	body = `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"thinking_blocks":[{"type":"thinking","thinking":"why","cache_control":{"type":"ephemeral"}}]}}]}`
	out, err = ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	block = m["content"].([]any)[0].(map[string]any)
	if _, extra := block["cache_control"]; extra {
		t.Fatalf("extra keys rode along: %#v", block)
	}
	if block["signature"] != "" {
		t.Fatalf("unsigned thinking block = %#v, want signature:\"\" stated", block)
	}
	if len(block) != 3 {
		t.Fatalf("thinking block = %#v, want exactly type/thinking/signature", block)
	}

	body = `{"id":"c1","model":"m","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}`
	out, err = ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["content"].([]any)[0].(map[string]any)
	if _, present := tool["provider_specific_fields"]; !present {
		t.Fatalf("tool_use = %#v, want provider_specific_fields stated", tool)
	}
}

// The streaming frames carry the reference's shapes: the closing
// message_delta names only the stop_reason, and a thinking block opened from
// incremental reasoning states its empty signature (litellm:
// adapters/transformation.py:1553, 1613; streaming_iterator.py:548).
func TestChatToMessagesStreamFrameShapesMatchReference(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"reasoning_content":"why"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	for _, e := range chatMsgParityEvents(t, dst.String()) {
		if e["type"] == "content_block_start" {
			if cb, ok := e["content_block"].(map[string]any); ok && cb["type"] == "thinking" {
				if cb["signature"] != "" {
					t.Fatalf("thinking start = %#v, want signature:\"\" stated", cb)
				}
			}
		}
		if e["type"] == "message_delta" {
			d, ok := e["delta"].(map[string]any)
			if !ok {
				t.Fatalf("message_delta = %#v", e)
			}
			if _, has := d["stop_sequence"]; has {
				t.Fatalf("message_delta.delta = %#v, want no stop_sequence key", d)
			}
			if d["stop_reason"] != "end_turn" {
				t.Fatalf("stop_reason = %v", d["stop_reason"])
			}
		}
	}
}

// A chat tool call id rides the Messages wire only as [a-zA-Z0-9_-]+, so a
// cross-provider id like "functions.Bash:0" is normalized, an empty id gets a
// placeholder, and a thought-signature suffix moves off the id — in the
// streaming fold onto provider_specific_fields (litellm:
// anthropic/common_utils.py:1121-1132).
func TestChatToMessagesNormalizesToolUseIDs(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"functions.Bash:0","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	out, err := ConvertResponse(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if got := m["content"].([]any)[0].(map[string]any)["id"]; got != "functions_Bash_0" {
		t.Fatalf("id = %v, want functions_Bash_0", got)
	}

	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"toolu_1__thought__SIGBYTES","type":"function","function":{"name":"f","arguments":""}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	for _, e := range chatMsgParityEvents(t, dst.String()) {
		if e["type"] != "content_block_start" {
			continue
		}
		cb, ok := e["content_block"].(map[string]any)
		if !ok || cb["type"] != "tool_use" {
			continue
		}
		if cb["id"] != "toolu_1" {
			t.Fatalf("stream id = %v, want the suffix stripped", cb["id"])
		}
		psf, ok := cb["provider_specific_fields"].(map[string]any)
		if !ok || psf["signature"] != "SIGBYTES" {
			t.Fatalf("provider_specific_fields = %#v, want the thought signature", cb)
		}
		return
	}
	t.Fatalf("no tool_use block:\n%s", dst.String())
}

// The count of server-tool (web search) requests survives the fold as its own
// usage object, the explicit spelling outranking the token-details one
// (litellm: adapters/transformation.py:1395-1431).
func TestChatToMessagesUsageCarriesServerToolUse(t *testing.T) {
	details := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"web_search_requests":3}}}`
	out, err := ConvertResponse(Chat, Messages, []byte(details))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	stu, ok := m["usage"].(map[string]any)["server_tool_use"].(map[string]any)
	if !ok || stu["web_search_requests"] != float64(3) {
		t.Fatalf("usage = %#v, want server_tool_use.web_search_requests=3", m["usage"])
	}

	explicit := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"server_tool_use":{"web_search_requests":7},"prompt_tokens_details":{"web_search_requests":3}}}`
	out, err = ConvertResponse(Chat, Messages, []byte(explicit))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	stu = m["usage"].(map[string]any)["server_tool_use"].(map[string]any)
	if stu["web_search_requests"] != float64(7) {
		t.Fatalf("usage = %#v, want the explicit count to win", m["usage"])
	}

	none := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`
	out, err = ConvertResponse(Chat, Messages, []byte(none))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if _, has := m["usage"].(map[string]any)["server_tool_use"]; has {
		t.Fatalf("usage = %#v, want no server_tool_use when none counted", m["usage"])
	}
}

// The chat usage states its detail objects unconditionally, zeros included:
// prompt_tokens_details (cached/cache-creation/cache-write/text), the
// top-level cache halves, and completion_tokens_details with the thinking
// split when the source reports one (litellm:
// anthropic/chat/transformation.py:2259-2360).
func TestMessagesResponseToChatUsageStatesDetails(t *testing.T) {
	body := `{"id":"m1","model":"c","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":30,"output_tokens":5,"cache_read_input_tokens":10,"cache_creation_input_tokens":60}}`
	out, err := ConvertResponse(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(100) || usage["completion_tokens"] != float64(5) || usage["total_tokens"] != float64(105) {
		t.Fatalf("token math changed: %#v", usage)
	}
	if usage["cache_creation_input_tokens"] != float64(60) || usage["cache_read_input_tokens"] != float64(10) {
		t.Fatalf("top-level cache halves = %#v", usage)
	}
	ptd := usage["prompt_tokens_details"].(map[string]any)
	if ptd["cached_tokens"] != float64(10) || ptd["cache_creation_tokens"] != float64(60) || ptd["cache_write_tokens"] != float64(60) || ptd["text_tokens"] != float64(30) {
		t.Fatalf("prompt_tokens_details = %#v", ptd)
	}
	ctd := usage["completion_tokens_details"].(map[string]any)
	if ctd["reasoning_tokens"] != float64(0) || ctd["text_tokens"] != float64(5) {
		t.Fatalf("completion_tokens_details = %#v", ctd)
	}

	// No cache: the halves are stated as zeros, not dropped.
	body = `{"id":"m1","model":"c","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":30,"output_tokens":5}}`
	out, err = ConvertResponse(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage = m["usage"].(map[string]any)
	if usage["cache_creation_input_tokens"] != float64(0) || usage["cache_read_input_tokens"] != float64(0) {
		t.Fatalf("cache halves = %#v, want zeros stated", usage)
	}
	if ptd := usage["prompt_tokens_details"].(map[string]any); ptd["cached_tokens"] != float64(0) || ptd["text_tokens"] != float64(30) {
		t.Fatalf("prompt_tokens_details = %#v, want zeros stated", ptd)
	}

	// A reported thinking share splits completion_tokens_details, capped at
	// the reported total.
	body = `{"id":"m1","model":"c","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":3,"output_tokens":5,"output_tokens_details":{"thinking_tokens":2}}}`
	out, err = ConvertResponse(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	ctd = m["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	if ctd["reasoning_tokens"] != float64(2) || ctd["text_tokens"] != float64(3) {
		t.Fatalf("completion_tokens_details = %#v, want the thinking split", ctd)
	}
}

// A streamed content part list is read part by part, the non-streaming fold's
// rule: rendering it with a generic formatter writes Go syntax the model
// cannot read (the reference has no handling here at all — it errors out).
func TestChatToMessagesStreamReadsContentParts(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":[{"type":"text","text":"Hello"}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"text":"Hello"`) {
		t.Fatalf("part-list text lost:\n%s", out)
	}
	if strings.Contains(out, "map[") {
		t.Fatalf("part list rendered with Go syntax:\n%s", out)
	}
}

// A chat stream that ends without ever naming a finish is closed, not
// answered: the client gets a bare message_stop — the one frame the reference
// emits there — with no synthesized stop_reason or zero-usage message_delta
// dressing the cut as a clean turn. A held finish_reason whose usage never
// arrived still closes with the real reason (litellm:
// adapters/streaming_iterator.py:663-690).
func TestChatToMessagesEOFSynthesizesOnlyAMessageStop(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"text_delta"`) || !strings.Contains(out, `"text":"hi"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
	if !strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("the turn no longer closes for the client:\n%s", out)
	}
	if strings.Contains(out, `"type":"message_delta"`) {
		t.Fatalf("a cut stream was dressed as a clean turn:\n%s", out)
	}
	if strings.Contains(out, `"type":"content_block_stop"`) {
		t.Fatalf("synthesized content_block_stop the reference does not emit:\n%s", out)
	}

	// A held finish_reason whose usage never came still closes with the real
	// reason, not a bare stop.
	src = strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
	}, "\n")
	dst.Reset()
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out = dst.String()
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("held stop_reason lost:\n%s", out)
	}
	if n := strings.Count(out, `"type":"message_stop"`); n != 1 {
		t.Fatalf("message_stop events = %d, want 1\n%s", n, out)
	}
}

// A server tool call is a tool call on the chat wire: the same id/name open
// chunk, its own index, and no collision with the tools after it. A
// server-tool result's json delta is not tool arguments — appending it to the
// previous call corrupts them (litellm: anthropic/chat/handler.py:746-762,
// 914-917).
func TestMessagesToChatFoldsServerToolUseAndGuardsResultDeltas(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtool_01","name":"web_search","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtool_01","content":[]}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"results\":[]}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`,
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
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var openServer, serverArgs, openTool *map[string]any
	for _, c := range chatMsgParityEvents(t, dst.String()) {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
		calls, ok := delta["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			continue
		}
		call := calls[0].(map[string]any)
		fn, _ := call["function"].(map[string]any)
		args, _ := fn["arguments"].(string)
		if fn["name"] == "web_search" {
			openServer = &call
		}
		if strings.Contains(args, "query") {
			serverArgs = &call
		}
		if fn["name"] == "get_weather" {
			openTool = &call
		}
		if strings.Contains(args, "results") {
			t.Fatalf("a server-tool result's json became tool arguments: %#v", call)
		}
	}
	if openServer == nil {
		t.Fatalf("server_tool_use open chunk lost:\n%s", dst.String())
	}
	if (*openServer)["id"] != "srvtool_01" || (*openServer)["index"] != float64(0) {
		t.Fatalf("server tool open chunk = %#v, want id srvtool_01 at index 0", *openServer)
	}
	if serverArgs == nil || (*serverArgs)["index"] != float64(0) {
		t.Fatalf("server tool arguments delta lost or misindexed:\n%s", dst.String())
	}
	if openTool == nil {
		t.Fatalf("tool after a server tool lost:\n%s", dst.String())
	}
	if (*openTool)["index"] != float64(1) {
		t.Fatalf("tool index = %v, want 1 — the server tool must hold index 0", (*openTool)["index"])
	}
}

// Each thinking block folds on its own: the accumulator resets when a block
// opens, so two signed blocks yield two signed thinking_blocks chunks — never
// one merged block that loses the first signature — and a redacted block's
// data survives a thinking block after it (litellm:
// anthropic/chat/handler.py:778-793, 906).
func TestMessagesToChatFoldsEachThinkingBlockSeparately(t *testing.T) {
	run := func(blocks string) []map[string]any {
		src := strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
			``,
			blocks,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		}, "\n")
		var dst bytes.Buffer
		if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
			t.Fatal(err)
		}
		var signed []map[string]any
		for _, c := range chatMsgParityEvents(t, dst.String()) {
			choices, _ := c["choices"].([]any)
			if len(choices) == 0 {
				continue
			}
			delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
			if blocks, ok := delta["thinking_blocks"].([]any); ok && len(blocks) == 1 {
				if b, _ := blocks[0].(map[string]any); b["signature"] != nil && b["signature"] != "" {
					signed = append(signed, b)
				}
				if b, _ := blocks[0].(map[string]any); b["type"] == "redacted_thinking" {
					signed = append(signed, b)
				}
			}
		}
		return signed
	}

	two := run(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"A"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sigA"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"B"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sigB"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
	}, "\n"))
	if len(two) != 2 {
		t.Fatalf("signed blocks = %d, want one per thinking block:\n%v", len(two), two)
	}
	if two[0]["thinking"] != "A" || two[0]["signature"] != "sigA" {
		t.Fatalf("first block = %#v, want {A, sigA}", two[0])
	}
	if two[1]["thinking"] != "B" || two[1]["signature"] != "sigB" {
		t.Fatalf("second block = %#v, want {B, sigB} — the first block's text must not leak in", two[1])
	}

	mixed := run(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"OPAQUE"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"B"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sigB"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
	}, "\n"))
	if len(mixed) != 2 {
		t.Fatalf("mixed blocks = %#v, want the redacted block and the signed thinking block", mixed)
	}
	if mixed[0]["type"] != "redacted_thinking" || mixed[0]["data"] != "OPAQUE" {
		t.Fatalf("redacted block = %#v, want its data intact", mixed[0])
	}
	if mixed[1]["thinking"] != "B" || mixed[1]["signature"] != "sigB" {
		t.Fatalf("thinking block after redacted = %#v", mixed[1])
	}
}

// A tool-arguments continuation entry names its kind on every chunk, the
// spelling the reference emits (litellm: anthropic/chat/handler.py:751).
func TestMessagesToChatToolArgsContinuationNamesItsType(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	sawContinuation := false
	for _, c := range chatMsgParityEvents(t, dst.String()) {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
		calls, ok := delta["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			continue
		}
		call := calls[0].(map[string]any)
		fn, _ := call["function"].(map[string]any)
		if _, hasName := fn["name"]; hasName {
			continue // the open chunk; the continuation is what must name its type
		}
		sawContinuation = true
		if call["type"] != "function" {
			t.Fatalf("continuation entry = %#v, want type:function stated", call)
		}
	}
	if !sawContinuation {
		t.Fatalf("no continuation chunk:\n%s", dst.String())
	}
}

// The stream of chat chunks follows the reference's shapes: the role rides
// the first chunk that carries content (no chunk of its own), the terminal
// chunk is empty with no usage on it, and usage rides its own trailing chunk
// with empty choices — the include_usage spelling (litellm:
// streaming_handler.py:987-999, 1137-1178).
func TestMessagesToChatStreamChunkShapesMatchReference(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":2,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	chunks := chatMsgParityEvents(t, dst.String())
	if len(chunks) == 0 {
		t.Fatalf("no chunks:\n%s", dst.String())
	}
	first := chunks[0]["choices"].([]any)[0].(map[string]any)
	firstDelta := first["delta"].(map[string]any)
	if firstDelta["role"] != "assistant" || firstDelta["content"] != "Hi" {
		t.Fatalf("first chunk = %#v, want the role riding the first content", firstDelta)
	}
	for i, c := range chunks {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue // the trailing usage-only chunk
		}
		choice := choices[0].(map[string]any)
		if choice["finish_reason"] == nil {
			continue
		}
		if i == 0 {
			t.Fatalf("finish chunk is also the first chunk: %#v", c)
		}
		if _, has := c["usage"]; has {
			t.Fatalf("terminal chunk carries usage: %#v", c)
		}
		if d := choice["delta"].(map[string]any); len(d) != 0 {
			t.Fatalf("terminal chunk delta = %#v, want it empty", d)
		}
		// The usage rides the chunk after the close: one empty choice, the
		// include_usage tail shape.
		if i+1 >= len(chunks) {
			t.Fatalf("no usage chunk after the close:\n%s", dst.String())
		}
		usageChunk := chunks[i+1]
		tailChoices, _ := usageChunk["choices"].([]any)
		if len(tailChoices) != 1 {
			t.Fatalf("usage chunk choices = %#v, want one empty choice", usageChunk["choices"])
		}
		tail := tailChoices[0].(map[string]any)
		if d := tail["delta"].(map[string]any); len(d) != 0 {
			t.Fatalf("usage chunk delta = %#v, want it empty", d)
		}
		if _, has := tail["finish_reason"]; has {
			t.Fatalf("usage chunk choice carries finish_reason: %#v", tail)
		}
		u, has := usageChunk["usage"].(map[string]any)
		if !has {
			t.Fatalf("usage chunk has no usage: %#v", usageChunk)
		}
		if u["prompt_tokens"] != float64(2) || u["completion_tokens"] != float64(4) {
			t.Fatalf("usage chunk = %#v", u)
		}
		return
	}
	t.Fatalf("no finish chunk:\n%s", dst.String())
}

// Thinking blocks fold per block: each signature_delta replays the block it
// closed — not the whole turn's thinking — so two signed blocks stay two
// blocks with their own signatures, and a redacted block between them keeps
// its opaque bytes (litellm: anthropic/chat/handler.py:765-793, 906).
func TestMessagesToChatThinkingBlocksFoldPerBlock(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"A"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sigA"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"REDACTED"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"B"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sigB"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var sawA, sawRedacted, sawB bool
	for _, c := range chatMsgParityEvents(t, dst.String()) {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
		blocks, ok := delta["thinking_blocks"].([]any)
		if !ok || len(blocks) != 1 {
			continue
		}
		b, _ := blocks[0].(map[string]any)
		switch b["type"] {
		case "thinking":
			sig, _ := b["signature"].(string)
			if sig == "sigA" {
				if b["thinking"] != "A" {
					t.Fatalf("block A = %#v, want its own text only", b)
				}
				sawA = true
			}
			if sig == "sigB" {
				if b["thinking"] != "B" {
					t.Fatalf("block B = %#v, want its own text only", b)
				}
				sawB = true
			}
		case "redacted_thinking":
			if b["data"] != "REDACTED" {
				t.Fatalf("redacted block = %#v, want its own data", b)
			}
			sawRedacted = true
		}
	}
	if !sawA || !sawB {
		t.Fatalf("signed blocks A=%v B=%v, want each block replayed with its own signature:\n%s", sawA, sawB, dst.String())
	}
	if !sawRedacted {
		t.Fatalf("redacted block lost or overwritten:\n%s", dst.String())
	}
	if strings.Contains(dst.String(), `"thinking":"AB"`) {
		t.Fatalf("blocks merged across content_block_start:\n%s", dst.String())
	}
}

// The include_usage tail chunk carries one empty choice, not zero choices —
// and the reverse fold still reads its usage (litellm:
// streaming_handler.py:2069-2079).
func TestChatToMessagesFoldsTheOneChoiceUsageTail(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":7,"completion_tokens":3}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"input_tokens":7`) || !strings.Contains(out, `"output_tokens":3`) {
		t.Fatalf("usage tail with one empty choice not folded:\n%s", out)
	}
	if n := strings.Count(out, `"type":"message_stop"`); n != 1 {
		t.Fatalf("message_stop events = %d, want 1\n%s", n, out)
	}
}

// A tool argument continuation names its kind on every chunk: the entry
// carries type:"function" beside the index and arguments, as the reference
// spells it (litellm: anthropic/chat/handler.py:750-762).
func TestMessagesToChatToolArgumentDeltasCarryType(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
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
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	continuation := false
	for _, c := range chatMsgParityEvents(t, dst.String()) {
		choices, _ := c["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
		calls, ok := delta["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			continue
		}
		call := calls[0].(map[string]any)
		fn, _ := call["function"].(map[string]any)
		if _, has := fn["name"]; has {
			continue // the open chunk, not a continuation
		}
		continuation = true
		if call["type"] != "function" {
			t.Fatalf("continuation entry = %#v, want type:\"function\"", call)
		}
	}
	if !continuation {
		t.Fatalf("no continuation chunk:\n%s", dst.String())
	}
}
