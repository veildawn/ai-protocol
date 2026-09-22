package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestPipeChatToMessagesText(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"He"}}]}`,
		``,
		`data: {"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"llo"}}]}`,
		``,
		`data: {"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	n, err := PipeStream(Chat, Messages, &dst, strings.NewReader(src), StreamOpts{Model: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("wrote 0")
	}
	out := dst.String()
	for _, want := range []string{"event: message_start", "text_delta", "He", "llo", "message_stop", "end_turn"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in %s", want, out)
		}
	}
}

func TestPipeMessagesToChatText(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","role":"assistant","content":[]}}`,
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
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	_, err := PipeStream(Messages, Chat, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"content":"Hi"`) && !strings.Contains(out, `"content": "Hi"`) {
		t.Fatalf("missing content: %s", out)
	}
	if !strings.Contains(out, "[DONE]") {
		t.Fatalf("missing DONE: %s", out)
	}
}

func TestPipeChatToResponses(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"A"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	_, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, "response.output_text.delta") || !strings.Contains(out, "response.completed") {
		t.Fatalf("%s", out)
	}
}

func TestPipeSameDialectCopies(t *testing.T) {
	in := "data: hello\n\n"
	var dst bytes.Buffer
	n, err := PipeStream(Chat, "openai", &dst, strings.NewReader(in), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(len(in)) || dst.String() != in {
		t.Fatalf("copy failed n=%d out=%q", n, dst.String())
	}
}

func TestPipeMessagesToChatMergesCacheUsage(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":2,"output_tokens":1,"cache_read_input_tokens":54033}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
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
	var usage map[string]any
	for _, line := range strings.Split(dst.String(), "\n") {
		payload, ok := strings.CutPrefix(strings.TrimSpace(line), "data:")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		if u, ok := chunk["usage"].(map[string]any); ok {
			usage = u
		}
	}
	if usage == nil {
		t.Fatalf("no usage frame:\n%s", dst.String())
	}
	if usage["prompt_tokens"] != float64(54035) {
		t.Fatalf("prompt_tokens=%v want 54035 inclusive of cache; body=%v\n%s", usage["prompt_tokens"], usage, dst.String())
	}
	details, _ := usage["prompt_tokens_details"].(map[string]any)
	if details["cached_tokens"] != float64(54033) {
		t.Fatalf("cached_tokens=%v want 54033; body=%v", details["cached_tokens"], usage)
	}
	if usage["completion_tokens"] != float64(4) {
		t.Fatalf("completion_tokens=%v want 4", usage["completion_tokens"])
	}
}

func TestPipeStreamWithReportsSourceTerminal(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Terminal || result.Truncated || result.InBandErr != nil {
		t.Fatalf("result=%+v", result)
	}
}

func TestPipeStreamWithReportsTruncation(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","choices":[{"delta":{"content":"hi"}}]}`,
		``,
		``,
	}, "\n")
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Terminal || !result.Truncated || result.InBandErr != nil {
		t.Fatalf("result=%+v", result)
	}
}

func TestPipeStreamWithReportsInBandError(t *testing.T) {
	src := "data: {\"error\":{\"message\":\"boom\"}}\n\n"
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatalf("PipeStreamWith error should be metadata: %v", err)
	}
	if !result.Terminal || result.Truncated {
		t.Fatalf("result=%+v", result)
	}
	if result.InBandErr == nil || result.InBandErr.Error() != "boom" {
		t.Fatalf("InBandErr=%v", result.InBandErr)
	}
}

// chatToolCallSource mirrors a real Chat Completions tool-call stream: every
// chunk repeats the call id while only the first carries the function name.
// A converter that keys "new call" off either field emits output_item.added
// once per chunk and buries the call.
const chatToolCallSource = `data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec_command","arguments":""}}]}}]}

data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"cmd\": \"cat"}}]}}]}

data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":" secret.txt\"}"}}]}}]}

data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]

`

// responseEvents parses the SSE output into (event, decoded-body) pairs.
func responseEvents(t *testing.T, raw string) []map[string]any {
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

func eventsOfType(evs []map[string]any, typ string) []map[string]any {
	var out []map[string]any
	for _, e := range evs {
		if e["type"] == typ {
			out = append(out, e)
		}
	}
	return out
}

func completedOutput(t *testing.T, evs []map[string]any) []any {
	t.Helper()
	for _, e := range evs {
		if e["type"] == "response.completed" {
			resp, _ := e["response"].(map[string]any)
			out, _ := resp["output"].([]any)
			return out
		}
	}
	t.Fatal("no response.completed event")
	return nil
}

func TestPipeChatToResponsesToolCallLifecycle(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(chatToolCallSource), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	evs := responseEvents(t, dst.String())

	added := eventsOfType(evs, "response.output_item.added")
	if len(added) != 1 {
		t.Fatalf("output_item.added count=%d, want 1 (one per call, not per chunk):\n%s", len(added), dst.String())
	}
	item, _ := added[0]["item"].(map[string]any)
	if item["type"] != "function_call" {
		t.Fatalf("item type=%v want function_call", item["type"])
	}
	itemID, _ := item["id"].(string)
	if itemID == "" {
		t.Fatalf("function_call item missing id: %v", item)
	}
	if item["call_id"] != "call_1" || item["name"] != "exec_command" {
		t.Fatalf("item=%v", item)
	}

	argDeltas := eventsOfType(evs, "response.function_call_arguments.delta")
	if len(argDeltas) == 0 {
		t.Fatalf("no function_call_arguments.delta events:\n%s", dst.String())
	}
	var args strings.Builder
	for _, d := range argDeltas {
		if d["item_id"] != itemID {
			t.Fatalf("delta missing item_id %q: %v", itemID, d)
		}
		args.WriteString(d["delta"].(string))
	}
	if args.String() != `{"cmd": "cat secret.txt"}` {
		t.Fatalf("args=%q", args.String())
	}

	if done := eventsOfType(evs, "response.function_call_arguments.done"); len(done) != 1 {
		t.Fatalf("function_call_arguments.done count=%d", len(done))
	} else if done[0]["arguments"] != `{"cmd": "cat secret.txt"}` {
		t.Fatalf("done arguments=%v", done[0]["arguments"])
	}
	if n := len(eventsOfType(evs, "response.output_item.done")); n != 1 {
		t.Fatalf("output_item.done count=%d", n)
	}

	out := completedOutput(t, evs)
	if len(out) != 1 {
		t.Fatalf("completed output items=%d, want the function_call: %v", len(out), out)
	}
	fc, _ := out[0].(map[string]any)
	if fc["type"] != "function_call" || fc["arguments"] != `{"cmd": "cat secret.txt"}` {
		t.Fatalf("completed output item=%v", fc)
	}

	var usage map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			usage, _ = e["response"].(map[string]any)["usage"].(map[string]any)
		}
	}
	if usage["input_tokens"] != float64(10) || usage["output_tokens"] != float64(5) {
		t.Fatalf("usage=%v", usage)
	}
}

// messagesToolStream exercises every block kind the Anthropic fold sees on a
// Codex turn: thinking, tool_use, text — plus the usage that only exists on
// message_start/message_delta.
const messagesToolStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"glm","role":"assistant","content":[],"usage":{"input_tokens":42,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"exec_command","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"ls\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"done"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":77}}

event: message_stop
data: {"type":"message_stop"}

`

func TestPipeMessagesToResponsesToolAndText(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(messagesToolStream), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := responseEvents(t, dst.String())

	added := eventsOfType(evs, "response.output_item.added")
	if len(added) != 3 {
		t.Fatalf("output_item.added count=%d, want reasoning+function_call+message:\n%s", len(added), dst.String())
	}
	var itemIDs []string
	var wantTypes = []string{"reasoning", "function_call", "message"}
	for i, a := range added {
		item, _ := a["item"].(map[string]any)
		if item["type"] != wantTypes[i] {
			t.Fatalf("item %d type=%v want %v", i, item["type"], wantTypes[i])
		}
		id, _ := item["id"].(string)
		if id == "" {
			t.Fatalf("item %d missing id: %v", i, item)
		}
		if _, ok := a["output_index"]; !ok {
			t.Fatalf("added %d missing output_index: %v", i, a)
		}
		itemIDs = append(itemIDs, id)
	}
	rsID, fcID, msgID := itemIDs[0], itemIDs[1], itemIDs[2]

	for _, d := range eventsOfType(evs, "response.reasoning_summary_text.delta") {
		if d["item_id"] != rsID {
			t.Fatalf("reasoning delta item_id=%v want %v", d["item_id"], rsID)
		}
		if _, ok := d["summary_index"]; !ok {
			t.Fatalf("reasoning delta missing summary_index: %v", d)
		}
	}
	var fcArgs strings.Builder
	for _, d := range eventsOfType(evs, "response.function_call_arguments.delta") {
		if d["item_id"] != fcID {
			t.Fatalf("args delta item_id=%v want %v", d["item_id"], fcID)
		}
		fcArgs.WriteString(d["delta"].(string))
	}
	if fcArgs.String() != `{"cmd": "ls"}` {
		t.Fatalf("fc args=%q", fcArgs.String())
	}
	for _, d := range eventsOfType(evs, "response.output_text.delta") {
		if d["item_id"] != msgID {
			t.Fatalf("text delta item_id=%v want %v", d["item_id"], msgID)
		}
		if _, ok := d["content_index"]; !ok {
			t.Fatalf("text delta missing content_index: %v", d)
		}
	}
	for _, typ := range []string{
		"response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.content_part.added", "response.content_part.done",
		"response.output_text.done", "response.function_call_arguments.done",
	} {
		if len(eventsOfType(evs, typ)) == 0 {
			t.Fatalf("missing %s:\n%s", typ, dst.String())
		}
	}
	if n := len(eventsOfType(evs, "response.output_item.done")); n != 3 {
		t.Fatalf("output_item.done count=%d want 3", n)
	}

	out := completedOutput(t, evs)
	if len(out) != 3 {
		t.Fatalf("completed output items=%d: %v", len(out), out)
	}
	rs, _ := out[0].(map[string]any)
	if rs["type"] != "reasoning" {
		t.Fatalf("output[0]=%v", rs)
	}
	fc, _ := out[1].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "toolu_1" || fc["name"] != "exec_command" || fc["arguments"] != `{"cmd": "ls"}` {
		t.Fatalf("output[1]=%v", fc)
	}
	msg, _ := out[2].(map[string]any)
	if msg["type"] != "message" {
		t.Fatalf("output[2]=%v", msg)
	}
	parts, _ := msg["content"].([]any)
	p0, _ := parts[0].(map[string]any)
	if p0["text"] != "done" {
		t.Fatalf("message content=%v", msg["content"])
	}

	var usage map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			usage, _ = e["response"].(map[string]any)["usage"].(map[string]any)
		}
	}
	if usage["input_tokens"] != float64(42) || usage["output_tokens"] != float64(77) {
		t.Fatalf("usage=%v want in=42 out=77 (message_delta carries the real output count)", usage)
	}
}

func TestPipeMessagesToResponsesTextLifecycle(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_9","model":"glm","role":"assistant","content":[],"usage":{"input_tokens":9,"output_tokens":1}}}`,
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
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := responseEvents(t, dst.String())

	added := eventsOfType(evs, "response.output_item.added")
	if len(added) != 1 {
		t.Fatalf("text block must open a message item; added=%d:\n%s", len(added), dst.String())
	}
	item, _ := added[0]["item"].(map[string]any)
	itemID, _ := item["id"].(string)
	if item["type"] != "message" || itemID == "" {
		t.Fatalf("item=%v", item)
	}
	if len(eventsOfType(evs, "response.content_part.added")) != 1 {
		t.Fatalf("content_part.added missing:\n%s", dst.String())
	}
	for _, d := range eventsOfType(evs, "response.output_text.delta") {
		if d["item_id"] != itemID {
			t.Fatalf("delta without item_id: %v", d)
		}
	}

	out := completedOutput(t, evs)
	msg, _ := out[0].(map[string]any)
	if msg["type"] != "message" || msg["id"] != itemID {
		t.Fatalf("completed output=%v", out[0])
	}
	var usage map[string]any
	for _, e := range evs {
		if e["type"] == "response.completed" {
			usage, _ = e["response"].(map[string]any)["usage"].(map[string]any)
		}
	}
	if usage["input_tokens"] != float64(9) || usage["output_tokens"] != float64(3) {
		t.Fatalf("usage dropped: %v", usage)
	}
}

func TestPipeStreamWithSameDialectReportsTerminal(t *testing.T) {
	src := strings.Join([]string{
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	result, err := PipeStreamWith(Messages, "anthropic", &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Terminal || result.Truncated || result.InBandErr != nil {
		t.Fatalf("result=%+v", result)
	}
	if dst.String() != src {
		t.Fatalf("same dialect changed bytes: %q", dst.String())
	}
}
