package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// Regression pins for the chatresponses response/stream folds, each aligned to
// a LiteLLM v1.100.1 behavior the audit found this codec diverging from. The
// litellm references are named per case; the venv-verified behaviors were run
// against litellm 1.100.1's own transformers.

// An incomplete response rewrites EVERY choice's finish_reason — the tool-call
// choice included (litellm transform_response, completed_extras/
// litellm_responses_transformation/transformation.py:886-894). A ceiling cut
// that happened to contain a call is still a ceiling cut; reporting
// "tool_calls" tells the client the model chose to call a tool when it ran out
// of room. A completed turn with a call still finishes "tool_calls".
func TestResponsesToChatIncompleteToolCallFinishReason(t *testing.T) {
	incomplete := `{"id":"r1","model":"m","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"function_call","call_id":"c1","name":"fn","arguments":"{}"}]}`
	out, err := ConvertResponse(Responses, Chat, []byte(incomplete))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	choice := m["choices"].([]any)[0].(map[string]any)
	if got := choice["finish_reason"]; got != "length" {
		t.Fatalf("finish_reason = %v, want length for an incomplete turn", got)
	}
	msg := choice["message"].(map[string]any)
	if calls, ok := msg["tool_calls"].([]any); !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v, want the call preserved on the message", msg["tool_calls"])
	}

	completed := `{"id":"r1","model":"m","status":"completed","output":[{"type":"function_call","call_id":"c1","name":"fn","arguments":"{}"}]}`
	out2, err := ConvertResponse(Responses, Chat, []byte(completed))
	if err != nil {
		t.Fatal(err)
	}
	var m2 map[string]any
	if err := json.Unmarshal(out2, &m2); err != nil {
		t.Fatal(err)
	}
	if got := m2["choices"].([]any)[0].(map[string]any)["finish_reason"]; got != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls for a completed call turn", got)
	}
}

// The non-streaming chat->responses fold shapes a reasoning item the way the
// reference does (litellm _extract_reasoning_output_items + _encode_thinking_
// blocks, responses/litellm_completion_transformation/transformation.py:2369+
// and :2355+): one item carrying the reasoning text as an output_text part
// (annotations included, no summary field) and every replayable thinking block
// — signature or data — inside one compact encrypted_content JSON array.
// Blocks with neither are left out.
func TestChatResponseToResponsesReasoningItemShape(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi","reasoning_content":"chain of thought","thinking_blocks":[` +
		`{"type":"thinking","thinking":"t1","signature":"sig1"},` +
		`{"type":"thinking","thinking":"t2","signature":"sig2","data":"d2"},` +
		`{"type":"thinking","thinking":"no replay material"}` +
		`]}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	var reasoning map[string]any
	for _, item := range m["output"].([]any) {
		im := item.(map[string]any)
		if im["type"] == "reasoning" {
			if reasoning != nil {
				t.Fatalf("two reasoning items for one choice: %#v", m["output"])
			}
			reasoning = im
		}
	}
	if reasoning == nil {
		t.Fatalf("no reasoning item in %#v", m["output"])
	}
	if _, has := reasoning["summary"]; has {
		t.Fatalf("reasoning item carries a summary; the non-streaming fold uses content: %#v", reasoning)
	}
	if reasoning["status"] != "completed" || reasoning["role"] != "assistant" {
		t.Fatalf("reasoning item = %#v, want status completed and role assistant", reasoning)
	}
	parts := reasoning["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("content = %#v, want the one output_text part", reasoning["content"])
	}
	part := parts[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "chain of thought" {
		t.Fatalf("content part = %#v, want the reasoning text as output_text", part)
	}
	if anns, ok := part["annotations"].([]any); !ok || len(anns) != 0 {
		t.Fatalf("content part annotations = %#v, want an empty list", part["annotations"])
	}
	var blocks []any
	if err := json.Unmarshal([]byte(reasoning["encrypted_content"].(string)), &blocks); err != nil {
		t.Fatalf("encrypted_content is not a JSON array: %#v", reasoning["encrypted_content"])
	}
	if len(blocks) != 2 {
		t.Fatalf("encrypted_content blocks = %#v, want the two blocks carrying signature/data only", blocks)
	}
	b0 := blocks[0].(map[string]any)
	if b0["signature"] != "sig1" || b0["thinking"] != "t1" {
		t.Fatalf("first preserved block = %#v", b0)
	}
	b1 := blocks[1].(map[string]any)
	if b1["signature"] != "sig2" || b1["data"] != "d2" {
		t.Fatalf("second preserved block = %#v", b1)
	}
}

// Item status comes from the finish_reason (litellm
// _map_chat_completion_finish_reason_to_responses_status: length/content_filter/
// refusal -> incomplete). A message a ceiling cut left behind reports
// incomplete; a function_call item keeps completed — the reference derives that
// one from the call itself (function_definition.get("status") or "completed").
func TestChatResponseToResponsesMessageStatusFromFinish(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"length","message":{"role":"assistant","content":"partial","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	items := m["output"].([]any)
	if len(items) != 2 {
		t.Fatalf("output = %#v, want message then function_call", items)
	}
	if got := items[0].(map[string]any)["status"]; got != "incomplete" {
		t.Fatalf("message item status = %v, want incomplete for finish_reason length", got)
	}
	if got := items[1].(map[string]any)["status"]; got != "completed" {
		t.Fatalf("function_call item status = %v, want completed", got)
	}
}

// The same finish-derived status lands in the streaming response.completed (or
// .incomplete) output array, while the per-item done events keep "completed" —
// both as litellm's own split does (non-stream transform builds the completed
// payload; streaming_iterator.py:693 hardcodes the done events).
func TestPipeChatToResponsesCompletedOutputMessageStatus(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"content":"partial"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var terminal, itemDone map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			switch obj["type"] {
			case "response.incomplete":
				terminal = obj["response"].(map[string]any)
			case "response.output_item.done":
				itemDone = obj["item"].(map[string]any)
			}
		}
	}
	if terminal == nil {
		t.Fatalf("no response.incomplete terminal:\n%s", dst.String())
	}
	if terminal["status"] != "incomplete" {
		t.Fatalf("terminal status = %v", terminal["status"])
	}
	msg := terminal["output"].([]any)[0].(map[string]any)
	if msg["type"] != "message" || msg["status"] != "incomplete" {
		t.Fatalf("completed output message = %#v, want an incomplete message item", msg)
	}
	if itemDone == nil || itemDone["status"] != "completed" {
		t.Fatalf("output_item.done message = %#v, want status completed as litellm's streaming path does", itemDone)
	}
}

// Each reasoning item OVERWRITES reasoning_content rather than joining (litellm
// _convert_response_output_to_choices, completed_extras/
// litellm_responses_transformation/transformation.py:656): two reasoning items
// before the message leave only the second chain on the chat message.
func TestResponsesToChatKeepsLastReasoningItem(t *testing.T) {
	body := `{"id":"r1","model":"m","status":"completed","output":[` +
		`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"first chain"}]},` +
		`{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"second chain"}]},` +
		`{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer","annotations":[]}]}` +
		`]}`
	out, err := ConvertResponse(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	msg := m["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if got := msg["reasoning_content"]; got != "second chain" {
		t.Fatalf("reasoning_content = %#v, want only the last item's text", got)
	}
	if got := msg["content"]; got != "answer" {
		t.Fatalf("content = %#v", got)
	}
}

// output_text annotations (url_citation and friends) cross onto the chat wire
// as message.annotations, verbatim (litellm
// _convert_annotations_to_chat_format — the two dialects spell them the same).
func TestResponsesToChatKeepsOutputTextAnnotations(t *testing.T) {
	body := `{"id":"r1","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"see A","annotations":[{"type":"url_citation","start_index":0,"end_index":4,"url":"https://ex.example/a","title":"A"}]}]}]}`
	out, err := ConvertResponse(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	msg := m["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	anns, ok := msg["annotations"].([]any)
	if !ok || len(anns) != 1 {
		t.Fatalf("annotations = %#v, want the url_citation preserved", msg["annotations"])
	}
	cit := anns[0].(map[string]any)
	if cit["type"] != "url_citation" || cit["url"] != "https://ex.example/a" || cit["title"] != "A" {
		t.Fatalf("citation = %#v", cit)
	}
	if cit["start_index"] != float64(0) || cit["end_index"] != float64(4) {
		t.Fatalf("citation offsets = %#v", cit)
	}
}

// created_at is required on every Responses payload — the OpenAI SDK's
// Response model rejects one without it. litellm stamps the opening shells
// from the clock and the completed payload from the chunk's created
// (streaming_iterator.py:418, transformation.py:2229); the no-clock contract
// means both here take the source chunks' created, which is present either way.
func TestPipeChatToResponsesCarriesCreatedAt(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":1741290958,"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{
		"response.created":     1741290958,
		"response.in_progress": 1741290958,
		"response.completed":   1741290958,
	}
	seen := map[string]float64{}
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if _, wanted := want[obj["type"].(string)]; obj["type"] != nil && wanted {
				resp := obj["response"].(map[string]any)
				created, ok := resp["created_at"].(float64)
				if !ok {
					t.Fatalf("%v payload has no created_at: %#v", obj["type"], resp)
				}
				seen[obj["type"].(string)] = created
			}
		}
	}
	for typ, wantCreated := range want {
		if got, ok := seen[typ]; !ok || got != wantCreated {
			t.Fatalf("%v created_at = %v (seen %v), want %v", typ, got, seen[typ], wantCreated)
		}
	}
}

// Every chat.completion.chunk carries `created` — a required field a strict
// chat client rejects the chunk without. The reference stamps its chunks from
// the wall clock; this library has no clock by contract, so the Responses
// response's created_at is mapped onto it instead.
func TestResponsesToChatStreamChunksCarryCreated(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m","created_at":1741290958}}`,
		``,
		`data: {"type":"response.output_text.delta","delta":"hi"}`,
		``,
		`data: {"type":"response.completed","response":{"id":"r1","model":"m","created_at":1741290958,"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	chunks := 0
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok || strings.TrimSpace(data) == "[DONE]" {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["object"] != "chat.completion.chunk" {
				continue
			}
			chunks++
			if got := obj["created"]; got != float64(1741290958) {
				t.Fatalf("chunk created = %#v, want the source response's created_at", got)
			}
		}
	}
	if chunks < 3 {
		t.Fatalf("saw %d chunks, want the role, content and finish chunks all checked:\n%s", chunks, dst.String())
	}
}

// A function_call output item's id IS its call id: the reference puts the chat
// tool_call id in both fields (ResponseFunctionToolCall(id=tool.id,
// call_id=tool.id)), both non-streaming and in the streaming added frame.
func TestChatResponseToResponsesFunctionCallIdIsCallId(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"hi","tool_calls":[{"id":"call_9","type":"function","function":{"name":"fn","arguments":"{\"a\":1}"}}]}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	var fc map[string]any
	for _, item := range m["output"].([]any) {
		if im := item.(map[string]any); im["type"] == "function_call" {
			fc = im
		}
	}
	if fc == nil {
		t.Fatalf("no function_call item in %#v", m["output"])
	}
	if fc["id"] != "call_9" || fc["call_id"] != "call_9" {
		t.Fatalf("function_call id=%v call_id=%v, want both to be the chat tool_call id", fc["id"], fc["call_id"])
	}

	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"fn","arguments":""}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var addedItem map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["type"] == "response.output_item.added" {
				addedItem, _ = obj["item"].(map[string]any)
			}
		}
	}
	if addedItem == nil {
		t.Fatalf("no output_item.added:\n%s", dst.String())
	}
	if addedItem["id"] != "call_9" || addedItem["call_id"] != "call_9" {
		t.Fatalf("added item id=%v call_id=%v, want the chat tool_call id in both", addedItem["id"], addedItem["call_id"])
	}
}

// A message item is folded for EVERY choice, empty or missing text included —
// the reference emits one even for a tool-call-only turn, with the text part
// rendering null (no content) or empty (empty content) rather than the item
// vanishing from the output array.
func TestChatResponseToResponsesToolOnlyKeepsMessageItem(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		want    any
	}{
		{"null content", `"content":null,`, nil},
		{"empty content", `"content":"",`, ""},
	} {
		body := `{"id":"c1","model":"m","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant",` + tc.content + `"tool_calls":[{"id":"call_9","type":"function","function":{"name":"fn","arguments":"{}"}}]}}]}`
		out, err := ConvertResponse(Chat, Responses, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		items := m["output"].([]any)
		if len(items) != 2 {
			t.Fatalf("%s: output = %#v, want message then function_call", tc.name, items)
		}
		msg := items[0].(map[string]any)
		if msg["type"] != "message" {
			t.Fatalf("%s: output[0] = %#v, want the message item", tc.name, msg)
		}
		part := msg["content"].([]any)[0].(map[string]any)
		if part["text"] != tc.want {
			t.Fatalf("%s: message text = %#v, want %#v", tc.name, part["text"], tc.want)
		}
	}
}

// The streaming bridge shares output_index 0 between the text items: whichever
// of reasoning/message opens first owns the slot's single output_item.added,
// the second rides it with no added frame of its own, and no part.added exists
// for a reasoning item; tool calls count from 1 with the chat call id as the
// item id, and an argument fragment is re-sliced into ~10-character deltas
// whose concatenation is the arguments unchanged. (The reference drops the
// first reasoning delta to its one-shot queueing; this library keeps every
// delta — the completed text is the same and no content is lost mid-stream.)
func TestPipeChatToResponsesSharedTextSlotAndToolCallIndexes(t *testing.T) {
	longArgs := `{"q":"abcdefghijklmnopqrstuvwxyz0123"}` // 38 chars
	argsJSON, err := json.Marshal(longArgs)
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"reasoning_content":"thinking hard"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hello world"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":` + string(argsJSON) + `}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var evs []map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok || strings.TrimSpace(data) == "[DONE]" {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			evs = append(evs, obj)
		}
	}
	var added []map[string]any
	for _, e := range evs {
		if e["type"] == "response.output_item.added" {
			added = append(added, e)
		}
	}
	if len(added) != 2 {
		t.Fatalf("output_item.added count = %d, want 2 (reasoning owns slot 0, message never opens):\n%s", len(added), dst.String())
	}
	if got := added[0]["output_index"]; got != float64(0) {
		t.Fatalf("first added output_index = %v, want 0", got)
	}
	if it := added[0]["item"].(map[string]any); it["type"] != "reasoning" {
		t.Fatalf("first added item = %#v, want reasoning", it)
	}
	if got := added[1]["output_index"]; got != float64(1) {
		t.Fatalf("function_call added output_index = %v, want 1 (slot 0 is reserved for the text item)", got)
	}
	fcItem := added[1]["item"].(map[string]any)
	if fcItem["type"] != "function_call" || fcItem["id"] != "call_1" || fcItem["call_id"] != "call_1" {
		t.Fatalf("function_call added item = %#v, want the chat call id as id and call_id", fcItem)
	}
	for _, partAdded := range []string{"response.content_part.added", "response.reasoning_summary_part.added"} {
		for _, e := range evs {
			if e["type"] == partAdded {
				t.Fatalf("%s emitted; the reference opens no part frame for a slot the reasoning item owns", partAdded)
			}
		}
	}
	var deltas []string
	for _, e := range evs {
		switch e["type"] {
		case "response.function_call_arguments.delta":
			deltas = append(deltas, e["delta"].(string))
			if e["item_id"] != "call_1" || e["output_index"] != float64(1) {
				t.Fatalf("args delta = %#v, want item_id call_1 at output_index 1", e)
			}
		}
	}
	if strings.Join(deltas, "") != longArgs {
		t.Fatalf("args deltas joined = %q, want %q", strings.Join(deltas, ""), longArgs)
	}
	for i, want := range []int{10, 10, 10, 8} {
		if len(deltas[i]) != want {
			t.Fatalf("args delta %d length = %d, want %d (deltas %v)", i, len(deltas[i]), want, deltas)
		}
	}
	// The message rides slot 0 unopened but still streams and closes.
	var textDeltas, msgDones int
	for _, e := range evs {
		if e["type"] == "response.output_text.delta" {
			if e["output_index"] != float64(0) {
				t.Fatalf("text delta output_index = %v, want the shared slot 0", e["output_index"])
			}
			textDeltas++
		}
		if e["type"] == "response.output_item.done" {
			if it := e["item"].(map[string]any); it["type"] == "message" {
				msgDones++
			}
		}
	}
	if textDeltas != 1 || msgDones != 1 {
		t.Fatalf("text deltas = %d, message item dones = %d, want the message to stream and close on slot 0", textDeltas, msgDones)
	}
}

// A tool-first chunk burns the shared text slot's opening frame without
// emitting it — late text rides slot 0 unopened (deltas and close frames, no
// output_item.added / content_part.added), exactly as the reference's one-shot
// gate behaves.
func TestPipeChatToResponsesToolFirstBurnsTextSlot(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"late text"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var evs []map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			evs = append(evs, obj)
		}
	}
	var addedTypes []string
	textDelta, textDone := false, false
	for _, e := range evs {
		switch e["type"] {
		case "response.output_item.added":
			addedTypes = append(addedTypes, e["item"].(map[string]any)["type"].(string))
			if e["output_index"] != float64(1) {
				t.Fatalf("function_call added output_index = %v, want 1", e["output_index"])
			}
		case "response.output_text.delta":
			textDelta = true
			if e["output_index"] != float64(0) {
				t.Fatalf("late text delta output_index = %v, want slot 0", e["output_index"])
			}
		case "response.output_text.done":
			textDone = true
		case "response.content_part.added":
			t.Fatalf("content_part.added emitted for a slot a tool call burned:\n%s", dst.String())
		}
	}
	if len(addedTypes) != 1 || addedTypes[0] != "function_call" {
		t.Fatalf("added items = %v, want only the function_call", addedTypes)
	}
	if !textDelta || !textDone {
		t.Fatalf("late text must still stream and close on slot 0 (delta=%v done=%v):\n%s", textDelta, textDone, dst.String())
	}
}

// The chat->responses usage fold reads only the chat spellings and passes the
// reported total through untouched; the details objects are rebuilt on the
// reference's key whitelist — cached_tokens/reasoning_tokens filled in as 0
// once the details object exists, unknown keys dropped, and no details object
// means no details field at all.
func TestChatToResponsesUsageWhitelistAndTotalPassthrough(t *testing.T) {
	body := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":999,"prompt_tokens_details":{"cached_tokens":4,"custom_thing":7},"completion_tokens_details":{"audio_tokens":2}}}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if usage["input_tokens"] != float64(10) || usage["output_tokens"] != float64(5) {
		t.Fatalf("usage parts = %#v", usage)
	}
	if usage["total_tokens"] != float64(999) {
		t.Fatalf("total_tokens = %v, want the reported 999 passed through, not recomputed", usage["total_tokens"])
	}
	inDetails := usage["input_tokens_details"].(map[string]any)
	if inDetails["cached_tokens"] != float64(4) {
		t.Fatalf("input_tokens_details = %#v, want cached_tokens 4", inDetails)
	}
	if _, ok := inDetails["custom_thing"]; ok {
		t.Fatalf("input_tokens_details kept an off-whitelist key: %#v", inDetails)
	}
	outDetails := usage["output_tokens_details"].(map[string]any)
	if outDetails["reasoning_tokens"] != float64(0) {
		t.Fatalf("output_tokens_details = %#v, want reasoning_tokens filled in as 0", outDetails)
	}
	if outDetails["audio_tokens"] != float64(2) {
		t.Fatalf("output_tokens_details = %#v, want audio_tokens 2", outDetails)
	}

	// No details objects, total 0: nothing invented, nothing recomputed.
	bare := `{"id":"c1","model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":0}}`
	out2, err := ConvertResponse(Chat, Responses, []byte(bare))
	if err != nil {
		t.Fatal(err)
	}
	var m2 map[string]any
	if err := json.Unmarshal(out2, &m2); err != nil {
		t.Fatal(err)
	}
	u2 := m2["usage"].(map[string]any)
	if u2["total_tokens"] != float64(0) {
		t.Fatalf("total_tokens = %v, want 0 passed through", u2["total_tokens"])
	}
	for _, k := range []string{"input_tokens_details", "output_tokens_details"} {
		if _, ok := u2[k]; ok {
			t.Fatalf("usage invented %s: %#v", k, u2)
		}
	}
}

// The responses->chat total is always the sum of the parts — a reported total
// that disagrees is discarded, the way the reference folds it.
func TestResponsesToChatUsageTotalIsPartsSum(t *testing.T) {
	body := `{"id":"r1","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":999}}`
	out, err := ConvertResponse(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	usage := m["usage"].(map[string]any)
	if usage["total_tokens"] != float64(30) {
		t.Fatalf("total_tokens = %v, want 30 = prompt + completion", usage["total_tokens"])
	}
}

// The fold opens with a bare empty-content delta — no role — exactly the
// reference's first chunk.
func TestResponsesToChatFirstChunkIsBareEmptyContent(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m","created_at":5}}`,
		``,
		`data: {"type":"response.completed","response":{"id":"r1","model":"m","created_at":5,"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var first map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok || strings.TrimSpace(data) == "[DONE]" {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["object"] == "chat.completion.chunk" && first == nil {
				first = obj
			}
		}
	}
	if first == nil {
		t.Fatalf("no chunks:\n%s", dst.String())
	}
	delta := first["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if len(delta) != 1 || delta["content"] != "" {
		t.Fatalf("first delta = %#v, want exactly {content:%q}", delta, "")
	}
}

// An event the fold maps nothing onto still advances the stream with a minimal
// empty-content chunk, the reference's cadence. The reasoning aliases are this
// library's deliberate exception: they carry model output the reference
// discards, and they fold to reasoning_content instead.
func TestResponsesToChatUnhandledEventsEmitEmptyChunk(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed"}}`,
		``,
		`data: {"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0}`,
		``,
		`data: {"type":"response.reasoning_summary.delta","item_id":"rs_1","output_index":0,"delta":"alias reasoning"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var deltas []map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["object"] == "chat.completion.chunk" {
				choice := obj["choices"].([]any)[0].(map[string]any)
				deltas = append(deltas, choice["delta"].(map[string]any))
			}
		}
	}
	if len(deltas) != 3 {
		t.Fatalf("chunk count = %d, want one per event:\n%s", len(deltas), dst.String())
	}
	for i := 0; i < 2; i++ {
		if len(deltas[i]) != 1 || deltas[i]["content"] != "" {
			t.Fatalf("delta %d = %#v, want the minimal {content:%q}", i, deltas[i], "")
		}
	}
	if deltas[2]["reasoning_content"] != "alias reasoning" {
		t.Fatalf("alias delta = %#v, want the reasoning text preserved", deltas[2])
	}
}

// Usage rides the terminal chunk only when the source reported one: a terminal
// without usage carries usage:null — not an invented zeros object.
func TestResponsesToChatTerminalUsageNullWhenAbsent(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m","created_at":5}}`,
		``,
		`data: {"type":"response.completed","response":{"id":"r1","model":"m","created_at":5,"status":"completed","output":[]}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(dst.String(), `"usage":null`) {
		t.Fatalf("terminal without usage should carry usage:null:\n%s", dst.String())
	}
	if strings.Contains(dst.String(), `"prompt_tokens":0`) {
		t.Fatalf("usage was invented as zeros for a source that reported none:\n%s", dst.String())
	}
}

// A Response object's required members — output, parallel_tool_calls,
// tool_choice, tools — ride every payload this fold emits, with the
// reference's own defaults. A strict Responses client (the OpenAI SDK's
// Response model) rejects a body without them; the request's real tool list
// is request context a pure response fold never sees.
func TestChatToResponsesBodyCarriesRequiredToolFields(t *testing.T) {
	body := `{"id":"c1","model":"m","created":5,"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`
	out, err := ConvertResponse(Chat, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["output"].([]any); !ok {
		t.Fatalf("output = %#v, want an array", m["output"])
	}
	if m["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v, want false", m["parallel_tool_calls"])
	}
	if m["tool_choice"] != "auto" {
		t.Fatalf("tool_choice = %v, want auto", m["tool_choice"])
	}
	if tools, ok := m["tools"].([]any); !ok || len(tools) != 0 {
		t.Fatalf("tools = %#v, want an empty array", m["tools"])
	}
}

// The streaming shells and the terminal payload carry the same required
// members, so a client validating either the opening frames or the closing
// snapshot accepts the stream: the shells with the reference's opening values
// (parallel_tool_calls true), the terminal with its closing ones (false).
func TestPipeChatToResponsesShellCarriesRequiredFields(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	shells := 0
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			typ, _ := obj["type"].(string)
			if typ != "response.created" && typ != "response.in_progress" && typ != "response.completed" {
				continue
			}
			resp := obj["response"].(map[string]any)
			if _, ok := resp["output"].([]any); !ok {
				t.Fatalf("%v shell missing output: %#v", typ, resp)
			}
			if _, ok := resp["tool_choice"].(string); !ok {
				t.Fatalf("%v shell missing tool_choice: %#v", typ, resp)
			}
			if _, ok := resp["tools"].([]any); !ok {
				t.Fatalf("%v shell missing tools: %#v", typ, resp)
			}
			if _, ok := resp["parallel_tool_calls"].(bool); !ok {
				t.Fatalf("%v shell missing parallel_tool_calls: %#v", typ, resp)
			}
			if typ != "response.completed" {
				if resp["parallel_tool_calls"] != true {
					t.Fatalf("%v shell parallel_tool_calls = %v, want the reference's opening true", typ, resp["parallel_tool_calls"])
				}
				shells++
			} else {
				if resp["parallel_tool_calls"] != false {
					t.Fatalf("terminal parallel_tool_calls = %v, want the reference's closing false", resp["parallel_tool_calls"])
				}
				// The terminal's message parts carry their annotations member
				// even when empty — the SDK's ResponseOutputText requires it.
				for _, it := range resp["output"].([]any) {
					if im, ok := it.(map[string]any); ok && im["type"] == "message" {
						for _, part := range im["content"].([]any) {
							if pm, ok := part.(map[string]any); ok {
								if _, ok := pm["annotations"].([]any); !ok {
									t.Fatalf("output part missing annotations: %#v", pm)
								}
							}
						}
					}
				}
			}
		}
	}
	if shells != 2 {
		t.Fatalf("validated %d shells, want created + in_progress:\n%s", shells, dst.String())
	}
}

// Each message item folds to its own choice, index counting up, with the
// accumulated calls closing the turn as one trailing tool-calls choice — the
// reference's per-message choices rather than one concatenated turn. An
// incomplete turn rewrites every choice's finish_reason.
func TestResponsesToChatMultiMessageItemsAreSeparateChoices(t *testing.T) {
	body := `{"id":"r1","model":"m","status":"completed","output":[` +
		`{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello"}]},` +
		`{"type":"message","id":"m2","role":"assistant","status":"completed","content":[{"type":"output_text","text":"world"}]},` +
		`{"type":"function_call","call_id":"call_1","name":"fn","arguments":"{}"}` +
		`]}`
	out, err := ConvertResponse(Responses, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	choices := m["choices"].([]any)
	if len(choices) != 3 {
		t.Fatalf("choices = %#v, want one per message item plus the tool-call choice", choices)
	}
	c0 := choices[0].(map[string]any)
	if c0["index"] != float64(0) || c0["finish_reason"] != "stop" {
		t.Fatalf("choice 0 = %#v", c0)
	}
	if msg := c0["message"].(map[string]any); msg["content"] != "Hello" {
		t.Fatalf("choice 0 content = %#v, want Hello distinct from world", msg["content"])
	}
	c1 := choices[1].(map[string]any)
	if c1["index"] != float64(1) || c1["message"].(map[string]any)["content"] != "world" {
		t.Fatalf("choice 1 = %#v", c1)
	}
	c2 := choices[2].(map[string]any)
	if c2["finish_reason"] != "tool_calls" {
		t.Fatalf("tool choice finish = %v, want tool_calls", c2["finish_reason"])
	}
	if calls := c2["message"].(map[string]any)["tool_calls"].([]any); len(calls) != 1 {
		t.Fatalf("tool choice calls = %#v", calls)
	}

	incomplete := `{"id":"r1","model":"m","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[` +
		`{"type":"message","id":"m1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello"}]},` +
		`{"type":"function_call","call_id":"call_1","name":"fn","arguments":"{}"}` +
		`]}`
	out2, err := ConvertResponse(Responses, Chat, []byte(incomplete))
	if err != nil {
		t.Fatal(err)
	}
	var m2 map[string]any
	if err := json.Unmarshal(out2, &m2); err != nil {
		t.Fatal(err)
	}
	for i, ch := range m2["choices"].([]any) {
		if got := ch.(map[string]any)["finish_reason"]; got != "length" {
			t.Fatalf("incomplete choice %d finish = %v, want length on every choice", i, got)
		}
	}
}

// Every output_text part shape the stream emits carries its annotations member
// even when empty — the added part, the done part, and the message item's
// content inside output_item.done — because a strict Responses client (the
// OpenAI SDK's ResponseOutputText) requires it on each. litellm's own frames
// were verified to carry it too; its remaining event-model rejections
// (logprobs on text deltas/dones, name on arguments.done) hit the reference's
// shapes identically, so nothing further aligns there.
func TestPipeChatToResponsesPartFramesCarryAnnotations(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	checked := map[string]bool{}
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			var part map[string]any
			switch obj["type"] {
			case "response.content_part.added", "response.content_part.done":
				part, _ = obj["part"].(map[string]any)
			case "response.output_item.done":
				item, _ := obj["item"].(map[string]any)
				parts, _ := item["content"].([]any)
				if len(parts) > 0 {
					part, _ = parts[0].(map[string]any)
				}
			}
			if part == nil {
				continue
			}
			if anns, ok := part["annotations"].([]any); !ok {
				t.Fatalf("%v part missing annotations: %#v", obj["type"], part)
			} else if anns == nil {
				t.Fatalf("%v part annotations is null: %#v", obj["type"], part)
			}
			checked[obj["type"].(string)] = true
		}
	}
	for _, typ := range []string{"response.content_part.added", "response.content_part.done", "response.output_item.done"} {
		if !checked[typ] {
			t.Fatalf("no %v event seen:\n%s", typ, dst.String())
		}
	}
}

// The reasoning item's added frame declares status in_progress the way the
// reference's does, and the completed snapshot's reasoning item takes the
// reference's non-streaming shape — role, finish-derived status, the reasoning
// text as an annotated output_text part — while the in-stream output_item.done
// keeps the summary shape the reference's own streaming path emits. A ceiling
// cut marks BOTH the reasoning and the message item incomplete in the snapshot.
func TestPipeChatToResponsesReasoningAddedStatusAndCompletedShape(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"reasoning_content":"pondering"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"answer"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":9,"total_tokens":10}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var addedReasoning, doneReasoning, terminal map[string]any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			switch obj["type"] {
			case "response.output_item.added":
				if item, _ := obj["item"].(map[string]any); item["type"] == "reasoning" {
					addedReasoning = item
				}
			case "response.output_item.done":
				if item, _ := obj["item"].(map[string]any); item["type"] == "reasoning" {
					doneReasoning = item
				}
			case "response.incomplete":
				terminal, _ = obj["response"].(map[string]any)
			}
		}
	}
	if addedReasoning == nil || addedReasoning["status"] != "in_progress" {
		t.Fatalf("reasoning added item = %#v, want status in_progress", addedReasoning)
	}
	if doneReasoning == nil {
		t.Fatalf("no reasoning output_item.done:\n%s", dst.String())
	}
	if _, has := doneReasoning["summary"]; !has {
		t.Fatalf("in-stream reasoning done item = %#v, want the summary shape the reference's streaming path emits", doneReasoning)
	}
	if terminal == nil {
		t.Fatalf("no response.incomplete terminal:\n%s", dst.String())
	}
	out := terminal["output"].([]any)
	if len(out) != 2 {
		t.Fatalf("completed output = %#v, want reasoning then message", out)
	}
	rs := out[0].(map[string]any)
	if _, has := rs["summary"]; has {
		t.Fatalf("snapshot reasoning item = %#v, want the non-streaming content shape", rs)
	}
	if rs["role"] != "assistant" || rs["status"] != "incomplete" {
		t.Fatalf("snapshot reasoning item = %#v, want role assistant and the finish-derived status", rs)
	}
	part := rs["content"].([]any)[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "pondering" {
		t.Fatalf("snapshot reasoning content = %#v", part)
	}
	if _, ok := part["annotations"].([]any); !ok {
		t.Fatalf("snapshot reasoning content part missing annotations: %#v", part)
	}
	if msg := out[1].(map[string]any); msg["status"] != "incomplete" {
		t.Fatalf("snapshot message item = %#v, want the finish-derived incomplete status", msg)
	}
}

// One call addressed by its chat slot index on some fragments and by its call
// id on others is ONE call: both spellings resolve through the same item (the
// reference's own index-to-id map does), so the arguments concatenate instead
// of splitting across two function_call items that each hold half a payload.
func TestPipeChatToResponsesMixedKeyFragmentsStayOneCall(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"fn","arguments":""}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"{\"a\":"}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"1}"}}]}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var args strings.Builder
	addedCount := 0
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			switch obj["type"] {
			case "response.output_item.added":
				if item, _ := obj["item"].(map[string]any); item["type"] == "function_call" {
					addedCount++
				}
			case "response.function_call_arguments.delta":
				args.WriteString(obj["delta"].(string))
			}
		}
	}
	if addedCount != 1 {
		t.Fatalf("function_call added count = %d, want 1 (index- and id-addressed fragments are the same call):\n%s", addedCount, dst.String())
	}
	if args.String() != `{"a":1}` {
		t.Fatalf("arguments = %q, want the fragments concatenated", args.String())
	}
	if !strings.Contains(dst.String(), `"arguments":"{\"a\":1}"`) {
		t.Fatalf("done/completed arguments missing the whole payload:\n%s", dst.String())
	}
}

// The responses->chat fold types every tool_calls entry — argument deltas
// included — and closes with the reference's bare empty-content terminal
// delta; the id and model pass through from the source response rather than
// being re-minted per stream.
func TestResponsesToChatArgsDeltaTypedAndTerminalDeltaBare(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"resp_9","model":"m","created_at":7}}`,
		``,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"fn","arguments":""}}`,
		``,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{}"}`,
		``,
		`data: {"type":"response.completed","response":{"id":"resp_9","model":"m","created_at":7,"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var terminalDelta map[string]any
	sawArgsDelta := false
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok || strings.TrimSpace(data) == "[DONE]" {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			choice := obj["choices"].([]any)[0].(map[string]any)
			delta, _ := choice["delta"].(map[string]any)
			if choice["finish_reason"] != nil {
				terminalDelta = delta
			}
			if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
				entry := tcs[0].(map[string]any)
				if entry["function"] != nil && entry["id"] == nil {
					if entry["type"] != "function" {
						t.Fatalf("args delta entry = %#v, want type function on every entry", entry)
					}
					sawArgsDelta = true
				}
			}
		}
	}
	if !sawArgsDelta {
		t.Fatalf("no argument delta seen:\n%s", dst.String())
	}
	if terminalDelta == nil || len(terminalDelta) != 1 || terminalDelta["content"] != "" {
		t.Fatalf("terminal delta = %#v, want exactly {content:%q}", terminalDelta, "")
	}
	if !strings.Contains(dst.String(), `"id":"resp_9"`) || !strings.Contains(dst.String(), `"model":"m"`) {
		t.Fatalf("id/model should pass through from the source response:\n%s", dst.String())
	}
}

// The opening tool-call chunk's id is the item's call_id, falling back to the
// item id when the event carries no call_id — an empty id would break the
// pairing the tool result answers to. (The reference's further call_\d+
// guard is a vendor patch this library's no-vendor-knowledge contract keeps
// out.)
func TestResponsesToChatToolIdFallsBackToItemID(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m","created_at":5}}`,
		``,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_only","name":"fn","arguments":""}}`,
		``,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"gn","arguments":""}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var ids []any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			delta, _ := obj["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
			if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
				ids = append(ids, tcs[0].(map[string]any)["id"])
			}
		}
	}
	if len(ids) != 2 {
		t.Fatalf("tool ids = %#v, want both added events folded:\n%s", ids, dst.String())
	}
	if ids[0] != "fc_only" {
		t.Fatalf("id without call_id = %#v, want the item id fallback", ids[0])
	}
	if ids[1] != "call_2" {
		t.Fatalf("id with call_id = %#v, want the call_id", ids[1])
	}
}

// A multi-segment reasoning summary mirrors its summary_index into the chunk's
// choice index, the way the reference folds those events; an index-less delta
// stays at the single choice slot 0.
func TestResponsesToChatReasoningSummaryIndexMirrorsChoiceIndex(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m","created_at":5}}`,
		``,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":2,"delta":"part two"}`,
		``,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"no index"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Chat, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var indexes []any
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			choice := obj["choices"].([]any)[0].(map[string]any)
			delta, _ := choice["delta"].(map[string]any)
			if delta != nil && delta["reasoning_content"] != nil {
				indexes = append(indexes, choice["index"])
			}
		}
	}
	if len(indexes) != 2 {
		t.Fatalf("reasoning chunks = %d, want both:\n%s", len(indexes), dst.String())
	}
	if indexes[0] != float64(2) {
		t.Fatalf("summary_index=2 chunk choice index = %#v, want 2", indexes[0])
	}
	if indexes[1] != float64(0) {
		t.Fatalf("index-less chunk choice index = %#v, want 0", indexes[1])
	}
}

// The reasoning item's opening frame declares its summary slot null — the
// reference's own added shape — filling it only on the closing frames.
func TestPipeChatToResponsesReasoningAddedSummaryIsNull(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"reasoning_content":"pondering"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["type"] == "response.output_item.added" {
				if item, _ := obj["item"].(map[string]any); item["type"] == "reasoning" {
					if v, has := item["summary"]; !has || v != nil {
						t.Fatalf("reasoning added summary = %#v, want null", item["summary"])
					}
					return
				}
			}
		}
	}
	t.Fatalf("no reasoning added frame:\n%s", dst.String())
}

// The terminal payload carries the empty text object, as the non-streaming
// body does and the reference's completed snapshot does.
func TestPipeChatToResponsesTerminalCarriesTextObject(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","created":7,"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	for _, chunk := range strings.Split(dst.String(), "\n\n") {
		for _, line := range strings.Split(chunk, "\n") {
			data, ok := strings.CutPrefix(line, "data:")
			if !ok {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &obj); err != nil {
				continue
			}
			if obj["type"] == "response.completed" {
				resp, _ := obj["response"].(map[string]any)
				if text, ok := resp["text"].(map[string]any); !ok || len(text) != 0 {
					t.Fatalf("terminal text = %#v, want the empty object", resp["text"])
				}
				return
			}
		}
	}
	t.Fatalf("no terminal:\n%s", dst.String())
}
