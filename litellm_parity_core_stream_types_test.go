package protocol

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/stream"
)

// This file pins the core/stream-types folds a LiteLLM v1.100.1 audit found
// diverging from the reference: losses the API reported for features both
// sides carry, a file part that vanished without even a loss entry, and an
// in-band error frame the stream observer lost to its own buffer handling.

// An in-band error that arrives mid-stream, with events after it and a read
// boundary inside the event after that, is still the metadata the contract
// promises: InBandErr set, the pipe returning nil, and the turn not counted as
// truncated. The observer's tail compaction used to overwrite the very bytes
// it was about to scan, so the error frame vanished (and a corrupted remnant
// could read as a phantom terminal). The unterminated tail is deliberately
// longer than the events before the error: that is the shape whose compaction
// clobbers the error frame itself.
func TestPipeStreamWithMidStreamInBandErrorStaysMetadata(t *testing.T) {
	longTail := strings.Repeat("a", 400)
	chatSrc := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"error":{"message":"boom"}}`,
		``,
		`data: [DONE]`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"` + longTail,
	}, "\n")
	messagesSrc := strings.Join([]string{
		`event: message_start`, `data: {"type":"message_start","message":{"id":"m1","model":"m"}}`, ``,
		`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`, ``,
		`event: error`, `data: {"type":"error","error":{"message":"boom"}}`, ``,
		`event: message_stop`, `data: {"type":"message_stop"}`, ``,
	}, "\n") + "\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"" + longTail

	for _, tc := range []struct {
		name     string
		from, to Dialect
		src      string
	}{
		{"chat", Chat, Responses, chatSrc},
		{"messages", Messages, Chat, messagesSrc},
	} {
		for _, size := range []int{4096, 17} {
			var dst bytes.Buffer
			result, err := PipeStreamWith(tc.from, tc.to, &dst, &chunkedReader{data: []byte(tc.src), size: size}, StreamOpts{})
			if err != nil {
				t.Fatalf("%s size=%d: in-band error must be metadata, got pipe error %v", tc.name, size, err)
			}
			if result.InBandErr == nil || result.InBandErr.Error() != "boom" {
				t.Errorf("%s size=%d: InBandErr=%v, want boom", tc.name, size, result.InBandErr)
			}
			if !result.Terminal || result.Truncated {
				t.Errorf("%s size=%d: result=%+v, want a terminal failure and no truncation", tc.name, size, result)
			}
			if !strings.Contains(dst.String(), "hi") {
				t.Errorf("%s size=%d: content before the error was not folded:\n%s", tc.name, size, dst.String())
			}
		}
	}
}

// json_schema survives the chat->messages fold: litellm keeps it as
// output_format, and so does this library; reporting structured_output as a
// loss pinned a feature both codecs carry. json_object is different: it names
// no schema, and litellm v1.100.1 sends nothing for it on both its
// structured-output paths (transformation.py:1287-1301 — no output_format, no
// simulating tool), so the fold carries no output_format for it either.
func TestChatToMessagesStructuredOutputIsNotALoss(t *testing.T) {
	if StaticLosses(Chat, Messages)[LossStructuredOutput] {
		t.Fatal("structured output cannot be lost on chat->messages")
	}
	rf := `{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object"}}}`
	body := []byte(`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"response_format":` + rf + `}`)
	if got := BodyLosses(Chat, Messages, body, ConvertOptions{}); got[LossStructuredOutput] {
		t.Fatalf("response_format json_schema reported as lost: %v", got)
	}
	out, err := ConvertRequest(Chat, Messages, body)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["output_format"]; !ok {
		t.Fatalf("response_format json_schema produced no output_format: %s", out.Body)
	}
}

// The responses->messages fold keeps json_schema (both sides fold it to
// output_format), so that body reports no loss; json_object has no branch in
// the fold and is genuinely dropped, so it stays a loss.
func TestResponsesToMessagesStructuredOutputLossIsJsonObjectOnly(t *testing.T) {
	schema := []byte(`{"model":"m","max_output_tokens":10,"input":"x","text":{"format":{"type":"json_schema","name":"n","schema":{"type":"object"}}}}`)
	if got := BodyLosses(Responses, Messages, schema, ConvertOptions{}); got[LossStructuredOutput] {
		t.Fatalf("json_schema reported as lost: %v", got)
	}
	out, err := ConvertRequest(Responses, Messages, schema)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["output_format"]; !ok {
		t.Fatalf("json_schema produced no output_format: %s", out.Body)
	}

	object := []byte(`{"model":"m","max_output_tokens":10,"input":"x","text":{"format":{"type":"json_object"}}}`)
	if got := BodyLosses(Responses, Messages, object, ConvertOptions{}); !got[LossStructuredOutput] {
		t.Fatalf("json_object is dropped by this fold and must stay a loss: %v", got)
	}
}

// The responses->messages fold drops every input_file part, whether it rode a
// file_id or inline file_data. The loss API used to require a file_id, so an
// inline-only part vanished without even a loss entry.
func TestResponsesInputFileDataOnlyIsReportedAsFileLoss(t *testing.T) {
	body := []byte(`{"model":"m","max_output_tokens":10,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"x"},{"type":"input_file","file_data":"data:application/pdf;base64,AA"}]}]}`)
	got := BodyLosses(Responses, Messages, body, ConvertOptions{})
	if !got[LossFileReference] {
		t.Fatalf("losses=%v, want file_reference for the dropped input_file", got)
	}
	// The loss is real: the fold emits no document block for the part.
	out, err := ConvertRequest(Responses, Messages, body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out.Body), "document") {
		t.Fatalf("input_file unexpectedly folded: %s", out.Body)
	}
}

// litellm folds empty or missing tool-call arguments to input:{} and so does
// this library; a tool_arguments loss for input both sides carry was a phantom.
func TestEmptyToolArgumentsAreNotALoss(t *testing.T) {
	bodies := map[string][]byte{
		"chat empty":   []byte(`{"model":"m","max_tokens":10,"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":""}}]},{"role":"tool","tool_call_id":"c","content":"ok"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}`),
		"chat missing": []byte(`{"model":"m","max_tokens":10,"messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f"}}]},{"role":"tool","tool_call_id":"c","content":"ok"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}`),
	}
	for name, body := range bodies {
		if got := BodyLosses(Chat, Messages, body, ConvertOptions{}); got[LossToolArguments] {
			t.Fatalf("%s: arguments reported as lost: %v", name, got)
		}
		out, err := ConvertRequest(Chat, Messages, body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		block := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		if block["type"] != "tool_use" || !reflect.DeepEqual(block["input"], map[string]any{}) {
			t.Fatalf("%s: tool_use block = %#v, want input {}", name, block)
		}
	}

	responses := []byte(`{"model":"m","max_output_tokens":10,"input":[{"type":"function_call","call_id":"c","name":"f","arguments":""}]}`)
	if got := BodyLosses(Responses, Messages, responses, ConvertOptions{}); got[LossToolArguments] {
		t.Fatalf("responses empty arguments reported as lost: %v", got)
	}
	out, err := ConvertRequest(Responses, Messages, responses)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	block := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_use" || !reflect.DeepEqual(block["input"], map[string]any{}) {
		t.Fatalf("tool_use block = %#v, want input {}", block)
	}
}

// A redacted thinking block rides its whole opaque payload in `data` — the
// fold delivers it and the reference hands it to the client as thinking
// content — so the fact layer must report delivered reasoning, not an empty
// shell. An empty redacted block stays a prelude.
func TestRedactedThinkingStartIsReasoning(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "content_block_start",
		Data:  `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"RW5jcnlwdGVk"}}`,
	})
	if facts.Kind != stream.EventReasoning {
		t.Fatalf("kind=%v, want EventReasoning for a redacted payload", facts.Kind)
	}
	empty := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "content_block_start",
		Data:  `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":""}}`,
	})
	if empty.Kind != stream.EventPrelude {
		t.Fatalf("empty redacted block kind=%v, want EventPrelude", empty.Kind)
	}
}

// A Chat chunk closing with the literal finish_reason "error" is the wire's
// own failure close: the fact layer reports a failure, and a settlement built
// on the facts records InBandErr instead of booking the turn as a clean stop.
func TestChatFinishReasonErrorIsAnInBandFailure(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Chat, stream.Event{
		Data: `{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"error"}]}`,
	})
	if facts.Kind != stream.EventFailure || !facts.Terminal {
		t.Fatalf("facts=%+v, want a terminal EventFailure", facts)
	}
	// finish_reason "stop" stays a logical completion, not a wire terminal.
	stop := stream.ClassifyEvent(stream.Chat, stream.Event{
		Data: `{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	})
	if stop.Kind != stream.EventComplete || stop.Terminal {
		t.Fatalf("facts=%+v, want a non-terminal EventComplete", stop)
	}

	src := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"error"}]}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Responses, &dst, &chunkedReader{data: []byte(src), size: 4096}, StreamOpts{})
	if err != nil {
		t.Fatalf("in-band failure must be metadata, got pipe error %v", err)
	}
	if result.InBandErr == nil {
		t.Fatal("finish_reason error settled as a success")
	}
	if !result.Terminal || result.Truncated {
		t.Fatalf("result=%+v, want a terminal failure and no truncation", result)
	}
}

// A usage-only chunk that names no choices key at all is the closing shape
// several OpenAI-compatible backends use. The reference parses it tolerantly
// (chunk.get("choices", [])), so the usage evidence survives; a choices value
// that is present but unparseable is still an unknown event.
func TestUsageOnlyChunkWithoutChoicesKeepsUsage(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Chat, stream.Event{
		Data: `{"id":"c1","usage":{"prompt_tokens":42,"completion_tokens":7}}`,
	})
	if facts.Kind != stream.EventPrelude || !facts.HasUsage {
		t.Fatalf("facts=%+v, want a prelude carrying usage", facts)
	}
	malformed := stream.ClassifyEvent(stream.Chat, stream.Event{
		Data: `{"id":"c1","choices":"not-a-list"}`,
	})
	if malformed.Kind != stream.EventUnknown {
		t.Fatalf("facts=%+v, want EventUnknown for unparseable choices", malformed)
	}
}

// Anthropic reports usage twice on a Messages stream — inside message_start's
// message object and again on message_delta — and HasUsage states the event
// carries it, either time.
func TestMessagesUsageEventsCarryHasUsage(t *testing.T) {
	start := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "message_start",
		Data:  `{"type":"message_start","message":{"id":"m1","usage":{"input_tokens":10,"output_tokens":1}}}`,
	})
	if start.Kind != stream.EventPrelude || !start.HasUsage {
		t.Fatalf("facts=%+v, want message_start to carry usage", start)
	}
	delta := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "message_delta",
		Data:  `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
	})
	if delta.Kind != stream.EventComplete || !delta.HasUsage {
		t.Fatalf("facts=%+v, want a message_delta completion carrying usage", delta)
	}
	plain := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "message_delta",
		Data:  `{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
	})
	if plain.HasUsage {
		t.Fatalf("facts=%+v, usage invented for a chunk without one", plain)
	}
}

// A citations delta is search-attribution metadata for text already delivered:
// the reference ships the citation in provider-specific fields, and no fold
// here forwards it as client content, so the fact layer must not claim the
// not-switch-upstream guarantee for it.
func TestCitationsDeltaIsMetadataNotContent(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Messages, stream.Event{
		Event: "content_block_delta",
		Data:  `{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://x","title":"x"}}}`,
	})
	if facts.Kind != stream.EventPrelude {
		t.Fatalf("facts=%+v, want EventPrelude for citation metadata", facts)
	}
}

// custom_tool_call_input is the freeform-input twin of function_call_arguments:
// both folds here and the reference deliver it as a tool-call increment, so a
// non-empty delta is actionable tool content rather than an unknown shape.
func TestCustomToolCallInputDeltaIsContent(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Responses, stream.Event{
		Event: "response.custom_tool_call_input.delta",
		Data:  `{"type":"response.custom_tool_call_input.delta","item_id":"i","output_index":0,"delta":"ls -la"}`,
	})
	if facts.Kind != stream.EventContent {
		t.Fatalf("facts=%+v, want EventContent for a tool-input increment", facts)
	}
	empty := stream.ClassifyEvent(stream.Responses, stream.Event{
		Event: "response.custom_tool_call_input.delta",
		Data:  `{"type":"response.custom_tool_call_input.delta","item_id":"i","output_index":0,"delta":""}`,
	})
	if empty.Kind != stream.EventPrelude {
		t.Fatalf("facts=%+v, want EventPrelude for an empty increment", empty)
	}

	// The fold agrees: piping the event to Chat carries the increment.
	src := strings.Join([]string{
		`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"c1","call_id":"c1","name":"shell","input":""}}`, ``,
		`event: response.custom_tool_call_input.delta`, `data: {"type":"response.custom_tool_call_input.delta","item_id":"c1","output_index":0,"delta":"ls -la"}`, ``,
		`event: response.completed`, `data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`, ``,
	}, "\n")
	var dst bytes.Buffer
	result, err := PipeStreamWith(Responses, Chat, &dst, &chunkedReader{data: []byte(src), size: 4096}, StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Terminal || result.Truncated {
		t.Fatalf("result=%+v", result)
	}
	if !strings.Contains(dst.String(), "ls -la") {
		t.Fatalf("fold dropped the custom tool input:\n%s", dst.String())
	}
}

// A custom_tool_call item is a tool call the way a function_call item is: the
// reference's fold and both folds here translate it, so its output_item events
// carry actionable tool content instead of reading as an unknown shape.
func TestCustomToolCallItemIsContent(t *testing.T) {
	facts := stream.ClassifyEvent(stream.Responses, stream.Event{
		Event: "response.output_item.added",
		Data:  `{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"c1","call_id":"c1","name":"shell","input":""}}`,
	})
	if facts.Kind != stream.EventContent {
		t.Fatalf("facts=%+v, want EventContent for a custom_tool_call item", facts)
	}
}

// Events no fold delivers — refusal deltas, mcp argument deltas, and the
// call-shaped items outside function_call/custom_tool_call — must not claim
// client-visible content: neither this library's folds nor the reference
// forward them, so the fact layer leaves them unknown for the host to retain.
func TestUndeliveredResponsesEventsAreNotContent(t *testing.T) {
	notContent := []struct {
		name string
		ev   stream.Event
	}{
		{"refusal delta", stream.Event{Event: "response.refusal.delta", Data: `{"type":"response.refusal.delta","item_id":"m","output_index":0,"delta":"cannot"}`}},
		{"mcp arguments delta", stream.Event{Event: "response.mcp_call_arguments.delta", Data: `{"type":"response.mcp_call_arguments.delta","item_id":"c","output_index":0,"delta":"{}"}`}},
		{"web_search item", stream.Event{Event: "response.output_item.added", Data: `{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws","name":"search"}}`}},
		{"computer item", stream.Event{Event: "response.output_item.added", Data: `{"type":"response.output_item.added","output_index":0,"item":{"type":"computer_call","id":"cc","action":{"type":"key"}}}`}},
		{"image_generation item", stream.Event{Event: "response.output_item.added", Data: `{"type":"response.output_item.added","output_index":0,"item":{"type":"image_generation_call","id":"ig","input":"a cat"}}`}},
		{"mcp item", stream.Event{Event: "response.output_item.added", Data: `{"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_call","id":"mc","name":"tool","arguments":"{}"}}`}},
	}
	for _, tc := range notContent {
		if facts := stream.ClassifyEvent(stream.Responses, tc.ev); facts.Kind == stream.EventContent {
			t.Errorf("%s: facts=%+v, must not claim delivered content", tc.name, facts)
		}
	}
	// The delivered twin stays content.
	fn := stream.ClassifyEvent(stream.Responses, stream.Event{
		Event: "response.output_item.added",
		Data:  `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"f","name":"f","arguments":"{}"}}`,
	})
	if fn.Kind != stream.EventContent {
		t.Fatalf("function_call item facts=%+v, want EventContent", fn)
	}
}

// SSE's grammar ends a line with \n, \r\n, or a lone \r, and the reference's
// decoder accepts all three. A lone-\r stream must frame and settle like any
// other — events dispatched, [DONE] terminal — including when a read boundary
// splits a pending \r from the \n that would complete its CRLF.
func TestLoneCRLineEndingsStillFrameAndSettle(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\r\r" +
		"data: [DONE]\r\r"
	for _, size := range []int{4096, 9, 1} {
		var dst bytes.Buffer
		result, err := PipeStreamWith(Chat, Responses, &dst, &chunkedReader{data: []byte(src), size: size}, StreamOpts{})
		if err != nil {
			t.Fatalf("size=%d: %v", size, err)
		}
		if !result.Terminal || result.Truncated {
			t.Errorf("size=%d: result=%+v, want the [DONE] sentinel to settle the stream", size, result)
		}
		if !strings.Contains(dst.String(), "hi") {
			t.Errorf("size=%d: content lost:\n%s", size, dst.String())
		}
	}

	// Direct Framer use, with the \r|\n boundary split across feeds: the
	// held \r ends its line either way, so the events are identical.
	var fr stream.Framer
	var got []string
	feed := func(chunk string) {
		if err := fr.Feed([]byte(chunk), func(ev stream.Event) error {
			got = append(got, ev.Data)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	feed("data: a\r")
	feed("\rdata: b\r\r")
	if err := fr.Finish(func(ev stream.Event) error {
		got = append(got, ev.Data)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("events=%q, want [a b]", got)
	}
}

// A Chat stream that closes on its usage chunk — the OpenRouter-style ending,
// no [DONE] after it — is a closed turn, not a truncation: the counts arrive
// when the turn is over, and the host must not book a failed delivery for it.
// A bare EOF with no closing frame at all is still a truncation.
func TestChatUsageCloseIsTerminalNotTruncation(t *testing.T) {
	for name, src := range map[string]string{
		"usage chunk with empty choices": strings.Join([]string{
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: {"id":"c1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}`,
			``,
		}, "\n"),
		"usage chunk without a choices key": strings.Join([]string{
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: {"id":"c1","usage":{"prompt_tokens":10,"completion_tokens":5}}`,
			``,
		}, "\n"),
	} {
		var dst bytes.Buffer
		result, err := PipeStreamWith(Chat, Messages, &dst, &chunkedReader{data: []byte(src), size: 4096}, StreamOpts{Model: "m"})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !result.Terminal || result.Truncated || result.InBandErr != nil {
			t.Errorf("%s: result=%+v, want a settled turn", name, result)
		}
	}

	// The truncation contract survives: EOF with no closing frame is a cut.
	bare := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Messages, &dst, strings.NewReader(bare), StreamOpts{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Terminal || !result.Truncated {
		t.Fatalf("result=%+v, want a bare EOF to stay truncated", result)
	}
}

// The SSE grammar ends lines with \n, \r\n, or a lone \r, and the reference's
// decoder splits on all three: a lone-\r stream must dispatch its events and
// settle, not merge into one unreadable blob.
func TestLoneCRLineEndingsStillDispatch(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\r\r" +
		"data: [DONE]\r\r"
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Responses, &dst, &chunkedReader{data: []byte(src), size: 4096}, StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Terminal || result.Truncated {
		t.Fatalf("result=%+v, want the lone-CR stream settled on [DONE]", result)
	}
	if !strings.Contains(dst.String(), "hi") {
		t.Fatalf("lone-CR content lost:\n%s", dst.String())
	}
}
