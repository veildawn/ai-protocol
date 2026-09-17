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
