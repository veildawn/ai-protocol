package protocol

import (
	"bytes"
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
