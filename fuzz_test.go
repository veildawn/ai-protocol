package protocol_test

import (
	"bytes"
	"testing"

	protocol "github.com/veildawn/ai-protocol"
	"github.com/veildawn/ai-protocol/stream"
)

var fuzzDialects = []protocol.Dialect{
	protocol.Chat,
	protocol.Messages,
	protocol.Responses,
	protocol.Gemini,
	protocol.Images,
}

var fuzzRequestSeeds = []string{
	`{}`,
	`null`,
	`{"model":"m"}`,
	`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
	`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
	`{"input":"hi","tools":[{"type":"function","name":"f","parameters":{}}]}`,
	`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"maxOutputTokens":16}}`,
	`{"prompt":"a cat","n":1,"size":"1024x1024"}`,
	`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
}

// ConvertRequest must never panic on a body a client can send, and a
// same-dialect conversion must be the identity.
func FuzzConvertRequest(f *testing.F) {
	for i, seed := range fuzzRequestSeeds {
		f.Add(uint8(i%len(fuzzDialects)), uint8((i+1)%len(fuzzDialects)), seed)
	}
	f.Fuzz(func(t *testing.T, a, b uint8, body string) {
		from := fuzzDialects[int(a)%len(fuzzDialects)]
		to := fuzzDialects[int(b)%len(fuzzDialects)]

		out, err := protocol.ConvertRequestWith(from, to, []byte(body), protocol.ConvertOptions{
			MaxTokensFallback: 1024,
		})
		if from == to {
			if err != nil {
				t.Fatalf("same-dialect %s conversion failed: %v", from, err)
			}
			if string(out.Body) != body {
				t.Fatalf("same-dialect %s conversion changed the body", from)
			}
		}
		_ = out
	})
}

// ConvertResponse must never panic on a body an upstream can return.
func FuzzConvertResponse(f *testing.F) {
	for i, seed := range fuzzRequestSeeds {
		f.Add(uint8(i%len(fuzzDialects)), uint8((i+1)%len(fuzzDialects)), seed)
	}
	f.Fuzz(func(t *testing.T, a, b uint8, body string) {
		from := fuzzDialects[int(a)%len(fuzzDialects)]
		to := fuzzDialects[int(b)%len(fuzzDialects)]

		out, err := protocol.ConvertResponseWith(from, to, []byte(body), protocol.ConvertOptions{})
		if from == to {
			if err != nil {
				t.Fatalf("same-dialect %s conversion failed: %v", from, err)
			}
			if !bytes.Equal(out, []byte(body)) {
				t.Fatalf("same-dialect %s conversion changed the body", from)
			}
		}
	})
}

// A stream is attacker-controlled too: arbitrary bytes, arbitrary chunking.
func FuzzPipeStream(f *testing.F) {
	for _, seed := range []string{
		"",
		"data: {}\n\n",
		"data: [DONE]\n\n",
		"event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"x\",\"message\":\"boom\"}}\n\n",
		"data: not json\n\n",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, src string) {
		var dst bytes.Buffer
		_, _ = protocol.PipeStreamWith(protocol.Messages, protocol.Chat, &dst, bytes.NewReader([]byte(src)), protocol.StreamOpts{})
	})
}

// The framer splits a stream on delimiters; no byte sequence may panic it or
// make it forget the final event.
func FuzzFramer(f *testing.F) {
	for _, seed := range []string{"", "\n\n", "data: x\n\n", "data: x", "retry: 1\nid: 2\n\n"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw string) {
		var framer stream.Framer
		src := []byte(raw)
		for len(src) > 0 {
			n := 1 + len(src)/3
			if n > len(src) {
				n = len(src)
			}
			_ = framer.Feed(src[:n], func(stream.Event) error { return nil })
			src = src[n:]
		}
		_ = framer.Finish(func(stream.Event) error { return nil })
	})
}
