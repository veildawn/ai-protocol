package protocol_test

import (
	"io"
	"strings"
	"testing"

	protocol "github.com/veildawn/ai-protocol"
	"github.com/veildawn/ai-protocol/stream"
)

var benchChatRequest = []byte(`{
	"model":"example-model",
	"max_tokens":256,
	"messages":[
		{"role":"system","content":"You are terse."},
		{"role":"user","content":"Summarize the change."}
	],
	"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
	"temperature":0.2
}`)

var benchMessagesResponse = []byte(`{
	"id":"msg_1","type":"message","role":"assistant","model":"example-model",
	"content":[{"type":"text","text":"The change adds a parser."}],
	"stop_reason":"end_turn",
	"usage":{"input_tokens":12,"output_tokens":9}
}`)

const benchSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"example-model","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

func BenchmarkConvertRequestChatToMessages(b *testing.B) {
	opts := protocol.ConvertOptions{MaxTokensFallback: 8192}
	b.ReportAllocs()
	b.SetBytes(int64(len(benchChatRequest)))
	for i := 0; i < b.N; i++ {
		if _, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, benchChatRequest, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvertResponseMessagesToChat(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(int64(len(benchMessagesResponse)))
	for i := 0; i < b.N; i++ {
		if _, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, benchMessagesResponse); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBodyLosses(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = protocol.BodyLosses(protocol.Chat, protocol.Messages, benchChatRequest, protocol.ConvertOptions{})
	}
}

func BenchmarkPipeStreamMessagesToChat(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(int64(len(benchSSE)))
	for i := 0; i < b.N; i++ {
		if _, err := protocol.PipeStreamWith(protocol.Messages, protocol.Chat, io.Discard, strings.NewReader(benchSSE), protocol.StreamOpts{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFramer(b *testing.B) {
	raw := []byte(benchSSE)
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	for i := 0; i < b.N; i++ {
		var framer stream.Framer
		if err := framer.Feed(raw, func(stream.Event) error { return nil }); err != nil {
			b.Fatal(err)
		}
		if err := framer.Finish(func(stream.Event) error { return nil }); err != nil {
			b.Fatal(err)
		}
	}
}
