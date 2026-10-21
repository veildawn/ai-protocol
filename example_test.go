package protocol_test

import (
	"errors"
	"fmt"
	"strings"

	protocol "github.com/veildawn/ai-protocol"
)

// The smallest request conversion: a chat-completions body folded into the
// messages wire. MaxTokensFallback supplies the target's required max_tokens,
// which this source body does not carry.
func ExampleConvertRequest() {
	body := []byte(`{"model":"example-model","messages":[{"role":"user","content":"hello"}]}`)

	converted, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{
		MaxTokensFallback: 8192,
	})
	if err != nil {
		fmt.Println("convert:", err)
		return
	}
	fmt.Println("model:", converted.Model)
	fmt.Println("has max_tokens:", strings.Contains(string(converted.Body), "max_tokens"))
}

// A response conversion runs the other direction, so the client receives the
// shape it asked for.
func ExampleConvertResponse() {
	upstream := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"example-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)

	out, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, upstream)
	if err != nil {
		fmt.Println("convert:", err)
		return
	}
	fmt.Println(string(out))
}

// A conversion that cannot carry a parameter reports it as a typed error
// instead of silently dropping it.
func ExampleUnsupportedParamError() {
	body := []byte(`{"model":"example-model","logprobs":true,"messages":[{"role":"user","content":"hi"}]}`)

	_, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{
		MaxTokensFallback: 8192,
	})
	var unsupported protocol.UnsupportedParamError
	if errors.As(err, &unsupported) {
		fmt.Println("cannot carry:", unsupported.Param)
	}
}

// Losses are inspected before a conversion runs, so the caller can decide
// whether the omission is acceptable.
func ExampleBodyLosses() {
	body := []byte(`{"model":"example-model","logprobs":true,"messages":[{"role":"user","content":"hi"}]}`)

	losses := protocol.BodyLosses(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{})
	fmt.Println("logprobs lost:", losses[protocol.LossLogprobs])
}
