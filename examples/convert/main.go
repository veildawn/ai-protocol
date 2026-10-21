// Command convert shows the smallest useful conversion: a request body and a
// response body between two public dialects, with no network and no I/O
// beyond the printed output.
package main

import (
	"fmt"
	"os"

	protocol "github.com/veildawn/ai-protocol"
)

func main() {
	request := []byte(`{"model":"example-model","messages":[{"role":"user","content":"hello"}]}`)

	converted, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, request, protocol.ConvertOptions{
		// The messages wire requires max_tokens. This source body does not
		// carry one, so the caller supplies the model's ceiling.
		MaxTokensFallback: 8192,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "request:", err)
		os.Exit(1)
	}
	fmt.Printf("model: %s\n", converted.Model)
	fmt.Printf("messages request: %s\n\n", converted.Body)

	response := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"example-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)

	out, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, response)
	if err != nil {
		fmt.Fprintln(os.Stderr, "response:", err)
		os.Exit(1)
	}
	fmt.Printf("chat response: %s\n", out)
}
