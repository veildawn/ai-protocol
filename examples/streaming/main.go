// Command streaming pipes a Messages SSE stream into the chat dialect and
// reports how the source ended. The bytes on stdout are the client's wire.
package main

import (
	"fmt"
	"os"
	"strings"

	protocol "github.com/veildawn/ai-protocol"
)

// A Messages turn: start, one text block, the stop reason, then the close.
const turn = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"example-model","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

func main() {
	result, err := protocol.PipeStreamWith(protocol.Messages, protocol.Chat, os.Stdout, strings.NewReader(turn), protocol.StreamOpts{
		Flush: func() {}, // a real host flushes its http.ResponseWriter here
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "pipe:", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "\nbytes=%d terminal=%t truncated=%t\n", result.Written, result.Terminal, result.Truncated)
	if !result.Terminal {
		// The source never closed the turn. A host decides what to do about it;
		// the fold deliberately left the turn open.
		os.Exit(1)
	}
}
