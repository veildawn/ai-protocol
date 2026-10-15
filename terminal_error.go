package protocol

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/veildawn/ai-protocol/stream"
)

// WriteTerminalError writes one terminal error frame onto a stream, in
// dialect d's own shape, so a host that aborts a delivery mid-stream can end
// it with a frame the client's parser can place instead of a connection that
// simply goes quiet.
//
// The mold is this library's because only it knows each wire's framing rules;
// the verdict is the host's, which is why the folds emit nothing of their own
// for a failed source — the frame the client has to see is the host's error
// frame for the failed delivery, and this function is that frame. errType
// names the failure on the wires that carry one (the host's own vocabulary),
// and message should already be bounded: nothing here truncates, because a
// bound is the host's policy.
//
// The three shapes, and the rules that fix them:
//
//   - Gemini has no [DONE]: the sentinel is chat framing, and leaking another
//     protocol's close into a generateContent stream is a frame the client
//     cannot parse. Google's own stream errors are error-object data frames,
//     so that is the shape minted here, with the status fixed at UNAVAILABLE
//     (503, Service Unavailable) — a mid-flight abort is the host's verdict,
//     not an upstream code it could relay.
//   - Responses names every frame (event: ...) and has no [DONE] either; an
//     unnamed data line is a frame its parser cannot place, and the sentinel
//     is the same foreign framing as above. Messages shares the named-error
//     shape, which is Anthropic's own.
//   - Chat closes with the bare [DONE] sentinel after the error payload.
func WriteTerminalError(w io.Writer, d Dialect, errType, message string) error {
	norm, ok := Normalize(string(d))
	if !ok {
		return fmt.Errorf("dialect %q has no terminal error frame", d)
	}
	switch norm {
	case Gemini:
		payload, err := json.Marshal(map[string]any{
			"error": map[string]any{
				"code":    503,
				"message": message,
				"status":  "UNAVAILABLE",
			},
		})
		if err != nil {
			return err
		}
		_, err = stream.WriteEvent(w, stream.Event{Data: string(payload)}, nil)
		return err
	case Messages, Responses:
		payload, err := errorPayload(errType, message)
		if err != nil {
			return err
		}
		_, err = stream.WriteEvent(w, stream.Event{Event: "error", Data: string(payload)}, nil)
		return err
	case Chat:
		payload, err := errorPayload(errType, message)
		if err != nil {
			return err
		}
		if _, err = stream.WriteEvent(w, stream.Event{Data: string(payload)}, nil); err != nil {
			return err
		}
		_, err = stream.WriteDone(w, nil)
		return err
	default:
		return fmt.Errorf("dialect %q has no terminal error frame", d)
	}
}

// errorPayload is the named-error envelope the Messages, Responses and Chat
// shapes share: {"type":"error","error":{"type","message"}}. Map-marshal's
// key order is part of the mold — hosts' clients have been reading these
// bytes since before the shape moved here.
func errorPayload(errType, message string) ([]byte, error) {
	return json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": errType, "message": message},
	})
}
