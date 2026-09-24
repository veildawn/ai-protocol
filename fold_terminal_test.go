package protocol

import (
	"strings"
	"testing"
)

// A fold may close a stream for the client only when the SOURCE closed it too.
// Responses and Messages sources have exactly one legal ending, so a stream that
// stops without it was cut; a fold that invents the target's terminal for such a
// source tells the client it holds a finished answer while the host records a
// failed delivery for the same turn. That disagreement is what these tests pin,
// in the shape it was found: a chat client rendered a completed answer, and the
// gateway logged a 502 and cooled the account for it.
const (
	responsesOpen     = `data: {"type":"response.created","response":{"id":"resp_1","model":"m"}}` + "\n\n" + `data: {"type":"response.output_text.delta","delta":"hello"}` + "\n\n"
	responsesComplete = `data: {"type":"response.completed","response":{"id":"resp_1","model":"m","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}` + "\n\n"
	messagesOpen      = `event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}` + "\n\n" + `event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n"
	messagesDelta     = `event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}` + "\n\n"
	messagesStop      = `event: message_stop` + "\n" + `data: {"type":"message_stop"}` + "\n\n"
	chatOpen          = `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hello"}}]}` + "\n\n"
	relayDone         = "data: [DONE]\n\n"
)

func pipeFor(t *testing.T, from, to Dialect, src string) (string, StreamResult) {
	t.Helper()
	var dst strings.Builder
	res, err := PipeStreamWith(from, to, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatalf("PipeStreamWith(%s→%s): %v", from, to, err)
	}
	return dst.String(), res
}

func assertNoTerminal(t *testing.T, out string) {
	t.Helper()
	for _, marker := range []string{`"finish_reason":"stop"`, "[DONE]", "message_stop", "response.completed"} {
		if strings.Contains(out, marker) {
			t.Fatalf("truncated source was closed for the client with %q:\n%s", marker, out)
		}
	}
}

// The happy path, pinned next to the truncation cases so "no terminal" can
// never be conflated with "no terminal when the source had one": a source that
// closes properly still yields the client's finish frame, with the usage the
// host bills from.
func TestTypedSourcesStillCloseOnTheirOwnTerminal(t *testing.T) {
	out, res := pipeFor(t, Responses, Chat, responsesOpen+responsesComplete)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("responses→chat result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("responses→chat lost its terminal:\n%s", out)
	}
	if !strings.Contains(out, `"usage"`) {
		t.Fatalf("responses→chat lost the usage the finish frame carries:\n%s", out)
	}

	out, res = pipeFor(t, Messages, Chat, messagesOpen+messagesDelta+messagesStop)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("messages→chat result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("messages→chat lost its terminal:\n%s", out)
	}
}

func TestResponsesToChatTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Responses, Chat, responsesOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"content":"hello"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestResponsesToChatIncompleteIsLength(t *testing.T) {
	src := responsesOpen + `data: {"type":"response.incomplete","response":{"id":"resp_1","model":"m","status":"incomplete","usage":{"input_tokens":3,"output_tokens":4}}}` + "\n\n"
	out, res := pipeFor(t, Responses, Chat, src)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"length"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("ceiling cut not reported as length:\n%s", out)
	}
}

func TestResponsesToChatRelayDoneStillCloses(t *testing.T) {
	out, res := pipeFor(t, Responses, Chat, responsesOpen+relayDone)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("relay close not folded into a chat terminal:\n%s", out)
	}
}

func TestResponsesToChatInBandFailureIsNotACompletion(t *testing.T) {
	src := responsesOpen + `data: {"type":"response.failed","response":{"error":{"code":"server_error","message":"boom"}}}` + "\n\n" + relayDone
	out, res := pipeFor(t, Responses, Chat, src)
	if res.InBandErr == nil || res.InBandErr.Error() != "boom" {
		t.Fatalf("in-band error lost: %+v", res)
	}
	assertNoTerminal(t, out)
}

func TestMessagesToChatTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Messages, Chat, messagesOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"content":"hello"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestMessagesToChatRelayDoneStillCloses(t *testing.T) {
	out, res := pipeFor(t, Messages, Chat, messagesOpen+relayDone)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("relay close not folded into a chat terminal:\n%s", out)
	}
}

func TestMessagesToResponsesTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Messages, Responses, messagesOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
}

func TestResponsesToMessagesTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Responses, Messages, responsesOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
}

// A Chat source's own ending is optional — several OpenAI-compatible vendors
// close on a usage chunk instead of [DONE] — so its folds keep closing the turn
// for the client. The host still hears Truncated and owns the verdict; this test
// exists so the asymmetry with the typed sources above is a decision, not an
// accident.
func TestChatSourceFoldsStillCloseAnOpenEndedStream(t *testing.T) {
	out, res := pipeFor(t, Chat, Responses, chatOpen)
	if !res.Truncated || res.Terminal {
		t.Fatalf("chat→responses result=%+v", res)
	}
	if !strings.Contains(out, "response.completed") {
		t.Fatalf("chat→responses lost its synthesized terminal:\n%s", out)
	}

	out, res = pipeFor(t, Chat, Messages, chatOpen)
	if !res.Truncated || res.Terminal {
		t.Fatalf("chat→messages result=%+v", res)
	}
	if !strings.Contains(out, "message_stop") {
		t.Fatalf("chat→messages lost its synthesized terminal:\n%s", out)
	}
}
