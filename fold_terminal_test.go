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

	geminiOpen    = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"index":0}],"responseId":"r1"}` + "\n\n"
	geminiStop    = `data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}` + "\n\n"
	geminiBlocked = `data: {"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}` + "\n\n"
	geminiError   = `data: {"error":{"code":429,"message":"quota exceeded"}}` + "\n\n"
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
	for _, marker := range []string{`"finish_reason":"stop"`, "[DONE]", "message_stop", "response.completed", "finishReason"} {
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

func TestResponsesToMessagesIncompleteCloses(t *testing.T) {
	src := responsesOpen + `data: {"type":"response.incomplete","response":{"id":"resp_1","model":"m","status":"incomplete","usage":{"input_tokens":3,"output_tokens":4}}}` + "\n\n"
	out, res := pipeFor(t, Responses, Messages, src)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"stop_reason":"max_tokens"`) || !strings.Contains(out, "message_stop") {
		t.Fatalf("ceiling cut not closed for the messages client:\n%s", out)
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

// A ceiling cut is a terminal, but it is NOT a completion: the Responses dialect
// spells it response.incomplete and requires incomplete_details.reason. Emitting
// response.completed with status "incomplete" and no reason is what left clients
// reporting a bare "Response incomplete without a provider reason" — unable to
// tell a max_output_tokens cut from a content filter, and reading the event name
// as proof the turn finished.
func TestChatToResponsesCeilingCutIsIncompleteWithReason(t *testing.T) {
	src := chatOpen +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":65536,"total_tokens":65546}}` + "\n\n" +
		relayDone
	out, res := pipeFor(t, Chat, Responses, src)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, "event: response.incomplete") {
		t.Fatalf("ceiling cut did not close as response.incomplete:\n%s", out)
	}
	if strings.Contains(out, "event: response.completed") {
		t.Fatalf("ceiling cut was reported as a clean completion:\n%s", out)
	}
	if !strings.Contains(out, `"incomplete_details":{"reason":"max_output_tokens"}`) {
		t.Fatalf("ceiling cut carried no reason:\n%s", out)
	}
}

// The reason is what distinguishes the two non-success endings, so a content
// filter must not be reported as a token ceiling.
func TestChatToResponsesContentFilterKeepsItsOwnReason(t *testing.T) {
	src := chatOpen +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n" +
		relayDone
	out, _ := pipeFor(t, Chat, Responses, src)
	if !strings.Contains(out, "event: response.incomplete") {
		t.Fatalf("content filter did not close as response.incomplete:\n%s", out)
	}
	if !strings.Contains(out, `"incomplete_details":{"reason":"content_filter"}`) {
		t.Fatalf("content filter was mislabelled:\n%s", out)
	}
}

// A clean finish must stay a clean completion, with no incomplete_details.
func TestChatToResponsesCleanFinishStaysCompleted(t *testing.T) {
	src := chatOpen +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n" +
		relayDone
	out, res := pipeFor(t, Chat, Responses, src)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, "event: response.completed") {
		t.Fatalf("clean finish did not close as response.completed:\n%s", out)
	}
	if strings.Contains(out, "response.incomplete") || strings.Contains(out, "incomplete_details") {
		t.Fatalf("clean finish was marked incomplete:\n%s", out)
	}
}

// The Messages dialect's ceiling cut is stop_reason max_tokens. On this fold
// the reference's chat→responses bridge closes EVERY turn with
// response.completed (litellm v1.100.1
// responses/litellm_completion_transformation/streaming_iterator.py:1165
// _emit_response_completed_event — it has no incomplete variant); a ceiling
// cut says so in the embedded response.status, and incomplete_details stays
// null because the builder only copies it off the chat completion
// (transformation.py:2233), which the Anthropic path never sets. The turn is
// still not a completion — the status the client reads must say incomplete.
func TestMessagesToResponsesCeilingCutIsIncompleteWithReason(t *testing.T) {
	src := messagesOpen +
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":65536}}` + "\n\n" +
		messagesStop
	out, res := pipeFor(t, Messages, Responses, src)
	if !res.Terminal || res.Truncated {
		t.Fatalf("result=%+v", res)
	}
	if strings.Contains(out, "event: response.incomplete") {
		t.Fatalf("ceiling cut closed as response.incomplete:\n%s", out)
	}
	if !strings.Contains(out, `event: response.completed`) {
		t.Fatalf("ceiling cut did not close with response.completed:\n%s", out)
	}
	if !strings.Contains(out, `"status":"incomplete"`) {
		t.Fatalf("terminal response.status did not say incomplete:\n%s", out)
	}
	if strings.Contains(out, "incomplete_details") {
		t.Fatalf("ceiling cut carried an incomplete_details the reference leaves null:\n%s", out)
	}
}

// The gemini wire's only close signals are a candidate's finishReason and a
// blocked promptFeedback — there is no sentinel and no separate terminal
// frame. A source that ends with neither was cut, and the fold must not
// invent one; a source that did state one is closed, terminal and all.
func TestGeminiToChatTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Gemini, Chat, geminiOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"content":"hello"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestGeminiToChatClosesOnItsFinishReasonFrame(t *testing.T) {
	out, res := pipeFor(t, Gemini, Chat, geminiOpen+geminiStop)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("finishReason frame was not folded into a chat terminal:\n%s", out)
	}
	// The usage rides the same wire frame as the finish and must survive the
	// fold onto the chat finish chunk.
	if !strings.Contains(out, `"usage"`) {
		t.Fatalf("terminal usage lost:\n%s", out)
	}
}

// A blocked prompt is a close the source stated itself: it ends the turn, but
// as a filter, not a success.
func TestGeminiToChatBlockedPromptClosesAsContentFilter(t *testing.T) {
	out, res := pipeFor(t, Gemini, Chat, geminiBlocked)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finish_reason":"content_filter"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("blocked prompt not folded into a filtered close:\n%s", out)
	}
}

func TestGeminiToMessagesTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Gemini, Messages, geminiOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"text_delta"`) {
		t.Fatalf("delivered text lost:\n%s", out)
	}
}

func TestGeminiToResponsesTruncationIsNotACompletion(t *testing.T) {
	// Two text frames: the first spends its delta on the item announcement
	// (the reference's announce-then-delta shape), so the second is the one
	// whose delta the client sees before the cut.
	src := geminiOpen + `data: {"candidates":[{"content":{"role":"model","parts":[{"text":" there"}]},"index":0}],"responseId":"r1"}` + "\n\n"
	out, res := pipeFor(t, Gemini, Responses, src)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"delta":" there"`) {
		t.Fatalf("delivered text lost:\n%s", out)
	}
}

// Folding INTO the gemini wire follows the same rule from the other side: the
// terminal frame (finishReason + usage on one frame) is written only when the
// source closed itself; a cut source leaves the target stream open.
func TestChatToGeminiTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Chat, Gemini, chatOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"text":"hello"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestMessagesToGeminiTruncationIsNotACompletion(t *testing.T) {
	out, res := pipeFor(t, Messages, Gemini, messagesOpen)
	if !res.Truncated || res.Terminal || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	assertNoTerminal(t, out)
	if !strings.Contains(out, `"text":"hello"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestChatToGeminiClosesOnSourceFinish(t *testing.T) {
	src := chatOpen +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}` + "\n\n" +
		relayDone
	out, res := pipeFor(t, Chat, Gemini, src)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finishReason":"STOP"`) || !strings.Contains(out, `"usageMetadata"`) {
		t.Fatalf("source close not folded into the wire's terminal frame:\n%s", out)
	}
}

// An in-band error frame ends the stream without an answer. It must surface as
// InBandErr for the host and close nothing for the client — the same contract
// the typed dialects' error frames already follow.
func TestGeminiInBandFailureIsNotACompletion(t *testing.T) {
	src := geminiOpen + geminiError
	out, res := pipeFor(t, Gemini, Chat, src)
	if res.InBandErr == nil || res.InBandErr.Error() != "quota exceeded" {
		t.Fatalf("in-band error lost: %+v", res)
	}
	assertNoTerminal(t, out)
}

// A Chat source that closes on the bare usage tail — the ending several
// vendors send instead of a finish_reason or [DONE] — has still closed: the
// observer settles it Terminal, so every Chat-source fold must close the turn
// for its client too, or the host books a delivered turn while the client's
// stream never ends. The gemini fold names the wire's natural stop and states
// the counts the tail carried.
func TestChatToGeminiUsageCloseStillCloses(t *testing.T) {
	src := chatOpen + `data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}` + "\n\n"
	out, res := pipeFor(t, Chat, Gemini, src)
	if !res.Terminal || res.Truncated || res.InBandErr != nil {
		t.Fatalf("result=%+v", res)
	}
	if !strings.Contains(out, `"finishReason":"STOP"`) || !strings.Contains(out, `"usageMetadata"`) {
		t.Fatalf("usage close not folded into the wire's terminal frame:\n%s", out)
	}
}
