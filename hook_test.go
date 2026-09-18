package protocol

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/stream"
)

// messagesToChatSource is a small Anthropic stream that converts into several
// chat chunks plus the converter's own synthesized terminal frames, which is
// what makes it a good witness for "the hook saw everything that was written".
const messagesToChatSource = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude","role":"assistant","content":[]}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

func scanEvents(t *testing.T, raw string) []stream.Event {
	t.Helper()
	var out []stream.Event
	if err := stream.Scan(strings.NewReader(raw), func(ev stream.Event) error {
		out = append(out, ev)
		return nil
	}); err != nil {
		t.Fatalf("scan: %v", err)
	}
	return out
}

// The hook sees exactly the events that end up in the output, in the order they
// are written — including the terminal frames the CONVERTER synthesizes, which
// never appeared in the source. That is the property the host depends on: it
// can rewrite a terminal frame (a response.completed carrying the full output)
// without having to notice it is synthesized.
func TestStreamHookSeesEveryWrittenEventInOrder(t *testing.T) {
	var seen []stream.Event
	var dst bytes.Buffer
	_, err := PipeStream(Messages, Chat, &dst, strings.NewReader(messagesToChatSource), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) {
			seen = append(seen, ev)
			return ev, nil
		},
	})
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}

	written := scanEvents(t, dst.String())
	if len(seen) != len(written) {
		t.Fatalf("hook saw %d events, %d were written", len(seen), len(written))
	}
	for i := range written {
		if seen[i] != written[i] {
			t.Fatalf("event %d: hook saw %+v, written %+v", i, seen[i], written[i])
		}
	}
	if last := seen[len(seen)-1]; !stream.IsDone(last.Data) {
		t.Fatalf("hook never saw the synthesized [DONE]; last = %+v", last)
	}
}

// A converter that synthesizes its terminal frame without a [DONE] sentinel —
// Responses — must still expose that frame to the hook.
func TestStreamHookSeesSynthesizedTerminalWithoutSentinel(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\ndata: [DONE]\n\n"
	var seen []string
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Responses, &dst, strings.NewReader(src), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) {
			seen = append(seen, ev.Event)
			return ev, nil
		},
	}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	joined := strings.Join(seen, ",")
	if !strings.Contains(joined, "response.completed") {
		t.Fatalf("hook never saw the synthesized terminal: %v", seen)
	}
	if got := scanEvents(t, dst.String()); len(got) != len(seen) {
		t.Fatalf("hook saw %d events, %d written", len(seen), len(got))
	}
}

// The event the hook RETURNS is the event written; the original must not leak
// through anywhere.
func TestStreamHookReplacementIsWhatIsWritten(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &dst, strings.NewReader(messagesToChatSource), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) {
			ev.Data = strings.Replace(ev.Data, "Hi", "HOOKED", 1)
			return ev, nil
		},
	}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	out := dst.String()
	if !strings.Contains(out, "HOOKED") {
		t.Fatalf("replacement missing: %s", out)
	}
	if strings.Contains(out, `"Hi"`) {
		t.Fatalf("original survived the hook: %s", out)
	}
}

// An error from the hook stops the stream immediately and is reported as a pipe
// error, not as an in-band failure or a clean truncation: the host asked to
// stop, and settlement must not read that as the upstream's verdict.
func TestStreamHookErrorAbortsTheStream(t *testing.T) {
	sentinel := errors.New("hook refused")
	seen := 0
	var dst bytes.Buffer
	result, err := PipeStreamWith(Messages, Chat, &dst, strings.NewReader(messagesToChatSource), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) {
			seen++
			if seen == 2 {
				return stream.Event{}, sentinel
			}
			return ev, nil
		},
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want the hook's sentinel", err)
	}
	if result.Written <= 0 {
		t.Fatal("Written = 0; the first event should already have been written")
	}
	if result.Truncated || result.InBandErr != nil {
		t.Fatalf("settlement misread a hook abort: %+v", result)
	}
	// Terminal is deliberately not asserted: it reports what the SOURCE bytes
	// the reader consumed contained, and the observer is a tee on that reader —
	// this fixture arrives in one read, so the source's own terminal frame was
	// seen even though the pipe stopped converting before reaching it. The
	// aborting caller has err for that verdict. See StreamResult.Terminal, whose
	// contract is about the upstream, not about what was converted.
	if got := scanEvents(t, dst.String()); len(got) != 1 {
		t.Fatalf("wrote %d events after aborting on the second", len(got))
	}
}

// Same-dialect is a byte copy with no decoding, so there is nothing to hang a
// hook on. Pinned because it is the one shape a caller could reasonably expect
// the hook to cover.
func TestStreamHookIsNotCalledOnSameDialectCopy(t *testing.T) {
	in := "data: hello\n\n"
	called := false
	var dst bytes.Buffer
	result, err := PipeStreamWith(Chat, Chat, &dst, strings.NewReader(in), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) {
			called = true
			return ev, nil
		},
	})
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if called {
		t.Fatal("the hook ran on the same-dialect copy path")
	}
	if dst.String() != in || result.Written != int64(len(in)) {
		t.Fatalf("copy changed: %q (%d bytes)", dst.String(), result.Written)
	}
}

// Nil and identity hooks must be indistinguishable, byte for byte: that is what
// makes the hook safe to add to an existing pipeline.
func TestStreamHookNilAndIdentityAreIdentical(t *testing.T) {
	var without, with bytes.Buffer
	if _, err := PipeStream(Messages, Chat, &without, strings.NewReader(messagesToChatSource), StreamOpts{}); err != nil {
		t.Fatalf("nil hook: %v", err)
	}
	if _, err := PipeStream(Messages, Chat, &with, strings.NewReader(messagesToChatSource), StreamOpts{
		OnEvent: func(ev stream.Event) (stream.Event, error) { return ev, nil },
	}); err != nil {
		t.Fatalf("identity hook: %v", err)
	}
	if without.String() != with.String() {
		t.Fatalf("identity hook changed the bytes:\n%q\n%q", without.String(), with.String())
	}
}
