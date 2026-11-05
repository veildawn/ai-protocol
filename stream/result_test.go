package stream

import (
	"strings"
	"testing"
)

func TestObserverSeparatesLogicalChatFinishFromWireTerminal(t *testing.T) {
	var observer Observer
	if err := observer.Feed(Event{Data: `{"choices":[{"delta":{},"finish_reason":"stop"}]}`}); err != nil {
		t.Fatal(err)
	}
	if observer.Result.Terminal {
		t.Fatal("Chat finish_reason was mistaken for the source [DONE] terminal")
	}
	if err := observer.Feed(Event{Data: "[DONE]"}); err != nil {
		t.Fatal(err)
	}
	if !observer.Result.Terminal {
		t.Fatal("source [DONE] was not classified as a wire terminal")
	}
}

// A hook drops an event by returning the zero Event. Encoding that value
// writes a bare blank line — an empty keep-alive at best, a protocol break at
// worst — so the writer must write nothing for it.
func TestWriteEventWithDropsZeroValue(t *testing.T) {
	var b strings.Builder
	n, err := WriteEventWith(&b, Event{Event: "content_block_delta", Data: `{"type":"content_block_delta"}`}, nil,
		func(Event) (Event, error) { return Event{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 || b.Len() != 0 {
		t.Fatalf("wrote %d bytes (%q), want nothing", n, b.String())
	}
}

func TestObserverUsesSharedFactsForInBandFailure(t *testing.T) {
	var observer Observer
	err := observer.Feed(Event{Data: `{"type":"response.failed","response":{"error":{"message":"overloaded"}}}`})
	if err == nil || err.Error() != "overloaded" {
		t.Fatalf("Observer.Feed error = %v, want overloaded", err)
	}
	if !observer.Result.Terminal || observer.Result.InBandErr == nil {
		t.Fatalf("observer result = %+v, want terminal failure", observer.Result)
	}
}
