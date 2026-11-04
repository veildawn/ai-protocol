package stream

import (
	"errors"
	"strings"
	"testing"
)

func TestFramerEmitsOnlyCompleteEventsAcrossChunks(t *testing.T) {
	input := "event: response.output_text.delta\r\n" +
		"data: {\"type\":\"response.output_text.delta\",\r\n" +
		"data: \"delta\":\"hello\"}\r\n\r\n" +
		": keepalive\n\n" +
		"data: [DONE]\n\n"
	var framer Framer
	var events []Event
	for _, b := range []byte(input) {
		if err := framer.Feed([]byte{b}, func(ev Event) error {
			events = append(events, ev)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if len(events) != 2 {
		t.Fatalf("Feed emitted %d events before Finish; want the two complete frames", len(events))
	}
	if events[0].Event != "response.output_text.delta" || events[0].Data != "{\"type\":\"response.output_text.delta\",\n\"delta\":\"hello\"}" {
		t.Fatalf("first event = %+v", events[0])
	}
	if events[1].Data != "[DONE]" {
		t.Fatalf("second event = %+v", events[1])
	}
	if err := framer.Finish(func(ev Event) error {
		events = append(events, ev)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("Finish duplicated a dispatched event: got %d events", len(events))
	}
}

func TestFramerFinishesAnEventAtEOF(t *testing.T) {
	var framer Framer
	var events []Event
	if err := framer.Feed([]byte("data: [DONE]"), func(ev Event) error {
		events = append(events, ev)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 0 {
		t.Fatal("Feed must not dispatch an unterminated event")
	}
	if err := framer.Finish(func(ev Event) error {
		events = append(events, ev)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Data != "[DONE]" {
		t.Fatalf("events at EOF = %+v, want one DONE event", events)
	}
}

func TestFramerPropagatesHandlerErrorAndRemainsUsable(t *testing.T) {
	want := errors.New("stop")
	var framer Framer
	if err := framer.Feed([]byte("data: first\n\n"), func(Event) error { return want }); !errors.Is(err, want) {
		t.Fatalf("Feed error = %v, want %v", err, want)
	}
	var got []Event
	if err := framer.Feed([]byte("data: second\n\n"), func(ev Event) error {
		got = append(got, ev)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Data != "second" {
		t.Fatalf("events after handler error = %+v", got)
	}
}

func TestFramerRejectsOversizedLines(t *testing.T) {
	var framer Framer
	err := framer.Feed([]byte("data: "+strings.Repeat("x", maxEventBytes)), func(Event) error { return nil })
	if err == nil {
		t.Fatal("Feed accepted a line over the existing scanner limit")
	}
}

func TestFramerRejectsOversizedMultiLineEvent(t *testing.T) {
	var framer Framer
	half := strings.Repeat("a", (maxEventBytes/2)+1)
	// First line fits within maxEventBytes
	if err := framer.Feed([]byte("data: "+half+"\n"), func(Event) error { return nil }); err != nil {
		t.Fatalf("first line failed: %v", err)
	}
	// Second line by itself fits within line limit, but aggregate exceeds maxEventBytes
	err := framer.Feed([]byte("data: "+half+"\n\n"), func(Event) error { return nil })
	if err == nil {
		t.Fatal("Feed accepted multi-line data payload exceeding maxEventBytes")
	}
}
