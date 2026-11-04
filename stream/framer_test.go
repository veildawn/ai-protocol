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
	if err := framer.Feed([]byte("data: "+half+"\n"), func(Event) error { return nil }); err != nil {
		t.Fatalf("first line: %v", err)
	}
	err := framer.Feed([]byte("data: "+half+"\n\n"), func(Event) error { return nil })
	if err == nil {
		t.Fatal("Feed accepted multi-line data over the event ceiling")
	}
}

func TestFramerCountsJoinNewlinesTowardTheEventCeiling(t *testing.T) {
	// Leave room for the "data: " prefix so each line stays within the line
	// cap, and land the joined payload on the event ceiling. One more value
	// byte must fail: the newline between the fields is part of the payload,
	// so a counter that only added len(value) would still accept it.
	const lead = "abcdef"
	restN := maxEventBytes - len(lead) - 1
	rest := strings.Repeat("c", restN)
	var got Event
	var framer Framer
	if err := framer.Feed([]byte("data: "+lead+"\ndata: "+rest+"\n\n"), func(ev Event) error {
		got = ev
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != maxEventBytes || got.Data != lead+"\n"+rest {
		t.Fatalf("joined length = %d, want %d", len(got.Data), maxEventBytes)
	}

	over := strings.Repeat("c", restN+1)
	var overFramer Framer
	if err := overFramer.Feed([]byte("data: "+lead+"\ndata: "+over+"\n\n"), func(Event) error { return nil }); err == nil {
		t.Fatal("Feed accepted joined data one byte over the ceiling")
	}
}

func TestFramerDispatchesEmptyDataFields(t *testing.T) {
	var framer Framer
	var got []Event
	if err := framer.Feed([]byte("data:\n\ndata:\ndata:\n\n"), func(ev Event) error {
		got = append(got, ev)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Data != "" || got[1].Data != "\n" {
		t.Fatalf("events = %#v, want one empty field and two empty fields joined as a newline", got)
	}
}

func TestFramerEmptyDataFieldsCountTowardTheCeiling(t *testing.T) {
	// One legal line fills the event to just under the ceiling. Each empty
	// field after it adds the newline the payload will contain. A counter
	// that ignores empty values would keep accepting them.
	const prefix = "data: "
	value := strings.Repeat("a", maxEventBytes-len(prefix))
	var b strings.Builder
	b.WriteString(prefix)
	b.WriteString(value)
	b.WriteByte('\n')
	for i := 0; i < len(prefix); i++ {
		b.WriteString("data:\n")
	}
	body := b.String()

	var got Event
	var framer Framer
	if err := framer.Feed([]byte(body+"\n"), func(ev Event) error {
		got = ev
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != maxEventBytes || !strings.HasPrefix(got.Data, value) || !strings.HasSuffix(got.Data, strings.Repeat("\n", len(prefix))) {
		t.Fatalf("empty fields did not fill the ceiling: length %d", len(got.Data))
	}

	var over Framer
	if err := over.Feed([]byte(body+"data:\n"), func(Event) error { return nil }); err == nil {
		t.Fatal("Feed accepted empty data fields past the event ceiling")
	}
}
