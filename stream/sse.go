package stream

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const maxEventBytes = 8 << 20

// Event is one SSE event (optional event name + data payload).
type Event struct {
	Event string
	Data  string
}

// Dialect infers the public source dialect from an event so generic stream
// settlement code can remain in the root package.
func (e Event) Dialect() Dialect {
	if IsDone(e.Data) {
		return Chat
	}
	var obj struct {
		Type string `json:"type"`
	}
	if json.Unmarshal([]byte(e.Data), &obj) != nil {
		return ""
	}
	switch obj.Type {
	case "message_start", "message_delta", "message_stop",
		"content_block_start", "content_block_delta", "content_block_stop":
		return Messages
	case "response.created", "response.completed", "response.incomplete", "response.failed",
		"response.output_text.delta", "response.output_item.added":
		return Responses
	}
	return ""
}

func (e Event) Encode() string {
	var b strings.Builder
	if e.Event != "" {
		b.WriteString("event: ")
		b.WriteString(e.Event)
		b.WriteByte('\n')
	}
	if e.Data != "" {
		for _, line := range strings.Split(e.Data, "\n") {
			b.WriteString("data: ")
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	b.WriteByte('\n')
	return b.String()
}

// WriteEventWith writes one SSE event, giving hook the TARGET event first — the
// one about to be written — and writing whatever it returns. Returning the
// argument unchanged is a pass-through, and a nil hook is the identity, so the
// bytes written are exactly what WriteEvent has always written.
//
// The hook exists so a host can apply policy this library has no opinion about
// to DECODED events instead of re-parsing serialized output. The policy stays
// with the host: nothing here knows what a hook does.
//
// A hook error aborts the stream and is returned as a pipe error.
func WriteEventWith(w io.Writer, e Event, flush func(), hook func(Event) (Event, error)) (int, error) {
	if hook != nil {
		next, err := hook(e)
		if err != nil {
			return 0, Fail("event hook", err)
		}
		e = next
	}
	return WriteEvent(w, e, flush)
}

func WriteEvent(w io.Writer, e Event, flush func()) (int, error) {
	s := e.Encode()
	n, err := io.WriteString(w, s)
	if err == nil && flush != nil {
		flush()
	}
	return n, err
}

func WriteDone(w io.Writer, flush func()) (int, error) {
	return WriteEvent(w, Event{Data: "[DONE]"}, flush)
}

// Scan reads SSE events from r. Multi-line data fields are joined with '\n'.
func Scan(r io.Reader, fn func(Event) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(nil, maxEventBytes)
	var ev Event
	var data []string
	flush := func() error {
		if ev.Event == "" && len(data) == 0 {
			return nil
		}
		ev.Data = strings.Join(data, "\n")
		err := fn(ev)
		ev = Event{}
		data = data[:0]
		return err
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		name, val, _ := strings.Cut(line, ":")
		if strings.HasPrefix(val, " ") {
			val = val[1:]
		}
		switch name {
		case "event":
			ev.Event = val
		case "data":
			data = append(data, val)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if err := flush(); err != nil {
		return err
	}
	return nil
}

func IsDone(data string) bool {
	return bytes.Equal(bytes.TrimSpace([]byte(data)), []byte("[DONE]"))
}

func Fail(op string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("stream %s: %w", op, err)
}
