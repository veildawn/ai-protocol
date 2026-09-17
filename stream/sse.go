package stream

import (
	"bufio"
	"bytes"
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
