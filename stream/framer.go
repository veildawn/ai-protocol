package stream

import (
	"bytes"
	"fmt"
	"strings"
)

// Framer incrementally decodes SSE events. Data received by Feed is not
// dispatched until a complete event delimiter is seen; Finish dispatches the
// last event at EOF, as SSE requires. Callers that also need exact wire bytes
// should retain them separately: Event intentionally contains only protocol
// fields, not id/retry or the original line endings.
type Framer struct {
	line []byte
	// crPending records that the last chunk ended with \r, which may yet turn
	// out to be the first half of a CRLF ending split across chunks.
	crPending bool
	event     string
	data      []string
}

// Feed adds a chunk and calls fn for each complete event in it. The partial
// line and event are retained for the next call. The scanner's existing
// per-line limit is enforced before appending more bytes.
func (f *Framer) Feed(p []byte, fn func(Event) error) error {
	for len(p) > 0 {
		if f.crPending {
			// The previous chunk ended in \r. An \n now completes one CRLF
			// ending; anything else leaves that \r as the lone-CR ending it
			// already was. Either way the line it ended stays ended.
			f.crPending = false
			if p[0] == '\n' {
				p = p[1:]
				continue
			}
		}
		nl := bytes.IndexByte(p, '\n')
		cr := bytes.IndexByte(p, '\r')
		// A line ends with \n, \r\n, or a lone \r — the SSE grammar accepts
		// all three, and so does the reference's decoder. The earliest
		// terminator wins.
		end, skip := nl, 1
		if cr >= 0 && (nl < 0 || cr < nl) {
			if cr == len(p)-1 {
				// A trailing \r may be half of a CRLF split across chunks:
				// the line's content is settled either way, so process it and
				// hold only the ending decision.
				if err := f.takeLine(p[:cr], fn); err != nil {
					return err
				}
				f.crPending = true
				return nil
			}
			end = cr
			if p[cr+1] == '\n' {
				skip = 2
			}
		}
		if end < 0 {
			if len(f.line)+len(p) > maxEventBytes {
				return fmt.Errorf("SSE line exceeds %d bytes", maxEventBytes)
			}
			f.line = append(f.line, p...)
			return nil
		}
		if err := f.takeLine(p[:end], fn); err != nil {
			return err
		}
		p = p[end+skip:]
	}
	return nil
}

// takeLine appends content to the pending line and processes the result as one
// complete line.
func (f *Framer) takeLine(content []byte, fn func(Event) error) error {
	if len(f.line)+len(content) > maxEventBytes {
		return fmt.Errorf("SSE line exceeds %d bytes", maxEventBytes)
	}
	f.line = append(f.line, content...)
	err := f.processLine(f.line, fn)
	f.line = f.line[:0]
	return err
}

// Finish processes an unterminated final line and dispatches any event still
// pending at EOF. It is safe to call on an empty or already-finished framer.
func (f *Framer) Finish(fn func(Event) error) error {
	if len(f.line) > 0 {
		if err := f.processLine(f.line, fn); err != nil {
			f.line = f.line[:0]
			return err
		}
		f.line = f.line[:0]
	}
	return f.flush(fn)
}

func (f *Framer) processLine(line []byte, fn func(Event) error) error {
	if len(line) > 0 && line[len(line)-1] == '\r' {
		line = line[:len(line)-1]
	}
	if len(line) == 0 {
		return f.flush(fn)
	}
	if line[0] == ':' {
		return nil
	}
	name, value, _ := bytes.Cut(line, []byte{':'})
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	switch string(name) {
	case "event":
		f.event = string(value)
	case "data":
		f.data = append(f.data, string(value))
	}
	return nil
}

func (f *Framer) flush(fn func(Event) error) error {
	if f.event == "" && len(f.data) == 0 {
		return nil
	}
	ev := Event{Event: f.event, Data: strings.Join(f.data, "\n")}
	f.event = ""
	f.data = f.data[:0]
	return fn(ev)
}
