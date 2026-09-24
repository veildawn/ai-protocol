package protocol

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

// chunkedReader hands the stream out in fixed-size pieces, so SSE events
// straddle reads the way they do on a real connection.
type chunkedReader struct {
	data []byte
	size int
	off  int
}

func (r *chunkedReader) Read(p []byte) (int, error) {
	if r.off >= len(r.data) {
		return 0, io.EOF
	}
	n := min(r.size, len(p), len(r.data)-r.off)
	copy(p, r.data[r.off:r.off+n])
	r.off += n
	return n, nil
}

// A complete stream must report Terminal however its bytes are divided: an SSE
// event routinely straddles two reads, and the terminal frame is the largest one
// in the stream. Reading each chunk in isolation lost it, so a host that bills
// on StreamResult booked delivered turns as failed deliveries — the gateway
// logged "upstream stream truncated before terminal frame" (and cooled the
// account for it) for turns whose clients held a complete answer.
func TestStreamResultSurvivesSplitEvents(t *testing.T) {
	responses := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`, "",
		`data: {"type":"response.output_text.delta","delta":"hi"}`, "",
		`data: {"type":"response.completed","response":{"id":"r1","model":"m","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":2}}}`, "",
	}, "\n")
	chat := strings.Join([]string{
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`, "",
		`data: [DONE]`, "",
	}, "\n")

	for _, tc := range []struct {
		name     string
		from, to Dialect
		src      string
	}{
		{"responses terminal", Responses, Chat, responses},
		{"chat sentinel", Chat, Responses, chat},
	} {
		for _, size := range []int{3, 7, 64, 4096} {
			var dst bytes.Buffer
			result, err := PipeStreamWith(tc.from, tc.to, &dst, &chunkedReader{data: []byte(tc.src), size: size}, StreamOpts{})
			if err != nil {
				t.Fatalf("%s size=%d: %v", tc.name, size, err)
			}
			if !result.Terminal || result.Truncated || result.InBandErr != nil {
				t.Errorf("%s size=%d: result=%+v, want a terminal and no truncation", tc.name, size, result)
			}
		}
	}
}

// The fix must not launder a real truncation: a stream that stops before its
// terminal frame is still reported as truncated, split across reads or not.
func TestStreamResultStillReportsTruncationWhenSplit(t *testing.T) {
	src := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`, "",
		`data: {"type":"response.output_text.delta","delta":"hi"}`, "",
	}, "\n")
	for _, size := range []int{3, 7, 64, 4096} {
		var dst bytes.Buffer
		result, err := PipeStreamWith(Responses, Chat, &dst, &chunkedReader{data: []byte(src), size: size}, StreamOpts{})
		if err != nil {
			t.Fatalf("size=%d: %v", size, err)
		}
		if result.Terminal || !result.Truncated || result.InBandErr != nil {
			t.Errorf("size=%d: result=%+v, want truncated with no terminal", size, result)
		}
	}
}
