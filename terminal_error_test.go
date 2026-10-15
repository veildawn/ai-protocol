package protocol

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteTerminalError(t *testing.T) {
	cases := []struct {
		name string
		d    Dialect
		want string
	}{
		{
			"gemini is an error-object data frame with no sentinel",
			Gemini,
			"data: {\"error\":{\"code\":503,\"message\":\"m\",\"status\":\"UNAVAILABLE\"}}\n\n",
		},
		{
			"messages is a named error event with no sentinel",
			Messages,
			"event: error\ndata: {\"error\":{\"message\":\"m\",\"type\":\"upstream_stream_error\"},\"type\":\"error\"}\n\n",
		},
		{
			"responses is a named error event with no sentinel",
			Responses,
			"event: error\ndata: {\"error\":{\"message\":\"m\",\"type\":\"upstream_stream_error\"},\"type\":\"error\"}\n\n",
		},
		{
			"chat carries the sentinel after the payload",
			Chat,
			"data: {\"error\":{\"message\":\"m\",\"type\":\"upstream_stream_error\"},\"type\":\"error\"}\n\ndata: [DONE]\n\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := WriteTerminalError(&buf, tc.d, "upstream_stream_error", "m"); err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if got := buf.String(); got != tc.want {
				t.Fatalf("frame = %q, want %q", got, tc.want)
			}
			switch tc.d {
			case Gemini, Messages, Responses:
				if strings.Contains(buf.String(), "[DONE]") {
					t.Fatalf("dialect %s must not carry chat framing", tc.d)
				}
			}
		})
	}
}

func TestWriteTerminalErrorAcceptsGatewaySpellings(t *testing.T) {
	// Normalize powers the switch, so the gateway's openai/anthropic aliases
	// land on the same molds as the dialect's own name.
	var buf bytes.Buffer
	if err := WriteTerminalError(&buf, "openai", "t", "m"); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if !strings.HasSuffix(buf.String(), "data: [DONE]\n\n") {
		t.Fatalf("alias openai must take the chat shape, got %q", buf.String())
	}
}

func TestWriteTerminalErrorUnknownDialect(t *testing.T) {
	// Images is a real dialect with no streaming wire: it has no terminal
	// frame to mint, and saying so beats folding it onto chat framing.
	for _, d := range []Dialect{Images, "nonsense"} {
		var buf bytes.Buffer
		if err := WriteTerminalError(&buf, d, "t", "m"); err == nil {
			t.Fatalf("dialect %q: err = nil, want one", d)
		}
		if buf.Len() != 0 {
			t.Fatalf("dialect %q wrote %q before failing", d, buf.String())
		}
	}
}
