package protocol

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestPeekModel(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantModel string
		wantErr   bool
	}{
		{"present", `{"model":"m","messages":[]}`, "m", false},
		{"absent", `{"messages":[]}`, "", false},
		{"non-string model is a decode error", `{"model":7}`, "", true},
		{"malformed json", `{"model":`, "", true},
		{"not an object", `[]`, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := PeekModel([]byte(tc.body))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("err = nil, want a json error")
				}
				var syntax *json.SyntaxError
				var unmarshalType *json.UnmarshalTypeError
				if !errors.As(err, &syntax) && !errors.As(err, &unmarshalType) {
					t.Fatalf("err = %v, want a json decoding error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if model != tc.wantModel {
				t.Fatalf("model = %q, want %q", model, tc.wantModel)
			}
		})
	}
}

// The same-dialect passthrough is the one internal consumer: it must keep
// returning the original bytes and the peeked model without ever failing.
func TestPeekModelFeedsTheSameDialectPassthrough(t *testing.T) {
	body := []byte(`{"model":"m","messages":[]}`)
	conv, err := ConvertRequest(Chat, Chat, body)
	if err != nil {
		t.Fatal(err)
	}
	if string(conv.Body) != string(body) || conv.Model != "m" {
		t.Fatalf("passthrough = (%s, %q), want the exact bytes and the peeked model", conv.Body, conv.Model)
	}
}
