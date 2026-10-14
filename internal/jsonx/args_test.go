package jsonx

import (
	"encoding/json"
	"testing"
)

func TestArgsValue(t *testing.T) {
	cases := []struct{ in, want string }{
		// Key order is the string's own; separators are the reference
		// encoder's at every level.
		{`{"zeta": 1, "alpha": 2}`, `{"zeta": 1, "alpha": 2}`},
		{`{"zeta":1,"alpha":2}`, `{"zeta": 1, "alpha": 2}`},
		{`{"mid": {"y": true, "x": null}, "list": [3, 1]}`, `{"mid": {"y": true, "x": null}, "list": [3, 1]}`},
		{`{}`, `{}`},
		{`{"a": {}, "b": []}`, `{"a": {}, "b": []}`},
		// Strings re-escape the way the reference encoder does: the JSON
		// escapes, and \uXXXX for everything past ASCII — & and < stay bare.
		{`{"k": "caf\u00e9 \"q\" \\ \n&<>"}`, `{"k": "caf\u00e9 \"q\" \\ \n&<>"}`},
	}
	for _, tc := range cases {
		raw, ok := ArgsValue(tc.in)
		if !ok {
			t.Fatalf("%s: not ok", tc.in)
		}
		if string(raw) != tc.want {
			t.Fatalf("%s\n got  %s\n want %s", tc.in, raw, tc.want)
		}
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			t.Fatalf("%s: rendered bytes are not JSON: %v", tc.in, err)
		}
	}
	for _, bad := range []string{``, `5`, `[1,2]`, `"s"`, `{"a":1} trailing`, `not json`, `{"a": 1`} {
		if _, ok := ArgsValue(bad); ok {
			t.Fatalf("%q: want not-ok", bad)
		}
	}
}
