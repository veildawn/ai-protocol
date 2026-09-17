package jsonx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

func UnmarshalMap(data []byte) (map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, fmt.Errorf("expected JSON object")
	}
	return m, nil
}

func Marshal(v any) ([]byte, error) {
	return json.Marshal(v)
}

func String(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case json.Number:
		return t.String()
	case fmt.Stringer:
		return t.String()
	case nil:
		return ""
	default:
		return fmt.Sprint(t)
	}
}

func AsMap(v any) (map[string]any, bool) {
	m, ok := v.(map[string]any)
	return m, ok
}

func AsSlice(v any) ([]any, bool) {
	s, ok := v.([]any)
	return s, ok
}

func GetString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	return String(m[key])
}

func CloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func Int(v any) (int, bool) {
	switch t := v.(type) {
	case int:
		return t, true
	case int64:
		return int(t), true
	case float64:
		return int(t), true
	case json.Number:
		i, err := t.Int64()
		if err != nil {
			f, ferr := t.Float64()
			if ferr != nil {
				return 0, false
			}
			return int(f), true
		}
		return int(i), true
	default:
		return 0, false
	}
}

func Bool(v any) (bool, bool) {
	b, ok := v.(bool)
	return b, ok
}

func JoinText(parts []string) string {
	return strings.Join(parts, "")
}
