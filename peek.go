package protocol

import "encoding/json"

func jsonxUnmarshalModel(body []byte, v any) error {
	return json.Unmarshal(body, v)
}

// PeekModel reads the top-level "model" member without a full decode. err is
// non-nil for any body the probe cannot read — malformed JSON, or a model
// member that is not a string; an ABSENT model is "" with no error, because
// whether that is an error is host policy, not codec fact. The same-dialect
// passthrough inside ConvertRequest uses this and deliberately ignores the
// error: a raw copy never fails on the bytes it is copying.
func PeekModel(body []byte) (string, error) {
	var p struct {
		Model string `json:"model"`
	}
	if err := jsonxUnmarshalModel(body, &p); err != nil {
		return "", err
	}
	return p.Model, nil
}
