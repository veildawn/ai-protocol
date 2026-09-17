package protocol

import "encoding/json"

func jsonxUnmarshalModel(body []byte, v any) error {
	return json.Unmarshal(body, v)
}
