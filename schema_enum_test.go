package protocol_test

import (
	"encoding/json"
	"testing"

	"github.com/veildawn/ai-protocol"
)

// A gemini client declares its tools with the wire's native SCREAMING_SNAKE
// type enums; every dialect the request folds onto validates the lowercase
// JSON Schema vocabulary and refuses the uppercase spelling. The out-of-gemini
// folds must re-spell the schema they relay.
const uppercaseSchemaTools = `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[` +
	`{"function_declarations":[{"name":"get_weather","parameters":{"type":"OBJECT",` +
	`"properties":{"city":{"type":"STRING"},"days":{"type":"INTEGER"},` +
	`"opts":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"k":{"type":"BOOLEAN"}}}}},` +
	`"required":["city"]}}]}]}`

func TestGeminiToolsToChatLowercasesSchemaEnums(t *testing.T) {
	converted, err := protocol.ConvertRequest(protocol.Gemini, protocol.Chat, []byte(uppercaseSchemaTools))
	if err != nil {
		t.Fatalf("fold failed: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(converted.Body, &out); err != nil {
		t.Fatalf("fold output not JSON: %v", err)
	}
	tools, _ := out["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("root type = %v, want object", params["type"])
	}
	props := params["properties"].(map[string]any)
	for key, want := range map[string]string{"city": "string", "days": "integer"} {
		if got := props[key].(map[string]any)["type"]; got != want {
			t.Errorf("%s type = %v, want %s", key, got, want)
		}
	}
	items := props["opts"].(map[string]any)["items"].(map[string]any)
	if items["type"] != "object" {
		t.Errorf("items type = %v, want object", items["type"])
	}
	inner := items["properties"].(map[string]any)["k"].(map[string]any)
	if inner["type"] != "boolean" {
		t.Errorf("nested type = %v, want boolean", inner["type"])
	}
}
