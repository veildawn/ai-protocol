package messagesresponses

import (
	"encoding/json"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

func MessagesResponseToResponses(body map[string]any) map[string]any {
	content, _ := jsonx.AsSlice(body["content"])
	var output []any
	for _, b := range content {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "thinking", "redacted_thinking":
			text := jsonx.GetString(bm, "thinking")
			item := map[string]any{
				"type":    "reasoning",
				"summary": []any{map[string]any{"type": "summary_text", "text": text}},
			}
			if sig := jsonx.GetString(bm, "signature"); sig != "" {
				item["encrypted_content"] = sig
			}
			output = append(output, item)
		case "text":
			output = append(output, map[string]any{
				"type":    "message",
				"role":    "assistant",
				"content": []any{map[string]any{"type": "output_text", "text": jsonx.GetString(bm, "text")}},
			})
		case "tool_use":
			args, _ := json.Marshal(bm["input"])
			if bm["input"] == nil {
				args = []byte("{}")
			}
			output = append(output, map[string]any{
				"type":      "function_call",
				"call_id":   jsonx.GetString(bm, "id"),
				"name":      jsonx.GetString(bm, "name"),
				"arguments": string(args),
			})
		}
	}
	if output == nil {
		output = []any{}
	}
	status := "completed"
	if jsonx.GetString(body, "stop_reason") == "max_tokens" {
		status = "incomplete"
	}
	usage := body["usage"]
	in, out := 0, 0
	if um, ok := jsonx.AsMap(usage); ok {
		in, _ = jsonx.Int(um["input_tokens"])
		out, _ = jsonx.Int(um["output_tokens"])
	}
	return map[string]any{
		"id":     jsonx.GetString(body, "id"),
		"object": "response",
		"model":  jsonx.GetString(body, "model"),
		"status": status,
		"output": output,
		"usage": map[string]any{
			"input_tokens":  in,
			"output_tokens": out,
			"total_tokens":  in + out,
		},
	}
}
