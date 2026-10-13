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
			// A redacted block carries its payload in `data` and has no
			// summary text; reading only `signature` would emit an empty
			// reasoning item and lose the opaque bytes the next turn replays.
			// Mirrors the streaming path.
			if jsonx.GetString(bm, "type") == "redacted_thinking" {
				item := map[string]any{"type": "reasoning", "summary": []any{}}
				if data := jsonx.GetString(bm, "data"); data != "" {
					item["encrypted_content"] = data
				}
				output = append(output, item)
				continue
			}
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
	incomplete := ""
	switch jsonx.GetString(body, "stop_reason") {
	case "max_tokens":
		status, incomplete = "incomplete", "max_output_tokens"
	case "refusal":
		// A safeguard refusal is not a clean stop; the Responses spelling for
		// it is a content filter, matching the chat direction.
		status, incomplete = "incomplete", "content_filter"
	}
	usage := body["usage"]
	in, out := 0, 0
	cacheRead := 0
	if um, ok := jsonx.AsMap(usage); ok {
		in, _ = jsonx.Int(um["input_tokens"])
		out, _ = jsonx.Int(um["output_tokens"])
		cacheRead, _ = jsonx.Int(um["cache_read_input_tokens"])
	}
	usageOut := map[string]any{
		"input_tokens":  in,
		"output_tokens": out,
		"total_tokens":  in + out,
	}
	if cacheRead > 0 {
		usageOut["input_tokens_details"] = map[string]any{"cached_tokens": cacheRead}
	}
	resp := map[string]any{
		"id":     jsonx.GetString(body, "id"),
		"object": "response",
		"model":  jsonx.GetString(body, "model"),
		"status": status,
		"output": output,
		"usage":  usageOut,
	}
	if status == "incomplete" {
		resp["incomplete_details"] = map[string]any{"reason": incomplete}
	}
	return resp
}
