package messagesresponses

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

func MessagesToResponses(body map[string]any) (map[string]any, error) {
	out := map[string]any{
		"model": jsonx.GetString(body, "model"),
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	input := messagesToInput(msgs)
	system := body["system"]
	if s, ok := system.(string); ok && s != "" {
		out["instructions"] = s
	} else if list, ok := jsonx.AsSlice(system); ok {
		var texts []string
		for _, b := range list {
			if bm, ok := jsonx.AsMap(b); ok && jsonx.GetString(bm, "type") == "text" {
				texts = append(texts, jsonx.GetString(bm, "text"))
			}
		}
		if len(texts) > 0 {
			out["instructions"] = strings.Join(texts, "\n")
		}
	}
	out["input"] = input
	if v, ok := body["max_tokens"]; ok {
		out["max_output_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	if tools, ok := body["tools"]; ok {
		out["tools"] = toolsToResponses(tools)
	}
	if tc, ok := body["tool_choice"]; ok {
		out["tool_choice"] = toolChoiceToResponses(tc)
	}
	if thinking, ok := jsonx.AsMap(body["thinking"]); ok {
		if r := thinkingToReasoning(thinking, body["output_config"]); r != nil {
			out["reasoning"] = r
		}
	}
	of := body["output_format"]
	if _, ok := jsonx.AsMap(of); !ok {
		if oc, ok := jsonx.AsMap(body["output_config"]); ok {
			of = oc["format"]
		}
	}
	if fm, ok := jsonx.AsMap(of); ok && jsonx.GetString(fm, "type") == "json_schema" {
		if schema, ok := fm["schema"]; ok && schema != nil {
			strict := fm["strict"]
			if strict == nil {
				strict = false
			}
			out["text"] = map[string]any{"format": map[string]any{
				"type":   "json_schema",
				"name":   "structured_output",
				"schema": schema,
				"strict": strict,
			}}
		}
	}
	if cm, ok := jsonx.AsMap(body["context_management"]); ok {
		if mapped := contextManagementToResponses(cm); mapped != nil {
			out["context_management"] = mapped
		}
	}
	if md, ok := jsonx.AsMap(body["metadata"]); ok {
		if uid := jsonx.GetString(md, "user_id"); uid != "" {
			trunc := uid
			if len(trunc) > 64 {
				trunc = trunc[:64]
			}
			out["user"] = trunc
			out["prompt_cache_key"] = trunc
		}
	}
	// stop_sequences, top_k and speed have no Messages counterpart and are
	// dropped; the omission is reported through the loss API.
	return out, nil
}

func ResponsesToMessages(body map[string]any) (map[string]any, error) {
	output, _ := jsonx.AsSlice(body["output"])
	var content []any
	stop := "end_turn"
	for _, item := range output {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "type") {
		case "reasoning":
			content = append(content, thinkingFromReasoning(m)...)
		case "message":
			if parts, ok := jsonx.AsSlice(m["content"]); ok {
				for _, p := range parts {
					pm, ok := jsonx.AsMap(p)
					if !ok {
						continue
					}
					if jsonx.GetString(pm, "type") == "output_text" {
						content = append(content, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
					}
				}
			}
		case "function_call":
			var input any = map[string]any{}
			if args := jsonx.GetString(m, "arguments"); args != "" {
				var parsed any
				if err := json.Unmarshal([]byte(args), &parsed); err == nil {
					input = parsed
				}
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    first(jsonx.GetString(m, "call_id"), jsonx.GetString(m, "id")),
				"name":  jsonx.GetString(m, "name"),
				"input": input,
			})
			stop = "tool_use"
		}
	}
	if jsonx.GetString(body, "status") == "incomplete" {
		stop = "max_tokens"
	}
	if content == nil {
		content = []any{}
	}
	model := jsonx.GetString(body, "model")
	if model == "" {
		model = "unknown-model"
	}
	return map[string]any{
		"id":            jsonx.GetString(body, "id"),
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"stop_sequence": nil,
		"usage":         usageToAnthropic(body["usage"]),
		"content":       content,
		"stop_reason":   stop,
	}, nil
}

func copyIf(src, dst map[string]any, key string) {
	if v, ok := src[key]; ok {
		dst[key] = v
	}
}

func first(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func messagesToInput(msgs []any) []any {
	var input []any
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		role := jsonx.GetString(m, "role")
		if role == "system" {
			text := contentText(m["content"])
			if text != "" {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "system",
					"content": []any{map[string]any{"type": "input_text", "text": text}},
				})
			}
			continue
		}
		content := m["content"]
		if role == "user" {
			if s, ok := content.(string); ok {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "user",
					"content": []any{map[string]any{"type": "input_text", "text": s}},
				})
				continue
			}
			list, _ := jsonx.AsSlice(content)
			var userParts []any
			for _, b := range list {
				bm, ok := jsonx.AsMap(b)
				if !ok {
					continue
				}
				switch jsonx.GetString(bm, "type") {
				case "text":
					userParts = append(userParts, map[string]any{"type": "input_text", "text": jsonx.GetString(bm, "text")})
				case "image":
					if url := imageURL(bm); url != "" {
						userParts = append(userParts, map[string]any{"type": "input_image", "image_url": url})
					}
				case "document":
					if fp := documentFilePart(bm); fp != nil {
						userParts = append(userParts, fp)
					}
				case "tool_result":
					input = append(input, map[string]any{
						"type":    "function_call_output",
						"call_id": jsonx.GetString(bm, "tool_use_id"),
						"output":  toolResultOutput(bm["content"]),
					})
				}
			}
			if len(userParts) > 0 {
				input = append(input, map[string]any{"type": "message", "role": "user", "content": userParts})
			}
			continue
		}
		if role == "assistant" {
			if s, ok := content.(string); ok {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": s}},
				})
				continue
			}
			list, _ := jsonx.AsSlice(content)
			var textParts []any
			flushText := func() {
				if len(textParts) == 0 {
					return
				}
				input = append(input, map[string]any{"type": "message", "role": "assistant", "content": textParts})
				textParts = nil
			}
			for _, b := range list {
				bm, ok := jsonx.AsMap(b)
				if !ok {
					continue
				}
				switch jsonx.GetString(bm, "type") {
				case "text":
					textParts = append(textParts, map[string]any{"type": "output_text", "text": jsonx.GetString(bm, "text")})
				case "thinking":
					flushText()
					th := jsonx.GetString(bm, "thinking")
					item := map[string]any{
						"type":    "reasoning",
						"summary": []any{map[string]any{"type": "summary_text", "text": th}},
					}
					if sig := jsonx.GetString(bm, "signature"); sig != "" {
						item["encrypted_content"] = sig
					}
					input = append(input, item)
				case "tool_use":
					flushText()
					args, _ := json.Marshal(bm["input"])
					if bm["input"] == nil {
						args = []byte("{}")
					}
					input = append(input, map[string]any{
						"type":      "function_call",
						"call_id":   jsonx.GetString(bm, "id"),
						"name":      jsonx.GetString(bm, "name"),
						"arguments": string(args),
					})
				}
			}
			flushText()
		}
	}
	return input
}

func contentText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, p := range c {
			if pm, ok := jsonx.AsMap(p); ok {
				parts = append(parts, jsonx.GetString(pm, "text"))
			}
		}
		return strings.Join(parts, "\n")
	default:
		return jsonx.String(c)
	}
}

func imageURL(bm map[string]any) string {
	src, ok := jsonx.AsMap(bm["source"])
	if !ok {
		return ""
	}
	switch jsonx.GetString(src, "type") {
	case "url":
		return jsonx.GetString(src, "url")
	case "base64":
		media := jsonx.GetString(src, "media_type")
		if media == "" {
			media = "image/jpeg"
		}
		return "data:" + media + ";base64," + jsonx.GetString(src, "data")
	}
	return ""
}

func documentFilePart(bm map[string]any) map[string]any {
	src, ok := jsonx.AsMap(bm["source"])
	if !ok {
		return nil
	}
	part := map[string]any{"type": "input_file"}
	if jsonx.GetString(src, "type") == "base64" {
		part["file_data"] = jsonx.GetString(src, "data")
		if mt := jsonx.GetString(src, "media_type"); mt != "" {
			part["media_type"] = mt
		}
	}
	return part
}

func toolResultOutput(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, p := range c {
			if pm, ok := jsonx.AsMap(p); ok && jsonx.GetString(pm, "type") == "text" {
				parts = append(parts, jsonx.GetString(pm, "text"))
			}
		}
		return strings.Join(parts, "\n")
	case nil:
		return ""
	default:
		return jsonx.String(c)
	}
}

func toolsToResponses(v any) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var out []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		t := jsonx.GetString(m, "type")
		if strings.HasPrefix(t, "web_search") || jsonx.GetString(m, "name") == "web_search" {
			out = append(out, map[string]any{"type": "web_search_preview"})
			continue
		}
		tool := map[string]any{"type": "function", "name": jsonx.GetString(m, "name")}
		if d := jsonx.GetString(m, "description"); d != "" {
			tool["description"] = d
		}
		if s, ok := m["input_schema"]; ok {
			tool["parameters"] = s
		}
		out = append(out, tool)
	}
	return out
}

func toolChoiceToResponses(v any) any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return v
	}
	switch jsonx.GetString(m, "type") {
	case "any":
		return "required"
	case "tool":
		return map[string]any{"type": "function", "name": jsonx.GetString(m, "name")}
	case "none":
		return "none"
	default:
		return "auto"
	}
}

func thinkingToReasoning(thinking map[string]any, outputConfig any) map[string]any {
	typ := jsonx.GetString(thinking, "type")
	var effort string
	switch typ {
	case "adaptive":
		effort = "medium"
		if oc, ok := jsonx.AsMap(outputConfig); ok {
			if e := jsonx.GetString(oc, "effort"); e != "" {
				effort = e
			}
		}
	case "enabled":
		b, _ := jsonx.Int(thinking["budget_tokens"])
		effort = types.EffortFromBudget(b)
	default:
		return nil
	}
	r := map[string]any{"effort": effort}
	if s := jsonx.GetString(thinking, "summary"); s != "" {
		r["summary"] = s
	} else {
		r["summary"] = "detailed"
	}
	return r
}

func contextManagementToResponses(cm map[string]any) []any {
	edits, ok := jsonx.AsSlice(cm["edits"])
	if !ok {
		return nil
	}
	var result []any
	for _, e := range edits {
		em, ok := jsonx.AsMap(e)
		if !ok {
			continue
		}
		if jsonx.GetString(em, "type") == "compact_20260112" {
			entry := map[string]any{"type": "compaction"}
			if tr, ok := jsonx.AsMap(em["trigger"]); ok {
				if v, ok := jsonx.Int(tr["value"]); ok {
					entry["compact_threshold"] = v
				}
			}
			result = append(result, entry)
		}
	}
	return result
}

func thinkingFromReasoning(m map[string]any) []any {
	var out []any
	if list, ok := jsonx.AsSlice(m["summary"]); ok {
		for _, p := range list {
			text := ""
			if pm, ok := jsonx.AsMap(p); ok {
				text = jsonx.GetString(pm, "text")
			} else {
				text = jsonx.String(p)
			}
			if text != "" {
				out = append(out, map[string]any{"type": "thinking", "thinking": text})
			}
		}
	}
	return out
}

func usageToAnthropic(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"input_tokens": 0, "output_tokens": 0}
	}
	in, _ := jsonx.Int(m["input_tokens"])
	out, _ := jsonx.Int(m["output_tokens"])
	return map[string]any{"input_tokens": in, "output_tokens": out}
}
