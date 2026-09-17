package chatresponses

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

func ChatToResponses(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	instructions, input := chatMessagesToResponsesInput(msgs)
	out["input"] = input
	if instructions != "" {
		out["instructions"] = instructions
	}
	if v, ok := body["max_completion_tokens"]; ok {
		out["max_output_tokens"] = v
	} else if v, ok := body["max_tokens"]; ok {
		out["max_output_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	copyIf(body, out, "user")
	copyIf(body, out, "parallel_tool_calls")
	copyIf(body, out, "metadata")
	if v, ok := body["tools"]; ok {
		out["tools"] = chatToolsToResponses(v)
	}
	if v, ok := body["tool_choice"]; ok {
		out["tool_choice"] = chatToolChoiceToResponses(v)
	}
	if rf, ok := jsonx.AsMap(body["response_format"]); ok {
		if text := responseFormatToText(rf); text != nil {
			out["text"] = text
		}
	}
	if effort, ok := body["reasoning_effort"]; ok {
		switch t := effort.(type) {
		case string:
			if t != "" && t != "none" {
				out["reasoning"] = map[string]any{"effort": t}
			}
		case map[string]any:
			out["reasoning"] = t
		}
	}
	if th, ok := jsonx.AsMap(body["thinking"]); ok {
		if jsonx.GetString(th, "type") == "enabled" {
			b, _ := jsonx.Int(th["budget_tokens"])
			out["reasoning"] = map[string]any{"effort": types.EffortFromBudget(b)}
		}
	}
	if _, ok := body["stream"]; ok {
		if b, ok := jsonx.Bool(body["stream"]); ok && b {
			out["stream"] = true
		}
	}
	return out, nil
}

func ResponsesToChat(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	input := body["input"]
	msgs := responsesInputToChatMessages(input)
	if inst := jsonx.GetString(body, "instructions"); inst != "" {
		msgs = append([]any{map[string]any{"role": "system", "content": inst}}, msgs...)
	}
	out["messages"] = msgs
	if v, ok := body["max_output_tokens"]; ok {
		out["max_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	copyIf(body, out, "user")
	copyIf(body, out, "parallel_tool_calls")
	copyIf(body, out, "metadata")
	if v, ok := body["tools"]; ok {
		out["tools"] = responsesToolsToChat(v)
	}
	if v, ok := body["tool_choice"]; ok {
		out["tool_choice"] = responsesToolChoiceToChat(v)
	}
	if text, ok := jsonx.AsMap(body["text"]); ok {
		if rf := textFormatToResponseFormat(text); rf != nil {
			out["response_format"] = rf
		}
	}
	if r, ok := jsonx.AsMap(body["reasoning"]); ok {
		if _, hasSummary := r["summary"]; hasSummary {
			out["reasoning_effort"] = r
		} else if e, ok := r["effort"]; ok {
			out["reasoning_effort"] = e
		} else {
			out["reasoning_effort"] = r
		}
	}
	if b, ok := jsonx.Bool(body["stream"]); ok && b {
		out["stream"] = true
		out["stream_options"] = map[string]any{"include_usage": true}
	}
	// drop nil-ish
	for k, v := range out {
		if v == nil {
			delete(out, k)
		}
	}
	return out, nil
}

func copyIf(src, dst map[string]any, key string) {
	if v, ok := src[key]; ok && v != nil {
		dst[key] = v
	}
}

func chatMessagesToResponsesInput(msgs []any) (instructions string, input []any) {
	var inst []string
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		role := jsonx.GetString(m, "role")
		if role == "system" || role == "developer" {
			inst = append(inst, contentToPlain(m["content"]))
			continue
		}
		if role == "tool" {
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": jsonx.GetString(m, "tool_call_id"),
				"output":  contentToPlain(m["content"]),
			})
			continue
		}
		if role == "assistant" {
			if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok {
				for _, tc := range tcs {
					tm, ok := jsonx.AsMap(tc)
					if !ok {
						continue
					}
					fn, _ := jsonx.AsMap(tm["function"])
					args := "{}"
					name := ""
					if fn != nil {
						args = jsonx.GetString(fn, "arguments")
						if args == "" {
							args = "{}"
						}
						name = jsonx.GetString(fn, "name")
					}
					input = append(input, map[string]any{
						"type":      "function_call",
						"call_id":   jsonx.GetString(tm, "id"),
						"name":      name,
						"arguments": args,
					})
				}
			}
			if rc := jsonx.GetString(m, "reasoning_content"); rc != "" {
				input = append(input, map[string]any{
					"type":    "reasoning",
					"summary": []any{map[string]any{"type": "summary_text", "text": rc}},
				})
			}
			text := contentToPlain(m["content"])
			if text != "" {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": text}},
				})
			}
			continue
		}
		// user
		input = append(input, map[string]any{
			"type":    "message",
			"role":    "user",
			"content": chatContentToInputParts(m["content"]),
		})
	}
	return strings.Join(inst, "\n"), input
}

func chatContentToInputParts(content any) []any {
	switch c := content.(type) {
	case string:
		return []any{map[string]any{"type": "input_text", "text": c}}
	case []any:
		var parts []any
		for _, p := range c {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			switch jsonx.GetString(pm, "type") {
			case "text", "":
				parts = append(parts, map[string]any{"type": "input_text", "text": jsonx.GetString(pm, "text")})
			case "image_url":
				url := ""
				if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
					url = jsonx.GetString(iu, "url")
				}
				parts = append(parts, map[string]any{"type": "input_image", "image_url": url})
			}
		}
		if len(parts) == 0 {
			return []any{map[string]any{"type": "input_text", "text": ""}}
		}
		return parts
	default:
		return []any{map[string]any{"type": "input_text", "text": jsonx.String(content)}}
	}
}

func responsesInputToChatMessages(input any) []any {
	if s, ok := input.(string); ok {
		return []any{map[string]any{"role": "user", "content": s}}
	}
	list, ok := jsonx.AsSlice(input)
	if !ok {
		return []any{}
	}
	var msgs []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		typ := jsonx.GetString(m, "type")
		switch typ {
		case "function_call_output":
			msgs = append(msgs, map[string]any{
				"role":         "tool",
				"tool_call_id": jsonx.GetString(m, "call_id"),
				"content":      jsonx.GetString(m, "output"),
			})
		case "function_call":
			msgs = append(msgs, map[string]any{
				"role": "assistant",
				"tool_calls": []any{map[string]any{
					"id":   jsonx.GetString(m, "call_id"),
					"type": "function",
					"function": map[string]any{
						"name":      jsonx.GetString(m, "name"),
						"arguments": jsonx.GetString(m, "arguments"),
					},
				}},
			})
		case "reasoning":
			text := reasoningText(m)
			if text != "" {
				msgs = append(msgs, map[string]any{
					"role":              "assistant",
					"content":           nil,
					"reasoning_content": text,
				})
			}
		case "message", "":
			role := jsonx.GetString(m, "role")
			if role == "" {
				role = "user"
			}
			if role == "system" || role == "developer" {
				msgs = append(msgs, map[string]any{"role": "system", "content": partsToText(m["content"], "input_text", "output_text", "text")})
				continue
			}
			content, images := partsToChatContent(m["content"], role)
			msg := map[string]any{"role": role, "content": content}
			_ = images
			msgs = append(msgs, msg)
		}
	}
	return msgs
}

func partsToChatContent(content any, role string) (any, bool) {
	if s, ok := content.(string); ok {
		return s, false
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content, false
	}
	var texts []string
	var parts []any
	hasNonText := false
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "output_text", "text":
			t := jsonx.GetString(pm, "text")
			texts = append(texts, t)
			parts = append(parts, map[string]any{"type": "text", "text": t})
		case "input_image":
			hasNonText = true
			url := jsonx.GetString(pm, "image_url")
			parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
		}
	}
	if !hasNonText {
		return strings.Join(texts, ""), false
	}
	return parts, true
}

func partsToText(content any, types ...string) string {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return jsonx.String(content)
	}
	want := map[string]struct{}{}
	for _, t := range types {
		want[t] = struct{}{}
	}
	var parts []string
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		if _, ok := want[jsonx.GetString(pm, "type")]; ok || jsonx.GetString(pm, "type") == "" {
			parts = append(parts, jsonx.GetString(pm, "text"))
		}
	}
	return strings.Join(parts, "\n")
}

func reasoningText(m map[string]any) string {
	if s := jsonx.GetString(m, "content"); s != "" {
		return s
	}
	if list, ok := jsonx.AsSlice(m["summary"]); ok {
		var parts []string
		for _, p := range list {
			if pm, ok := jsonx.AsMap(p); ok {
				parts = append(parts, jsonx.GetString(pm, "text"))
			} else if s, ok := p.(string); ok {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "")
	}
	return ""
}

func contentToPlain(content any) string {
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
		return strings.Join(parts, "")
	default:
		if c == nil {
			return ""
		}
		return jsonx.String(c)
	}
}

func chatToolsToResponses(v any) []any {
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
		if jsonx.GetString(m, "type") == "function" {
			fn, _ := jsonx.AsMap(m["function"])
			tool := map[string]any{"type": "function"}
			if fn != nil {
				tool["name"] = jsonx.GetString(fn, "name")
				if d := jsonx.GetString(fn, "description"); d != "" {
					tool["description"] = d
				}
				if p, ok := fn["parameters"]; ok {
					tool["parameters"] = p
				}
				if st, ok := fn["strict"]; ok {
					tool["strict"] = st
				}
			}
			out = append(out, tool)
			continue
		}
		if strings.HasPrefix(jsonx.GetString(m, "type"), "web_search") || jsonx.GetString(m, "name") == "web_search" {
			out = append(out, map[string]any{"type": "web_search_preview"})
			continue
		}
		out = append(out, m)
	}
	return out
}

func responsesToolsToChat(v any) []any {
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
		typ := jsonx.GetString(m, "type")
		if typ == "function" || typ == "" && jsonx.GetString(m, "name") != "" {
			fn := map[string]any{"name": jsonx.GetString(m, "name")}
			if d := jsonx.GetString(m, "description"); d != "" {
				fn["description"] = d
			}
			if p, ok := m["parameters"]; ok {
				fn["parameters"] = p
			}
			if st, ok := m["strict"]; ok {
				fn["strict"] = st
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
			continue
		}
		if strings.Contains(typ, "web_search") {
			out = append(out, map[string]any{"type": "function", "function": map[string]any{
				"name":        "web_search",
				"description": "web search",
				"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
			}})
			continue
		}
		out = append(out, m)
	}
	return out
}

func chatToolChoiceToResponses(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		if fn, ok := jsonx.AsMap(t["function"]); ok {
			name := jsonx.GetString(fn, "name")
			if name != "" {
				return map[string]any{"type": "function", "name": name}
			}
		}
		if typ == "function" && jsonx.GetString(t, "name") != "" {
			return t
		}
		return t
	default:
		return v
	}
}

func responsesToolChoiceToChat(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto", "none", "required":
			return typ
		case "tool", "any":
			return "required"
		case "function":
			if jsonx.GetString(t, "name") != "" && t["function"] == nil {
				return map[string]any{"type": "function", "function": map[string]any{"name": jsonx.GetString(t, "name")}}
			}
			return t
		default:
			return t
		}
	default:
		return v
	}
}

func responseFormatToText(rf map[string]any) map[string]any {
	switch jsonx.GetString(rf, "type") {
	case "json_schema":
		js, _ := jsonx.AsMap(rf["json_schema"])
		format := map[string]any{"type": "json_schema"}
		if js != nil {
			format["name"] = jsonx.GetString(js, "name")
			if format["name"] == "" {
				format["name"] = "response_schema"
			}
			format["schema"] = js["schema"]
			if st, ok := js["strict"]; ok {
				format["strict"] = st
			} else {
				format["strict"] = false
			}
		}
		return map[string]any{"format": format}
	case "json_object":
		return map[string]any{"format": map[string]any{"type": "json_object"}}
	default:
		return nil
	}
}

func textFormatToResponseFormat(text map[string]any) map[string]any {
	format, ok := jsonx.AsMap(text["format"])
	if !ok {
		return nil
	}
	switch jsonx.GetString(format, "type") {
	case "json_schema":
		return map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   first(jsonx.GetString(format, "name"), "response_schema"),
				"schema": format["schema"],
				"strict": format["strict"],
			},
		}
	case "json_object":
		return map[string]any{"type": "json_object"}
	default:
		return nil
	}
}

func first(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func mustArgs(v any) string {
	switch t := v.(type) {
	case string:
		return t
	default:
		b, _ := json.Marshal(v)
		if len(b) == 0 {
			return "{}"
		}
		return string(b)
	}
}
