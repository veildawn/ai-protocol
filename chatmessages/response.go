package chatmessages

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

func ChatResponseToMessages(body map[string]any) map[string]any {
	choices, _ := jsonx.AsSlice(body["choices"])
	content := openaiChoicesToAnthropicContent(choices)
	stop := "end_turn"
	if len(choices) > 0 {
		if c, ok := jsonx.AsMap(choices[0]); ok {
			stop = finishToStopReason(jsonx.GetString(c, "finish_reason"))
		}
	}
	usage := openaiUsageToAnthropic(body["usage"])
	model := jsonx.GetString(body, "model")
	if model == "" {
		model = "unknown-model"
	}
	id := jsonx.GetString(body, "id")
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"stop_sequence": nil,
		"usage":         usage,
		"content":       content,
		"stop_reason":   stop,
	}
}

func MessagesResponseToChat(body map[string]any) map[string]any {
	content, _ := jsonx.AsSlice(body["content"])
	msg := anthropicAssistantToOpenAI(map[string]any{"role": "assistant", "content": content})
	finish := stopReasonToFinish(jsonx.GetString(body, "stop_reason"))
	usage := anthropicUsageToOpenAI(body["usage"])
	id := jsonx.GetString(body, "id")
	if id == "" {
		id = "chatcmpl-unknown"
	}
	model := jsonx.GetString(body, "model")
	created := body["created"]
	if created == nil {
		created = 0
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       msg,
				"finish_reason": finish,
			},
		},
		"usage": usage,
	}
}

func openaiChoicesToAnthropicContent(choices []any) []any {
	var content []any
	for _, ch := range choices {
		cm, ok := jsonx.AsMap(ch)
		if !ok {
			continue
		}
		msg, _ := jsonx.AsMap(cm["message"])
		if msg == nil {
			continue
		}
		if tbs, ok := jsonx.AsSlice(msg["thinking_blocks"]); ok {
			for _, tb := range tbs {
				if tm, ok := jsonx.AsMap(tb); ok {
					content = append(content, tm)
				}
			}
		} else if rc := jsonx.GetString(msg, "reasoning_content"); rc != "" {
			content = append(content, map[string]any{"type": "thinking", "thinking": rc})
		}
		if msg["content"] != nil {
			if s, ok := msg["content"].(string); ok {
				content = append(content, map[string]any{"type": "text", "text": s})
			} else if s := jsonx.String(msg["content"]); s != "" && msg["content"] != nil {
				content = append(content, map[string]any{"type": "text", "text": s})
			}
		}
		if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
			for _, tc := range tcs {
				tm, ok := jsonx.AsMap(tc)
				if !ok {
					continue
				}
				fn, _ := jsonx.AsMap(tm["function"])
				input := any(map[string]any{})
				if fn != nil {
					if args := jsonx.GetString(fn, "arguments"); args != "" {
						var parsed any
						if err := json.Unmarshal([]byte(args), &parsed); err == nil {
							input = parsed
						}
					}
				}
				name := ""
				if fn != nil {
					name = jsonx.GetString(fn, "name")
				}
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    jsonx.GetString(tm, "id"),
					"name":  name,
					"input": input,
				})
			}
		}
	}
	if content == nil {
		content = []any{}
	}
	return content
}

func finishToStopReason(fr string) string {
	switch fr {
	case "stop", "":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func stopReasonToFinish(sr string) string {
	switch sr {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return "stop"
	}
}

func openaiUsageToAnthropic(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"input_tokens": 0, "output_tokens": 0}
	}
	in, _ := jsonx.Int(m["prompt_tokens"])
	if in == 0 {
		in, _ = jsonx.Int(m["input_tokens"])
	}
	out, _ := jsonx.Int(m["completion_tokens"])
	if out == 0 {
		out, _ = jsonx.Int(m["output_tokens"])
	}
	u := map[string]any{"input_tokens": in, "output_tokens": out}
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		if c, ok := jsonx.Int(d["cached_tokens"]); ok && c > 0 {
			u["cache_read_input_tokens"] = c
		}
	}
	if c, ok := jsonx.Int(m["cache_read_input_tokens"]); ok && c > 0 {
		u["cache_read_input_tokens"] = c
	}
	if c, ok := jsonx.Int(m["cache_creation_input_tokens"]); ok && c > 0 {
		u["cache_creation_input_tokens"] = c
	}
	return u
}

func anthropicUsageToOpenAI(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	}
	in, _ := jsonx.Int(m["input_tokens"])
	out, _ := jsonx.Int(m["output_tokens"])
	cacheRead, _ := jsonx.Int(m["cache_read_input_tokens"])
	cacheWrite, _ := jsonx.Int(m["cache_creation_input_tokens"])
	// OpenAI prompt_tokens is inclusive of cache read/write; Anthropic
	// input_tokens is the uncached remainder.
	prompt := in + cacheRead + cacheWrite
	u := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": out,
		"total_tokens":      prompt + out,
	}
	if cacheRead > 0 {
		u["prompt_tokens_details"] = map[string]any{"cached_tokens": cacheRead}
	}
	return u
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
