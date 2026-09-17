package chatresponses

import (
	"encoding/json"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

func ChatResponseToResponses(body map[string]any) map[string]any {
	choices, _ := jsonx.AsSlice(body["choices"])
	output := chatChoicesToResponsesOutput(choices)
	finish := ""
	if len(choices) > 0 {
		if c, ok := jsonx.AsMap(choices[0]); ok {
			finish = jsonx.GetString(c, "finish_reason")
		}
	}
	created := body["created"]
	if created == nil {
		created = 0
	}
	return map[string]any{
		"id":                  jsonx.GetString(body, "id"),
		"created_at":          created,
		"model":               jsonx.GetString(body, "model"),
		"object":              "response",
		"output":              output,
		"parallel_tool_calls": false,
		"status":              finishToStatus(finish),
		"text":                map[string]any{},
		"usage":               chatUsageToResponses(body["usage"]),
	}
}

func ResponsesResponseToChat(body map[string]any) map[string]any {
	output, _ := jsonx.AsSlice(body["output"])
	msg := map[string]any{"role": "assistant"}
	var texts []string
	var toolCalls []any
	var reasoning string
	for _, item := range output {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "type") {
		case "message":
			texts = append(texts, partsToText(m["content"], "output_text", "text"))
		case "function_call":
			args := jsonx.GetString(m, "arguments")
			toolCalls = append(toolCalls, map[string]any{
				"id":   first(jsonx.GetString(m, "call_id"), jsonx.GetString(m, "id")),
				"type": "function",
				"function": map[string]any{
					"name":      jsonx.GetString(m, "name"),
					"arguments": args,
				},
			})
		case "reasoning":
			reasoning += reasoningText(m)
		}
	}
	if len(texts) == 0 {
		msg["content"] = nil
	} else if len(texts) == 1 {
		msg["content"] = texts[0]
	} else {
		s := ""
		for _, t := range texts {
			s += t
		}
		msg["content"] = s
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	if reasoning != "" {
		msg["reasoning_content"] = reasoning
	}
	finish := statusToFinish(jsonx.GetString(body, "status"), len(toolCalls) > 0)
	id := jsonx.GetString(body, "id")
	created := body["created_at"]
	if created == nil {
		created = body["created"]
	}
	if created == nil {
		created = 0
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   jsonx.GetString(body, "model"),
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       msg,
				"finish_reason": finish,
			},
		},
		"usage": responsesUsageToChat(body["usage"]),
	}
}

func chatChoicesToResponsesOutput(choices []any) []any {
	var out []any
	for _, ch := range choices {
		cm, ok := jsonx.AsMap(ch)
		if !ok {
			continue
		}
		msg, _ := jsonx.AsMap(cm["message"])
		if msg == nil {
			continue
		}
		if rc := jsonx.GetString(msg, "reasoning_content"); rc != "" {
			out = append(out, map[string]any{
				"type":    "reasoning",
				"summary": []any{map[string]any{"type": "summary_text", "text": rc}},
			})
		}
		if tbs, ok := jsonx.AsSlice(msg["thinking_blocks"]); ok {
			var summary []any
			for _, tb := range tbs {
				if tm, ok := jsonx.AsMap(tb); ok {
					if t := jsonx.GetString(tm, "thinking"); t != "" {
						summary = append(summary, map[string]any{"type": "summary_text", "text": t})
					}
				}
			}
			if len(summary) > 0 {
				out = append(out, map[string]any{"type": "reasoning", "summary": summary})
			}
		}
		if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
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
					name = jsonx.GetString(fn, "name")
				}
				out = append(out, map[string]any{
					"type":      "function_call",
					"call_id":   jsonx.GetString(tm, "id"),
					"name":      name,
					"arguments": args,
				})
			}
		}
		text := contentToPlain(msg["content"])
		if text != "" {
			out = append(out, map[string]any{
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": []any{map[string]any{"type": "output_text", "text": text}},
			})
		}
	}
	if out == nil {
		out = []any{}
	}
	return out
}

func finishToStatus(fr string) string {
	switch fr {
	case "length":
		return "incomplete"
	case "tool_calls":
		return "completed"
	default:
		return "completed"
	}
}

func statusToFinish(status string, hasTools bool) string {
	if hasTools {
		return "tool_calls"
	}
	if status == "incomplete" {
		return "length"
	}
	return "stop"
}

func chatUsageToResponses(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	}
	in, _ := jsonx.Int(m["prompt_tokens"])
	if in == 0 {
		in, _ = jsonx.Int(m["input_tokens"])
	}
	out, _ := jsonx.Int(m["completion_tokens"])
	if out == 0 {
		out, _ = jsonx.Int(m["output_tokens"])
	}
	total, _ := jsonx.Int(m["total_tokens"])
	if total == 0 {
		total = in + out
	}
	u := map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": total}
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		u["input_tokens_details"] = d
	}
	return u
}

func responsesUsageToChat(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	}
	in, _ := jsonx.Int(m["input_tokens"])
	out, _ := jsonx.Int(m["output_tokens"])
	total, _ := jsonx.Int(m["total_tokens"])
	if total == 0 {
		total = in + out
	}
	return map[string]any{"prompt_tokens": in, "completion_tokens": out, "total_tokens": total}
}

func marshalJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
