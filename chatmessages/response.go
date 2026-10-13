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
				tm, ok := jsonx.AsMap(tb)
				if !ok {
					continue
				}
				// A thinking block with no text and no signature holds nothing
				// a client could need, and Anthropic rejects an empty one. A
				// signature-only block is a real response and is kept: the
				// client needs the signature to replay reasoning.
				if isEmptyUnsignedThinkingBlock(tm) {
					continue
				}
				content = append(content, tm)
			}
		} else if rc := jsonx.GetString(msg, "reasoning_content"); rc != "" {
			content = append(content, map[string]any{"type": "thinking", "thinking": rc})
		}
		if msg["content"] != nil {
			// A chat message's content is a string or a part list. A list must
			// be read part by part: rendering it with a generic formatter
			// produces Go syntax the model cannot read.
			if s := chatContentToText(msg["content"]); s != "" {
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

// chatContentToText renders a chat message's content for the Messages wire. A
// part list contributes only its text parts; anything else has no Messages
// equivalent here and is skipped rather than stringified.
func chatContentToText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, p := range c {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			if jsonx.GetString(pm, "type") == "text" {
				parts = append(parts, jsonx.GetString(pm, "text"))
			}
		}
		return strings.Join(parts, "")
	default:
		return ""
	}
}

// isEmptyUnsignedThinkingBlock reports a thinking block whose text is empty or
// whitespace-only and which carries no signature — a block with nothing a
// client could replay.
func isEmptyUnsignedThinkingBlock(block map[string]any) bool {
	if jsonx.GetString(block, "type") != "thinking" {
		return false
	}
	if strings.TrimSpace(jsonx.GetString(block, "thinking")) != "" {
		return false
	}
	return jsonx.GetString(block, "signature") == ""
}

func finishToStopReason(fr string) string {
	switch fr {
	case "stop", "":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "refusal"
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
	case "refusal", "content_filtered":
		return "content_filter"
	case "compaction":
		return "length"
	default:
		return "stop"
	}
}

// openaiUsageToAnthropic maps a chat usage object onto the Messages shape.
// OpenAI's prompt_tokens counts the whole prompt, cache reads and writes
// included, while Anthropic's input_tokens is the uncached remainder and
// reports the cache halves as fields of their own.
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
	cacheRead := 0
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		cacheRead, _ = jsonx.Int(d["cached_tokens"])
	}
	if c, ok := jsonx.Int(m["cache_read_input_tokens"]); ok && c > 0 {
		cacheRead = c
	}
	cacheWrite := 0
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		cacheWrite, _ = jsonx.Int(d["cache_creation_tokens"])
		if cacheWrite == 0 {
			cacheWrite, _ = jsonx.Int(d["cache_write_tokens"])
		}
	}
	if c, ok := jsonx.Int(m["cache_creation_input_tokens"]); ok && c > 0 {
		cacheWrite = c
	}
	u := map[string]any{"input_tokens": max(in-cacheRead-cacheWrite, 0), "output_tokens": out}
	if cacheRead > 0 {
		u["cache_read_input_tokens"] = cacheRead
	}
	if cacheWrite > 0 {
		u["cache_creation_input_tokens"] = cacheWrite
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
	details := map[string]any{}
	if cacheRead > 0 {
		details["cached_tokens"] = cacheRead
	}
	if cacheWrite > 0 {
		details["cache_creation_tokens"] = cacheWrite
	}
	if len(details) > 0 {
		u["prompt_tokens_details"] = details
	}
	return u
}
