package messagesresponses

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

func ResponsesRequestToMessages(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	if inst := jsonx.GetString(body, "instructions"); inst != "" {
		out["system"] = inst
	}
	out["messages"] = inputToMessages(body["input"])
	if v, ok := body["max_output_tokens"]; ok {
		out["max_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	if tools, ok := body["tools"]; ok {
		out["tools"] = responsesToolsToMessages(tools)
	}
	if tc, ok := body["tool_choice"]; ok {
		out["tool_choice"] = responsesToolChoiceToMessages(tc)
	}
	if r, ok := jsonx.AsMap(body["reasoning"]); ok {
		if th := reasoningToThinking(r); th != nil {
			out["thinking"] = th
		}
	}
	if text, ok := jsonx.AsMap(body["text"]); ok {
		if fm, ok := jsonx.AsMap(text["format"]); ok && jsonx.GetString(fm, "type") == "json_schema" {
			out["output_format"] = map[string]any{
				"type":   "json_schema",
				"schema": fm["schema"],
				"strict": fm["strict"],
			}
		}
	}
	if uid := jsonx.GetString(body, "user"); uid != "" {
		out["metadata"] = map[string]any{"user_id": uid}
	}
	return out, nil
}

func inputToMessages(input any) []any {
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
		switch jsonx.GetString(m, "type") {
		case "function_call_output":
			msgs = append(msgs, map[string]any{
				"role": "user",
				"content": []any{map[string]any{
					"type":        "tool_result",
					"tool_use_id": jsonx.GetString(m, "call_id"),
					"content":     functionOutputToAnthropic(m["output"]),
				}},
			})
		case "function_call":
			var parsed any = map[string]any{}
			if args := jsonx.GetString(m, "arguments"); args != "" {
				_ = json.Unmarshal([]byte(args), &parsed)
			}
			msgs = append(msgs, map[string]any{
				"role": "assistant",
				"content": []any{map[string]any{
					"type":  "tool_use",
					"id":    jsonx.GetString(m, "call_id"),
					"name":  jsonx.GetString(m, "name"),
					"input": parsed,
				}},
			})
		case "reasoning":
			sig := jsonx.GetString(m, "encrypted_content")
			var blocks []any
			if sum, ok := jsonx.AsSlice(m["summary"]); ok {
				for _, p := range sum {
					text := jsonx.String(p)
					if pm, ok := jsonx.AsMap(p); ok {
						text = jsonx.GetString(pm, "text")
					}
					if text != "" {
						block := map[string]any{"type": "thinking", "thinking": text}
						if sig != "" {
							block["signature"] = sig
						}
						blocks = append(blocks, block)
					}
				}
			}
			if len(blocks) == 0 {
				// Reasoning the summary did not spell out still has to reach the
				// target, or the assistant turn it belongs to arrives empty.
				if sig == "" {
					continue
				}
				blocks = []any{map[string]any{"type": "redacted_thinking", "data": sig}}
			}
			// A reasoning item joins the assistant turn it precedes rather than
			// opening one of its own: two consecutive assistant messages are a
			// request the target rejects.
			if last, ok := trailingAssistant(msgs); ok {
				last["content"] = appendBlocks(last["content"], blocks)
				continue
			}
			msgs = append(msgs, map[string]any{"role": "assistant", "content": blocks})
		case "message", "":
			role := jsonx.GetString(m, "role")
			if role == "" {
				role = "user"
			}
			if role == "system" || role == "developer" {
				msgs = append(msgs, map[string]any{"role": "user", "content": partsText(m["content"])})
				continue
			}
			if role == "assistant" {
				blocks := outputPartsToAnthropic(m["content"])
				// An assistant message the previous reasoning item opened a turn
				// for joins that turn. Two consecutive assistant messages are a
				// request the target rejects.
				if last, ok := trailingAssistant(msgs); ok {
					last["content"] = appendBlocks(last["content"], blocks)
					continue
				}
				msgs = append(msgs, map[string]any{"role": "assistant", "content": blocks})
				continue
			}
			msgs = append(msgs, map[string]any{"role": "user", "content": inputPartsToAnthropic(m["content"])})
		}
	}
	return msgs
}

// trailingAssistant returns the trailing message when it is an assistant turn —
// the turn a following reasoning item or function_call belongs to.
func trailingAssistant(msgs []any) (map[string]any, bool) {
	if len(msgs) == 0 {
		return nil, false
	}
	last, ok := jsonx.AsMap(msgs[len(msgs)-1])
	if !ok || jsonx.GetString(last, "role") != "assistant" {
		return nil, false
	}
	return last, true
}

// appendBlocks adds content to an assistant turn, promoting a bare string to a
// text block so blocks have somewhere to go.
func appendBlocks(content, add any) []any {
	var out []any
	switch c := content.(type) {
	case string:
		if c != "" {
			out = append(out, map[string]any{"type": "text", "text": c})
		}
	case []any:
		out = append(out, c...)
	}
	switch a := add.(type) {
	case string:
		if a != "" {
			out = append(out, map[string]any{"type": "text", "text": a})
		}
	case []any:
		out = append(out, a...)
	}
	return out
}

// functionOutputToAnthropic renders a Responses function_call_output for an
// Anthropic tool_result. A plain string stays a string; a part list becomes a
// block list, and anything else is re-encoded as JSON so the bytes stay
// parseable rather than rendering as Go syntax.
func functionOutputToAnthropic(v any) any {
	if s, ok := v.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(v)
	if !ok {
		if v == nil {
			return ""
		}
		if b, err := jsonx.Marshal(v); err == nil {
			return string(b)
		}
		return ""
	}
	var blocks []any
	var texts []string
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			if b, err := jsonx.Marshal(p); err == nil {
				texts = append(texts, string(b))
				blocks = append(blocks, map[string]any{"type": "text", "text": string(b)})
			}
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "output_text", "text", "":
			t := jsonx.GetString(pm, "text")
			texts = append(texts, t)
			blocks = append(blocks, map[string]any{"type": "text", "text": t})
		case "input_image":
			url := jsonx.GetString(pm, "image_url")
			if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			if url != "" {
				blocks = append(blocks, inputImageToAnthropicBlock(url))
			}
		}
	}
	// Text-only output stays the flat string a tool result usually carries.
	if len(blocks) == len(texts) {
		return strings.Join(texts, "\n")
	}
	return blocks
}

// inputImageToAnthropicBlock builds an Anthropic image block from a URL or data
// URL.
func inputImageToAnthropicBlock(url string) map[string]any {
	block := map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": url}}
	if strings.HasPrefix(url, "data:") {
		media, data := splitDataURL(url)
		block["source"] = map[string]any{"type": "base64", "media_type": media, "data": data}
	}
	return block
}

func partsText(content any) any {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content
	}
	var texts []string
	for _, p := range list {
		if pm, ok := jsonx.AsMap(p); ok {
			texts = append(texts, jsonx.GetString(pm, "text"))
		}
	}
	if len(texts) == 1 {
		return texts[0]
	}
	return texts
}

func inputPartsToAnthropic(content any) any {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content
	}
	var blocks []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "text":
			blocks = append(blocks, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
		case "input_image":
			url := jsonx.GetString(pm, "image_url")
			blocks = append(blocks, inputImageToAnthropicBlock(url))
		}
	}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
			return jsonx.GetString(b, "text")
		}
	}
	return blocks
}

func outputPartsToAnthropic(content any) any {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content
	}
	var blocks []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		if jsonx.GetString(pm, "type") == "output_text" || jsonx.GetString(pm, "type") == "text" {
			blocks = append(blocks, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
		}
	}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok {
			return jsonx.GetString(b, "text")
		}
	}
	return blocks
}

func splitDataURL(url string) (media, data string) {
	media = "image/jpeg"
	rest := url[5:]
	if i := index(rest, ";base64,"); i >= 0 {
		if i > 0 {
			media = rest[:i]
		}
		data = rest[i+len(";base64,"):]
	}
	return media, data
}

func index(s, sep string) int {
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i:i+len(sep)] == sep {
			return i
		}
	}
	return -1
}

func responsesToolsToMessages(v any) []any {
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
		if jsonx.GetString(m, "type") == "web_search_preview" || jsonx.GetString(m, "type") == "web_search" {
			out = append(out, map[string]any{"type": "web_search_20260209", "name": "web_search"})
			continue
		}
		tool := map[string]any{"name": jsonx.GetString(m, "name")}
		if d := jsonx.GetString(m, "description"); d != "" {
			tool["description"] = d
		}
		if p, ok := m["parameters"]; ok {
			tool["input_schema"] = p
		}
		out = append(out, tool)
	}
	return out
}

func responsesToolChoiceToMessages(v any) any {
	switch t := v.(type) {
	case string:
		switch t {
		case "required":
			return map[string]any{"type": "any"}
		case "none":
			return map[string]any{"type": "none"}
		default:
			return map[string]any{"type": "auto"}
		}
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		if typ == "function" || typ == "custom" {
			// A custom choice nests its name under `custom`; a named choice
			// means "call this tool", and dropping the name would tell the
			// model to pick any tool instead.
			name := jsonx.GetString(t, "name")
			if name == "" {
				if nested, ok := jsonx.AsMap(t[typ]); ok {
					name = jsonx.GetString(nested, "name")
				}
			}
			if name != "" {
				return map[string]any{"type": "tool", "name": name}
			}
			return map[string]any{"type": "any"}
		}
	}
	return map[string]any{"type": "auto"}
}

func reasoningToThinking(r map[string]any) map[string]any {
	effort := jsonx.GetString(r, "effort")
	if effort == "" {
		return nil
	}
	budget, ok := types.BudgetFromEffort(effort)
	if !ok || effort == "none" {
		return nil
	}
	// Only type and budget_tokens: the Messages thinking object has no
	// summary member, and carrying the Responses-only r.summary here draws
	// "thinking.enabled.summary: Extra inputs are not permitted" from the
	// upstream. Summary is a Responses-API concept with no Messages
	// equivalent, so the faithful fold drops it.
	return map[string]any{"type": "enabled", "budget_tokens": budget}
}
