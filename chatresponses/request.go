package chatresponses

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/dialectvocab"
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
	if len(input) == 0 {
		// The Responses API rejects an empty input, and a system-only request
		// would otherwise serialize `"input": null` — a body the upstream
		// refuses. Carry the instructions as a system-role input item instead.
		if instructions != "" {
			input = []any{map[string]any{
				"type":    "message",
				"role":    "system",
				"content": []any{map[string]any{"type": "input_text", "text": instructions}},
			}}
			instructions = ""
		} else {
			input = []any{}
		}
	}
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
			// "none" is a member of the effort vocabulary — it asks for no
			// reasoning, which is a request, not an absence of one.
			if t != "" {
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
	// Top-level instructions become their own leading system message. A
	// system-role input item stays where the caller put it: hoisting it to the
	// front reorders the conversation the caller wrote.
	if inst := jsonx.GetString(body, "instructions"); strings.TrimSpace(inst) != "" {
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
	// reasoning_effort is a scalar string field on the chat wire: only the
	// effort value carries over. reasoning.summary (OpenAI Responses clients
	// always set one, e.g. "auto") and every other reasoning.* key have no
	// chat equivalent, so they drop here.
	if r, ok := jsonx.AsMap(body["reasoning"]); ok {
		if e, ok := r["effort"]; ok {
			out["reasoning_effort"] = e
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
			inst = appendInstruction(inst, contentToPlain(m["content"]))
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
			// Order matters: a Responses replay expects the reasoning item
			// before the call it produced, so the assistant turn is emitted
			// reasoning -> message -> function_call.
			//
			// thinking_blocks is the fallback when reasoning_content is absent:
			// a turn that arrived over a surface carrying signed blocks still
			// has reasoning to replay, and dropping it loses the signature the
			// next turn needs.
			if rc := jsonx.GetString(m, "reasoning_content"); rc != "" {
				input = append(input, map[string]any{
					"type":    "reasoning",
					"summary": []any{map[string]any{"type": "summary_text", "text": rc}},
				})
			} else if item := reasoningItemFromThinkingBlocks(m["thinking_blocks"]); item != nil {
				input = append(input, item)
			}
			text := contentToPlain(m["content"])
			if text != "" {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": text}},
				})
			}
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
			continue
		}
		// user
		input = append(input, map[string]any{
			"type":    "message",
			"role":    "user",
			"content": chatContentToInputParts(m["content"]),
		})
	}
	// Multiple system turns concatenate with a space, the separator the
	// reference uses for a folded instruction string.
	return strings.Join(inst, " "), input
}

func appendInstruction(instructions []string, text string) []string {
	if text = strings.TrimSpace(text); text != "" {
		return append(instructions, text)
	}
	return instructions
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
			// A tool message must name the call it answers; without one the
			// upstream rejects the whole request, so an item with no call id
			// contributes nothing rather than an invalid turn.
			callID := jsonx.GetString(m, "call_id")
			if callID == "" {
				continue
			}
			msgs = append(msgs, map[string]any{
				"role":         "tool",
				"tool_call_id": callID,
				"content":      functionCallOutputContent(m["output"]),
			})
		case "function_call":
			call := map[string]any{
				"id":   jsonx.GetString(m, "call_id"),
				"type": "function",
				"function": map[string]any{
					"name":      jsonx.GetString(m, "name"),
					"arguments": jsonx.GetString(m, "arguments"),
				},
			}
			// The call belongs to the assistant turn it was emitted in. Folding
			// it into a separate assistant message splits one turn in two, and
			// a strict upstream rejects a replayed turn whose reasoning and
			// tool_calls are not carried by the same assistant message.
			if last, ok := lastAssistant(msgs); ok {
				if existing, ok := jsonx.AsSlice(last["tool_calls"]); ok {
					last["tool_calls"] = append(existing, call)
				} else {
					last["tool_calls"] = []any{call}
				}
				continue
			}
			msgs = append(msgs, map[string]any{
				"role":       "assistant",
				"tool_calls": []any{call},
			})
		case "reasoning":
			text := reasoningText(m)
			if text != "" {
				// Join the reasoning onto the adjacent assistant turn instead of
				// opening one of its own. A second reasoning item appends rather
				// than splitting the turn, which would strand the first item on
				// a message with no tool_calls of its own.
				if last, ok := lastAssistant(msgs); ok {
					last["reasoning_content"] = joinReasoning(jsonx.GetString(last, "reasoning_content"), text)
					continue
				}
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
			// A system-role item keeps its place and its role: the chat wire
			// takes a system message anywhere, so there is no reason to lift it
			// out of the turn order the caller wrote.
			if role == "system" || role == "developer" {
				if text := partsToText(m["content"], "input_text", "output_text", "text"); text != "" {
					msgs = append(msgs, map[string]any{"role": "system", "content": text})
				}
				continue
			}
			content, images := partsToChatContent(m["content"], role)
			_ = images
			// An assistant message whose text was split from its own reasoning
			// rejoins it. Only a message the previous turn left EMPTY is
			// absorbed: a second message with text of its own is a distinct
			// turn and dropping it would lose content.
			if role == "assistant" {
				if last, ok := lastAssistant(msgs); ok {
					if last["content"] == nil || last["content"] == "" {
						last["content"] = content
						continue
					}
				}
			}
			msg := map[string]any{"role": role, "content": content}
			msgs = append(msgs, msg)
		}
	}
	return msgs
}

// lastAssistant reports the trailing message when it is an assistant message —
// the message a following function_call, reasoning item or split text belongs to.
func lastAssistant(msgs []any) (map[string]any, bool) {
	if len(msgs) == 0 {
		return nil, false
	}
	last, ok := msgs[len(msgs)-1].(map[string]any)
	if !ok || jsonx.GetString(last, "role") != "assistant" {
		return nil, false
	}
	return last, true
}

// joinReasoning appends a later reasoning item to the text already carried, so
// several items in one turn read as one block rather than being concatenated
// without a separator.
func joinReasoning(prev, next string) string {
	if strings.TrimSpace(prev) == "" {
		return next
	}
	return prev + "\n" + next
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
			if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			img := map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
			if d := jsonx.GetString(pm, "detail"); d != "" {
				img["image_url"].(map[string]any)["detail"] = d
			}
			parts = append(parts, img)
		case "input_file":
			// A chat upstream reads a file's bytes as an image_url data URL;
			// dropping the part would lose the attachment entirely.
			if url := inputFileToDataURL(pm); url != "" {
				hasNonText = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
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
			parts = appendInstruction(parts, jsonx.GetString(pm, "text"))
		}
	}
	return strings.Join(parts, "\n")
}

// functionCallOutputContent renders a Responses function_call_output for the
// chat wire. A plain string output stays a string, and any other JSON value is
// re-encoded so the bytes stay parseable rather than rendering as Go syntax. A
// part list becomes a content-part array when it carries media — flattening an
// image to its JSON would lose it — and a joined string when it is text only.
func functionCallOutputContent(v any) any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		if s, ok := v.(string); ok {
			return s
		}
		if v == nil {
			return ""
		}
		if b, err := jsonx.Marshal(v); err == nil {
			return string(b)
		}
		return ""
	}
	var texts []string
	var parts []any
	hasMedia := false
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			if b, err := jsonx.Marshal(p); err == nil {
				texts = append(texts, string(b))
				parts = append(parts, map[string]any{"type": "text", "text": string(b)})
			}
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "output_text", "text", "":
			t := jsonx.GetString(pm, "text")
			texts = append(texts, t)
			parts = append(parts, map[string]any{"type": "text", "text": t})
		case "input_image", "image_url":
			url := jsonx.GetString(pm, "image_url")
			if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			if url != "" {
				hasMedia = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
		case "input_file":
			// The chat wire has no file part on a tool message; the bytes ride
			// a data URL, which is what a chat upstream reads for a PDF.
			if url := inputFileToDataURL(pm); url != "" {
				hasMedia = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
		}
	}
	if !hasMedia {
		return strings.Join(texts, "\n")
	}
	return parts
}

// inputFileToDataURL renders a Responses input_file part as a data URL for the
// chat wire, which carries a file's bytes as an image_url data URL.
func inputFileToDataURL(pm map[string]any) string {
	if data := jsonx.GetString(pm, "file_data"); data != "" {
		return data
	}
	return jsonx.GetString(pm, "file_url")
}

// reasoningItemFromThinkingBlocks builds a Responses reasoning item from a chat
// message's thinking_blocks, the fallback when reasoning_content is absent. A
// redacted block carries no readable text, so it contributes no summary part.
func reasoningItemFromThinkingBlocks(v any) map[string]any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var summary []any
	for _, raw := range list {
		bm, ok := jsonx.AsMap(raw)
		if !ok || jsonx.GetString(bm, "type") != "thinking" {
			continue
		}
		if t := jsonx.GetString(bm, "thinking"); t != "" {
			summary = append(summary, map[string]any{"type": "summary_text", "text": t})
		}
	}
	if len(summary) == 0 {
		return nil
	}
	return map[string]any{"type": "reasoning", "summary": summary}
}

// reasoningText reads the plaintext a Responses reasoning item carries, joining
// its summary parts the way the reference does.
func reasoningText(m map[string]any) string {
	if s := jsonx.GetString(m, "content"); s != "" {
		return s
	}
	if list, ok := jsonx.AsSlice(m["summary"]); ok {
		var parts []string
		for _, p := range list {
			if pm, ok := jsonx.AsMap(p); ok {
				if t := strings.TrimSpace(jsonx.GetString(pm, "text")); t != "" {
					parts = append(parts, t)
				}
			} else if s, ok := p.(string); ok && strings.TrimSpace(s) != "" {
				parts = append(parts, strings.TrimSpace(s))
			}
		}
		return strings.Join(parts, "\n")
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
				tool["description"] = jsonx.GetString(fn, "description")
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
		if jsonx.GetString(m, "type") == "custom" {
			if custom, ok := jsonx.AsMap(m["custom"]); ok {
				tool := map[string]any{"type": "custom", "name": jsonx.GetString(custom, "name")}
				if d := jsonx.GetString(custom, "description"); d != "" {
					tool["description"] = d
				}
				if f, ok := jsonx.AsMap(custom["format"]); ok {
					tool["format"] = customToolFormatToResponses(f)
				}
				out = append(out, tool)
				continue
			}
		}
		out = append(out, m)
	}
	return out
}

// customToolFormatToResponses unwraps a chat grammar object into the flat
// grammar shape the Responses API takes.
func customToolFormatToResponses(format map[string]any) map[string]any {
	if jsonx.GetString(format, "type") != "grammar" {
		return format
	}
	grammar, ok := jsonx.AsMap(format["grammar"])
	if !ok {
		return format
	}
	out := map[string]any{"type": "grammar"}
	if v, ok := grammar["definition"]; ok {
		out["definition"] = v
	}
	if v, ok := grammar["syntax"]; ok {
		out["syntax"] = v
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
			// A function tool needs an object schema; a provider rejects one
			// that carries no `type`, so an absent or empty parameters object
			// becomes `{"type":"object"}`.
			params, ok := jsonx.AsMap(m["parameters"])
			if !ok || params["type"] == nil {
				if !ok {
					params = map[string]any{}
				}
				params["type"] = "object"
			}
			fn["parameters"] = params
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
		// A Responses-only tool has no chat equivalent, and forwarding it makes
		// the upstream reject the whole request for a missing `function`. It is
		// dropped rather than passed through.
		if dialectvocab.IsResponsesOnlyTool(typ) {
			continue
		}
		out = append(out, m)
	}
	return out
}

// chatToolChoiceToResponses maps a chat tool_choice to the Responses shape.
// Chat nests a named choice under `function`/`custom`; Responses takes the name
// at the top level, so the nested object is flattened rather than forwarded.
func chatToolChoiceToResponses(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto", "none", "required":
			return typ
		case "function", "custom":
			if name := jsonx.GetString(t, "name"); name != "" {
				return map[string]any{"type": typ, "name": name}
			}
			if nested, ok := jsonx.AsMap(t[typ]); ok {
				if name := jsonx.GetString(nested, "name"); name != "" {
					return map[string]any{"type": typ, "name": name}
				}
			}
		}
		return t
	default:
		return v
	}
}

// responsesToolChoiceToChat maps a Responses tool_choice to the chat shape.
// Chat nests a named choice under `function`, and a choice that names no tool
// means "any tool" — the chat spelling for that is "required".
func responsesToolChoiceToChat(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		// A choice that already carries a chat-shaped function name is
		// forwarded untouched.
		if fn, ok := jsonx.AsMap(t["function"]); ok && jsonx.GetString(fn, "name") != "" {
			return t
		}
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto", "none", "required":
			return typ
		case "tool", "any":
			return "required"
		case "function", "custom":
			name := jsonx.GetString(t, "name")
			if name == "" {
				if nested, ok := jsonx.AsMap(t[typ]); ok {
					name = jsonx.GetString(nested, "name")
				}
			}
			if name != "" {
				return map[string]any{"type": "function", "function": map[string]any{"name": name}}
			}
			return "required"
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
	case "text":
		// An explicit "no structured output" request; the Responses spelling is
		// a text format rather than an absent one.
		return map[string]any{"format": map[string]any{"type": "text"}}
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
		js := map[string]any{
			"name":   first(jsonx.GetString(format, "name"), "response_schema"),
			"schema": format["schema"],
		}
		// A missing strict is stated as false rather than left null, which is
		// not a value the schema takes.
		if st, ok := jsonx.Bool(format["strict"]); ok {
			js["strict"] = st
		} else {
			js["strict"] = false
		}
		return map[string]any{"type": "json_schema", "json_schema": js}
	case "json_object":
		return map[string]any{"type": "json_object"}
	case "text":
		return map[string]any{"type": "text"}
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
