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
	restoreToolCallNames(msg, content)
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

// restoreToolCallNames puts each tool_use's own name back on the folded call.
// anthropicAssistantToOpenAI caps names at the chat wire's 64 characters for
// the request history, but a response must hand the client back the name the
// Messages body carries: the client matches its registered tools by name, and
// a hash-suffixed rename never matches. The reference does the same on this
// path (its reverse map is a no-op when nothing was rewritten).
// The calls come out in content order, so positions pair one to one.
func restoreToolCallNames(msg map[string]any, content []any) {
	calls, ok := jsonx.AsSlice(msg["tool_calls"])
	if !ok {
		return
	}
	var names []string
	for _, b := range content {
		if bm, ok := jsonx.AsMap(b); ok && jsonx.GetString(bm, "type") == "tool_use" {
			names = append(names, jsonx.GetString(bm, "name"))
		}
	}
	for i, c := range calls {
		if i >= len(names) {
			break
		}
		cm, ok := jsonx.AsMap(c)
		if !ok {
			continue
		}
		if fn, ok := jsonx.AsMap(cm["function"]); ok {
			fn["name"] = names[i]
		}
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
				content = append(content, rebuildThinkingBlock(tm))
			}
		} else if rc := jsonx.GetString(msg, "reasoning_content"); rc != "" {
			// The block states its signature even here — as a null, the
			// unsigned spelling on the Messages wire.
			content = append(content, map[string]any{"type": "thinking", "thinking": rc, "signature": nil})
		}
		if msg["content"] != nil {
			// A chat message's content is a string or a part list. A list must
			// be read part by part: rendering it with a generic formatter
			// produces Go syntax the model cannot read. A present-but-empty
			// string still folds to an empty text block — the reference emits
			// one whenever content is not null, and a client distinguishing
			// "" from absent reads the difference.
			if s := chatContentToText(msg["content"]); s != "" {
				content = append(content, map[string]any{"type": "text", "text": s})
			} else if _, isStr := msg["content"].(string); isStr {
				content = append(content, map[string]any{"type": "text", "text": ""})
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
					"id":    normalizeToolUseID(jsonx.GetString(tm, "id")),
					"name":  name,
					"input": input,
					// The Messages block shape states the field even when
					// nothing rides it.
					"provider_specific_fields": nil,
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

// rebuildThinkingBlock re-emits a chat thinking block in the fixed Messages
// shape: {type, thinking, signature} for a thinking block, {type, data} for a
// redacted one. Extra chat-side keys (a cache_control marker, say) do not ride
// along, and the signature is always stated — an absent one as "" (the block
// says "unsigned"), an explicit null as null.
func rebuildThinkingBlock(block map[string]any) map[string]any {
	if jsonx.GetString(block, "type") == "redacted_thinking" {
		return map[string]any{"type": "redacted_thinking", "data": jsonx.GetString(block, "data")}
	}
	sig, present := block["signature"]
	if !present {
		sig = ""
	}
	return map[string]any{
		"type":      "thinking",
		"thinking":  jsonx.GetString(block, "thinking"),
		"signature": sig,
	}
}

// thoughtSignatureSeparator marks a thought-signature suffix some chat
// clients append to a tool call id after this marker. The Messages id pattern
// cannot carry it, so the fold moves it off the id.
const thoughtSignatureSeparator = "__thought__"

// splitThoughtSignature splits a chat tool call id into its base and the
// thought-signature suffix, when one rides it.
func splitThoughtSignature(id string) (base, sig string) {
	if i := strings.Index(id, thoughtSignatureSeparator); i >= 0 {
		return id[:i], id[i+len(thoughtSignatureSeparator):]
	}
	return id, ""
}

// normalizeToolUseID makes a chat tool call id legal on the Messages wire,
// whose ids match ^[a-zA-Z0-9_-]+$: a thought-signature suffix is stripped
// first, every remaining character outside the class becomes '_', and an id
// with nothing left becomes "tool_use_id". Cross-provider clients replay ids
// like "functions.Bash:0"; the target would refuse them as they stand.
func normalizeToolUseID(id string) string {
	base, _ := splitThoughtSignature(id)
	var b strings.Builder
	b.Grow(len(base))
	for _, r := range base {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "tool_use_id"
	}
	return b.String()
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

// finishToStopReason maps a chat finish_reason onto the Messages wire.
// content_filter has no branch of its own: the reference folds it onto
// end_turn together with every other vendor-specific reason, and "refusal"
// would tell a Messages client the model declined when the upstream said the
// turn was filtered.
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
	// The count of server-tool (web search) requests rides its own usage
	// object on the Messages wire; the explicit spelling outranks the one
	// nested in the token details.
	webSearch := 0
	if stu, ok := jsonx.AsMap(m["server_tool_use"]); ok {
		webSearch, _ = jsonx.Int(stu["web_search_requests"])
	}
	if webSearch == 0 {
		if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
			webSearch, _ = jsonx.Int(d["web_search_requests"])
		}
	}
	u := map[string]any{"input_tokens": max(in-cacheRead-cacheWrite, 0), "output_tokens": out}
	if cacheRead > 0 {
		u["cache_read_input_tokens"] = cacheRead
	}
	if cacheWrite > 0 {
		u["cache_creation_input_tokens"] = cacheWrite
	}
	if webSearch > 0 {
		u["server_tool_use"] = map[string]any{"web_search_requests": webSearch}
	}
	return u
}

func anthropicUsageToOpenAI(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		m = map[string]any{}
	}
	in, _ := jsonx.Int(m["input_tokens"])
	out, _ := jsonx.Int(m["output_tokens"])
	cacheRead, _ := jsonx.Int(m["cache_read_input_tokens"])
	cacheWrite, _ := jsonx.Int(m["cache_creation_input_tokens"])
	// OpenAI prompt_tokens is inclusive of cache read/write; Anthropic
	// input_tokens is the uncached remainder.
	prompt := in + cacheRead + cacheWrite
	// Anthropic may state the thinking share of output_tokens under
	// output_tokens_details; the chat wire spells that split in
	// completion_tokens_details, capped at the reported total.
	reasoning := 0
	if d, ok := jsonx.AsMap(m["output_tokens_details"]); ok {
		if r, ok := jsonx.Int(d["thinking_tokens"]); ok && r > 0 {
			reasoning = min(r, out)
		}
	}
	// The details objects are stated unconditionally, zeros included: the
	// reference folds them on every response, and a client probing cache or
	// reasoning support reads zeros, not absent keys. cache_write_tokens
	// mirrors cache_creation_tokens, the reference's spelling.
	return map[string]any{
		"prompt_tokens":               prompt,
		"completion_tokens":           out,
		"total_tokens":                prompt + out,
		"prompt_tokens_details":       map[string]any{"cached_tokens": cacheRead, "cache_creation_tokens": cacheWrite, "cache_write_tokens": cacheWrite, "text_tokens": in},
		"cache_creation_input_tokens": cacheWrite,
		"cache_read_input_tokens":     cacheRead,
		"completion_tokens_details":   map[string]any{"reasoning_tokens": reasoning, "text_tokens": out - reasoning},
	}
}
