package messagesresponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// MessagesResponseToResponses folds an Anthropic Messages response into a
// Responses response, in the shape the v1.100.1 reference produces when its
// Anthropic adapter feeds a chat completion through the responses bridge:
// one merged reasoning item (when any thinking block carried text or opaque
// bytes), exactly one message item, then one function_call per tool_use and
// server_tool_use block, in content order.
func MessagesResponseToResponses(body map[string]any) map[string]any {
	content, _ := jsonx.AsSlice(body["content"])
	var thinkingBlocks []any // thinking + redacted_thinking, in content order
	var texts []string
	var calls []map[string]any // tool_use + server_tool_use, in content order
	for _, b := range content {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "thinking", "redacted_thinking":
			thinkingBlocks = append(thinkingBlocks, bm)
		case "text":
			texts = append(texts, jsonx.GetString(bm, "text"))
		case "tool_use", "server_tool_use":
			// A server-side call the model made is still a call: the reference
			// folds server_tool_use through the same tool_call path as
			// tool_use, and dropping it would lose the call from the turn.
			calls = append(calls, bm)
		}
	}
	status := stopReasonToStatus(jsonx.GetString(body, "stop_reason"))
	// Item ids are minted the way the streaming path mints them, so a
	// non-streamed body and the same body streamed agree on the shape.
	seq := 0
	nextID := func(prefix string) string {
		id := fmt.Sprintf("%s_%d", prefix, seq)
		seq++
		return id
	}
	var output []any
	if text, preserved := mergeThinking(thinkingBlocks); text != "" || len(preserved) > 0 {
		output = append(output, reasoningItem(text, preserved, nextID("rs"), status))
	}
	// The message item is emitted even when the turn carries no text at all:
	// its single output_text part then has no text, which is the shape a
	// tool-only turn takes on the wire.
	part := map[string]any{"type": "output_text", "annotations": []any{}}
	if text := strings.Join(texts, ""); text != "" {
		part["text"] = text
	}
	output = append(output, map[string]any{
		"id":      nextID("msg"),
		"type":    "message",
		"role":    "assistant",
		"status":  status,
		"content": []any{part},
	})
	for _, bm := range calls {
		args, _ := json.Marshal(bm["input"])
		if bm["input"] == nil {
			args = []byte("{}")
		}
		callID := jsonx.GetString(bm, "id")
		output = append(output, map[string]any{
			"type":      "function_call",
			"id":        callID,
			"call_id":   callID,
			"name":      jsonx.GetString(bm, "name"),
			"arguments": string(args),
			"status":    "completed",
		})
	}
	// The message item is always emitted, so the output is never empty here.
	usage := body["usage"]
	in, out, cacheRead, cacheCreate := 0, 0, 0, 0
	if um, ok := jsonx.AsMap(usage); ok {
		in, _ = jsonx.Int(um["input_tokens"])
		out, _ = jsonx.Int(um["output_tokens"])
		cacheRead, _ = jsonx.Int(um["cache_read_input_tokens"])
		cacheCreate, _ = jsonx.Int(um["cache_creation_input_tokens"])
	}
	// The reference's calculate_usage folds cache_creation and cache_read into
	// prompt_tokens before the responses usage is built, and the details
	// object is always present — cached_tokens is simply 0 without a cache.
	usageOut := map[string]any{
		"input_tokens":  in + cacheRead + cacheCreate,
		"output_tokens": out,
		"total_tokens":  in + cacheRead + cacheCreate + out,
		"input_tokens_details": map[string]any{
			"cached_tokens":      cacheRead,
			"text_tokens":        in,
			"cache_write_tokens": cacheCreate,
		},
	}
	return map[string]any{
		"id":     jsonx.GetString(body, "id"),
		"object": "response",
		"model":  jsonx.GetString(body, "model"),
		"status": status,
		"output": output,
		"usage":  usageOut,
	}
}

// stopReasonToStatus maps an Anthropic stop_reason onto the Responses status
// the way the reference does it in two hops (stop_reason → chat finish_reason
// → Responses status): end_turn, stop_sequence and tool_use complete the turn;
// max_tokens (length) and refusal (content_filter) leave it incomplete.
func stopReasonToStatus(stop string) string {
	switch stop {
	case "max_tokens", "refusal":
		return "incomplete"
	}
	return "completed"
}

// mergeThinking folds the turn's thinking and redacted_thinking blocks into
// the concatenated thinking text and the blocks that carry opaque bytes (a
// signature or redacted data) — the two things the reference's single merged
// reasoning item is built from.
func mergeThinking(blocks []any) (text string, preserved []any) {
	for _, b := range blocks {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		if t := jsonx.GetString(bm, "thinking"); t != "" {
			text += t
		}
		if jsonx.GetString(bm, "signature") != "" || jsonx.GetString(bm, "data") != "" {
			preserved = append(preserved, bm)
		}
	}
	return text, preserved
}

// reasoningItem builds the single merged reasoning item the reference emits:
// the concatenated thinking text rides one output_text part in content, and
// the preserved blocks are re-encoded as the compact JSON array
// encrypted_content is on the wire. Callers only emit it when the turn carried
// thinking text or opaque bytes — the reference emits no reasoning item
// otherwise.
func reasoningItem(text string, preserved []any, id, status string) map[string]any {
	item := map[string]any{
		"id":     id,
		"type":   "reasoning",
		"status": status,
		"role":   "assistant",
	}
	parts := []any{}
	if text != "" {
		parts = append(parts, map[string]any{"type": "output_text", "text": text, "annotations": []any{}})
	}
	item["content"] = parts
	if len(preserved) > 0 {
		if b, err := json.Marshal(preserved); err == nil {
			item["encrypted_content"] = string(b)
		}
	}
	return item
}
