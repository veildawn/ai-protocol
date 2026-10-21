package chatresponses

import (
	"encoding/json"
	"fmt"

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
	status := finishToStatus(finish)
	// tool_choice/tools/parallel_tool_calls are required members of a Response
	// object — a strict client (the OpenAI SDK's Response model) rejects the
	// body without them. The defaults are the reference's own (auto, none,
	// off); the request's real tool list is request context this pure fold
	// never sees.
	out := map[string]any{
		"id":                  jsonx.GetString(body, "id"),
		"created_at":          created,
		"model":               jsonx.GetString(body, "model"),
		"object":              "response",
		"output":              output,
		"parallel_tool_calls": false,
		"status":              status,
		"text":                map[string]any{},
		"tool_choice":         "auto",
		"tools":               []any{},
		"usage":               chatUsageToResponses(body["usage"]),
	}
	if status == "incomplete" {
		out["incomplete_details"] = map[string]any{"reason": incompleteReason(finish)}
	}
	return out
}

func ResponsesResponseToChat(body map[string]any) map[string]any {
	output, _ := jsonx.AsSlice(body["output"])
	// Each message item becomes its own choice, the way the reference folds
	// (_convert_response_output_to_choices appends one per ResponseOutputMessage,
	// index counting up); accumulated tool calls close the turn as one trailing
	// choice. A reasoning item's text attaches to the choice that follows it
	// and is consumed — each item overwrites the field rather than joining, so
	// an earlier chain is dropped, not run together with a later one.
	var choices []any
	var toolCalls []any
	reasoning := ""
	nextIndex := 0
	for _, item := range output {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "type") {
		case "message":
			role := jsonx.GetString(m, "role")
			if role == "" {
				role = "assistant"
			}
			msg := map[string]any{
				"role":    role,
				"content": partsToText(m["content"], "output_text", "text"),
			}
			// Citations ride the chat wire in message.annotations — the
			// reference keeps them as-is (_convert_annotations_to_chat_format)
			// — and dropping them loses the provenance a cited answer owes the
			// client.
			if anns := partsAnnotations(m["content"]); len(anns) > 0 {
				msg["annotations"] = anns
			}
			if reasoning != "" {
				msg["reasoning_content"] = reasoning
				reasoning = ""
			}
			choices = append(choices, map[string]any{
				"index":   nextIndex,
				"message": msg,
			})
			nextIndex++
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
		case "custom_tool_call":
			// A custom call carries its freeform payload in `input`; the chat
			// wire reads it as arguments, and dropping the item would lose the
			// call the model made.
			toolCalls = append(toolCalls, map[string]any{
				"id":   first(jsonx.GetString(m, "call_id"), jsonx.GetString(m, "id")),
				"type": "function",
				"function": map[string]any{
					"name":      jsonx.GetString(m, "name"),
					"arguments": jsonx.GetString(m, "input"),
				},
			})
		case "reasoning":
			reasoning = reasoningText(m)
		}
	}
	if len(toolCalls) > 0 {
		toolMsg := map[string]any{"role": "assistant", "content": nil, "tool_calls": toolCalls}
		if reasoning != "" {
			toolMsg["reasoning_content"] = reasoning
			reasoning = ""
		}
		choices = append(choices, map[string]any{
			"index":   nextIndex,
			"message": toolMsg,
		})
	} else if len(choices) == 0 {
		// Nothing the fold maps: still one assistant turn, so a chat client
		// that expects a choice has one to read.
		choices = append(choices, map[string]any{
			"index":   0,
			"message": map[string]any{"role": "assistant", "content": nil},
		})
	}
	detailReason := ""
	if d, ok := jsonx.AsMap(body["incomplete_details"]); ok {
		detailReason = jsonx.GetString(d, "reason")
	}
	// An incomplete turn rewrites EVERY choice's finish_reason — a ceiling cut
	// that happened to contain a call is still a ceiling cut. A completed turn
	// closes its message choices with "stop" and the tool-call choice with
	// "tool_calls".
	status := jsonx.GetString(body, "status")
	toolFinish := statusToFinish(status, len(toolCalls) > 0, detailReason)
	msgFinish := "stop"
	if status == "incomplete" {
		msgFinish = toolFinish
	}
	for _, ch := range choices {
		cm := ch.(map[string]any)
		if cm["message"].(map[string]any)["tool_calls"] != nil {
			cm["finish_reason"] = toolFinish
		} else {
			cm["finish_reason"] = msgFinish
		}
	}
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
		"choices": choices,
		"usage":   responsesUsageToChat(body["usage"]),
	}
}

func chatChoicesToResponsesOutput(choices []any) []any {
	var out []any
	// Item ids are minted here the way the streaming path mints them, so a
	// non-streamed body and the same body streamed agree on the ids. The
	// Responses schema requires `id` on every output item, and a client that
	// replays this output rejects an item without one.
	seq := 0
	nextID := func(prefix string) string {
		id := fmt.Sprintf("%s_%d", prefix, seq)
		seq++
		return id
	}
	for _, ch := range choices {
		cm, ok := jsonx.AsMap(ch)
		if !ok {
			continue
		}
		msg, _ := jsonx.AsMap(cm["message"])
		if msg == nil {
			continue
		}
		finish := jsonx.GetString(cm, "finish_reason")
		// The reference's non-streaming fold is one reasoning item per choice,
		// carrying the visible reasoning text as an output_text part and every
		// replayable thinking block inside one encrypted_content JSON array
		// (_extract_reasoning_output_items + _encode_thinking_blocks). Its own
		// streaming path keeps the summary shape instead; this body follows
		// the non-streaming one.
		rc := jsonx.GetString(msg, "reasoning_content")
		enc := encodeThinkingBlocks(msg["thinking_blocks"])
		if rc != "" || enc != "" {
			content := []any{}
			if rc != "" {
				content = append(content, map[string]any{"type": "output_text", "text": rc, "annotations": []any{}})
			}
			item := map[string]any{
				"id":      nextID("rs"),
				"type":    "reasoning",
				"status":  finishToStatus(finish),
				"role":    "assistant",
				"content": content,
			}
			if enc != "" {
				item["encrypted_content"] = enc
			}
			out = append(out, item)
		}
		// The assistant turn is emitted reasoning -> message -> function_call,
		// the order the Responses API itself uses, and a message item is folded
		// for EVERY choice — the reference emits one even when the turn is only
		// a tool call or only reasoning, with the text rendering null/empty
		// rather than the item vanishing. A function_call before the message it
		// was decided in reads as a call with no text turn, and a client that
		// replays this output against a strict target rejects the turn.
		var text any
		if msg["content"] != nil {
			text = contentToPlain(msg["content"])
		}
		// The part carries its annotations member — an empty list when there
		// are none, the way the reference shapes it and a strict client
		// (the OpenAI SDK's ResponseOutputText) requires.
		out = append(out, map[string]any{
			"id":   nextID("msg"),
			"type": "message",
			"role": "assistant",
			// A message the finish_reason cut short is an incomplete item,
			// the way the reference maps it
			// (_map_chat_completion_finish_reason_to_responses_status). A
			// function_call item keeps "completed" — the reference derives
			// that one from the call itself, not the finish.
			"status":  finishToStatus(finish),
			"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
		})
		if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
			for _, tc := range tcs {
				tm, ok := jsonx.AsMap(tc)
				if !ok {
					continue
				}
				fn, _ := jsonx.AsMap(tm["function"])
				args := ""
				name := ""
				if fn != nil {
					args = jsonx.GetString(fn, "arguments")
					name = jsonx.GetString(fn, "name")
				}
				// The item id IS the call id — the reference puts the chat
				// tool_call id in both fields. A chat id that is missing still
				// needs some id: the Responses schema requires one on every
				// output item, so a minted id is the fallback.
				callID := jsonx.GetString(tm, "id")
				if callID == "" {
					callID = nextID("fc")
				}
				out = append(out, map[string]any{
					"id":        callID,
					"type":      "function_call",
					"call_id":   callID,
					"name":      name,
					"arguments": args,
					"status":    "completed",
				})
			}
		}
	}
	if out == nil {
		out = []any{}
	}
	return out
}

// partsAnnotations collects the annotation lists a message item's content
// parts carry. They pass through unchanged — the two dialects spell them the
// same — so what the model cited survives the fold onto the chat wire.
func partsAnnotations(content any) []any {
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return nil
	}
	var out []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok || jsonx.GetString(pm, "type") != "output_text" {
			continue
		}
		anns, ok := jsonx.AsSlice(pm["annotations"])
		if !ok {
			continue
		}
		for _, a := range anns {
			if am, ok := jsonx.AsMap(a); ok {
				out = append(out, am)
			}
		}
	}
	return out
}

// encodeThinkingBlocks renders the thinking blocks that carry replay material
// — a signature or raw data — as one compact JSON array: the encrypted_content
// the reference attaches to a reasoning item (_encode_thinking_blocks). Blocks
// with neither are left out, and nothing to preserve means no field.
func encodeThinkingBlocks(v any) string {
	tbs, ok := jsonx.AsSlice(v)
	if !ok {
		return ""
	}
	var preserved []any
	for _, tb := range tbs {
		tm, ok := jsonx.AsMap(tb)
		if !ok {
			continue
		}
		if truthyField(tm, "signature") || truthyField(tm, "data") {
			preserved = append(preserved, tm)
		}
	}
	if len(preserved) == 0 {
		return ""
	}
	b, err := json.Marshal(preserved)
	if err != nil {
		return ""
	}
	return string(b)
}

// truthyField reports whether a map field is present with a non-empty value,
// the check the reference's `block.get(...) or ...` performs.
func truthyField(m map[string]any, key string) bool {
	v, ok := m[key]
	if !ok || v == nil {
		return false
	}
	if s, ok := v.(string); ok && s == "" {
		return false
	}
	return true
}

// incompleteReason is the Responses incomplete_details.reason for a chat
// finish_reason, or "" when that finish is a clean completion.
func incompleteReason(fr string) string {
	switch fr {
	case "length":
		return "max_output_tokens"
	case "content_filter":
		return "content_filter"
	case "refusal":
		// A refusal is the model declining, not a clean stop; reporting it as
		// completed would tell the client the turn finished normally.
		return "content_filter"
	default:
		return ""
	}
}

// finishToStatus maps a chat finish_reason onto a Responses item status the
// way the reference does
// (_map_chat_completion_finish_reason_to_responses_status): a length cut, a
// content filter and a refusal leave the item incomplete, everything else
// completes it.
func finishToStatus(fr string) string {
	if incompleteReason(fr) != "" {
		return "incomplete"
	}
	return "completed"
}

// statusToFinish maps a Responses status onto a chat finish_reason. An
// incomplete turn keeps its reason and outranks a tool call — the reference
// rewrites every choice's finish_reason for an incomplete response, the
// tool-call choice included (transform_response), because a ceiling cut that
// happened to contain a call is still a ceiling cut. A length cut and a content
// filter stay distinct for the same reason.
func statusToFinish(status string, hasTools bool, incompleteReason string) string {
	if status == "incomplete" {
		if incompleteReason == "content_filter" {
			return "content_filter"
		}
		return "length"
	}
	if hasTools {
		return "tool_calls"
	}
	return "stop"
}

// chatUsageToResponses folds a chat usage onto the Responses spellings. The
// reference reads only the chat keys — no input/output_tokens fallback — and
// passes the reported total through rather than recomputing it from the parts;
// the details objects are rebuilt on a key whitelist, and only when the source
// carries the object at all (an absent details field stays absent, one present
// gets its headline counter filled in as 0 rather than dropped).
func chatUsageToResponses(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	}
	in, _ := jsonx.Int(m["prompt_tokens"])
	out, _ := jsonx.Int(m["completion_tokens"])
	total, _ := jsonx.Int(m["total_tokens"])
	u := map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": total}
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		u["input_tokens_details"] = inputTokensDetails(d)
	}
	if d, ok := jsonx.AsMap(m["completion_tokens_details"]); ok {
		u["output_tokens_details"] = outputTokensDetails(d)
	}
	return u
}

// inputTokensDetails rebuilds the input breakdown: cached_tokens is always
// present once the details object is (0 when the source omits it); text,
// audio and cache-write counters ride only when the source carries them.
func inputTokensDetails(d map[string]any) map[string]any {
	out := map[string]any{"cached_tokens": 0}
	if v, ok := jsonx.Int(d["cached_tokens"]); ok {
		out["cached_tokens"] = v
	}
	for _, k := range []string{"text_tokens", "audio_tokens"} {
		if v, ok := jsonx.Int(d[k]); ok {
			out[k] = v
		}
	}
	cw := d["cache_write_tokens"]
	if cw == nil {
		cw = d["cache_creation_tokens"]
	}
	if v, ok := jsonx.Int(cw); ok {
		out["cache_write_tokens"] = v
	}
	return out
}

// outputTokensDetails rebuilds the output breakdown: reasoning_tokens is
// always present once the details object is (0 when the source omits it);
// audio, text and image counters ride only when the source carries them.
func outputTokensDetails(d map[string]any) map[string]any {
	out := map[string]any{"reasoning_tokens": 0}
	if v, ok := jsonx.Int(d["reasoning_tokens"]); ok {
		out["reasoning_tokens"] = v
	}
	for _, k := range []string{"audio_tokens", "text_tokens", "image_tokens"} {
		if v, ok := jsonx.Int(d[k]); ok {
			out[k] = v
		}
	}
	return out
}

// responsesUsageToChat folds a Responses usage onto the chat spellings. The
// total is always the sum of the parts — the reference discards a reported
// total rather than trusting a number that disagrees with its own parts.
func responsesUsageToChat(v any) map[string]any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	}
	in, _ := jsonx.Int(m["input_tokens"])
	out, _ := jsonx.Int(m["output_tokens"])
	u := map[string]any{"prompt_tokens": in, "completion_tokens": out, "total_tokens": in + out}
	if d, ok := jsonx.AsMap(m["input_tokens_details"]); ok {
		u["prompt_tokens_details"] = d
	}
	if d, ok := jsonx.AsMap(m["output_tokens_details"]); ok {
		u["completion_tokens_details"] = d
	}
	return u
}
