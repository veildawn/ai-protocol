package stream

import (
	"bytes"
	"encoding/json"
	"strings"
)

// EventKind is the protocol-level meaning of one decoded public stream event.
// It intentionally describes facts, not host policy such as retrying, billing,
// or routing.
type EventKind uint8

const (
	// EventUnknown means the event does not match a known public shape. Callers
	// must retain it rather than treating it as proof of output or failure.
	EventUnknown EventKind = iota
	// EventPrelude is framing, metadata, a keepalive, or an empty content shell.
	EventPrelude
	// EventContent carries non-empty user-facing content or an actionable tool
	// call. A caller that has exposed it to a client must not silently switch the
	// upstream attempt.
	EventContent
	// EventReasoning carries non-empty reasoning rather than an empty reasoning
	// shell or an opaque signature by itself.
	EventReasoning
	// EventComplete is an explicit successful terminal event.
	EventComplete
	// EventIncomplete is an explicit non-success terminal event. Whether the
	// host can retry it depends on whether anything was already delivered.
	EventIncomplete
	// EventFailure is an explicit in-band failure frame.
	EventFailure
)

// EventFacts describes a single decoded SSE event. EventFacts is deliberately
// small so providers, transport, and accounting policy remain outside this
// protocol package.
type EventFacts struct {
	Kind EventKind
	// HasUsage reports that the event carries a non-empty usage object. Usage
	// metadata is not itself user-visible content, but may establish that a
	// terminal Chat sentinel closed a valid zero-text response.
	HasUsage bool
	// Terminal reports whether this wire event is a terminal frame in its
	// source dialect. A Chat finish_reason is a logical completion fact but is
	// not itself the Chat stream's [DONE] terminal.
	Terminal bool
}

// ClassifyEvent returns protocol-level facts for one public event. The dialect
// is explicit because event names and payload shapes are not globally unique.
// An empty dialect uses the existing event inference for compatibility.
func ClassifyEvent(d Dialect, ev Event) EventFacts {
	if strings.EqualFold(ev.Event, "ping") {
		return EventFacts{Kind: EventPrelude}
	}
	if strings.TrimSpace(ev.Data) == "" {
		return EventFacts{Kind: EventPrelude}
	}
	if IsDone(ev.Data) {
		if d == "" || d == Chat {
			return EventFacts{Kind: EventComplete, Terminal: true}
		}
		return EventFacts{Kind: EventUnknown}
	}

	var obj map[string]json.RawMessage
	if json.Unmarshal([]byte(ev.Data), &obj) != nil || obj == nil {
		return EventFacts{Kind: EventUnknown}
	}

	typ := rawString(obj["type"])
	if typ == "" {
		typ = ev.Event
	}
	if d == "" {
		d = inferEventDialect(typ)
		if d == "" {
			if _, ok := obj["error"]; ok {
				d = Chat
			}
		}
	}
	if isFailureEvent(typ, d, obj) {
		return EventFacts{Kind: EventFailure, Terminal: true}
	}

	var facts EventFacts
	switch d {
	case Chat:
		facts = classifyChatEvent(obj)
	case Messages:
		facts = classifyMessagesEvent(typ, obj)
	case Responses:
		facts = classifyResponsesEvent(typ, obj)
	default:
		facts = EventFacts{Kind: EventUnknown}
	}
	facts.Terminal = facts.Kind == EventIncomplete ||
		(d == Messages && typ == "message_stop") ||
		(d == Responses && typ == "response.completed")
	return facts
}

func inferEventDialect(typ string) Dialect {
	switch {
	case typ == "message_start" || typ == "message_delta" || typ == "message_stop" ||
		typ == "content_block_start" || typ == "content_block_delta" || typ == "content_block_stop" ||
		strings.HasPrefix(typ, "message_") || strings.HasPrefix(typ, "content_block_"):
		return Messages
	case strings.HasPrefix(typ, "response."):
		return Responses
	default:
		return ""
	}
}

func isFailureEvent(typ string, d Dialect, obj map[string]json.RawMessage) bool {
	if typ == "error" || typ == "response.failed" {
		return true
	}
	if _, ok := obj["error"]; !ok {
		return false
	}
	return d == Chat
}

func classifyChatEvent(obj map[string]json.RawMessage) EventFacts {
	var chunk struct {
		Choices []struct {
			Delta        map[string]json.RawMessage `json:"delta"`
			FinishReason json.RawMessage            `json:"finish_reason"`
		} `json:"choices"`
	}
	if len(obj["choices"]) == 0 || json.Unmarshal(obj["choices"], &chunk.Choices) != nil {
		return EventFacts{Kind: EventUnknown}
	}
	if len(chunk.Choices) == 0 {
		// Usage-only chunks are useful evidence for the stream observer, but are
		// not by themselves client-visible output.
		return EventFacts{Kind: EventPrelude, HasUsage: hasJSONValue(obj["usage"])}
	}
	for _, choice := range chunk.Choices {
		if hasJSONValue(choice.FinishReason) && rawString(choice.FinishReason) != "" && rawString(choice.FinishReason) != "null" {
			return EventFacts{Kind: EventComplete}
		}
		for key, value := range choice.Delta {
			if key == "role" || key == "index" || !hasJSONValue(value) {
				continue
			}
			if key == "reasoning" || key == "reasoning_content" || key == "reasoning_details" || key == "reasoning_summary" {
				return EventFacts{Kind: EventReasoning}
			}
			if key == "tool_calls" || key == "function_call" {
				if hasActionableTool(value) {
					return EventFacts{Kind: EventContent}
				}
				continue
			}
			return EventFacts{Kind: EventContent}
		}
	}
	return EventFacts{Kind: EventPrelude}
}

func classifyMessagesEvent(typ string, obj map[string]json.RawMessage) EventFacts {
	switch typ {
	case "message_start", "content_block_stop":
		return EventFacts{Kind: EventPrelude}
	case "message_stop":
		return EventFacts{Kind: EventComplete}
	case "message_delta":
		var delta struct {
			StopReason string `json:"stop_reason"`
		}
		if json.Unmarshal(obj["delta"], &delta) == nil && delta.StopReason != "" {
			return EventFacts{Kind: EventComplete}
		}
		return EventFacts{Kind: EventPrelude}
	case "content_block_start":
		block := rawObject(obj["content_block"])
		if block == nil {
			return EventFacts{Kind: EventUnknown}
		}
		switch rawString(block["type"]) {
		case "thinking", "redacted_thinking":
			if hasJSONValue(block["thinking"]) {
				return EventFacts{Kind: EventReasoning}
			}
			return EventFacts{Kind: EventPrelude}
		case "text":
			if hasJSONValue(block["text"]) {
				return EventFacts{Kind: EventContent}
			}
			return EventFacts{Kind: EventPrelude}
		case "tool_use":
			if hasJSONValue(block["name"]) || hasJSONValue(block["input"]) {
				return EventFacts{Kind: EventContent}
			}
			return EventFacts{Kind: EventPrelude}
		default:
			if hasVisibleField(block) {
				return EventFacts{Kind: EventContent}
			}
			return EventFacts{Kind: EventPrelude}
		}
	case "content_block_delta":
		delta := rawObject(obj["delta"])
		if delta == nil {
			return EventFacts{Kind: EventUnknown}
		}
		switch {
		case hasJSONValue(delta["thinking"]), hasJSONValue(delta["redacted_thinking"]):
			return EventFacts{Kind: EventReasoning}
		case hasJSONValue(delta["text"]), hasJSONValue(delta["partial_json"]), hasJSONValue(delta["data"]):
			return EventFacts{Kind: EventContent}
		case delta["type"] != nil && (rawString(delta["type"]) == "signature_delta" || rawString(delta["type"]) == "signature"):
			return EventFacts{Kind: EventPrelude}
		case rawString(delta["type"]) == "thinking_delta":
			return EventFacts{Kind: EventPrelude}
		case hasVisibleField(delta):
			return EventFacts{Kind: EventContent}
		default:
			return EventFacts{Kind: EventPrelude}
		}
	default:
		return EventFacts{Kind: EventUnknown}
	}
}

func classifyResponsesEvent(typ string, obj map[string]json.RawMessage) EventFacts {
	switch typ {
	case "response.created", "response.in_progress", "response.queued", "response.metadata", "ping":
		return EventFacts{Kind: EventPrelude}
	case "response.completed":
		// Some relays close a ceiling cut under this name with status incomplete.
		if responsesStatus(obj) == "incomplete" {
			return EventFacts{Kind: EventIncomplete}
		}
		return EventFacts{Kind: EventComplete}
	case "response.incomplete":
		return EventFacts{Kind: EventIncomplete}
	case "response.output_text.delta", "response.refusal.delta", "response.audio.delta", "response.audio_transcript.delta":
		if hasJSONValue(obj["delta"]) {
			return EventFacts{Kind: EventContent}
		}
		return EventFacts{Kind: EventPrelude}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary.delta", "response.reasoning.delta":
		if hasJSONValue(obj["delta"]) {
			return EventFacts{Kind: EventReasoning}
		}
		return EventFacts{Kind: EventPrelude}
	case "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.mcp_call_arguments.delta", "response.mcp_call_arguments.done":
		if hasJSONValue(obj["delta"]) {
			return EventFacts{Kind: EventContent}
		}
		return EventFacts{Kind: EventPrelude}
	case "response.output_item.added", "response.output_item.done", "response.content_part.added", "response.content_part.done":
		return classifyResponsesPart(obj)
	default:
		return EventFacts{Kind: EventUnknown}
	}
}

// responsesStatus reads the explicit response.status of a Responses terminal
// event, or "" when the payload carries none. Both the top-level and the
// nested response object are checked because relays differ on which they
// populate.
func responsesStatus(obj map[string]json.RawMessage) string {
	if s := rawString(obj["status"]); s != "" {
		return s
	}
	if resp := rawObject(obj["response"]); len(resp) > 0 {
		return rawString(resp["status"])
	}
	return ""
}

func classifyResponsesPart(obj map[string]json.RawMessage) EventFacts {
	item := rawObject(obj["item"])
	if len(item) == 0 {
		item = rawObject(obj["part"])
	}
	if len(item) == 0 {
		return EventFacts{Kind: EventUnknown}
	}
	typ := rawString(item["type"])
	switch typ {
	case "function_call", "mcp_call", "computer_call", "web_search_call", "image_generation_call":
		if hasJSONValue(item["name"]) || hasJSONValue(item["arguments"]) || hasJSONValue(item["input"]) {
			return EventFacts{Kind: EventContent}
		}
		return EventFacts{Kind: EventPrelude}
	case "reasoning":
		if hasJSONValue(item["summary"]) || hasJSONValue(item["content"]) {
			return EventFacts{Kind: EventReasoning}
		}
		return EventFacts{Kind: EventPrelude}
	case "message":
		var content []map[string]json.RawMessage
		if json.Unmarshal(item["content"], &content) == nil {
			for _, part := range content {
				switch rawString(part["type"]) {
				case "reasoning", "thinking":
					if hasJSONValue(part["text"]) || hasJSONValue(part["summary"]) {
						return EventFacts{Kind: EventReasoning}
					}
				case "output_text", "text", "refusal":
					if hasJSONValue(part["text"]) {
						return EventFacts{Kind: EventContent}
					}
				case "image", "audio", "output_audio", "input_audio":
					if hasVisibleField(part) {
						return EventFacts{Kind: EventContent}
					}
				default:
					if hasVisibleField(part) {
						return EventFacts{Kind: EventContent}
					}
				}
			}
		}
		return EventFacts{Kind: EventPrelude}
	case "output_text", "refusal", "input_text", "input_image", "input_audio", "output_audio":
		if hasVisibleField(item) {
			return EventFacts{Kind: EventContent}
		}
		return EventFacts{Kind: EventPrelude}
	default:
		return EventFacts{Kind: EventUnknown}
	}
}

func hasActionableTool(value json.RawMessage) bool {
	var calls []map[string]json.RawMessage
	if json.Unmarshal(value, &calls) == nil {
		for _, call := range calls {
			if hasJSONValue(call["name"]) || hasJSONValue(call["arguments"]) || hasJSONValue(call["input"]) {
				return true
			}
			fn := rawObject(call["function"])
			if hasJSONValue(fn["name"]) || hasJSONValue(fn["arguments"]) {
				return true
			}
		}
		return false
	}
	var call map[string]json.RawMessage
	if json.Unmarshal(value, &call) != nil {
		return false
	}
	return hasJSONValue(call["name"]) || hasJSONValue(call["arguments"])
}

func hasVisibleField(fields map[string]json.RawMessage) bool {
	for key, value := range fields {
		switch key {
		case "id", "index", "type", "status", "role", "speaker", "object", "model", "created_at":
			continue
		}
		if hasJSONValue(value) {
			return true
		}
	}
	return false
}

func rawObject(value json.RawMessage) map[string]json.RawMessage {
	var obj map[string]json.RawMessage
	if json.Unmarshal(value, &obj) != nil {
		return nil
	}
	return obj
}

func rawString(value json.RawMessage) string {
	var s string
	if json.Unmarshal(value, &s) != nil {
		return ""
	}
	return s
}

func hasJSONValue(value json.RawMessage) bool {
	if len(value) == 0 || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return false
	}
	var s string
	if json.Unmarshal(value, &s) == nil {
		return s != ""
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(value, &fields) == nil {
		return len(fields) > 0
	}
	var items []json.RawMessage
	if json.Unmarshal(value, &items) == nil {
		return len(items) > 0
	}
	return true
}
