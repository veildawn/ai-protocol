package protocol

import (
	"encoding/json"
	"strings"
)

// Loss names a request feature that a conversion cannot preserve. Losses are
// descriptive protocol facts, never routing policy.
type Loss string

const (
	LossLogitBias        Loss = "logit_bias"
	LossLogprobs         Loss = "logprobs"
	LossMultiChoice      Loss = "multi_choice"
	LossSeed             Loss = "seed"
	LossPenaltyControls  Loss = "penalty_controls"
	LossFileReference    Loss = "file_reference"
	LossStructuredOutput Loss = "structured_output"
	LossToolArguments    Loss = "tool_arguments"
	LossUnknownContent   Loss = "unknown_content"
)

// LossSet is a set of losses keyed for cheap union and membership checks.
type LossSet map[Loss]bool

// Add returns the receiver with loss present, allocating lazily.
func (s LossSet) Add(losses ...Loss) LossSet {
	if s == nil {
		s = make(LossSet, len(losses))
	}
	for _, loss := range losses {
		s[loss] = true
	}
	return s
}

// Union returns a new set containing both operands' losses.
func (s LossSet) Union(other LossSet) LossSet {
	out := make(LossSet, len(s)+len(other))
	for loss := range s {
		out[loss] = true
	}
	for loss := range other {
		out[loss] = true
	}
	return out
}

// StaticLosses describes every loss possible on one directed conversion,
// independent of a particular request body.
func StaticLosses(from, to Dialect) LossSet {
	src, ok := Normalize(string(from))
	if !ok || src == to {
		return nil
	}
	dst, ok := Normalize(string(to))
	if !ok {
		return nil
	}
	out := LossSet{}.Add(LossLogitBias, LossLogprobs, LossMultiChoice, LossToolArguments, LossUnknownContent)
	switch {
	case src == Chat && dst == Messages:
		return out.Add(LossSeed, LossPenaltyControls, LossFileReference, LossStructuredOutput)
	case src == Responses && dst == Messages:
		return out.Add(LossFileReference, LossStructuredOutput)
	default:
		return out
	}
}

// BodyLosses inspects one request body and returns only the losses applicable
// to that body. opts controls the same omission policy as conversion.
func BodyLosses(from, to Dialect, body []byte, opts ConvertOptions) LossSet {
	src, ok := Normalize(string(from))
	if !ok || src == to {
		return nil
	}
	dst, ok := Normalize(string(to))
	if !ok {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return LossSet{}.Add(LossUnknownContent)
	}
	static := StaticLosses(src, dst)
	out := LossSet{}
	has := func(loss Loss) bool { return static[loss] }
	if has(LossLogitBias) && payloadPresent(payload, "logit_bias") {
		out.Add(LossLogitBias)
	}
	if has(LossLogprobs) && (payloadPresent(payload, "logprobs") || payloadPresent(payload, "top_logprobs")) {
		out.Add(LossLogprobs)
	}
	if has(LossMultiChoice) && number(payload["n"]) > 1 {
		out.Add(LossMultiChoice)
	}
	if has(LossSeed) && payloadPresent(payload, "seed") {
		out.Add(LossSeed)
	}
	if has(LossPenaltyControls) && (payloadPresent(payload, "presence_penalty") || payloadPresent(payload, "frequency_penalty")) {
		out.Add(LossPenaltyControls)
	}
	if has(LossFileReference) && bodyHasFileReference(payload, src) {
		out.Add(LossFileReference)
	}
	if has(LossStructuredOutput) && bodyHasStructuredOutput(payload, src) {
		out.Add(LossStructuredOutput)
	}
	if has(LossToolArguments) && bodyHasInvalidToolArguments(payload, src) {
		out.Add(LossToolArguments)
	}
	return out
}

func payloadPresent(payload map[string]any, key string) bool {
	value, ok := payload[key]
	return ok && value != nil
}

func number(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	default:
		return 0
	}
}

func bodyHasFileReference(payload map[string]any, dialect Dialect) bool {
	switch dialect {
	case Chat:
		return messagesHaveFileReference(payload["messages"])
	case Responses:
		return responsesHasFileReference(payload["input"])
	default:
		return false
	}
}

func messagesHaveFileReference(value any) bool {
	messages, ok := value.([]any)
	if !ok {
		return false
	}
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if msg == nil {
			continue
		}
		blocks, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, blockRaw := range blocks {
			block, _ := blockRaw.(map[string]any)
			if block == nil || block["type"] != "file" {
				continue
			}
			inner, _ := block["file"].(map[string]any)
			if inner != nil && strings.TrimSpace(stringField(inner["file_id"])) != "" {
				return true
			}
		}
	}
	return false
}

func responsesHasFileReference(value any) bool {
	items, ok := value.([]any)
	if !ok {
		return false
	}
	for _, raw := range items {
		item, _ := raw.(map[string]any)
		if item == nil {
			continue
		}
		blocks, ok := item["content"].([]any)
		if !ok {
			continue
		}
		for _, blockRaw := range blocks {
			block, _ := blockRaw.(map[string]any)
			if block != nil && block["type"] == "input_file" && strings.TrimSpace(stringField(block["file_id"])) != "" {
				return true
			}
		}
	}
	return false
}

func bodyHasStructuredOutput(payload map[string]any, dialect Dialect) bool {
	switch dialect {
	case Chat:
		rf, _ := payload["response_format"].(map[string]any)
		return rf != nil && (rf["type"] == "json_object" || rf["type"] == "json_schema")
	case Responses:
		text, _ := payload["text"].(map[string]any)
		format, _ := text["format"].(map[string]any)
		return format != nil && (format["type"] == "json_object" || format["type"] == "json_schema")
	default:
		return false
	}
}

func bodyHasInvalidToolArguments(payload map[string]any, dialect Dialect) bool {
	switch dialect {
	case Chat:
		messages, _ := payload["messages"].([]any)
		for _, raw := range messages {
			msg, _ := raw.(map[string]any)
			if msg == nil {
				continue
			}
			calls, _ := msg["tool_calls"].([]any)
			for _, callRaw := range calls {
				call, _ := callRaw.(map[string]any)
				fn, _ := call["function"].(map[string]any)
				if fn == nil {
					continue
				}
				var value any
				if err := json.Unmarshal([]byte(stringField(fn["arguments"])), &value); err != nil {
					return true
				}
				if _, ok := value.(map[string]any); !ok && value != nil {
					return true
				}
			}
		}
	case Responses:
		items, _ := payload["input"].([]any)
		for _, raw := range items {
			item, _ := raw.(map[string]any)
			if item == nil || item["type"] != "function_call" {
				continue
			}
			var value any
			if err := json.Unmarshal([]byte(stringField(item["arguments"])), &value); err != nil {
				return true
			}
			if _, ok := value.(map[string]any); !ok && value != nil {
				return true
			}
		}
	}
	return false
}

func stringField(v any) string {
	s, _ := v.(string)
	return s
}
