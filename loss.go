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
	// LossNativeParam names a source parameter (or tool form) that only the
	// source wire defines: the target request body has no spelling for it, so
	// it is dropped rather than carried in a shape the target rejects.
	LossNativeParam Loss = "native_param"
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
	// The universal predicates self-filter on the body, so they are always in.
	// The baseline beyond them is per-direction: the six pairs that do not
	// touch the generateContent wire keep their sets name-for-name, and the
	// six that do carry only what that wire can express (logprobs and
	// candidate counts fold onto it, so they stop being losses there; see
	// geminiLosses for the rows that name what it cannot hold).
	out := LossSet{}.Add(LossToolArguments, LossUnknownContent)
	switch {
	case src == Chat && dst == Messages:
		// Structured output is NOT listed for chat->messages: both forms fold
		// (json_schema and json_object become output_format), matching the
		// reference, which keeps response_format for Anthropic targets.
		return out.Add(LossLogitBias, LossLogprobs, LossMultiChoice, LossSeed, LossPenaltyControls, LossFileReference)
	case src == Responses && dst == Messages:
		// Only the json_object body form is lost here (the fold has a branch
		// for json_schema alone); see bodyHasStructuredOutput.
		return out.Add(LossLogitBias, LossLogprobs, LossMultiChoice, LossFileReference, LossStructuredOutput)
	// The images wire pairs only with the generateContent wire, so its branch
	// must be tested first: every images direction also matches the Gemini
	// test below and would otherwise fall into that switch's rows.
	case src == Images || dst == Images:
		return out.Add(imagesLosses(src, dst)...)
	case src == Gemini || dst == Gemini:
		return out.Add(geminiLosses(src, dst)...)
	default:
		return out.Add(LossLogitBias, LossLogprobs, LossMultiChoice)
	}
}

// imagesLosses lists what one conversion touching the images wire can lose
// beyond the two universal predicates. The images request's own extras —
// quality, response_format, background, and the edits-only mask — have no
// spelling on the generateContent wire, so a body carrying any of them loses
// it in both refusal paths. On the way back, the wire's logprobs knobs have no
// images spelling; the candidate count does (it folds to n), so multi-choice
// is not a loss here.
func imagesLosses(src, dst Dialect) []Loss {
	switch {
	case src == Images && dst == Gemini:
		return []Loss{LossNativeParam}
	default: // src == Gemini && dst == Images
		return []Loss{LossLogprobs, LossNativeParam}
	}
}

// geminiLosses lists what one conversion touching the generateContent wire can
// lose beyond the two universal predicates. Logprobs and multi-choice ARE
// expressible there (they fold onto the wire's own response config fields), so
// they appear only where the target is Messages or Responses, whose loss rows
// already list them. Seed and penalty controls fold in both directions and are
// never lost; structured output and file references fold both ways too.
func geminiLosses(src, dst Dialect) []Loss {
	switch {
	case src == Chat && dst == Gemini:
		return []Loss{LossLogitBias, LossNativeParam}
	case src == Messages && dst == Gemini, src == Responses && dst == Gemini, src == Gemini && dst == Chat:
		return []Loss{LossNativeParam}
	default: // src == Gemini && (dst == Messages || dst == Responses)
		return []Loss{LossLogprobs, LossMultiChoice, LossNativeParam}
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
	if has(LossNativeParam) && bodyHasNativeParam(payload, src, dst) {
		out.Add(LossNativeParam)
	}
	return out
}

// bodyHasNativeParam reports whether the source body carries a key (or tool
// form) only the source wire defines. Key presence is the predicate: what the
// key folds to either lands somewhere on the target wire or it does not, and
// the rows here are the ones that do not.
func bodyHasNativeParam(payload map[string]any, src, dst Dialect) bool {
	present := func(keys ...string) bool {
		for _, key := range keys {
			if payloadPresent(payload, key) {
				return true
			}
		}
		return false
	}
	switch {
	case src == Chat && dst == Gemini:
		// The first five are refused without DropParams and dropped with it;
		// the rest are accepted but the target request builder never reads
		// them. Either way the feature does not survive.
		return present("function_call", "metadata", "prediction", "store",
			"context_management", "user", "stream_options", "timeout",
			"extra_headers", "speed", "cache_control")
	case src == Messages && dst == Gemini:
		// metadata.user_id folds to a chat parameter the builder ignores; the
		// rest of metadata has no landing at all. cache_control lands the same
		// nowhere; context_management is refused or dropped.
		return present("metadata", "cache_control", "context_management")
	case src == Responses && dst == Gemini:
		// reasoning.effort folds onto the wire's thinking config; the summary
		// it may carry does not, so a reasoning key is a native-param loss.
		return present("metadata", "user", "context_management", "reasoning")
	case src == Gemini && dst == Images:
		// The images wire can carry one thing the other targets cannot: the
		// generation image config folds to a size, so unlike the rows below it
		// is not a native-param loss on this direction.
		return geminiBodyHasNativeParamExceptImageConfig(payload)
	case src == Images:
		// The images parameters the generateContent wire cannot spell; the
		// edits-only mask rides the same refusal set either way.
		return present("quality", "response_format", "mask", "background")
	case src == Gemini:
		return geminiBodyHasNativeParam(payload)
	default:
		return false
	}
}

// geminiBodyHasNativeParam reports the generateContent body's native-only
// surface: safety and cache handles the other dialects have no parameter for,
// generation knobs with no counterpart, and tool forms beyond function
// declarations and hosted search.
func geminiBodyHasNativeParam(payload map[string]any) bool {
	if payloadPresent(payload, "safetySettings") || payloadPresent(payload, "cachedContent") {
		return true
	}
	if gc, _ := payload["generationConfig"].(map[string]any); gc != nil {
		for _, key := range []string{"speechConfig", "imageConfig", "responseFormat", "mediaResolution"} {
			if payloadPresent(gc, key) {
				return true
			}
		}
	}
	if tc, _ := payload["toolConfig"].(map[string]any); tc != nil {
		if payloadPresent(tc, "includeServerSideToolInvocations") {
			return true
		}
	}
	tools, _ := payload["tools"].([]any)
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		if tool == nil {
			continue
		}
		for key := range tool {
			if key != "function_declarations" && key != "googleSearch" {
				return true
			}
		}
	}
	return false
}

// geminiBodyHasNativeParamExceptImageConfig is the native-only surface as seen
// from the images target: the same rows as above minus generationConfig
// imageConfig, which folds to the images request's size.
func geminiBodyHasNativeParamExceptImageConfig(payload map[string]any) bool {
	if payloadPresent(payload, "safetySettings") || payloadPresent(payload, "cachedContent") {
		return true
	}
	if gc, _ := payload["generationConfig"].(map[string]any); gc != nil {
		for _, key := range []string{"speechConfig", "responseFormat", "mediaResolution"} {
			if payloadPresent(gc, key) {
				return true
			}
		}
	}
	if tc, _ := payload["toolConfig"].(map[string]any); tc != nil {
		if payloadPresent(tc, "includeServerSideToolInvocations") {
			return true
		}
	}
	tools, _ := payload["tools"].([]any)
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		if tool == nil {
			continue
		}
		for key := range tool {
			if key != "function_declarations" && key != "googleSearch" {
				return true
			}
		}
	}
	return false
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
			if block == nil || block["type"] != "input_file" {
				continue
			}
			// The fold drops every input_file part, so both spellings are a loss:
			// a referenced file_id and an inline file_data payload alike.
			if strings.TrimSpace(stringField(block["file_id"])) != "" || strings.TrimSpace(stringField(block["file_data"])) != "" {
				return true
			}
		}
	}
	return false
}

func bodyHasStructuredOutput(payload map[string]any, dialect Dialect) bool {
	switch dialect {
	case Chat:
		// Both response_format forms fold to output_format, so neither body
		// loses structured output; this branch reports nothing.
		return false
	case Responses:
		// A text.format json_schema folds to output_format and survives;
		// json_object has no branch in the fold and is dropped, so only that
		// form is a loss.
		text, _ := payload["text"].(map[string]any)
		format, _ := text["format"].(map[string]any)
		return format != nil && format["type"] == "json_object"
	default:
		return false
	}
}

func bodyHasInvalidToolArguments(payload map[string]any, dialect Dialect) bool {
	// Empty, missing, or whitespace-only arguments fold to input:{} — the
	// reference does the same — so only a value that is present AND neither a
	// JSON object nor parseable JSON is a loss.
	isInvalid := func(arguments string) bool {
		if strings.TrimSpace(arguments) == "" {
			return false
		}
		var value any
		if err := json.Unmarshal([]byte(arguments), &value); err != nil {
			return true
		}
		_, ok := value.(map[string]any)
		return !ok && value != nil
	}
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
				if isInvalid(stringField(fn["arguments"])) {
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
			if isInvalid(stringField(item["arguments"])) {
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
