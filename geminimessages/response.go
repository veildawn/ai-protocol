package geminimessages

// Response-direction folds between a gemini generateContent response body and
// an Anthropic Messages response. The forward direction (gemini to Messages)
// is the reference's composed fold, translated into one direct pass; the
// reverse direction inverts it field by field.

import (
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/internal/thinksig"
)

// GeminiResponseToMessages folds a gemini generateContent response into an
// Anthropic Messages response.
//
// The source body names no model, so the Messages model falls back to the
// package convention rather than an invented name; the id is the source's
// responseId, or empty when it stated none.
func GeminiResponseToMessages(body map[string]any) map[string]any {
	id := jsonx.GetString(body, "responseId")
	model := "unknown-model"

	// A blocked prompt or a flagged finish on the first candidate replaces the
	// whole response with one content-filter choice: no content blocks at all,
	// the usage read from the three totals alone, and the Messages dialect's
	// spelling of a filtered turn — which is end_turn, not a refusal.
	if blocked, ok := jsonx.AsMap(body["promptFeedback"]); ok && jsonx.GetString(blocked, "blockReason") != "" {
		return messagesBlockedResponse(id, model, usageMetadataOf(body))
	}
	candidates, _ := jsonx.AsSlice(body["candidates"])
	if len(candidates) > 0 {
		if c, ok := jsonx.AsMap(candidates[0]); ok {
			if _, flagged := flaggedFinishReasons[jsonx.GetString(c, "finishReason")]; flagged {
				return messagesBlockedResponse(id, model, usageMetadataOf(body))
			}
		}
	}

	var content []any
	sawToolCalls := false
	for _, raw := range candidates {
		c, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		blocks, hasCalls := geminiCandidateBlocks(c)
		sawToolCalls = sawToolCalls || hasCalls
		content = append(content, blocks...)
	}
	if content == nil {
		content = []any{}
	}

	stop := "end_turn"
	if len(candidates) > 0 {
		if c, ok := jsonx.AsMap(candidates[0]); ok {
			stop = geminiFinishToStopReason(jsonx.GetString(c, "finishReason"), firstCandidateHasCalls(c))
		}
	}
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"stop_sequence": nil,
		"usage":         geminiUsageToMessages(usageMetadataOf(body), body),
		"content":       content,
		"stop_reason":   stop,
	}
}

// messagesBlockedResponse is the shape a blocked or flagged turn takes: an
// empty content list and the three usage totals only — the reference builds
// that usage without the detail splits, so no cache or reasoning fields
// appear.
func messagesBlockedResponse(id, model string, u map[string]any) map[string]any {
	in, _ := jsonx.Int(u["promptTokenCount"])
	out, _ := jsonx.Int(u["candidatesTokenCount"])
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"stop_sequence": nil,
		"usage":         map[string]any{"input_tokens": in, "output_tokens": out},
		"content":       []any{},
		"stop_reason":   "end_turn",
	}
}

// flaggedFinishReasons are the finish reasons the reference replaces the whole
// response over: the candidate was filtered, so nothing it might have carried
// is delivered.
var flaggedFinishReasons = map[string]struct{}{
	"SAFETY": {}, "RECITATION": {}, "BLOCKLIST": {}, "PROHIBITED_CONTENT": {},
	"SPII": {}, "IMAGE_SAFETY": {}, "IMAGE_PROHIBITED_CONTENT": {},
}

// firstCandidateHasCalls reports whether the candidate carries a function
// call, which outranks the finish reason on the wire the fold lands on.
func firstCandidateHasCalls(c map[string]any) bool {
	_, hasCalls := geminiCandidateBlocks(c)
	return hasCalls
}

// geminiCandidateBlocks folds one candidate's parts into Messages blocks. The
// block order is the reference's own: thinking blocks first, then the text,
// then the tool calls. Media has three destinations on this fold — images and
// audio are dropped (the Messages response has no spelling for model-made
// media), while any other inline payload rides the text as a data URL, the
// same string the composed fold would have delivered.
func geminiCandidateBlocks(c map[string]any) (blocks []any, hasCalls bool) {
	content, _ := jsonx.AsMap(c["content"])
	parts, _ := jsonx.AsSlice(content["parts"])
	var thoughts []thoughtBlock
	var calls []any
	var text strings.Builder
	hasText := false
	callSeq := 0
	for _, raw := range parts {
		p, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		// The response wire spells these camelCase; the request wire is the
		// snake_case one.
		if fc, ok := jsonx.AsMap(p["functionCall"]); ok {
			hasCalls = true
			id := jsonx.GetString(fc, "id")
			sig := ""
			if id != "" {
				var base string
				base, sig = stripThoughtSignature(id)
				id = base
			}
			if s := jsonx.GetString(p, "thoughtSignature"); s != "" {
				sig = s
			}
			if id == "" {
				id = "call_" + itoa(callSeq)
			}
			callSeq++
			input := any(map[string]any{})
			if args, ok := fc["args"]; ok && args != nil {
				input = args
			}
			block := map[string]any{
				"type":  "tool_use",
				"id":    normalizeToolUseID(id),
				"name":  jsonx.GetString(fc, "name"),
				"input": input,
				// The block states the field even when nothing rides it, the
				// shape the reference emits.
				"provider_specific_fields": nil,
			}
			if sig != "" {
				block["provider_specific_fields"] = map[string]any{"signature": sig}
			}
			calls = append(calls, block)
			continue
		}
		if inline, ok := jsonx.AsMap(p["inlineData"]); ok {
			mime := jsonx.GetString(inline, "mimeType")
			switch {
			case strings.HasPrefix(mime, "image/"):
				// Model-made images are dropped on this fold: the Messages
				// response has no spelling for them, and the composed fold
				// would have delivered them nowhere.
			case strings.HasPrefix(mime, "audio/"):
				// Audio sets the content aside on the composed fold, which
				// leaves nothing for the Messages body either.
			default:
				text.WriteString("data:" + mime + ";base64," + jsonx.GetString(inline, "data"))
				hasText = true
			}
			continue
		}
		if raw, present := p["text"]; present {
			s := jsonx.String(raw)
			thought, _ := jsonx.Bool(p["thought"])
			if thought {
				sig, hasSig := "", false
				if v, present := p["thoughtSignature"]; present && v != nil {
					sig = jsonx.String(v)
					hasSig = true
				}
				thoughts = append(thoughts, thoughtBlock{s, sig, hasSig})
				continue
			}
			if isAudioDataURL(s) {
				// A text part that is exactly an audio data URL is audio
				// output in disguise; the composed fold recognizes it and
				// keeps it out of the text.
				continue
			}
			text.WriteString(s)
			hasText = true
		}
	}
	for _, t := range thoughts {
		if t.text == "" && !t.hasSig {
			// A thought part with no text and no signature holds nothing a
			// client could replay, and the Messages dialect rejects an empty
			// thinking block outright.
			continue
		}
		// A thought signature the source stated rides through; an unsigned
		// thought gets the fold's stand-in rather than "", which Messages
		// clients read as "no signature" and answer by downgrading the
		// thinking to text on replay — see internal/thinksig.
		sig := any(thinksig.Synthetic)
		if t.hasSig {
			sig = t.sig
		}
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": t.text, "signature": sig})
	}
	if hasText {
		blocks = append(blocks, map[string]any{"type": "text", "text": text.String()})
	}
	// The calls close the turn: the reference emits thinking, then the text,
	// then the tool calls, whatever order the parts arrived in.
	blocks = append(blocks, calls...)
	return blocks, hasCalls
}

// thoughtBlock is one thought part's Messages shape: its text and the
// signature it carried, if any.
type thoughtBlock struct {
	text   string
	sig    string
	hasSig bool
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// geminiFinishToStopReason maps a finish reason onto the Messages stop_reason
// by way of the chat finish the composed fold produces: a call-bearing turn
// closes as a tool turn whatever the reason says, a ceiling cut maps to
// max_tokens, and everything else — the filtered reasons included — is
// end_turn on the Messages wire.
func geminiFinishToStopReason(finish string, hasToolCalls bool) string {
	if hasToolCalls || finish == "tool_calls" || finish == "function_call" {
		return "tool_use"
	}
	switch finish {
	case "STOP", "FINISH_REASON_UNSPECIFIED", "MALFORMED_FUNCTION_CALL",
		"TOO_MANY_TOOL_CALLS", "MALFORMED_RESPONSE", "":
		return "end_turn"
	case "MAX_TOKENS":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

// usageMetadataOf reads the usage block, tolerating its absence with an empty
// map — the reference fails a response with no usage at all, where this fold
// settles for zeros rather than failing a body it was handed.
func usageMetadataOf(body map[string]any) map[string]any {
	if u, ok := jsonx.AsMap(body["usageMetadata"]); ok {
		return u
	}
	return map[string]any{}
}

// geminiUsageToMessages folds the usage block through the composed fold's
// arithmetic: the prompt count plus any billable tool-use tokens; the
// completion count widened by the thinking tokens when the totals show the
// candidate count excluded them; and the cached tokens taken off the input
// count, because the Messages input is the uncached remainder. A response
// with search grounding also names the searches it performed: the composed
// fold counts the distinct queries as the server-tool usage the Messages
// dialect reports.
func geminiUsageToMessages(u, body map[string]any) map[string]any {
	promptTok, _ := jsonx.Int(u["promptTokenCount"])
	candTok, _ := jsonx.Int(u["candidatesTokenCount"])
	totalTok, _ := jsonx.Int(u["totalTokenCount"])
	toolUse, hasToolUse := jsonx.Int(u["toolUsePromptTokenCount"])
	thoughts, _ := jsonx.Int(u["thoughtsTokenCount"])
	cached, _ := jsonx.Int(u["cachedContentTokenCount"])

	// When the response carries billable grounding, the tool-use tokens are
	// not billable on top of the prompt; otherwise they are added to it.
	webSearches, billableGrounding := searchGroundingRequests(body)
	billable := 0
	if !billableGrounding && hasToolUse {
		billable = toolUse
	}
	prompt := promptTok + billable

	completion := candTok
	inclusive := promptTok+candTok+billable == totalTok
	if !inclusive && thoughts > 0 {
		completion = thoughts + candTok
	}
	in := prompt - cached
	if in < 0 {
		in = 0
	}
	out := map[string]any{"input_tokens": in, "output_tokens": completion}
	if cached > 0 {
		out["cache_read_input_tokens"] = cached
	}
	if webSearches > 0 {
		out["server_tool_use"] = map[string]any{"web_search_requests": webSearches}
	}
	return out
}

// searchGroundingRequests counts the billable grounding a response carries,
// the way the composed fold counts it: every distinct non-empty web search
// query across the candidates' grounding metadata is one request, queries a
// maps-attributed item reported belong to the maps counter instead, and a
// maps entry (a maps grounding chunk or a widget token) marks the response as
// grounded even when no query was reported. Grounded tool use is billed
// inside the prompt count, which is why the caller asks.
func searchGroundingRequests(body map[string]any) (webSearches int, billable bool) {
	seenWeb := map[string]struct{}{}
	anyMaps := false
	candidates, _ := jsonx.AsSlice(body["candidates"])
	for _, raw := range candidates {
		c, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		g, ok := jsonx.AsMap(c["groundingMetadata"])
		if !ok {
			continue
		}
		itemMaps, itemWeb := false, false
		if chunks, ok := jsonx.AsSlice(g["groundingChunks"]); ok {
			for _, chunk := range chunks {
				cm, ok := jsonx.AsMap(chunk)
				if !ok {
					continue
				}
				if _, has := cm["maps"]; has {
					itemMaps = true
				}
				if _, has := cm["web"]; has {
					itemWeb = true
				}
			}
		}
		if v, has := g["googleMapsWidgetContextToken"]; has && v != nil {
			itemMaps = true
		}
		anyMaps = anyMaps || itemMaps
		if itemMaps && !itemWeb {
			// This item's queries belong to the maps counter; the Messages
			// usage does not read that one.
			continue
		}
		if queries, ok := jsonx.AsSlice(g["webSearchQueries"]); ok {
			for _, q := range queries {
				if s, ok := q.(string); ok && s != "" {
					seenWeb[s] = struct{}{}
				}
			}
		}
	}
	return len(seenWeb), len(seenWeb) > 0 || anyMaps
}

// MessagesResponseToGemini folds an Anthropic Messages response into a gemini
// generateContent response body.
//
// The target body has no model member, so the Messages model does not cross.
func MessagesResponseToGemini(body map[string]any) map[string]any {
	content, _ := jsonx.AsSlice(body["content"])
	parts := messagesBlocksToParts(content)
	out := map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": messagesStopToFinishReason(jsonx.GetString(body, "stop_reason")),
			"index":        0,
		}},
		"usageMetadata": messagesUsageToGemini(body["usage"]),
	}
	if id := jsonx.GetString(body, "id"); id != "" {
		out["responseId"] = id
	}
	return out
}

// messagesBlocksToParts folds the content blocks into parts in source order:
// text becomes a text part, a signed thinking block becomes a flagged,
// signature-carrying part and an unsigned one a thought part (the shapes the
// reference's response reader reconstructs reasoning from), and a tool_use
// block becomes a function call
// with the signature its provider fields carry. Blocks with no target
// spelling — redacted thinking among them, which the request-side fold never
// reads — are dropped.
func messagesBlocksToParts(content []any) []any {
	var parts []any
	for _, raw := range content {
		bm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "text":
			parts = append(parts, map[string]any{"text": jsonx.GetString(bm, "text")})
		case "thinking":
			thinking := jsonx.GetString(bm, "thinking")
			sig, hasSig := "", false
			if v, present := bm["signature"]; present && v != nil {
				if s, ok := v.(string); ok && s != "" {
					sig = s
					hasSig = true
				}
			}
			if thinking == "" && !hasSig {
				continue
			}
			if hasSig {
				// A signed thinking block rides a flagged thought part: the
				// reference's response reader folds an unflagged part's text
				// into the content (verified by executing the reader), while
				// its request writer replays blocks unflagged — the request
				// fold keeps the writer's spelling, this one the reader's.
				parts = append(parts, map[string]any{"text": thinking, "thought": true, "thoughtSignature": sig})
			} else {
				parts = append(parts, map[string]any{"text": thinking, "thought": true})
			}
		case "tool_use":
			call := map[string]any{"name": jsonx.GetString(bm, "name")}
			if input, ok := bm["input"]; ok && input != nil {
				call["args"] = roundTripJSON(input)
			} else {
				call["args"] = map[string]any{}
			}
			// A response body spells the call camelCase — the response wire
			// and the request wire differ on exactly this word.
			part := map[string]any{"functionCall": call}
			if sig := toolUseSignature(bm); sig != "" {
				part["thoughtSignature"] = sig
			}
			parts = append(parts, part)
		}
	}
	if parts == nil {
		parts = []any{}
	}
	return parts
}

// messagesStopToFinishReason maps a stop_reason onto the target's finishReason
// by way of the chat finish the composed fold produces: end_turn and
// stop_sequence close cleanly, max_tokens is a ceiling cut, a tool turn
// closes cleanly too (the call, not the reason, is what ended it), and the
// filtered reasons map to the target's representative value.
func messagesStopToFinishReason(stop string) string {
	switch stop {
	case "max_tokens":
		return "MAX_TOKENS"
	case "refusal", "content_filtered":
		return "SAFETY"
	default:
		return "STOP"
	}
}

// messagesUsageToGemini folds the Messages usage onto the target's usage
// block. The prompt count is the inclusive total (the Messages input is the
// uncached remainder, and the cache halves are counted back in), the thinking
// share of the output splits off into its own counter, and the total is
// recomputed from the parts so the body is self-consistent.
func messagesUsageToGemini(v any) map[string]any {
	u, _ := jsonx.AsMap(v)
	in, _ := jsonx.Int(u["input_tokens"])
	out, _ := jsonx.Int(u["output_tokens"])
	cacheRead, _ := jsonx.Int(u["cache_read_input_tokens"])
	cacheWrite, _ := jsonx.Int(u["cache_creation_input_tokens"])
	reasoning := 0
	if d, ok := jsonx.AsMap(u["output_tokens_details"]); ok {
		if t, ok := jsonx.Int(d["thinking_tokens"]); ok && t > 0 && t < out {
			reasoning = t
		} else if ok && t >= out && out > 0 {
			reasoning = out
		}
	}
	prompt := in + cacheRead + cacheWrite
	candidates := out - reasoning
	total := prompt + candidates + reasoning
	g := map[string]any{
		"promptTokenCount":     prompt,
		"candidatesTokenCount": candidates,
		"totalTokenCount":      total,
	}
	if cacheRead > 0 {
		g["cachedContentTokenCount"] = cacheRead
	}
	if reasoning > 0 {
		g["thoughtsTokenCount"] = reasoning
	}
	if in > 0 {
		g["promptTokensDetails"] = []any{map[string]any{"modality": "TEXT", "tokenCount": in}}
	}
	if candidates > 0 {
		g["candidatesTokensDetails"] = []any{map[string]any{"modality": "TEXT", "tokenCount": candidates}}
	}
	// A cache-write count has no key on the target wire — it only reads one
	// cached counter — so the write half folds into the prompt total and stops
	// there.
	return g
}
