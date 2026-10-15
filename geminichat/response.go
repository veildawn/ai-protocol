package geminichat

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// finishReasonToChat maps the wire's finish reasons onto the chat vocabulary.
// Unknown values fall to "stop" — the reference warns and does the same rather
// than refusing a turn over a reason it cannot name.
var finishReasonToChat = map[string]string{
	"STOP":                      "stop",
	"MAX_TOKENS":                "length",
	"SAFETY":                    "content_filter",
	"RECITATION":                "content_filter",
	"FINISH_REASON_UNSPECIFIED": "stop",
	"MALFORMED_FUNCTION_CALL":   "stop",
	"LANGUAGE":                  "content_filter",
	"OTHER":                     "content_filter",
	"BLOCKLIST":                 "content_filter",
	"PROHIBITED_CONTENT":        "content_filter",
	"SPII":                      "content_filter",
	"IMAGE_SAFETY":              "content_filter",
	"IMAGE_PROHIBITED_CONTENT":  "content_filter",
	"TOO_MANY_TOOL_CALLS":       "stop",
	"MALFORMED_RESPONSE":        "stop",
}

// flaggedFinishReasons are the reasons that mean the content was refused. When
// the first candidate carries one, the whole response is replaced by a single
// filtered choice — there is no partial answer to hand the client.
var flaggedFinishReasons = map[string]bool{
	"SAFETY": true, "RECITATION": true, "BLOCKLIST": true,
	"PROHIBITED_CONTENT": true, "SPII": true, "IMAGE_SAFETY": true,
	"IMAGE_PROHIBITED_CONTENT": true,
}

// finishReasonFromChat maps a chat finish_reason back onto the wire. A filtered
// or refused turn has no more precise spelling here than the generic safety
// reason; a tool-call close is a natural stop on the wire.
func finishReasonFromChat(fr string) string {
	switch fr {
	case "stop":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "tool_calls", "function_call":
		return "STOP"
	case "content_filter", "refusal":
		return "SAFETY"
	default:
		return "STOP"
	}
}

// GeminiResponseToChat folds a generateContent response body into a chat
// completion response.
//
// The source names no model and carries no clock, so the chat body's model is
// empty and its created is zero — the model a client needs rides the request,
// which this pure fold never sees. The id falls back to the package's own
// deterministic placeholder: the reference mints a uuid there, which this
// library has no way to reproduce.
func GeminiResponseToChat(body map[string]any) map[string]any {
	id := jsonx.GetString(body, "responseId")
	if id == "" {
		id = "chatcmpl-unknown"
	}
	if pf, ok := jsonx.AsMap(body["promptFeedback"]); ok {
		if _, blocked := pf["blockReason"]; blocked {
			return map[string]any{
				"id":      id,
				"object":  "chat.completion",
				"created": 0,
				"model":   jsonx.GetString(body, "model"),
				"choices": []any{blockedChoice()},
				"usage":   blockedUsage(body),
			}
		}
	}
	if candidates, _ := jsonx.AsSlice(body["candidates"]); len(candidates) > 0 {
		if c0, ok := jsonx.AsMap(candidates[0]); ok {
			if flaggedFinishReasons[jsonx.GetString(c0, "finishReason")] {
				return map[string]any{
					"id":      id,
					"object":  "chat.completion",
					"created": 0,
					"model":   jsonx.GetString(body, "model"),
					"choices": []any{blockedChoice()},
					"usage":   blockedUsage(body),
				}
			}
		}
	}
	choices := []any{}
	toolIndex := 0
	callSeq := 0
	for idx, raw := range candidatesOrEmpty(body) {
		cm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		content, hasContent := jsonx.AsMap(cm["content"])
		if !hasContent {
			// A candidate with no content at all contributes nothing — the
			// reference skips it rather than emitting an empty answer.
			continue
		}
		msg := map[string]any{"content": nil, "role": "assistant", "tool_calls": nil, "function_call": nil}
		finish := "stop"
		parts, hasParts := jsonx.AsSlice(content["parts"])
		if hasParts {
			msg["images"] = []any{}
			msg["thinking_blocks"] = []any{}
			finish = foldCandidateParts(parts, msg, jsonx.GetString(cm, "finishReason"), &toolIndex, &callSeq)
		}
		if _, has := cm["index"]; has {
			msg["_index"] = cm["index"]
		} else {
			msg["_index"] = idx
		}
		choice := map[string]any{
			"index":         msg["_index"],
			"message":       msg,
			"finish_reason": finish,
		}
		delete(msg, "_index")
		choices = append(choices, choice)
	}
	return map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": 0,
		"model":   jsonx.GetString(body, "model"),
		"choices": choices,
		"usage":   geminiUsageToChat(body),
	}
}

func candidatesOrEmpty(body map[string]any) []any {
	candidates, _ := jsonx.AsSlice(body["candidates"])
	return candidates
}

// blockedChoice is the single filtered choice a refused response collapses to.
func blockedChoice() map[string]any {
	return map[string]any{
		"finish_reason": "content_filter",
		"index":         0,
		"message": map[string]any{
			"content": nil, "role": "assistant", "tool_calls": nil, "function_call": nil,
		},
	}
}

// blockedUsage reports the three headline counts only: a refused turn produced
// no answer to break down, so the detail objects stay null.
func blockedUsage(body map[string]any) map[string]any {
	u, _ := jsonx.AsMap(body["usageMetadata"])
	prompt, _ := jsonx.Int(u["promptTokenCount"])
	completion, _ := jsonx.Int(u["candidatesTokenCount"])
	total, _ := jsonx.Int(u["totalTokenCount"])
	return map[string]any{
		"prompt_tokens":             prompt,
		"completion_tokens":         completion,
		"total_tokens":              total,
		"completion_tokens_details": nil,
		"prompt_tokens_details":     nil,
	}
}

// geminiPartsFold accumulates what one candidate's parts say. The response fold
// turns it into a chat message; the stream fold turns it into a chat delta —
// the same parts, read the same way, so the two faces cannot drift apart.
type geminiPartsFold struct {
	content      strings.Builder
	hasContent   bool
	reasoning    strings.Builder
	hasReasoning bool
	images       []any
	blocks       []any
	sigs         []string
	toolCalls    []any
	audio        map[string]any
}

// feed folds one part. Text and non-image, non-audio inline media accumulate
// onto the content; a thought part accumulates onto the reasoning and a
// thinking block; images, audio and calls each ride their own member.
func (f *geminiPartsFold) feed(pm map[string]any, toolIndex, callSeq *int) {
	sig := partThoughtSignature(pm)
	if sig != "" {
		f.sigs = append(f.sigs, sig)
	}
	if fc, ok := jsonx.AsMap(pm["functionCall"]); ok {
		args := fc["args"]
		if args == nil {
			args = map[string]any{}
		}
		id := jsonx.GetString(fc, "id")
		if id == "" {
			// The wire's stable ids exist only on the newest models; this fold
			// mints a positional one so the result that answers the call has
			// something to name.
			id = fmt.Sprintf("call_%d", *callSeq)
			*callSeq++
		}
		call := map[string]any{
			"index": *toolIndex,
			"function": map[string]any{
				"arguments": mustArgs(args),
				"name":      jsonx.GetString(fc, "name"),
			},
			"id":   id,
			"type": "function",
		}
		if sig != "" {
			call["provider_specific_fields"] = map[string]any{"thought_signature": sig}
			call["id"] = id + thoughtSignatureSeparator + sig
		}
		f.toolCalls = append(f.toolCalls, call)
		*toolIndex++
		return
	}
	if inline, ok := jsonx.AsMap(pm["inlineData"]); ok {
		mime := jsonx.GetString(inline, "mimeType")
		data := jsonx.GetString(inline, "data")
		switch {
		case strings.HasPrefix(mime, "image/"):
			// Images ride their own member with the detail a replay needs;
			// folding one into the text would hand the client a URI where it
			// reads an answer.
			f.images = append(f.images, map[string]any{
				"image_url": map[string]any{"url": "data:" + mime + ";base64," + data, "detail": "auto"},
				"index":     0,
				"type":      "image_url",
			})
		case strings.HasPrefix(mime, "audio/"):
			if f.audio == nil {
				f.audio = wireAudio(data)
			}
		default:
			f.content.WriteString("data:" + mime + ";base64," + data)
			f.hasContent = true
		}
		return
	}
	if raw, ok := pm["text"]; ok {
		text := jsonx.String(raw)
		if data, isAudio := audioDataURL(text); isAudio {
			// Speech the model emitted as a data URL is audio output, not
			// answer text; it rides the audio member and the content the
			// client reads stays empty.
			if f.audio == nil {
				f.audio = wireAudio(data)
			}
			return
		}
		if thought, _ := pm["thought"].(bool); thought {
			if len(text) > 0 {
				f.reasoning.WriteString(text)
				f.hasReasoning = true
			}
			block := map[string]any{"type": "thinking", "thinking": text}
			if sig != "" {
				block["signature"] = sig
			}
			f.blocks = append(f.blocks, block)
			return
		}
		// Only a part that says something opens the content member: a turn of
		// nothing but empty texts has no content to state, and the reference
		// leaves it null rather than empty.
		if len(text) > 0 {
			f.content.WriteString(text)
			f.hasContent = true
		}
	}
}

// finishReason is the choice's close: a turn that made a call closes as a
// tool-call turn whatever the wire's own reason says.
func (f *geminiPartsFold) finishReason(fr string) string {
	if len(f.toolCalls) > 0 {
		return "tool_calls"
	}
	if v, ok := finishReasonToChat[fr]; ok {
		return v
	}
	return "stop"
}

// applyMessage writes the fold onto a chat message. Images and thinking blocks
// are stated even when empty — the shape the reference always sends for a
// candidate whose content carried parts.
func (f *geminiPartsFold) applyMessage(msg map[string]any) {
	if f.audio != nil {
		// A spoken turn states its audio member; the text content is written
		// after it and wins when the turn also said something — only a turn
		// with no words at all leaves the content null.
		msg["audio"] = f.audio
	}
	if f.hasContent {
		msg["content"] = f.content.String()
	}
	if len(f.images) > 0 {
		msg["images"] = f.images
	}
	if len(f.blocks) > 0 {
		msg["thinking_blocks"] = f.blocks
	}
	if f.hasReasoning {
		msg["reasoning_content"] = f.reasoning.String()
	}
	if len(f.toolCalls) > 0 {
		msg["tool_calls"] = f.toolCalls
	}
	if len(f.sigs) > 0 {
		msg["provider_specific_fields"] = map[string]any{"thought_signatures": f.sigs}
	}
}

// applyDelta writes the fold onto a chat delta. A streaming delta never carries
// thinking blocks: the blocks are the reference's non-streaming spelling, and
// its own stream folds the thought text into reasoning alone. Spoken audio
// rides no delta either — the reference's delta shape has no member for it, so
// the bytes wait for the non-streaming fold or the host's own framing.
func (f *geminiPartsFold) applyDelta() map[string]any {
	delta := map[string]any{}
	if f.hasContent {
		delta["content"] = f.content.String()
	}
	if f.hasReasoning {
		delta["reasoning_content"] = f.reasoning.String()
	}
	if len(f.images) > 0 {
		delta["images"] = f.images
	}
	if len(f.toolCalls) > 0 {
		delta["tool_calls"] = f.toolCalls
	}
	if len(f.sigs) > 0 {
		delta["provider_specific_fields"] = map[string]any{"thought_signatures": f.sigs}
	}
	return delta
}

// foldCandidateParts folds one candidate's parts onto a chat message and
// returns the choice's finish reason.
func foldCandidateParts(parts []any, msg map[string]any, finishReason string, toolIndex, callSeq *int) string {
	var f geminiPartsFold
	for _, p := range parts {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		f.feed(pm, toolIndex, callSeq)
	}
	f.applyMessage(msg)
	return f.finishReason(finishReason)
}

// wireAudio is the audio member a spoken turn carries. The reference fills an
// id, an expiry and a transcript alongside the bytes; this library has neither
// a uuid nor a clock, so those stand in as the empty id, zero and "" — the
// bytes are the only part any fold reads back.
func wireAudio(data string) map[string]any {
	return map[string]any{"id": "", "data": data, "expires_at": 0, "transcript": ""}
}

// audioDataURL reports the payload when a text part is exactly a base64 audio
// data URL — the shape the reference's own speech output takes on the wire.
func audioDataURL(text string) (string, bool) {
	if !strings.HasPrefix(text, "data:audio") || !strings.Contains(text, ";base64,") {
		return "", false
	}
	if !isBase64DataURL(text) {
		return "", false
	}
	before, after, found := strings.Cut(text[5:], ";base64,")
	if !found || !strings.HasPrefix(before, "audio/") {
		return "", false
	}
	return after, true
}

// isBase64DataURL reports whether a data URL's payload is strict base64 that
// re-encodes to itself, the check the reference applies before treating a data
// URL as media.
func isBase64DataURL(s string) bool {
	if !strings.HasPrefix(s, "data:") {
		return false
	}
	_, payload, found := strings.Cut(s, ",")
	if !found {
		return false
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(payload)
	if err != nil {
		return false
	}
	return base64.StdEncoding.EncodeToString(decoded) == payload
}

// geminiUsageToChat folds usageMetadata onto the chat usage shape.
//
// The completion count has to decide whether the candidate count already
// includes the thoughts: the wire states it by whether the token totals add up
// without them, so that arithmetic is what the fold reads. Tool-use tokens are
// billed as prompt tokens unless the turn used billable grounding, whose
// separate rate the totals must not double-count. Per-modality prompt counts
// are stated inclusive of the cache and subtracted back out here, so a cache
// reading folds from whichever of the two spellings the wire offered.
func geminiUsageToChat(body map[string]any) map[string]any {
	u, _ := jsonx.AsMap(body["usageMetadata"])
	at := func(key string) (int, bool) {
		if u == nil {
			return 0, false
		}
		v, ok := u[key]
		if !ok || v == nil {
			return 0, false
		}
		return jsonx.Int(v)
	}
	prompt, _ := at("promptTokenCount")
	cand, _ := at("candidatesTokenCount")
	toolUse, toolUsePresent := at("toolUsePromptTokenCount")
	total, _ := at("totalTokenCount")
	cached, cachedPresent := at("cachedContentTokenCount")
	reasoning, reasoningPresent := at("thoughtsTokenCount")
	respTokens, respPresent := at("responseTokenCount")

	grounding := groundingRequests(body["candidates"])
	billable := 0
	if !grounding.billable {
		billable = toolUse
	}
	promptTokens := prompt + billable

	completion := cand
	if respPresent && respTokens != 0 {
		completion = respTokens
	}
	inclusive := prompt+cand+toolUse == total
	if !inclusive && reasoningPresent && reasoning != 0 {
		completion = reasoning + completion
	}

	promptDetails := map[string]any{
		"cached_tokens": nil, "audio_tokens": nil, "text_tokens": nil,
		"image_tokens": nil, "video_tokens": nil,
	}
	if cachedPresent {
		promptDetails["cached_tokens"] = cached
	}
	var promptAudio, promptText, promptImage, promptVideo *int
	add := func(p **int, n int) {
		if *p == nil {
			*p = new(int)
		}
		**p += n
	}
	if list, ok := jsonx.AsSlice(u["promptTokensDetails"]); ok {
		for _, d := range list {
			dm, ok := jsonx.AsMap(d)
			if !ok {
				continue
			}
			n, _ := jsonx.Int(dm["tokenCount"])
			switch strings.ToUpper(jsonx.GetString(dm, "modality")) {
			case "AUDIO":
				add(&promptAudio, n)
			case "TEXT":
				add(&promptText, n)
			case "IMAGE":
				add(&promptImage, n)
			case "VIDEO":
				add(&promptVideo, n)
			case "DOCUMENT":
				add(&promptText, n)
			}
		}
	}
	var cacheAudio, cacheText, cacheImage, cacheVideo *int
	_, hasCacheDetails := jsonx.AsSlice(u["cacheTokensDetails"])
	if hasCacheDetails {
		if list, ok := jsonx.AsSlice(u["cacheTokensDetails"]); ok {
			for _, d := range list {
				dm, ok := jsonx.AsMap(d)
				if !ok {
					continue
				}
				n, _ := jsonx.Int(dm["tokenCount"])
				switch strings.ToUpper(jsonx.GetString(dm, "modality")) {
				case "AUDIO":
					add(&cacheAudio, n)
				case "TEXT":
					add(&cacheText, n)
				case "IMAGE":
					add(&cacheImage, n)
				case "VIDEO":
					add(&cacheVideo, n)
				case "DOCUMENT":
					add(&cacheText, n)
				}
			}
		}
	}
	set := func(key string, p *int) {
		if p != nil {
			promptDetails[key] = *p
		}
	}
	if cacheText != nil && promptText != nil {
		promptDetails["text_tokens"] = *promptText - *cacheText
	} else if cachedPresent && promptText != nil && cacheText == nil && !hasCacheDetails {
		promptDetails["text_tokens"] = *promptText - cached
	} else {
		set("text_tokens", promptText)
	}
	if cacheAudio != nil && promptAudio != nil {
		promptDetails["audio_tokens"] = *promptAudio - *cacheAudio
	} else {
		set("audio_tokens", promptAudio)
	}
	if cacheImage != nil && promptImage != nil {
		promptDetails["image_tokens"] = *promptImage - *cacheImage
	} else {
		set("image_tokens", promptImage)
	}
	if cacheVideo != nil && promptVideo != nil {
		promptDetails["video_tokens"] = *promptVideo - *cacheVideo
	} else {
		set("video_tokens", promptVideo)
	}
	if toolUsePresent && toolUse != 0 {
		promptDetails["tool_use_tokens"] = toolUse
	}
	// Search and maps grounding are billed per request, not per token; the
	// reference states the counts beside the prompt breakdown so a client can
	// price them. The spec's usage key list omits them, but the reference's
	// own fold states them and dropping a billed count is the worse loss.
	if grounding.webSearch > 0 {
		promptDetails["web_search_requests"] = grounding.webSearch
	}
	if grounding.maps > 0 {
		promptDetails["google_maps_grounding_requests"] = grounding.maps
	}

	completionDetails := map[string]any(nil)
	var outAudio, outText, outImage, outVideo *int
	hasDetails := false
	if list, ok := jsonx.AsSlice(u["responseTokensDetails"]); ok {
		hasDetails = true
		for _, d := range list {
			dm, ok := jsonx.AsMap(d)
			if !ok {
				continue
			}
			n, _ := jsonx.Int(dm["tokenCount"])
			switch strings.ToUpper(jsonx.GetString(dm, "modality")) {
			case "AUDIO":
				add(&outAudio, n)
			case "TEXT", "DOCUMENT":
				add(&outText, n)
			}
		}
	}
	if list, ok := jsonx.AsSlice(u["candidatesTokensDetails"]); ok {
		hasDetails = true
		for _, d := range list {
			dm, ok := jsonx.AsMap(d)
			if !ok {
				continue
			}
			n, _ := jsonx.Int(dm["tokenCount"])
			switch strings.ToUpper(jsonx.GetString(dm, "modality")) {
			case "AUDIO":
				add(&outAudio, n)
			case "TEXT", "DOCUMENT":
				add(&outText, n)
			case "IMAGE":
				add(&outImage, n)
			case "VIDEO":
				add(&outVideo, n)
			}
		}
	}
	if reasoningPresent {
		hasDetails = true
	}
	if cand > 0 {
		hasDetails = true
		if outText == nil {
			zero := 0
			outText = &zero
			audioN, imageN, videoN := 0, 0, 0
			if outAudio != nil {
				audioN = *outAudio
			}
			if outImage != nil {
				imageN = *outImage
			}
			if outVideo != nil {
				videoN = *outVideo
			}
			*outText = cand - imageN - audioN - videoN
		}
	}
	if hasDetails {
		completionDetails = map[string]any{}
		if outText != nil {
			completionDetails["text_tokens"] = *outText
		}
		if reasoningPresent {
			completionDetails["reasoning_tokens"] = reasoning
		}
		if outAudio != nil {
			completionDetails["audio_tokens"] = *outAudio
		}
		if outImage != nil {
			completionDetails["image_tokens"] = *outImage
		}
		if outVideo != nil {
			completionDetails["video_tokens"] = *outVideo
		}
	}

	usage := map[string]any{
		"prompt_tokens":             promptTokens,
		"completion_tokens":         completion,
		"total_tokens":              total,
		"completion_tokens_details": completionDetails,
		"prompt_tokens_details":     promptDetails,
	}
	if cachedPresent {
		usage["cache_read_input_tokens"] = cached
	} else {
		usage["cache_read_input_tokens"] = nil
	}
	return usage
}

// groundingRequests reads the billable lookups a turn's grounding metadata
// names. Search and maps grounding are charged per request rather than per
// token, so their tool-use tokens leave the prompt a client bills, and the
// request counts ride the prompt breakdown for the client to price. Each
// distinct query is one request, duplicates across candidates collapse, and a
// maps lookup floors at one request even when no query was reported — the
// per-prompt charge applies either way.
type groundingCounts struct {
	billable  bool
	webSearch int
	maps      int
}

func groundingRequests(candidates any) groundingCounts {
	var out groundingCounts
	list, ok := jsonx.AsSlice(candidates)
	if !ok {
		return out
	}
	web := map[string]bool{}
	mapsQueries := map[string]bool{}
	hasMaps := false
	for _, raw := range list {
		cm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		items, ok := jsonx.AsSlice(cm["groundingMetadata"])
		if !ok {
			if gm, isMap := jsonx.AsMap(cm["groundingMetadata"]); isMap {
				items = []any{gm}
			} else {
				continue
			}
		}
		for _, item := range items {
			gm, ok := jsonx.AsMap(item)
			if !ok {
				continue
			}
			kinds := map[string]bool{}
			if chunks, ok := jsonx.AsSlice(gm["groundingChunks"]); ok {
				for _, c := range chunks {
					if cmm, ok := jsonx.AsMap(c); ok {
						for k := range cmm {
							kinds[k] = true
						}
					}
				}
			}
			isMaps := kinds["maps"] || jsonx.GetString(gm, "googleMapsWidgetContextToken") != ""
			attributesToMaps := isMaps && !kinds["web"]
			if isMaps {
				hasMaps = true
			}
			queries, _ := jsonx.AsSlice(gm["webSearchQueries"])
			for _, q := range queries {
				s, isStr := q.(string)
				if !isStr || s == "" {
					continue
				}
				if attributesToMaps {
					mapsQueries[s] = true
				} else {
					web[s] = true
				}
			}
		}
	}
	if len(web) > 0 {
		out.webSearch = len(web)
	}
	if hasMaps {
		out.maps = len(mapsQueries)
		if out.maps < 1 {
			out.maps = 1
		}
	}
	out.billable = len(web) > 0 || hasMaps
	return out
}

// ChatResponseToGemini folds a chat completion response body into the target
// dialect's response body: one candidate per choice, the turn's reason mapped
// back, and the usage recomputed so the inclusive-or-exclusive arithmetic the
// forward fold performs lands on the counts this body started from.
//
// The responseId is the chat id verbatim; the target's own id member is the
// only spelling it has.
func ChatResponseToGemini(body map[string]any) map[string]any {
	choices, _ := jsonx.AsSlice(body["choices"])
	candidates := []any{}
	for idx, raw := range choices {
		cm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		msg, _ := jsonx.AsMap(cm["message"])
		if msg == nil {
			msg = map[string]any{}
		}
		index := idx
		if v, ok := jsonx.Int(cm["index"]); ok {
			index = v
		}
		candidates = append(candidates, map[string]any{
			"content": map[string]any{
				"role":  "model",
				"parts": chatMessageToParts(msg),
			},
			"finishReason": finishReasonFromChat(jsonx.GetString(cm, "finish_reason")),
			"index":        index,
		})
	}
	out := map[string]any{
		"candidates":    candidates,
		"usageMetadata": chatUsageToMetadata(body["usage"]),
	}
	if id := jsonx.GetString(body, "id"); id != "" {
		out["responseId"] = id
	}
	return out
}

// replayedThoughtPart renders one signed thinking block for a RESPONSE body.
// The reference's own response reader reconstructs reasoning only from parts
// flagged thought: an unflagged part's text folds into the content instead
// (executed against the reader — the request wire's replay spelling comes
// back with the reasoning concatenated onto the content). The request folds
// keep that unflagged spelling, which is what the reference's writer emits;
// each side mirrors its own reference.
func replayedThoughtPart(text, sig string) map[string]any {
	return map[string]any{"text": text, "thought": true, "thoughtSignature": sig}
}

// chatMessageToParts folds one choice's message onto the parts of a candidate.
// The order is the forward fold's own: the reasoning summary, signed thinking
// blocks, the text, generated media, then the calls.
func chatMessageToParts(msg map[string]any) []any {
	var parts []any
	if rc := jsonx.GetString(msg, "reasoning_content"); rc != "" {
		parts = append(parts, map[string]any{"text": rc, "thought": true})
	}
	if tbs, ok := jsonx.AsSlice(msg["thinking_blocks"]); ok && msg["thinking_blocks"] != nil {
		for _, b := range tbs {
			bm, isMap := jsonx.AsMap(b)
			if !isMap || jsonx.GetString(bm, "type") != "thinking" {
				// A redacted block is opaque on the chat wire too; no fold can
				// read it back.
				continue
			}
			text, hasText := bm["thinking"]
			sig, hasSig := bm["signature"]
			if !hasText || text == nil || !hasSig || sig == nil {
				// The forward fold drops an unsigned block: it is the same fact
				// as the reasoning summary already carried, and sending both
				// would double the reasoning the target bills on replay.
				continue
			}
			parts = append(parts, replayedThoughtPart(jsonx.String(text), jsonx.String(sig)))
		}
	}
	text := ""
	stringContent := false
	switch c := msg["content"].(type) {
	case string:
		text = c
		stringContent = true
	case []any:
		for _, p := range c {
			if pm, ok := jsonx.AsMap(p); ok && jsonx.GetString(pm, "type") == "text" {
				text += jsonx.GetString(pm, "text")
			}
		}
	}
	if text != "" {
		// The signature a replay attaches to the turn's text rides the text
		// part, and only for the string content the forward fold reads it from.
		sig := ""
		if stringContent {
			if psf, ok := jsonx.AsMap(msg["provider_specific_fields"]); ok {
				if sigs, ok := jsonx.AsSlice(psf["thought_signatures"]); ok && len(sigs) > 0 {
					sig = jsonx.String(sigs[0])
				}
			}
		}
		if sig != "" {
			parts = append(parts, map[string]any{"text": text, "thoughtSignature": sig})
		} else {
			parts = append(parts, map[string]any{"text": text})
		}
	}
	if imgs, ok := jsonx.AsSlice(msg["images"]); ok && msg["images"] != nil {
		for _, item := range imgs {
			im, isMap := jsonx.AsMap(item)
			if !isMap {
				continue
			}
			url := jsonx.GetString(im, "image_url")
			if iu, ok := jsonx.AsMap(im["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			if mime, data, ok := splitDataURLParts(url); ok {
				// The response wire spells inline media in the camel case its
				// own reader expects; the request wire is the snake-case one.
				parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": mime, "data": data}})
			}
		}
	}
	if audio, ok := jsonx.AsMap(msg["audio"]); ok {
		if data := jsonx.GetString(audio, "data"); data != "" {
			// The chat audio member names no media type; the reference's speech
			// output is pcm, and the wire's spelling for it is fixed here.
			parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": "audio/wav", "data": data}})
		}
	}
	if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok && msg["tool_calls"] != nil {
		for _, tc := range tcs {
			tm, isMap := jsonx.AsMap(tc)
			if !isMap {
				continue
			}
			fn, _ := jsonx.AsMap(tm["function"])
			if fn == nil {
				fn = map[string]any{}
			}
			name := jsonx.GetString(fn, "name")
			var args any = map[string]any{}
			if s := jsonx.GetString(fn, "arguments"); s != "" {
				if parsed, err := parseJSONValue(s); err == nil {
					args = parsed
				}
				// Arguments that are not JSON cannot ride a part whose args
				// member must be an object; the call keeps its name and the
				// unusable payload drops rather than the whole call.
			}
			part := map[string]any{"functionCall": map[string]any{"name": name, "args": args}}
			sig := toolCallThoughtSignature(tm)
			if sig != "" {
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

// chatUsageToMetadata folds a chat usage object back onto the wire's usage
// metadata. The total is recomputed rather than copied: the forward fold reads
// whether the candidate count includes the thoughts from the token arithmetic,
// so a body whose numbers agree with itself always reads back the way it was
// written, whatever the source's total claimed.
func chatUsageToMetadata(usage any) map[string]any {
	m, ok := jsonx.AsMap(usage)
	if !ok {
		m = map[string]any{}
	}
	prompt, _ := jsonx.Int(m["prompt_tokens"])
	completion, _ := jsonx.Int(m["completion_tokens"])
	reasoning := 0
	if d, ok := jsonx.AsMap(m["completion_tokens_details"]); ok {
		if n, ok := jsonx.Int(d["reasoning_tokens"]); ok {
			reasoning = n
		}
	}
	candidates := completion - reasoning
	thoughts := 0
	if reasoning > 0 {
		thoughts = reasoning
	}
	out := map[string]any{
		"promptTokenCount":     prompt,
		"candidatesTokenCount": candidates,
		"totalTokenCount":      prompt + candidates + thoughts,
	}
	if thoughts > 0 {
		out["thoughtsTokenCount"] = thoughts
	}
	if d, ok := jsonx.AsMap(m["prompt_tokens_details"]); ok {
		if cached, ok := jsonx.Int(d["cached_tokens"]); ok && cached > 0 {
			out["cachedContentTokenCount"] = cached
			// The forward fold states prompt text inclusive of the cache and
			// subtracts it back out, so the inverse adds it here.
			if text, ok := jsonx.Int(d["text_tokens"]); ok {
				out["promptTokensDetails"] = []any{map[string]any{"modality": "TEXT", "tokenCount": text + cached}}
			}
		} else if text, ok := jsonx.Int(d["text_tokens"]); ok {
			out["promptTokensDetails"] = []any{map[string]any{"modality": "TEXT", "tokenCount": text}}
		}
	}
	if d, ok := jsonx.AsMap(m["completion_tokens_details"]); ok {
		if text, ok := jsonx.Int(d["text_tokens"]); ok {
			out["candidatesTokensDetails"] = []any{map[string]any{"modality": "TEXT", "tokenCount": text}}
		}
	}
	return out
}

// splitDataURLParts reads a data URL into its media type and payload, the
// response-side spelling of inline media.
func splitDataURLParts(url string) (string, string, bool) {
	media, data, err := splitDataURL(url)
	if err != nil {
		return "", "", false
	}
	return media, data, true
}
