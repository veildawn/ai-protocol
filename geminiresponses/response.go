package geminiresponses

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// flaggedFinishReasons are the finish reasons that mark a whole response as
// content-policy-violating: the reference replaces every choice with one
// content-filtered turn when the prompt was blocked or the first candidate
// finished for one of these reasons.
var flaggedFinishReasons = map[string]bool{
	"SAFETY": true, "RECITATION": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true,
	"SPII": true, "IMAGE_SAFETY": true, "IMAGE_PROHIBITED_CONTENT": true,
}

// ResponsesResponseToGemini folds a Responses response body onto the target
// wire. This is the reverse of GeminiResponseToResponses, derived from the
// forward mapping: reasoning items replay as thought parts (the signature a
// replay needs rides the part), message items replay as text parts, function
// calls replay as function_call parts, and the status and usage map back onto
// a finishReason and a usageMetadata block.
func ResponsesResponseToGemini(body map[string]any) map[string]any {
	var parts []any
	if output, ok := jsonx.AsSlice(body["output"]); ok {
		for _, raw := range output {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			switch jsonx.GetString(m, "type") {
			case "reasoning":
				if text := reasoningItemText(m); text != "" {
					parts = append(parts, map[string]any{"text": text, "thought": true})
				}
				for _, b := range decodeThinkingBlocks(m["encrypted_content"]) {
					bm, ok := b.(map[string]any)
					if !ok {
						continue
					}
					// Flagged thought part: the reference's response reader
					// folds an unflagged part's text into the content, so a
					// signed block replayed for a reader carries the flag (the
					// request fold keeps the writer's unflagged spelling).
					parts = append(parts, map[string]any{
						"text":             bm["thinking"],
						"thought":          true,
						"thoughtSignature": bm["signature"],
					})
				}
			case "message":
				if content, ok := jsonx.AsSlice(m["content"]); ok {
					for _, p := range content {
						pm, ok := p.(map[string]any)
						if !ok {
							continue
						}
						if s, ok := pm["text"].(string); ok {
							parts = append(parts, map[string]any{"text": s})
						}
					}
				}
			case "function_call":
				name := jsonx.GetString(m, "name")
				var args any
				if err := json.Unmarshal([]byte(jsonx.GetString(m, "arguments")), &args); err != nil {
					// The reference refuses a call whose arguments are not one
					// JSON value; a response fold has no error to return, so
					// the call keeps its name and carries an empty argument
					// object rather than losing the call outright.
					args = map[string]any{}
				}
				// The response side of this wire spells its part keys in
				// camelCase — the spelling the wire's own responses use and
				// every reader of a response body reads.
				part := map[string]any{"functionCall": map[string]any{"name": name, "args": args}}
				if sig := callThoughtSignature(m); sig != "" {
					part["thoughtSignature"] = sig
				}
				parts = append(parts, part)
			case "image_generation_call":
				// The image item carries its bytes with no media type; the
				// wire needs one, and the wire's own image output is the type
				// stated here — a documented approximation, the same one the
				// reference's audio fold makes.
				parts = append(parts, map[string]any{"inlineData": map[string]any{
					"mimeType": "image/png",
					"data":     jsonx.GetString(m, "result"),
				}})
			}
		}
	}
	finishReason := "STOP"
	if jsonx.GetString(body, "status") == "incomplete" {
		if d, ok := jsonx.AsMap(body["incomplete_details"]); ok && jsonx.GetString(d, "reason") == "content_filter" {
			finishReason = "SAFETY"
		} else {
			finishReason = "MAX_TOKENS"
		}
	}
	return map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": finishReason,
			"index":        0,
		}},
		"usageMetadata": responsesUsageToMetadata(body["usage"]),
		"responseId":    jsonx.GetString(body, "id"),
	}
}

// GeminiResponseToResponses folds a target-wire response body onto the
// Responses dialect, reproducing the reference's two-hop result directly: the
// wire's candidates become one reasoning item (when the turn reasoned), one
// message item per candidate and one function_call item per call, in that
// order, with the finish reason deciding the status and the usage block
// folded through the chat-dialect accounting.
func GeminiResponseToResponses(body map[string]any) map[string]any {
	id := jsonx.GetString(body, "responseId")
	if blocked, usage := blockedTurn(body); blocked {
		return map[string]any{
			"id":         id,
			"created_at": 0,
			"model":      "",
			"object":     "response",
			// The blocked turn keeps its (empty) message item: the reference
			// emits one for every choice, null text included.
			"output":              []any{messageResponseItem("msg_0", nil, "incomplete")},
			"parallel_tool_calls": false,
			"status":              "incomplete",
			"text":                map[string]any{},
			"tool_choice":         "auto",
			"tools":               []any{},
			"usage":               usage,
		}
	}
	var output []any
	var callItems []any
	seq := 0
	nextID := func(prefix string) string {
		s := prefix + "_" + itoa(seq)
		seq++
		return s
	}
	reasoningEmitted := false
	candidates, _ := jsonx.AsSlice(body["candidates"])
	status := "completed"
	for i, raw := range candidates {
		cm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		parsed := parseCandidate(cm)
		if i == 0 {
			status = finishToStatus(parsed.finish)
		}
		if !reasoningEmitted && (parsed.reasoning != "" || encodedThinking(parsed.blocks) != "") {
			reasoningEmitted = true
			item := map[string]any{
				"id": nextID("rs"), "type": "reasoning",
				"status": finishToStatus(parsed.finish), "role": "assistant",
			}
			content := []any{}
			if parsed.reasoning != "" {
				content = append(content, map[string]any{"type": "output_text", "text": parsed.reasoning, "annotations": []any{}})
			}
			item["content"] = content
			if enc := encodedThinking(parsed.blocks); enc != "" {
				item["encrypted_content"] = enc
			}
			output = append(output, item)
		}
		if len(parsed.images) > 0 {
			// A turn that produced images reports them as image items in place
			// of the message item — the reference's own message extractor does
			// exactly that, and the bytes are the only thing it keeps.
			for _, img := range parsed.images {
				output = append(output, map[string]any{
					"id":     nextID("ig"),
					"type":   "image_generation_call",
					"status": imageGenerationStatus(parsed.finish),
					"result": img,
				})
			}
		} else {
			var text any
			if parsed.hasText {
				text = parsed.text
			}
			output = append(output, messageResponseItem(nextID("msg"), text, finishToStatus(parsed.finish)))
		}
		for _, call := range parsed.calls {
			callID := call.id
			if callID == "" {
				callID = nextID("call")
			}
			if call.signature != "" {
				callID += thoughtSeparator + call.signature
			}
			item := map[string]any{
				"id":        callID,
				"type":      "function_call",
				"call_id":   callID,
				"name":      call.name,
				"arguments": call.arguments,
				"status":    "completed",
			}
			if call.signature != "" {
				item["provider_specific_fields"] = map[string]any{"thought_signature": call.signature}
			}
			callItems = append(callItems, item)
		}
	}
	output = append(output, callItems...)
	if output == nil {
		output = []any{}
	}
	return map[string]any{
		"id":                  id,
		"created_at":          0,
		"model":               "",
		"object":              "response",
		"output":              output,
		"parallel_tool_calls": false,
		"status":              status,
		"text":                map[string]any{},
		"tool_choice":         "auto",
		"tools":               []any{},
		"usage":               geminiUsageToResponses(body["usageMetadata"], responseHasGrounding(body)),
	}
}

// messageResponseItem builds the one message item every choice contributes:
// its single output_text part carries the turn's text, null when the turn had
// none — the shape the reference's bridge emits rather than dropping the item.
func messageResponseItem(id string, text any, status string) map[string]any {
	return map[string]any{
		"id":     id,
		"type":   "message",
		"role":   "assistant",
		"status": status,
		"content": []any{map[string]any{
			"type": "output_text", "text": text, "annotations": []any{},
		}},
	}
}

// candidateParts is one candidate's contribution, in the vocabulary the
// Responses fold reads: the visible text (with non-image, non-audio inline
// media folded in as data URLs the way the reference's chat hop does), the
// reasoning text, the signed thinking blocks, the calls, and the images.
type candidateParts struct {
	text      string
	hasText   bool
	reasoning string
	blocks    []any
	calls     []parsedCall
	images    []string
	finish    string
}

type parsedCall struct {
	id        string
	name      string
	arguments string
	signature string
}

// parseCandidate reads one candidate, applying the finish-reason mapping and
// the tool-call override the reference applies: a turn that made any call
// reports "tool_calls" whatever the wire's finish reason said.
func parseCandidate(cm map[string]any) candidateParts {
	out := candidateParts{}
	content, _ := jsonx.AsMap(cm["content"])
	parts, _ := jsonx.AsSlice(content["parts"])
	for _, raw := range parts {
		pm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		text, hasText := pm["text"].(string)
		thought, _ := jsonx.Bool(pm["thought"])
		if hasText && thought {
			out.reasoning += text
			block := map[string]any{"type": "thinking", "thinking": text}
			if sig, ok := pm["thoughtSignature"]; ok && sig != nil {
				block["signature"] = sig
			}
			out.blocks = append(out.blocks, block)
			continue
		}
		if hasText {
			// A text part that is itself an audio data URL is the audio
			// object, not visible text.
			if mime, _, ok := splitDataURL(text); ok && strings.HasPrefix(mime, "audio/") {
				continue
			}
			out.text += text
			out.hasText = true
			continue
		}
		if fc, ok := jsonx.AsMap(firstNonNil(pm["functionCall"], pm["function_call"])); ok {
			call := parsedCall{
				id:        jsonx.GetString(fc, "id"),
				name:      jsonx.GetString(fc, "name"),
				arguments: dumpArgsJSON(fc["args"]),
				signature: jsonx.GetString(pm, "thoughtSignature"),
			}
			out.calls = append(out.calls, call)
			continue
		}
		idata, ok := jsonx.AsMap(firstNonNil(pm["inlineData"], pm["inline_data"]))
		if !ok {
			continue
		}
		mime := firstNonEmpty(jsonx.GetString(idata, "mimeType"), jsonx.GetString(idata, "mime_type"))
		data := jsonx.GetString(idata, "data")
		switch {
		case strings.HasPrefix(mime, "image/"):
			out.images = append(out.images, data)
		case strings.HasPrefix(mime, "audio/"):
			// Audio rides the chat hop's audio object, which the Responses
			// fold does not read; it drops there.
		default:
			out.text += "data:" + mime + ";base64," + data
			out.hasText = true
		}
	}
	finish := mapFinishReason(jsonx.GetString(cm, "finishReason"))
	if len(out.calls) > 0 {
		finish = "tool_calls"
	}
	out.finish = finish
	return out
}

// mapFinishReason maps the wire's finish reasons onto the chat vocabulary the
// status mapping reads. Values outside the table read as a clean stop — the
// reference warns and maps them so, refusing nothing.
func mapFinishReason(fr string) string {
	switch fr {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "LANGUAGE", "OTHER", "BLOCKLIST",
		"PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT":
		return "content_filter"
	default:
		return "stop"
	}
}

// finishToStatus maps a chat finish reason onto a Responses status: a ceiling
// cut or a content filter leaves the turn incomplete, everything else
// completes it.
func finishToStatus(fr string) string {
	switch fr {
	case "length", "content_filter", "refusal":
		return "incomplete"
	default:
		return "completed"
	}
}

// imageGenerationStatus maps a finish reason onto the status an image item
// takes — its vocabulary has no queued or cancelled member.
func imageGenerationStatus(fr string) string {
	switch fr {
	case "stop":
		return "completed"
	case "length":
		return "incomplete"
	case "content_filter", "error":
		return "failed"
	default:
		return "completed"
	}
}

// blockedTurn reports whether the whole response was content-filtered — a
// blocked prompt or a flagged first candidate — and returns the three-counter
// usage block the reference reports for such a turn.
func blockedTurn(body map[string]any) (bool, map[string]any) {
	if pf, ok := jsonx.AsMap(body["promptFeedback"]); ok && jsonx.GetString(pf, "blockReason") != "" {
		return true, blockedUsage(body)
	}
	candidates, _ := jsonx.AsSlice(body["candidates"])
	if len(candidates) > 0 {
		if cm, ok := candidates[0].(map[string]any); ok {
			if flaggedFinishReasons[jsonx.GetString(cm, "finishReason")] {
				return true, blockedUsage(body)
			}
		}
	}
	return false, nil
}

func blockedUsage(body map[string]any) map[string]any {
	um, _ := jsonx.AsMap(body["usageMetadata"])
	p, _ := jsonx.Int(um["promptTokenCount"])
	c, _ := jsonx.Int(um["candidatesTokenCount"])
	t, _ := jsonx.Int(um["totalTokenCount"])
	return map[string]any{"input_tokens": p, "output_tokens": c, "total_tokens": t}
}

// encodedThinking renders the thinking blocks a replay needs — those carrying
// a signature — as the compact JSON array the Responses reasoning item's
// encrypted_content field takes. Blocks with nothing to verify drop.
func encodedThinking(blocks []any) string {
	var preserved []any
	for _, b := range blocks {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		if truthyField(bm, "signature") || truthyField(bm, "data") {
			preserved = append(preserved, bm)
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

// callThoughtSignature reads the signature a Responses function call item
// carries: stated in its provider fields, or embedded in the id the forward
// fold wrote.
func callThoughtSignature(m map[string]any) string {
	if pf, ok := jsonx.AsMap(m["provider_specific_fields"]); ok {
		if s := jsonx.GetString(pf, "thought_signature"); s != "" {
			return s
		}
	}
	for _, key := range []string{"call_id", "id"} {
		if v := jsonx.GetString(m, key); v != "" {
			if i := strings.Index(v, thoughtSeparator); i >= 0 {
				return v[i+len(thoughtSeparator):]
			}
		}
	}
	return ""
}

// geminiUsageToResponses folds the wire's usageMetadata onto the Responses
// usage block via the chat-dialect accounting the reference's two hops apply:
// a candidate count that excludes the thinking tokens has them added back,
// prompt details are reported per modality with cached tokens subtracted, and
// a candidate count above zero always states its text share.
func geminiUsageToResponses(v any, grounded bool) map[string]any {
	um, ok := jsonx.AsMap(v)
	if !ok {
		return map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	}
	cached := intField(um["cachedContentTokenCount"])
	hasCached := cached != nil
	prompt := modalityCounts(um["promptTokensDetails"])
	cache := modalityCounts(um["cacheTokensDetails"])
	if cache.text != nil && prompt.text != nil {
		prompt.text = prompt.text.(int) - cache.text.(int)
	} else if hasCached && prompt.text != nil && cache.absent {
		prompt.text = prompt.text.(int) - cached.(int)
	}
	reasoning := intField(um["thoughtsTokenCount"])
	comp := modalityCounts(um["candidatesTokensDetails"])
	hasCompDetails := hasKey(um, "candidatesTokensDetails")
	cand, _ := jsonx.Int(um["candidatesTokenCount"])
	if cand > 0 {
		hasCompDetails = true
		if comp.text == nil {
			comp.text = cand - intOf(comp.image) - intOf(comp.audio) - intOf(comp.video)
		}
	}
	if reasoning != nil {
		hasCompDetails = true
	}
	// The wire bills tool-use prompt tokens as input tokens unless the turn
	// used search grounding, whose per-request fee replaces them.
	billableTool := intField(um["toolUsePromptTokenCount"])
	if billableTool != nil && grounded {
		billableTool = nil
	}
	promptTokens, _ := jsonx.Int(um["promptTokenCount"])
	total, _ := jsonx.Int(um["totalTokenCount"])
	completion := cand
	if !candidateCountInclusive(um) && intOf(reasoning) > 0 {
		completion = cand + intOf(reasoning)
	}
	cu := map[string]any{
		"prompt_tokens":     promptTokens + intOf(billableTool),
		"completion_tokens": completion,
		"total_tokens":      total,
		"prompt_tokens_details": map[string]any{
			"cached_tokens":   cached,
			"audio_tokens":    prompt.audio,
			"text_tokens":     prompt.text,
			"image_tokens":    prompt.image,
			"video_tokens":    prompt.video,
			"tool_use_tokens": billableTool,
		},
	}
	if hasCompDetails {
		cu["completion_tokens_details"] = map[string]any{
			"audio_tokens":     comp.audio,
			"image_tokens":     comp.image,
			"text_tokens":      comp.text,
			"video_tokens":     comp.video,
			"reasoning_tokens": reasoning,
		}
	}
	return chatUsageToResponses(cu)
}

// chatUsageToResponses folds a chat-shaped usage block onto the Responses
// spellings: the totals pass through as reported, the input breakdown always
// states its cached share once it exists at all, and the output breakdown
// always states its reasoning share.
func chatUsageToResponses(cu map[string]any) map[string]any {
	in, _ := jsonx.Int(cu["prompt_tokens"])
	out, _ := jsonx.Int(cu["completion_tokens"])
	total, _ := jsonx.Int(cu["total_tokens"])
	u := map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": total}
	if d, ok := jsonx.AsMap(cu["prompt_tokens_details"]); ok {
		det := map[string]any{"cached_tokens": 0}
		if v, ok := jsonx.Int(d["cached_tokens"]); ok {
			det["cached_tokens"] = v
		}
		for _, k := range []string{"text_tokens", "audio_tokens"} {
			if v, ok := jsonx.Int(d[k]); ok {
				det[k] = v
			}
		}
		u["input_tokens_details"] = det
	}
	if d, ok := jsonx.AsMap(cu["completion_tokens_details"]); ok {
		det := map[string]any{"reasoning_tokens": 0}
		if v, ok := jsonx.Int(d["reasoning_tokens"]); ok {
			det["reasoning_tokens"] = v
		}
		for _, k := range []string{"audio_tokens", "text_tokens", "image_tokens"} {
			if v, ok := jsonx.Int(d[k]); ok {
				det[k] = v
			}
		}
		u["output_tokens_details"] = det
	}
	return u
}

// responsesUsageToMetadata folds a Responses usage block back onto the wire's
// usageMetadata: the prompt count rides as reported, the candidate count sheds
// the reasoning share (stated separately when positive), and the total is
// recomputed so the block is self-consistent with either reading of the
// candidate count. The tool-use count has no inverse — the Responses block
// folds it into its input count — so it is never written.
func responsesUsageToMetadata(v any) map[string]any {
	um := map[string]any{}
	m, ok := jsonx.AsMap(v)
	if !ok {
		return um
	}
	prompt, _ := jsonx.Int(m["input_tokens"])
	output, _ := jsonx.Int(m["output_tokens"])
	reasoning := 0
	if d, ok := jsonx.AsMap(m["output_tokens_details"]); ok {
		if n, ok := jsonx.Int(d["reasoning_tokens"]); ok {
			reasoning = n
		}
	}
	cand := output - reasoning
	if reasoning < 0 {
		cand = output
		reasoning = 0
	}
	um["promptTokenCount"] = prompt
	um["candidatesTokenCount"] = cand
	if reasoning > 0 {
		um["thoughtsTokenCount"] = reasoning
	}
	um["totalTokenCount"] = prompt + cand + reasoning
	if d, ok := jsonx.AsMap(m["input_tokens_details"]); ok {
		if n, ok := jsonx.Int(d["cached_tokens"]); ok && n > 0 {
			um["cachedContentTokenCount"] = n
		}
		if n, ok := jsonx.Int(d["text_tokens"]); ok {
			um["promptTokensDetails"] = []any{modalityCount("TEXT", n)}
		}
	}
	if d, ok := jsonx.AsMap(m["output_tokens_details"]); ok {
		if n, ok := jsonx.Int(d["text_tokens"]); ok {
			um["candidatesTokensDetails"] = []any{modalityCount("TEXT", n)}
		}
	}
	return um
}

func modalityCount(modality string, n int) map[string]any {
	return map[string]any{"modality": modality, "tokenCount": n}
}

// dumpArgsJSON renders decoded arguments as the JSON string the reference's
// chat hop carries them in: the default separators of its own encoder — a
// space after every comma and colon — with map keys sorted, because the
// decode the fold starts from keeps no insertion order to restore. Absent
// arguments read as the empty object, the placeholder the hop itself uses.
func dumpArgsJSON(v any) string {
	if v == nil {
		return "{}"
	}
	var b strings.Builder
	writeDumpJSON(&b, v)
	if b.Len() == 0 {
		return "{}"
	}
	return b.String()
}

// writeDumpJSON renders one decoded-JSON value with the spacing described
// above, escaping nothing an ASCII payload does not need.
func writeDumpJSON(b *strings.Builder, v any) {
	switch t := v.(type) {
	case map[string]any:
		if len(t) == 0 {
			b.WriteString("{}")
			return
		}
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writeDumpJSONString(b, k)
			b.WriteString(": ")
			writeDumpJSON(b, t[k])
		}
		b.WriteByte('}')
	case []any:
		if len(t) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		for i, item := range t {
			if i > 0 {
				b.WriteString(", ")
			}
			writeDumpJSON(b, item)
		}
		b.WriteByte(']')
	case json.Number:
		if t.String() == "" {
			b.WriteString("0")
			return
		}
		b.WriteString(t.String())
	case string:
		writeDumpJSONString(b, t)
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	default:
		// A value the decode never produces (e.g. a raw float): let the
		// standard encoder spell it rather than guess.
		raw, err := json.Marshal(v)
		if err != nil {
			b.WriteString("null")
			return
		}
		b.Write(raw)
	}
}

// writeDumpJSONString quotes one key or string value, leaving non-ASCII as the
// bytes it came in and escaping exactly the characters a JSON string must.
func writeDumpJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
				continue
			}
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
}

// candidateHasGrounding reports whether one candidate used search grounding.
func candidateHasGrounding(cand map[string]any) bool {
	if gm, ok := jsonx.AsMap(cand["groundingMetadata"]); ok {
		if q, ok := jsonx.AsSlice(gm["webSearchQueries"]); ok && len(q) > 0 {
			return true
		}
		if chunks, ok := jsonx.AsSlice(gm["groundingChunks"]); ok {
			for _, c := range chunks {
				if chm, ok := c.(map[string]any); ok {
					if _, has := chm["maps"]; has {
						return true
					}
				}
			}
		}
		return jsonx.GetString(gm, "googleMapsWidgetContextToken") != ""
	}
	return false
}

// candidateCountInclusive reports whether the candidate count already
// includes the thinking tokens: the wire states it by making the non-thinking
// totals add up, and anything else means the thinking tokens ride separately.
func candidateCountInclusive(um map[string]any) bool {
	p, _ := jsonx.Int(um["promptTokenCount"])
	c, _ := jsonx.Int(um["candidatesTokenCount"])
	t, _ := jsonx.Int(um["toolUsePromptTokenCount"])
	total, _ := jsonx.Int(um["totalTokenCount"])
	return p+c+t == total
}

// responseHasGrounding reports whether the turn used search grounding, which
// the wire bills per request instead of folding into the input count.
func responseHasGrounding(body any) bool {
	um, ok := body.(map[string]any)
	if !ok {
		return false
	}
	candidates, ok := jsonx.AsSlice(um["candidates"])
	if !ok {
		return false
	}
	for _, raw := range candidates {
		if cm, ok := raw.(map[string]any); ok && candidateHasGrounding(cm) {
			return true
		}
	}
	return false
}

// modalityTotals accumulates a token count per modality out of a details
// list, mapping DOCUMENT onto the text share. Absent modalities stay nil,
// which the chat accounting reads as "not reported".
type modalitySet struct {
	audio, image, text, video any
	// absent marks a details list that carried no cache split at all — the
	// signal that implicit caching should come off the text share instead.
	absent bool
}

func modalityCounts(v any) modalitySet {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return modalitySet{absent: true}
	}
	add := func(slot *any, n int) {
		if *slot == nil {
			*slot = n
			return
		}
		*slot = (*slot).(int) + n
	}
	var out modalitySet
	for _, raw := range list {
		dm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		n, _ := jsonx.Int(firstNonNil(dm["tokenCount"], dm["token_count"]))
		switch strings.ToUpper(jsonx.GetString(dm, "modality")) {
		case "AUDIO":
			add(&out.audio, n)
		case "IMAGE":
			add(&out.image, n)
		case "TEXT", "DOCUMENT":
			add(&out.text, n)
		case "VIDEO":
			add(&out.video, n)
		}
	}
	return out
}

// intField reads a value that is either absent or a number, keeping the
// distinction the accounting needs.
func intField(v any) any {
	if v == nil {
		return nil
	}
	if n, ok := jsonx.Int(v); ok {
		return n
	}
	return nil
}

func intOf(v any) int {
	if n, ok := v.(int); ok {
		return n
	}
	return 0
}

func hasKey(m map[string]any, key string) bool {
	_, ok := m[key]
	return ok
}

// itoa renders a small non-negative integer without the import this file
// does not otherwise need.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
