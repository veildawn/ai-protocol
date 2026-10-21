package imagesgemini

// Response-direction folds between the gemini generateContent response body
// and the Images dialect's response shape. The forward fold is the reference
// image adapter's response transform; the reverse is the spec's inverse. See
// the reference specification §4-§5.

import (
	"encoding/base64"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// GeminiResponseToImages folds a generateContent response body into the
// images response shape.
//
// Every candidate's inline image parts become one flat `data` array — the
// reference scans all candidates and all parts and reads only non-empty
// `inlineData` members; text parts, finish reasons and thought signatures are
// not read. There is no refused-turn branch on this path: a blocked response
// folds to an empty `data` with the usage still counted (the reference's own
// behavior, confirmed by running it). The fold has no clock, so `created` is
// the codec's zero — the reference stamps wall time in its handler. `url` and
// `revised_prompt` are never produced: the reference leaves both unset, since
// this wire returns base64 bytes and no revised prompt.
func GeminiResponseToImages(body map[string]any) map[string]any {
	out := map[string]any{"created": 0, "data": []any{}}
	data := out["data"].([]any)
	for _, raw := range candidatesOrEmpty(body) {
		candidate, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		content, _ := jsonx.AsMap(candidate["content"])
		parts, _ := jsonx.AsSlice(content["parts"])
		for _, partRaw := range parts {
			part, ok := jsonx.AsMap(partRaw)
			if !ok {
				continue
			}
			inline, _ := jsonx.AsMap(part["inlineData"])
			if d := jsonx.GetString(inline, "data"); d != "" {
				data = append(data, map[string]any{"b64_json": d})
			}
		}
	}
	out["data"] = data
	if um, ok := jsonx.AsMap(body["usageMetadata"]); ok {
		out["usage"] = usageToImages(um)
	}
	return out
}

// ImagesResponseToGemini folds the images response shape back onto the gemini
// wire. The reference has no reverse path; this is the spec's inverse: one
// candidate carries every `b64_json` entry as an inlineData part (entries
// naming only a url are skipped — the wire's response parts are inline bytes;
// `revised_prompt` has no member to land on), with a STOP finish reason the
// wire's terminal shape requires. The usage folds back to the three headline
// counts plus the per-modality detail lists, rebuilt as the non-zero entries
// the forward fold's sums started from.
func ImagesResponseToGemini(body map[string]any) map[string]any {
	parts := []any{}
	if data, ok := jsonx.AsSlice(body["data"]); ok {
		for _, raw := range data {
			entry, ok := jsonx.AsMap(raw)
			if !ok {
				continue
			}
			d := jsonx.GetString(entry, "b64_json")
			if d == "" {
				continue
			}
			parts = append(parts, map[string]any{
				"inlineData": map[string]any{"mimeType": sniffImageMime(d), "data": d},
			})
		}
	}
	out := map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": "STOP",
			"index":        0,
		}},
	}
	if usage, ok := jsonx.AsMap(body["usage"]); ok {
		out["usageMetadata"] = usageToGemini(usage)
	}
	return out
}

// usageToImages folds usageMetadata into the images usage shape: the three
// headline counts plus image/text breakdowns summed from the wire's
// per-modality detail lists. When the candidate side names no detail list at
// all, the whole output token count is image tokens; when the details sum to
// less than the headline, the difference is text — the reference's own two
// fallbacks, kept verbatim.
func usageToImages(um map[string]any) map[string]any {
	inputTokens := tokenCount(um["promptTokenCount"])
	outputTokens := tokenCount(um["candidatesTokenCount"])
	inputImage, inputText := sumModalityDetails(um, "promptTokensDetails", "prompt_tokens_details")

	outputImage, outputText, summed := 0, 0, false
	for _, key := range []string{"candidatesTokensDetails", "candidates_tokens_details"} {
		if list, ok := jsonx.AsSlice(um[key]); ok {
			// An empty list counts as "no details" — the reference's emptiness
			// test runs on the list it found, not on the key.
			if len(list) > 0 {
				outputImage, outputText = sumModality(list)
				summed = true
			}
			break
		}
	}
	if !summed {
		outputImage = outputTokens
	} else if known := outputImage + outputText; outputTokens > known {
		outputText += outputTokens - known
	}

	details := func(image, text int) map[string]any {
		return map[string]any{"image_tokens": image, "text_tokens": text}
	}
	inputDetails := details(inputImage, inputText)
	outputDetails := details(outputImage, outputText)
	return map[string]any{
		"input_tokens":              inputTokens,
		"input_tokens_details":      inputDetails,
		"output_tokens":             outputTokens,
		"total_tokens":              tokenCount(um["totalTokenCount"]),
		"prompt_tokens":             inputTokens,
		"prompt_tokens_details":     inputDetails,
		"completion_tokens":         outputTokens,
		"completion_tokens_details": outputDetails,
		"output_tokens_details":     outputDetails,
	}
}

// usageToGemini rebuilds usageMetadata from the images usage: the headline
// counts ride their wire names, and the detail lists come back as one entry
// per non-zero modality — the mechanical inverse of the forward fold's sums.
// All-zero details list nothing.
func usageToGemini(usage map[string]any) map[string]any {
	inputTokens := tokenCount(usage["input_tokens"])
	outputTokens := tokenCount(usage["output_tokens"])
	um := map[string]any{
		"promptTokenCount":     inputTokens,
		"candidatesTokenCount": outputTokens,
		"totalTokenCount":      tokenCount(usage["total_tokens"]),
	}
	if d, ok := jsonx.AsMap(usage["input_tokens_details"]); ok {
		if list := modalityList(d); list != nil {
			um["promptTokensDetails"] = list
		}
	}
	if d, ok := jsonx.AsMap(usage["output_tokens_details"]); ok {
		if list := modalityList(d); list != nil {
			um["candidatesTokensDetails"] = list
		}
	}
	return um
}

// modalityList turns one details object into the wire's list spelling, dropping
// zero entries; nil when nothing survives.
func modalityList(details map[string]any) []any {
	var list []any
	if n := tokenCount(details["image_tokens"]); n > 0 {
		list = append(list, map[string]any{"modality": "IMAGE", "tokenCount": n})
	}
	if n := tokenCount(details["text_tokens"]); n > 0 {
		list = append(list, map[string]any{"modality": "TEXT", "tokenCount": n})
	}
	return list
}

// sumModalityDetails sums the first detail list found under the camelCase or
// snake_case key, in the reference's preference order.
func sumModalityDetails(um map[string]any, keys ...string) (image, text int) {
	for _, key := range keys {
		if list, ok := jsonx.AsSlice(um[key]); ok {
			return sumModality(list)
		}
	}
	return 0, 0
}

// sumModality totals one detail list by modality; entries that are not
// objects and modalities it cannot name contribute nothing, as in the
// reference.
func sumModality(list []any) (image, text int) {
	for _, raw := range list {
		details, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		switch upper(details["modality"]) {
		case "TEXT":
			text += tokenCount(details["tokenCount"])
		case "IMAGE":
			image += tokenCount(details["tokenCount"])
		}
	}
	return image, text
}

// tokenCount reads a wire token count: integers only — anything else counts
// as zero. Bodies decode with json.Number, so the read goes through jsonx.Int.
func tokenCount(v any) int {
	n, _ := jsonx.Int(v)
	return n
}

func upper(v any) string {
	s, _ := v.(string)
	return strings.ToUpper(s)
}

func candidatesOrEmpty(body map[string]any) []any {
	candidates, _ := jsonx.AsSlice(body["candidates"])
	return candidates
}

// sniffImageMime decodes the head of a base64 payload and names the image type
// by magic bytes, the reference's detection table; anything it cannot name —
// including a payload that does not decode — falls back to PNG.
func sniffImageMime(b64 string) string {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		// A short or padded-head payload still names its type off the decodable
		// prefix; only an undecodable payload falls through to the default.
		if raw = decodePrefix(b64); len(raw) == 0 {
			return "image/png"
		}
	}
	switch {
	case len(raw) >= 8 && string(raw[0:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(raw) >= 6 && string(raw[0:4]) == "GIF8" && raw[5] == 'a':
		return "image/gif"
	case len(raw) >= 3 && string(raw[0:3]) == "\xff\xd8\xff":
		return "image/jpeg"
	case len(raw) >= 8 && string(raw[4:8]) == "ftyp":
		return "image/heic"
	case len(raw) >= 12 && string(raw[0:4]) == "RIFF" && string(raw[8:12]) == "WEBP":
		return "image/webp"
	default:
		return "image/png"
	}
}

// decodePrefix decodes as much of the base64 head as forms complete quantum,
// for payloads whose tail is cut or carries foreign padding.
func decodePrefix(b64 string) []byte {
	n := len(b64) / 4 * 4
	if n == 0 {
		return nil
	}
	raw, err := base64.StdEncoding.DecodeString(b64[:n])
	if err != nil {
		return nil
	}
	return raw
}
