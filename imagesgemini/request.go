// Package imagesgemini folds the Images dialect (the image-generation wire:
// a generations shape `{model,prompt,n,size,...}` and an edits shape that adds
// `image[]`/`mask`) onto the gemini generateContent request body, and back.
//
// The semantics are the reference's own image adapters, translated directly:
// the image adapters are a separate path from the chat adapter, and their
// spelling quirks are reproduced verbatim — snake_case `response_modalities`
// (the chat path writes the camelCase twin of the same key), contents parts
// with no `role`, and input images as `inlineData` parts ahead of the prompt
// text. See the reference specification for the pinned rules and the samples
// that were run.
package imagesgemini

import (
	"fmt"
	"math"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// imagesVocabulary is the Images dialect's request vocabulary: the parameters
// an images body may carry that name something at all. It is the OpenAI Images
// API surface the design names, plus `background`, which the reference's edit
// adapter collects and the generation check refuses alike. Keys outside it are
// neither mapped nor refused — the reference's adapters read a fixed parameter
// set, so an unknown key rides nothing onto the wire.
var imagesVocabulary = map[string]struct{}{
	"model": {}, "prompt": {}, "n": {}, "size": {}, "quality": {},
	"response_format": {}, "image": {}, "mask": {}, "background": {},
}

// imagesRefusedParams are the images parameters the target has no spelling for
// at all. The reference refuses them at its parameter check on both shapes —
// dropping them is a caller's explicit choice, not a fold's silent one.
var imagesRefusedParams = map[string]struct{}{
	"quality": {}, "response_format": {}, "mask": {}, "background": {},
}

// ImagesToGemini folds an images request body into the target's
// generateContent request body.
//
// The body has no `model` member on the target side (the model rides the URL
// path, not the payload), so the member is consumed rather than carried. The
// two shapes share one body space and are told apart by `image`: an edits
// request always carries at least one input image and a generations request
// never carries any, so presence is the discriminator (both cannot be named
// otherwise). Parameters the target cannot represent are refused unless
// dropParams is set.
func ImagesToGemini(body map[string]any, dropParams bool) (map[string]any, error) {
	if err := rejectImagesParams(body, dropParams); err != nil {
		return nil, err
	}
	entries, hasImage, err := imageEntries(body["image"])
	if err != nil {
		return nil, err
	}
	var prompt any
	if p, ok := body["prompt"]; ok {
		s, isStr := p.(string)
		if !isStr {
			return nil, fmt.Errorf("images request prompt is not a string")
		}
		prompt = s
	}
	var parts []any
	if hasImage {
		// Edits shape: input images first, in order, prompt text last and only
		// when non-empty — exactly the reference's part assembly.
		if len(entries) == 0 {
			return nil, fmt.Errorf("images edit request requires at least one image")
		}
		for _, e := range entries {
			parts = append(parts, e.part())
		}
		if s, _ := prompt.(string); s != "" {
			parts = append(parts, map[string]any{"text": s})
		}
	} else {
		// Generations shape: the prompt is the request's one required member —
		// the caller's-side API takes it as a positional argument, so there is
		// no reference behavior for its absence to fold.
		s, ok := prompt.(string)
		if !ok {
			return nil, fmt.Errorf("images generation request carries no prompt")
		}
		parts = []any{map[string]any{"text": s}}
	}
	out := map[string]any{
		"contents": []any{map[string]any{"parts": parts}},
	}
	gc, err := imageGenerationConfig(body)
	if err != nil {
		return nil, err
	}
	out["generationConfig"] = gc
	return out, nil
}

// GeminiToImages folds a generateContent request body back into the images
// dialect. The reference has no reverse path here; this is the spec's inverse
// of the forward assembly: inlineData parts become `image` entries (an edits
// shape), text parts join into the prompt, candidateCount becomes n and the
// image config reverses to a size through the reference's own dimension
// table. Members with no images spelling — the response modality list, the
// sampling knobs, safety and cache handles — ride nothing; the static loss
// rows name them. The body names no model (on the target wire it rides the
// URL), so the images body carries no `model` member either.
func GeminiToImages(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	var images []any
	var texts []string
	contents, _ := jsonx.AsSlice(body["contents"])
	for _, raw := range contents {
		content, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		parts, _ := jsonx.AsSlice(content["parts"])
		for _, partRaw := range parts {
			part, ok := jsonx.AsMap(partRaw)
			if !ok {
				continue
			}
			inline := inlineDataOf(part)
			if inline != nil {
				entry := map[string]any{"b64_json": jsonx.GetString(inline, "data")}
				if mime := jsonx.GetString(inline, "mimeType"); mime != "" {
					entry["mime_type"] = mime
				}
				images = append(images, entry)
				continue
			}
			if text, ok := part["text"].(string); ok {
				texts = append(texts, text)
			}
		}
	}
	if len(images) > 0 {
		out["image"] = images
	}
	if prompt := strings.Join(texts, "\n"); prompt != "" {
		out["prompt"] = prompt
	}
	gc, _ := jsonx.AsMap(body["generationConfig"])
	if n, ok := jsonx.Int(gc["candidateCount"]); ok {
		out["n"] = n
	}
	ic, _ := jsonx.AsMap(gc["imageConfig"])
	if size := imageSizeReverse(jsonx.GetString(ic, "aspectRatio"), jsonx.GetString(ic, "imageSize")); size != "" {
		out["size"] = size
	}
	return out, nil
}

func rejectImagesParams(body map[string]any, drop bool) error {
	for k := range body {
		if _, refused := imagesRefusedParams[k]; !refused {
			continue
		}
		if _, isVocab := imagesVocabulary[k]; !isVocab {
			continue
		}
		if drop {
			continue
		}
		return cerr.UnsupportedParam{Param: k}
	}
	return nil
}

// imageGenerationConfig assembles the target's generationConfig: the response
// modality list is always on (snake_case — the image path's own spelling), the
// image config lands when the size folds to one, and the candidate count when
// the body names n.
func imageGenerationConfig(body map[string]any) (map[string]any, error) {
	gc := map[string]any{"response_modalities": []any{"IMAGE", "TEXT"}}
	if ic := sizeToImageConfig(jsonx.GetString(body, "size"), jsonx.GetString(body, "model")); ic != nil {
		gc["imageConfig"] = ic
	}
	if n, ok := jsonx.Int(body["n"]); ok {
		gc["candidateCount"] = n
	} else if _, present := body["n"]; present {
		return nil, fmt.Errorf("images request parameter n is not an integer")
	}
	return gc, nil
}

// imageEntry is one decoded `image` array member: base64 payload plus the MIME
// type the fold will spell on the wire.
type imageEntry struct {
	mimeType string
	data     string
}

func (e imageEntry) part() map[string]any {
	return map[string]any{"inlineData": map[string]any{"mimeType": e.mimeType, "data": e.data}}
}

// imageEntries reads the `image` member. A single object or string is one
// image, an array is many — the caller-side API takes either. The bool
// reports presence, which is the shape discriminator between edits and
// generations.
func imageEntries(v any) ([]imageEntry, bool, error) {
	items, isArray := jsonx.AsSlice(v)
	if !isArray {
		if v == nil {
			return nil, false, nil
		}
		items = []any{v}
	}
	var out []imageEntry
	for _, item := range items {
		switch e := item.(type) {
		case string:
			out = append(out, imageEntryOfDataURL(e))
		case map[string]any:
			data := jsonx.GetString(e, "b64_json")
			if data == "" {
				return nil, true, fmt.Errorf("image entry carries no b64_json")
			}
			out = append(out, imageEntry{mimeType: jsonx.GetString(e, "mime_type"), data: data})
		default:
			return nil, true, fmt.Errorf("image entry is neither a string nor an object")
		}
	}
	return out, v != nil, nil
}

// imageEntryOfDataURL resolves one string entry: a data URL names its own MIME
// type, a bare payload is sniffed from the decoded magic bytes.
func imageEntryOfDataURL(s string) imageEntry {
	if mime, data, ok := strings.Cut(s, ";base64,"); strings.HasPrefix(mime, "data:") && ok {
		return imageEntry{mimeType: strings.TrimPrefix(mime, "data:"), data: data}
	}
	return imageEntry{mimeType: sniffImageMime(s), data: s}
}

// inlineDataOf reads a part's inline bytes under either spelling of the key:
// this pair's wire writes the camelCase form, the chat-folded bodies the
// snake_case one, and a reverse fold should read both.
func inlineDataOf(part map[string]any) map[string]any {
	if inline, ok := jsonx.AsMap(part["inlineData"]); ok {
		return inline
	}
	inline, _ := jsonx.AsMap(part["inline_data"])
	return inline
}

// aspectRatioValues is the reference's aspect-ratio palette, in its literal
// order: the nearest-ratio lookup iterates in this order and keeps the first
// minimum, exactly as the reference's min() does.
var aspectRatioValues = []struct {
	ratio string
	value float64
}{
	{"1:1", 1.0 / 1}, {"1:4", 1.0 / 4}, {"1:8", 1.0 / 8}, {"2:3", 2.0 / 3},
	{"3:2", 3.0 / 2}, {"3:4", 3.0 / 4}, {"4:1", 4.0 / 1}, {"4:3", 4.0 / 3},
	{"4:5", 4.0 / 5}, {"5:4", 5.0 / 4}, {"8:1", 8.0 / 1}, {"9:16", 9.0 / 16},
	{"16:9", 16.0 / 9}, {"21:9", 21.0 / 9},
}

// sizeToAspectRatio is the reference's dimension table in its literal order.
// Order matters twice: an exact (w,h) hit reads the ratio straight off, and
// the reverse lookup (imageSizeReverse below) takes the FIRST row that matches
// a ratio and size bucket.
var sizeToAspectRatio = []struct {
	w, h  int
	ratio string
}{
	{512, 512, "1:1"}, {1024, 1024, "1:1"}, {2048, 2048, "1:1"}, {4096, 4096, "1:1"},
	{256, 1024, "1:4"}, {512, 2048, "1:4"}, {1024, 4096, "1:4"}, {2048, 8192, "1:4"},
	{192, 1536, "1:8"}, {384, 3072, "1:8"}, {768, 6144, "1:8"}, {1536, 12288, "1:8"},
	{424, 632, "2:3"}, {848, 1264, "2:3"}, {1696, 2528, "2:3"}, {3392, 5056, "2:3"},
	{632, 424, "3:2"}, {1264, 848, "3:2"}, {2528, 1696, "3:2"}, {5056, 3392, "3:2"},
	{448, 600, "3:4"}, {896, 1200, "3:4"}, {1792, 2400, "3:4"}, {3584, 4800, "3:4"},
	{1024, 256, "4:1"}, {2048, 512, "4:1"}, {4096, 1024, "4:1"}, {8192, 2048, "4:1"},
	{600, 448, "4:3"}, {1200, 896, "4:3"}, {2400, 1792, "4:3"}, {4800, 3584, "4:3"},
	{464, 576, "4:5"}, {928, 1152, "4:5"}, {1856, 2304, "4:5"}, {3712, 4608, "4:5"},
	{576, 464, "5:4"}, {1152, 928, "5:4"}, {2304, 1856, "5:4"}, {4608, 3712, "5:4"},
	{1536, 192, "8:1"}, {3072, 384, "8:1"}, {6144, 768, "8:1"}, {12288, 1536, "8:1"},
	{384, 688, "9:16"}, {768, 1376, "9:16"}, {1536, 2752, "9:16"}, {3072, 5504, "9:16"},
	{688, 384, "16:9"}, {1376, 768, "16:9"}, {2752, 1536, "16:9"}, {5504, 3072, "16:9"},
	{792, 336, "21:9"}, {1584, 672, "21:9"}, {3168, 1344, "21:9"}, {6336, 2688, "21:9"},
	{1280, 896, "4:3"}, {896, 1280, "3:4"},
}

// sizeToImageConfig folds an images `size` onto the target's imageConfig:
// aspectRatio from the exact dimension table first — the table's trailing rows
// exist precisely because the nearest-by-log-distance lookup would name a
// different ratio for them — and the nearest palette entry otherwise.
// imageSize comes from the square-equivalent side. "auto", unparseable and
// non-positive sizes name nothing and land nowhere — the reference tolerates
// them silently.
//
// imageSize is dropped for the models the reference's own fallback names: the
// registry flag the reference consults first is data this codec cannot carry,
// so the pinned rule is its fallback — a model name containing "2.5-flash"
// takes no imageSize.
func sizeToImageConfig(size, model string) map[string]any {
	w, h, ok := parseSize(size)
	if !ok {
		return nil
	}
	ic := map[string]any{"aspectRatio": aspectRatioOf(w, h)}
	if !isGeminiImageModel(model) || supportsImageSize(model) {
		ic["imageSize"] = imageSizeBucket(w, h)
	}
	return ic
}

func parseSize(size string) (int, int, bool) {
	if size == "auto" {
		return 0, 0, false
	}
	ws, hs, found := strings.Cut(strings.ToLower(size), "x")
	if !found {
		return 0, 0, false
	}
	var w, h int
	if _, err := fmt.Sscanf(ws, "%d", &w); err != nil || w <= 0 {
		return 0, 0, false
	}
	if _, err := fmt.Sscanf(hs, "%d", &h); err != nil || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}

// aspectRatioOf reads the exact dimension table first, falling back to the
// nearest palette entry by log distance — ties keep the palette's first row,
// as the reference's min() does.
func aspectRatioOf(w, h int) string {
	for _, row := range sizeToAspectRatio {
		if row.w == w && row.h == h {
			return row.ratio
		}
	}
	requested := float64(w) / float64(h)
	best := ""
	var bestDist float64
	for i, ar := range aspectRatioValues {
		dist := math.Abs(math.Log(ar.value / requested))
		if i == 0 || dist < bestDist {
			best, bestDist = ar.ratio, dist
		}
	}
	return best
}

// imageSizeBucket buckets the square-equivalent side: under 768 the small
// square, then one K each up to 3072, above that the four-K bucket.
func imageSizeBucket(w, h int) string {
	side := math.Sqrt(float64(w) * float64(h))
	switch {
	case side < 768:
		return "512"
	case side < 1536:
		return "1K"
	case side < 3072:
		return "2K"
	default:
		return "4K"
	}
}

// isGeminiImageModel is the reference's own model test: the name after any
// provider prefix contains "gemini". Models outside it speak a different wire
// (instances/parameters, not contents/generationConfig) and are not this
// dialect's target, but the test keeps the fold's branching honest.
func isGeminiImageModel(model string) bool {
	base := model
	if _, rest, ok := strings.Cut(model, "/"); ok {
		base = rest
	}
	return strings.Contains(base, "gemini")
}

// supportsImageSize is the reference's fallback rule, verbatim: models without
// registry data take the image size unless the name says 2.5-flash. The
// registry lookup that runs first in the reference is data this codec cannot
// carry; see images-spec.md §2.
func supportsImageSize(model string) bool {
	return !strings.Contains(model, "2.5-flash")
}

// imageSizeReverse is the dimension table read backwards: the first row whose
// ratio matches and whose square-equivalent buckets to the named size, or the
// ratio's first row when no bucket is named. An unknown ratio names no size.
func imageSizeReverse(ratio, size string) string {
	if ratio == "" {
		return ""
	}
	for _, row := range sizeToAspectRatio {
		if row.ratio != ratio {
			continue
		}
		if size == "" || imageSizeBucket(row.w, row.h) == size {
			return fmt.Sprintf("%dx%d", row.w, row.h)
		}
	}
	return ""
}
