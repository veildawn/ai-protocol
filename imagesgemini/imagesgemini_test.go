package imagesgemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// The folds here are pinned against the reference's own image adapters' output
// shapes for the same inputs (the request bodies below are ran samples; see
// .zcode/images-spec.md), and the rest are the smallest inputs that expose
// each rule this package owns: the shape discriminator, the refusal set, the
// size table's exact-hit rows and the usage fallbacks.

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func fold(t *testing.T, body string, drop bool) map[string]any {
	t.Helper()
	obj, err := jsonx.UnmarshalMap([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	out, err := ImagesToGemini(obj, drop)
	if err != nil {
		t.Fatalf("ImagesToGemini: %v", err)
	}
	return out
}

// Minimal generations: the prompt is one text part in a content with no role,
// and the response modality list is always on — in the image path's own
// snake_case spelling.
func TestImagesToGeminiGenerationsMinimal(t *testing.T) {
	out := fold(t, `{"model":"m","prompt":"a red apple on a table"}`, false)
	want := `{"contents":[{"parts":[{"text":"a red apple on a table"}]}],` +
		`"generationConfig":{"response_modalities":["IMAGE","TEXT"]}}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

// n and size land on the generation config; the model the name says takes no
// image size, and the one it does not takes the square-equivalent bucket.
func TestImagesToGeminiGenerationsNSSize(t *testing.T) {
	out := fold(t, `{"model":"gemini-2.5-flash-image","prompt":"x","n":2,"size":"1024x1024"}`, false)
	gc, _ := jsonx.AsMap(out["generationConfig"])
	if n, _ := jsonx.Int(gc["candidateCount"]); n != 2 {
		t.Fatalf("candidateCount = %s", mustJSON(gc))
	}
	ic, _ := jsonx.AsMap(gc["imageConfig"])
	if jsonx.GetString(ic, "aspectRatio") != "1:1" {
		t.Fatalf("imageConfig = %s", mustJSON(ic))
	}
	if _, has := ic["imageSize"]; has {
		t.Fatalf("2.5-flash took an imageSize: %s", mustJSON(ic))
	}

	out = fold(t, `{"model":"gemini-3-pro-image","prompt":"x","n":1,"size":"1792x1024"}`, false)
	gc, _ = jsonx.AsMap(out["generationConfig"])
	if n, _ := jsonx.Int(gc["candidateCount"]); n != 1 {
		t.Fatalf("explicit n=1 still lands: %s", mustJSON(gc))
	}
	ic, _ = jsonx.AsMap(gc["imageConfig"])
	if got := mustJSON(ic); got != `{"aspectRatio":"16:9","imageSize":"1K"}` {
		t.Fatalf("imageConfig = %s", got)
	}
}

// The dimension table is read exact-hit first: 1280x896 names 4:3 by table row
// though the nearest palette entry by log distance would be 3:2. "auto" and
// unparseable sizes land nowhere.
func TestImagesToGeminiSizeExactHitWins(t *testing.T) {
	out := fold(t, `{"prompt":"x","size":"1280x896"}`, false)
	gc, _ := jsonx.AsMap(out["generationConfig"])
	ic, _ := jsonx.AsMap(gc["imageConfig"])
	if jsonx.GetString(ic, "aspectRatio") != "4:3" {
		t.Fatalf("exact hit lost to nearest: %s", mustJSON(ic))
	}
	for _, size := range []string{"auto", "wide", "0x0", "-5x5"} {
		out := fold(t, `{"prompt":"x","size":"`+size+`"}`, false)
		gc, _ := jsonx.AsMap(out["generationConfig"])
		if _, has := gc["imageConfig"]; has {
			t.Fatalf("%s landed an imageConfig: %s", size, mustJSON(gc))
		}
	}
}

// The parameters the target has no spelling for are refused unless the caller
// asked for drops; the rest of the vocabulary rides or drops silently.
func TestImagesToGeminiRefusedParams(t *testing.T) {
	for _, p := range []string{"quality", "response_format", "mask", "background"} {
		body := `{"prompt":"x","` + p + `":"v"}`
		obj, err := jsonx.UnmarshalMap([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		_, err = ImagesToGemini(obj, false)
		if err == nil {
			t.Errorf("%s: want refusal", p)
		} else if _, ok := err.(cerr.UnsupportedParam); !ok {
			t.Errorf("%s: want cerr.UnsupportedParam, got %T (%v)", p, err, err)
		}
		if out := fold(t, body, true); strings.Contains(mustJSON(out), `"v"`) {
			t.Errorf("%s: drop path still carries it: %s", p, mustJSON(out))
		}
	}
}

// Edits: input images ride first as inlineData parts in order, the prompt
// rides last and only when non-empty, and the mime comes from the entry, a
// data URL prefix, or the payload's magic bytes.
func TestImagesToGeminiEditsShape(t *testing.T) {
	const png1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	out := fold(t, `{"prompt":"make it blue","image":[{"b64_json":"`+png1x1+`","mime_type":"image/png"}]}`, false)
	want := `{"contents":[{"parts":[{"inlineData":{"data":"` + png1x1 + `","mimeType":"image/png"}},{"text":"make it blue"}]}],` +
		`"generationConfig":{"response_modalities":["IMAGE","TEXT"]}}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}

	// Two images, no prompt: no text part at all. A single string entry is one
	// image; a data URL names its own mime.
	out = fold(t, `{"image":["`+png1x1+`","data:image/jpeg;base64,/9j/4AAQ"]}`, false)
	contents, _ := jsonx.AsSlice(out["contents"])
	content, _ := jsonx.AsMap(contents[0])
	parts, _ := jsonx.AsSlice(content["parts"])
	if len(parts) != 2 {
		t.Fatalf("parts = %s", mustJSON(content))
	}
	first, _ := jsonx.AsMap(parts[0])
	inline, _ := jsonx.AsMap(first["inlineData"])
	if jsonx.GetString(inline, "mimeType") != "image/png" {
		t.Fatalf("sniffed mime = %s", mustJSON(inline))
	}
	second, _ := jsonx.AsMap(parts[1])
	inline, _ = jsonx.AsMap(second["inlineData"])
	if jsonx.GetString(inline, "mimeType") != "image/jpeg" {
		t.Fatalf("data URL mime = %s", mustJSON(inline))
	}
}

// An edits body names the shape but carries no images: the reference raises
// rather than folding an empty edit.
func TestImagesToGeminiEditsRequiresImage(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{"prompt":"x","image":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ImagesToGemini(obj, false); err == nil {
		t.Fatal("empty image array folded")
	}
	if _, err := ImagesToGemini(obj, true); err == nil {
		t.Fatal("empty image array folded on the drop path")
	}
}

// The reverse request fold: inlineData parts come back as image entries, text
// parts join into the prompt, candidateCount becomes n and the image config
// reverses through the dimension table. Either spelling of the inline key is
// read.
func TestGeminiToImagesReverse(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"contents":[{"parts":[{"inlineData":{"mimeType":"image/png","data":"AAAB"}},
			{"inline_data":{"data":"BBB"}},{"text":"a"},{"text":"b"}]}],
		"generationConfig":{"response_modalities":["IMAGE","TEXT"],
			"candidateCount":2,"imageConfig":{"aspectRatio":"1:1","imageSize":"1K"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := GeminiToImages(obj)
	if err != nil {
		t.Fatal(err)
	}
	images, _ := jsonx.AsSlice(out["image"])
	if len(images) != 2 {
		t.Fatalf("image = %s", mustJSON(out))
	}
	first, _ := jsonx.AsMap(images[0])
	if jsonx.GetString(first, "b64_json") != "AAAB" || jsonx.GetString(first, "mime_type") != "image/png" {
		t.Fatalf("entry = %s", mustJSON(first))
	}
	second, _ := jsonx.AsMap(images[1])
	if jsonx.GetString(second, "b64_json") != "BBB" {
		t.Fatalf("snake-spelling entry lost: %s", mustJSON(second))
	}
	if jsonx.GetString(out, "prompt") != "a\nb" {
		t.Fatalf("prompt = %s", mustJSON(out))
	}
	if n, _ := jsonx.Int(out["n"]); n != 2 {
		t.Fatalf("n = %s", mustJSON(out))
	}
	if jsonx.GetString(out, "size") != "1024x1024" {
		t.Fatalf("size = %s", mustJSON(out))
	}
	if _, has := out["model"]; has {
		t.Fatalf("the wire names no model: %s", mustJSON(out))
	}

	// A body with no input images folds to the generations shape; a ratio with
	// no bucket takes the table's first row.
	bare, err := jsonx.UnmarshalMap([]byte(`{"contents":[{"parts":[{"text":"q"}]}],
		"generationConfig":{"imageConfig":{"aspectRatio":"21:9"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err = GeminiToImages(bare)
	if err != nil {
		t.Fatal(err)
	}
	if jsonx.GetString(out, "prompt") != "q" || jsonx.GetString(out, "size") != "792x336" {
		t.Fatalf("generations reverse = %s", mustJSON(out))
	}
}

// Forward and back: the shapes the fold produces survive the round trip on
// every member both sides can name. The size read back is the table's
// canonical member for the ratio and bucket (1792x1024, say, reverses to
// 1376x768 — the inverse is table-canonical by design), so the trip here uses
// a size the table itself names.
func TestImagesGeminiRequestRoundTrip(t *testing.T) {
	body := `{"model":"m","prompt":"a cup","n":2,"size":"1024x1024"}`
	out := fold(t, body, false)
	back, err := GeminiToImages(out)
	if err != nil {
		t.Fatal(err)
	}
	if jsonx.GetString(back, "prompt") != "a cup" || jsonx.GetString(back, "size") != "1024x1024" {
		t.Fatalf("round trip = %s", mustJSON(back))
	}
	if n, _ := jsonx.Int(back["n"]); n != 2 {
		t.Fatalf("round trip n = %s", mustJSON(back))
	}
}

// The response fold: every candidate's inline parts flatten into data in
// order, text parts and finish reasons are not read, the usage keeps the
// reference's nine-key shape with its modality arithmetic, and created stays
// the codec's zero.
func TestGeminiResponseToImages(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"candidates":[{"content":{"role":"model","parts":[
			{"text":"Here is your image."},
			{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}},
			{"inlineData":{"mimeType":"image/png","data":"d29ybGQ="}}]},
			"finishReason":"STOP","index":0}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290,"totalTokenCount":1295,
			"promptTokensDetails":[{"modality":"TEXT","tokenCount":5}],
			"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1290}]},
		"responseId":"resp-1"}`))
	if err != nil {
		t.Fatal(err)
	}
	out := GeminiResponseToImages(obj)
	if out["created"] != 0 {
		t.Fatalf("created = %s", mustJSON(out))
	}
	data, _ := jsonx.AsSlice(out["data"])
	if len(data) != 2 {
		t.Fatalf("data = %s", mustJSON(out))
	}
	first, _ := jsonx.AsMap(data[0])
	if jsonx.GetString(first, "b64_json") != "aGVsbG8=" {
		t.Fatalf("entry = %s", mustJSON(first))
	}
	if _, has := first["url"]; has {
		t.Fatalf("url produced: %s", mustJSON(first))
	}
	if _, has := first["revised_prompt"]; has {
		t.Fatalf("revised_prompt produced: %s", mustJSON(first))
	}
	usage, _ := jsonx.AsMap(out["usage"])
	if got := mustJSON(usage); got != `{"completion_tokens":1290,`+
		`"completion_tokens_details":{"image_tokens":1290,"text_tokens":0},`+
		`"input_tokens":5,"input_tokens_details":{"image_tokens":0,"text_tokens":5},`+
		`"output_tokens":1290,"output_tokens_details":{"image_tokens":1290,"text_tokens":0},`+
		`"prompt_tokens":5,"prompt_tokens_details":{"image_tokens":0,"text_tokens":5},`+
		`"total_tokens":1295}` {
		t.Fatalf("usage = %s", got)
	}
}

// A refused response is no branch at all on this path: the data array empties
// and the usage still counts — the reference's own behavior.
func TestGeminiResponseToImagesBlocked(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"promptFeedback":{"blockReason":"SAFETY"},
		"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5}}`))
	if err != nil {
		t.Fatal(err)
	}
	out := GeminiResponseToImages(obj)
	data, _ := jsonx.AsSlice(out["data"])
	if len(data) != 0 {
		t.Fatalf("data = %s", mustJSON(out))
	}
	usage, _ := jsonx.AsMap(out["usage"])
	if n, _ := jsonx.Int(usage["output_tokens"]); n != 0 {
		t.Fatalf("usage = %s", mustJSON(usage))
	}
	details, _ := jsonx.AsMap(usage["output_tokens_details"])
	if n, _ := jsonx.Int(details["image_tokens"]); n != 0 {
		t.Fatalf("usage = %s", mustJSON(usage))
	}
}

// The candidate-side fallback: no detail list means the whole output count is
// image tokens; a list that sums short of the headline puts the difference on
// text.
func TestGeminiResponseToImagesUsageFallbacks(t *testing.T) {
	bare, _ := jsonx.UnmarshalMap([]byte(`{"candidates":[],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290,"totalTokenCount":1295}}`))
	usage, _ := jsonx.AsMap(GeminiResponseToImages(bare)["usage"])
	details, _ := jsonx.AsMap(usage["output_tokens_details"])
	if n, _ := jsonx.Int(details["image_tokens"]); n != 1290 {
		t.Fatalf("bare fallback = %s", mustJSON(usage))
	}

	short, _ := jsonx.UnmarshalMap([]byte(`{"candidates":[],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1300,"totalTokenCount":1305,
			"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1290}]}}`))
	usage, _ = jsonx.AsMap(GeminiResponseToImages(short)["usage"])
	details, _ = jsonx.AsMap(usage["output_tokens_details"])
	if n, _ := jsonx.Int(details["text_tokens"]); n != 10 {
		t.Fatalf("short-sum fallback = %s", mustJSON(usage))
	}
}

// The reverse response fold: b64 entries become inlineData parts with the mime
// sniffed from the magic bytes, url-only entries are skipped, and the usage
// rebuilds the headline counts and the non-zero detail entries.
func TestImagesResponseToGemini(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{"created":0,
		"data":[{"b64_json":"iVBORw0KGgo="},{"url":"https://x/a.png"},
			{"b64_json":"/9j/4AAQ"}],
		"usage":{"input_tokens":5,"input_tokens_details":{"image_tokens":0,"text_tokens":5},
			"output_tokens":1290,"total_tokens":1295,
			"output_tokens_details":{"image_tokens":1290,"text_tokens":0}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out := ImagesResponseToGemini(obj)
	candidates, _ := jsonx.AsSlice(out["candidates"])
	candidate, _ := jsonx.AsMap(candidates[0])
	if jsonx.GetString(candidate, "finishReason") != "STOP" {
		t.Fatalf("candidate = %s", mustJSON(candidate))
	}
	content, _ := jsonx.AsMap(candidate["content"])
	parts, _ := jsonx.AsSlice(content["parts"])
	if len(parts) != 2 {
		t.Fatalf("parts = %s", mustJSON(content))
	}
	first, _ := jsonx.AsMap(parts[0])
	inline, _ := jsonx.AsMap(first["inlineData"])
	if jsonx.GetString(inline, "mimeType") != "image/png" || jsonx.GetString(inline, "data") != "iVBORw0KGgo=" {
		t.Fatalf("part = %s", mustJSON(first))
	}
	second, _ := jsonx.AsMap(parts[1])
	inline, _ = jsonx.AsMap(second["inlineData"])
	if jsonx.GetString(inline, "mimeType") != "image/jpeg" {
		t.Fatalf("jpeg sniff = %s", mustJSON(inline))
	}
	um, _ := jsonx.AsMap(out["usageMetadata"])
	if n, _ := jsonx.Int(um["promptTokenCount"]); n != 5 {
		t.Fatalf("usageMetadata = %s", mustJSON(um))
	}
	if n, _ := jsonx.Int(um["candidatesTokenCount"]); n != 1290 {
		t.Fatalf("usageMetadata = %s", mustJSON(um))
	}
	details, _ := jsonx.AsSlice(um["promptTokensDetails"])
	if got := mustJSON(details); got != `[{"modality":"TEXT","tokenCount":5}]` {
		t.Fatalf("prompt details = %s", got)
	}
	details, _ = jsonx.AsSlice(um["candidatesTokensDetails"])
	if got := mustJSON(details); got != `[{"modality":"IMAGE","tokenCount":1290}]` {
		t.Fatalf("candidate details = %s", got)
	}
}

// Response round trip: the fold's own output reads back to the same data
// bytes and the same headline usage.
func TestImagesGeminiResponseRoundTrip(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AAAB"}}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290,"totalTokenCount":1295,
			"promptTokensDetails":[{"modality":"TEXT","tokenCount":5}],
			"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1290}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	out := GeminiResponseToImages(obj)
	back := ImagesResponseToGemini(out)
	imagesOut, _ := jsonx.AsSlice(out["data"])
	candidates, _ := jsonx.AsSlice(back["candidates"])
	candidate, _ := jsonx.AsMap(candidates[0])
	content, _ := jsonx.AsMap(candidate["content"])
	parts, _ := jsonx.AsSlice(content["parts"])
	if len(parts) != len(imagesOut) {
		t.Fatalf("round trip changed the image count: %s", mustJSON(back))
	}
	first, _ := jsonx.AsMap(parts[0])
	inline, _ := jsonx.AsMap(first["inlineData"])
	if jsonx.GetString(inline, "data") != "AAAB" {
		t.Fatalf("round trip data = %s", mustJSON(back))
	}
	um, _ := jsonx.AsMap(back["usageMetadata"])
	if n, _ := jsonx.Int(um["candidatesTokenCount"]); n != 1290 {
		t.Fatalf("round trip usage = %s", mustJSON(um))
	}
}
