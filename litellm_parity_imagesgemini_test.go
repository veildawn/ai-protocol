package protocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// This file pins the Images<->Gemini folds against LiteLLM v1.100.1.
//
// Truth source for the images->gemini request direction and the
// gemini->images response direction (the reference has native image adapters
// there — a separate path from its chat adapter): every expected body below
// was produced by running the reference from the repo venv:
//
//	python3
//	  litellm/llms/gemini/image_generation/transformation.py:
//	    GoogleImageGenConfig.map_openai_params -> transform_image_generation_request
//	  litellm/llms/gemini/image_edit/transformation.py:
//	    GeminiImageEditConfig.map_openai_params -> transform_image_edit_request
//	  ... -> transform_image_generation_response / transform_image_edit_response
//
// Top-level refusal behavior (quality/response_format/mask/background without
// drop) comes from the reference's own checks: utils.py get_optional_params_
// image_gen._check_valid_arg for generations, images/utils.py
// get_optional_params_image_edit for edits.
//
// The gemini->images request direction and the images->gemini response
// direction have no forward path in the reference: their expectations are
// derived from the spec's inverse rules (the reference specification §5, each row
// cites the forward mapping it inverts) and are validated below by feeding
// this codec's output back through its own forward folds.
//
// Two documented divergences from the reference's bytes, both codec-fact not
// mapping-fact: `created` is 0 here (the reference's handler stamps wall
// time; this library has no clock), and the edits `image` member's JSON
// spelling ({b64_json,mime_type} entries) is this dialect's own convention —
// the reference takes file handles there, and everything downstream of that
// member matches its output.

// imagesCanonical renders a decoded JSON value so two bodies compare
// independent of key order (Go maps marshal with sorted keys).
func imagesCanonical(t *testing.T, v string) string {
	t.Helper()
	var parsed any
	if err := json.Unmarshal([]byte(v), &parsed); err != nil {
		t.Fatalf("bad JSON %q: %v", v, err)
	}
	b, err := json.Marshal(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func imagesConvertRequest(t *testing.T, body string, drop bool) string {
	t.Helper()
	out, err := ConvertRequestWith(Images, Gemini, []byte(body), ConvertOptions{DropParams: drop})
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return string(out.Body)
}

// TestImagesGeminiParityImagesToGemini pins the request fold against
// reference-ran truth. The generationConfig key is snake_case
// response_modalities — the image path's own spelling (common_utils.py:272),
// unlike the chat path's camelCase twin; the contents parts carry no role
// (transformation.py:139-140, image_edit/transformation.py:103-107).
func TestImagesGeminiParityImagesToGemini(t *testing.T) {
	const png1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	cases := []struct{ name, images, want string }{
		// Minimal: prompt -> one text part, no role; modality list always on
		// (common_utils.py:272).
		{"minimal", `{"model":"gemini-2.5-flash-image","prompt":"a red apple on a table"}`,
			`{"contents":[{"parts":[{"text":"a red apple on a table"}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"]}}`},
		// n and size land on the config (common_utils.py:143-144, 284-293, 146-157);
		// the 2.5-flash fallback takes no imageSize (common_utils.py:115-123).
		{"n size", `{"model":"gemini-2.5-flash-image","prompt":"a red apple on a table","n":2,"size":"1024x1024"}`,
			`{"contents":[{"parts":[{"text":"a red apple on a table"}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"],"imageConfig":{"aspectRatio":"1:1"},"candidateCount":2}}`},
		// 1792x1024 hits the dimension table exactly; n=1 still lands
		// (common_utils.py:37-96, 318-337).
		{"wide size", `{"model":"gemini-2.5-flash-image","prompt":"x","n":1,"size":"1792x1024"}`,
			`{"contents":[{"parts":[{"text":"x"}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"],"imageConfig":{"aspectRatio":"16:9"},"candidateCount":1}}`},
		// Edits: inlineData parts first in order, non-empty prompt last
		// (image_edit/transformation.py:94-117).
		{"edits one image", `{"model":"gemini-2.5-flash-image","prompt":"make it blue","image":[{"b64_json":"` + png1x1 + `","mime_type":"image/png"}]}`,
			`{"contents":[{"parts":[{"inlineData":{"data":"` + png1x1 + `","mimeType":"image/png"}},{"text":"make it blue"}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"]}}`},
		// Edits with n/size; two images and no prompt means no text part.
		{"edits n size two images", `{"model":"gemini-2.5-flash-image","image":["` + png1x1 + `"],"n":2,"size":"1024x1024"}`,
			`{"contents":[{"parts":[{"inlineData":{"data":"` + png1x1 + `","mimeType":"image/png"}}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"],"imageConfig":{"aspectRatio":"1:1"},"candidateCount":2}}`},
		// size "auto" names nothing (common_utils.py:298-300); the rest of the
		// config is unchanged.
		{"size auto", `{"model":"gemini-2.5-flash-image","prompt":"x","size":"auto"}`,
			`{"contents":[{"parts":[{"text":"x"}]}],"generationConfig":{"response_modalities":["IMAGE","TEXT"]}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := imagesConvertRequest(t, tc.images, false); imagesCanonical(t, got) != imagesCanonical(t, tc.want) {
				t.Fatalf("got  %s\nwant %s", got, tc.want)
			}
		})
	}
}

// The parameters the reference refuses at its own checks come back as
// UnsupportedParamError unless the caller opted into omission; with it, the
// dropped body equals the minimal fold.
func TestImagesGeminiParityRefusals(t *testing.T) {
	for _, p := range []string{"quality", "response_format", "background", "mask"} {
		body := `{"model":"m","prompt":"x","` + p + `":"v"}`
		_, err := ConvertRequest(Images, Gemini, []byte(body))
		var unsupported UnsupportedParamError
		if !errors.As(err, &unsupported) || unsupported.Param != p {
			t.Fatalf("%s: err = %v, want UnsupportedParamError{%s}", p, err, p)
		}
		dropped := imagesConvertRequest(t, body, true)
		if strings.Contains(dropped, `"v"`) {
			t.Fatalf("%s: drop path kept the param: %s", p, dropped)
		}
	}
}

// TestImagesGeminiParityGeminiResponseToImages pins the response fold against
// reference-ran truth: every candidate's inline parts flatten to b64_json
// entries, the usage keeps the reference's nine-key shape with its modality
// arithmetic (image_usage_transformation.py:37-65), a refused response is no
// branch at all (data empties, usage still counts — ran behavior), and the
// edit response folds through the same shape
// (image_edit/transformation.py:119-154).
func TestImagesGeminiParityGeminiResponseToImages(t *testing.T) {
	gemini := `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"Here is your image."},
		{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}},
		{"inlineData":{"mimeType":"image/png","data":"d29ybGQ="}}]},
		"finishReason":"STOP","index":0}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290,"totalTokenCount":1295,
		"promptTokensDetails":[{"modality":"TEXT","tokenCount":5}],
		"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1290}]},
		"responseId":"resp-1"}`
	want := `{"created":0,"data":[{"b64_json":"aGVsbG8="},{"b64_json":"d29ybGQ="}],
		"usage":{"input_tokens":5,"input_tokens_details":{"image_tokens":0,"text_tokens":5},
		"output_tokens":1290,"total_tokens":1295,"prompt_tokens":5,
		"prompt_tokens_details":{"image_tokens":0,"text_tokens":5},
		"completion_tokens":1290,"completion_tokens_details":{"image_tokens":1290,"text_tokens":0},
		"output_tokens_details":{"image_tokens":1290,"text_tokens":0}}}`
	out, err := ConvertResponse(Gemini, Images, []byte(gemini))
	if err != nil {
		t.Fatal(err)
	}
	if got := imagesCanonical(t, string(out)); got != imagesCanonical(t, want) {
		t.Fatalf("got  %s\nwant %s", got, want)
	}

	// Ran blocked behavior: promptFeedback.blockReason empties the data array,
	// the usage still counts, and the no-details fallback books the output
	// count as image tokens (image_usage_transformation.py:47-48).
	blocked := `{"promptFeedback":{"blockReason":"SAFETY"},
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290,"totalTokenCount":1295}}`
	out, err = ConvertResponse(Gemini, Images, []byte(blocked))
	if err != nil {
		t.Fatal(err)
	}
	got := imagesCanonical(t, string(out))
	if !strings.Contains(got, `"data":[]`) || !strings.Contains(got, `"output_tokens_details":{"image_tokens":1290,"text_tokens":0}`) {
		t.Fatalf("blocked response = %s", got)
	}
}

// The derived directions (spec §5): the reverse request fold reads back the
// members the forward fold wrote, and the reverse response fold feeds this
// codec's own forward response output — the loop closes on the members both
// sides can name.
func TestImagesGeminiParityDerivedReverse(t *testing.T) {
	forward := imagesConvertRequest(t, `{"model":"m","prompt":"a cup","n":2,"size":"1024x1024"}`, false)
	back, err := ConvertRequest(Gemini, Images, []byte(forward))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"n":2,"prompt":"a cup","size":"1024x1024"}`
	if got := imagesCanonical(t, string(back.Body)); got != imagesCanonical(t, want) {
		t.Fatalf("request reverse got  %s\nwant %s", got, want)
	}

	editsForward := imagesConvertRequest(t, `{"model":"m","image":[{"b64_json":"AAAB","mime_type":"image/png"},{"b64_json":"BBB"}],"prompt":"edit this"}`, false)
	back, err = ConvertRequest(Gemini, Images, []byte(editsForward))
	if err != nil {
		t.Fatal(err)
	}
	want = `{"image":[{"b64_json":"AAAB","mime_type":"image/png"},{"b64_json":"BBB"}],"prompt":"edit this"}`
	if got := imagesCanonical(t, string(back.Body)); got != imagesCanonical(t, want) {
		t.Fatalf("edits reverse got  %s\nwant %s", got, want)
	}

	resp := `{"created":0,"data":[{"b64_json":"AAAB"}],
		"usage":{"input_tokens":5,"input_tokens_details":{"image_tokens":0,"text_tokens":5},
		"output_tokens":1290,"total_tokens":1295,"prompt_tokens":5,
		"prompt_tokens_details":{"image_tokens":0,"text_tokens":5},
		"completion_tokens":1290,"completion_tokens_details":{"image_tokens":1290,"text_tokens":0},
		"output_tokens_details":{"image_tokens":1290,"text_tokens":0}}}`
	gemResp, err := ConvertResponse(Images, Gemini, []byte(resp))
	if err != nil {
		t.Fatal(err)
	}
	backAgain, err := ConvertResponse(Gemini, Images, gemResp)
	if err != nil {
		t.Fatal(err)
	}
	if got := imagesCanonical(t, string(backAgain)); got != imagesCanonical(t, resp) {
		t.Fatalf("response round trip got  %s\nwant %s", got, resp)
	}
}
