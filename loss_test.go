package protocol

import (
	"encoding/json"
	"testing"
)

func TestBodyLossesChatToMessages(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7,"frequency_penalty":0.5}`)
	got := BodyLosses(Chat, Messages, body, ConvertOptions{})
	if !got[LossSeed] || !got[LossPenaltyControls] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesMissingBodyIsUnknownContent(t *testing.T) {
	got := BodyLosses(Chat, Messages, []byte(`{`), ConvertOptions{})
	if !got[LossUnknownContent] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesFileReference(t *testing.T) {
	chat := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"file","file":{"file_id":"f"}}]}]}`)
	got := BodyLosses(Chat, Messages, chat, ConvertOptions{})
	if !got[LossFileReference] {
		t.Fatalf("losses=%v", got)
	}
	responses := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"x"},{"type":"input_file","file_id":"f"}]}]}`)
	got = BodyLosses(Responses, Messages, responses, ConvertOptions{})
	if !got[LossFileReference] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesInvalidToolArguments(t *testing.T) {
	chat := []byte(`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"["}}]}]}`)
	got := BodyLosses(Chat, Responses, chat, ConvertOptions{})
	if !got[LossToolArguments] {
		t.Fatalf("losses=%v", got)
	}
	responses := []byte(`{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":"[]"}]}`)
	got = BodyLosses(Responses, Messages, responses, ConvertOptions{})
	if !got[LossToolArguments] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesMatchConverterRejection(t *testing.T) {
	bodies := map[string][]byte{
		"seed":       []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7}`),
		"logit_bias": []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"logit_bias":{"x":1}}`),
		"logprobs":   []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"logprobs":true}`),
	}
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			// A fallback keeps the required-field rule out of this test's way:
			// the assertion is about LOSSES, not about max_tokens.
			losses := BodyLosses(Chat, Messages, body, ConvertOptions{MaxTokensFallback: 8192})
			_, err := ConvertRequestWith(Chat, Messages, body, ConvertOptions{MaxTokensFallback: 8192})
			if losses != nil && err == nil {
				t.Fatalf("loss=%v but conversion succeeded", losses)
			}
			if losses == nil && err != nil {
				t.Fatalf("no loss but conversion failed: %v", err)
			}
		})
	}
}

func TestConvertRequestWithDropsUnsupportedParams(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7}`)
	got, err := ConvertRequestWith(Chat, Messages, body, ConvertOptions{DropParams: true, MaxTokensFallback: 8192})
	if err != nil {
		t.Fatalf("drop params conversion failed: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(got.Body, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["seed"]; ok {
		t.Fatalf("seed was not dropped: %s", got.Body)
	}
}

// StaticLosses' name sets are part of the loss API's contract: the six pairs
// that predate the generateContent wire keep their sets name-for-name, and the
// six that touch it carry only what that wire cannot express (logprobs and
// candidate counts fold onto it, so they stop being losses except where the
// target is Messages or Responses, whose own rows already list them). The
// images pair carries the native-param row both ways — the images extras have
// no wire spelling, and the wire's logprobs have no images spelling — while
// the candidate count folds to n, so multi-choice is not a loss there.
func TestStaticLossesNameSets(t *testing.T) {
	tests := []struct {
		from, to Dialect
		want     []Loss
	}{
		{Chat, Messages, []Loss{LossFileReference, LossLogitBias, LossLogprobs, LossMultiChoice, LossPenaltyControls, LossSeed, LossToolArguments, LossUnknownContent}},
		{Chat, Responses, []Loss{LossLogitBias, LossLogprobs, LossMultiChoice, LossToolArguments, LossUnknownContent}},
		{Messages, Chat, []Loss{LossLogitBias, LossLogprobs, LossMultiChoice, LossToolArguments, LossUnknownContent}},
		{Messages, Responses, []Loss{LossLogitBias, LossLogprobs, LossMultiChoice, LossToolArguments, LossUnknownContent}},
		{Responses, Chat, []Loss{LossLogitBias, LossLogprobs, LossMultiChoice, LossToolArguments, LossUnknownContent}},
		{Responses, Messages, []Loss{LossFileReference, LossLogitBias, LossLogprobs, LossMultiChoice, LossStructuredOutput, LossToolArguments, LossUnknownContent}},
		{Chat, Gemini, []Loss{LossLogitBias, LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Messages, Gemini, []Loss{LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Responses, Gemini, []Loss{LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Gemini, Chat, []Loss{LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Gemini, Messages, []Loss{LossLogprobs, LossMultiChoice, LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Gemini, Responses, []Loss{LossLogprobs, LossMultiChoice, LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Images, Gemini, []Loss{LossNativeParam, LossToolArguments, LossUnknownContent}},
		{Gemini, Images, []Loss{LossLogprobs, LossNativeParam, LossToolArguments, LossUnknownContent}},
	}
	for _, tc := range tests {
		got := StaticLosses(tc.from, tc.to)
		if len(got) != len(tc.want) {
			t.Fatalf("StaticLosses(%s→%s) = %d losses, want %d", tc.from, tc.to, len(got), len(tc.want))
		}
		for _, loss := range tc.want {
			if !got[loss] {
				t.Fatalf("StaticLosses(%s→%s) is missing %q", tc.from, tc.to, loss)
			}
		}
	}
}

// LossNativeParam fires on the keys only the source wire defines — whichever
// fold direction reads the body.
func TestBodyLossesNativeParam(t *testing.T) {
	tests := []struct {
		name     string
		from, to Dialect
		body     string
	}{
		{"chat user", Chat, Gemini, `{"model":"m","messages":[{"role":"user","content":"x"}],"user":"u1"}`},
		{"chat context_management", Chat, Gemini, `{"model":"m","messages":[{"role":"user","content":"x"}],"context_management":{"edits":[]}}`},
		{"messages metadata", Messages, Gemini, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],"metadata":{"user_id":"u1"}}`},
		{"responses reasoning summary", Responses, Gemini, `{"model":"m","input":"hi","reasoning":{"effort":"low","summary":"auto"}}`},
		{"gemini safetySettings", Gemini, Chat, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"safetySettings":[{"category":"H","threshold":"B"}]}`},
		{"gemini cachedContent", Gemini, Messages, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"cachedContent":"cc"}`},
		{"gemini speechConfig", Gemini, Responses, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"generationConfig":{"speechConfig":{"voice":{}}}}`},
		{"gemini native tool", Gemini, Chat, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"tools":[{"url_context":{}}]}`},
		{"gemini server-side invocations", Gemini, Chat, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],"toolConfig":{"includeServerSideToolInvocations":true}}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := BodyLosses(tc.from, tc.to, []byte(tc.body), ConvertOptions{})
			if !got[LossNativeParam] {
				t.Fatalf("BodyLosses(%s→%s) missing native_param: %v", tc.from, tc.to, got)
			}
		})
	}
}

// The keys that DO fold onto the generateContent wire must not report a
// native-param loss: logprobs, seed, penalties, candidate counts, structured
// output and file references all have a spelling there.
func TestBodyLossesGeminiCarriesTheFordableParams(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"file","file":{"file_data":"data:application/pdf;base64,AA=="}}]}],"logprobs":true,"top_logprobs":3,"seed":7,"presence_penalty":0.1,"n":2,"response_format":{"type":"json_schema","json_schema":{"name":"o","schema":{"type":"object"}}}}`)
	got := BodyLosses(Chat, Gemini, body, ConvertOptions{})
	for _, loss := range []Loss{LossNativeParam, LossLogprobs, LossMultiChoice, LossSeed, LossPenaltyControls, LossStructuredOutput, LossFileReference} {
		if got[loss] {
			t.Fatalf("BodyLosses(chat→gemini) reported %q for a body the wire can carry: %v", loss, got)
		}
	}
}

// The images pair's native-param rows: the images extras the wire cannot
// spell fire in both refusal paths, and the wire-side knobs that do not fold
// fire on the way back — while the ones that do fold stay silent.
func TestBodyLossesImagesPair(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"images quality", `{"model":"m","prompt":"x","quality":"high"}`},
		{"images response_format", `{"model":"m","prompt":"x","response_format":"b64_json"}`},
		{"images background", `{"model":"m","prompt":"x","background":"transparent"}`},
		{"images edits mask", `{"model":"m","prompt":"x","image":[{"b64_json":"AA"}],"mask":"m"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := BodyLosses(Images, Gemini, []byte(tc.body), ConvertOptions{})
			if !got[LossNativeParam] {
				t.Fatalf("BodyLosses(images→gemini) missing native_param: %v", got)
			}
		})
	}

	native := []byte(`{"contents":[{"role":"user","parts":[{"text":"x"}]}],"safetySettings":[{"category":"H","threshold":"B"}]}`)
	if got := BodyLosses(Gemini, Images, native, ConvertOptions{}); !got[LossNativeParam] {
		t.Fatalf("BodyLosses(gemini→images) missing native_param: %v", got)
	}

	// The candidate count folds to n and the generation image config folds to
	// a size, so neither reports a loss on this pair.
	folds := []byte(`{"contents":[{"role":"user","parts":[{"text":"x"}]}],"generationConfig":{"imageConfig":{"aspectRatio":"1:1"},"candidateCount":2}}`)
	got := BodyLosses(Gemini, Images, folds, ConvertOptions{})
	if got[LossNativeParam] {
		t.Fatalf("BodyLosses(gemini→images) reported native_param for a body the images wire can carry: %v", got)
	}
	nBody := []byte(`{"model":"m","prompt":"x","n":2}`)
	got = BodyLosses(Images, Gemini, nBody, ConvertOptions{})
	if got[LossNativeParam] || got[LossMultiChoice] {
		t.Fatalf("BodyLosses(images→gemini) reported %v for a body the wire can carry", got)
	}
}
