package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// This file pins the Responses<->Gemini response and stream folds against
// LiteLLM v1.100.1, run through the reference's virtualenv. The reference
// has no direct path between these dialects; every forward expectation was
// produced by composing its two hops —
//   VGS._transform_google_generate_content_to_openai_model_response
//   (litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2406-2505)
// then
//   RSP.transform_chat_completion_response_to_responses_api_response
//   (litellm/responses/litellm_completion_transformation/transformation.py:2211-2360)
// and, for streams,
//   VGS.ModelResponseIterator.chunk_parser (VGS:3239-3284) ->
//   LiteLLMCompletionStreamingIterator
//   (litellm/responses/litellm_completion_transformation/streaming_iterator.py:72) —
// and pasting the result. Where the reference fills a field with a uuid or a
// clock reading, this library's deterministic stand-ins are asserted instead
// (spec D10); everything semantic is pinned to the reference bytes.
//
// The reverse directions (ResponsesResponseToGemini, PipeResponsesToGemini)
// have no forward path in the reference; their expectations are the spec's
// inverse rules, and each body was fed back through the reference's forward
// composition to confirm it restores the original (script:
// the reverse-round-trip audit script).

// gemRespParityResponse folds one gemini response body onto the Responses
// dialect and returns the decoded result.
func gemRespParityResponse(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := ConvertResponse(Gemini, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// A plain text turn: one message item, the finish mapping, and the usage
// folded through the chat accounting (reference rs_text run).
func TestGeminiResponseToResponsesParityText(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hello there"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-1"}`)
	if m["id"] != "resp-1" || m["object"] != "response" || m["status"] != "completed" {
		t.Fatalf("shell = %#v", m)
	}
	output := m["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %#v", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" || item["status"] != "completed" {
		t.Fatalf("item = %#v", item)
	}
	if item["id"] != "msg_0" {
		t.Fatalf("id = %v (deterministic stand-in for the reference's uuid, D10)", item["id"])
	}
	part := item["content"].([]any)[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "hello there" || fmtJSON(part["annotations"]) != `[]` {
		t.Fatalf("part = %#v", part)
	}
	usage := m["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(3) || usage["total_tokens"] != float64(8) {
		t.Fatalf("usage = %#v", usage)
	}
	// The input breakdown states its cached share at 0 once it exists at all;
	// the output breakdown states reasoning 0 / text 3 (reference: input_tokens_details
	// {cached_tokens: 0}, output_tokens_details {reasoning_tokens: 0, text_tokens: 3}).
	in := usage["input_tokens_details"].(map[string]any)
	if in["cached_tokens"] != float64(0) {
		t.Fatalf("input details = %#v", in)
	}
	out := usage["output_tokens_details"].(map[string]any)
	if out["reasoning_tokens"] != float64(0) || out["text_tokens"] != float64(3) {
		t.Fatalf("output details = %#v", out)
	}
	if m["parallel_tool_calls"] != false || fmtJSON(m["tools"]) != `[]` || m["tool_choice"] != "auto" {
		t.Fatalf("shell extras = %#v", m)
	}
	// No responseId means no id to stand on (D10: empty, not invented).
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	if m["id"] != "" {
		t.Fatalf("id = %#v, want empty", m["id"])
	}
}

// A reasoned tool turn: the output order is fixed — reasoning, message,
// function_call — and the message keeps its null text (reference rs_thought_tool
// run; RSP:2362-2393).
func TestGeminiResponseToResponsesParityThoughtAndCalls(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"let me think","thought":true},
		{"functionCall":{"name":"get_weather","args":{"city":"Boston"}}}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-2"}`)
	output := m["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("output = %#v", output)
	}
	types := []string{"reasoning", "message", "function_call"}
	for i, want := range types {
		if output[i].(map[string]any)["type"] != want {
			t.Fatalf("item %d = %#v, want %s", i, output[i], want)
		}
	}
	rsn := output[0].(map[string]any)
	if rsn["id"] != "rs_0" || rsn["role"] != "assistant" || rsn["status"] != "completed" {
		t.Fatalf("reasoning item = %#v", rsn)
	}
	if rsn["content"].([]any)[0].(map[string]any)["text"] != "let me think" {
		t.Fatalf("reasoning text = %#v", rsn["content"])
	}
	msg := output[1].(map[string]any)
	if msg["content"].([]any)[0].(map[string]any)["text"] != nil {
		t.Fatalf("message text = %#v, want null", msg["content"])
	}
	call := output[2].(map[string]any)
	if call["call_id"] != "call_2" || call["name"] != "get_weather" || call["arguments"] != `{"city": "Boston"}` {
		t.Fatalf("call = %#v", call)
	}
	if call["status"] != "completed" {
		t.Fatalf("call status = %#v", call["status"])
	}

	// Two calls in one turn each become their own item with minted ids
	// (reference rs_two_calls).
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"name":"f","args":{"a":1}}},
		{"functionCall":{"name":"g","args":{"b":2}}}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	output = m["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("output = %#v", output)
	}
	for i, want := range []string{"msg_0", "call_1", "call_2"} {
		if output[i].(map[string]any)["id"] != want {
			t.Fatalf("item %d id = %v, want %s", i, output[i].(map[string]any)["id"], want)
		}
	}
}

// A call's signature rides the id (the __thought__ separator) and the
// provider fields; a native id on the wire wins (reference rs_tool_sig,
// rs_tool_native_id; FAC:1142-1157, VGS:1548-1600).
func TestGeminiResponseToResponsesParityCallSignatures(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"SIG"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-3"}`)
	call := m["output"].([]any)[1].(map[string]any)
	if call["call_id"] != "call_1__thought__SIG" || call["id"] != "call_1__thought__SIG" {
		t.Fatalf("call id = %#v", call)
	}
	if fmtJSON(call["provider_specific_fields"]) != `{"thought_signature":"SIG"}` {
		t.Fatalf("provider fields = %s", fmtJSON(call["provider_specific_fields"]))
	}

	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"name":"f","args":{"a":1},"id":"native_1"}}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	call = m["output"].([]any)[1].(map[string]any)
	if call["call_id"] != "native_1" {
		t.Fatalf("native id = %#v", call["call_id"])
	}
	if _, has := call["provider_specific_fields"]; has {
		t.Fatalf("no signature, no provider fields: %#v", call)
	}
}

// A thought part with a signature: the non-stream bridge encodes the signed
// blocks into the reasoning item's encrypted_content (RSP:2356); the
// reference's own string is
// [{"type":"thinking","thinking":"ponder","signature":"TSIG"}].
func TestGeminiResponseToResponsesParityThoughtSignature(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"ponder","thought":true,"thoughtSignature":"TSIG"},
		{"text":"answer"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-4"}`)
	output := m["output"].([]any)
	rsn := output[0].(map[string]any)
	if rsn["type"] != "reasoning" {
		t.Fatalf("output = %#v", output)
	}
	if rsn["content"].([]any)[0].(map[string]any)["text"] != "ponder" {
		t.Fatalf("reasoning text = %#v", rsn["content"])
	}
	if rsn["encrypted_content"] != `[{"signature":"TSIG","thinking":"ponder","type":"thinking"}]` {
		t.Fatalf("encrypted_content = %#v", rsn["encrypted_content"])
	}
	if output[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] != "answer" {
		t.Fatalf("message text = %#v", output[1])
	}
	// An unsigned thought contributes text but nothing to encrypt.
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"ponder","thought":true},{"text":"answer"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	if _, has := m["output"].([]any)[0].(map[string]any)["encrypted_content"]; has {
		t.Fatalf("unsigned thought must not encrypt: %#v", m["output"])
	}
}

// Media walk: images replace the message item with image items carrying the
// bytes alone (the text is dropped there), other inline media folds into the
// text as data URLs upstream of that, and audio — as object or data-URL text —
// vanishes (reference rs_media, rs_image_only, rs_audio_text; VGS:1353-1375,
// VGS:1491-1546, RSP's message extractor).
func TestGeminiResponseToResponsesParityMedia(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"see"},
		{"inlineData":{"mimeType":"image/png","data":"AAAB"}},
		{"inlineData":{"mimeType":"application/pdf","data":"JVBE"}},
		{"inlineData":{"mimeType":"audio/wav","data":"QUJD"}}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-5"}`)
	output := m["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %#v, want the image item alone", output)
	}
	ig := output[0].(map[string]any)
	if ig["type"] != "image_generation_call" || ig["id"] != "ig_0" || ig["result"] != "AAAB" || ig["status"] != "completed" {
		t.Fatalf("image item = %#v", ig)
	}

	// Without images, the pdf data URL rides the message text and audio drops.
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"see"},
		{"inlineData":{"mimeType":"application/pdf","data":"JVBE"}},
		{"inlineData":{"mimeType":"audio/wav","data":"QUJD"}}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	msg := m["output"].([]any)[0].(map[string]any)
	if got := msg["content"].([]any)[0].(map[string]any)["text"]; got != "seedata:application/pdf;base64,JVBE" {
		t.Fatalf("text = %#v", got)
	}

	// A text part that is an audio data URL is audio, not text.
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"data:audio/wav;base64,QUJD"}]},
		"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	msg = m["output"].([]any)[0].(map[string]any)
	if got := msg["content"].([]any)[0].(map[string]any)["text"]; got != nil {
		t.Fatalf("audio text part should fold to null text, got %#v", got)
	}
}

// A blocked prompt and a flagged first candidate both replace the whole turn
// with one content-filtered message item and the three-counter usage block —
// no details (reference rs_blocked, rs_flagged; VGS:1630-1697).
func TestGeminiResponseToResponsesParityBlocked(t *testing.T) {
	m := gemRespParityResponse(t, `{"promptFeedback":{"blockReason":"SAFETY"},
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`)
	if m["status"] != "incomplete" {
		t.Fatalf("status = %#v", m["status"])
	}
	output := m["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output = %#v", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["status"] != "incomplete" {
		t.Fatalf("item = %#v", item)
	}
	if item["content"].([]any)[0].(map[string]any)["text"] != nil {
		t.Fatalf("blocked text = %#v, want null", item["content"])
	}
	usage := m["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(0) || usage["total_tokens"] != float64(5) {
		t.Fatalf("usage = %#v", usage)
	}
	if _, has := usage["input_tokens_details"]; has {
		t.Fatalf("blocked usage states no details: %#v", usage)
	}
	if _, has := usage["output_tokens_details"]; has {
		t.Fatalf("blocked usage states no details: %#v", usage)
	}

	// A flagged first candidate takes the same shape with its own counters.
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		"finishReason":"SAFETY","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`)
	if m["status"] != "incomplete" || m["output"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] != nil {
		t.Fatalf("flagged turn = %#v", m["output"])
	}
	if m["usage"].(map[string]any)["output_tokens"] != float64(3) {
		t.Fatalf("flagged usage = %#v", m["usage"])
	}
}

// The finish reason enumeration, end to end: the flagged set replaces the
// choice (null text), the content_filter-mapped rest keep their text and read
// incomplete, and everything else completes (reference rs_fr_* runs; VGS:1294-1345
// with core_helpers.py:151-206).
func TestGeminiResponseToResponsesParityFinishReasonEnumeration(t *testing.T) {
	cases := []struct {
		finish   string
		status   string
		nullText bool
	}{
		{"STOP", "completed", false},
		{"MAX_TOKENS", "incomplete", false},
		{"SAFETY", "incomplete", true},
		{"RECITATION", "incomplete", true},
		{"LANGUAGE", "incomplete", false},
		{"OTHER", "incomplete", false},
		{"BLOCKLIST", "incomplete", true},
		{"PROHIBITED_CONTENT", "incomplete", true},
		{"SPII", "incomplete", true},
		{"IMAGE_SAFETY", "incomplete", true},
		{"IMAGE_PROHIBITED_CONTENT", "incomplete", true},
		{"FINISH_REASON_UNSPECIFIED", "completed", false},
		{"MALFORMED_FUNCTION_CALL", "completed", false},
		{"TOO_MANY_TOOL_CALLS", "completed", false},
		{"MALFORMED_RESPONSE", "completed", false},
		{"WAT", "completed", false},
	}
	for _, tc := range cases {
		m := gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
			"finishReason":"`+tc.finish+`","index":0}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r"}`)
		if m["status"] != tc.status {
			t.Fatalf("%s: status = %v, want %s", tc.finish, m["status"], tc.status)
		}
		item := m["output"].([]any)[0].(map[string]any)
		text := item["content"].([]any)[0].(map[string]any)["text"]
		if tc.nullText && text != nil {
			t.Fatalf("%s: text = %#v, want null", tc.finish, text)
		}
		if !tc.nullText && text != "x" {
			t.Fatalf("%s: text = %#v, want x", tc.finish, text)
		}
	}
}

// The usage accounting, end to end: a candidate count that excludes thinking
// has it added back; per-modality details report with cached tokens
// subtracted; a positive candidate count always states its text share; the
// tool-use prompt tokens fold into the input count unless the turn used
// search grounding (reference rs_usage_* runs; VGS:1699-1917).
func TestGeminiResponseToResponsesParityUsage(t *testing.T) {
	body := func(um string) string {
		return `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],
			"usageMetadata":` + um + `}`
	}
	m := gemRespParityResponse(t, body(`{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":10,"thoughtsTokenCount":2}`))
	u := m["usage"].(map[string]any)
	if u["output_tokens"] != float64(5) {
		t.Fatalf("exclusive completion = %#v", u["output_tokens"])
	}
	if od := u["output_tokens_details"].(map[string]any); od["reasoning_tokens"] != float64(2) || od["text_tokens"] != float64(3) {
		t.Fatalf("exclusive details = %#v", od)
	}

	m = gemRespParityResponse(t, body(`{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,
		"cachedContentTokenCount":2,"thoughtsTokenCount":2,
		"promptTokensDetails":[{"modality":"TEXT","tokenCount":10}],
		"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":4}]}`))
	u = m["usage"].(map[string]any)
	if u["input_tokens"] != float64(10) || u["output_tokens"] != float64(6) || u["total_tokens"] != float64(16) {
		t.Fatalf("rich totals = %#v", u)
	}
	if id := u["input_tokens_details"].(map[string]any); id["cached_tokens"] != float64(2) || id["text_tokens"] != float64(8) {
		t.Fatalf("rich input details = %#v", id)
	}
	if od := u["output_tokens_details"].(map[string]any); od["reasoning_tokens"] != float64(2) || od["text_tokens"] != float64(4) {
		t.Fatalf("rich output details = %#v", od)
	}

	// No candidate count: zero output, no output details at all.
	m = gemRespParityResponse(t, body(`{"promptTokenCount":7,"totalTokenCount":7,"cachedContentTokenCount":3}`))
	u = m["usage"].(map[string]any)
	if u["output_tokens"] != float64(0) {
		t.Fatalf("sparse output = %#v", u["output_tokens"])
	}
	if _, has := u["output_tokens_details"]; has {
		t.Fatalf("sparse states no output details: %#v", u)
	}
	if id := u["input_tokens_details"].(map[string]any); id["cached_tokens"] != float64(3) {
		t.Fatalf("sparse input details = %#v", id)
	}

	// Tool-use prompt tokens fold into the input count.
	m = gemRespParityResponse(t, body(`{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,"toolUsePromptTokenCount":3}`))
	u = m["usage"].(map[string]any)
	if u["input_tokens"] != float64(8) {
		t.Fatalf("tool-use input = %#v", u["input_tokens"])
	}

	// ...unless the turn used search grounding, whose fee replaces them.
	m = gemRespParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		"finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["q"]}}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,"toolUsePromptTokenCount":3}}`)
	u = m["usage"].(map[string]any)
	if u["input_tokens"] != float64(5) {
		t.Fatalf("grounded input = %#v", u["input_tokens"])
	}
}

// Multi-candidate turns contribute one message item each; a body with no
// candidates contributes none (reference rs_multi, rs_no_candidates,
// rs_empty_candidates).
func TestGeminiResponseToResponsesParityMultiAndEmpty(t *testing.T) {
	m := gemRespParityResponse(t, `{"candidates":[
		{"content":{"role":"model","parts":[{"text":"one"}]},"finishReason":"STOP","index":0},
		{"content":{"role":"model","parts":[{"text":"two"}]},"finishReason":"STOP","index":1}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r-multi"}`)
	output := m["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v", output)
	}
	for i, want := range []string{"one", "two"} {
		if output[i].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] != want {
			t.Fatalf("item %d = %#v", i, output[i])
		}
	}

	for _, body := range []string{
		`{"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5}}`,
		`{"candidates":[],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
	} {
		m := gemRespParityResponse(t, body)
		if fmtJSON(m["output"]) != `[]` {
			t.Fatalf("output = %s, want empty", fmtJSON(m["output"]))
		}
		if m["status"] != "completed" {
			t.Fatalf("status = %#v", m["status"])
		}
	}
}

// ---------------------------------------------------------------------------
// Reverse: Responses response -> gemini response (derived, round-trip checked)
// ---------------------------------------------------------------------------

// The reverse fold emits the response-side spellings (camelCase part keys)
// and the self-consistent usage block the spec's inverse rules define: the
// candidate count sheds the reasoning share, the total is recomputed, and the
// tool-use count never appears (spec D5; validated by feeding this body back
// through the reference's gemini->chat->responses composition).
func TestResponsesResponseToGeminiReverseParity(t *testing.T) {
	out, err := ConvertResponse(Responses, Gemini, []byte(`{"id":"resp_1","object":"response","created_at":0,
		"status":"completed","model":"m",
		"output":[
			{"type":"reasoning","id":"rs_1","status":"completed","role":"assistant",
			 "content":[{"type":"output_text","text":"ponder","annotations":[]}],
			 "encrypted_content":"[{\"type\":\"thinking\",\"thinking\":\"deep\",\"signature\":\"SIG\"}]"},
			{"type":"message","id":"msg_1","status":"completed","role":"assistant",
			 "content":[{"type":"output_text","text":"hello","annotations":[]}]},
			{"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\": 1}","id":"call_1","status":"completed"}],
		"usage":{"input_tokens":10,"output_tokens":6,"total_tokens":16,
			"input_tokens_details":{"cached_tokens":2,"text_tokens":8},
			"output_tokens_details":{"reasoning_tokens":2,"text_tokens":4}}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["responseId"] != "resp_1" {
		t.Fatalf("responseId = %#v", m["responseId"])
	}
	cand := m["candidates"].([]any)[0].(map[string]any)
	if cand["finishReason"] != "STOP" || cand["index"] != float64(0) {
		t.Fatalf("candidate shell = %#v", cand)
	}
	parts := cand["content"].(map[string]any)["parts"].([]any)
	if len(parts) != 4 {
		t.Fatalf("parts = %#v", parts)
	}
	if p0 := parts[0].(map[string]any); p0["text"] != "ponder" || p0["thought"] != true {
		t.Fatalf("thought part = %#v", p0)
	}
	if p1 := parts[1].(map[string]any); p1["text"] != "deep" || p1["thoughtSignature"] != "SIG" {
		t.Fatalf("signed replay part = %#v", p1)
	}
	if parts[2].(map[string]any)["text"] != "hello" {
		t.Fatalf("text part = %#v", parts[2])
	}
	fc := parts[3].(map[string]any)["functionCall"].(map[string]any)
	if fc["name"] != "f" || fmtJSON(fc["args"]) != `{"a":1}` {
		t.Fatalf("functionCall = %#v", fc)
	}
	um := m["usageMetadata"].(map[string]any)
	if um["promptTokenCount"] != float64(10) || um["candidatesTokenCount"] != float64(4) ||
		um["thoughtsTokenCount"] != float64(2) || um["totalTokenCount"] != float64(16) {
		t.Fatalf("usageMetadata = %#v", um)
	}
	if um["cachedContentTokenCount"] != float64(2) {
		t.Fatalf("cached = %#v", um["cachedContentTokenCount"])
	}
	if fmtJSON(um["promptTokensDetails"]) != `[{"modality":"TEXT","tokenCount":8}]` {
		t.Fatalf("promptTokensDetails = %s", fmtJSON(um["promptTokensDetails"]))
	}
	if fmtJSON(um["candidatesTokensDetails"]) != `[{"modality":"TEXT","tokenCount":4}]` {
		t.Fatalf("candidatesTokensDetails = %s", fmtJSON(um["candidatesTokensDetails"]))
	}
	if _, has := um["toolUsePromptTokenCount"]; has {
		t.Fatalf("tool-use count has no inverse and must not be written: %#v", um)
	}
}

// The reverse status mapping: incomplete by a ceiling reads MAX_TOKENS,
// incomplete by a content filter reads SAFETY, and everything else STOP
// (inverse of RSP:2103-2109 + VGS:1310-1345).
func TestResponsesResponseToGeminiReverseParityStatus(t *testing.T) {
	shell := func(status, details, usage string) string {
		return `{"id":"r","status":"` + status + `"` + details + `,"output":[
			{"type":"message","id":"msg_1","role":"assistant","status":"` + status + `",
			 "content":[{"type":"output_text","text":"cut","annotations":[]}]}],"usage":` + usage + `}`
	}
	cases := []struct {
		name, body, want string
	}{
		{"completed", shell("completed", "", `{"input_tokens":5,"output_tokens":3,"total_tokens":8}`), "STOP"},
		{"max_output_tokens", shell("incomplete", `,"incomplete_details":{"reason":"max_output_tokens"}`, `{"input_tokens":5,"output_tokens":3,"total_tokens":8}`), "MAX_TOKENS"},
		{"content_filter", shell("incomplete", `,"incomplete_details":{"reason":"content_filter"}`, `{"input_tokens":5,"output_tokens":0,"total_tokens":5}`), "SAFETY"},
	}
	for _, tc := range cases {
		out, err := ConvertResponse(Responses, Gemini, []byte(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		cand := m["candidates"].([]any)[0].(map[string]any)
		if cand["finishReason"] != tc.want {
			t.Fatalf("%s: finishReason = %v, want %s", tc.name, cand["finishReason"], tc.want)
		}
	}
	// An image item replays as inline media with the response-side spelling.
	out, err := ConvertResponse(Responses, Gemini, []byte(`{"id":"r","status":"completed",
		"output":[{"type":"image_generation_call","id":"ig_1","status":"completed","result":"AAAB"}],
		"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	parts := m["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)
	if fmtJSON(parts[0]) != `{"inlineData":{"data":"AAAB","mimeType":"image/png"}}` {
		t.Fatalf("image part = %s", fmtJSON(parts[0]))
	}
	// The self-contradictory total is recomputed, not copied (D5).
	um := m["usageMetadata"].(map[string]any)
	if um["totalTokenCount"] != float64(8) {
		t.Fatalf("recomputed total = %#v, want 8", um["totalTokenCount"])
	}
}

// ---------------------------------------------------------------------------
// Streams: gemini -> responses (forward, reference-pinned)
// ---------------------------------------------------------------------------

// gemRespParityEvents splits a Responses SSE stream into its parsed payloads.
func gemRespParityEvents(t *testing.T, out string) []map[string]any {
	t.Helper()
	var evs []map[string]any
	for _, block := range strings.Split(out, "\n\n") {
		var data string
		for _, line := range strings.Split(block, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("bad event data %q: %v", data, err)
		}
		evs = append(evs, m)
	}
	return evs
}

// gemRespParityTypes lists the event names of a parsed stream.
func gemRespParityTypes(evs []map[string]any) []string {
	var out []string
	for _, e := range evs {
		out = append(out, e["type"].(string))
	}
	return out
}

// gemRespParityFind returns the first event of the given type.
func gemRespParityFind(evs []map[string]any, typ string) map[string]any {
	for _, e := range evs {
		if e["type"] == typ {
			return e
		}
	}
	return nil
}

// gemRespParityFrames splits a gemini SSE byte stream into its raw frame
// payloads (this wire has no event names and no sentinel).
func gemRespParityFrames(t *testing.T, out string) []map[string]any {
	t.Helper()
	var frames []map[string]any
	for _, block := range strings.Split(out, "\n\n") {
		var data string
		for _, line := range strings.Split(block, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("bad frame %q: %v", data, err)
		}
		frames = append(frames, m)
	}
	return frames
}

// The full event walk for a reasoned tool turn: created, in_progress, the
// reasoning item announced at its own index with its delta, the message item
// announced when its text arrives, the tool after them, and the close order
// tools-then-text. (The reference's bridge announces only the first content
// item and swallows that frame's own delta — the message item of this shape
// streamed its text under a silently-minted id; this library announces every
// item and keeps every delta.)
func TestPipeGeminiToResponsesParityTextTool(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added",
		"response.reasoning_summary_text.delta",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	if got := gemRespParityTypes(evs); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event types = %v\nwant %v\n%s", got, want, dst.String())
	}
	rsAdded := evs[2]
	if item := rsAdded["item"].(map[string]any); item["type"] != "reasoning" || item["summary"] != nil || item["id"] != "rs_0" {
		t.Fatalf("reasoning announcement = %#v", item)
	}
	if rsAdded["output_index"] != float64(0) {
		t.Fatalf("reasoning output_index = %#v", rsAdded["output_index"])
	}
	if d := evs[3]; d["delta"] != "think" || d["item_id"] != "rs_0" {
		t.Fatalf("reasoning delta = %#v", d)
	}
	msgAdded := evs[4]
	if item := msgAdded["item"].(map[string]any); item["type"] != "message" || item["id"] != "msg_1" {
		t.Fatalf("message announcement = %#v", item)
	}
	if msgAdded["output_index"] != float64(1) {
		t.Fatalf("message output_index = %#v", msgAdded["output_index"])
	}
	delta := evs[6]
	if delta["delta"] != "hi" || delta["item_id"] != "msg_1" || delta["output_index"] != float64(1) {
		t.Fatalf("text delta = %#v", delta)
	}
	toolAdded := evs[7]
	if toolAdded["output_index"] != float64(2) {
		t.Fatalf("tool output_index = %#v", toolAdded["output_index"])
	}
	if item := toolAdded["item"].(map[string]any); item["id"] != "call_2" || item["call_id"] != "call_2" || item["name"] != "f" || item["arguments"] != "" {
		t.Fatalf("tool announcement = %#v", item)
	}
	if d := evs[8]; d["delta"] != `{"a": 1}` {
		t.Fatalf("args delta = %#v", d["delta"])
	}
	// The message part done carries the whole reasoning text — the shape the
	// reference's bridge closes a reasoned turn with.
	part := gemRespParityFind(evs, "response.content_part.done")
	if fmtJSON(part["part"]) != `{"reasoning":"think","type":"reasoning_text"}` {
		t.Fatalf("part done = %s", fmtJSON(part["part"]))
	}
	if td := gemRespParityFind(evs, "response.output_text.done"); td["text"] != "hi" {
		t.Fatalf("text done = %#v", td["text"])
	}
	// The terminal: output in index order, usage through the accounting.
	comp := gemRespParityFind(evs, "response.completed")
	resp := comp["response"].(map[string]any)
	if resp["status"] != "completed" || resp["model"] != "m" {
		t.Fatalf("terminal shell = %#v", resp)
	}
	output := resp["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("terminal output = %#v", output)
	}
	for i, want := range []string{"reasoning", "message", "function_call"} {
		if output[i].(map[string]any)["type"] != want {
			t.Fatalf("terminal item %d = %#v", i, output[i])
		}
	}
	if text := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "think" {
		t.Fatalf("terminal reasoning text = %#v", text)
	}
	if text := output[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "hi" {
		t.Fatalf("terminal message text = %#v", text)
	}
	if call := output[2].(map[string]any); call["arguments"] != `{"a": 1}` || call["call_id"] != "call_2" {
		t.Fatalf("terminal call = %#v", call)
	}
	usage := resp["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(3) || usage["total_tokens"] != float64(8) {
		t.Fatalf("terminal usage = %#v", usage)
	}
	// The reasoning item in the stream terminal never carries an
	// encrypted_content — that spelling belongs to the non-stream fold; a
	// streamed signature becomes the response-level list asserted below.
	if _, has := output[0].(map[string]any)["encrypted_content"]; has {
		t.Fatalf("stream reasoning item must not encrypt: %#v", output[0])
	}
}

// Every thought frame streams its delta — the reference's iterator swallows
// the first one to its one-shot announcement queueing; this library keeps all
// of them (deltas ["A","B"], part reasoning "AB").
func TestPipeGeminiToResponsesParityThoughtSwallow(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"A","thought":true}]},"index":0}],"responseId":"r2"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"B","thought":true}]},"index":0}],"responseId":"r2"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r2"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	var deltas []string
	for _, e := range evs {
		if e["type"] == "response.reasoning_summary_text.delta" {
			deltas = append(deltas, e["delta"].(string))
		}
	}
	if strings.Join(deltas, ",") != "A,B" {
		t.Fatalf("reasoning deltas = %v, want [A B]:\n%s", deltas, dst.String())
	}
	part := gemRespParityFind(evs, "response.content_part.done")
	if fmtJSON(part["part"]) != `{"reasoning":"AB","type":"reasoning_text"}` {
		t.Fatalf("part done = %s", fmtJSON(part["part"]))
	}
	comp := gemRespParityFind(evs, "response.completed")
	output := comp["response"].(map[string]any)["output"].([]any)
	if text := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "AB" {
		t.Fatalf("terminal reasoning text = %#v", text)
	}
}

// A plain text stream: the first frame announces the message item (plus its
// empty content part) AND streams its delta — the reference spends the first
// frame's delta on the announcement; this library keeps every frame's content
// (deltas ["hel","lo"], done text "hello").
func TestPipeGeminiToResponsesParityPlainText(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}],"responseId":"r3"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}],"responseId":"r3"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r3"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	added := gemRespParityFind(evs, "response.output_item.added")
	if item := added["item"].(map[string]any); item["type"] != "message" || item["role"] != "assistant" || item["status"] != "in_progress" {
		t.Fatalf("message announcement = %#v", item)
	}
	if cp := gemRespParityFind(evs, "response.content_part.added"); cp == nil || cp["content_index"] != float64(0) {
		t.Fatalf("content_part.added missing: %#v", cp)
	}
	var deltas []string
	for _, e := range evs {
		if e["type"] == "response.output_text.delta" {
			deltas = append(deltas, e["delta"].(string))
		}
	}
	if strings.Join(deltas, ",") != "hel,lo" {
		t.Fatalf("text deltas = %v, want [hel lo]:\n%s", deltas, dst.String())
	}
	if td := gemRespParityFind(evs, "response.output_text.done"); td["text"] != "hello" {
		t.Fatalf("text done = %#v", td["text"])
	}
	if part := gemRespParityFind(evs, "response.content_part.done"); fmtJSON(part["part"]) != `{"annotations":[],"text":"hello","type":"output_text"}` {
		t.Fatalf("part done = %s", fmtJSON(part["part"]))
	}
}

// A blocked prompt closes the stream itself: one message item announced, the
// empty part done, and the terminal reporting incomplete with the
// three-counter usage (reference st_blocked run).
func TestPipeGeminiToResponsesParityBlocked(t *testing.T) {
	src := `data: {"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5},"responseId":"r5"}` + "\n\n"
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	comp := gemRespParityFind(evs, "response.completed")
	if comp == nil {
		t.Fatalf("no terminal:\n%s", dst.String())
	}
	resp := comp["response"].(map[string]any)
	if resp["status"] != "incomplete" {
		t.Fatalf("status = %#v", resp["status"])
	}
	output := resp["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["type"] != "message" {
		t.Fatalf("output = %#v", output)
	}
	if item := output[0].(map[string]any); item["status"] != "incomplete" {
		t.Fatalf("message status = %#v", item["status"])
	}
	usage := resp["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(0) || usage["total_tokens"] != float64(5) {
		t.Fatalf("usage = %#v", usage)
	}
	if _, has := usage["output_tokens_details"]; has {
		t.Fatalf("blocked stream states no output details: %#v", usage)
	}
}

// A ceiling cut completes the turn with an incomplete status (reference
// st_maxtokens run).
func TestPipeGeminiToResponsesParityMaxTokens(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"cut"}]},"index":0}],"responseId":"r6"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r6"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	comp := gemRespParityFind(gemRespParityEvents(t, dst.String()), "response.completed")
	resp := comp["response"].(map[string]any)
	if resp["status"] != "incomplete" {
		t.Fatalf("status = %#v", resp["status"])
	}
	if text := resp["output"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "cut" {
		t.Fatalf("text = %#v", text)
	}
}

// An image frame announces the message item like any first content frame, and
// the terminal reports the image item alone (reference st_image run).
func TestPipeGeminiToResponsesParityImage(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AAAB"}}]},"index":0}],"responseId":"r7"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r7"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	if gemRespParityFind(evs, "response.output_item.added") == nil || gemRespParityFind(evs, "response.content_part.added") == nil {
		t.Fatalf("image frame should announce the message item:\n%s", dst.String())
	}
	comp := gemRespParityFind(evs, "response.completed")
	output := comp["response"].(map[string]any)["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["type"] != "image_generation_call" {
		t.Fatalf("output = %#v", output)
	}
	if ig := output[0].(map[string]any); ig["result"] != "AAAB" || ig["id"] != "ig_1" || ig["status"] != "completed" {
		t.Fatalf("image item = %#v", ig)
	}
}

// Streamed thought signatures: the terminal's reasoning item carries only its
// text, and the signatures surface as the response-level
// provider_specific_fields.thought_signatures — the value the reference's
// stream accumulator keeps is the last frame's list (reference st_thought_sig
// run; streaming_iterator.py:520-544).
func TestPipeGeminiToResponsesParityThoughtSignatures(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ponder","thought":true,"thoughtSignature":"TSIG"}]},"index":0}],"responseId":"r8"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"answer"}]},"index":0}],"responseId":"r8"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r8"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	comp := gemRespParityFind(evs, "response.completed")
	resp := comp["response"].(map[string]any)
	output := resp["output"].([]any)
	rsn := output[0].(map[string]any)
	if rsn["type"] != "reasoning" {
		t.Fatalf("output = %#v", output)
	}
	if _, has := rsn["encrypted_content"]; has {
		t.Fatalf("stream reasoning item must not carry encrypted_content: %#v", rsn)
	}
	if fmtJSON(resp["provider_specific_fields"]) != `{"thought_signatures":["TSIG"]}` {
		t.Fatalf("provider fields = %s", fmtJSON(resp["provider_specific_fields"]))
	}
	if text := rsn["content"].([]any)[0].(map[string]any)["text"]; text != "ponder" {
		t.Fatalf("reasoning text = %#v", text)
	}
}

// Tool-first streams: the call announces itself at index 0, and a later text
// frame opens and announces the message item at the next index — the text
// rides an item the client has seen declared. (The reference's bridge burns
// its one announcement on the tool frame; the late text then streams under a
// silently-minted id.)
func TestPipeGeminiToResponsesParityToolFirst(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1},"id":"native_1"}}]},"index":0}],"responseId":"r9"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"after"}]},"index":0}],"responseId":"r9"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r9"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	var added []map[string]any
	for _, e := range evs {
		if e["type"] == "response.output_item.added" {
			added = append(added, e)
		}
	}
	if len(added) != 2 {
		t.Fatalf("announcements = %#v, want the tool then the message", added)
	}
	if item := added[0]["item"].(map[string]any); item["id"] != "native_1" || added[0]["output_index"] != float64(0) {
		t.Fatalf("tool announcement = %#v", added[0])
	}
	if item := added[1]["item"].(map[string]any); item["type"] != "message" || added[1]["output_index"] != float64(1) {
		t.Fatalf("message announcement = %#v", added[1])
	}
	delta := gemRespParityFind(evs, "response.output_text.delta")
	if delta == nil || delta["delta"] != "after" || delta["item_id"] != "msg_0" {
		t.Fatalf("late text delta = %#v", delta)
	}
	comp := gemRespParityFind(evs, "response.completed")
	output := comp["response"].(map[string]any)["output"].([]any)
	if len(output) != 2 || output[0].(map[string]any)["type"] != "function_call" || output[1].(map[string]any)["type"] != "message" {
		t.Fatalf("terminal output = %#v", output)
	}
	if call := output[0].(map[string]any); call["call_id"] != "native_1" || call["arguments"] != `{"a": 1}` {
		t.Fatalf("terminal call = %#v", call)
	}
	if text := output[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "after" {
		t.Fatalf("terminal message text = %#v, want the streamed text", text)
	}
}

// A usage-only frame crashes the reference's bridge (IndexError in its
// iterator — verified against v1.100.1); this fold reads the usage it came
// for and emits nothing for the frame itself.
func TestPipeGeminiToResponsesParityUsageOnlyFrameSurvives(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r4"}`,
		``,
		`data: {"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r4"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r4"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	// One announcement with its content part, the first frame's delta, the
	// terminal — nothing for the usage-only frame itself.
	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done", "response.content_part.done",
		"response.output_item.done", "response.completed",
	}
	if got := gemRespParityTypes(evs); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event types = %v\nwant %v\n%s", got, want, dst.String())
	}
	comp := gemRespParityFind(evs, "response.completed")
	usage := comp["response"].(map[string]any)["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(3) {
		t.Fatalf("usage = %#v", usage)
	}
}

// An in-band error frame raises in the reference (VertexAIError); this fold
// stays silent — no events for the frame and nothing after it (D7.9).
func TestPipeGeminiToResponsesParityErrorFrameStaysSilent(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r12"}`,
		``,
		`data: {"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r12"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	if gemRespParityFind(evs, "response.completed") != nil {
		t.Fatalf("an error frame must not dress up as a completion:\n%s", dst.String())
	}
}

// ---------------------------------------------------------------------------
// Reverse: responses stream -> gemini frames (derived, round-trip checked)
// ---------------------------------------------------------------------------

// The reverse stream emits one bare frame per delta — text, thought, and the
// whole function call once its fragments have grown into one JSON object —
// then the terminal frame carrying the finish reason and the usage on the
// same frame (validated by feeding these frames back through the reference's
// chunk parser + iterator: same terminal items, text and usage).
func TestPipeResponsesToGeminiReverseParity(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress"}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hel"}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo"}`,
		``,
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"ponder"}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","status":"in_progress","call_id":"call_1","name":"f","arguments":""}}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\"a\": "}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"1}"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	frames := gemRespParityFrames(t, dst.String())
	if len(frames) != 5 {
		t.Fatalf("frames = %#v, want five", frames)
	}
	text1 := frames[0]["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if text1["text"] != "hel" {
		t.Fatalf("frame 0 = %#v", text1)
	}
	if frames[1]["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"] != "lo" {
		t.Fatalf("frame 1 = %#v", frames[1])
	}
	thought := frames[2]["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if thought["text"] != "ponder" || thought["thought"] != true {
		t.Fatalf("thought frame = %#v", thought)
	}
	// The unparseable fragment produced no frame; the completed object is one
	// frame, in the response-side spelling.
	call := frames[3]["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if fmtJSON(call["functionCall"]) != `{"args":{"a":1},"name":"f"}` {
		t.Fatalf("call frame = %s", fmtJSON(call))
	}
	// The terminal carries the finish reason and the usage on the same frame.
	term := frames[4]["candidates"].([]any)[0].(map[string]any)
	if term["finishReason"] != "STOP" {
		t.Fatalf("terminal finish = %#v", term["finishReason"])
	}
	um := frames[4]["usageMetadata"].(map[string]any)
	if um["promptTokenCount"] != float64(5) || um["candidatesTokenCount"] != float64(3) || um["totalTokenCount"] != float64(8) {
		t.Fatalf("terminal usage = %#v", um)
	}
	// Frames are bare data lines on this wire; no event name may ride along.
	if strings.Contains(dst.String(), "event:") {
		t.Fatalf("frames must be bare data lines:\n%s", dst.String())
	}
}

// The reverse stream's close rules: an incomplete turn terminates with
// MAX_TOKENS, a relay sentinel terminates with STOP and no usage, and a
// source that never closed terminates nothing (D7.8).
func TestPipeResponsesToGeminiReverseParityTermination(t *testing.T) {
	inc := strings.Join([]string{
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"cut"}`,
		``,
		`data: {"type":"response.incomplete","response":{"id":"resp_2","status":"incomplete","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(inc), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	frames := gemRespParityFrames(t, dst.String())
	if len(frames) != 2 {
		t.Fatalf("frames = %#v", frames)
	}
	if frames[1]["candidates"].([]any)[0].(map[string]any)["finishReason"] != "MAX_TOKENS" {
		t.Fatalf("incomplete finish = %#v", frames[1])
	}

	sentinel := strings.Join([]string{
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"orphan"}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	dst.Reset()
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(sentinel), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	frames = gemRespParityFrames(t, dst.String())
	if len(frames) != 2 {
		t.Fatalf("sentinel frames = %#v", frames)
	}
	term := frames[1]["candidates"].([]any)[0].(map[string]any)
	if term["finishReason"] != "STOP" {
		t.Fatalf("sentinel finish = %#v", term)
	}
	if _, has := frames[1]["usageMetadata"]; has {
		t.Fatalf("a sentinel close carries no usage: %#v", frames[1])
	}

	// A source that simply ends: the text frame arrives, no terminal invented.
	cut := `data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lost"}` + "\n\n"
	dst.Reset()
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(cut), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	frames = gemRespParityFrames(t, dst.String())
	if len(frames) != 1 {
		t.Fatalf("truncated frames = %#v, want the text frame alone", frames)
	}
	if strings.Contains(dst.String(), "finishReason") {
		t.Fatalf("a cut source must not gain a terminal:\n%s", dst.String())
	}
}

// Long argument strings are re-chunked into ten-character deltas — the piece
// size the reference's iterator streams them in (verified against v1.100.1:
// pieces of exactly 10, remainder last).
func TestPipeGeminiToResponsesParityArgsChunking(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"query":"a rather long argument string that exceeds ten runes","n":2}}}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	var deltas []string
	for _, e := range evs {
		if e["type"] == "response.function_call_arguments.delta" {
			deltas = append(deltas, e["delta"].(string))
		}
	}
	// Key order is the one documented approximation this fold (like the
	// chat-side fold of the same wire) carries: the reference renders the
	// args in the order the wire stated them, and a decode into plain maps
	// has no order to restore — keys come out sorted, content identical.
	want := []string{
		`{"n": 2, "`, `query": "a`, ` rather lo`, `ng argumen`, `t string t`,
		`hat exceed`, `s ten rune`, `s"}`,
	}
	if strings.Join(deltas, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("delta pieces = %q\nwant %q\n%s", deltas, want, dst.String())
	}
	done := gemRespParityFind(evs, "response.function_call_arguments.done")
	if done["arguments"] != `{"n": 2, "query": "a rather long argument string that exceeds ten runes"}` {
		t.Fatalf("done arguments = %#v", done["arguments"])
	}
}

// A relay sentinel in a gemini stream is the source's own close with nothing
// stated about how the turn ended. The reference never sees a [DONE] on this
// wire (its iterator reads generateContent frames), so the relay sentinel is
// this library's own convention — and the convention is settlement
// consistency, the same rule messagesresponses/stream.go:77-86 writes down
// and the other folds of a Chat-style source follow: StreamResult.Terminal is
// true for a [DONE], and the Responses dialect has exactly one legal ending,
// so the fold writes response.completed stating a plain completion (the
// source ended the turn and never said it failed). Withholding it would leave
// the host booking a delivered turn while the client's stream never closes.
func TestPipeGeminiToResponsesParitySentinelClosesTheTurn(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	if gemRespParityFind(evs, "response.completed") == nil {
		t.Fatalf("a sentinel close must write the dialect's one legal ending:\n%s", dst.String())
	}
	term := gemRespParityFind(evs, "response.completed")
	resp, _ := term["response"].(map[string]any)
	if resp == nil || resp["status"] != "completed" {
		t.Fatalf("a sentinel close states a plain completion, got %#v", term)
	}
	if gemRespParityFind(evs, "response.output_item.done") == nil {
		t.Fatalf("the open items should still close:\n%s", dst.String())
	}
}

// The reference's own terminal split: a pure-tool stream's message item in
// the terminal response states text null — the assembled item's fact —
// while the output_text.done event the same bridge emits says "" (an
// accumulator artifact). Reproduced from the reference's stream bridge
// during this audit.
func TestGeminiToResponsesParityPureToolTerminalNullText(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}]}`,
		``,
		`data: {"candidates":[{"content":{"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gemRespParityEvents(t, dst.String())
	if td := gemRespParityFind(evs, "response.output_text.done"); td["text"] != "" {
		t.Fatalf("text done = %#v, want the accumulator's \"\"", td["text"])
	}
	comp := gemRespParityFind(evs, "response.completed")
	for _, raw := range comp["response"].(map[string]any)["output"].([]any) {
		item := raw.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		part := item["content"].([]any)[0].(map[string]any)
		if text, has := part["text"]; !has || text != nil {
			t.Fatalf("terminal message text = %#v, want explicit null (the assembled item states the absence)", part)
		}
	}
}

// The reasoning item carrying BOTH a visible summary and an encrypted block
// is the one body no gemini shape can round-trip — the reference's own
// reader merges the two thought texts into one reasoning stream and its
// responses bridge re-encrypts only the SIGNED block, dropping the visible
// summary text from the item entirely (executed against the reference's
// two-hop during this audit: reasoning_content "sumblk", thinking_blocks
// [{sum},{blk,SIG}], responses item summary null, encrypted_content
// [{blk,SIG}]). The reverse fold's shape — both texts flagged, the signature
// on its own part — is the one that yields exactly that reading; the loss
// is the reference's own, documented here rather than papered over.
func TestGeminiResponsesParityReasoningDualTextRoundTrip(t *testing.T) {
	src := `{"id":"r1","model":"m","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"sum"}],"encrypted_content":"[{\"type\":\"thinking\",\"thinking\":\"blk\",\"signature\":\"S\"}]"},{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ans"}]}],"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}`
	gem, err := ConvertResponse(Responses, Gemini, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	// The optimal shape: both texts as flagged thought parts, the signature
	// riding the encrypted block's own part.
	if !strings.Contains(string(gem), `"text":"sum","thought":true`) ||
		!strings.Contains(string(gem), `"text":"blk","thought":true,"thoughtSignature":"S"`) {
		t.Fatalf("reverse fold shape = %s", gem)
	}
	back, err := ConvertResponse(Gemini, Responses, gem)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(back, &m); err != nil {
		t.Fatal(err)
	}
	for _, raw := range m["output"].([]any) {
		item := raw.(map[string]any)
		if item["type"] != "reasoning" {
			continue
		}
		// Only the signed block re-encrypts; the visible summary text is the
		// reference's own drop.
		if enc := item["encrypted_content"]; enc != `[{`+
			`"signature":"S","thinking":"blk","type":"thinking"}]` {
			t.Fatalf("encrypted_content = %#v", enc)
		}
	}
	if strings.Contains(string(back), `"summary":[{"type":"summary_text","text":"sum"}]`) {
		t.Fatalf("visible summary text must not survive: %s", back)
	}
}

// The image branch of the gemini->responses response fold, executed against
// the reference's two-hop during this audit: an inline image part replaces
// the message item entirely — even the turn's visible text drops — and a
// call in the same turn keeps its place after the image item (the reference
// mints ig_<uuid>, this codec's sequential id is its documented stand-in).
func TestGeminiToResponsesParityImageBranch(t *testing.T) {
	for _, tc := range []struct{ name, parts string }{
		{"text+img", `[{"text":"see"},{"inlineData":{"mimeType":"image/png","data":"AAAB"}}]`},
		{"img-only", `[{"inlineData":{"mimeType":"image/png","data":"AAAB"}}]`},
	} {
		out, err := ConvertResponse(Gemini, Responses, []byte(`{"candidates":[{"content":{"role":"model","parts":`+
			tc.parts+`},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		output := m["output"].([]any)
		if len(output) != 1 {
			t.Fatalf("%s: output = %#v, want the image item alone", tc.name, output)
		}
		item := output[0].(map[string]any)
		if item["type"] != "image_generation_call" || item["result"] != "AAAB" || item["status"] != "completed" {
			t.Fatalf("%s: image item = %#v", tc.name, item)
		}
	}
	// A call in the same turn keeps its place after the image item.
	out, err := ConvertResponse(Gemini, Responses, []byte(`{"candidates":[{"content":{"role":"model","parts":[
		{"inlineData":{"mimeType":"image/png","data":"AAAB"}},{"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	output := m["output"].([]any)
	if len(output) != 2 || output[0].(map[string]any)["type"] != "image_generation_call" ||
		output[1].(map[string]any)["type"] != "function_call" {
		t.Fatalf("img+call output = %#v", output)
	}
}

// A Responses "response.failed" / "error" event is the source declaring the
// turn failed: the fold writes nothing further — no synthesized terminal
// would dress the failure as a finished answer — and the host's error frame
// is what the client must see. The branch mirrors the gemini-source error
// frames the suite already pins, on the reverse direction.
func TestPipeResponsesToGeminiErrorEventIsSilent(t *testing.T) {
	src := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"hi"}`,
		``,
		`event: response.failed`,
		`data: {"type":"response.failed","response":{"id":"r1","status":"failed","error":{"code":"server_error","message":"boom"}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if strings.Contains(out, `"finishReason"`) {
		t.Fatalf("an error close must not synthesize a terminal frame:\n%s", out)
	}
	if !strings.Contains(out, `"text":"hi"`) {
		t.Fatalf("the delivered text frames must stand:\n%s", out)
	}
}

// A single-frame answer — the real endpoint's common short-answer shape,
// found live during this audit — leaves the responses stream with NO
// output_text.delta: the text rides only the done events and the assembled
// terminal item, and the streamed signature only the terminal's
// provider_specific_fields. The reverse fold must recover both, and its
// output must read back with the text and the signature intact.
func TestPipeResponsesToGeminiRecoversDoneOnlyTextAndSignature(t *testing.T) {
	src := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
		``,
		`event: response.output_text.done`,
		`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_1","text":"4"}`,
		``,
		`event: response.content_part.done`,
		`data: {"type":"response.content_part.done","output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"4","annotations":[]}}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"provider_specific_fields":{"thought_signatures":["SIG"]},"usage":{"input_tokens":14,"output_tokens":1,"total_tokens":15}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	res, err := PipeStreamWith(Responses, Gemini, &dst, strings.NewReader(src), StreamOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Terminal {
		t.Fatalf("the typed terminal must count: %+v", res)
	}
	out := dst.String()
	if !strings.Contains(out, `"text":"4"`) {
		t.Fatalf("done-only text lost:\n%s", out)
	}
	if !strings.Contains(out, `"thoughtSignature":"SIG"`) {
		t.Fatalf("streamed signature lost:\n%s", out)
	}
	// The roundtrip: the recovered frames read back with text and signature.
	var back bytes.Buffer
	if _, err := PipeStream(Gemini, Responses, &back, strings.NewReader(out), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	rt := back.String()
	if !strings.Contains(rt, `"text":"4"`) || !strings.Contains(rt, "SIG") {
		t.Fatalf("roundtrip lost the text or the signature:\n%s", rt)
	}
}

// Text a delta already delivered must not be re-emitted by the done events
// or the assembled terminal: the done event states the whole text, and the
// fold owes only the undelivered remainder (here: nothing).
func TestPipeResponsesToGeminiDeltaTextNotDuplicated(t *testing.T) {
	src := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"4"}`,
		``,
		`event: response.output_text.done`,
		`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"4"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"4","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Responses, Gemini, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if n := strings.Count(out, `"text":"4"`); n != 1 {
		t.Fatalf("delta-delivered text must appear exactly once, got %d:\n%s", n, out)
	}
}
