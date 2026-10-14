package protocol

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

// This file pins the Messages<->gemini request folds against LiteLLM v1.100.1.
// The reference has no direct bridge between the two dialects: it composes the
// Anthropic adapter onto the chat shape and folds that onto the gemini body
// (litellm: llms/anthropic/experimental_pass_through/adapters/transformation.py:1118
// translate_anthropic_to_openai -> litellm.get_optional_params(custom_llm_provider=
// "vertex_ai") -> llms/vertex_ai/gemini/transformation.py sync_transform_request_body).
// Every want in the forward cases below is the JSON that composition actually
// printed for the same input, run with .zcode/litellm-venv/bin/python during
// this audit; nothing here is hand-derived. The reverse direction has no
// forward path in the reference, so its wants are the spec's inverse rules,
// each cross-checked by feeding this codec's output back through the
// reference's forward composition (the roundtrip cases).

// gmParityDecode decodes a JSON literal with float64 numbers so decoded
// literals and decoded codec output compare identically.
func gmParityDecode(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("decode %s: %v", s, err)
	}
	return m
}

// gmParityFold runs the Messages->gemini request fold through the root
// dispatcher — the wiring a host exercises, not just the leaf package.
func gmParityFold(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := ConvertRequest(Messages, Gemini, []byte(body))
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return gmParityDecode(t, string(out.Body))
}

// gmParityFoldBack runs the gemini->Messages request fold the same way.
func gmParityFoldBack(t *testing.T, body string, fallback int) map[string]any {
	t.Helper()
	out, err := ConvertRequestWith(Gemini, Messages, []byte(body), ConvertOptions{MaxTokensFallback: fallback})
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return gmParityDecode(t, string(out.Body))
}

func gmParityEqual(t *testing.T, got, want any, what string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(want)
		t.Fatalf("%s mismatch:\n got  %s\n want %s", what, gb, wb)
	}
}

// The headline two-hop sample: a tool round trip, a system prompt, an auto
// tool choice, thinking and stop sequences. This is the body the reference's
// composed fold prints for the Messages request the spec's §2.2 quotes (run
// this audit via .zcode/verify_spec_two_hop.py's M2 entry, which asserts this
// same JSON; the exact print was taken from
// adapter.translate_anthropic_to_openai -> sync_transform_request_body).
func TestGeminiMessagesParityRequestTwoHopSample(t *testing.T) {
	got := gmParityFold(t, `{"model":"gemini-2.5-pro","max_tokens":32,"system":"be brief",
	 "messages":[{"role":"user","content":"hello"},
	   {"role":"assistant","content":[{"type":"text","text":"hi"},
	     {"type":"tool_use","id":"tu_1","name":"lookup","input":{"q":"x"}}]},
	   {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"res"}]}],
	 "tools":[{"name":"lookup","description":"d",
	   "input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],
	 "tool_choice":{"type":"auto"},"thinking":{"type":"enabled","budget_tokens":2048},
	 "stop_sequences":["END"]}`)
	want := gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"hello"}]},
	    {"role":"model","parts":[{"text":"hi"},{"function_call":{"name":"lookup","args":{"q":"x"}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"lookup","response":{"content":"res"}}}]}],
	  "system_instruction":{"parts":[{"text":"be brief"}]},
	  "tools":[{"function_declarations":[{"name":"lookup","description":"d",
	     "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"AUTO"}},
	  "generationConfig":{"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true},
	    "stop_sequences":["END"],"max_output_tokens":32}}`)
	gmParityEqual(t, got, want, "two-hop fold")
}

// Consecutive same-role turns merge into one content per role run, and a
// tool result whose content parses as a JSON object rides as that object
// rather than a wrapped string (litellm:
// prompt_templates/factory.py:1465-1476). Reference prints, this audit.
func TestGeminiMessagesParityRequestMergingAndToolResults(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[
	    {"role":"user","content":"a"},
	    {"role":"user","content":[{"type":"text","text":"b"}]},
	    {"role":"assistant","content":"mid"},
	    {"role":"assistant","content":[{"type":"text","text":"end"}]},
	    {"role":"user","content":"c"}]}`)
	want := gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"a"},{"text":"b"}]},
	    {"role":"model","parts":[{"text":"mid"},{"text":"end"}]},
	    {"role":"user","parts":[{"text":"c"}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "role merging")

	got = gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[
	    {"role":"user","content":"go"},
	    {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"{\"ok\": true, \"n\": 2}"}]}]}`)
	want = gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"go"}]},
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"n":2,"ok":true}}}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "json-object tool result")

	got = gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[
	    {"role":"user","content":"go"},
	    {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
	      "content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]}]}]}`)
	want = gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"go"}]},
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"xy"}}}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "multi-part tool result")
}

// All four tool_choice spellings fold onto the functionCallingConfig the
// reference prints (litellm: vertex_and_google_ai_studio_gemini.py:331-346).
func TestGeminiMessagesParityRequestToolChoiceModes(t *testing.T) {
	for _, tc := range []struct {
		choice string
		want   string
	}{
		{`{"type":"auto"}`, `{"functionCallingConfig":{"mode":"AUTO"}}`},
		{`{"type":"any"}`, `{"functionCallingConfig":{"mode":"ANY"}}`},
		{`{"type":"none"}`, `{"functionCallingConfig":{"mode":"NONE"}}`},
		{`{"type":"tool","name":"f"}`, `{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}}`},
	} {
		got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],
		  "tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":`+tc.choice+`}`)
		gmParityEqual(t, got["toolConfig"], gmParityDecode(t, tc.want), tc.choice)
	}
}

// A thinking block folds to ONE thought part carrying its signature itself.
// This deliberately deviates from the reference's bridge, which turns the
// block into reasoning_content plus thinking_blocks and writes one gemini
// part for each (litellm: transformation.py request fold of reasoning_content
// at :880-883 and thinking_blocks at :884-901 — the two-part duplication is
// the reference's real output, printed this audit). The duplication was
// measured against the live upstream and dropped: every replayed turn bills
// its reasoning text twice, and the reference's signature carrier — a plain
// text part — loses the signature on the way back (the reverse fold reads
// signatures only off thought parts). On the thought part the signature
// round-trips.
func TestGeminiMessagesParityRequestThinkingBlocks(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[
	    {"role":"user","content":"q"},
	    {"role":"assistant","content":[
	      {"type":"thinking","thinking":"why","signature":"SIG"},
	      {"type":"text","text":"answer"}]}]}`)
	want := gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"},
	      {"text":"answer"}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "thinking block fold")

	// An unsigned block folds to a bare thought part — no empty
	// thoughtSignature key, no duplicate text part.
	got = gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[
	    {"role":"user","content":"q"},
	    {"role":"assistant","content":[
	      {"type":"thinking","thinking":"why"},
	      {"type":"text","text":"answer"}]}]}`)
	want = gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true},
	      {"text":"answer"}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "unsigned thinking block fold")
}

// The output_format schema strictifies the way the bridge's strict mode does
// (required naming every property) and additionalProperties:false survives on
// the responseJsonSchema path — the reference's print, this audit.
func TestGeminiMessagesParityRequestOutputFormat(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],
		"output_format":{"type":"json_schema","strict":true,
		"schema":{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}}}`)
	want := gmParityDecode(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":8,
	    "response_mime_type":"application/json",
	    "response_json_schema":{"type":"object","properties":{"a":{"type":"string"}},
	      "required":["a"],"additionalProperties":false}}}`)
	gmParityEqual(t, got, want, "output_format fold")
}

// Sampling knobs land in generationConfig under their own names, and two
// function tools merge into one function_declarations block (litellm:
// vertex_and_google_ai_studio_gemini.py:699-703). Reference prints.
func TestGeminiMessagesParityRequestParamsAndToolMerge(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],
		"temperature":0.5,"top_p":0.9,"top_k":3,"stop_sequences":["END","STOP"]}`)
	want := gmParityDecode(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":8,"stop_sequences":["END","STOP"],
	    "temperature":0.5,"top_k":3,"top_p":0.9}}`)
	gmParityEqual(t, got, want, "params fold")

	got = gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],
		"tools":[{"name":"a","description":"da","input_schema":{"type":"object","properties":{}}},
		  {"name":"b","input_schema":{"type":"object"}}]}`)
	want = gmParityDecode(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":8},
	  "tools":[{"function_declarations":[
	    {"name":"a","description":"da","parameters":{"type":"object"}},
	    {"name":"b","parameters":{"type":"object"}}]}]}`)
	gmParityEqual(t, got, want, "tool merge fold")
}

// An image block rides as inline_data; a user turn with no text part at all
// gains the blank text part the target requires (litellm:
// common_utils.py:494-505 _check_text_in_content — the " " append). The
// text+image print comes from this audit; the blank-append rule from the same
// run family (.zcode/verify_spec.py, spec §2.1 rule 5).
func TestGeminiMessagesParityRequestImageTurn(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
	    {"type":"text","text":"look"},
	    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAB"}}]}]}`)
	want := gmParityDecode(t, `{"contents":[{"role":"user","parts":[
	    {"text":"look"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]}],
	  "generationConfig":{"max_output_tokens":10}}`)
	gmParityEqual(t, got, want, "text+image fold")

	got = gmParityFold(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
	    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAB"}}]}]}`)
	want = gmParityDecode(t, `{"contents":[{"role":"user","parts":[
	    {"inline_data":{"data":"AAAB","mime_type":"image/png"}},{"text":" "}]}],
	  "generationConfig":{"max_output_tokens":10}}`)
	gmParityEqual(t, got, want, "image-only fold")
}

// The file service's URIs pass straight through as file references — that
// service refuses direct fetches, so the reference does not fetch them either
// — with or without the media parameter, and mime_type stays absent (litellm:
// transformation.py:559-573; printed this audit for both a document and an
// image url source). The blank text part rides because the turn has no text
// of its own.
func TestGeminiMessagesParityRequestFilesAPIPassThrough(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
	    {"type":"document","source":{"type":"url","url":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}},
	    {"type":"image","source":{"type":"url","url":"https://generativelanguage.googleapis.com/v1beta/files/xyz"}}]}]}`)
	want := gmParityDecode(t, `{"contents":[{"role":"user","parts":[
	    {"file_data":{"file_uri":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}},
	    {"file_data":{"file_uri":"https://generativelanguage.googleapis.com/v1beta/files/xyz"}},
	    {"text":" "}]}],
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "files api passthrough")
}

// A system prompt that is a block list joins into ONE part with no separator
// — the reference reads a system message's list content as one text (litellm:
// transformation.py:1399-1412; printed this audit: system [A, B] folds to
// system_instruction parts [{"text":"AB"}]).
func TestGeminiMessagesParityRequestSystemBlockList(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":8,
	  "system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],
	  "messages":[{"role":"user","content":"x"}]}`)
	want := gmParityDecode(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "system_instruction":{"parts":[{"text":"AB"}]},
	  "generationConfig":{"max_output_tokens":8}}`)
	gmParityEqual(t, got, want, "system block list")
}

// The thinking budget takes the two-level ladder: the Messages budget is
// bucketed into an effort and the effort names the target budget, so 5000
// lands on 4096 and 0 on 128 (litellm: types/utils EffortFromBudget buckets
// plus vertex_and_google_ai_studio_gemini.py:799-846 effort->budget). Every
// want printed by the composed reference this audit (same runs as
// .zcode/verify_spec_two_hop.py's ladder checks).
func TestGeminiMessagesParityRequestThinkingLadder(t *testing.T) {
	for _, tc := range []struct {
		thinking string
		want     string
	}{
		{`{"type":"enabled","budget_tokens":128}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":700}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":1}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":0}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":2048}`, `{"thinkingBudget":2048,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":4095}`, `{"thinkingBudget":2048,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":4096}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":5000}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":8192}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":16384}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"disabled"}`, `{"thinkingBudget":0,"includeThoughts":false}`},
		{`{"type":"adaptive"}`, `{"thinkingBudget":2048,"includeThoughts":true}`},
	} {
		got := gmParityFold(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],
		  "thinking":`+tc.thinking+`}`)
		gc, _ := got["generationConfig"].(map[string]any)
		var cfg map[string]any
		if gc != nil {
			cfg, _ = gc["thinkingConfig"].(map[string]any)
		}
		gmParityEqual(t, cfg, gmParityDecode(t, tc.want), tc.thinking)
	}
}

// The hosted web-search tool becomes the target's own search tool, and is
// dropped when function declarations ride with it — that wire refuses the
// mixture (litellm: vertex_and_google_ai_studio_gemini.py:516-563). Prints
// from .zcode/verify_spec_two_hop.py, run this audit.
func TestGeminiMessagesParityRequestHostedWebSearch(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],
		"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)
	gmParityEqual(t, got["tools"], []any{map[string]any{"googleSearch": map[string]any{}}}, "search alone")

	got = gmParityFold(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"},
		  {"name":"f","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}]}`)
	gmParityEqual(t, got["tools"], []any{map[string]any{"function_declarations": []any{
		map[string]any{"name": "f", "parameters": map[string]any{
			"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}},
	}}}, "search dropped when mixed")
}

// metadata.user_id crosses the bridge as the unvalidated user parameter and
// never reaches the gemini body; context_management crosses as its own chat
// parameter and is REFUSED by the target unless drops were asked for — both
// printed by the reference this audit (litellm: utils.py:4186-4213
// _check_valid_arg raising UnsupportedParamsError for vertex_ai).
func TestGeminiMessagesParityRequestParamFates(t *testing.T) {
	got := gmParityFold(t, `{"model":"m","max_tokens":10,"metadata":{"user_id":"u1"},
		"messages":[{"role":"user","content":"x"}]}`)
	if _, has := got["user"]; has {
		t.Fatalf("user crossed into the gemini body: %v", got["user"])
	}
	for _, k := range []string{"metadata", "user_id"} {
		if _, has := got[k]; has {
			t.Fatalf("%s crossed into the gemini body", k)
		}
	}

	cm := `{"model":"m","max_tokens":10,"context_management":{"edits":[]},
		"messages":[{"role":"user","content":"x"}]}`
	_, err := ConvertRequest(Messages, Gemini, []byte(cm))
	if err == nil {
		t.Fatal("context_management was not refused")
	}
	var unsup UnsupportedParamError
	if !errors.As(err, &unsup) || unsup.Param != "context_management" {
		t.Fatalf("err = %T(%v), want UnsupportedParamError(context_management)", err, err)
	}
	if _, err := ConvertRequestWith(Messages, Gemini, []byte(cm), ConvertOptions{DropParams: true}); err != nil {
		t.Fatalf("context_management with drops: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Reverse direction: gemini request -> Messages request. The reference has no
// forward path here; wants follow the spec's §3.1 inverse rules and were
// cross-checked by feeding this codec's output back through the reference's
// forward composition during this audit.

// The full-body reverse fold. Verified by roundtrip: the reference's forward
// composition applied to this exact output restored the source body's
// contents, system_instruction, tools, toolConfig and every generationConfig
// knob including serviceTier (run this audit).
func TestGeminiMessagesParityReverseFullBody(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"text":"hi"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]},
	    {"role":"model","parts":[{"text":"answer"},{"function_call":{"name":"f","args":{"a":1}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"res"}}}]}],
	  "system_instruction":{"parts":[{"text":"be brief"}]},
	  "tools":[{"function_declarations":[{"name":"f","description":"d",
	    "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}},
	  "generationConfig":{"temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,
	    "stop_sequences":["END"],"max_output_tokens":32,
	    "thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}},
	  "serviceTier":"priority"}`, 64)
	want := gmParityDecode(t, `{"max_tokens":32,
	  "messages":[
	    {"role":"user","content":[{"type":"text","text":"hi"},
	      {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAB"}}]},
	    {"role":"assistant","content":[{"type":"text","text":"answer"},
	      {"type":"tool_use","id":"call_0","name":"f","input":{"a":1}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_0","content":"res"}]}],
	  "system":[{"type":"text","text":"be brief"}],
	  "tools":[{"name":"f","description":"d",
	    "input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],
	  "tool_choice":{"type":"tool","name":"f"},
	  "temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,"stop_sequences":["END"],
	  "thinking":{"type":"enabled","budget_tokens":2048},
	  "service_tier":"auto"}`)
	gmParityEqual(t, got, want, "reverse full body")
}

// The roundtrip itself: this codec's reverse output, folded back by THIS
// codec's forward fold (which the forward cases above pinned to the
// reference), must restore the source body. The googleSearch-only body
// restores exactly; the reference's own refold of this codec's output (run
// this audit) printed the same JSON for both, so the fixed point is the
// reference's, not merely ours.
func TestGeminiMessagesParityReverseRoundTrip(t *testing.T) {
	src := `{"contents":[
	    {"role":"user","parts":[{"text":"hi"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]},
	    {"role":"model","parts":[{"text":"answer"},{"function_call":{"name":"f","args":{"a":1}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"res"}}}]}],
	  "system_instruction":{"parts":[{"text":"be brief"}]},
	  "tools":[{"function_declarations":[{"name":"f","description":"d",
	    "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}},
	  "generationConfig":{"temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,
	    "stop_sequences":["END"],"max_output_tokens":32,
	    "thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}},
	  "safetySettings":[{"category":"H","threshold":"BLOCK_NONE"}],
	  "cachedContent":"cc-1","serviceTier":"priority"}`
	back := gmParityFoldBack(t, src, 64)
	refolded := gmParityRefold(t, back)
	// Documented drops on this direction: safetySettings and cachedContent have
	// no Messages spelling (spec §3.1 rows) — assert they stay dropped rather
	// than being smuggled back.
	if _, has := refolded["safetySettings"]; has {
		t.Fatalf("safetySettings came back: %v", refolded["safetySettings"])
	}
	if _, has := refolded["cachedContent"]; has {
		t.Fatalf("cachedContent came back: %v", refolded["cachedContent"])
	}
	want := gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"hi"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]},
	    {"role":"model","parts":[{"text":"answer"},{"function_call":{"name":"f","args":{"a":1}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"res"}}}]}],
	  "system_instruction":{"parts":[{"text":"be brief"}]},
	  "tools":[{"function_declarations":[{"name":"f","description":"d",
	    "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}},
	  "generationConfig":{"temperature":0.5,"top_p":0.9,"top_k":3,"seed":7,
	    "stop_sequences":["END"],"max_output_tokens":32,
	    "thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}},
	  "serviceTier":"priority"}`)
	gmParityEqual(t, refolded, want, "roundtrip")

	// The search-only body restores its tools exactly (same refold the
	// reference printed for this codec's output, this audit).
	searchSrc := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
	  "tools":[{"googleSearch":{}}],"generationConfig":{"max_output_tokens":16}}`
	refolded = gmParityRefold(t, gmParityFoldBack(t, searchSrc, 64))
	gmParityEqual(t, refolded["tools"], []any{map[string]any{"googleSearch": map[string]any{}}}, "search roundtrip")
}

// gmParityRefold folds a Messages body back through this codec's forward
// fold, serialized the way a host would hand it over.
func gmParityRefold(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ConvertRequest(Messages, Gemini, b)
	if err != nil {
		t.Fatalf("refold: %v", err)
	}
	return gmParityDecode(t, string(out.Body))
}

// A thought part takes the §3.1 shape: a thinking block whose signature is
// the part's own, and the signature on a plain text part has no Messages
// spelling and drops. An unsigned thought part folds back to a thinking block
// with NO signature key (the forward fold writes the key only for a real
// signature), so the fold pair is lossless in both directions — the
// reference's own forward used to print any thinking block as a thought part
// PLUS a signature part (proven this audit), duplicating the reasoning text
// on every replay; this fold dropped that duplication.
func TestGeminiMessagesParityReverseThoughtParts(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true},{"text":"answer"}]}]}`, 64)
	asst := got["messages"].([]any)[1].(map[string]any)
	blocks := asst["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("blocks = %#v, want thinking then text", blocks)
	}
	th := blocks[0].(map[string]any)
	if th["type"] != "thinking" || th["thinking"] != "why" {
		t.Fatalf("thinking block = %#v", th)
	}
	if _, has := th["signature"]; has {
		t.Fatalf("unsigned thought part grew a signature key: %#v", th)
	}
	if blocks[1].(map[string]any)["text"] != "answer" {
		t.Fatalf("text block = %#v", blocks[1])
	}

	// A signature on a plain text part drops (§3.1 row: the Messages wire has
	// no field for it on a text block).
	got = gmParityFoldBack(t, `{"contents":[
	    {"role":"model","parts":[{"text":"plain","thoughtSignature":"TSIG"}]}]}`, 64)
	blocks = got["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 1 || blocks[0].(map[string]any)["type"] != "text" {
		t.Fatalf("blocks = %#v, want one plain text block", blocks)
	}
	if _, has := blocks[0].(map[string]any)["signature"]; has {
		t.Fatalf("text block grew a signature: %#v", blocks[0])
	}

	// The full roundtrip of a thought part: the refold returns the same
	// single thought part, signature and all — the compact encoding the
	// forward fold now writes.
	refolded := gmParityRefold(t, gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true},{"text":"answer"}]}]}`, 64))
	want := gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true},{"text":"answer"}]}],
	  "generationConfig":{"max_output_tokens":64}}`)
	gmParityEqual(t, refolded, want, "thought roundtrip")

	// A signed thought part round-trips its signature onto the same part.
	refolded = gmParityRefold(t, gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"},{"text":"answer"}]}]}`, 64))
	want = gmParityDecode(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"},{"text":"answer"}]}],
	  "generationConfig":{"max_output_tokens":64}}`)
	gmParityEqual(t, refolded, want, "signed thought roundtrip")
}

// top_k, seed and the penalty controls ride INTO the Messages body: the
// reference's bridge passes them straight through (printed this audit —
// {"top_k":3,"seed":7,"frequency_penalty":0.5,"presence_penalty":0.2} all
// survive translate_anthropic_to_openai untouched) and the chat->gemini hop
// maps them back under the same names, so the faithful inverse carries them.
// (The spec's §3.1 table marks these rows "dropped"; its own D12 loss table
// says the reverse "fits" for seed and the penalty controls and lists neither
// as a loss — this codec follows the reference's real behavior.)
func TestGeminiMessagesParityReverseSamplingKnobs(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"top_k":3,"seed":7,"frequency_penalty":0.5,"presence_penalty":0.2,
	    "candidate_count":2,"responseLogprobs":true,"logprobs":5,
	    "responseModalities":["TEXT"]}}`, 64)
	for _, k := range []string{"top_k", "seed", "frequency_penalty", "presence_penalty"} {
		if _, has := got[k]; !has {
			t.Fatalf("%s dropped from the Messages body: %v", k, got)
		}
	}
	// The knobs the Messages dialect cannot spell stay dropped (D12's
	// LossLogprobs/LossMultiChoice rows for this direction).
	for _, k := range []string{"candidate_count", "responseLogprobs", "logprobs", "responseModalities"} {
		if _, has := got[k]; has {
			t.Fatalf("%s crossed into the Messages body", k)
		}
	}
}

// max_tokens derives per D11: the config's own ceiling wins, a thinking
// budget alone adds the dialect default (4096), and neither present leaves
// the field to the caller's fallback — or the MissingRequiredFieldError.
func TestGeminiMessagesParityReverseMaxTokens(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":32}}`, 64)
	if got["max_tokens"] != float64(32) {
		t.Fatalf("max_tokens = %v, want the config ceiling", got["max_tokens"])
	}
	got = gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true}}}`, 64)
	if got["max_tokens"] != float64(2048+4096) {
		t.Fatalf("max_tokens = %v, want budget+4096", got["max_tokens"])
	}
	_, err := ConvertRequest(Gemini, Messages, []byte(`{"contents":[{"role":"user","parts":[{"text":"x"}]}]}`))
	if err == nil {
		t.Fatal("want MissingRequiredFieldError without a fallback")
	}
	var missing MissingRequiredFieldError
	if !errors.As(err, &missing) {
		t.Fatalf("err = %T(%v), want MissingRequiredFieldError", err, err)
	}
}

// function_call ids pair with function_responses per §3.1's ladder: an id the
// call carried rides as-is; one the call lacked is minted positionally; a
// response pairs by name with the nearest same-named call; a response with no
// own id and no namesake still folds — minted — because dropping it would
// leave the conversation without an answer.
func TestGeminiMessagesParityReverseFunctionResponseIDs(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{},"id":"native_1"}},
	      {"function_call":{"name":"g","args":{}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r1"}}},
	      {"function_response":{"name":"g","response":{"content":"r2"}}}]}]}`, 64)
	asst := got["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if asst[0].(map[string]any)["id"] != "native_1" {
		t.Fatalf("call id = %#v, want the stated native_1", asst[0])
	}
	if asst[1].(map[string]any)["id"] != "call_1" {
		t.Fatalf("minted id = %#v, want call_1 (position after the named one)", asst[1])
	}
	user := got["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if user[0].(map[string]any)["tool_use_id"] != "native_1" {
		t.Fatalf("paired tool_use_id = %#v, want native_1 by name", user[0])
	}
	if user[1].(map[string]any)["tool_use_id"] != "call_1" {
		t.Fatalf("paired tool_use_id = %#v, want call_1 by name", user[1])
	}

	// A response matching no call at all still folds, minted positionally.
	got = gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"function_response":{"name":"orphan","response":{"content":"r"}}}]}]}`, 64)
	block := got["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_result" || block["tool_use_id"] != "call_0" {
		t.Fatalf("orphan result = %#v, want a minted tool_result", block)
	}
}

// The thinking budget buckets back through the same ladder the forward fold
// uses, so the written value is the one the ladder reproduces — 5000 reads as
// 4096, because that is what the reference's two-level fold would produce
// (spec §3.1 thinkingConfig row; forward ladder printed above).
func TestGeminiMessagesParityReverseThinkingLadder(t *testing.T) {
	for _, tc := range []struct {
		budget float64
		want   float64
	}{
		{128, 128},
		{700, 128},
		{2048, 2048},
		{4095, 2048},
		{4096, 4096},
		{5000, 4096},
		{16384, 4096},
	} {
		got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
		  "generationConfig":{"max_output_tokens":8,
		    "thinkingConfig":{"thinkingBudget":`+jsonNumber(tc.budget)+`,"includeThoughts":true}}}`, 64)
		thinking := got["thinking"].(map[string]any)
		if thinking["type"] != "enabled" || thinking["budget_tokens"] != tc.want {
			t.Fatalf("budget %v: thinking = %#v, want enabled/%v", tc.budget, thinking, tc.want)
		}
	}
	// includeThoughts:false is the off switch.
	got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":8,
	    "thinkingConfig":{"thinkingBudget":0,"includeThoughts":false}}}`, 64)
	gmParityEqual(t, got["thinking"], map[string]any{"type": "disabled"}, "disabled")
}

// jsonNumber renders a float as a JSON integer literal when it is one.
func jsonNumber(f float64) string {
	if f == float64(int(f)) {
		return itoaGemini(int(f))
	}
	b, _ := json.Marshal(f)
	return string(b)
}

func itoaGemini(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		return "-" + string(b)
	}
	return string(b)
}

// toolConfig modes fold back to their Messages spellings, and the
// includeServerSideToolInvocations flag has no Messages spelling and drops
// (§3.1 rows; VGS:331-346 forward basis).
func TestGeminiMessagesParityReverseToolConfig(t *testing.T) {
	for _, tc := range []struct {
		cfg  string
		want string
	}{
		{`{"mode":"AUTO"}`, `{"type":"auto"}`},
		{`{"mode":"ANY"}`, `{"type":"any"}`},
		{`{"mode":"NONE"}`, `{"type":"none"}`},
	} {
		got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
		  "toolConfig":{"functionCallingConfig":`+tc.cfg+`}}`, 64)
		gmParityEqual(t, got["tool_choice"], gmParityDecode(t, tc.want), tc.cfg)
	}
	got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}}}`, 64)
	gmParityEqual(t, got["tool_choice"], map[string]any{"type": "tool", "name": "f"}, "named")

	got = gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "toolConfig":{"includeServerSideToolInvocations":true}}`, 64)
	if _, has := got["tool_choice"]; has {
		t.Fatalf("includeServerSideToolInvocations crossed into tool_choice: %v", got["tool_choice"])
	}
}

// serviceTier inverts the forward rewrite: priority->auto, standard->default,
// anything else as written (TRF:1239-1246's inverse).
func TestGeminiMessagesParityReverseServiceTier(t *testing.T) {
	for _, tc := range []struct{ tier, want string }{
		{"priority", "auto"},
		{"standard", "default"},
		{"batch", "batch"},
	} {
		got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
		  "serviceTier":"`+tc.tier+`"}`, 64)
		if got["service_tier"] != tc.want {
			t.Fatalf("tier %s: service_tier = %v, want %s", tc.tier, got["service_tier"], tc.want)
		}
	}
}

// The response_mime_type/response_json_schema pair folds back to the
// output_format the Messages dialect reads, strict:false and schema-only
// (§3.1 row; forward basis VGS:754-797, printed in the output_format case
// above).
func TestGeminiMessagesParityReverseOutputFormat(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[{"role":"user","parts":[{"text":"x"}]}],
	  "generationConfig":{"max_output_tokens":8,
	    "response_mime_type":"application/json",
	    "response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}}`, 64)
	want := map[string]any{
		"type": "json_schema",
		"schema": map[string]any{
			"type":       "object",
			"properties": map[string]any{"a": map[string]any{"type": "string"}},
		},
		"strict": false,
	}
	gmParityEqual(t, got["output_format"], want, "output_format")
}

// file_data folds back to a url-source document block (§3.1 row; forward
// basis TRF:556-573), and the googleSearch-only tools case is covered by the
// roundtrip above. This also pins that mime_type does not ride the block —
// the Messages url source carries only the url.
func TestGeminiMessagesParityReverseFileData(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"file_data":{"mime_type":"application/pdf","file_uri":"https://x/a.pdf"}}]}]}`, 64)
	block := got["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if block["type"] != "document" {
		t.Fatalf("block = %#v, want a document", block)
	}
	src, _ := block["source"].(map[string]any)
	if src["type"] != "url" || src["url"] != "https://x/a.pdf" {
		t.Fatalf("source = %#v, want the url source", src)
	}
	if _, has := src["media_type"]; has {
		t.Fatalf("url source grew a media_type: %#v", src)
	}
}

// gmParityAssertAlternating fails when two neighbouring messages share a
// role — the adjacent-same-role shape the Messages upstream answers with a
// 400, which per convert.go's policy (a body that would 400 at the upstream
// is a conversion bug reported late) is a fold defect, not caller input.
func gmParityAssertAlternating(t *testing.T, msgs []any) {
	t.Helper()
	for i := 1; i < len(msgs); i++ {
		prev, _ := msgs[i-1].(map[string]any)
		cur, _ := msgs[i].(map[string]any)
		if prev["role"] == cur["role"] {
			t.Fatalf("messages[%d] and messages[%d] are both %v: the target rejects adjacent same-role messages", i-1, i, cur["role"])
		}
	}
}

// Same-role runs in the contents list fold into ONE message per run (2026-10-14).
// The Messages wire requires user/assistant alternation — the same requirement
// that made the two existing Messages folds merge runs (chatmessages/request.go
// "Anthropic requires the wire to alternate" and messagesresponses/from_responses.go
// "two consecutive assistant messages are a request the target rejects"). This
// direction has no forward path in the reference (file header above), so there
// is no parity conflict: the merge rule is the target's own wire law, and the
// reverse fold now mirrors the forward fold's run-merging (geminimessages
// request.go geminiContents) instead of re-splitting runs it had joined.
//
// The host scenario this guards: many gemini SDKs send each parallel function
// response as its own user content and append the follow-up prompt as another —
// a contents run of [model calls, user response, user response+text]. Folding
// that per-content produced adjacent user messages, a 400 shape at the target.
func TestGeminiMessagesParityReverseSameRoleRunsMerge(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{}}},
	      {"function_call":{"name":"g","args":{}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r1"}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"g","response":{"content":"r2"}}},
	      {"text":"now continue"}]}],
	  "generationConfig":{"max_output_tokens":16}}`, 64)
	want := gmParityDecode(t, `{"messages":[
	    {"role":"assistant","content":[
	      {"type":"tool_use","id":"call_0","name":"f","input":{}},
	      {"type":"tool_use","id":"call_1","name":"g","input":{}}]},
	    {"role":"user","content":[
	      {"type":"tool_result","tool_use_id":"call_0","content":"r1"},
	      {"type":"tool_result","tool_use_id":"call_1","content":"r2"},
	      {"type":"text","text":"now continue"}]}],
	  "max_tokens":16}`)
	gmParityEqual(t, got, want, "parallel tool results")
	msgs := got["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want exactly 2", len(msgs))
	}
	gmParityAssertAlternating(t, msgs)
}

// The assistant side of the same rule: consecutive model contents join one
// assistant message with the blocks in part order — text the model streamed
// before its calls stays ahead of the tool_use blocks (source order, no
// reordering).
func TestGeminiMessagesParityReverseModelRunsMerge(t *testing.T) {
	got := gmParityFoldBack(t, `{"contents":[
	    {"role":"user","parts":[{"text":"q"}]},
	    {"role":"model","parts":[{"text":"partial"}]},
	    {"role":"model","parts":[{"function_call":{"name":"f","args":{}}}]},
	    {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r"}}}]}],
	  "generationConfig":{"max_output_tokens":16}}`, 64)
	want := gmParityDecode(t, `{"messages":[
	    {"role":"user","content":[{"type":"text","text":"q"}]},
	    {"role":"assistant","content":[{"type":"text","text":"partial"},
	      {"type":"tool_use","id":"call_0","name":"f","input":{}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_0","content":"r"}]}],
	  "max_tokens":16}`)
	gmParityEqual(t, got, want, "model run merge")
	msgs := got["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want exactly 3", len(msgs))
	}
	gmParityAssertAlternating(t, msgs)
}

// A content that contributes no blocks — empty parts, parts absent — folds to
// no message at all. Keeping it produced an assistant message with an empty
// content array (the target rejects empty content) AND wedged the two user
// turns apart so they could not merge; skipping it lets the run close.
func TestGeminiMessagesParityReverseEmptyPartsSkip(t *testing.T) {
	for _, body := range []string{`{"contents":[
	    {"role":"user","parts":[{"text":"a"}]},
	    {"role":"model","parts":[]},
	    {"role":"user","parts":[{"text":"b"}]}],
	  "generationConfig":{"max_output_tokens":16}}`,
		`{"contents":[
	    {"role":"user","parts":[{"text":"a"}]},
	    {"role":"model"},
	    {"role":"user","parts":[{"text":"b"}]}],
	  "generationConfig":{"max_output_tokens":16}}`} {
		got := gmParityFoldBack(t, body, 64)
		msgs, _ := got["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("%s: messages = %d, want the two user contents joined into 1", body, len(msgs))
		}
		gmParityEqual(t, msgs[0].(map[string]any)["content"], gmParityDecode(t,
			`{"b":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`)["b"], "joined user content")
		gmParityAssertAlternating(t, msgs)
		for i, raw := range msgs {
			if blocks, ok := raw.(map[string]any)["content"].([]any); ok && len(blocks) == 0 {
				t.Fatalf("messages[%d] has an empty content array", i)
			}
		}
	}
}

// The library's own roundtrip must not manufacture the 400 shape: the forward
// fold (reference-pinned above) rewrites one Messages user turn carrying
// [tool_result, text] into TWO gemini user contents — the tool responses as
// their own content, the text as the next. Folding that back per-content
// returned adjacent user messages; the run merge restores the one turn the
// caller wrote, tool_result first and text after. The tool ids change
// (toolu_1 -> call_0) because the forward fold drops the id — the reference's
// own encoding, pinned by the two-hop case above — and the reverse mints
// positionally per §3.1.
func TestGeminiMessagesParityReverseRoundTripToolResultText(t *testing.T) {
	src := `{"model":"m","max_tokens":64,"messages":[
	    {"role":"user","content":"call the tool"},
	    {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"r"},
	      {"type":"text","text":"now continue"}]}]}`
	g, err := ConvertRequest(Messages, Gemini, []byte(src))
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	got := gmParityFoldBack(t, string(g.Body), 64)
	want := gmParityDecode(t, `{"messages":[
	    {"role":"user","content":[{"type":"text","text":"call the tool"}]},
	    {"role":"assistant","content":[{"type":"tool_use","id":"call_0","name":"f","input":{}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_0","content":"r"},
	      {"type":"text","text":"now continue"}]}],
	  "max_tokens":64}`)
	gmParityEqual(t, got, want, "roundtrip")
	gmParityAssertAlternating(t, got["messages"].([]any))
}
