package protocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// This file pins the Responses<->Gemini request folds against LiteLLM v1.100.1,
// run through the venv in .zcode/litellm-venv. The reference has no direct
// path between these dialects: every forward expectation below was produced by
// composing its two hops —
//   RSP.transform_responses_api_request_to_chat_completion_request
//   (litellm/responses/litellm_completion_transformation/transformation.py:275)
// then
//   litellm.get_optional_params(custom_llm_provider="vertex_ai") +
//   TRF.sync_transform_request_body
//   (litellm/llms/vertex_ai/gemini/transformation.py:1217-1247) —
// and pasting the result. The reverse direction (GeminiToResponses) has no
// forward path in the reference; its expectations are the spec's inverse rules,
// and each body was fed back through the reference's forward composition to
// confirm it restores the original (script: .zcode/roundtrip_reverse.py).

// gemRespParityRequest folds one Responses request body onto the gemini wire
// and returns the decoded result.
func gemRespParityRequest(t *testing.T, body string, drop bool) map[string]any {
	t.Helper()
	opts := ConvertOptions{}
	if drop {
		opts.DropParams = true
	}
	out, err := ConvertRequestWith(Responses, Gemini, []byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// gemRespParityRequestErr folds a body that is expected to fail.
func gemRespParityRequestErr(t *testing.T, body string, drop bool) error {
	t.Helper()
	opts := ConvertOptions{}
	if drop {
		opts.DropParams = true
	}
	_, err := ConvertRequestWith(Responses, Gemini, []byte(body), opts)
	if err == nil {
		t.Fatalf("body %s: want an error", body)
	}
	return err
}

// gemRespParityContents returns the folded contents list.
func gemRespParityContents(t *testing.T, m map[string]any) []any {
	t.Helper()
	list, ok := m["contents"].([]any)
	if !ok {
		t.Fatalf("contents missing or not a list: %#v", m["contents"])
	}
	return list
}

// gemRespParityParts returns one content's parts.
func gemRespParityParts(t *testing.T, content any) []any {
	t.Helper()
	parts, ok := content.(map[string]any)["parts"].([]any)
	if !ok {
		t.Fatalf("parts missing: %#v", content)
	}
	return parts
}

// A string input and a one-item input list fold to the same single user turn
// (RSP:275 builds one chat message either way; TRF:850-852 turns a string
// content into one text part as-is).
func TestResponsesToGeminiParityMinimalInput(t *testing.T) {
	for _, body := range []string{
		`{"model":"m","input":"hi"}`,
		`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
	} {
		m := gemRespParityRequest(t, body, false)
		contents := gemRespParityContents(t, m)
		if len(contents) != 1 {
			t.Fatalf("%s: contents = %#v", body, contents)
		}
		c := contents[0].(map[string]any)
		if c["role"] != "user" {
			t.Fatalf("%s: role = %v", body, c["role"])
		}
		parts := gemRespParityParts(t, c)
		if len(parts) != 1 || parts[0].(map[string]any)["text"] != "hi" {
			t.Fatalf("%s: parts = %#v", body, parts)
		}
		if _, has := m["generationConfig"]; has {
			t.Fatalf("%s: generationConfig should be absent: %#v", body, m["generationConfig"])
		}
	}
	// An empty-string input is a stated empty turn, not the blank fallback.
	m := gemRespParityRequest(t, `{"model":"m","input":""}`, false)
	parts := gemRespParityParts(t, gemRespParityContents(t, m)[0])
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "" {
		t.Fatalf(`input "": parts = %#v, want [{"text":""}]`, parts)
	}
}

// Instructions and system-role items both land in the system instruction
// block (RSP:311-320 turns instructions into a system message; TRF:1399-1412
// extracts every system message into one part each).
func TestResponsesToGeminiParitySystemInstruction(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","instructions":"be terse",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`, false)
	sys, ok := m["system_instruction"].(map[string]any)
	if !ok {
		t.Fatalf("system_instruction missing: %#v", m)
	}
	parts := sys["parts"].([]any)
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "be terse" {
		t.Fatalf("system parts = %#v", parts)
	}

	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"message","role":"system","content":[{"type":"input_text","text":"sys here"}]},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`, false)
	sys = m["system_instruction"].(map[string]any)
	if got := sys["parts"].([]any)[0].(map[string]any)["text"]; got != "sys here" {
		t.Fatalf("system item text = %v", got)
	}
}

// The full role walk: user question, model call, user tool result, user
// follow-up — four separate contents, the tool-result turn its own user
// content (FAC:1443-1476 pairs the result with the call by id and
// TRF:723-724 keeps the following user turn separate).
func TestResponsesToGeminiParityToolHistory(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
		{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\": \"Boston\"}"},
		{"type":"function_call_output","call_id":"call_1","output":"sunny"},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"thanks"}]}]}`, false)
	contents := gemRespParityContents(t, m)
	if len(contents) != 4 {
		t.Fatalf("contents = %#v, want 4 turns", contents)
	}
	wantRoles := []string{"user", "model", "user", "user"}
	for i, want := range wantRoles {
		if contents[i].(map[string]any)["role"] != want {
			t.Fatalf("content %d role = %v, want %v", i, contents[i].(map[string]any)["role"], want)
		}
	}
	call := gemRespParityParts(t, contents[1])[0].(map[string]any)["function_call"].(map[string]any)
	if call["name"] != "get_weather" || fmtJSON(call["args"]) != `{"city":"Boston"}` {
		t.Fatalf("function_call = %#v", call)
	}
	// 2.5 semantics: the call id is not written back (VGS:278-285).
	if _, has := call["id"]; has {
		t.Fatalf("function_call must not carry an id: %#v", call)
	}
	resp := gemRespParityParts(t, contents[2])[0].(map[string]any)["function_response"].(map[string]any)
	if resp["name"] != "get_weather" || fmtJSON(resp["response"]) != `{"content":"sunny"}` {
		t.Fatalf("function_response = %#v", resp)
	}

	// A result whose output parses as a JSON object rides as that object.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c2","name":"f","arguments":"{\"a\": 1}"},
		{"type":"function_call_output","call_id":"c2","output":"{\"ok\": true}"}]}`, false)
	fr := gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"ok":true}` {
		t.Fatalf("parsed response = %#v", fr["response"])
	}
}

// fmtJSON renders a decoded value compactly for exact comparison.
func fmtJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "<marshal error>"
	}
	return string(b)
}

// Consecutive same-role items merge into one content, each keeping its own
// part (TRF:723-724 groups consecutive same-role messages).
func TestResponsesToGeminiParityConsecutiveRolesMerge(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"one"}]},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"two"}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"r1"}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"r2"}]}]}`, false)
	contents := gemRespParityContents(t, m)
	if len(contents) != 2 {
		t.Fatalf("contents = %#v, want two merged turns", contents)
	}
	for i, want := range [][]string{{"one", "two"}, {"r1", "r2"}} {
		parts := gemRespParityParts(t, contents[i])
		if len(parts) != len(want) {
			t.Fatalf("content %d parts = %#v", i, parts)
		}
		for j, w := range want {
			if parts[j].(map[string]any)["text"] != w {
				t.Fatalf("content %d part %d = %#v, want %q", i, j, parts[j], w)
			}
		}
	}
}

// Reasoning replay: a reasoning item ahead of an assistant message merges into
// that turn (thought part first); its encrypted content replays as a signed
// thinking part; a standalone reasoning item ahead of a user turn becomes its
// own model turn (RSP:1296-1330; TRF:880-901).
func TestResponsesToGeminiParityReasoningReplay(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"reasoning","summary":[{"type":"summary_text","text":"ponder"}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}]}`, false)
	parts := gemRespParityParts(t, gemRespParityContents(t, m)[0])
	if len(parts) != 2 {
		t.Fatalf("parts = %#v, want [thought, text]", parts)
	}
	th := parts[0].(map[string]any)
	if th["text"] != "ponder" || th["thought"] != true {
		t.Fatalf("thought part = %#v", th)
	}
	if parts[1].(map[string]any)["text"] != "a" {
		t.Fatalf("text part = %#v", parts[1])
	}

	enc := `[{"type":"thinking","thinking":"deep","signature":"SIG"}]`
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"reasoning","encrypted_content":`+quoteJSON(enc)+`},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}]}`, false)
	parts = gemRespParityParts(t, gemRespParityContents(t, m)[0])
	sig := parts[0].(map[string]any)
	if sig["text"] != "deep" || sig["thoughtSignature"] != "SIG" {
		t.Fatalf("signed thinking part = %#v", sig)
	}
	if _, has := sig["thought"]; has {
		t.Fatalf("signed replay part must not carry thought: %#v", sig)
	}

	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"reasoning","summary":[{"type":"summary_text","text":"just thinking"}]},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`, false)
	contents := gemRespParityContents(t, m)
	if len(contents) != 2 || contents[0].(map[string]any)["role"] != "model" {
		t.Fatalf("standalone reasoning should be its own model turn: %#v", contents)
	}
}

// quoteJSON embeds a JSON fragment as a string literal inside a test body.
func quoteJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// Media walk: data URLs become inline data, https URLs with a known extension
// and gs:// URIs become file references with their media type, and the wire's
// own file storage passes through untouched (TRF:556-579). An image-only turn
// gains the blank text part the reference appends (TRF:856-870).
func TestResponsesToGeminiParityMedia(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[{"type":"message","role":"user","content":[
		{"type":"input_text","text":"look"},
		{"type":"input_image","image_url":"data:image/png;base64,AAAB"},
		{"type":"input_file","filename":"d.pdf","file_data":"data:application/pdf;base64,JVBE"},
		{"type":"input_file","file_url":"https://example.com/cat.png"},
		{"type":"input_file","file_url":"gs://bucket/vid.mp4"},
		{"type":"input_file","file_url":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}]}]}`, false)
	parts := gemRespParityParts(t, gemRespParityContents(t, m)[0])
	if len(parts) != 6 {
		t.Fatalf("parts = %#v, want six", parts)
	}
	if parts[0].(map[string]any)["text"] != "look" {
		t.Fatalf("text part = %#v", parts[0])
	}
	want := []string{
		`{"data":"AAAB","mime_type":"image/png"}`,
		`{"data":"JVBE","mime_type":"application/pdf"}`,
		`{"file_uri":"https://example.com/cat.png","mime_type":"image/png"}`,
		`{"file_uri":"gs://bucket/vid.mp4","mime_type":"video/mp4"}`,
		`{"file_uri":"https://generativelanguage.googleapis.com/v1beta/files/abc123?alt=media"}`,
	}
	for i, w := range want {
		pm := parts[i+1].(map[string]any)
		var key string
		if _, ok := pm["inline_data"]; ok {
			key = "inline_data"
		} else {
			key = "file_data"
		}
		if fmtJSON(pm[key]) != w {
			t.Fatalf("part %d %s = %s, want %s", i+1, key, fmtJSON(pm[key]), w)
		}
	}

	// An image-only turn carries the blank text part at the end.
	m = gemRespParityRequest(t, `{"model":"m","input":[{"type":"message","role":"user","content":[
		{"type":"input_image","image_url":"data:image/png;base64,AAAB"}]}]}`, false)
	parts = gemRespParityParts(t, gemRespParityContents(t, m)[0])
	if len(parts) != 2 || parts[1].(map[string]any)["text"] != " " {
		t.Fatalf("image-only parts = %#v, want media + blank text", parts)
	}

	// An audio part has no gemini spelling: the bridge drops it and the whole
	// user turn vanishes when nothing else is in it (RSP:1903 drops input_audio;
	// TRF's empty-list content produces no content at all).
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_audio","input_audio":{"data":"QUJD","format":"wav"}}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}]}`, false)
	contents := gemRespParityContents(t, m)
	if len(contents) != 1 || contents[0].(map[string]any)["role"] != "model" {
		t.Fatalf("audio-only turn should vanish: %#v", contents)
	}
}

// Function tools merge into one declarations block: strict and a false
// additionalProperties are stripped, a missing description is stated blank,
// and absent parameters become the bare object (VGS:655-703; CUT:577-626).
func TestResponsesToGeminiParityFunctionTools(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[
			{"type":"function","name":"lookup","description":"d","strict":true,
			 "parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}},
			{"type":"function","name":"noargs"}]}`, false)
	tools := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %#v, want one merged block", tools)
	}
	decls := tools[0].(map[string]any)["function_declarations"].([]any)
	if len(decls) != 2 {
		t.Fatalf("declarations = %#v", decls)
	}
	d0 := decls[0].(map[string]any)
	if d0["description"] != "d" || fmtJSON(d0["parameters"]) != `{"properties":{"q":{"type":"string"}},"required":["q"],"type":"object"}` {
		t.Fatalf("declaration 0 = %#v", d0)
	}
	d1 := decls[1].(map[string]any)
	if d1["description"] != "" || fmtJSON(d1["parameters"]) != `{"type":"object"}` {
		t.Fatalf("declaration 1 = %#v", d1)
	}
}

// The schema builder's pass order is observable: the null-to-nullable pass
// runs before the type-array widening, so a type array keeps its null branch
// while a pre-built anyOf loses it (CUT:604-607 calls
// convert_anyof_null_to_nullable before _convert_schema_types; CUT:773-812,
// CUT:876-951). Enums drop empty strings to null and vanish off non-string
// types; array items are completed; $refs are inlined.
func TestResponsesToGeminiParitySchemaNormalisation(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"function","name":"s","parameters":{
			"type":"object",
			"properties":{
				"u":{"$ref":"#/$defs/Unit"},
				"n":{"type":["string","null"]},
				"m":{"anyOf":[{"type":"string"},{"type":"null"}]},
				"e":{"type":"string","enum":["a",""]},
				"i":{"type":"integer","enum":[1,2]},
				"arr":{"type":"array"}},
			"$defs":{"Unit":{"type":"string"}}}}]}`, false)
	decl := m["tools"].([]any)[0].(map[string]any)["function_declarations"].([]any)[0].(map[string]any)
	want := `{"properties":{"arr":{"items":{"type":"object"},"type":"array"},` +
		`"e":{"enum":["a",null],"type":"string"},` +
		`"i":{"type":"integer"},` +
		`"m":{"anyOf":[{"nullable":true,"type":"string"}]},` +
		`"n":{"anyOf":[{"type":"string"},{"type":"null"}]},` +
		`"u":{"type":"string"}},"type":"object"}`
	if fmtJSON(decl["parameters"]) != want {
		t.Fatalf("parameters = %s\nwant %s", fmtJSON(decl["parameters"]), want)
	}
}

// A custom tool rides as a function whose description carries the grammar and
// whose parameters wrap the one freeform string; a namespace flattens to
// `namespace__tool` names (RSP:1583 wraps custom input; the reference's tool
// translation builds both shapes).
func TestResponsesToGeminiParityCustomAndNamespaceTools(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"custom","name":"roll","description":"roll dice",
			"format":{"type":"grammar","syntax":"lark","definition":"start: DIGIT"}}]}`, false)
	decl := m["tools"].([]any)[0].(map[string]any)["function_declarations"].([]any)[0].(map[string]any)
	if decl["name"] != "roll" {
		t.Fatalf("name = %v", decl["name"])
	}
	if decl["description"] != "roll dice\n\nFormat:\n```lark\nstart: DIGIT\n```" {
		t.Fatalf("description = %q", decl["description"])
	}
	if fmtJSON(decl["parameters"]) != `{"properties":{"content":{"description":"The roll content following the specified format","type":"string"}},"required":["content"],"type":"object"}` {
		t.Fatalf("parameters = %s", fmtJSON(decl["parameters"]))
	}

	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"namespace","name":"calc","description":"math ops",
			"tools":[{"type":"function","name":"add","description":"add","parameters":{"type":"object"}}]}]}`, false)
	decl = m["tools"].([]any)[0].(map[string]any)["function_declarations"].([]any)[0].(map[string]any)
	if decl["name"] != "calc__add" || decl["description"] != "math ops\n\nadd" {
		t.Fatalf("namespace declaration = %#v", decl)
	}
}

// The search tool becomes the wire's own search tool, is dropped when function
// declarations are also present, and tool types the chat hop drops never
// arrive while the ones it forwards but the wire ignores leave the key empty
// (VGS:348-354, VGS:516-563, VGS:699-741; RSP:1903).
func TestResponsesToGeminiParitySearchAndServerTools(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"web_search_preview"}]}`, false)
	if fmtJSON(m["tools"]) != `[{"googleSearch":{}}]` {
		t.Fatalf("search tools = %s", fmtJSON(m["tools"]))
	}

	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"web_search_preview"},{"type":"function","name":"f","parameters":{"type":"object"}}]}`, false)
	if fmtJSON(m["tools"]) != `[{"function_declarations":[{"description":"","name":"f","parameters":{"type":"object"}}]}]` {
		t.Fatalf("mixed tools = %s", fmtJSON(m["tools"]))
	}

	// image_generation is dropped on the chat hop: no tools key at all.
	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"image_generation"}]}`, false)
	if _, has := m["tools"]; has {
		t.Fatalf("image_generation should leave no tools key: %#v", m["tools"])
	}

	// code_interpreter is forwarded by the chat hop and ignored by the wire's
	// tool builder: the key survives with no entries.
	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}`, false)
	tools, ok := m["tools"].([]any)
	if !ok || len(tools) != 0 {
		t.Fatalf("forwarded server tool should leave an empty tools list: %#v", m["tools"])
	}
}

// tool_choice maps onto the calling config's three modes plus the named
// function shape; both Responses dict spellings converge on the same chat
// shape first (VGS:331-346; RSP:337-341).
func TestResponsesToGeminiParityToolChoice(t *testing.T) {
	cases := []struct {
		choice string
		want   string
	}{
		{`"auto"`, `{"functionCallingConfig":{"mode":"AUTO"}}`},
		{`"none"`, `{"functionCallingConfig":{"mode":"NONE"}}`},
		{`"required"`, `{"functionCallingConfig":{"mode":"ANY"}}`},
		{`{"type":"function","name":"lookup"}`, `{"functionCallingConfig":{"allowed_function_names":["lookup"],"mode":"ANY"}}`},
		{`{"type":"function","function":{"name":"lookup"}}`, `{"functionCallingConfig":{"allowed_function_names":["lookup"],"mode":"ANY"}}`},
	}
	for _, tc := range cases {
		m := gemRespParityRequest(t, `{"model":"m",
			"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
			"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
			"tool_choice":`+tc.choice+`}`, false)
		if fmtJSON(m["toolConfig"]) != tc.want {
			t.Fatalf("tool_choice %s: toolConfig = %s, want %s", tc.choice, fmtJSON(m["toolConfig"]), tc.want)
		}
	}
}

// The effort ladder carries fixed budgets; a string effort reads the same; the
// summary half of the reasoning object has no landing; an unknown level is
// refused (VGS:799-846, VGS:1155-1187).
func TestResponsesToGeminiParityThinking(t *testing.T) {
	cases := []struct {
		reasoning string
		want      string
	}{
		{`{"effort":"minimal"}`, `{"includeThoughts":true,"thinkingBudget":128}`},
		{`{"effort":"low"}`, `{"includeThoughts":true,"thinkingBudget":1024}`},
		{`{"effort":"medium"}`, `{"includeThoughts":true,"thinkingBudget":2048}`},
		{`{"effort":"high"}`, `{"includeThoughts":true,"thinkingBudget":4096}`},
		{`{"effort":"none"}`, `{"includeThoughts":false,"thinkingBudget":0}`},
		{`"low"`, `{"includeThoughts":true,"thinkingBudget":1024}`},
		{`{"effort":"low","summary":"auto"}`, `{"includeThoughts":true,"thinkingBudget":1024}`},
	}
	for _, tc := range cases {
		m := gemRespParityRequest(t, `{"model":"m",
			"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
			"reasoning":`+tc.reasoning+`}`, false)
		gc := m["generationConfig"].(map[string]any)
		if fmtJSON(gc["thinkingConfig"]) != tc.want {
			t.Fatalf("reasoning %s: thinkingConfig = %s, want %s", tc.reasoning, fmtJSON(gc["thinkingConfig"]), tc.want)
		}
	}
	err := gemRespParityRequestErr(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"reasoning":{"effort":"xhigh"}}`, false)
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("xhigh effort: err = %T(%v), want ConversionError", err, err)
	}
}

// Sampling and structured output: the envelope's strict is stripped while a
// false additionalProperties inside the schema rides; json_object names no
// schema; a text format carries nothing at all (VGS:754-797; the reference's
// _remove_strict_from_schema, litellm/utils.py:3790-3808).
func TestResponsesToGeminiParityGenerationConfig(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"max_output_tokens":20,"temperature":0.5,"top_p":0.9}`, false)
	gc := m["generationConfig"].(map[string]any)
	if fmtJSON(gc) != `{"max_output_tokens":20,"temperature":0.5,"top_p":0.9}` {
		t.Fatalf("generationConfig = %s", fmtJSON(gc))
	}

	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"text":{"format":{"type":"json_schema","name":"o","strict":true,
			"schema":{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}}}}`, false)
	gc = m["generationConfig"].(map[string]any)
	if gc["response_mime_type"] != "application/json" {
		t.Fatalf("mime = %v", gc["response_mime_type"])
	}
	if fmtJSON(gc["response_json_schema"]) != `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"type":"object"}` {
		t.Fatalf("response_json_schema = %s", fmtJSON(gc["response_json_schema"]))
	}

	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"text":{"format":{"type":"json_object"}}}`, false)
	gc = m["generationConfig"].(map[string]any)
	if fmtJSON(gc) != `{"response_mime_type":"application/json"}` {
		t.Fatalf("json_object generationConfig = %s", fmtJSON(gc))
	}

	m = gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"text":{"format":{"type":"text"}}}`, false)
	if _, has := m["generationConfig"]; has {
		t.Fatalf("text format must fold nothing: %#v", m["generationConfig"])
	}
}

// Parameters with no landing on the wire: metadata, user, service_tier,
// truncation and parallel_tool_calls all vanish from the body (the bridge
// forwards only user, and the request fold reads none of them; RSP:337-341,
// TRF:1203-1247).
func TestResponsesToGeminiParityDroppedParams(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"metadata":{"foo":"bar"},"user":"u1","service_tier":"auto","truncation":"auto",
		"parallel_tool_calls":false}`, false)
	for _, key := range []string{"metadata", "user", "serviceTier", "truncation", "parallel_tool_calls"} {
		if _, has := m[key]; has {
			t.Fatalf("%s should not ride the wire body: %#v", key, m[key])
		}
	}
}

// context_management is the one Responses parameter the reference forwards and
// then refuses: an error unless the caller asked for silent dropping
// (litellm/utils.py:4186-4213 via _check_valid_arg).
func TestResponsesToGeminiParityContextManagement(t *testing.T) {
	body := `{"model":"m",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"context_management":{"edits":[{"type":"clear_tool_uses_20250919"}]}}`
	err := gemRespParityRequestErr(t, body, false)
	var unsup UnsupportedParamError
	if !errors.As(err, &unsup) || unsup.Param != "context_management" {
		t.Fatalf("err = %T(%v), want UnsupportedParamError(context_management)", err, err)
	}
	m := gemRespParityRequest(t, body, true)
	if _, has := m["context_management"]; has {
		t.Fatalf("context_management should drop silently: %#v", m["context_management"])
	}
}

// Tool results in every shape the bridge produces: None, a text list, a list
// with output_image parts (dropped — the bridge only promotes input_image),
// a list with input_image parts (kept as inline data beside the response), a
// bare image data URL string, a custom tool call, and a signature embedded in
// the call id (RSP:1431-1500; FAC:1465-1500).
func TestResponsesToGeminiParityToolResultShapes(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":null}]}`, false)
	fr := gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"content":""}` {
		t.Fatalf("null output response = %s", fmtJSON(fr["response"]))
	}

	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":[
			{"type":"output_text","text":"line1\n"},{"type":"output_text","text":"line2"}]}]}`, false)
	fr = gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"content":"line1\nline2"}` {
		t.Fatalf("text list response = %s", fmtJSON(fr["response"]))
	}
	if _, has := fr["parts"]; has {
		t.Fatalf("no media should ride a text-only result: %#v", fr)
	}

	// output_image is not a shape the bridge promotes: texts join, no parts.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":[
			{"type":"output_text","text":"chart"},
			{"type":"output_image","image_url":"data:image/png;base64,AAAB"}]}]}`, false)
	fr = gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"content":"chart"}` || len(fr) != 2 {
		t.Fatalf("output_image result = %#v", fr)
	}

	// input_image IS promoted: the texts fold into the response and the image
	// rides as nested inline data.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":[
			{"type":"output_text","text":"chart"},
			{"type":"input_image","image_url":"data:image/png;base64,AAAB"}]}]}`, false)
	fr = gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"content":"chart"}` {
		t.Fatalf("mixed result response = %s", fmtJSON(fr["response"]))
	}
	if fmtJSON(fr["parts"]) != `[{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]` {
		t.Fatalf("mixed result parts = %s", fmtJSON(fr["parts"]))
	}

	// A bare image data URL string is promoted the same way.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":"data:image/png;base64,AAAB"}]}`, false)
	fr = gemRespParityParts(t, gemRespParityContents(t, m)[1])[0].(map[string]any)["function_response"].(map[string]any)
	if fmtJSON(fr["response"]) != `{"content":""}` || fmtJSON(fr["parts"]) != `[{"inline_data":{"data":"AAAB","mime_type":"image/png"}}]` {
		t.Fatalf("data URL result = %#v", fr)
	}

	// A custom tool call wraps its freeform input as the content argument.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"custom_tool_call","call_id":"ct1","name":"roll","input":"3"},
		{"type":"custom_tool_call_output","call_id":"ct1","output":"rolled"}]}`, false)
	call := gemRespParityParts(t, gemRespParityContents(t, m)[0])[0].(map[string]any)["function_call"].(map[string]any)
	if fmtJSON(call["args"]) != `{"content":"3"}` {
		t.Fatalf("custom call args = %s", fmtJSON(call["args"]))
	}

	// A signature embedded in the call id rides the function_call part.
	m = gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"call_1__thought__SIG","name":"f","arguments":"{\"a\": 1}"},
		{"type":"function_call_output","call_id":"call_1__thought__SIG","output":"ok"}]}`, false)
	part := gemRespParityParts(t, gemRespParityContents(t, m)[0])[0].(map[string]any)
	if part["thoughtSignature"] != "SIG" {
		t.Fatalf("thoughtSignature = %v", part["thoughtSignature"])
	}
}

// Two identical calls in one turn deduplicate onto one part, but both results
// still pair by their own ids — the id-to-name table keeps every call the
// caller made (FAC:1443-1449 answers each tool message by its own id).
func TestResponsesToGeminiParityDuplicateCalls(t *testing.T) {
	m := gemRespParityRequest(t, `{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\": 1}"},
		{"type":"function_call","call_id":"c2","name":"f","arguments":"{\"a\": 1}"},
		{"type":"function_call_output","call_id":"c1","output":"r1"},
		{"type":"function_call_output","call_id":"c2","output":"r2"}]}`, false)
	contents := gemRespParityContents(t, m)
	modelParts := gemRespParityParts(t, contents[1])
	if len(modelParts) != 1 {
		t.Fatalf("model parts = %#v, want the deduplicated call", modelParts)
	}
	resultParts := gemRespParityParts(t, contents[2])
	if len(resultParts) != 2 {
		t.Fatalf("result parts = %#v, want both answers", resultParts)
	}
	for i, want := range []string{"r1", "r2"} {
		fr := resultParts[i].(map[string]any)["function_response"].(map[string]any)
		if fmtJSON(fr["response"]) != `{"content":"`+want+`"}` {
			t.Fatalf("result %d = %#v, want %q", i, fr, want)
		}
	}
}

// Arguments that are not one JSON value fail the request — the reference's
// own refusal (FAC:1130).
func TestResponsesToGeminiParityBadArgumentsFail(t *testing.T) {
	err := gemRespParityRequestErr(t, `{"model":"m","input":[
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\": "}]}`, false)
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("err = %T(%v), want ConversionError", err, err)
	}
}

// ---------------------------------------------------------------------------
// Reverse: gemini request -> Responses request (derived, round-trip checked)
// ---------------------------------------------------------------------------

// The reverse fold's outputs below were fed back through the reference's
// forward composition (.zcode/roundtrip_reverse.py) and restored the original
// bodies up to the documented forward-side drops: safetySettings and
// cachedContent have no chat landing, the bridge drops service_tier, system
// parts join into one instructions string, a blank description is added to
// declarations that lacked one, a text/plain format folds nothing, ids are
// not written back under 2.5 semantics, and a bare thoughtSignature has no
// Responses spelling.
func TestGeminiToResponsesReverseParityFull(t *testing.T) {
	body := `{"contents":[
		{"role":"user","parts":[{"text":"hello"},{"inline_data":{"data":"AAAB","mime_type":"image/png"}},{"file_data":{"mime_type":"application/pdf","file_uri":"https://x/d.pdf"}}]},
		{"role":"model","parts":[{"text":"ponder","thought":true},{"text":"hi there"},{"function_call":{"name":"f","args":{"a":1}}}]},
		{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"ok"}}}]}],
		"system_instruction":{"parts":[{"text":"line one"},{"text":"line two"}]},
		"tools":[{"function_declarations":[{"name":"f","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}},{"name":"g","parameters":{"type":"object"}}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}},
		"generationConfig":{"max_output_tokens":20,"temperature":0.5,"top_p":0.9,
			"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true},
			"response_mime_type":"application/json","response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}}},
		"serviceTier":"priority","safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT"}],"cachedContent":"cc-1"}`
	out, err := ConvertRequest(Gemini, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["instructions"] != "line one\nline two" {
		t.Fatalf("instructions = %#v", m["instructions"])
	}
	input := m["input"].([]any)
	if len(input) != 5 {
		t.Fatalf("input = %#v, want five items", input)
	}
	// user message: text + image + file
	msg := input[0].(map[string]any)
	if msg["type"] != "message" || msg["role"] != "user" {
		t.Fatalf("first item = %#v", msg)
	}
	content := msg["content"].([]any)
	if content[0].(map[string]any)["type"] != "input_text" || content[0].(map[string]any)["text"] != "hello" {
		t.Fatalf("text part = %#v", content[0])
	}
	img := content[1].(map[string]any)
	if img["type"] != "input_image" || img["image_url"] != "data:image/png;base64,AAAB" {
		t.Fatalf("image part = %#v", img)
	}
	file := content[2].(map[string]any)
	if file["type"] != "input_file" || file["file_url"] != "https://x/d.pdf" {
		t.Fatalf("file part = %#v", file)
	}
	// reasoning ahead of the assistant turn
	rsn := input[1].(map[string]any)
	if rsn["type"] != "reasoning" || fmtJSON(rsn["summary"]) != `[{"text":"ponder","type":"summary_text"}]` {
		t.Fatalf("reasoning item = %#v", rsn)
	}
	// assistant message
	asst := input[2].(map[string]any)
	if asst["role"] != "assistant" || asst["content"].([]any)[0].(map[string]any)["text"] != "hi there" {
		t.Fatalf("assistant item = %#v", asst)
	}
	// call and output pair on the minted id
	call := input[3].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_0" || call["name"] != "f" || call["arguments"] != `{"a":1}` {
		t.Fatalf("call item = %#v", call)
	}
	output := input[4].(map[string]any)
	if output["type"] != "function_call_output" || output["call_id"] != "call_0" || output["output"] != "ok" {
		t.Fatalf("output item = %#v", output)
	}
	// parameters
	if m["max_output_tokens"] != float64(20) || m["temperature"] != 0.5 || m["top_p"] != 0.9 {
		t.Fatalf("sampling = %#v", m)
	}
	if fmtJSON(m["reasoning"]) != `{"effort":"medium"}` {
		t.Fatalf("reasoning = %s", fmtJSON(m["reasoning"]))
	}
	if fmtJSON(m["text"]) != `{"format":{"name":"response_schema","schema":{"properties":{"a":{"type":"string"}},"type":"object"},"strict":false,"type":"json_schema"}}` {
		t.Fatalf("text = %s", fmtJSON(m["text"]))
	}
	if fmtJSON(m["tool_choice"]) != `{"name":"f","type":"function"}` {
		t.Fatalf("tool_choice = %s", fmtJSON(m["tool_choice"]))
	}
	tools := m["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %#v", tools)
	}
	if fmtJSON(tools[0]) != `{"description":"d","name":"f","parameters":{"properties":{"q":{"type":"string"}},"type":"object"},"strict":false,"type":"function"}` {
		t.Fatalf("tool 0 = %s", fmtJSON(tools[0]))
	}
	if fmtJSON(tools[1]) != `{"description":"","name":"g","parameters":{"type":"object"},"strict":false,"type":"function"}` {
		t.Fatalf("tool 1 = %s", fmtJSON(tools[1]))
	}
	if m["service_tier"] != "auto" {
		t.Fatalf("service_tier = %#v", m["service_tier"])
	}
	// documented drops: safetySettings and cachedContent have no landing here
	for _, key := range []string{"safetySettings", "cachedContent"} {
		if _, has := m[key]; has {
			t.Fatalf("%s must not ride the Responses body: %#v", key, m[key])
		}
	}
}

// The wire's search tool folds to the Responses search spelling; the calling
// config's AUTO mode folds to the auto choice.
func TestGeminiToResponsesReverseParitySearch(t *testing.T) {
	out, err := ConvertRequest(Gemini, Responses, []byte(`{"contents":[{"role":"user","parts":[{"text":"q"}]}],
		"tools":[{"googleSearch":{}}],"toolConfig":{"functionCallingConfig":{"mode":"AUTO"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if fmtJSON(m["tools"]) != `[{"type":"web_search_preview"}]` {
		t.Fatalf("tools = %s", fmtJSON(m["tools"]))
	}
	if m["tool_choice"] != "auto" {
		t.Fatalf("tool_choice = %#v", m["tool_choice"])
	}
}

// Reverse thinking: a budget buckets back onto the effort level whose budget
// it meets; thinking switched off maps to none (types.EffortFromBudget).
func TestGeminiToResponsesReverseParityThinking(t *testing.T) {
	cases := []struct {
		config string
		want   string
	}{
		{`{"thinkingBudget":2048,"includeThoughts":true}`, `{"effort":"medium"}`},
		{`{"thinkingBudget":128,"includeThoughts":true}`, `{"effort":"minimal"}`},
		{`{"thinkingBudget":4096,"includeThoughts":true}`, `{"effort":"high"}`},
		{`{"thinkingBudget":1024,"includeThoughts":true}`, `{"effort":"low"}`},
		{`{"thinkingBudget":0,"includeThoughts":false}`, `{"effort":"none"}`},
	}
	for _, tc := range cases {
		out, err := ConvertRequest(Gemini, Responses, []byte(`{"contents":[{"role":"model","parts":[{"text":"a"}]}],
			"generationConfig":{"thinkingConfig":`+tc.config+`}}`))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if fmtJSON(m["reasoning"]) != tc.want {
			t.Fatalf("config %s: reasoning = %s, want %s", tc.config, fmtJSON(m["reasoning"]), tc.want)
		}
	}
}

// Reverse structured output and tiers: json without a schema is the plain
// object request; text/plain folds the explicit text format; the standard
// tier maps back to default.
func TestGeminiToResponsesReverseParityFormatsAndTiers(t *testing.T) {
	out, err := ConvertRequest(Gemini, Responses, []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"generationConfig":{"response_mime_type":"application/json"}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if fmtJSON(m["text"]) != `{"format":{"type":"json_object"}}` {
		t.Fatalf("json only = %s", fmtJSON(m["text"]))
	}

	out, err = ConvertRequest(Gemini, Responses, []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"generationConfig":{"response_mime_type":"text/plain"}}`))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if fmtJSON(m["text"]) != `{"format":{"type":"text"}}` {
		t.Fatalf("text/plain = %s", fmtJSON(m["text"]))
	}

	out, err = ConvertRequest(Gemini, Responses, []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"serviceTier":"standard"}`))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if m["service_tier"] != "default" {
		t.Fatalf("service_tier = %#v", m["service_tier"])
	}
}

// A function_response carrying its own id (Gemini 3 provenance) keeps it; the
// reverse fold's minted ids are stable in document order.
func TestGeminiToResponsesReverseParityCallIDs(t *testing.T) {
	out, err := ConvertRequest(Gemini, Responses, []byte(`{"contents":[
		{"role":"model","parts":[{"function_call":{"name":"f","args":{},"id":"call_9"}}]},
		{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"ok"},"id":"call_9"}}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	call := input[0].(map[string]any)
	if call["call_id"] != "call_9" {
		t.Fatalf("call_id = %#v", call["call_id"])
	}
	if call["arguments"] != "{}" {
		t.Fatalf("arguments = %#v", call["arguments"])
	}
	output := input[1].(map[string]any)
	if output["call_id"] != "call_9" || output["output"] != "ok" {
		t.Fatalf("output item = %#v", output)
	}
}

// A system-only body folds the instructions into a system-role input item —
// the Responses upstream rejects an empty input, so the instructions ride an
// item instead of serialising input:null.
func TestGeminiToResponsesReverseParitySystemOnly(t *testing.T) {
	out, err := ConvertRequest(Gemini, Responses, []byte(`{"system_instruction":{"parts":[{"text":"sys only"}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	input := m["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input = %#v", input)
	}
	item := input[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "system" {
		t.Fatalf("item = %#v", item)
	}
	if item["content"].([]any)[0].(map[string]any)["text"] != "sys only" {
		t.Fatalf("content = %#v", item["content"])
	}
	if _, has := m["instructions"]; has {
		t.Fatalf("instructions should have moved into the item: %#v", m["instructions"])
	}
}

// Bare thought signatures and generation-config keys with no Responses
// spelling (top_k, candidate_count, stop sequences, penalties, seed,
// logprobs, modalities, speech config) are dropped, not invented.
func TestGeminiToResponsesReverseParityDrops(t *testing.T) {
	out, err := ConvertRequest(Gemini, Responses, []byte(`{"contents":[
		{"role":"model","parts":[{"text":"deep","thoughtSignature":"SIG"},{"text":"answer"}]}],
		"generationConfig":{"thinkingConfig":{"thinkingBudget":0,"includeThoughts":false},
			"top_k":3,"candidate_count":2,"stop_sequences":["END"],"presence_penalty":0.1,
			"frequency_penalty":0.2,"seed":7,"responseLogprobs":true,"logprobs":3,
			"responseModalities":["TEXT"],"speechConfig":{"voice":{}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if fmtJSON(m["reasoning"]) != `{"effort":"none"}` {
		t.Fatalf("reasoning = %s", fmtJSON(m["reasoning"]))
	}
	for _, key := range []string{"top_k", "candidate_count", "stop_sequences", "presence_penalty",
		"frequency_penalty", "seed", "logprobs", "responseLogprobs", "responseModalities", "speechConfig"} {
		if _, has := m[key]; has {
			t.Fatalf("%s has no Responses spelling and must drop: %#v", key, m[key])
		}
	}
	// The bare signature is gone (documented: no Responses landing), the texts
	// survive as the message content.
	content := m["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[0].(map[string]any)["text"] != "deep" || content[1].(map[string]any)["text"] != "answer" {
		t.Fatalf("content = %#v", content)
	}
	if strings.Contains(string(out.Body), "SIG") {
		t.Fatalf("bare thoughtSignature must not survive: %s", out.Body)
	}
}

// A multi-key arguments object rides the source string's own key order into
// the target's args — executed against the reference's two-hop
// (responses->chat->gemini) during this audit: args {"zeta": 1, "alpha": 2,
// "mid": {"y": true, "x": null}} kept the stated order; the body marshal
// compacts the bytes, the ORDER is what a decoded map could not give back.
func TestGeminiResponsesParityMultiKeyArgsOrder(t *testing.T) {
	body := `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"zeta\": 1, \"alpha\": 2, \"mid\": {\"y\": true, \"x\": null}}"}]}`
	out, err := ConvertRequest(Responses, Gemini, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	const want = `"args":{"zeta":1,"alpha":2,"mid":{"y":true,"x":null}}`
	if !strings.Contains(string(out.Body), want) {
		t.Fatalf("multi-key args order not preserved:\n%s", out.Body)
	}
}
