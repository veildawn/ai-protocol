package geminiresponses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// mustBody decodes a JSON object for the tests below.
func mustBody(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return m
}

// wantJSON compares a fold output against an expected body, ignoring object
// key order (which JSON does not define).
func wantJSON(t *testing.T, name string, got map[string]any, want string) {
	t.Helper()
	var w map[string]any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("decode want: %v", err)
	}
	g, _ := jsonx.Marshal(got)
	if string(g) != marshalSorted(w) {
		t.Errorf("%s:\n got %s\nwant %s", name, g, marshalSorted(w))
	}
}

func marshalSorted(v any) string {
	var b strings.Builder
	writeSorted(&b, v)
	return b.String()
}

func writeSorted(b *strings.Builder, v any) {
	switch t := v.(type) {
	case map[string]any:
		b.WriteByte('{')
		first := true
		for _, k := range sortedKeys(t) {
			if !first {
				b.WriteByte(',')
			}
			first = false
			kb, _ := json.Marshal(k)
			b.Write(kb)
			b.WriteByte(':')
			writeSorted(b, t[k])
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
			}
			writeSorted(b, e)
		}
		b.WriteByte(']')
	default:
		eb, _ := json.Marshal(v)
		b.Write(eb)
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// TestRequestForwardPinnedCases pins the fold against two-hop results run on
// the reference: the spec's §2.3 sample and a tool round. The mediaResolution
// key the reference derives from an image `detail` is deliberately absent —
// this codec carries no model knowledge, so it never writes that key.
func TestRequestForwardPinnedCases(t *testing.T) {
	out, err := ResponsesToGemini(mustBody(t, `{"model":"gemini-2.5-pro","instructions":"be terse",
	 "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
	 "max_output_tokens":20,"reasoning":{"effort":"low"},
	 "text":{"format":{"type":"json_schema","name":"o","strict":true,"schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
	 "tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"tool_choice":"auto"}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	wantJSON(t, "P3", out, `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
	 "system_instruction":{"parts":[{"text":"be terse"}]},
	 "tools":[{"function_declarations":[{"name":"f","description":"","parameters":{"type":"object"}}]}],
	 "toolConfig":{"functionCallingConfig":{"mode":"AUTO"}},
	 "generationConfig":{"max_output_tokens":20,"response_mime_type":"application/json",
	   "response_json_schema":{"type":"object","properties":{"a":{"type":"string"}}},
	   "thinkingConfig":{"thinkingBudget":1024,"includeThoughts":true}}}`)

	out, err = ResponsesToGemini(mustBody(t, `{"model":"m","input":[
	 {"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
	 {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\": \"Boston\"}"},
	 {"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	wantJSON(t, "tool round", out, `{"contents":[
	 {"role":"user","parts":[{"text":"weather?"}]},
	 {"role":"model","parts":[{"function_call":{"name":"get_weather","args":{"city":"Boston"}},"thoughtSignature":"skip_thought_signature_validator"}]},
	 {"role":"user","parts":[{"function_response":{"name":"get_weather","response":{"content":"sunny"}}}]}]}`)

	// context_management is the one source parameter the wire cannot carry.
	if _, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":"x","context_management":[{"type":"compaction"}]}`), false); err == nil {
		t.Error("context_management should be refused without dropParams")
	}
	if _, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":"x","context_management":[{"type":"compaction"}]}`), true); err != nil {
		t.Errorf("context_management with dropParams: %v", err)
	}
}

// A replayed function_call the client preserved no signature for must still
// carry a signature onto the target: the upstream rejects a replayed call
// WITHOUT one outright (INVALID_ARGUMENT, "Function call is missing a
// thought_signature in functionCall parts"), and it validates the value too, so
// the only honest spelling is the upstream's own skip marker. Observed
// 2026-10-15: a /v1/responses tool loop whose first turn streamed a
// default_api:bash call, the client replayed it with no signature, and the
// upstream 400'd on the second request.
func TestResponsesToGeminiUnsignedCallTakesSkipMarker(t *testing.T) {
	out, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":[
	 {"type":"message","role":"user","content":[{"type":"input_text","text":"look up"}]},
	 {"type":"function_call","call_id":"call_1","name":"default_api:bash","arguments":"{\"command\": \"ls\"}"},
	 {"type":"function_call_output","call_id":"call_1","output":"ok"}]}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	wantJSON(t, "unsigned call", out, `{"contents":[
	 {"role":"user","parts":[{"text":"look up"}]},
	 {"role":"model","parts":[{"function_call":{"name":"default_api:bash","args":{"command":"ls"}},"thoughtSignature":"skip_thought_signature_validator"}]},
	 {"role":"user","parts":[{"function_response":{"name":"default_api:bash","response":{"content":"ok"}}}]}]}`)
}

// The signature the client DID preserve rides verbatim: the skip marker is the
// fallback, not the default. A Responses upstream mints it as the call id's
// suffix, which is where this fold reads it back from.
func TestResponsesToGeminiSignedCallRidesVerbatim(t *testing.T) {
	out, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":[
	 {"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},
	 {"type":"function_call","call_id":"call_1__thought__SIG","name":"f","arguments":"{\"a\": 1}"},
	 {"type":"function_call_output","call_id":"call_1__thought__SIG","output":"ok"}]}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	wantJSON(t, "signed call", out, `{"contents":[
	 {"role":"user","parts":[{"text":"q"}]},
	 {"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}},"thoughtSignature":"SIG"}]},
	 {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"ok"}}}]}]}`)
}

// Google's own wire spells the member BOTH ways — the plugin's reader says
// "thoughtSignature comes and goes" — and it keeps both keys in the map, so a
// reader that consults one spelling alone silently drops the signature half
// the time. A dropped signature is not cosmetic: it rides the call_id the
// client replays, and the upstream refuses an unsigned replayed call.
func TestResponsesSignatureReadsEitherSpelling(t *testing.T) {
	for _, tc := range []struct{ name, member string }{
		{"camel", `"thoughtSignature":"SIG"`},
		{"snake", `"thought_signature":"SIG"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := GeminiResponseToResponses(mustBody(t,
				`{"candidates":[{"content":{"role":"model","parts":[`+
					`{"text":"let me think","thought":true,`+tc.member+`},`+
					`{"functionCall":{"name":"lookup","args":{"q":"x"}},`+tc.member+`}]},`+
					`"finishReason":"STOP","index":0}],"responseId":"resp-1"}`))
			got := marshalSorted(out)
			// The thinking block keeps its signature. The block rides the
			// reasoning item's encrypted_content, a JSON string, so the wire
			// carries it escaped.
			if !strings.Contains(got, `signature`) || !strings.Contains(got, `SIG`) {
				t.Errorf("the thinking block lost its signature: %s", got)
			}
			// The call rides it as the id suffix, so a replay reaches the
			// upstream signed.
			if !strings.Contains(got, "__thought__SIG") {
				t.Errorf("the call did not carry the signature: %s", got)
			}
		})
	}
}

// The skip-marker fallback also reads the signature from the item's own
// provider fields (the reverse fold writes it there as well as the call id),
// so a client that relayed only the field — not the id suffix — still
// replays a signed call.
func TestResponsesToGeminiReadsProviderFieldSignature(t *testing.T) {
	out, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":[
	 {"type":"message","role":"user","content":[{"type":"input_text","text":"q"}]},
	 {"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"a\": 1}",
	  "provider_specific_fields":{"thought_signature":"PSIG"}},
	 {"type":"function_call_output","call_id":"call_1","output":"ok"}]}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	wantJSON(t, "provider field signature", out, `{"contents":[
	 {"role":"user","parts":[{"text":"q"}]},
	 {"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}},"thoughtSignature":"PSIG"}]},
	 {"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"ok"}}}]}]}`)
}

// A tool result is promoted to inline image bytes ONLY when the whole string
// is one image data URL: the scheme must sit at the string's start and the
// payload must decode as base64. The unanchored search this fold used to run
// promoted any text that merely MENTIONED a data URL — the tail of the text
// was sliced out as the payload — and a payload that is not base64 is refused
// by the upstream with a 400 (INVALID_ARGUMENT, "Base64 decoding failed"),
// killing the whole turn.
//
// The first two payloads below are verbatim from the live 400s of 2026-10-16
// (admin/debug-logs, antigravity arm): a Read of a manifest whose icon field
// holds a data URL, and a Read of Go test source containing one as a literal.
// In each the text must stay text, whatever it mentions.
func TestToolResultMentioningDataURLStaysText(t *testing.T) {
	for _, tc := range []struct{ name, output string }{
		{"manifest with icon data URL", "1\t{\n2\t  \"icon\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==\"\n3\t}"},
		{"source line with data URL literal", "x\"}}}),\n219\t\t})\n220\t"},
		{"leading text then data URL", "see data:image/png;base64,AAAB for details"},
		{"data URL scheme without base64 marker", "data:image/png,QUJD"},
		{"junk payload after the scheme", "data:image/png;base64,!!!"},
		{"payload with trailing prose", "data:image/png;base64,AAAB\"}\nmore"},
		{"empty payload", "data:image/png;base64,"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ResponsesToGemini(map[string]any{"model": "m", "input": []any{
				map[string]any{"type": "function_call", "call_id": "c1", "name": "Read", "arguments": "{}"},
				map[string]any{"type": "function_call_output", "call_id": "c1", "output": tc.output},
			}}, false)
			if err != nil {
				t.Fatalf("fold: %v", err)
			}
			fr := toolResultFunctionResponse(t, out)
			if _, hasParts := fr["parts"]; hasParts {
				t.Fatalf("text mentioning a data URL must not be promoted to media: %#v", fr)
			}
			response, _ := fr["response"].(map[string]any)
			if response["content"] != tc.output {
				t.Fatalf("text result content = %#v, want it verbatim", response["content"])
			}
		})
	}
}

// The promotion still happens for the shape it is for: a string that IS an
// image data URL, scheme first and a real base64 payload behind the marker.
// The unpadded payload is deliberate — clients send those, and a gate that
// only accepted padded base64 would demote real media to text.
func TestToolResultBareImageDataURLIsPromoted(t *testing.T) {
	out, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":[
	 {"type":"function_call","call_id":"c1","name":"Read","arguments":"{}"},
	 {"type":"function_call_output","call_id":"c1","output":"data:image/png;base64,iVBORw0KGgo="}]}`), false)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	fr := toolResultFunctionResponse(t, out)
	if marshalSorted(fr["parts"]) != `[{"inline_data":{"data":"iVBORw0KGgo=","mime_type":"image/png"}}]` {
		t.Fatalf("bare image data URL parts = %s", marshalSorted(fr["parts"]))
	}
	if marshalSorted(fr["response"]) != `{"content":""}` {
		t.Fatalf("bare image data URL response = %s", marshalSorted(fr["response"]))
	}
}

// toolResultFunctionResponse digs the one function_response out of a folded
// body produced by the tool-result cases above.
func toolResultFunctionResponse(t *testing.T, out map[string]any) map[string]any {
	t.Helper()
	contents, _ := out["contents"].([]any)
	if len(contents) != 2 {
		t.Fatalf("contents = %#v, want the call turn and the result turn", out["contents"])
	}
	resultTurn, _ := contents[1].(map[string]any)
	parts, _ := resultTurn["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("result turn parts = %#v, want one function_response", resultTurn["parts"])
	}
	part, _ := parts[0].(map[string]any)
	fr, _ := part["function_response"].(map[string]any)
	if fr == nil {
		t.Fatalf("part = %#v, want a function_response", part)
	}
	return fr
}

// TestResponseForwardPinnedCases pins the response fold against the
// reference's own two-hop output, with the ids this contract mints in place
// of the reference's uuids.
func TestResponseForwardPinnedCases(t *testing.T) {
	out := GeminiResponseToResponses(mustBody(t, `{"candidates":[{"content":{"role":"model","parts":[
	 {"text":"let me think","thought":true},{"text":"hello"},{"functionCall":{"name":"lookup","args":{"q":"x"}}}]},
	 "finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"resp-1"}`))
	wantJSON(t, "thinking and tools", out, `{"id":"resp-1","created_at":0,"model":"","object":"response",
	 "status":"completed","parallel_tool_calls":false,"text":{},"tool_choice":"auto","tools":[],
	 "output":[
	  {"type":"reasoning","id":"rs_0","status":"completed","role":"assistant","content":[{"type":"output_text","text":"let me think","annotations":[]}]},
	  {"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]},
	  {"type":"function_call","id":"call_2","call_id":"call_2","name":"lookup","arguments":"{\"q\": \"x\"}","status":"completed"}],
	 "usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8,
	   "input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0,"text_tokens":3}}}`)

	// A blocked prompt folds into one incomplete message item with null text.
	out = GeminiResponseToResponses(mustBody(t, `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`))
	wantJSON(t, "blocked", out, `{"id":"","created_at":0,"model":"","object":"response","status":"incomplete",
	 "parallel_tool_calls":false,"text":{},"tool_choice":"auto","tools":[],
	 "output":[{"type":"message","id":"msg_0","status":"incomplete","role":"assistant","content":[{"type":"output_text","text":null,"annotations":[]}]}],
	 "usage":{"input_tokens":5,"output_tokens":0,"total_tokens":5}}`)
}

// TestStreamForwardSequence pins the gemini->responses stream fold's event
// order: every item announces itself when it opens — the reasoning item off
// its first thought frame (delta included), the message item off the text
// frame that follows, the tool after them — and the turn closes with the done
// trio plus response.completed.
func TestStreamForwardSequence(t *testing.T) {
	frames := []string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
	}
	var src strings.Builder
	for _, f := range frames {
		src.WriteString("data: " + f + "\n\n")
	}
	var out strings.Builder
	if _, err := PipeGeminiToResponses(&out, strings.NewReader(src.String()), StreamOpts{Model: "gemini-2.5-pro"}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	var names []string
	for _, line := range strings.Split(out.String(), "\n") {
		if ev, ok := strings.CutPrefix(line, "event: "); ok {
			names = append(names, ev)
		}
	}
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
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Errorf("event order:\n got %v\nwant %v", names, want)
	}
	if !strings.Contains(out.String(), `"part":{"reasoning":"think","type":"reasoning_text"}`) &&
		!strings.Contains(out.String(), `"part":{"type":"reasoning_text","reasoning":"think"}`) {
		t.Error("the closing content_part.done should carry the full reasoning text as a reasoning_text part")
	}
}

// TestStreamTruncationAndSentinel pins the terminal ruling: an EOF that
// arrived with no finishReason closes nothing, and a relay sentinel closes
// the turn — items and terminal both. Withholding response.completed on a
// sentinel would leave the host booking a delivered turn (StreamResult says
// Terminal for a [DONE]) while the Responses client's stream never closes;
// the Responses dialect has exactly one legal ending.
func TestStreamTruncationAndSentinel(t *testing.T) {
	frame := `{"candidates":[{"content":{"role":"model","parts":[{"text":"par"}]},"index":0}],"responseId":"r1"}`
	var out strings.Builder
	if _, err := PipeGeminiToResponses(&out, strings.NewReader("data: "+frame+"\n\n"), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if s := out.String(); strings.Contains(s, "response.completed") || strings.Contains(s, "response.output_item.done") {
		t.Errorf("a truncated source must not be closed by the fold:\n%s", s)
	}
	out.Reset()
	if _, err := PipeGeminiToResponses(&out, strings.NewReader("data: "+frame+"\n\ndata: [DONE]\n\n"), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	s := out.String()
	if !strings.Contains(s, "response.output_item.done") {
		t.Errorf("a sentinel close should close the open items:\n%s", s)
	}
	if !strings.Contains(s, "response.completed") {
		t.Errorf("a sentinel close must write the dialect's one legal ending:\n%s", s)
	}
	if !strings.Contains(s, `"status":"completed"`) {
		t.Errorf("a sentinel close states a plain completion — the source never said it failed:\n%s", s)
	}
}

// TestStreamReverseFold pins the responses->gemini stream fold: text and
// reasoning deltas become frames, and a call's arguments are only sent once
// they parse as a whole JSON object.
func TestStreamReverseFold(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"r1"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_0","type":"message"}}`,
		`{"type":"response.output_text.delta","item_id":"msg_0","output_index":0,"content_index":0,"delta":"hi"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"call_0","type":"function_call","call_id":"call_0","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"call_0","output_index":1,"delta":"{\"a\""}`,
		`{"type":"response.function_call_arguments.delta","item_id":"call_0","output_index":1,"delta":": 1}"}`,
		`{"type":"response.completed","response":{"id":"r1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
	}
	var src strings.Builder
	for _, e := range events {
		src.WriteString("event: " + jsonEventType(e) + "\ndata: " + e + "\n\n")
	}
	var out strings.Builder
	if _, err := PipeResponsesToGemini(&out, strings.NewReader(src.String()), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	s := out.String()
	if !strings.Contains(s, `{"text":"hi"}`) {
		t.Errorf("text delta frame missing:\n%s", s)
	}
	if c := strings.Count(s, `"functionCall"`); c != 1 {
		t.Errorf("exactly one functionCall frame should be sent (got %d):\n%s", c, s)
	}
	if !strings.Contains(s, `"args":{"a":1}`) {
		t.Errorf("the whole arguments object should arrive:\n%s", s)
	}
	if !strings.Contains(s, `"finishReason":"STOP"`) || !strings.Contains(s, `"promptTokenCount":5`) {
		t.Errorf("the terminal frame should carry the finish reason and the usage:\n%s", s)
	}
	if strings.Contains(s, "[DONE]") {
		t.Errorf("this wire has no sentinel:\n%s", s)
	}
}

func jsonEventType(data string) string {
	var m map[string]any
	json.Unmarshal([]byte(data), &m)
	if t, ok := m["type"].(string); ok {
		return t
	}
	return ""
}
