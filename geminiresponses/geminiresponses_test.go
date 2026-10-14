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
	 {"role":"model","parts":[{"function_call":{"name":"get_weather","args":{"city":"Boston"}}}]},
	 {"role":"user","parts":[{"function_response":{"name":"get_weather","response":{"content":"sunny"}}}]}]}`)

	// context_management is the one source parameter the wire cannot carry.
	if _, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":"x","context_management":[{"type":"compaction"}]}`), false); err == nil {
		t.Error("context_management should be refused without dropParams")
	}
	if _, err := ResponsesToGemini(mustBody(t, `{"model":"m","input":"x","context_management":[{"type":"compaction"}]}`), true); err != nil {
		t.Errorf("context_management with dropParams: %v", err)
	}
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
