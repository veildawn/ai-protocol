package geminimessages

// Minimal checks against the golden bodies the spec quotes from real runs of
// the reference's composed fold (spec §2.2, §2.5). They are not the parity
// suite — the integration stage owns that — only a guard that the direct fold
// produces the documented shapes.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/internal/thinksig"
	"github.com/veildawn/ai-protocol/stream"
)

func decode(t *testing.T, s string) map[string]any {
	t.Helper()
	// Decode twice so number literals (1) and floats (1.0) compare the same
	// whichever side wrote them.
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("decode %s: %v", s, err)
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func decodeAny(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("decode %s: %v", s, err)
	}
	return normalize(v)
}

func normalize(v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return v
	}
	return out
}

func assertEqual(t *testing.T, got, want any, what string) {
	t.Helper()
	g, w := normalize(got), normalize(want)
	if !reflect.DeepEqual(g, w) {
		gb, _ := json.Marshal(g)
		wb, _ := json.Marshal(w)
		t.Fatalf("%s mismatch:\n got  %s\n want %s", what, gb, wb)
	}
}

// sseEvents decodes an SSE stream into its (event name, data) pairs.
func sseEvents(t *testing.T, s string) []stream.Event {
	t.Helper()
	var evs []stream.Event
	sc := bufio.NewScanner(strings.NewReader(s))
	var name string
	var data []string
	flush := func() {
		if len(data) == 0 {
			return
		}
		evs = append(evs, stream.Event{Event: name, Data: strings.Join(data, "\n")})
		name, data = "", nil
	}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event: "):
			name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			data = append(data, strings.TrimPrefix(line, "data: "))
		}
	}
	flush()
	return evs
}

func TestMessagesToGeminiSpecSample(t *testing.T) {
	// The spec's §2.2 sample: a tool round trip, a system prompt, tools, an
	// auto tool choice, thinking and stop sequences.
	src := decode(t, `{
	  "model":"gemini-2.5-pro","max_tokens":32,"system":"be brief",
	  "messages":[{"role":"user","content":"hello"},
	    {"role":"assistant","content":[{"type":"text","text":"hi"},
	      {"type":"tool_use","id":"tu_1","name":"lookup","input":{"q":"x"}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"res"}]}],
	  "tools":[{"name":"lookup","description":"d",
	    "input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}],
	  "tool_choice":{"type":"auto"},"thinking":{"type":"enabled","budget_tokens":2048},
	  "stop_sequences":["END"]}`)
	got, err := MessagesToGemini(src, false)
	if err != nil {
		t.Fatalf("MessagesToGemini: %v", err)
	}
	want := decode(t, `{
	  "contents":[
	    {"role":"user","parts":[{"text":"hello"}]},
	    {"role":"model","parts":[{"text":"hi"},{"function_call":{"name":"lookup","args":{"q":"x"}},"thoughtSignature":"skip_thought_signature_validator"}]},
	    {"role":"user","parts":[{"function_response":{"name":"lookup","response":{"content":"res"}}}]}],
	  "system_instruction":{"parts":[{"text":"be brief"}]},
	  "tools":[{"function_declarations":[{"name":"lookup","description":"d",
	     "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]}],
	  "toolConfig":{"functionCallingConfig":{"mode":"AUTO"}},
	  "generationConfig":{"thinkingConfig":{"thinkingBudget":2048,"includeThoughts":true},
	    "stop_sequences":["END"],"max_output_tokens":32}}`)
	assertEqual(t, got, want, "fold")
}

// A tool result that prints a JSON Schema keeps its text wrapped even though
// it parses as one object: `$ref` inside a function response is the gemini
// wire's reserved reference to a media part's display_name, and handing the
// schema over as structure made the upstream refuse the whole request (400,
// INVALID_ARGUMENT, "The referenced name ... does not match to a display_name
// in the function_response.parts"; live 2026-10-16). As text the schema
// reaches the model verbatim; an object with no `$ref` still rides as
// structure, exactly as before.
func TestMessagesToGeminiToolResultJSONSchemaRefStaysText(t *testing.T) {
	const schema = `{"type": "object", "properties": {"providers": {"items": {"$ref": "#/$defs/provider"}}}}`
	src := map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "schema?"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": map[string]any{}},
		}},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": "tu_1", "content": schema},
		}},
	}}
	got, err := MessagesToGemini(src, false)
	if err != nil {
		t.Fatalf("MessagesToGemini: %v", err)
	}
	contents, _ := got["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %#v, want the call turn and the result turn", got["contents"])
	}
	resultTurn, _ := contents[2].(map[string]any)
	parts, _ := resultTurn["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("result turn parts = %#v, want one function_response", resultTurn["parts"])
	}
	part, _ := parts[0].(map[string]any)
	fr, _ := part["function_response"].(map[string]any)
	response, _ := fr["response"].(map[string]any)
	if len(response) != 1 || response["content"] != schema {
		t.Fatalf("schema result response = %#v, want it wrapped verbatim under content", fr["response"])
	}

	// Control: no `$ref`, the parsed object rides as structure.
	plain := decode(t, `{"model":"m","messages":[
	    {"role":"user","content":"go"},
	    {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"f","input":{"a":1}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"{\"ok\": true, \"n\": 2}"}]}]}`)
	out, err := MessagesToGemini(plain, false)
	if err != nil {
		t.Fatalf("MessagesToGemini control: %v", err)
	}
	assertEqual(t, mustJSON(geminiResponseAt(out, 2)), `{"name":"f","response":{"n":2,"ok":true}}`, "plain object result")
}

// geminiResponseAt digs the function_response turn's single part, sorted for
// comparison, from the folded contents at index i.
func geminiResponseAt(out map[string]any, i int) any {
	contents, _ := out["contents"].([]any)
	turn, _ := contents[i].(map[string]any)
	parts, _ := turn["parts"].([]any)
	part, _ := parts[0].(map[string]any)
	return part["function_response"]
}

func TestMessagesToGeminiRejectsContextManagement(t *testing.T) {
	src := decode(t, `{"model":"m","max_tokens":8,"context_management":{"edits":[]},
		"messages":[{"role":"user","content":"hi"}]}`)
	if _, err := MessagesToGemini(src, false); err == nil {
		t.Fatal("context_management was not rejected with dropParams=false")
	}
	if _, err := MessagesToGemini(src, true); err != nil {
		t.Fatalf("context_management with dropParams=true: %v", err)
	}
}

func TestMessagesToGeminiThinkingLadder(t *testing.T) {
	// The two-level fold the spec pins: budget 5000 lands on 4096 and a
	// disabled request turns the thoughts off entirely.
	for _, tc := range []struct {
		thinking string
		want     string
	}{
		{`{"type":"enabled","budget_tokens":5000}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":128}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"enabled","budget_tokens":0}`, `{"thinkingBudget":128,"includeThoughts":true}`},
		{`{"type":"adaptive"}`, `{"thinkingBudget":2048,"includeThoughts":true}`},
		{`{"type":"adaptive"},"output_config":{"effort":"high"}`, `{"thinkingBudget":4096,"includeThoughts":true}`},
		{`{"type":"disabled"}`, `{"thinkingBudget":0,"includeThoughts":false}`},
		{`{"type":"weird"}`, ``},
	} {
		src := decode(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],"thinking":`+tc.thinking+`}`)
		got, err := MessagesToGemini(src, false)
		if err != nil {
			t.Fatalf("%s: %v", tc.thinking, err)
		}
		gc, _ := got["generationConfig"].(map[string]any)
		var cfg map[string]any
		if gc != nil {
			cfg, _ = gc["thinkingConfig"].(map[string]any)
		}
		if tc.want == "" {
			if cfg != nil {
				t.Fatalf("%s: thinkingConfig = %v, want none", tc.thinking, cfg)
			}
			continue
		}
		assertEqual(t, cfg, decode(t, tc.want), tc.thinking)
	}
}

func TestMessagesToGeminiEmptyAndMediaTurns(t *testing.T) {
	// An image-only user turn gains the blank text the target requires, and a
	// user turn with no foldable content leaves the contents fallback.
	img := decode(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
	  {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAB"}}]}]}`)
	got, err := MessagesToGemini(img, false)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, got, decode(t, `{"contents":[{"role":"user","parts":[
	   {"inline_data":{"data":"AAAB","mime_type":"image/png"}},{"text":" "}]}],
	   "generationConfig":{"max_output_tokens":8}}`), "image turn")

	empty := decode(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":""}]}`)
	got, err = MessagesToGemini(empty, false)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, got, decode(t, `{"contents":[{"role":"user","parts":[{"text":" "}]}],
	   "generationConfig":{"max_output_tokens":8}}`), "empty turn")

	sysOnly := decode(t, `{"model":"m","max_tokens":8,"system":"sys","messages":[]}`)
	got, err = MessagesToGemini(sysOnly, false)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, got, decode(t, `{"contents":[{"role":"user","parts":[{"text":"."}]}],
	   "system_instruction":{"parts":[{"text":"sys"}]},
	   "generationConfig":{"max_output_tokens":8}}`), "system only")
}

func TestGeminiResponseToMessagesRich(t *testing.T) {
	// The spec's §2.5 sample shape: thinking with a signature first, then the
	// text, then the tool call whose signature moves off the id.
	src := decode(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"think","thought":true,"thoughtSignature":"SIG1"},
	    {"text":"hello"},
	    {"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"SIG2"}]},
	  "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,
	   "cachedContentTokenCount":2,"thoughtsTokenCount":2},
	 "responseId":"resp-9"}`)
	got := GeminiResponseToMessages(src)
	want := decode(t, `{"id":"resp-9","type":"message","role":"assistant","model":"unknown-model",
	  "stop_sequence":null,
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2},
	  "content":[{"type":"thinking","thinking":"think","signature":"SIG1"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"call_0","name":"f","input":{"a":1},
	              "provider_specific_fields":{"signature":"SIG2"}}],
	  "stop_reason":"tool_use"}`)
	assertEqual(t, got, want, "fold")
}

func TestGeminiResponseToMessagesBlocked(t *testing.T) {
	src := decode(t, `{"promptFeedback":{"blockReason":"SAFETY"},
	  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`)
	got := GeminiResponseToMessages(src)
	if got["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", got["stop_reason"])
	}
	assertEqual(t, got["usage"], decode(t, `{"input_tokens":5,"output_tokens":0}`), "usage")
	if content, ok := got["content"].([]any); !ok || len(content) != 0 {
		t.Fatalf("content = %#v, want empty", got["content"])
	}
}

func TestGeminiResponseToMessagesMediaDestinations(t *testing.T) {
	// Images drop, non-image non-audio inline data rides the text as a data
	// URL, and audio drops — the three homes media have on this fold.
	src := decode(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"see"},{"inlineData":{"mimeType":"image/png","data":"AAAB"}},
	    {"inlineData":{"mimeType":"application/pdf","data":"JVBE"}},
	    {"inlineData":{"mimeType":"audio/wav","data":"AAAA"}}]},"finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	got := GeminiResponseToMessages(src)
	content, ok := got["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("content = %#v, want one text block", got["content"])
	}
	assertEqual(t, content[0], decode(t, `{"type":"text","text":"seedata:application/pdf;base64,JVBE"}`), "text block")
}

func TestMessagesResponseToGeminiRoundTrip(t *testing.T) {
	// A messages response folds to the target and back to the same usage the
	// composed fold would report.
	src := decode(t, `{"id":"m1","type":"message","role":"assistant","model":"any",
	  "content":[{"type":"thinking","thinking":"t","signature":"S"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"tu_1","name":"f","input":{"a":1}}],
	  "stop_reason":"tool_use",
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2}}`)
	gemini := MessagesResponseToGemini(src)
	assertEqual(t, gemini["responseId"], "m1", "responseId")
	gc, _ := gemini["candidates"].([]any)
	cand, _ := gc[0].(map[string]any)
	assertEqual(t, cand["finishReason"], "STOP", "finishReason")
	assertEqual(t, gemini["usageMetadata"], decode(t, `{
	  "promptTokenCount":10,"candidatesTokenCount":6,"totalTokenCount":16,
	  "cachedContentTokenCount":2,
	  "promptTokensDetails":[{"modality":"TEXT","tokenCount":8}],
	  "candidatesTokensDetails":[{"modality":"TEXT","tokenCount":6}]}`), "usageMetadata")
	back := GeminiResponseToMessages(gemini)
	assertEqual(t, back["usage"], decode(t, `{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2}`), "usage round trip")
	assertEqual(t, back["stop_reason"], "tool_use", "stop_reason round trip")
}

func TestPipeGeminiToMessagesSequence(t *testing.T) {
	// The spec's §2.8 sequence: thinking, text and tool blocks with dense
	// indexes, then exactly one message_delta with the real usage.
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]},"index":0}],"responseId":"r1"}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
	}, "\n\n") + "\n\n"
	var dst bytes.Buffer
	if _, err := PipeGeminiToMessages(&dst, strings.NewReader(src), StreamOpts{Model: "gemini-2.5-pro"}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	evs := sseEvents(t, dst.String())
	var types []string
	for _, ev := range evs {
		m := decode(t, ev.Data)
		types = append(types, jsonx.GetString(m, "type"))
	}
	assertEqual(t, types, []any{
		"message_start", "content_block_start", "content_block_delta", "content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}, "event types")

	assertEqual(t, decode(t, evs[1].Data), decode(t, `{"type":"content_block_start","index":0,
		"content_block":{"type":"thinking","thinking":"","signature":"dW5zaWduZWQtdGhpbmtpbmc="}}`), "thinking start")
	assertEqual(t, decode(t, evs[2].Data), decode(t, `{"type":"content_block_delta","index":0,
		"delta":{"type":"thinking_delta","thinking":"think"}}`), "thinking delta")
	assertEqual(t, decode(t, evs[4].Data), decode(t, `{"type":"content_block_start","index":1,
		"content_block":{"type":"text","text":""}}`), "text start")
	assertEqual(t, decode(t, evs[5].Data), decode(t, `{"type":"content_block_delta","index":1,
		"delta":{"type":"text_delta","text":"hi"}}`), "text delta")
	assertEqual(t, decode(t, evs[7].Data), decode(t, `{"type":"content_block_start","index":2,
		"content_block":{"type":"tool_use","id":"call_0","name":"f","input":{}}}`), "tool start")
	assertEqual(t, decode(t, evs[8].Data), decode(t, `{"type":"content_block_delta","index":2,
		"delta":{"type":"input_json_delta","partial_json":"{\"a\": 1}"}}`), "tool delta")
	assertEqual(t, decode(t, evs[10].Data), decode(t, `{"type":"message_delta",
		"delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":5,"output_tokens":3}}`), "message_delta")
	assertEqual(t, decode(t, evs[0].Data), decode(t, `{"type":"message_start","message":{
		"id":"r1","type":"message","role":"assistant","model":"gemini-2.5-pro","content":[],
		"stop_reason":null,"stop_sequence":null,
		"usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`), "message_start")
}

func TestPipeGeminiToMessagesTruncationIsNotACompletion(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]},\"index\":0}]}\n\n"
	var dst bytes.Buffer
	if _, err := PipeGeminiToMessages(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	out := dst.String()
	if strings.Contains(out, "message_delta") || strings.Contains(out, "message_stop") {
		t.Fatalf("truncated source was closed for the client:\n%s", out)
	}
	if !strings.Contains(out, `"text":"hi"`) {
		t.Fatalf("delivered content lost:\n%s", out)
	}
}

func TestPipeMessagesToGeminiEmitsTerminalOnlyOnSourceClose(t *testing.T) {
	src := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3}}}`,
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}`,
	}, "\n\n") + "\n\n"
	var dst bytes.Buffer
	if _, err := PipeMessagesToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	out := dst.String()
	if strings.Contains(out, "finishReason") {
		t.Fatalf("terminal written before the source closed:\n%s", out)
	}
	if !strings.Contains(out, `"text":"hi"`) {
		t.Fatalf("text frame lost:\n%s", out)
	}
	// Closing with message_stop writes the terminal frame with the usage the
	// stream accumulated.
	src += "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	dst.Reset()
	if _, err := PipeMessagesToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	frames := sseEvents(t, dst.String())
	last := decode(t, frames[len(frames)-1].Data)
	assertEqual(t, last, decode(t, `{"candidates":[{"content":{"role":"model","parts":[]},
		"finishReason":"STOP","index":0}],
		"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7,
		 "promptTokensDetails":[{"modality":"TEXT","tokenCount":3}],
		 "candidatesTokensDetails":[{"modality":"TEXT","tokenCount":4}]}}`), "terminal frame")
}

func TestPipeMessagesToGeminiToolArgumentsWhole(t *testing.T) {
	// Argument fragments that never become a JSON object emit no call; a
	// complete object emits exactly one, on the last parseable snapshot.
	src := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":1}}}`,
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}`,
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}`,
		"event: message_stop",
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
	var dst bytes.Buffer
	if _, err := PipeMessagesToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	frames := sseEvents(t, dst.String())
	var callFrame map[string]any
	calls := 0
	for _, ev := range frames {
		m := decode(t, ev.Data)
		cands, _ := m["candidates"].([]any)
		for _, craw := range cands {
			c, _ := craw.(map[string]any)
			content, _ := c["content"].(map[string]any)
			parts, _ := content["parts"].([]any)
			for _, praw := range parts {
				p, _ := praw.(map[string]any)
				if _, ok := p["function_call"]; ok {
					calls++
					callFrame = p
				}
			}
		}
	}
	if calls != 1 {
		t.Fatalf("call emitted %d times, want 1", calls)
	}
	assertEqual(t, callFrame, decode(t, `{"function_call":{"args":{"a":1},"name":"f"}}`), "call frame")
}

// A user-turn media URL this offline codec cannot resolve is an error, not a
// silent drop: the reference would fetch it (and fail offline reaching for
// it), and a prompt that quietly lost the caller's image answers the wrong
// question with no signal anywhere. Spec §2.1 rule 23; the chat-side and
// responses-side folds of the same wire refuse the same input.
func TestMessagesToGeminiUnresolvableMediaURLIsAnError(t *testing.T) {
	body := decode(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
	  {"type":"text","text":"see"},
	  {"type":"image","source":{"type":"url","url":"https://my-proxy.example/noext"}}]}]}`)
	if _, err := MessagesToGemini(body, false); err == nil {
		t.Fatal("an unresolvable media url was silently dropped")
	}
	// The same document spelling refuses the same way.
	doc := decode(t, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[
	  {"type":"document","source":{"type":"url","url":"https://my-proxy.example/noext"}}]}]}`)
	if _, err := MessagesToGemini(doc, false); err == nil {
		t.Fatal("an unresolvable document url was silently dropped")
	}
}

// The stand-in signature this package's response folds write onto a thinking
// block they rebuilt (internal/thinksig) is not a signature the target ever
// issued. The target verifies every signature it is handed, so replaying the
// stand-in is an INVALID_ARGUMENT — the 400 a gateway reports as a rejected
// model request. The fold must send that block the way it sends one that
// carried no signature at all: a bare thought part, and the call's own skip
// marker.
func TestMessagesToGeminiFoldStandInSignature(t *testing.T) {
	fold := func(t *testing.T, sentinelSig, callSig string) map[string]any {
		t.Helper()
		body := decode(t, `{"model":"m","max_tokens":8,"messages":[
		  {"role":"user","content":"q"},
		  {"role":"assistant","content":[
		    {"type":"thinking","thinking":"why","signature":"`+sentinelSig+`"},
		    {"type":"text","text":"answer"},
		    {"type":"tool_use","id":"t1","name":"f","input":{"a":1},
		     "provider_specific_fields":{"signature":"`+callSig+`"}}]}]}`)
		got, err := MessagesToGemini(body, false)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}

	// A stand-in on the thought and on the call folds exactly like an absent
	// one, so nothing the target could refuse rides out.
	standIn := fold(t, thinksig.Synthetic, thinksig.Synthetic)
	unsigned := fold(t, "", "")
	assertEqual(t, standIn, unsigned, "stand-in folds like an unsigned block")

	// The shape itself: bare thought part, skip marker on the replayed call.
	assertEqual(t, standIn["contents"], decodeAny(t, `[
	  {"role":"user","parts":[{"text":"q"}]},
	  {"role":"model","parts":[
	    {"text":"why","thought":true},
	    {"text":"answer"},
	    {"function_call":{"args":{"a":1},"name":"f"},
	     "thoughtSignature":"skip_thought_signature_validator"}]}]`), "stand-in parts")

	if encoded, err := json.Marshal(standIn); err != nil {
		t.Fatal(err)
	} else if strings.Contains(string(encoded), thinksig.Synthetic) {
		t.Fatalf("the fold stand-in reached the target: %s", encoded)
	}
}

// A thinking block that carries nothing but the stand-in (the empty block a
// streaming start frame opens) folds to no part at all: it names no reasoning
// and its signature is not one the target would accept.
func TestMessagesToGeminiEmptyStandInBlockDropped(t *testing.T) {
	body := decode(t, `{"model":"m","max_tokens":8,"messages":[
	  {"role":"user","content":"q"},
	  {"role":"assistant","content":[
	    {"type":"thinking","thinking":"","signature":"`+thinksig.Synthetic+`"},
	    {"type":"text","text":"answer"}]}]}`)
	got, err := MessagesToGemini(body, false)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, got["contents"], decodeAny(t, `[
	  {"role":"user","parts":[{"text":"q"}]},
	  {"role":"model","parts":[{"text":"answer"}]}]`), "empty stand-in dropped")
}

// The regression the stand-in caused end to end: a Gemini response whose
// thought carries no signature folds to a Messages block signed with the
// stand-in, and a client replays that block on the next turn. The replay must
// reach the target without the stand-in.
func TestFoldStandInDoesNotSurviveReplay(t *testing.T) {
	resp := GeminiResponseToMessages(decode(t, `{"candidates":[{"content":{"role":"model","parts":[
	   {"text":"why","thought":true},{"text":"answer"}]},"finishReason":"STOP","index":0}]}`))
	blocks, _ := resp["content"].([]any)
	if len(blocks) == 0 {
		t.Fatalf("response folded to no content: %v", resp)
	}
	first, _ := blocks[0].(map[string]any)
	if first["signature"] != thinksig.Synthetic {
		t.Fatalf("the response fold no longer marks a rebuilt thought: %v", first)
	}

	body := map[string]any{
		"model":      "m",
		"max_tokens": 8,
		"messages": []any{
			map[string]any{"role": "user", "content": "q"},
			map[string]any{"role": "assistant", "content": blocks},
		},
	}
	got, err := MessagesToGemini(body, false)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), thinksig.Synthetic) {
		t.Fatalf("a replayed stand-in reached the target: %s", encoded)
	}
}

// The same rule on the streamed request fold: a signature delta that repeats
// the stand-in contributes no thoughtSignature part, while a real signature
// still rides its own part.
func TestPipeMessagesToGeminiDropsStandInSignature(t *testing.T) {
	pipe := func(t *testing.T, sig string) string {
		t.Helper()
		src := strings.Join([]string{
			"event: message_start",
			`data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":1}}}`,
			"event: content_block_start",
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
			"event: content_block_delta",
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
			"event: content_block_delta",
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"` + sig + `"}}`,
			"event: message_stop",
			`data: {"type":"message_stop"}`,
		}, "\n\n") + "\n\n"
		var dst bytes.Buffer
		if _, err := PipeMessagesToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
			t.Fatalf("pipe: %v", err)
		}
		return dst.String()
	}

	if out := pipe(t, thinksig.Synthetic); strings.Contains(out, "thoughtSignature") {
		t.Fatalf("the stand-in reached the target: %s", out)
	}
	if out := pipe(t, "SIG"); !strings.Contains(out, `"thoughtSignature":"SIG"`) {
		t.Fatalf("a real signature was dropped: %s", out)
	}
}

// Gemini 3's thinkingLevel must reach the Messages wire as a budget. This is the
// third reader the shared package replaced: all three used to consult
// thinkingBudget alone, so a level-bearing history folded to "thinking enabled,
// no budget" here and to a specific depth elsewhere.
func TestGeminiToMessagesCarriesThinkingLevel(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],` +
			`"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH","includeThoughts":true}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := GeminiToMessages(obj)
	if err != nil {
		t.Fatal(err)
	}
	thinking, _ := jsonx.AsMap(out["thinking"])
	if got := jsonx.GetString(thinking, "type"); got != "enabled" {
		t.Fatalf("thinking.type = %q, want enabled: %s", got, mustJSON(out))
	}
	budget, ok := jsonx.Int(thinking["budget_tokens"])
	if !ok || budget <= 0 {
		t.Fatalf("thinking.budget_tokens = %v, want the high budget: %s", thinking["budget_tokens"], mustJSON(out))
	}
}

// A declaration's schema may ride the standard-JSON-Schema member the official
// SDKs state — camel or snake — rather than the legacy OpenAPI subset; the
// fold reads whichever member is present onto input_schema, and a declaration
// that stated none still gets the permissive object the target requires.
func TestGeminiToMessagesReadsJsonSchemaToolSpelling(t *testing.T) {
	for _, tc := range []struct{ name, member string }{
		{"camel", "parametersJsonSchema"},
		{"snake", "parameters_json_schema"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			obj, err := jsonx.UnmarshalMap([]byte(
				`{"contents":[{"role":"user","parts":[{"text":"q"}]}],` +
					`"tools":[{"function_declarations":[{"name":"lookup","description":"d","` + tc.member + `":` +
					`{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}}]}]}`))
			if err != nil {
				t.Fatal(err)
			}
			out, err := GeminiToMessages(obj)
			if err != nil {
				t.Fatal(err)
			}
			tools, _ := jsonx.AsSlice(out["tools"])
			if len(tools) != 1 {
				t.Fatalf("tools = %s", mustJSON(out))
			}
			tool, _ := jsonx.AsMap(tools[0])
			schema, _ := jsonx.AsMap(tool["input_schema"])
			if schema == nil {
				t.Fatalf("the schema was dropped: %s", mustJSON(out))
			}
			props, _ := jsonx.AsMap(schema["properties"])
			if _, ok := props["q"]; !ok {
				t.Errorf("the schema's properties were dropped: %s", mustJSON(out))
			}
		})
	}
}

// A structured-output schema crosses the fold enum-normalized: the official
// SDKs state the gemini wire's SCREAMING type enums on responseSchema too,
// and a strict Messages upstream validates lowercase JSON Schema vocabulary.
func TestGeminiToMessagesResponseSchemaEnumsNormalize(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(
		`{"contents":[{"role":"user","parts":[{"text":"q"}]}],` +
			`"generationConfig":{"response_mime_type":"application/json","response_schema":` +
			`{"type":"OBJECT","properties":{"city":{"type":"STRING"}}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := GeminiToMessages(obj)
	if err != nil {
		t.Fatal(err)
	}
	of, _ := jsonx.AsMap(out["output_format"])
	if of == nil {
		t.Fatalf("no output_format: %v", out)
	}
	schema, _ := jsonx.AsMap(of["schema"])
	if jsonx.GetString(schema, "type") != "object" {
		t.Fatalf("schema root type = %v, want re-spelled object", schema["type"])
	}
	props, _ := jsonx.AsMap(schema["properties"])
	city, _ := jsonx.AsMap(props["city"])
	if jsonx.GetString(city, "type") != "string" {
		t.Fatalf("city type = %v, want re-spelled string", city["type"])
	}
}
