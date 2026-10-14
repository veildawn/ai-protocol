package protocol

import (
	"bytes"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/stream"
)

// This file pins the Messages<->gemini response and SSE folds against LiteLLM
// v1.100.1. The forward direction's wants come from the reference's composed
// stack, run with .zcode/litellm-venv/bin/python during this audit:
//
//   responses: vertex_and_google_ai_studio_gemini.py:2406
//     _transform_google_generate_content_to_openai_model_response
//     -> adapters/transformation.py:1441 translate_openai_response_to_anthropic
//   streams:   vertex_and_google_ai_studio_gemini.py:3239 chunk_parser
//     -> the CustomStreamWrapper the vertex completion path really builds
//     -> adapters/streaming_iterator.py:258 AnthropicStreamWrapper
//
// The reverse direction has no forward path in the reference; its wants are
// the spec's inverse rules, cross-checked by feeding this codec's output back
// through that same composed stack during this audit.

// gmParityResponse folds a gemini response body to Messages through the root
// dispatcher.
func gmParityResponse(t *testing.T, body string) map[string]any {
	t.Helper()
	out, err := ConvertResponse(Gemini, Messages, []byte(body))
	if err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	return gmParityDecode(t, string(out))
}

// The rich response: a signed thought, text, a signed tool call whose
// signature moves off the id, and the full usage arithmetic (exclusive
// thoughts, cache split, detail-derived text tokens). This is the reference's
// print (run this audit; the spec's §2.5 quotes the same composition). Two
// substitutions are the codec's documented determinism, not drift: the
// reference mints call_<uuid> where this codec counts call_0, and it names
// the model from the request — which a pure function does not have — where
// this codec states the package fallback (spec D2/D10).
func TestGeminiMessagesParityResponseRich(t *testing.T) {
	got := gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"think","thought":true,"thoughtSignature":"SIG1"},
	    {"text":"hello"},
	    {"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"SIG2"}]},
	  "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,
	   "cachedContentTokenCount":2,"thoughtsTokenCount":2,
	   "promptTokensDetails":[{"modality":"TEXT","tokenCount":10}],
	   "candidatesTokensDetails":[{"modality":"TEXT","tokenCount":4}]},
	 "responseId":"resp-9"}`)
	want := gmParityDecode(t, `{"id":"resp-9","type":"message","role":"assistant","model":"unknown-model",
	  "stop_sequence":null,
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2},
	  "content":[{"type":"thinking","thinking":"think","signature":"SIG1"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"call_0","name":"f","input":{"a":1},
	              "provider_specific_fields":{"signature":"SIG2"}}],
	  "stop_reason":"tool_use"}`)
	gmParityEqual(t, got, want, "rich response")
}

// A blocked prompt replaces the whole response: no content blocks, the three
// usage totals only, and the Messages dialect's end_turn — the reference's
// print (litellm: vertex_and_google_ai_studio_gemini.py:1630-1663
// _handle_blocked_response -> adapters/transformation.py:1340-1347; the id is
// the reference's minted uuid, this codec leaves it empty per D10).
func TestGeminiMessagesParityResponseBlockedPrompt(t *testing.T) {
	got := gmParityResponse(t, `{"promptFeedback":{"blockReason":"SAFETY"},
	  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`)
	want := gmParityDecode(t, `{"id":"","type":"message","role":"assistant","model":"unknown-model",
	  "stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0},
	  "content":[],"stop_reason":"end_turn"}`)
	gmParityEqual(t, got, want, "blocked response")
}

// The full finishReason enumeration, folded by the reference this audit: the
// flagged reasons replace the response with the filtered shape (empty
// content, three totals), MAX_TOKENS is a ceiling cut, and everything else —
// including the unmapped values, which the reference warns about and folds
// to stop — closes the turn cleanly (litellm:
// core_helpers.py:151-206 _FINISH_REASON_MAP via get_finish_reason_mapping;
// flagged set at vertex_and_google_ai_studio_gemini.py:1294-1309).
func TestGeminiMessagesParityResponseFinishReasons(t *testing.T) {
	flagged := map[string]bool{
		"SAFETY": true, "RECITATION": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true,
		"SPII": true, "IMAGE_SAFETY": true, "IMAGE_PROHIBITED_CONTENT": true,
	}
	for _, fr := range []string{
		"FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER",
		"BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY",
		"IMAGE_PROHIBITED_CONTENT", "TOO_MANY_TOOL_CALLS", "MALFORMED_RESPONSE", "LANGUAGE",
		"WAT",
	} {
		got := gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		    "finishReason":"`+fr+`","index":0}],
		    "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		    "responseId":"r"}`)
		wantStop, wantBlocks := "end_turn", float64(1)
		if flagged[fr] {
			wantBlocks = 0
		}
		if fr == "MAX_TOKENS" {
			wantStop = "max_tokens"
		}
		if got["stop_reason"] != wantStop {
			t.Fatalf("%s: stop_reason = %v, want %s", fr, got["stop_reason"], wantStop)
		}
		if n := len(got["content"].([]any)); float64(n) != wantBlocks {
			t.Fatalf("%s: %d content blocks, want %v", fr, n, wantBlocks)
		}
		if u := got["usage"].(map[string]any); u["input_tokens"] != float64(5) || u["output_tokens"] != float64(3) {
			t.Fatalf("%s: usage = %#v", fr, u)
		}
	}
}

// The usage arithmetic, every branch the reference exercises (all printed
// this audit): tool-use tokens are billable on top of the prompt unless the
// response carries search grounding; the completion widens by the thinking
// tokens when the totals show the candidate count excluded them; cache comes
// off the input; a missing totalTokenCount reads as 0 (litellm:
// vertex_and_google_ai_studio_gemini.py:1699-1917 via the usage translation
// at adapters/transformation.py:1410-1432).
func TestGeminiMessagesParityResponseUsage(t *testing.T) {
	body := func(um string) map[string]any {
		return gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0}],"usageMetadata":`+um+`,"responseId":"r"}`)
	}
	for _, tc := range []struct {
		tag  string
		um   string
		want string
	}{
		{"inclusive", `{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}`,
			`{"input_tokens":5,"output_tokens":3}`},
		{"exclusive", `{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":10,"thoughtsTokenCount":2}`,
			`{"input_tokens":5,"output_tokens":5}`},
		{"tooluse", `{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,"toolUsePromptTokenCount":3}`,
			`{"input_tokens":8,"output_tokens":3}`},
		{"cached", `{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,"cachedContentTokenCount":2,"thoughtsTokenCount":2}`,
			`{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2}`},
		{"sparse", `{"promptTokenCount":7,"totalTokenCount":7,"cachedContentTokenCount":3}`,
			`{"input_tokens":4,"output_tokens":0,"cache_read_input_tokens":3}`},
		{"nototal", `{"promptTokenCount":4,"candidatesTokenCount":2}`,
			`{"input_tokens":4,"output_tokens":2}`},
	} {
		gmParityEqual(t, body(tc.um)["usage"], gmParityDecode(t, tc.want), tc.tag)
	}

	// Grounding: the searches count as the server-tool usage, distinct
	// queries only, and grounded tool use stops being billable (litellm:
	// grounding_requests.py calculate_grounding_requests via
	// _set_grounding_usage_counters at vertex_and_google_ai_studio_gemini.py:1987-1992).
	for _, tc := range []struct {
		tag  string
		body string
		want string
	}{
		{"grounded", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["q1","q2"]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":5,"output_tokens":3,"server_tool_use":{"web_search_requests":2}}`},
		{"duplicates", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["q","q","q2"]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":5,"output_tokens":3,"server_tool_use":{"web_search_requests":2}}`},
		{"empty-queries", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["",""]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":8,"output_tokens":3}`},
		{"maps-only", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"groundingChunks":[{"maps":{"uri":"m"}}]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":5,"output_tokens":3}`},
		{"maps-keeps-queries", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["mq"],"groundingChunks":[{"maps":{"uri":"m"}}]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":5,"output_tokens":3}`},
		{"maps-plus-web", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},
		  "finishReason":"STOP","index":0,"groundingMetadata":{"webSearchQueries":["q1"],"groundingChunks":[{"maps":{"uri":"m"}},{"web":{"uri":"w"}}]}}],
		  "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
		    "toolUsePromptTokenCount":3},"responseId":"r"}`,
			`{"input_tokens":5,"output_tokens":3,"server_tool_use":{"web_search_requests":1}}`},
	} {
		gmParityEqual(t, gmParityResponse(t, tc.body)["usage"], gmParityDecode(t, tc.want), tc.tag)
	}
}

// Media's three homes, printed by the reference this audit: images drop (the
// Messages response has no spelling for model-made images), any other inline
// payload rides the text as a data URL, audio drops, and a text part that IS
// an audio data URL is audio in disguise (litellm:
// vertex_and_google_ai_studio_gemini.py:1348-1532 for the chat-side assembly,
// adapters/transformation.py:1259-1338 for the Messages content fold).
func TestGeminiMessagesParityResponseMedia(t *testing.T) {
	got := gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"see"},{"inlineData":{"mimeType":"image/png","data":"AAAB"}},
	    {"inlineData":{"mimeType":"application/pdf","data":"JVBE"}},
	    {"inlineData":{"mimeType":"audio/wav","data":"AAAA"}}]},
	  "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},
	 "responseId":"r-m"}`)
	want := []any{map[string]any{"type": "text", "text": "seedata:application/pdf;base64,JVBE"}}
	gmParityEqual(t, got["content"], want, "media homes")

	got = gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"hi"},{"text":"data:audio/wav;base64,AAAA"}]},
	  "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},
	 "responseId":"r-a"}`)
	gmParityEqual(t, got["content"], []any{map[string]any{"type": "text", "text": "hi"}}, "audio data URL")

	// A signature on a plain text part has no Messages spelling: the
	// reference's chat fold parks it on provider fields the Messages fold
	// never reads (printed this audit).
	got = gmParityResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"plain","thoughtSignature":"TSIG"}]}, "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},
	 "responseId":"r-ts"}`)
	gmParityEqual(t, got["content"], []any{map[string]any{"type": "text", "text": "plain"}}, "text signature")

	// Two candidates fold into one content list, blocks in candidate order
	// (printed this audit).
	got = gmParityResponse(t, `{"candidates":[
	    {"content":{"role":"model","parts":[{"text":"one"}]},"finishReason":"STOP","index":0},
	    {"content":{"role":"model","parts":[{"text":"two"}]},"finishReason":"STOP","index":1}],
	 "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
	 "responseId":"r-2"}`)
	gmParityEqual(t, got["content"], []any{
		map[string]any{"type": "text", "text": "one"},
		map[string]any{"type": "text", "text": "two"},
	}, "multi candidate")
}

// ---------------------------------------------------------------------------
// Streams, gemini -> Messages. Every want below is the reference's composed
// stack printed for the same frames this audit (chunk_parser ->
// CustomStreamWrapper -> AnthropicStreamWrapper).

// gmParityEvents decodes an SSE stream into (event name, data) pairs.
func gmParityEvents(t *testing.T, out string) []stream.Event {
	t.Helper()
	var evs []stream.Event
	for _, block := range strings.Split(out, "\n\n") {
		var name string
		var data []string
		for _, line := range strings.Split(block, "\n") {
			if p, ok := strings.CutPrefix(line, "event: "); ok {
				name = p
			}
			if p, ok := strings.CutPrefix(line, "data: "); ok {
				data = append(data, p)
			}
		}
		if len(data) == 0 {
			continue
		}
		evs = append(evs, stream.Event{Event: name, Data: strings.Join(data, "\n")})
	}
	return evs
}

// gmParityPipeGeminiToMessages runs the stream fold through the root
// dispatcher and decodes the result.
func gmParityPipeGeminiToMessages(t *testing.T, frames string) []map[string]any {
	t.Helper()
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Messages, &dst, strings.NewReader(frames), StreamOpts{Model: "gemini-2.5-pro"}); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	var out []map[string]any
	for _, ev := range gmParityEvents(t, dst.String()) {
		out = append(out, gmParityDecode(t, ev.Data))
	}
	return out
}

// gmFrames joins gemini chunk literals into that wire's SSE shape: one data
// frame per chunk, no event names, no sentinel.
func gmFrames(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	return b.String()
}

// The spec §2.8 sequence, event by event: dense block indexes, each block
// opened before its delta and stopped after, exactly one message_delta
// carrying the real usage, message_stop last. The message_start id is the
// codec's documented determinism — the reference mints msg_<uuid>, this
// codec keeps the source's responseId (spec D10 and the stream fold's own
// doc).
func TestGeminiMessagesParityStreamSequence(t *testing.T) {
	evs := gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
	))
	wantSeq := []string{
		`{"type":"message_start","message":{"id":"r1","type":"message","role":"assistant","model":"gemini-2.5-pro","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_0","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\": 1}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":5,"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	}
	if len(evs) != len(wantSeq) {
		t.Fatalf("got %d events, want %d:\n%v", len(evs), len(wantSeq), evs)
	}
	for i, want := range wantSeq {
		gmParityEqual(t, evs[i], gmParityDecode(t, want), "event "+itoaGemini(i))
	}
}

// A blocked prompt frame is the source's own filtered close: the turn ends
// end_turn with that frame's usage — the reference's print (litellm:
// vertex_and_google_ai_studio_gemini.py:1935-1981 folds the frame into a
// content_filter close the Messages wrapper spells end_turn).
func TestGeminiMessagesParityStreamBlockedPrompt(t *testing.T) {
	evs := gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"par"}]},"index":0}],"responseId":"r2"}`,
		`{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5},"responseId":"r2"}`,
	))
	var stops []string
	for _, e := range evs {
		if e["type"] == "message_delta" {
			stops = append(stops, e["delta"].(map[string]any)["stop_reason"].(string))
			gmParityEqual(t, e["usage"], gmParityDecode(t, `{"input_tokens":5,"output_tokens":0}`), "blocked usage")
		}
	}
	if len(stops) != 1 || stops[0] != "end_turn" {
		t.Fatalf("message_delta stop_reasons = %v, want one end_turn", stops)
	}
	if n := countType(evs, "message_stop"); n != 1 {
		t.Fatalf("message_stop events = %d, want 1", n)
	}
}

// A usage-only frame (no candidates) parks its usage; the finish frame that
// names no usage of its own closes the turn with the parked counts — the
// reference's hold-and-merge, printed this audit.
func TestGeminiMessagesParityStreamUsageOnlyFrame(t *testing.T) {
	evs := gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r3"}`,
		`{"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r3"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"responseId":"r3"}`,
	))
	for _, e := range evs {
		if e["type"] == "message_delta" {
			gmParityEqual(t, e["usage"], gmParityDecode(t, `{"input_tokens":5,"output_tokens":3}`), "parked usage")
			return
		}
	}
	t.Fatalf("no message_delta:\n%v", evs)
}

// The terminal kinds the reference prints (all run this audit): MAX_TOKENS
// closes max_tokens; a thought's signature rides the chat delta's provider
// fields, which the Messages fold never reads back — dropped on both sides;
// consecutive thought frames share one block; two calls in separate frames
// open two blocks with their own indexes; non-image non-audio inline data
// rides the text as a data URL; the cached usage split reaches the
// message_delta; grounded usage counts the searches.
func TestGeminiMessagesParityStreamTerminalKinds(t *testing.T) {
	cases := []struct {
		tag       string
		frames    []string
		wantStop  string
		wantUsage string
	}{
		{"max_tokens", []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"cu"}]},"index":0}],"responseId":"r4"}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r4"}`,
		}, "max_tokens", `{"input_tokens":5,"output_tokens":3}`},
		{"thought-signature-drops", []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"t","thought":true,"thoughtSignature":"SIG"}]},"index":0}],"responseId":"r5"}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r5"}`,
		}, "end_turn", `{"input_tokens":5,"output_tokens":3}`},
		{"cached-usage", []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r9"}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,"cachedContentTokenCount":2,"thoughtsTokenCount":2},"responseId":"r9"}`,
		}, "end_turn", `{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2}`},
		{"grounding", []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r10"}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0},{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":1,"groundingMetadata":{"webSearchQueries":["q1","q2"]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,"toolUsePromptTokenCount":3},"responseId":"r10"}`,
		}, "end_turn", `{"input_tokens":5,"output_tokens":3,"server_tool_use":{"web_search_requests":2}}`},
	}
	for _, tc := range cases {
		evs := gmParityPipeGeminiToMessages(t, gmFrames(tc.frames...))
		var gotStop string
		for _, e := range evs {
			if e["type"] == "message_delta" {
				gotStop = e["delta"].(map[string]any)["stop_reason"].(string)
				gmParityEqual(t, e["usage"], gmParityDecode(t, tc.wantUsage), tc.tag+" usage")
			}
		}
		if gotStop != tc.wantStop {
			t.Fatalf("%s: stop_reason = %q, want %q", tc.tag, gotStop, tc.wantStop)
		}
	}

	// Consecutive thought frames share one block; a later text frame opens
	// the next index (reference print, this audit).
	evs := gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"A","thought":true}]},"index":0}],"responseId":"r6"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"B","thought":true}]},"index":0}],"responseId":"r6"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r6"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r6"}`,
	))
	var kinds []any
	var deltas []any
	for _, e := range evs {
		if e["type"] == "content_block_start" {
			kinds = append(kinds, e["content_block"].(map[string]any)["type"])
		}
		if e["type"] == "content_block_delta" && e["delta"].(map[string]any)["type"] == "thinking_delta" {
			deltas = append(deltas, e["delta"].(map[string]any)["thinking"])
		}
	}
	gmParityEqual(t, kinds, []any{"thinking", "text"}, "thought block kinds")
	gmParityEqual(t, deltas, []any{"A", "B"}, "thought deltas")

	// Two calls in separate frames open two blocks at their own indexes, and
	// the finish closes as a tool turn (reference print).
	evs = gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r7"}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"g","args":{"b":2}}}]},"index":0}],"responseId":"r7"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r7"}`,
	))
	var toolIdx []any
	var partials []any
	for _, e := range evs {
		if e["type"] == "content_block_start" && e["content_block"].(map[string]any)["type"] == "tool_use" {
			toolIdx = append(toolIdx, e["index"])
		}
		if e["type"] == "message_delta" && e["delta"].(map[string]any)["stop_reason"] != "tool_use" {
			t.Fatalf("two-call stop_reason = %v, want tool_use", e["delta"])
		}
		if d, ok := e["delta"].(map[string]any); ok && d["type"] == "input_json_delta" {
			partials = append(partials, d["partial_json"])
		}
	}
	gmParityEqual(t, toolIdx, []any{float64(0), float64(1)}, "tool block indexes")
	gmParityEqual(t, partials, []any{`{"a": 1}`, `{"b": 2}`}, "tool partial json")

	// Non-image non-audio inline data rides the text as a data URL.
	evs = gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"see"},{"inlineData":{"mimeType":"application/pdf","data":"JVBE"}}]},"index":0}],"responseId":"r8"}`,
		`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r8"}`,
	))
	for _, e := range evs {
		if d, ok := e["delta"].(map[string]any); ok && d["type"] == "text_delta" {
			if d["text"] != "seedata:application/pdf;base64,JVBE" {
				t.Fatalf("text delta = %v", d["text"])
			}
		}
	}
}

// A stream that ends with no finishReason and no sentinel was cut: the
// delivered content stands and nothing closes the turn for the client — no
// message_delta, no message_stop (spec D7.7; the reference's wrapper has no
// such ending either, every legal close names its reason).
func TestGeminiMessagesParityStreamTruncation(t *testing.T) {
	evs := gmParityPipeGeminiToMessages(t, gmFrames(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r"}`,
	))
	for _, e := range evs {
		if e["type"] == "message_delta" || e["type"] == "message_stop" {
			t.Fatalf("a cut stream was closed for the client: %v", e)
		}
	}
	sawText := false
	for _, e := range evs {
		if d, ok := e["delta"].(map[string]any); ok && d["type"] == "text_delta" && d["text"] == "hi" {
			sawText = true
		}
	}
	if !sawText {
		t.Fatalf("delivered content lost:\n%v", evs)
	}
}

func countType(evs []map[string]any, typ string) int {
	n := 0
	for _, e := range evs {
		if e["type"] == typ {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// Response reverse: Messages response -> gemini response. Wants follow the
// spec's §3.2 rules; the roundtrip case pins what the reference's forward
// composition printed when fed this codec's output (run this audit).

// The §3.2 shapes: text blocks become text parts, a signed thinking block a
// signature part and an unsigned one a thought part, a tool_use a camelCase
// functionCall, stop_reason maps back through the chat finish, and the usage
// follows D5's inverse pseudocode — the thinking share splits off, the cache
// halves fold back into the prompt, the total is recomputed.
func TestGeminiMessagesParityResponseReverseShapes(t *testing.T) {
	out, err := ConvertResponse(Messages, Gemini, []byte(`{"id":"m1","type":"message","role":"assistant","model":"mm",
	  "content":[{"type":"thinking","thinking":"t","signature":"S"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"tu_1","name":"f","input":{"a":1}}],
	  "stop_reason":"tool_use",
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2,"output_tokens_details":{"thinking_tokens":2}}}`))
	if err != nil {
		t.Fatal(err)
	}
	got := gmParityDecode(t, string(out))
	want := gmParityDecode(t, `{"candidates":[{"content":{"role":"model","parts":[
	    {"text":"t","thought":true,"thoughtSignature":"S"},
	    {"text":"hello"},
	    {"functionCall":{"name":"f","args":{"a":1}}}]},
	  "finishReason":"STOP","index":0}],
	 "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,
	   "cachedContentTokenCount":2,"thoughtsTokenCount":2,
	   "promptTokensDetails":[{"modality":"TEXT","tokenCount":8}],
	   "candidatesTokensDetails":[{"modality":"TEXT","tokenCount":4}]},
	 "responseId":"m1"}`)
	gmParityEqual(t, got, want, "reverse response")

	// stop_reason mapping back: max_tokens is the ceiling cut, refusal and
	// the filtered reasons take the target's representative value, everything
	// else closes cleanly (§3.2 row; D3's inverse).
	for _, tc := range []struct{ stop, want string }{
		{"max_tokens", "MAX_TOKENS"},
		{"refusal", "SAFETY"},
		{"end_turn", "STOP"},
		{"stop_sequence", "STOP"},
		{"tool_use", "STOP"},
	} {
		out, err := ConvertResponse(Messages, Gemini, []byte(`{"id":"m","content":[{"type":"text","text":"x"}],
		  "stop_reason":"`+tc.stop+`","usage":{"input_tokens":1,"output_tokens":1}}`))
		if err != nil {
			t.Fatal(err)
		}
		got := gmParityDecode(t, string(out))
		cand := got["candidates"].([]any)[0].(map[string]any)
		if cand["finishReason"] != tc.want {
			t.Fatalf("stop %s: finishReason = %v, want %s", tc.stop, cand["finishReason"], tc.want)
		}
	}

	// The D5 inverse for the plain cases: no thinking split, no cache halves.
	out, _ = ConvertResponse(Messages, Gemini, []byte(`{"id":"m","content":[{"type":"text","text":"x"}],
	  "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`))
	um := gmParityDecode(t, string(out))["usageMetadata"].(map[string]any)
	gmParityEqual(t, um, gmParityDecode(t, `{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8,
	  "promptTokensDetails":[{"modality":"TEXT","tokenCount":5}],
	  "candidatesTokensDetails":[{"modality":"TEXT","tokenCount":3}]}`), "plain usage inverse")
}

// The roundtrip: this codec's reverse output read back by this codec's
// forward fold (pinned to the reference above) must land on what the
// reference itself printed when reading this codec's output — the same
// content, usage and stop_reason (run this audit; the reference's
// call_<uuid> id is its own mint, this codec keeps the source's).
// The signed thinking block rides a flagged thought part, so it restores as
// a thinking block with its signature — the reference's reader folds an
// unflagged part's text into the content, which the old unflagged spelling
// demonstrated as pollution (executed during this audit).
func TestGeminiMessagesParityResponseReverseRoundTrip(t *testing.T) {
	src := `{"id":"m1","type":"message","role":"assistant","model":"mm",
	  "content":[{"type":"thinking","thinking":"t","signature":"S"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"tu_1","name":"f","input":{"a":1}}],
	  "stop_reason":"tool_use",
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2,"output_tokens_details":{"thinking_tokens":2}}}`
	out, err := ConvertResponse(Messages, Gemini, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	back, err := ConvertResponse(Gemini, Messages, out)
	if err != nil {
		t.Fatal(err)
	}
	got := gmParityDecode(t, string(back))
	want := gmParityDecode(t, `{"id":"m1","type":"message","role":"assistant","model":"unknown-model",
	  "stop_sequence":null,
	  "usage":{"input_tokens":8,"output_tokens":6,"cache_read_input_tokens":2},
	  "content":[{"type":"thinking","thinking":"t","signature":"S"},
	             {"type":"text","text":"hello"},
	             {"type":"tool_use","id":"call_0","name":"f","input":{"a":1},"provider_specific_fields":null}],
	  "stop_reason":"tool_use"}`)
	gmParityEqual(t, got, want, "reverse response roundtrip")

	// The unsigned thinking block survives its roundtrip as the reference
	// prints it: a thought part reads back as a thinking block with the
	// empty signature the Messages dialect always states.
	out, _ = ConvertResponse(Messages, Gemini, []byte(`{"id":"m2","content":[
	  {"type":"thinking","thinking":"t"},{"type":"text","text":"hello"}],
	  "stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`))
	back, _ = ConvertResponse(Gemini, Messages, out)
	got = gmParityDecode(t, string(back))
	blocks := got["content"].([]any)
	if len(blocks) != 2 || blocks[0].(map[string]any)["type"] != "thinking" {
		t.Fatalf("blocks = %#v, want thinking then text", blocks)
	}
	th := blocks[0].(map[string]any)
	if th["thinking"] != "t" || th["signature"] != "" {
		t.Fatalf("thinking block = %#v, want text t and the stated empty signature", th)
	}
}

// ---------------------------------------------------------------------------
// Stream reverse: Messages SSE -> gemini SSE. Wants follow the spec's §3.3
// rules; the roundtrip pins what the reference's composed stack printed when
// fed this codec's frames (run this audit).

// The §3.3 frames: thinking deltas become thought frames, the signature
// delta carries the block whole (the shape the chat delta's thinking_blocks
// take on the target wire), text deltas text frames, a tool call's arguments
// ride one frame on the last parseable snapshot, and the terminal — written
// only when the source closed — carries the finish reason with the usage.
func TestGeminiMessagesParityStreamReverseFrames(t *testing.T) {
	src := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3,"output_tokens":1}}}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		"",
		"event: content_block_stop",
		`data: {"type":"content_block_stop","index":0}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}`,
		"",
		"event: content_block_stop",
		`data: {"type":"content_block_stop","index":1}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":5}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &dst, strings.NewReader(src), StreamOpts{Model: "gemini-2.5-pro"}); err != nil {
		t.Fatal(err)
	}
	wantFrames := []string{
		`{"candidates":[{"content":{"parts":[{"text":"why","thought":true}],"role":"model"},"index":0}]}`,
		`{"candidates":[{"content":{"parts":[{"thoughtSignature":"SIG"}],"role":"model"},"index":0}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"index":0}]}`,
		`{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"candidatesTokenCount":5,"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":5}],"promptTokenCount":3,"promptTokensDetails":[{"modality":"TEXT","tokenCount":3}],"totalTokenCount":8}}`,
	}
	frames := gmParityEvents(t, dst.String())
	if len(frames) != len(wantFrames) {
		t.Fatalf("got %d frames, want %d:\n%s", len(frames), len(wantFrames), dst.String())
	}
	for i, want := range wantFrames {
		if frames[i].Event != "" {
			t.Fatalf("frame %d carries an event name %q — the target wire has none", i, frames[i].Event)
		}
		gmParityEqual(t, gmParityDecode(t, frames[i].Data), gmParityDecode(t, want), "frame "+itoaGemini(i))
	}
}

// The roundtrip: this codec's frames, read back by this codec's forward fold
// (pinned to the reference above), restore the turn exactly — the thinking
// text once (the signature rides a bare part, whose signature the reader
// collects without a second copy of the text), the text block holding "hi"
// alone, and exactly one message_delta with the original usage. The
// reference's message_start id is its own mint; this codec has no id to
// carry.
func TestGeminiMessagesParityStreamReverseRoundTrip(t *testing.T) {
	src := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":3,"output_tokens":1}}}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		"",
		"event: content_block_stop",
		`data: {"type":"content_block_stop","index":0}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}`,
		"",
		"event: content_block_stop",
		`data: {"type":"content_block_stop","index":1}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":5}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	var gemini bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &gemini, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	var back bytes.Buffer
	if _, err := PipeStream(Gemini, Messages, &back, strings.NewReader(gemini.String()), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	evs := gmParityPipeEventsOf(t, back.String())
	wantSeq := []string{
		`{"type":"message_start","message":{"id":"msg_0","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"why"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	}
	if len(evs) != len(wantSeq) {
		t.Fatalf("got %d events, want %d:\n%v", len(evs), len(wantSeq), evs)
	}
	for i, want := range wantSeq {
		gmParityEqual(t, evs[i], gmParityDecode(t, want), "roundtrip event "+itoaGemini(i))
	}
}

// gmParityPipeEventsOf decodes an already-written SSE stream.
func gmParityPipeEventsOf(t *testing.T, out string) []map[string]any {
	t.Helper()
	var evs []map[string]any
	for _, ev := range gmParityEvents(t, out) {
		evs = append(evs, gmParityDecode(t, ev.Data))
	}
	return evs
}

// The terminal frame is written only when the SOURCE closed the turn. A
// stream cut after the content writes no finishReason; a stream closed by
// message_stop (or the [DONE] sentinel a relay appends) writes it with the
// accumulated usage (spec D7.8).
func TestGeminiMessagesParityStreamReverseTerminalOnClose(t *testing.T) {
	prefix := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":1}}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		"",
	}, "\n") + "\n"
	var cut bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &cut, strings.NewReader(prefix), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(cut.String(), "finishReason") {
		t.Fatalf("terminal written before the source closed:\n%s", cut.String())
	}
	if !strings.Contains(cut.String(), `"text":"hi"`) {
		t.Fatalf("delivered text frame lost:\n%s", cut.String())
	}

	var closed bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &closed, strings.NewReader(prefix+"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(closed.String(), `"finishReason":"STOP"`) {
		t.Fatalf("close did not write the terminal:\n%s", closed.String())
	}
	if !strings.Contains(closed.String(), `"usageMetadata"`) {
		t.Fatalf("terminal carries no usage:\n%s", closed.String())
	}

	// [DONE] closes the turn the same way.
	var done bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &done, strings.NewReader(prefix+"data: [DONE]\n\n"), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(done.String(), `"finishReason":"STOP"`) {
		t.Fatalf("[DONE] did not close the turn:\n%s", done.String())
	}
}

// A tool_use block whose fragments never become a JSON object emits no call —
// the target wire reads each call as one object and cannot carry half of one
// (spec §3.3's argument rule; FAC:1130's json.loads on the request side).
func TestGeminiMessagesParityStreamReverseUnparseableToolArgs(t *testing.T) {
	src := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":1}}}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\": "}}`,
		"",
		"event: content_block_stop",
		`data: {"type":"content_block_stop","index":0}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(dst.String(), "function_call") {
		t.Fatalf("half an arguments object rode the wire:\n%s", dst.String())
	}
	// The stream still closes: the terminal is not held hostage by a call
	// that never completed.
	if !strings.Contains(dst.String(), `"finishReason":"STOP"`) {
		t.Fatalf("terminal missing:\n%s", dst.String())
	}
}

// A Messages "error" event is the source declaring the turn failed: the
// fold writes nothing further — no synthesized terminal would dress the
// failure as a finished answer — and the host's error frame is what the
// client must see. The branch mirrors the gemini-source error frames the
// suite already pins, on the reverse direction.
func TestPipeMessagesToGeminiErrorEventIsSilent(t *testing.T) {
	src := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"mm","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		``,
		`event: error`,
		`data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeStream(Messages, Gemini, &dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	frames := gmParityEvents(t, dst.String())
	if len(frames) == 0 {
		t.Fatalf("no frames written: %q", dst.String())
	}
	for _, f := range frames {
		if strings.Contains(f.Data, `"finishReason"`) {
			t.Fatalf("an error close must not synthesize a terminal frame:\n%s", dst.String())
		}
	}
}
