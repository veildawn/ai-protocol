package protocol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// This file pins the Chat<->Gemini response and stream folds against LiteLLM
// v1.100.1.
//
// Truth source for the gemini->chat directions (the reference has forward
// paths there): every expected value was produced by running the reference
// adapters from the repo venv:
//
//	.zcode/litellm-venv/bin/python
//	  responses: VertexGeminiConfig._transform_google_generate_content_to_openai_model_response
//	             (litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2406-2505)
//	  streams:   ModelResponseIterator(...).chunk_parser
//	             (litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:3239-3284)
//
// The chat->gemini response/stream directions have no forward path in the
// reference: expectations derive from the spec's inverse rules (each cited)
// and were validated in this audit by feeding this codec's output back
// through the reference's forward adapters (round-trip; lossy points are
// asserted as documented).
//
// Documented, deliberate non-reference values (the spec's no-clock/no-uuid
// ruling and this library's wire vocabulary):
//   - created is 0 (the reference stamps its clock),
//   - id falls back to "chatcmpl-unknown" (the reference mints a uuid;
//     responseId still wins when the wire states one),
//   - model is empty / the StreamOpts model (the reference copies the request
//     model; the gemini wire has none to read),
//   - the reference's dump always states provider metadata members and
//     always-null delta members (delta.role/tool_calls/function_call/audio,
//     logprobs, system_fingerprint); this codec omits absent members - the
//     same value space, and the provider member names are outside this
//     library's vocabulary.
//   - a chat-target relay writes the [DONE] sentinel after the source's own
//     close; the gemini wire has no sentinel (the reference's raw parser
//     emits none).
//   - the raw parser stamps finish_reason on every chunk (its default is
//     "stop", vertex_and_google_ai_studio_gemini.py:1919-1933); the fold
//     names a finish only on the frame that carries the wire's close, the
//     behavior the reference's full streaming stack surfaces.

// geminiChatParitySSE frames a list of JSON objects into an SSE source.
func geminiChatParitySSE(frames []string) string {
	var b strings.Builder
	for _, f := range frames {
		b.WriteString("data: " + f + "\n\n")
	}
	return b.String()
}

// geminiChatParityChunkData returns the parsed data payloads of an SSE
// output, with "[DONE]" kept as a marker element.
func geminiChatParityChunkData(t *testing.T, out string) []string {
	t.Helper()
	var evs []string
	for _, block := range strings.Split(out, "\n\n") {
		var data string
		for _, line := range strings.Split(block, "\n") {
			if p, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(p)
			}
		}
		if data == "" {
			continue
		}
		evs = append(evs, data)
	}
	return evs
}

// TestGeminiChatParityGeminiResponseToChat pins the response fold against
// reference-ran truth (adapter:
// _transform_google_generate_content_to_openai_model_response,
// vertex_and_google_ai_studio_gemini.py:2406-2505).
func TestGeminiChatParityGeminiResponseToChat(t *testing.T) {
	cases := []struct{ name, gemini, want string }{
		// Text turn: content from the text parts, images/thinking_blocks
		// stated empty (vertex_and_google_ai_studio_gemini.py:1348-1388,
		// 2263-2272); usage three totals plus the completion text detail.
		{"text", `{"candidates":[{"content":{"role":"model","parts":[{"text":"hello there"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"resp-1"}`,
			`{"id":"resp-1","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":"hello there","role":"assistant","tool_calls":null,"function_call":null,"images":[],"thinking_blocks":[]},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// Thought + call: reasoning_content and one thinking block per
		// thought part, the call with json.dumps-spaced arguments, and the
		// finish overridden to tool_calls
		// (vertex_and_google_ai_studio_gemini.py:1548-1600, 1919-1933,
		// 2307-2322). The id is this codec's positional stand-in (the
		// reference mints a uuid there - documented above).
		{"thought and tool", `{"candidates":[{"content":{"role":"model","parts":[{"text":"let me think","thought":true},{"functionCall":{"name":"get_weather","args":{"city":"Boston"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","reasoning_content":"let me think","thinking_blocks":[{"type":"thinking","thinking":"let me think"}],"tool_calls":[{"index":0,"function":{"arguments":"{\"city\": \"Boston\"}","name":"get_weather"},"id":"call_0","type":"function"}],"function_call":null,"images":[]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// A call with its own signature: the signature rides the call's
		// provider fields and the id suffix, and the message's
		// thought_signatures (vertex_and_google_ai_studio_gemini.py:1412-1428,
		// factory.py:1142-1157).
		{"tool with signature", `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}},"thoughtSignature":"SIG"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\"a\": 1}","name":"f"},"id":"call_0__thought__SIG","provider_specific_fields":{"thought_signature":"SIG"},"type":"function"}],"function_call":null,"images":[],"thinking_blocks":[],"provider_specific_fields":{"thought_signatures":["SIG"]}},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// The wire's own call id wins over the positional stand-in
		// (vertex_and_google_ai_studio_gemini.py:1562-1570).
		{"tool native id", `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{},"id":"native_1"}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{}","name":"f"},"id":"native_1","type":"function"}],"function_call":null,"images":[],"thinking_blocks":[]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// Two calls: indexes accumulate
		// (vertex_and_google_ai_studio_gemini.py:3097).
		{"two tools", `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}},{"functionCall":{"name":"g","args":{}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\"a\": 1}","name":"f"},"id":"call_0","type":"function"},{"index":1,"function":{"arguments":"{}","name":"g"},"id":"call_1","type":"function"}],"function_call":null,"images":[],"thinking_blocks":[]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// Inline media: images ride message.images with detail "auto", other
		// inline data is spliced into the text content as data URLs
		// (vertex_and_google_ai_studio_gemini.py:1365-1375, 1491-1509).
		{"inline media", `{"candidates":[{"content":{"role":"model","parts":[{"text":"see"},{"inlineData":{"mimeType":"image/png","data":"AAAB"}},{"inlineData":{"mimeType":"application/pdf","data":"JVBE"}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":"seedata:application/pdf;base64,JVBE","role":"assistant","images":[{"image_url":{"url":"data:image/png;base64,AAAB","detail":"auto"},"index":0,"type":"image_url"}],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// Audio inline data rides message.audio; the text content survives it
		// (audio nulls the content first, then the content assignment wins -
		// vertex_and_google_ai_studio_gemini.py:2267-2272). The id/expiry are
		// this codec's empty stand-ins for the reference's uuid/clock.
		{"audio inline with text", `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"},{"inlineData":{"mimeType":"audio/wav","data":"AAAA"}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":"hi","role":"assistant","audio":{"id":"","data":"AAAA","expires_at":0,"transcript":""},"images":[],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// A text part that is exactly a base64 audio data URL reads as audio
		// and leaves no content (vertex_and_google_ai_studio_gemini.py:1514-1532).
		{"text audio data url", `{"candidates":[{"content":{"role":"model","parts":[{"text":"data:audio/wav;base64,AAAA"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","audio":{"id":"","data":"AAAA","expires_at":0,"transcript":""},"images":[],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// A signed thought part: reasoning, the signed block, the message
		// signatures, and the answer text all together
		// (vertex_and_google_ai_studio_gemini.py:1389-1428).
		{"thought with signature", `{"candidates":[{"content":{"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"},{"text":"answer"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":"answer","role":"assistant","reasoning_content":"why","thinking_blocks":[{"type":"thinking","thinking":"why","signature":"SIG"}],"images":[],"tool_calls":null,"function_call":null,"provider_specific_fields":{"thought_signatures":["SIG"]}},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// A refused prompt collapses the response to one filtered choice
		// (vertex_and_google_ai_studio_gemini.py:1630-1663, 2423-2428);
		// usage states the three totals with null details.
		{"prompt feedback blocked", `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","tool_calls":null,"function_call":null},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5,"completion_tokens_details":null,"prompt_tokens_details":null}}`},
		// A flagged candidate finish collapses the same way
		// (vertex_and_google_ai_studio_gemini.py:1294-1309, 1664-1697, 2430-2437).
		{"candidate safety", `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"SAFETY","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","tool_calls":null,"function_call":null},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":null,"prompt_tokens_details":null}}`},
		// Multi-candidate: one choice per candidate, index from the candidate
		// (vertex_and_google_ai_studio_gemini.py:2344-2349); responseId is
		// the chat id verbatim (2439-2442).
		{"multi candidate", `{"candidates":[{"content":{"role":"model","parts":[{"text":"one"}]},"finishReason":"STOP","index":0},{"content":{"role":"model","parts":[{"text":"two"}]},"finishReason":"STOP","index":1}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r9"}`,
			`{"id":"r9","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":"one","role":"assistant","images":[],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"},{"index":1,"message":{"content":"two","role":"assistant","images":[],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// No candidates at all -> an empty choice list and no filter
		// (vertex_and_google_ai_studio_gemini.py:2430-2470).
		{"no candidates", `{"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// A candidate without content contributes no choice
		// (vertex_and_google_ai_studio_gemini.py:2216-2219).
		{"candidate without content", `{"candidates":[{"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
		// Empty parts: the members are stated, nothing else
		// (the content stays null - no part said anything).
		{"empty parts", `{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			`{"id":"chatcmpl-unknown","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"content":null,"role":"assistant","images":[],"thinking_blocks":[],"tool_calls":null,"function_call":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ConvertResponse(Gemini, Chat, []byte(tc.gemini))
			if err != nil {
				t.Fatalf("%s: %v", tc.gemini, err)
			}
			if g, w := geminiChatCanonical(t, string(out)), geminiChatCanonical(t, tc.want); g != w {
				t.Fatalf("gemini body %s\n got  %s\n want %s", tc.gemini, g, w)
			}
		})
	}
}

// The finishReason enumeration, reference-ran per value through the same
// adapter. The table is _GEMINI_FINISH_REASON_KEYS filtered through
// _FINISH_REASON_MAP (vertex_and_google_ai_studio_gemini.py:1310-1345,
// litellm_core_utils/core_helpers.py:151-206); unknown values warn and fold
// to stop (core_helpers.py:202-206).
func TestGeminiChatParityGeminiFinishReasons(t *testing.T) {
	cases := []struct{ reason, want string }{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"RECITATION", "content_filter"},
		{"LANGUAGE", "content_filter"},
		{"OTHER", "content_filter"},
		{"BLOCKLIST", "content_filter"},
		{"PROHIBITED_CONTENT", "content_filter"},
		{"SPII", "content_filter"},
		{"IMAGE_SAFETY", "content_filter"},
		{"IMAGE_PROHIBITED_CONTENT", "content_filter"},
		{"FINISH_REASON_UNSPECIFIED", "stop"},
		{"MALFORMED_FUNCTION_CALL", "stop"},
		{"TOO_MANY_TOOL_CALLS", "stop"},
		{"MALFORMED_RESPONSE", "stop"},
		// Off-table values: the mapper warns and returns stop
		// (reference-ran: both WAT and PROHIBITED).
		{"WAT", "stop"},
		{"PROHIBITED", "stop"},
	}
	for _, tc := range cases {
		body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"` + tc.reason + `","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`
		out, err := ConvertResponse(Gemini, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.reason, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		choices := m["choices"].([]any)
		if len(choices) != 1 {
			t.Fatalf("%s: choices = %#v", tc.reason, choices)
		}
		// The flagged reasons replace the whole choice; the mapped value is
		// still the reference's own replacement outcome.
		if got := choices[0].(map[string]any)["finish_reason"]; got != tc.want {
			t.Fatalf("finishReason %s -> %v, want %v", tc.reason, got, tc.want)
		}
	}
}

// The usage arithmetic, reference-ran per case through the same adapter:
// the inclusive/exclusive thoughts decision
// (vertex_and_google_ai_studio_gemini.py:1699-1716, 1902-1914), the
// tool-use prompt surcharge (1885-1900), the cache reading (1763-1764), and
// the per-modality details with the cache subtracted back out (1785-1876).
func TestGeminiChatParityGeminiUsage(t *testing.T) {
	cases := []struct{ name, metadata, want string }{
		// Inclusive: prompt+cand+toolUse == total, so the thoughts are
		// already inside the candidate count; completion stays 3 and the
		// details still state the reasoning tokens.
		{"inclusive", `{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":10,"thoughtsTokenCount":2}`,
			`{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10,"cache_read_input_tokens":null,"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"completion_tokens_details":{"reasoning_tokens":2,"text_tokens":3}}`},
		// Exclusive: the totals do not add up without the thoughts, so they
		// are added back into completion.
		{"exclusive", `{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,"thoughtsTokenCount":2}`,
			`{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16,"cache_read_input_tokens":null,"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"completion_tokens_details":{"reasoning_tokens":2,"text_tokens":4}}`},
		// Sparse: no candidatesTokenCount, a cache reading, total copied.
		{"sparse", `{"promptTokenCount":7,"totalTokenCount":7,"cachedContentTokenCount":3}`,
			`{"prompt_tokens":7,"completion_tokens":0,"total_tokens":7,"cache_read_input_tokens":3,"prompt_tokens_details":{"cached_tokens":3,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"completion_tokens_details":null}`},
		// No totalTokenCount: total is 0, not invented.
		{"no total", `{"promptTokenCount":5,"candidatesTokenCount":3}`,
			`{"prompt_tokens":5,"completion_tokens":3,"total_tokens":0,"cache_read_input_tokens":null,"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"completion_tokens_details":{"text_tokens":3}}`},
		// Tool-use prompt tokens are billed into prompt_tokens.
		{"tool use", `{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":16,"toolUsePromptTokenCount":3}`,
			`{"prompt_tokens":13,"completion_tokens":4,"total_tokens":16,"cache_read_input_tokens":null,"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null,"tool_use_tokens":3},"completion_tokens_details":{"text_tokens":4}}`},
		// Per-modality details: prompt text stated inclusive of the cache and
		// the cache subtracted back out; completion image+text split.
		{"details", `{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":14,"cachedContentTokenCount":2,"promptTokensDetails":[{"modality":"TEXT","tokenCount":8},{"modality":"IMAGE","tokenCount":2}],"cacheTokensDetails":[{"modality":"TEXT","tokenCount":2}],"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":3},{"modality":"IMAGE","tokenCount":1}]}`,
			`{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"cache_read_input_tokens":2,"prompt_tokens_details":{"cached_tokens":2,"audio_tokens":null,"text_tokens":6,"image_tokens":2,"video_tokens":null},"completion_tokens_details":{"text_tokens":3,"image_tokens":1}}`},
	}
	for _, tc := range cases {
		body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],"usageMetadata":` + tc.metadata + `}`
		out, err := ConvertResponse(Gemini, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		if g, w := geminiChatCanonical(t, mustMarshal(t, m["usage"])), geminiChatCanonical(t, tc.want); g != w {
			t.Fatalf("%s\n got  %s\n want %s", tc.name, g, w)
		}
	}
}

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestGeminiChatParityGeminiStreamToChat pins the stream fold against
// reference-ran truth (adapter: ModelResponseIterator.chunk_parser,
// vertex_and_google_ai_studio_gemini.py:3239-3284; delta construction
// 1996-2032; the has_seen_tool_calls rewrite 3132-3205). Expected chunks are
// the reference's values modulo the documented stand-ins listed at the top
// of this file.
func TestGeminiChatParityGeminiStreamToChat(t *testing.T) {
	cases := []struct {
		name   string
		frames []string
		want   []string // expected data payloads, "[DONE]" as marker
	}{
		{
			"text deltas then close",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
		{
			// A thought delta carries reasoning_content and the signature in
			// the delta's provider fields, never thinking blocks (the blocks
			// are the non-streaming spelling -
			// vertex_and_google_ai_studio_gemini.py:1996-2032, 2307-2322).
			"thought then answer",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"t","thought":true,"thoughtSignature":"SIG"}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"answer"}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"t","provider_specific_fields":{"thought_signatures":["SIG"]}},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
		{
			// A call delta then a plain STOP close: the rewrite turns the
			// close into tool_calls
			// (vertex_and_google_ai_studio_gemini.py:3132-3205).
			"tool then rewritten stop",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\": 1}","name":"f"},"id":"call_0","type":"function"}]},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
		{
			// Two calls on separate frames: the wire indexes accumulate
			// (vertex_and_google_ai_studio_gemini.py:3097).
			"two tools accumulate indexes",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"g","args":{}}}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\": 1}","name":"f"},"id":"call_0","type":"function"}]},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{}","name":"g"},"id":"call_1","type":"function"}]},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"text_tokens":1},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
		{
			// A usage-only frame yields the choices-less usage chunk
			// (vertex_and_google_ai_studio_gemini.py:3239-3284) and closes
			// nothing: no finish arrived, so no sentinel either.
			"usage only frame",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
				`{"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
			},
		},
		{
			// A refused prompt is one filtered chunk with the role stated
			// (vertex_and_google_ai_studio_gemini.py:1935-1981, 2423-2428).
			"blocked prompt",
			[]string{
				`{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5,"completion_tokens_details":null,"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
		{
			// MAX_TOKENS maps to length on the closing frame.
			"max tokens close",
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"index":0}],"responseId":"r1"}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r1"}`,
			},
			[]string{
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`,
				`{"id":"r1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"text_tokens":3},"prompt_tokens_details":{"cached_tokens":null,"audio_tokens":null,"text_tokens":null,"image_tokens":null,"video_tokens":null},"cache_read_input_tokens":null}}`,
				"[DONE]",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var dst bytes.Buffer
			if _, err := PipeStream(Gemini, Chat, &dst, strings.NewReader(geminiChatParitySSE(tc.frames)), StreamOpts{Model: "m"}); err != nil {
				t.Fatal(err)
			}
			got := geminiChatParityChunkData(t, dst.String())
			if len(got) != len(tc.want) {
				t.Fatalf("chunks = %d, want %d:\n%s", len(got), len(tc.want), dst.String())
			}
			for i, w := range tc.want {
				if w == "[DONE]" {
					if got[i] != "[DONE]" {
						t.Fatalf("chunk %d = %s, want the [DONE] sentinel", i, got[i])
					}
					continue
				}
				if g := geminiChatCanonical(t, got[i]); g != geminiChatCanonical(t, w) {
					t.Fatalf("chunk %d\n got  %s\n want %s", i, g, w)
				}
			}
		})
	}
}

// An in-band error frame ends the stream with nothing written for the
// client: the reference's parser raises
// (vertex_and_google_ai_studio_gemini.py:3108-3130) and the fold's silence
// leaves the failure to the host that saw it (the spec's error ruling).
func TestGeminiChatParityGeminiStreamErrorFrameIsSilent(t *testing.T) {
	var dst bytes.Buffer
	_, err := PipeStream(Gemini, Chat, &dst, strings.NewReader(geminiChatParitySSE([]string{
		`{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`,
	})), StreamOpts{Model: "m"})
	if err != nil {
		t.Fatalf("the fold must not fail the pipe: %v", err)
	}
	if dst.Len() != 0 {
		t.Fatalf("error frame produced output: %s", dst.String())
	}
}

// A source that stops without a finish reason was cut: nothing closes for
// the client (the truncation ruling - no synthetic finish, no sentinel).
func TestGeminiChatParityGeminiStreamTruncation(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Gemini, Chat, &dst, strings.NewReader(geminiChatParitySSE([]string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}],"responseId":"r1"}`,
	})), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	got := geminiChatParityChunkData(t, dst.String())
	if len(got) != 1 || strings.Contains(got[0], "finish_reason\":\"") {
		t.Fatalf("a cut source must leave the turn open, got %v", got)
	}
	if strings.Contains(dst.String(), "[DONE]") {
		t.Fatalf("a cut source must not gain a sentinel: %s", dst.String())
	}
}

// TestGeminiChatParityChatResponseToGemini pins the reverse response fold.
// The reference has no forward path here, so expectations follow the spec's
// inverse rules (cited per row) and were validated in this audit by feeding
// this codec's gemini body back through the reference's response adapter,
// which restored the original chat message, finish and usage (headline) for
// every case but the documented losses in the next two tests.
func TestGeminiChatParityChatResponseToGemini(t *testing.T) {
	cases := []struct{ name, chat, want string }{
		// Text turn (vertex_and_google_ai_studio_gemini.py:1348-1388
		// inverse); the chat id is the responseId
		// (vertex_and_google_ai_studio_gemini.py:2439-2442 inverse).
		{"text", `{"id":"chatcmpl-1","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"chatcmpl-1"}`},
		// reasoning_content -> a thought part
		// (transformation.py:880-883 inverse); the usage inverse splits the
		// reasoning tokens back out (D5: candidates = completion -
		// reasoning, total recomputed).
		{"reasoning", `{"id":"chatcmpl-2","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","reasoning_content":"why"}}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10,"completion_tokens_details":{"reasoning_tokens":2}}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"why","thought":true},{"text":"a"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"thoughtsTokenCount":2,"totalTokenCount":10},"responseId":"chatcmpl-2"}`},
		// Signed thinking blocks -> flagged, signature-carrying thought
		// parts - the one shape the reference's response reader reconstructs
		// as reasoning with its signature (executed during this audit; the
		// request wire's unflagged replay spelling reads back as content);
		// a redacted block has no request-side reader and drops.
		{"thinking blocks", `{"id":"chatcmpl-3","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","thinking_blocks":[{"type":"thinking","thinking":"signed","signature":"SIG"},{"type":"redacted_thinking","data":"x"}]}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"signed","thought":true,"thoughtSignature":"SIG"},{"text":"a"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"chatcmpl-3"}`},
		// tool_calls -> functionCall parts with parsed args and the
		// finish folded back to STOP (factory.py:1123-1138 inverse; a
		// tool-call close is a natural stop on the wire).
		{"tool calls", `{"id":"chatcmpl-4","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}]}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"chatcmpl-4"}`},
		// A call's signature rides the part whichever spelling carried it
		// (factory.py:1282-1290, 1142-1157 inverse).
		{"tool call signature", `{"id":"chatcmpl-5","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"c1__thought__SIG","type":"function","function":{"name":"f","arguments":"{}"},"provider_specific_fields":{"thought_signature":"SIG"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{}},"thoughtSignature":"SIG"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"chatcmpl-5"}`},
		// message.images -> inlineData parts (the request fold reads the
		// images field - transformation.py:943-966 inverse); camelCase is the
		// response wire's own spelling.
		{"images", `{"id":"chatcmpl-6","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","images":[{"image_url":{"url":"data:image/png;base64,AAAB","detail":"auto"},"index":0}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"},{"inlineData":{"mimeType":"image/png","data":"AAAB"}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"chatcmpl-6"}`},
		// message.audio -> an inlineData part; the chat audio member names
		// no media type, so the fixed wav spelling stands in (documented
		// approximation - the reference's speech output is pcm).
		{"audio", `{"id":"chatcmpl-7","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"audio":{"id":"","data":"AAAA","expires_at":0,"transcript":""}}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/wav","data":"AAAA"}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"responseId":"chatcmpl-7"}`},
		// The finish map (core_helpers.py:151-206 inverse): length ->
		// MAX_TOKENS; content_filter/refusal -> SAFETY (the representative
		// reason - the chat side cannot say which); anything else -> STOP.
		{"finish length", `{"id":"r-length","choices":[{"finish_reason":"length","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r-length"}`},
		{"finish content filter", `{"id":"r-cf","choices":[{"finish_reason":"content_filter","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"SAFETY","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r-cf"}`},
		{"finish refusal", `{"id":"r-ref","choices":[{"finish_reason":"refusal","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"SAFETY","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r-ref"}`},
		{"finish unknown", `{"id":"r-weird","choices":[{"finish_reason":"weird","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"r-weird"}`},
		// The usage inverse (D5): candidates = completion - reasoning,
		// thoughts stated only when positive, total recomputed, the cache
		// reading restored with prompt text stated inclusive of it (so the
		// forward's subtraction lands back on the original), and completion
		// text restored as the candidates detail.
		{"usage cached and reasoning", `{"id":"u3","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":2,"text_tokens":8},"completion_tokens_details":{"reasoning_tokens":2,"text_tokens":4}}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":16,"cachedContentTokenCount":2,"promptTokensDetails":[{"modality":"TEXT","tokenCount":10}],"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":4}]},"responseId":"u3"}`},
		{"usage zeros", `{"id":"u5","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0},"responseId":"u5"}`},
		// No id: no responseId member (the wire's id is optional).
		{"no id", `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`},
		// Several choices: one candidate each, the chat indexes kept
		// (vertex_and_google_ai_studio_gemini.py:2344-2349 inverse).
		{"multi choices", `{"id":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"one"}},{"index":1,"finish_reason":"stop","message":{"role":"assistant","content":"two"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"one"}]},"finishReason":"STOP","index":0},{"content":{"role":"model","parts":[{"text":"two"}]},"finishReason":"STOP","index":1}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"m"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ConvertResponse(Chat, Gemini, []byte(tc.chat))
			if err != nil {
				t.Fatalf("%s: %v", tc.chat, err)
			}
			if g, w := geminiChatCanonical(t, string(out)), geminiChatCanonical(t, tc.want); g != w {
				t.Fatalf("chat body %s\n got  %s\n want %s", tc.chat, g, w)
			}
		})
	}
}

// Round-trip: the gemini body this codec's reverse fold produces must
// restore the original chat response through a gemini->chat fold that is
// itself reference-exact (the reference round-trip ran during this audit;
// here the composition is exercised in-process over the same inputs).
func TestGeminiChatParityResponseRoundTrip(t *testing.T) {
	restore := []string{
		// A signed thinking block rides a flagged thought part, the one shape
		// the reference's response reader reconstructs as reasoning with its
		// signature (executed during this audit: the unflagged request-wire
		// spelling folds into the content instead).
		`{"id":"chatcmpl-3","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","thinking_blocks":[{"type":"thinking","thinking":"signed","signature":"SIG"}]}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
		`{"id":"chatcmpl-1","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
		`{"id":"chatcmpl-2","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","reasoning_content":"why"}}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10,"completion_tokens_details":{"reasoning_tokens":2}}}`,
		`{"id":"chatcmpl-4","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}}]}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
		`{"id":"chatcmpl-6","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"a","images":[{"image_url":{"url":"data:image/png;base64,AAAB","detail":"auto"},"index":0}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		`{"id":"r-length","choices":[{"finish_reason":"length","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
	}
	for _, body := range restore {
		gemini, err := ConvertResponse(Chat, Gemini, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", body, err)
		}
		back, err := ConvertResponse(Gemini, Chat, gemini)
		if err != nil {
			t.Fatalf("%s -> %s: %v", body, gemini, err)
		}
		var want, got map[string]any
		if err := json.Unmarshal([]byte(body), &want); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(back, &got); err != nil {
			t.Fatal(err)
		}
		// Compare the message's carried content, the finish and the headline
		// usage: the reference's reader always states the empty members
		// (function_call/images/thinking_blocks) alongside, so only the
		// members the source stated are required to survive verbatim; the
		// created/model members are the documented stand-ins.
		gotMsg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
		wantMsg := want["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
		for _, k := range []string{"content", "reasoning_content", "thinking_blocks"} {
			g, gok := gotMsg[k]
			w, wok := wantMsg[k]
			if !wok {
				continue
			}
			if !gok || mustMarshal(t, g) != mustMarshal(t, w) {
				t.Fatalf("%s\n gemini %s\n message[%q] got  %s\n           want %s", body, gemini, k, mustMarshal(t, g), mustMarshal(t, w))
			}
		}
		// Images restore by their URL, detail and index: the reference's
		// reader always adds the item's own type member on the way back
		// (vertex_and_google_ai_studio_gemini.py:1491-1509), so the member
		// set is a superset of what the source stated.
		if wantImgs, ok := wantMsg["images"].([]any); ok {
			gotImgs, _ := gotMsg["images"].([]any)
			if len(gotImgs) != len(wantImgs) {
				t.Fatalf("%s: images = %s, want %d", body, mustMarshal(t, gotImgs), len(wantImgs))
			}
			for i, raw := range wantImgs {
				wiu := raw.(map[string]any)["image_url"].(map[string]any)
				giu := gotImgs[i].(map[string]any)["image_url"].(map[string]any)
				if giu["url"] != wiu["url"] || giu["detail"] != wiu["detail"] {
					t.Fatalf("%s: image %d got %s, want url+detail of %s", body, i, mustMarshal(t, giu), mustMarshal(t, wiu))
				}
			}
		}
		// The calls restore by name and arguments; the chat-side ids cannot
		// ride the 2.5 wire (the forward fold writes none), which the
		// documented-losses test covers.
		if wantCalls, ok := wantMsg["tool_calls"].([]any); ok {
			gotCalls, _ := gotMsg["tool_calls"].([]any)
			if len(gotCalls) != len(wantCalls) {
				t.Fatalf("%s: calls = %s, want %d", body, mustMarshal(t, gotCalls), len(wantCalls))
			}
			for i, raw := range wantCalls {
				wf := raw.(map[string]any)["function"].(map[string]any)
				gf := gotCalls[i].(map[string]any)["function"].(map[string]any)
				if gf["name"] != wf["name"] || gf["arguments"] != wf["arguments"] {
					t.Fatalf("%s: call %d got %s, want name+args of %s", body, i, mustMarshal(t, gf), mustMarshal(t, wf))
				}
			}
		}
		if got["choices"].([]any)[0].(map[string]any)["finish_reason"] != want["choices"].([]any)[0].(map[string]any)["finish_reason"] {
			t.Fatalf("%s: finish not restored: %s", body, back)
		}
		for _, k := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
			if got["usage"].(map[string]any)[k] != want["usage"].(map[string]any)[k] {
				t.Fatalf("%s: usage[%s] not restored: %s", body, k, back)
			}
		}
		if got["id"] != want["id"] {
			t.Fatalf("%s: id not restored: %s", body, back)
		}
	}
}

// The documented lossy round-trips of the reverse response fold, asserted as
// the spec rules them rather than silently different:
//
//   - content_filter/refusal map to SAFETY, and the reference's reader
//     collapses a SAFETY candidate into the filtered single choice - the
//     chat side's finer reasons are a documented drop
//     (core_helpers.py:151-206 inverse);
//   - a finish_reason of tool_calls with no calls at all folds to STOP and
//     cannot come back (a tool-call close is a natural stop);
//   - a self-contradictory total is recomputed (D5's irreversibility: the
//     arithmetic must stay self-consistent so the inclusive/exclusive
//     decision reads back correctly).
func TestGeminiChatParityResponseRoundTripDocumentedLosses(t *testing.T) {
	roundTrip := func(t *testing.T, chat string) map[string]any {
		t.Helper()
		gemini, err := ConvertResponse(Chat, Gemini, []byte(chat))
		if err != nil {
			t.Fatal(err)
		}
		back, err := ConvertResponse(Gemini, Chat, gemini)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(back, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}

	// Signed thinking block: the flagged shape restores exactly (the exact
	// round-trip test above covers it); nothing left to document here.

	// content_filter -> SAFETY -> the filtered collapse.
	back := roundTrip(t, `{"id":"cf","choices":[{"finish_reason":"content_filter","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`)
	if back["choices"].([]any)[0].(map[string]any)["finish_reason"] != "content_filter" {
		t.Fatalf("content_filter round trip = %s", mustMarshal(t, back))
	}
	if back["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"] != nil {
		t.Fatalf("the SAFETY collapse must drop the content: %s", mustMarshal(t, back))
	}

	// tool_calls with no calls folds to STOP and stays stop.
	back = roundTrip(t, `{"id":"tc","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`)
	if fr := back["choices"].([]any)[0].(map[string]any)["finish_reason"]; fr != "stop" {
		t.Fatalf("call-less tool_calls round trip finish = %v, want stop (documented)", fr)
	}

	// Self-contradictory total: the recomputed one, per D5.
	gemini, err := ConvertResponse(Chat, Gemini, []byte(`{"id":"u4","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":10}}`))
	if err != nil {
		t.Fatal(err)
	}
	var gm map[string]any
	if err := json.Unmarshal(gemini, &gm); err != nil {
		t.Fatal(err)
	}
	um := gm["usageMetadata"].(map[string]any)
	if um["totalTokenCount"] != float64(8) {
		t.Fatalf("totalTokenCount = %#v, want the recomputed 8 (prompt 5 + candidates 3)", um["totalTokenCount"])
	}
	back2 := roundTrip(t, `{"id":"u4","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":10}}`)
	u := back2["usage"].(map[string]any)
	if u["prompt_tokens"] != float64(5) || u["completion_tokens"] != float64(3) {
		t.Fatalf("headline usage not restored: %s", mustMarshal(t, u))
	}
}

// TestGeminiChatParityChatToGeminiStream pins the reverse stream fold: the
// wire it emits is the response-body vocabulary (camelCase parts, the finish
// reason and the usage on one closing frame). Expectations follow the
// spec's inverse rules and were validated in this audit by feeding the
// frames back through the reference's chunk parser, which restored the
// accumulated content, reasoning, calls, close and usage.
func TestGeminiChatParityChatToGeminiStream(t *testing.T) {
	cases := []struct {
		name   string
		source []string
		want   []string
	}{
		{
			// Content deltas then a finish whose chunk already carries the
			// usage: the terminal frame states both together (the wire has
			// no separate usage frame).
			"content then finish with usage",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hel"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// A chat source that reports its usage on a choices-less chunk
			// after the finish: the held finish waits for it, so the single
			// terminal frame still carries both.
			"finish then usage tail",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				`{"id":"c1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// A chat source that closes on the bare usage tail — the ending
			// several vendors send instead of a finish_reason or [DONE] — is
			// the source's own close (the observer settles it Terminal via
			// its usage-close rule). The fold writes the terminal frame with
			// the wire's natural stop and the counts the tail carried, the
			// same ruling the chat→messages and chat→responses folds apply
			// to this ending.
			"usage tail without finish",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
				`{"id":"c1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// reasoning_content deltas become thought parts
			// (transformation.py:880-883 inverse).
			"reasoning",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"reasoning_content":"why"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"a"}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"why","thought":true}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// A signed thinking-block delta becomes a flagged thought part —
			// the shape the reference's response reader reconstructs as
			// reasoning with its signature; the request wire's unflagged
			// replay spelling reads back as content (executed this audit).
			// An unsigned one rides nothing (the forward fold drops it).
			"thinking block delta",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"thinking_blocks":[{"type":"thinking","thinking":"why","signature":"SIG"}]}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"why","thought":true,"thoughtSignature":"SIG"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// Argument fragments that do not parse yet ride nothing; the
			// completed object goes out as one whole call ahead of the
			// close (the wire's args member is a whole JSON object -
			// factory.py:1130's parse is the forward the inverse must
			// satisfy).
			"tool fragments",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":""}}]}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\""}}]}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": 1}"}}]}}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			},
		},
		{
			// A [DONE] sentinel with no finish anywhere: the source stated
			// its own close, so the fold names the natural stop and states
			// the (empty) usage the wire's terminal always carries.
			"done sentinel close",
			[]string{
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
				"[DONE]",
			},
			[]string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}]}`,
				`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}`,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var dst bytes.Buffer
			src := strings.Join(func() []string {
				out := make([]string, 0, len(tc.source))
				for _, s := range tc.source {
					if s == "[DONE]" {
						out = append(out, "data: [DONE]\n\n")
						continue
					}
					out = append(out, "data: "+s+"\n\n")
				}
				return out
			}(), "")
			if _, err := PipeStream(Chat, Gemini, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
				t.Fatal(err)
			}
			got := geminiChatParityChunkData(t, dst.String())
			if len(got) != len(tc.want) {
				t.Fatalf("frames = %d, want %d:\n%s", len(got), len(tc.want), dst.String())
			}
			for i, w := range tc.want {
				if g := geminiChatCanonical(t, got[i]); g != geminiChatCanonical(t, w) {
					t.Fatalf("frame %d\n got  %s\n want %s", i, g, w)
				}
			}
			// The gemini wire has no sentinel of its own.
			if strings.Contains(dst.String(), "[DONE]") {
				t.Fatalf("a sentinel leaked onto the sentinel-less wire: %s", dst.String())
			}
		})
	}
}

// A chat source cut without a finish and without the sentinel leaves the
// gemini stream open: no terminal frame is invented for it.
func TestGeminiChatParityChatToGeminiStreamTruncation(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Gemini, &dst, strings.NewReader(geminiChatParitySSE([]string{
		`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
	})), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	got := geminiChatParityChunkData(t, dst.String())
	if len(got) != 1 || strings.Contains(got[0], "finishReason") {
		t.Fatalf("a cut source must not gain a close: %v", got)
	}
}

// An error frame from a chat source ends the turn with nothing written: the
// host that saw the failure owns the client's error frame.
func TestGeminiChatParityChatToGeminiStreamErrorFrameIsSilent(t *testing.T) {
	var dst bytes.Buffer
	if _, err := PipeStream(Chat, Gemini, &dst, strings.NewReader(geminiChatParitySSE([]string{
		`{"error":{"message":"boom"}}`,
	})), StreamOpts{Model: "m"}); err != nil {
		t.Fatalf("the fold must not fail the pipe: %v", err)
	}
	if dst.Len() != 0 {
		t.Fatalf("error frame produced output: %s", dst.String())
	}
}

// Round-trip: the frames this codec's reverse stream fold emits must restore
// the original chat stream's accumulated content, reasoning, calls, close
// and usage through a gemini->chat fold that is itself reference-exact (the
// reference round-trip ran during this audit through the reference chunk
// parser over the same sources; here the composition runs in-process).
func TestGeminiChatParityStreamRoundTrip(t *testing.T) {
	sources := []string{
		strings.Join([]string{
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hel"}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, ``,
		}, "\n"),
		strings.Join([]string{
			`data: {"id":"c1","choices":[{"index":0,"delta":{"reasoning_content":"why"}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"a"}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, ``,
		}, "\n"),
		strings.Join([]string{
			`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{\"a\": 1}"}}]}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, ``,
		}, "\n"),
		strings.Join([]string{
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}`, ``,
			`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, ``,
		}, "\n"),
	}
	type squash struct {
		content, reasoning, finish string
		calls                      string
		usage                      [3]float64
	}
	fold := func(src string) squash {
		var dst bytes.Buffer
		if _, err := PipeStream(Chat, Gemini, &dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
			t.Fatal(err)
		}
		var back bytes.Buffer
		if _, err := PipeStream(Gemini, Chat, &back, &dst, StreamOpts{Model: "m"}); err != nil {
			t.Fatal(err)
		}
		var s squash
		for _, data := range geminiChatParityChunkData(t, back.String()) {
			if data == "[DONE]" || strings.TrimSpace(data) == "" {
				continue
			}
			var ch map[string]any
			if err := json.Unmarshal([]byte(data), &ch); err != nil {
				t.Fatal(err)
			}
			for _, raw := range ch["choices"].([]any) {
				choice := raw.(map[string]any)
				if d, ok := choice["delta"].(map[string]any); ok {
					if c, ok := d["content"].(string); ok {
						s.content += c
					}
					if r, ok := d["reasoning_content"].(string); ok {
						s.reasoning += r
					}
					if tcs, ok := d["tool_calls"].([]any); ok {
						for _, rawTC := range tcs {
							tc := rawTC.(map[string]any)
							fn := tc["function"].(map[string]any)
							s.calls += fn["name"].(string) + fn["arguments"].(string)
						}
					}
				}
				if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
					s.finish = fr
				}
			}
			if u, ok := ch["usage"].(map[string]any); ok {
				s.usage = [3]float64{u["prompt_tokens"].(float64), u["completion_tokens"].(float64), u["total_tokens"].(float64)}
			}
		}
		return s
	}
	want := []squash{
		{"hello", "", "stop", "", [3]float64{5, 3, 8}},
		{"a", "why", "stop", "", [3]float64{5, 3, 8}},
		{"", "", "tool_calls", "f{\"a\": 1}", [3]float64{5, 3, 8}},
		{"hi", "", "length", "", [3]float64{5, 3, 8}},
	}
	for i, src := range sources {
		if got := fold(src); got != want[i] {
			t.Fatalf("source %d round trip = %+v, want %+v", i, got, want[i])
		}
	}
}

// Grounded turns price their lookups inside the prompt breakdown: the
// request counts the reference's calculator states — distinct non-empty web
// queries (duplicates and empties collapse), queries a maps-attributed item
// reported belong to the maps counter instead, and a maps entry floors at
// one request. All four shapes executed against the reference's response
// adapter during this audit.
func TestGeminiChatParityGroundingUsageCounters(t *testing.T) {
	for _, tc := range []struct{ name, grounding, want string }{
		{"web dedup", `{"webSearchQueries":["q1","q1",""]}`, `"web_search_requests":1`},
		{"web distinct", `{"webSearchQueries":["q1","q2"]}`, `"web_search_requests":2`},
		{"maps chunk", `{"groundingChunks":[{"maps":{"map_title":"t"}}],"webSearchQueries":["q1"]}`, `"google_maps_grounding_requests":1`},
		{"maps token", `{"googleMapsWidgetContextToken":"T1"}`, `"google_maps_grounding_requests":1`},
	} {
		out, err := ConvertResponse(Gemini, Chat, []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]},"finishReason":"STOP","index":0,"groundingMetadata":`+
			tc.grounding+`}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		details := m["usage"].(map[string]any)["prompt_tokens_details"].(map[string]any)
		b, _ := json.Marshal(details)
		if !strings.Contains(string(b), tc.want) {
			t.Fatalf("%s: details = %s, want %s", tc.name, b, tc.want)
		}
	}
}
