package geminichat

import (
	"bytes"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// The folds here are pinned against the reference's own output shapes for the
// same inputs: the request bodies below are the reference's verified samples,
// and the rest are the smallest inputs that expose each rule this package owns
// (the deterministic ids, the refusal set, the stream close conditions).

func fold(t *testing.T, body string, drop bool) map[string]any {
	t.Helper()
	obj, err := jsonx.UnmarshalMap([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	out, err := ChatToGemini(obj, drop)
	if err != nil {
		t.Fatalf("ChatToGemini: %v", err)
	}
	return out
}

func TestChatToGeminiMinimal(t *testing.T) {
	out := fold(t, `{"messages":[{"role":"user","content":"hello"}]}`, false)
	if got := mustJSON(out); got != `{"contents":[{"parts":[{"text":"hello"}],"role":"user"}]}` {
		t.Fatalf("got %s", got)
	}
}

// The reference's own fold for a system turn, sampling parameters, a tool and
// a named tool choice — one sample that touches most of the request surface.
func TestChatToGeminiSystemParamsTools(t *testing.T) {
	out := fold(t, `{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
		"temperature":0.5,"top_p":0.9,"max_tokens":32,"stop":"END",
		"tools":[{"type":"function","function":{"name":"lookup","description":"d",
			"parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],
		"tool_choice":"required"}`, false)
	want := `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}],` +
		`"generationConfig":{"max_output_tokens":32,"stop_sequences":["END"],"temperature":0.5,"top_p":0.9},` +
		`"system_instruction":{"parts":[{"text":"be brief"}]},` +
		`"toolConfig":{"functionCallingConfig":{"mode":"ANY"}},` +
		`"tools":[{"function_declarations":[{"description":"d","name":"lookup","parameters":{"properties":{"q":{"type":"string"}},"type":"object"}}]}]}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

// The parameters the target has no spelling for are refused unless the caller
// asked for drops; everything else in the chat vocabulary rides or drops
// silently.
func TestChatToGeminiRefusedParams(t *testing.T) {
	for _, p := range []string{"logit_bias", "function_call", "metadata", "prediction", "store", "context_management"} {
		body := `{"messages":[{"role":"user","content":"x"}],"` + p + `":1}`
		obj, err := jsonx.UnmarshalMap([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ChatToGemini(obj, false); err == nil {
			t.Errorf("%s: want refusal", p)
		} else if _, ok := err.(cerr.UnsupportedParam); !ok {
			t.Errorf("%s: want cerr.UnsupportedParam, got %T (%v)", p, err, err)
		}
		if out := fold(t, body, true); mustJSON(out) != `{"contents":[{"parts":[{"text":"x"}],"role":"user"}]}` {
			t.Errorf("%s: drop path body = %s", p, mustJSON(out))
		}
	}
}

// A tool-call history folds to function_call and function_response parts, with
// the result paired to the call by its id.
func TestChatToGeminiToolHistory(t *testing.T) {
	out := fold(t, `{"messages":[{"role":"user","content":"weather?"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
			"function":{"name":"get_weather","arguments":"{\"city\": \"Boston\"}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"sunny"}]}`, false)
	want := `{"contents":[{"parts":[{"text":"weather?"}],"role":"user"},` +
		`{"parts":[{"function_call":{"args":{"city":"Boston"},"name":"get_weather"},"thoughtSignature":"skip_thought_signature_validator"}],"role":"model"},` +
		`{"parts":[{"function_response":{"name":"get_weather","response":{"content":"sunny"}}}],"role":"user"}]}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

// A replayed tool call the client preserved no signature for must still carry
// one onto the target: the upstream refuses a replayed functionCall part
// without a thoughtSignature (INVALID_ARGUMENT, "Function call is missing a
// thought_signature in functionCall parts") and validates the value, so the
// fallback is the upstream's own skip marker. Observed 2026-10-15 on the
// /v1/responses tool loop; the chat wire replays through the same fold.
func TestChatToGeminiUnsignedCallTakesSkipMarker(t *testing.T) {
	out := fold(t, `{"messages":[{"role":"user","content":"look up"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
			"function":{"name":"default_api:bash","arguments":"{\"command\": \"ls\"}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"}]}`, false)
	want := `{"contents":[{"parts":[{"text":"look up"}],"role":"user"},` +
		`{"parts":[{"function_call":{"args":{"command":"ls"},"name":"default_api:bash"},"thoughtSignature":"skip_thought_signature_validator"}],"role":"model"},` +
		`{"parts":[{"function_response":{"name":"default_api:bash","response":{"content":"ok"}}}],"role":"user"}]}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

// A signature the client DID preserve rides verbatim — on the call, on its
// function, or as the call id's suffix. The skip marker is the fallback.
func TestChatToGeminiSignedCallRidesVerbatim(t *testing.T) {
	for _, tc := range []struct{ name, call string }{
		{"call field", `"provider_specific_fields":{"thought_signature":"CSIG"}`},
		{"function field", `"function":{"name":"f","arguments":"{\"a\": 1}","provider_specific_fields":{"thought_signature":"CSIG"}}`},
	} {
		call := tc.call
		if !strings.HasPrefix(call, `"function"`) {
			call += `,"function":{"name":"f","arguments":"{\"a\": 1}"}`
		}
		body := `{"messages":[{"role":"user","content":"q"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",` + call + `}]}]}`
		out := fold(t, body, false)
		got := mustJSON(out)
		if !strings.Contains(got, `"thoughtSignature":"CSIG"`) {
			t.Errorf("%s: signature not carried verbatim: %s", tc.name, got)
		}
		if strings.Contains(got, "skip_thought_signature_validator") {
			t.Errorf("%s: skip marker must not override a real signature: %s", tc.name, got)
		}
	}
	// The id suffix the chat wire allows.
	out := fold(t, `{"messages":[{"role":"user","content":"q"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1__thought__SIG","type":"function",
			"function":{"name":"f","arguments":"{\"a\": 1}"}}]}]}`, false)
	if got := mustJSON(out); !strings.Contains(got, `"thoughtSignature":"SIG"`) {
		t.Errorf("id suffix: signature not carried verbatim: %s", got)
	}
}

// Google's own wire spells the thoughtSignature member BOTH ways — the
// plugin's reader says "thoughtSignature comes and goes" — and it keeps both
// keys in the map. The reverse fold must lift the signature off the call id in
// either spelling, or the client replays an unsigned call and the upstream
// refuses the next request.
func TestGeminiToChatSignatureReadsEitherSpelling(t *testing.T) {
	for _, tc := range []struct{ name, member string }{
		{"camel", `"thoughtSignature":"SIG"`},
		{"snake", `"thought_signature":"SIG"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			obj, err := jsonx.UnmarshalMap([]byte(
				`{"contents":[{"role":"user","parts":[{"text":"q"}]},` +
					`{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}},` + tc.member + `}]},` +
					`{"role":"user","parts":[{"function_response":{"name":"f","response":{"content":"r"}}}]}]}`))
			if err != nil {
				t.Fatal(err)
			}
			out, err := GeminiToChat(obj)
			if err != nil {
				t.Fatal(err)
			}
			got := mustJSON(out)
			if !strings.Contains(got, thoughtSignatureSeparator+"SIG") {
				t.Errorf("the signature was dropped: %s", got)
			}
		})
	}
}

// A media-only user turn gains the blank text part the target requires, and a
// data URL folds to inline bytes.
func TestChatToGeminiMediaBlankText(t *testing.T) {
	out := fold(t, `{"messages":[{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAB"}}]}]}`, false)
	want := `{"contents":[{"parts":[{"inline_data":{"data":"AAAB","mime_type":"image/png"}},{"text":" "}],"role":"user"}]}`
	if got := mustJSON(out); got != want {
		t.Fatalf("got %s", got)
	}
}

// The reverse request fold pairs a function response with the call it answers,
// mints positional ids where the wire named none, and carries the tool
// declarations back as chat tools.
func TestGeminiToChatToolRoundTrip(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"contents":[{"role":"user","parts":[{"text":"q"}]},
			{"role":"model","parts":[{"function_call":{"name":"f","args":{"a":1}}}]},
			{"role":"user","parts":[{"function_response":{"name":"f","response":{"a":1}}}]}],
		"tools":[{"function_declarations":[{"name":"f","parameters":{"type":"object"}}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowed_function_names":["f"]}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := GeminiToChat(obj)
	if err != nil {
		t.Fatal(err)
	}
	msgs, _ := jsonx.AsSlice(out["messages"])
	if len(msgs) != 3 {
		t.Fatalf("messages = %s", mustJSON(out))
	}
	assistant, _ := jsonx.AsMap(msgs[1])
	calls, _ := jsonx.AsSlice(assistant["tool_calls"])
	call, _ := jsonx.AsMap(calls[0])
	tool, _ := jsonx.AsMap(msgs[2])
	if jsonx.GetString(call, "id") == "" || jsonx.GetString(tool, "tool_call_id") != jsonx.GetString(call, "id") {
		t.Fatalf("call/result ids do not pair: %s", mustJSON(out))
	}
	if _, ok := out["tools"]; !ok {
		t.Fatalf("tools lost: %s", mustJSON(out))
	}
	if tc, _ := jsonx.AsMap(out["tool_choice"]); jsonx.GetString(tc, "type") != "function" {
		t.Fatalf("tool_choice = %s", mustJSON(out["tool_choice"]))
	}
	// Folding the chat body back must restore the parts the reference reads.
	again, err := ChatToGemini(out, false)
	if err != nil {
		t.Fatal(err)
	}
	contents, _ := jsonx.AsSlice(again["contents"])
	model, _ := jsonx.AsMap(contents[1])
	parts, _ := jsonx.AsSlice(model["parts"])
	if len(parts) != 1 {
		t.Fatalf("round trip lost the call part: %s", mustJSON(model))
	}
}

// A text response folds onto the chat shape with the deterministic id and no
// clock, and the usage keeps the reference's token arithmetic.
func TestGeminiResponseToChat(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"candidates":[{"content":{"role":"model","parts":[{"text":"hello there"}]},"finishReason":"STOP","index":0}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
		"responseId":"resp-1"}`))
	if err != nil {
		t.Fatal(err)
	}
	out := GeminiResponseToChat(obj)
	if jsonx.GetString(out, "id") != "resp-1" || out["created"] != 0 || out["model"] != "" {
		t.Fatalf("envelope = %s", mustJSON(out))
	}
	choices, _ := jsonx.AsSlice(out["choices"])
	choice, _ := jsonx.AsMap(choices[0])
	if jsonx.GetString(choice, "finish_reason") != "stop" {
		t.Fatalf("finish = %s", mustJSON(choice))
	}
	msg, _ := jsonx.AsMap(choice["message"])
	if jsonx.GetString(msg, "content") != "hello there" {
		t.Fatalf("message = %s", mustJSON(msg))
	}
	usage, _ := jsonx.AsMap(out["usage"])
	if n, _ := jsonx.Int(usage["completion_tokens"]); n != 3 {
		t.Fatalf("usage = %s", mustJSON(usage))
	}
}

// A candidate the safety layer refused collapses the response to one filtered
// choice, with the three headline counts only.
func TestGeminiResponseToChatBlocked(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"promptFeedback":{"blockReason":"SAFETY"},
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`))
	if err != nil {
		t.Fatal(err)
	}
	out := GeminiResponseToChat(obj)
	choices, _ := jsonx.AsSlice(out["choices"])
	if len(choices) != 1 {
		t.Fatalf("choices = %s", mustJSON(out))
	}
	choice, _ := jsonx.AsMap(choices[0])
	if jsonx.GetString(choice, "finish_reason") != "content_filter" {
		t.Fatalf("choice = %s", mustJSON(choice))
	}
	usage, _ := jsonx.AsMap(out["usage"])
	if _, has := usage["prompt_tokens_details"]; usage["prompt_tokens_details"] != nil || !has {
		t.Fatalf("usage = %s", mustJSON(usage))
	}
}

// The reverse response fold recomputes the token total so the inclusive-or-
// exclusive arithmetic reads back the counts the chat body started from.
func TestChatResponseToGeminiUsage(t *testing.T) {
	obj, err := jsonx.UnmarshalMap([]byte(`{
		"id":"c1","object":"chat.completion",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],
		"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":10,
			"completion_tokens_details":{"reasoning_tokens":2}}}`))
	if err != nil {
		t.Fatal(err)
	}
	out := ChatResponseToGemini(obj)
	um, _ := jsonx.AsMap(out["usageMetadata"])
	if n, _ := jsonx.Int(um["candidatesTokenCount"]); n != 1 {
		t.Fatalf("usageMetadata = %s", mustJSON(um))
	}
	if n, _ := jsonx.Int(um["thoughtsTokenCount"]); n != 2 {
		t.Fatalf("usageMetadata = %s", mustJSON(um))
	}
	if n, _ := jsonx.Int(um["totalTokenCount"]); n != 8 {
		t.Fatalf("usageMetadata = %s", mustJSON(um))
	}
	if jsonx.GetString(out, "responseId") != "c1" {
		t.Fatalf("responseId = %s", mustJSON(out))
	}
}

// A gemini frame folds to one chat chunk; the frame that carries the wire's
// own close is the one that names a finish reason, and the usage rides it.
func TestPipeGeminiToChat(t *testing.T) {
	src := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}],"responseId":"r1"}`,
		``,
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"r1"}`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeGeminiToChat(&dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `"content":"hel"`) || !strings.Contains(out, `"content":"lo"`) {
		t.Fatalf("content chunks lost:\n%s", out)
	}
	if n := strings.Count(out, `"finish_reason":"`); n != 1 {
		t.Fatalf("finish_reason chunks = %d, want the terminal only:\n%s", n, out)
	}
	if !strings.Contains(out, `"usage"`) {
		t.Fatalf("terminal lost its usage:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimSpace(out), "data: [DONE]") {
		t.Fatalf("stream not closed with the chat sentinel:\n%s", out)
	}
}

// A source that stops without a finish reason and without the relay sentinel
// was cut: the fold leaves the chat stream open rather than invent an ending.
func TestPipeGeminiToChatTruncation(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"par\"}]},\"index\":0}]}\n\n"
	var dst bytes.Buffer
	if _, err := PipeGeminiToChat(&dst, strings.NewReader(src), StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(dst.String(), `"finish_reason":"`) || strings.Contains(dst.String(), "[DONE]") {
		t.Fatalf("truncated source was closed for the client:\n%s", dst.String())
	}
	if !strings.Contains(dst.String(), `"content":"par"`) {
		t.Fatalf("delivered content lost:\n%s", dst.String())
	}
}

// Chat chunks fold onto the target's frames: content deltas become text parts,
// and the frame that closes the turn carries the finish reason with the usage.
func TestPipeChatToGemini(t *testing.T) {
	src := strings.Join([]string{
		`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		``,
		`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var dst bytes.Buffer
	if _, err := PipeChatToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	out := dst.String()
	if !strings.Contains(out, `{"text":"hel"}`) || !strings.Contains(out, `{"text":"lo"}`) {
		t.Fatalf("text frames lost:\n%s", out)
	}
	if !strings.Contains(out, `"finishReason":"STOP"`) {
		t.Fatalf("terminal frame lost:\n%s", out)
	}
	if !strings.Contains(out, `"usageMetadata"`) {
		t.Fatalf("terminal lost its usage:\n%s", out)
	}
	if strings.Contains(out, "[DONE]") {
		t.Fatalf("the target wire has no sentinel:\n%s", out)
	}
}

// A chat source that never said how it ended is not closed for the client.
func TestPipeChatToGeminiTruncation(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n"
	var dst bytes.Buffer
	if _, err := PipeChatToGemini(&dst, strings.NewReader(src), StreamOpts{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(dst.String(), "finishReason") {
		t.Fatalf("truncated source was closed for the client:\n%s", dst.String())
	}
	if !strings.Contains(dst.String(), `{"text":"par"}`) {
		t.Fatalf("delivered content lost:\n%s", dst.String())
	}
}
