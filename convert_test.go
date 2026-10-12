package protocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestPassthroughKeepsBytes(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0}`)
	got, err := ConvertRequest(Chat, Chat, body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Body, body) {
		t.Fatalf("passthrough re-encoded body")
	}
	if got.Model != "gpt-4o" {
		t.Fatalf("model=%s", got.Model)
	}
	out, err := ConvertResponse(Messages, Messages, body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, body) {
		t.Fatalf("response passthrough re-encoded")
	}
}

func TestNormalizeAliases(t *testing.T) {
	d, ok := Normalize("openai")
	if !ok || d != Chat {
		t.Fatalf("openai -> %s", d)
	}
	d, ok = Normalize("anthropic")
	if !ok || d != Messages {
		t.Fatalf("anthropic -> %s", d)
	}
}

func TestChatToMessagesPlainText(t *testing.T) {
	in := jsonMap{
		"model": "claude-sonnet-4-5",
		"messages": []any{
			map[string]any{"role": "system", "content": "be brief"},
			map[string]any{"role": "user", "content": "hello"},
		},
		"max_tokens":  32,
		"temperature": 0.2,
	}
	out := convertReq(t, Chat, Messages, in)
	if out["model"] != "claude-sonnet-4-5" {
		t.Fatalf("model=%v", out["model"])
	}
	if out["system"] != "be brief" {
		t.Fatalf("system=%v", out["system"])
	}
	msgs := out["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("msgs=%v", msgs)
	}
	if jsonMap(msgs[0].(map[string]any))["role"] != "user" {
		t.Fatalf("role")
	}
}

func TestMessagesToChatToolsAndImage(t *testing.T) {
	in := jsonMap{
		"model":  "gpt-4o",
		"system": "s",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "look"},
					map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": "https://x/a.png"}},
				},
			},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": map[string]any{"q": "a"}},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
				},
			},
		},
		"tools": []any{
			map[string]any{"name": "lookup", "description": "d", "input_schema": map[string]any{"type": "object"}},
		},
		"tool_choice":    map[string]any{"type": "any"},
		"stop_sequences": []any{"END"},
		"thinking":       map[string]any{"type": "enabled", "budget_tokens": 4096},
	}
	out := convertReq(t, Messages, Chat, in)
	if out["user"] != nil {
		t.Fatalf("unexpected user")
	}
	if out["stop"].([]any)[0] != "END" {
		t.Fatalf("stop")
	}
	if out["reasoning_effort"] != "high" {
		t.Fatalf("effort=%v", out["reasoning_effort"])
	}
	if out["tool_choice"] != "required" {
		t.Fatalf("tool_choice=%v", out["tool_choice"])
	}
	msgs := out["messages"].([]any)
	if jsonMap(msgs[0].(map[string]any))["role"] != "system" {
		t.Fatalf("system first")
	}
	foundTool := false
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] == "tool" {
			foundTool = true
			if mm["tool_call_id"] != "toolu_1" {
				t.Fatalf("tool_call_id")
			}
		}
		if mm["role"] == "assistant" {
			tcs := mm["tool_calls"].([]any)
			fn := tcs[0].(map[string]any)["function"].(map[string]any)
			if fn["name"] != "lookup" {
				t.Fatalf("name=%v", fn["name"])
			}
			if fn["arguments"] != `{"q":"a"}` && fn["arguments"] != `{"q": "a"}` {
				t.Fatalf("args=%v", fn["arguments"])
			}
		}
	}
	if !foundTool {
		t.Fatal("missing tool message")
	}
}

func TestChatToMessagesRejectsUnsupportedParam(t *testing.T) {
	in := jsonMap{
		"model":      "claude-3",
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"logit_bias": map[string]any{"1": 1},
	}
	_, err := ConvertRequest(Chat, Messages, mustJSON(in))
	if err == nil {
		t.Fatal("expected error")
	}
	var u UnsupportedParamError
	if !asUnsupported(err, &u) {
		t.Fatalf("err=%v", err)
	}
	if u.Param != "logit_bias" {
		t.Fatalf("param=%s", u.Param)
	}
}

func TestChatToMessagesToolCalls(t *testing.T) {
	in := jsonMap{
		"model": "claude-3",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{map[string]any{
					"id": "call_1", "type": "function",
					"function": map[string]any{"name": "sum", "arguments": `{"a":1}`},
				}},
			},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "2"},
		},
		"tools": []any{map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "sum", "parameters": map[string]any{"type": "object"},
			},
		}},
		"tool_choice":         "auto",
		"parallel_tool_calls": false,
		"max_tokens":          16,
	}
	out := convertReq(t, Chat, Messages, in)
	tc := out["tool_choice"].(map[string]any)
	if tc["type"] != "auto" {
		t.Fatalf("%v", tc)
	}
	if tc["disable_parallel_tool_use"] != true {
		t.Fatalf("parallel %v", tc)
	}
	tools := out["tools"].([]any)
	if tools[0].(map[string]any)["name"] != "sum" {
		t.Fatalf("tools=%v", tools)
	}
}

func TestMessagesToResponsesMapping(t *testing.T) {
	in := jsonMap{
		"model":      "gpt-4o",
		"system":     "sys",
		"max_tokens": 100,
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "thinking", "thinking": "hmm"},
				map[string]any{"type": "text", "text": "yo"},
				map[string]any{"type": "tool_use", "id": "t1", "name": "fn", "input": map[string]any{"x": 1}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
			}},
		},
		"tools":          []any{map[string]any{"name": "fn", "input_schema": map[string]any{"type": "object"}}},
		"tool_choice":    map[string]any{"type": "any"},
		"thinking":       map[string]any{"type": "enabled", "budget_tokens": 5000},
		"stop_sequences": []any{"NOPE"},
		"top_k":          5,
		"metadata":       map[string]any{"user_id": "abc"},
		"output_format":  map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}},
	}
	out := convertReq(t, Messages, Responses, in)
	if _, ok := out["stop_sequences"]; ok {
		t.Fatal("stop_sequences should be dropped")
	}
	if _, ok := out["top_k"]; ok {
		t.Fatal("top_k should be dropped")
	}
	if out["instructions"] != "sys" {
		t.Fatalf("instructions=%v", out["instructions"])
	}
	if out["max_output_tokens"] != json.Number("100") && fmtInt(out["max_output_tokens"]) != 100 {
		t.Fatalf("max_output_tokens=%T %v", out["max_output_tokens"], out["max_output_tokens"])
	}
	if out["tool_choice"] != "required" {
		t.Fatalf("tool_choice=%v", out["tool_choice"])
	}
	if out["user"] != "abc" || out["prompt_cache_key"] != "abc" {
		t.Fatalf("user=%v", out["user"])
	}
	reasoning := out["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" { // 5000 >= 4096
		t.Fatalf("effort=%v", reasoning["effort"])
	}
	if reasoning["summary"] != "detailed" {
		t.Fatalf("summary")
	}
	text := out["text"].(map[string]any)["format"].(map[string]any)
	if text["name"] != "structured_output" {
		t.Fatalf("text=%v", text)
	}
}

func TestChatResponsesRoundTripPlain(t *testing.T) {
	in := jsonMap{
		"model": "gpt-4o",
		"messages": []any{
			map[string]any{"role": "system", "content": "s"},
			map[string]any{"role": "user", "content": "hi"},
		},
		"max_tokens": 10,
	}
	mid := convertReq(t, Chat, Responses, in)
	if mid["instructions"] != "s" {
		t.Fatalf("instructions=%v", mid["instructions"])
	}
	back := convertReq(t, Responses, Chat, mid)
	msgs := back["messages"].([]any)
	if jsonMap(msgs[0].(map[string]any))["role"] != "system" {
		t.Fatalf("first=%v", msgs[0])
	}
}

func TestChatResponseToMessages(t *testing.T) {
	in := jsonMap{
		"id":    "chatcmpl-1",
		"model": "gpt-4o",
		"choices": []any{map[string]any{
			"finish_reason": "tool_calls",
			"message": map[string]any{
				"role":              "assistant",
				"content":           "hi",
				"reasoning_content": "think",
				"tool_calls": []any{map[string]any{
					"id": "c1", "type": "function",
					"function": map[string]any{"name": "fn", "arguments": `{"a":1}`},
				}},
			},
		}},
		"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
	}
	out := convertResp(t, Chat, Messages, in)
	if out["stop_reason"] != "tool_use" {
		t.Fatalf("stop=%v", out["stop_reason"])
	}
	content := out["content"].([]any)
	types := []string{}
	for _, c := range content {
		types = append(types, c.(map[string]any)["type"].(string))
	}
	if strings.Join(types, ",") != "thinking,text,tool_use" {
		t.Fatalf("content types=%s", strings.Join(types, ","))
	}
}

func TestResponsesResponseToMessages(t *testing.T) {
	in := jsonMap{
		"id":     "resp_1",
		"model":  "gpt-4o",
		"status": "incomplete",
		"output": []any{
			map[string]any{"type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": "why"}}},
			map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "hi"}}},
			map[string]any{"type": "function_call", "call_id": "c1", "name": "fn", "arguments": `{"z":true}`},
		},
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 2},
	}
	out := convertResp(t, Responses, Messages, in)
	if out["stop_reason"] != "max_tokens" {
		t.Fatalf("stop=%v", out["stop_reason"])
	}
}

type jsonMap = map[string]any

func convertReq(t *testing.T, from, to Dialect, in jsonMap) jsonMap {
	t.Helper()
	got, err := ConvertRequest(from, to, mustJSON(in))
	if err != nil {
		t.Fatal(err)
	}
	m, err := unmarshal(got.Body)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func convertResp(t *testing.T, from, to Dialect, in jsonMap) jsonMap {
	t.Helper()
	got, err := ConvertResponse(from, to, mustJSON(in))
	if err != nil {
		t.Fatal(err)
	}
	m, err := unmarshal(got)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func unmarshal(b []byte) (jsonMap, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var m jsonMap
	err := dec.Decode(&m)
	return m, err
}

func asUnsupported(err error, dest *UnsupportedParamError) bool {
	e, ok := err.(UnsupportedParamError)
	if !ok {
		return false
	}
	*dest = e
	return true
}

func fmtInt(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case float64:
		return int(t)
	case json.Number:
		i, _ := t.Int64()
		return int(i)
	default:
		return 0
	}
}

func TestChatToMessagesImage(t *testing.T) {
	in := jsonMap{
		"model": "claude-3",
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "see"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x/a.png"}},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,abc"}},
			},
		}},
		"max_tokens": 8,
	}
	out := convertReq(t, Chat, Messages, in)
	content := out["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if content[1].(map[string]any)["source"].(map[string]any)["type"] != "url" {
		t.Fatalf("%v", content[1])
	}
	if content[2].(map[string]any)["source"].(map[string]any)["type"] != "base64" {
		t.Fatalf("%v", content[2])
	}
}

func TestChatToResponsesStructuredOutput(t *testing.T) {
	in := jsonMap{
		"model":    "gpt-4o",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "item",
				"schema": map[string]any{"type": "object"},
				"strict": true,
			},
		},
		"reasoning_effort": "low",
	}
	out := convertReq(t, Chat, Responses, in)
	format := out["text"].(map[string]any)["format"].(map[string]any)
	if format["name"] != "item" || format["strict"] != true {
		t.Fatalf("%v", format)
	}
	if out["reasoning"].(map[string]any)["effort"] != "low" {
		t.Fatalf("%v", out["reasoning"])
	}
}

func TestResponsesToChatReasoningEffortStaysScalar(t *testing.T) {
	// OpenAI Responses clients (pi-ai among them) always set a summary next to
	// the effort. The chat wire has no summary and reasoning_effort is a string,
	// so the fold must carry the effort scalar and nothing else — an object
	// here is what made OpenRouter answer "reasoning_effort: Invalid option".
	in := jsonMap{
		"model":             "m",
		"input":             "hi",
		"max_output_tokens": 8,
		"reasoning":         map[string]any{"effort": "high", "summary": "auto"},
	}
	out := convertReq(t, Responses, Chat, in)
	if out["reasoning_effort"] != "high" {
		t.Fatalf("effort=%v (%T)", out["reasoning_effort"], out["reasoning_effort"])
	}

	// No effort: nothing to express on the chat wire, summary or not.
	delete(in, "effort")
	in["reasoning"] = map[string]any{"summary": "auto"}
	out = convertReq(t, Responses, Chat, in)
	if _, ok := out["reasoning_effort"]; ok {
		t.Fatalf("effort=%v", out["reasoning_effort"])
	}
}

func TestResponsesToMessagesThinkingDropsSummary(t *testing.T) {
	// pi-ai and other Responses clients always set a summary next to the
	// effort. The Messages thinking object has no summary member, so carrying
	// it draws "thinking.enabled.summary: Extra inputs are not permitted" from
	// an Anthropic-protocol upstream. The fold must emit type+budget_tokens
	// only, whatever the source set.
	in := jsonMap{
		"model":             "m",
		"input":             "hi",
		"max_output_tokens": 8,
		"reasoning":         map[string]any{"effort": "high", "summary": "auto"},
	}
	out := convertReq(t, Responses, Messages, in)
	thinking, ok := out["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking=%v", out["thinking"])
	}
	if _, has := thinking["summary"]; has {
		t.Fatalf("summary leaked into Messages thinking: %v", thinking)
	}
	if thinking["type"] != "enabled" {
		t.Fatalf("type=%v", thinking["type"])
	}
	if _, has := thinking["budget_tokens"]; !has {
		t.Fatalf("budget_tokens missing: %v", thinking)
	}
}

func TestMessagesResponseToChat(t *testing.T) {
	in := jsonMap{
		"id":    "msg_1",
		"model": "claude",
		"type":  "message",
		"role":  "assistant",
		"content": []any{
			map[string]any{"type": "thinking", "thinking": "t", "signature": "sig"},
			map[string]any{"type": "text", "text": "hello"},
		},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 2, "output_tokens": 3},
	}
	out := convertResp(t, Messages, Chat, in)
	msg := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "hello" {
		t.Fatalf("%v", msg["content"])
	}
	if msg["reasoning_content"] != "t" {
		t.Fatalf("%v", msg["reasoning_content"])
	}
	if out["usage"].(map[string]any)["total_tokens"] != json.Number("5") && fmtInt(out["usage"].(map[string]any)["total_tokens"]) != 5 {
		t.Fatalf("usage=%v", out["usage"])
	}
}

func TestGatewayAliases(t *testing.T) {
	in := jsonMap{
		"model":      "m",
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"max_tokens": 4,
	}
	got, err := ConvertRequest("openai", "anthropic", mustJSON(in))
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "m" {
		t.Fatalf("%s", got.Model)
	}
}

func TestChatReasoningEffortToThinking(t *testing.T) {
	in := jsonMap{
		"model":            "claude-3-7-sonnet",
		"messages":         []any{map[string]any{"role": "user", "content": "hi"}},
		"max_tokens":       100,
		"reasoning_effort": "medium",
	}
	out := convertReq(t, Chat, Messages, in)
	th := out["thinking"].(map[string]any)
	if th["type"] != "enabled" {
		t.Fatalf("%v", th)
	}
	if fmtInt(th["budget_tokens"]) != 2048 {
		t.Fatalf("budget=%v", th["budget_tokens"])
	}
}

func TestContextManagementMessagesToResponses(t *testing.T) {
	in := jsonMap{
		"model":      "gpt-4o",
		"max_tokens": 8,
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"context_management": map[string]any{
			"edits": []any{map[string]any{
				"type":    "compact_20260112",
				"trigger": map[string]any{"type": "input_tokens", "value": 150000},
			}},
		},
	}
	out := convertReq(t, Messages, Responses, in)
	cm := out["context_management"].([]any)[0].(map[string]any)
	if cm["type"] != "compaction" || fmtInt(cm["compact_threshold"]) != 150000 {
		t.Fatalf("%v", cm)
	}
}

func TestMessagesFoldRequiresMaxTokens(t *testing.T) {
	chat := []byte(`{"model":"some-model","messages":[{"role":"user","content":"hi"}]}`)

	// No fallback: the conversion names the missing field instead of emitting
	// a body the Messages upstream would reject.
	_, err := ConvertRequest(Chat, Messages, chat)
	var missing MissingRequiredFieldError
	if !errors.As(err, &missing) || missing.Field != "max_tokens" || missing.To != Messages {
		t.Fatalf("err = %v, want MissingRequiredFieldError{max_tokens, messages}", err)
	}

	// A host-supplied value fills the field.
	out, err := ConvertRequestWith(Chat, Messages, chat, ConvertOptions{MaxTokensFallback: 4096})
	if err != nil {
		t.Fatalf("with fallback: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body["max_tokens"] != float64(4096) {
		t.Fatalf("max_tokens = %v, want the host-supplied 4096", body["max_tokens"])
	}

	// A client-supplied value always wins over the fallback.
	withTokens := []byte(`{"model":"some-model","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}`)
	out, err = ConvertRequestWith(Chat, Messages, withTokens, ConvertOptions{MaxTokensFallback: 4096})
	if err != nil {
		t.Fatalf("client value: %v", err)
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if body["max_tokens"] != float64(512) {
		t.Fatalf("max_tokens = %v, want the client's 512", body["max_tokens"])
	}

	// Conversions not targeting Messages are untouched by the option.
	if _, err := ConvertRequest(Messages, Chat, []byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)); err != nil {
		t.Fatalf("messages->chat: %v", err)
	}

	// The Responses fold answers the same way.
	responses := []byte(`{"model":"some-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	if _, err := ConvertRequest(Responses, Messages, responses); !errors.As(err, &missing) {
		t.Fatalf("responses->messages without fallback: err = %v", err)
	}
}

func TestResponsesToChatMergesReasoningAndToolCalls(t *testing.T) {
	// The shape DSH actually replays: the reasoning item carries its text under
	// summary[].text (not a flat content string), and the function_call that
	// belongs to the same turn follows it as its own input item.
	in := jsonMap{
		"model": "deepseek-v4.1-flash",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "hi"},
			map[string]any{
				"type":    "reasoning",
				"id":      "rs_1",
				"status":  "completed",
				"summary": []any{map[string]any{"type": "summary_text", "text": "thinking about tools"}},
			},
			map[string]any{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "query",
				"arguments": "{\"q\":\"abc\"}",
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_1",
				"output":  "result_1",
			},
		},
	}
	out := convertReq(t, Responses, Chat, in)
	messages := out["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages (user, assistant, tool), got %d: %v", len(messages), messages)
	}
	assistant := messages[1].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("assistant role = %v", assistant["role"])
	}
	if assistant["reasoning_content"] != "thinking about tools" {
		t.Fatalf("reasoning_content = %v", assistant["reasoning_content"])
	}
	toolCalls, ok := assistant["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %v", assistant["tool_calls"])
	}
	call := toolCalls[0].(map[string]any)
	if call["id"] != "call_1" {
		t.Fatalf("call_id = %v", call["id"])
	}
}

func TestResponsesToChatJoinsMultipleReasoningItems(t *testing.T) {
	// Two reasoning items in one turn must land on ONE assistant message: a
	// second one that opened its own message would strand the first with no
	// tool_calls, which is the shape the upstream rejects.
	in := jsonMap{
		"model": "deepseek-v4.1-flash",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "hi"},
			map[string]any{"type": "reasoning", "content": "first thought"},
			map[string]any{"type": "reasoning", "content": "second thought"},
			map[string]any{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "query",
				"arguments": "{}",
			},
		},
	}
	out := convertReq(t, Responses, Chat, in)
	messages := out["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected user + one assistant message, got %d: %v", len(messages), messages)
	}
	assistant := messages[1].(map[string]any)
	if assistant["reasoning_content"] != "first thought\nsecond thought" {
		t.Fatalf("reasoning_content = %v", assistant["reasoning_content"])
	}
	if calls, ok := assistant["tool_calls"].([]any); !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %v", assistant["tool_calls"])
	}
}

func TestResponsesToChatKeepsDistinctAssistantText(t *testing.T) {
	// Two assistant messages that both carry text are two turns. Folding the
	// second into the first would silently drop its content.
	in := jsonMap{
		"model": "m",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "hi"},
			map[string]any{"type": "message", "role": "assistant", "content": "first answer"},
			map[string]any{"type": "message", "role": "user", "content": "again"},
			map[string]any{"type": "message", "role": "assistant", "content": "second answer"},
		},
	}
	out := convertReq(t, Responses, Chat, in)
	messages := out["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d: %v", len(messages), messages)
	}
	if messages[1].(map[string]any)["content"] != "first answer" {
		t.Fatalf("first assistant = %v", messages[1])
	}
	if messages[3].(map[string]any)["content"] != "second answer" {
		t.Fatalf("second assistant = %v", messages[3])
	}
}

func TestResponsesToChatKeepsTurnsSeparatedByUser(t *testing.T) {
	// A user message ends a turn: the next reasoning item belongs to the NEXT
	// assistant message, not to the one before the user.
	in := jsonMap{
		"model": "m",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "one"},
			map[string]any{"type": "message", "role": "assistant", "content": "answer one"},
			map[string]any{"type": "message", "role": "user", "content": "two"},
			map[string]any{"type": "reasoning", "content": "second turn thought"},
			map[string]any{
				"type":      "function_call",
				"call_id":   "call_2",
				"name":      "query",
				"arguments": "{}",
			},
		},
	}
	out := convertReq(t, Responses, Chat, in)
	messages := out["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d: %v", len(messages), messages)
	}
	if _, carried := messages[1].(map[string]any)["reasoning_content"]; carried {
		t.Fatalf("first assistant must not absorb the next turn's reasoning: %v", messages[1])
	}
	last := messages[3].(map[string]any)
	if last["reasoning_content"] != "second turn thought" {
		t.Fatalf("last assistant reasoning = %v", last["reasoning_content"])
	}
	if calls, ok := last["tool_calls"].([]any); !ok || len(calls) != 1 {
		t.Fatalf("last assistant tool_calls = %v", last["tool_calls"])
	}
}

// A tool_result that returned an image must reach a chat upstream as a content
// part, not as the Anthropic image block. The chat vocabulary has no "image"
// content type, so forwarding it verbatim makes an upstream that parses that
// vocabulary reject the whole request — and the tool result stays in history,
// so every later request fails too.
func TestMessagesToChatToolResultImage(t *testing.T) {
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "tool_use", "id": "t1", "name": "read", "input": map[string]any{}},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": []any{
						map[string]any{"type": "image", "source": map[string]any{
							"type": "base64", "media_type": "image/png", "data": "QUJD",
						}},
					}},
				},
			},
		},
	}
	out := convertReq(t, Messages, Chat, in)
	tool := out["messages"].([]any)[1].(map[string]any)
	if tool["role"] != "tool" {
		t.Fatalf("role = %v, want tool", tool["role"])
	}
	parts, ok := tool["content"].([]any)
	if !ok {
		t.Fatalf("content = %#v, want a content-part array", tool["content"])
	}
	if len(parts) != 1 {
		t.Fatalf("parts = %#v", parts)
	}
	part := parts[0].(map[string]any)
	if part["type"] != "image_url" {
		t.Fatalf("part type = %v, want image_url", part["type"])
	}
	if got := part["image_url"].(map[string]any)["url"]; got != "data:image/png;base64,QUJD" {
		t.Fatalf("url = %v", got)
	}
}

// A tool_result that returned only text keeps the flat string every chat
// upstream already accepted, so prompt prefixes stay stable.
func TestMessagesToChatToolResultTextStaysString(t *testing.T) {
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "tool_use", "id": "t1", "name": "read", "input": map[string]any{}},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": []any{
						map[string]any{"type": "text", "text": "first"},
						map[string]any{"type": "text", "text": "second"},
					}},
				},
			},
		},
	}
	out := convertReq(t, Messages, Chat, in)
	tool := out["messages"].([]any)[1].(map[string]any)
	if got, ok := tool["content"].(string); !ok || got != "first\nsecond" {
		t.Fatalf("content = %#v, want the joined string", tool["content"])
	}
}

// Text beside an image survives: the media is added, not substituted for the
// text the tool also returned.
func TestMessagesToChatToolResultImageKeepsText(t *testing.T) {
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "tool_use", "id": "t1", "name": "read", "input": map[string]any{}},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": []any{
						map[string]any{"type": "text", "text": "icon.png is 1024x1024"},
						map[string]any{"type": "image", "source": map[string]any{
							"type": "url", "url": "https://x/a.png",
						}},
					}},
				},
			},
		},
	}
	out := convertReq(t, Messages, Chat, in)
	parts := out["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0].(map[string]any)["text"] != "icon.png is 1024x1024" {
		t.Fatalf("text lost: %#v", parts[0])
	}
	if got := parts[1].(map[string]any)["image_url"].(map[string]any)["url"]; got != "https://x/a.png" {
		t.Fatalf("url = %v", got)
	}
}

// A tool_choice naming a tool must reference the name the request actually
// declares. Tool declarations are truncated to OpenAI's 64-char limit, so the
// reference has to be truncated the same way — otherwise it points at a tool
// the upstream never sees.
func TestMessagesToChatToolChoiceNameTruncated(t *testing.T) {
	long := strings.Repeat("a", 80)
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
		},
		"tools": []any{
			map[string]any{"name": long, "input_schema": map[string]any{"type": "object"}},
		},
		"tool_choice": map[string]any{"type": "tool", "name": long},
	}
	out := convertReq(t, Messages, Chat, in)
	tools := out["tools"].([]any)
	declared, _ := tools[0].(map[string]any)["function"].(map[string]any)["name"].(string)
	choice, _ := out["tool_choice"].(map[string]any)["function"].(map[string]any)["name"].(string)
	if choice != declared {
		t.Fatalf("tool_choice names %q but the declaration is %q", choice, declared)
	}
	if len(declared) > 64 {
		t.Fatalf("declared tool name is %d chars, over OpenAI's limit", len(declared))
	}
}

// A user message that attached a document (a PDF, say) must not lose it when
// the turn is folded onto the chat wire, where the attachment rides the same
// image_url spelling an image does.
func TestMessagesToChatUserDocumentKept(t *testing.T) {
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "summarize"},
				map[string]any{"type": "document", "source": map[string]any{
					"type": "base64", "media_type": "application/pdf", "data": "UERG",
				}},
			}},
		},
	}
	out := convertReq(t, Messages, Chat, in)
	parts, ok := out["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if !ok {
		t.Fatalf("content = %#v", out["messages"].([]any)[0].(map[string]any)["content"])
	}
	if len(parts) != 2 {
		t.Fatalf("document dropped: %#v", parts)
	}
	if got := parts[1].(map[string]any)["image_url"].(map[string]any)["url"]; got != "data:application/pdf;base64,UERG" {
		t.Fatalf("url = %v", got)
	}
}

// A redacted_thinking block carries its payload in `data`, not `signature`.
// The non-streaming response fold used to read only `signature`, emitting an
// empty reasoning item and losing the opaque bytes the next turn replays.
func TestMessagesResponseToResponsesRedactedThinkingKeepsData(t *testing.T) {
	body := `{"id":"m1","type":"message","role":"assistant","content":[{"type":"redacted_thinking","data":"OPAQUE"}]}`
	out, err := ConvertResponse(Messages, Responses, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	item := m["output"].([]any)[0].(map[string]any)
	if item["type"] != "reasoning" {
		t.Fatalf("item = %#v", item)
	}
	if item["encrypted_content"] != "OPAQUE" {
		t.Fatalf("encrypted_content = %v, want OPAQUE", item["encrypted_content"])
	}
	if s, _ := item["summary"].([]any); len(s) != 0 {
		t.Fatalf("redacted block should carry no summary text: %#v", s)
	}
}

// A document attached to a user message becomes a Responses input_file. The
// bytes ride a data URL and the part carries a filename; `media_type` is not a
// field of input_file, and a URL source is a different field again.
func TestMessagesToResponsesDocumentFilePart(t *testing.T) {
	in := jsonMap{
		"model": "m",
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "document", "source": map[string]any{
					"type": "base64", "media_type": "application/pdf", "data": "JVBE",
				}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "document", "source": map[string]any{
					"type": "url", "url": "https://x/a.pdf",
				}},
			}},
		},
	}
	out := convertReq(t, Messages, Responses, in)
	input := out["input"].([]any)
	part := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if part["type"] != "input_file" {
		t.Fatalf("part = %#v", part)
	}
	if part["file_data"] != "data:application/pdf;base64,JVBE" {
		t.Fatalf("file_data = %v", part["file_data"])
	}
	if part["filename"] != "document.pdf" {
		t.Fatalf("filename = %v", part["filename"])
	}
	if _, bad := part["media_type"]; bad {
		t.Fatalf("input_file has no media_type field: %#v", part)
	}
	urlPart := input[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if urlPart["file_url"] != "https://x/a.pdf" {
		t.Fatalf("url document = %#v", urlPart)
	}
}

// Responses turns strict mode on when a function tool omits `strict`, silently
// rewriting `required` to every property. An Anthropic tool is non-strict
// unless it asked otherwise, so the fold must state the flag.
func TestMessagesToResponsesToolStatesStrict(t *testing.T) {
	in := jsonMap{
		"model":    "m",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"tools": []any{
			map[string]any{"name": "plain", "input_schema": map[string]any{"type": "object"}},
			map[string]any{"name": "strict_one", "strict": true, "input_schema": map[string]any{"type": "object"}},
		},
	}
	out := convertReq(t, Messages, Responses, in)
	tools := out["tools"].([]any)
	if got, ok := tools[0].(map[string]any)["strict"].(bool); !ok || got {
		t.Fatalf("plain tool strict = %#v, want false", tools[0].(map[string]any)["strict"])
	}
	if got, ok := tools[1].(map[string]any)["strict"].(bool); !ok || !got {
		t.Fatalf("strict tool strict = %#v, want true", tools[1].(map[string]any)["strict"])
	}
}
