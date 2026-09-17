package protocol

import (
	"bytes"
	"encoding/json"
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
