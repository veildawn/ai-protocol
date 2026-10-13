package protocol

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/chatmessages"
)

// This file pins the Chat->Messages request folds a LiteLLM v1.100.1 audit
// found diverging: each case is a body the codec used to render as something
// the Messages upstream would refuse, or as a silent loss of what the caller
// sent.

// Two chat tool names can sanitize onto the same Messages name (foo/bar and
// foo_bar both want foo_bar). The wire names must stay unique, so the rewrite
// carries an _2 suffix — and which tool keeps the bare name may not depend on
// declaration order, because the already-legal name reserves it first.
func TestChatToMessagesDisambiguatesCollidingToolNames(t *testing.T) {
	tool := func(name string) string {
		return `{"type":"function","function":{"name":"` + name + `","parameters":{"type":"object","properties":{}}}}`
	}
	cases := []struct {
		name  string
		order []string
	}{
		{"invalid first", []string{"foo/bar", "foo_bar"}},
		{"valid first", []string{"foo_bar", "foo/bar"}},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"foo/bar","arguments":"{}"}}]}],` +
			`"tools":[` + tool(tc.order[0]) + `,` + tool(tc.order[1]) + `],` +
			`"tool_choice":{"type":"function","function":{"name":"foo/bar"}}}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		names := map[string]bool{}
		for _, item := range m["tools"].([]any) {
			names[item.(map[string]any)["name"].(string)] = true
		}
		if !names["foo_bar"] || !names["foo_bar_2"] {
			t.Fatalf("%s: tool names = %v, want foo_bar and foo_bar_2", tc.name, names)
		}
		// The history and the tool_choice must point at the rewritten name, or
		// they name a tool the request does not declare.
		use := m["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
		if use["name"] != "foo_bar_2" {
			t.Fatalf("%s: tool_use name = %v, want foo_bar_2", tc.name, use["name"])
		}
		choice := m["tool_choice"].(map[string]any)
		if choice["name"] != "foo_bar_2" {
			t.Fatalf("%s: tool_choice name = %v, want foo_bar_2", tc.name, choice["name"])
		}
	}
}

// The same chat name declared twice maps to the one rewrite, so the duplicate
// does not take a second suffix slot.
func TestChatToMessagesDuplicateToolNameSharesRewrite(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"foo bar","parameters":{"type":"object"}}},` +
		`{"type":"function","function":{"name":"foo/bar","parameters":{"type":"object"}}},` +
		`{"type":"function","function":{"name":"foo/bar","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	want := []string{"foo_bar", "foo_bar_2", "foo_bar_2"}
	for i, item := range m["tools"].([]any) {
		if got := item.(map[string]any)["name"].(string); got != want[i] {
			t.Fatalf("tool %d name = %q, want %q", i, got, want[i])
		}
	}
}

// Anthropic refuses a final assistant turn whose text ends in whitespace, so
// the last assistant message is right-trimmed — the whole string when content
// is one, every text block when content is a list. An earlier assistant turn
// keeps its text verbatim.
func TestChatToMessagesRstripsFinalAssistant(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[` +
		`{"role":"user","content":"q"},` +
		`{"role":"assistant","content":"mid  "},` +
		`{"role":"user","content":"q2"},` +
		`{"role":"assistant","content":"hello  "}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if got := msgs[1].(map[string]any)["content"]; got != "mid  " {
		t.Fatalf("earlier assistant content = %q, want trailing spaces kept", got)
	}
	if got := msgs[3].(map[string]any)["content"]; got != "hello" {
		t.Fatalf("final assistant content = %q, want %q", got, "hello")
	}

	blocks := `{"model":"m","max_tokens":10,"messages":[` +
		`{"role":"user","content":"q"},` +
		`{"role":"assistant","content":[{"type":"text","text":"a \n"},{"type":"text","text":"b"}]}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(blocks))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs = m["messages"].([]any)
	for i, want := range []string{"a", "b"} {
		block := msgs[1].(map[string]any)["content"].([]any)[i].(map[string]any)
		if got := block["text"]; got != want {
			t.Fatalf("text block %d = %q, want %q", i, got, want)
		}
	}

	// A final turn that was whitespace-only reads as the placeholder, not as
	// an empty text block the upstream rejects.
	wsOnly := `{"model":"m","max_tokens":10,"messages":[` +
		`{"role":"user","content":"q"},` +
		`{"role":"assistant","content":"   "}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(wsOnly))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs = m["messages"].([]any)
	if got := msgs[1].(map[string]any)["content"]; got != "[System: Empty message content sanitised to satisfy protocol]" {
		t.Fatalf("whitespace-only final assistant = %q, want the placeholder", got)
	}
}

// A tool result is user-role content, so it joins the user turn before it
// rather than forming a second consecutive user message, which the Messages
// upstream rejects.
func TestChatToMessagesMergesToolResultIntoUserTurn(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"tool","tool_call_id":"call_1","content":"res"}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want one merged user turn", msgs)
	}
	turn := msgs[0].(map[string]any)
	if turn["role"] != "user" {
		t.Fatalf("role = %v", turn["role"])
	}
	blocks := turn["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("content = %#v, want [text, tool_result]", blocks)
	}
	if blocks[0].(map[string]any)["type"] != "text" || blocks[0].(map[string]any)["text"] != "hi" {
		t.Fatalf("first block = %#v", blocks[0])
	}
	if blocks[1].(map[string]any)["type"] != "tool_result" {
		t.Fatalf("second block = %#v", blocks[1])
	}
}

// A Messages tool_use id may carry only [a-zA-Z0-9_-]; a chat caller's id with
// dots or spaces is rewritten in both places it appears, or the request names
// a block the upstream refuses.
func TestChatToMessagesSanitizesToolUseIDs(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call.abc-123 xyz","type":"function","function":{"name":"fn","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"call.abc-123 xyz","content":"r"}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	use := msgs[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if use["id"] != "call_abc-123_xyz" {
		t.Fatalf("tool_use id = %v, want call_abc-123_xyz", use["id"])
	}
	result := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if result["tool_use_id"] != "call_abc-123_xyz" {
		t.Fatalf("tool_use_id = %v, want call_abc-123_xyz", result["tool_use_id"])
	}
}

// A chat function tool folds to the Messages custom-tool shape: type "custom",
// the cache breakpoint lifted from the function level onto the tool (dropped,
// it would silently un-cache the prefix the caller marked), and no chat-only
// `strict` flag.
func TestChatToMessagesFunctionToolShape(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"fn","description":"d",` +
		`"parameters":{"type":"object","properties":{}},"strict":true,` +
		`"cache_control":{"type":"ephemeral"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "custom" {
		t.Fatalf("type = %v, want custom", tool["type"])
	}
	if tool["name"] != "fn" || tool["description"] != "d" {
		t.Fatalf("tool = %#v", tool)
	}
	if _, ok := tool["input_schema"]; !ok {
		t.Fatalf("input_schema dropped: %#v", tool)
	}
	cc, ok := tool["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control = %#v, want the function-level breakpoint", tool["cache_control"])
	}
	if _, ok := tool["strict"]; ok {
		t.Fatalf("strict must not ride the Messages tool: %#v", tool)
	}

	// A breakpoint on the tool itself wins over the function-level one.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","cache_control":{"type":"ephemeral","ttl":"5m"},` +
		`"function":{"name":"fn","parameters":{"type":"object"},"cache_control":{"type":"ephemeral"}}}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tool = m["tools"].([]any)[0].(map[string]any)
	cc, ok = tool["cache_control"].(map[string]any)
	if !ok || cc["ttl"] != "5m" {
		t.Fatalf("cache_control = %#v, want the tool-level breakpoint", tool["cache_control"])
	}
}

// Messages wants an integer ceiling of at least 1: a fractional value is
// rounded (halves to even, the way Python's round does) and clamped up, while
// a whole number — 0 included — rides through exactly as written.
func TestChatToMessagesMaxTokensIntegerCoercion(t *testing.T) {
	cases := []struct {
		body string
		want float64
	}{
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":100.7}`, 101},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":2.5}`, 2},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1.5}`, 2},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":100.5}`, 100},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0.4}`, 1},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0}`, 0},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0.0}`, 1},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":-2}`, -2},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":512}`, 512},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":2.5}`, 2},
	}
	for _, tc := range cases {
		out, err := ConvertRequest(Chat, Messages, []byte(tc.body))
		if err != nil {
			t.Fatalf("%s: %v", tc.body, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatalf("%s: %v", tc.body, err)
		}
		if got, ok := m["max_tokens"].(float64); !ok || got != tc.want {
			t.Fatalf("%s: max_tokens = %#v, want %v", tc.body, m["max_tokens"], tc.want)
		}
	}
}

// A truncated tool_call arguments string — the model cut the object
// mid-flight — is repaired by closing what was opened, not emptied out:
// silently replacing the caller's arguments with {} is a data change nothing
// downstream can see.
func TestChatToMessagesRepairsTruncatedToolArguments(t *testing.T) {
	cases := []struct {
		args string
		want map[string]any
	}{
		{`{"city": "Boston"`, map[string]any{"city": "Boston"}},
		{`{"a": 1,`, map[string]any{"a": float64(1)}},
		{`{"a":1}`, map[string]any{"a": float64(1)}},
		{`   `, map[string]any{}},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",` +
			`"function":{"name":"fn","arguments":` + strconv.Quote(tc.args) + `}}]}],` +
			`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%q: %v", tc.args, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		use := m["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
		got, ok := use["input"].(map[string]any)
		if !ok {
			t.Fatalf("%q: input = %#v, want an object", tc.args, use["input"])
		}
		if len(got) != len(tc.want) {
			t.Fatalf("%q: input = %#v, want %#v", tc.args, got, tc.want)
		}
		for k, v := range tc.want {
			if got[k] != v {
				t.Fatalf("%q: input[%q] = %#v, want %#v", tc.args, k, got[k], v)
			}
		}
	}
}

// Arguments that are neither valid JSON nor closable fail the request instead
// of masquerading as no arguments.
func TestChatToMessagesFailsUnparseableToolArguments(t *testing.T) {
	for _, args := range []string{`{"a":`, `not json`} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",` +
			`"function":{"name":"fn","arguments":` + strconv.Quote(args) + `}}]}],` +
			`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
		_, err := ConvertRequest(Chat, Messages, []byte(body))
		if err == nil {
			t.Fatalf("%q: want an error", args)
		}
		var conv ConversionError
		if !errors.As(err, &conv) {
			t.Fatalf("%q: err = %T(%v), want ConversionError", args, err, err)
		}
	}
}

// The chat web_search_options param rides the Messages dialect as the hosted
// web-search tool it declares there — dropped, the search capability the
// caller asked for would simply be gone.
func TestChatToMessagesMapsWebSearchOptionsToHostedTool(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}],` +
		`"web_search_options":{"search_context_size":"medium"}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %#v, want the function tool plus the hosted search tool", tools)
	}
	hosted := tools[1].(map[string]any)
	if hosted["type"] != "web_search_20250305" || hosted["name"] != "web_search" || hosted["max_uses"] != float64(5) {
		t.Fatalf("hosted tool = %#v", hosted)
	}

	// A location carries only the approximation fields the target defines.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"web_search_options":{"user_location":{"approximate":{"city":"Boston","country":"US","type":"exact","foo":"bar"}}}}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	hosted = m["tools"].([]any)[0].(map[string]any)
	loc, ok := hosted["user_location"].(map[string]any)
	if !ok || loc["type"] != "approximate" || loc["city"] != "Boston" || loc["country"] != "US" {
		t.Fatalf("user_location = %#v", hosted["user_location"])
	}
	if _, present := loc["foo"]; present {
		t.Fatalf("user_location kept a field the target does not define: %#v", loc)
	}

	// A non-dict value names nothing mappable and is dropped, tools included.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"web_search_options":"nonsense"}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["tools"]; present {
		t.Fatalf("tools = %#v, want none from a non-dict web_search_options", m["tools"])
	}
}

// The chat context_management list folds into the edits dict the Messages
// dialect reads; a value with no foldable entry is dropped rather than
// forwarded in a spelling the upstream rejects.
func TestChatToMessagesMapsContextManagementList(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"context_management":[{"type":"compaction","compact_threshold":200000}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	cm, ok := m["context_management"].(map[string]any)
	if !ok {
		t.Fatalf("context_management = %#v, want an edits dict", m["context_management"])
	}
	edits := cm["edits"].([]any)
	edit := edits[0].(map[string]any)
	if edit["type"] != "compact_20260112" {
		t.Fatalf("edit = %#v", edit)
	}
	trigger := edit["trigger"].(map[string]any)
	if trigger["type"] != "input_tokens" || trigger["value"] != float64(200000) {
		t.Fatalf("trigger = %#v", trigger)
	}

	// The Anthropic dict form passes through as-is.
	anthropic := `{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":7}}]}`
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"context_management":` + anthropic + `}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"compact_20260112"`) || !strings.Contains(string(out.Body), `"value":7`) {
		t.Fatalf("dict form not passed through: %s", out.Body)
	}

	// A list with no foldable entry carries nothing the target can act on.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"context_management":[{"type":"other"}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["context_management"]; present {
		t.Fatalf("context_management = %#v, want dropped", m["context_management"])
	}
}

// The Messages wire defaults to non-streaming, so an explicit false is the
// default itself and is omitted; only a request that actually streams says so.
func TestChatToMessagesOmitsStreamFalse(t *testing.T) {
	for _, tc := range []struct {
		body  string
		want  any
		wantS bool
	}{
		{`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stream":false}`, nil, false},
		{`{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stream":true}`, true, true},
	} {
		out, err := ConvertRequest(Chat, Messages, []byte(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		v, present := m["stream"]
		if present != tc.wantS || (present && v != tc.want) {
			t.Fatalf("%s: stream = %#v (present=%v)", tc.body, v, present)
		}
	}
}

// The Messages system prompt is a block list even for one plain text system
// turn — the folded string form is a spelling the reference fold does not
// produce.
func TestChatToMessagesSystemIsBlockList(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	sys, ok := m["system"].([]any)
	if !ok || len(sys) != 1 {
		t.Fatalf("system = %#v, want one block", m["system"])
	}
	block := sys[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "be brief" {
		t.Fatalf("system block = %#v", block)
	}
}

// A chat tool call is a function call or nothing the Messages dialect can
// name: entries of another type (or without one) are skipped rather than
// folded into a tool_use the upstream rejects.
func TestChatToMessagesSkipsNonFunctionToolCalls(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[` +
		`{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}},` +
		`{"id":"c2","type":"custom","function":{"name":"my_tool","arguments":"{}"}},` +
		`{"id":"c3","function":{"name":"no_type","arguments":"{}"}}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	content := m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	var uses []map[string]any
	for _, b := range content {
		if bm, ok := b.(map[string]any); ok && bm["type"] == "tool_use" {
			uses = append(uses, bm)
		}
	}
	if len(uses) != 1 || uses[0]["name"] != "fn" || uses[0]["id"] != "c1" {
		t.Fatalf("tool_use blocks = %#v, want only the function entry", uses)
	}
}

// The hosted web-search tool has no chat spelling as a tool: it folds into the
// request's web_search_options, and the remaining tools are what the chat
// target sees.
func TestMessagesToChatFoldsWebSearchToolIntoOptions(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[` +
		`{"type":"web_search_20250305","name":"web_search","max_uses":5},` +
		`{"type":"custom","name":"fn","input_schema":{"type":"object"}}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	tools := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %#v, want only the function tool", tools)
	}
	if tools[0].(map[string]any)["type"] != "function" {
		t.Fatalf("tool = %#v", tools[0])
	}
	wso, ok := m["web_search_options"].(map[string]any)
	if !ok || len(wso) != 0 {
		t.Fatalf("web_search_options = %#v, want {}", m["web_search_options"])
	}

	// A tools list that is only the hosted search folds to no tools key.
	only := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
	out, err = ConvertRequest(Messages, Chat, []byte(only))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if tools, present := m["tools"]; present && len(tools.([]any)) > 0 {
		t.Fatalf("tools = %#v, want none", tools)
	}
	if _, ok := m["web_search_options"]; !ok {
		t.Fatalf("web_search_options missing")
	}
}

// An assistant turn that carries nothing — no text, no tool call, no thinking
// — folds to no message at all; an assistant turn with empty string content
// stays, the same as the reference fold.
func TestMessagesToChatDropsEmptyAssistantTurn(t *testing.T) {
	cases := []struct {
		content string
		want    int
	}{
		{`[]`, 1},                          // assistant dropped, only the user turn remains
		{``, 1},                            // content absent: dropped
		{`""`, 2},                          // empty string: kept
		{`[{"type":"text","text":""}]`, 2}, // one empty text block: kept
	}
	for _, tc := range cases {
		content := tc.content
		if content == `` {
			// absent content: omit the key entirely
			content = ``
		}
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}`
		if tc.content == `` {
			body += `,{"role":"assistant"}`
		} else {
			body += `,{"role":"assistant","content":` + tc.content + `}`
		}
		body += `]}`
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		msgs := m["messages"].([]any)
		if len(msgs) != tc.want {
			t.Fatalf("content %s: messages = %#v, want %d", tc.content, msgs, tc.want)
		}
	}
}

// A tool_choice the chat dialect has no value for is refused here rather than
// forwarded as an Anthropic-shaped dict the target would reject.
func TestMessagesToChatRejectsUnknownToolChoice(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"banana"}}`
	_, err := ConvertRequest(Messages, Chat, []byte(body))
	if err == nil {
		t.Fatal("want an error for an incompatible tool_choice")
	}
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("err = %T(%v), want ConversionError", err, err)
	}

	ok := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"auto"}}`
	out, err := ConvertRequest(Messages, Chat, []byte(ok))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"tool_choice":"auto"`) {
		t.Fatalf("tool_choice not mapped: %s", out.Body)
	}
}

// A Messages system prompt that is a block list stays a parts list on the chat
// wire — one text part per block, its cache breakpoint with it — and blocks
// that are not text carry nothing the chat dialect reads.
func TestMessagesToChatSystemBlocksStayParts(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"system":[` +
		`{"type":"text","text":"be brief","cache_control":{"type":"ephemeral"}},` +
		`{"type":"text","text":"stay sharp"},` +
		`{"type":"image"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	first := msgs[0].(map[string]any)
	if first["role"] != "system" {
		t.Fatalf("first message = %#v, want the system turn", first)
	}
	parts, ok := first["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("system content = %#v, want two text parts", first["content"])
	}
	p0 := parts[0].(map[string]any)
	if p0["type"] != "text" || p0["text"] != "be brief" {
		t.Fatalf("first part = %#v", p0)
	}
	if cc, ok := p0["cache_control"].(map[string]any); !ok || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control = %#v, want the breakpoint carried", p0["cache_control"])
	}
	if parts[1].(map[string]any)["text"] != "stay sharp" {
		t.Fatalf("second part = %#v", parts[1])
	}

	// A system list with no text block prepends nothing.
	empty := `{"model":"m","max_tokens":10,"system":[{"type":"image"}],"messages":[{"role":"user","content":"hi"}]}`
	out, err = ConvertRequest(Messages, Chat, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs = m["messages"].([]any)
	if len(msgs) != 1 || msgs[0].(map[string]any)["role"] != "user" {
		t.Fatalf("messages = %#v, want no system turn", msgs)
	}
}

// An empty stop list names nothing to stop on: the chat wire omits the key
// rather than carrying an empty one.
func TestMessagesToChatOmitsEmptyStopSequences(t *testing.T) {
	for _, seq := range []string{`[]`, `null`, `""`} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stop_sequences":` + seq + `}`
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if _, present := m["stop"]; present {
			t.Fatalf("stop_sequences %s: stop = %#v, want omitted", seq, m["stop"])
		}
	}
	kept := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stop_sequences":["END"]}`
	out, err := ConvertRequest(Messages, Chat, []byte(kept))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"stop":["END"]`) {
		t.Fatalf("non-empty stop_sequences not carried: %s", out.Body)
	}
}

// Thinking blocks are separate turns of one chain of thought; they join on
// newlines — an empty block and a redacted one contribute nothing — the way
// the reference fold reads them back.
func TestMessagesToChatJoinsReasoningBlocksOnNewlines(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":[` +
		`{"type":"thinking","thinking":"AAA","signature":"s1"},` +
		`{"type":"thinking","thinking":"","signature":"s2"},` +
		`{"type":"thinking","thinking":"BBB","signature":"s3"},` +
		`{"type":"redacted_thinking","data":"xxx"}]}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	assistant := m["messages"].([]any)[1].(map[string]any)
	if assistant["reasoning_content"] != "AAA\nBBB" {
		t.Fatalf("reasoning_content = %#v, want %q", assistant["reasoning_content"], "AAA\nBBB")
	}
}

// An empty stop names a real stop and survives a drop-params fold; only a
// non-empty whitespace-only one is droppable, because the reference reads
// isspace(), which is false for the empty string.
func TestChatToMessagesDropsOnlyWhitespaceStops(t *testing.T) {
	cases := []struct {
		stop string
		want string
	}{
		{`""`, `[""]`},
		{`" "`, ``},
		{`["a"," ",""]`, `["a",""]`},
		{`["a"," "]`, `["a"]`},
	}
	for _, tc := range cases {
		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":` + tc.stop + `}`
		out, err := ConvertRequestWith(Chat, Messages, []byte(body), ConvertOptions{DropParams: true, MaxTokensFallback: 1})
		if err != nil {
			t.Fatalf("%s: %v", tc.stop, err)
		}
		if tc.want == "" {
			if strings.Contains(string(out.Body), "stop_sequences") {
				t.Fatalf("%s: stop_sequences not dropped: %s", tc.stop, out.Body)
			}
			continue
		}
		want := `"stop_sequences":` + tc.want
		if !strings.Contains(string(out.Body), want) {
			t.Fatalf("%s: want %s in %s", tc.stop, want, out.Body)
		}
	}
}

// An explicitly sent empty effort is a value, not an absence: it fails the
// request rather than silently asking for no thinking. "none" stays a real
// off-switch, and a mapped effort still turns into a thinking budget.
func TestChatToMessagesRejectsEmptyReasoningEffort(t *testing.T) {
	for _, effort := range []string{`""`, `{"effort":""}`} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"reasoning_effort":` + effort + `}`
		_, err := ConvertRequest(Chat, Messages, []byte(body))
		if err == nil {
			t.Fatalf("reasoning_effort %s: want an error", effort)
		}
		var conv ConversionError
		if !errors.As(err, &conv) {
			t.Fatalf("reasoning_effort %s: err = %T(%v), want ConversionError", effort, err, err)
		}
	}

	none := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}`
	out, err := ConvertRequest(Chat, Messages, []byte(none))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out.Body), "thinking") {
		t.Fatalf("reasoning_effort none produced thinking: %s", out.Body)
	}

	low := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low"}`
	out, err = ConvertRequest(Chat, Messages, []byte(low))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"thinking"`) {
		t.Fatalf("reasoning_effort low produced no thinking: %s", out.Body)
	}
}

// The user id the Messages dialect carries in metadata is refused only when
// it is shaped like a whole email address or phone number — a bare word with
// an @ in it, or digits with a + mid-string, is simply an id, and refusing it
// silently dropped the caller's metadata.
func TestChatToMessagesUserIDAcceptsPartialShapes(t *testing.T) {
	accepted := []string{"a@b", "1234+678", "abc123"}
	for _, uid := range accepted {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"user":"` + uid + `"}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("user %q: %v", uid, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		md, ok := m["metadata"].(map[string]any)
		if !ok || md["user_id"] != uid {
			t.Fatalf("user %q: metadata = %#v, want user_id carried", uid, m["metadata"])
		}
	}
	refused := []string{"a@b.com", "1234567", "+1 (555) 123-4567"}
	for _, uid := range refused {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"user":"` + uid + `"}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("user %q: %v", uid, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if _, present := m["metadata"]; present {
			t.Fatalf("user %q: metadata = %#v, want refused", uid, m["metadata"])
		}
	}
}

// A structured-output schema rides the Messages wire as written, and the
// target refuses a $ref it cannot follow: the definitions the schema carries
// are inlined into it and the defs blocks removed.
func TestChatToMessagesInlinesOutputSchemaDefs(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{` +
		`"type":"object","properties":{"a":{"$ref":"#/$defs/A"},"b":{"$ref":"#/$defs/B"}},` +
		`"$defs":{"A":{"type":"string"},"B":{"type":"array","items":{"$ref":"#/$defs/A"}}}}}}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	of := m["output_format"].(map[string]any)
	schema := of["schema"].(map[string]any)
	if _, present := schema["$defs"]; present {
		t.Fatalf("$defs survived: %#v", schema)
	}
	props := schema["properties"].(map[string]any)
	if a := props["a"].(map[string]any); a["type"] != "string" || len(a) != 1 {
		t.Fatalf("$ref not inlined: %#v", props["a"])
	}
	b := props["b"].(map[string]any)
	items := b["items"].(map[string]any)
	if items["type"] != "string" {
		t.Fatalf("nested $ref not inlined: %#v", b)
	}
}

// The Messages output format is the schema alone: the chat form's name and
// strict flag have no target spelling and stay behind.
func TestChatToMessagesOutputFormatDropsChatOnlyFields(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"myname","strict":true,` +
		`"schema":{"type":"object","properties":{"a":{"type":"string"}}}}}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	of := m["output_format"].(map[string]any)
	if of["type"] != "json_schema" {
		t.Fatalf("output_format = %#v", of)
	}
	for _, key := range []string{"name", "strict"} {
		if _, present := of[key]; present {
			t.Fatalf("output_format carries %q: %#v", key, of)
		}
	}
}

// json_object names no schema, and the reference fold sends nothing for it —
// no output_format describing a schema the caller never wrote.
func TestChatToMessagesJsonObjectFoldsNoOutputFormat(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"response_format":{"type":"json_object"}}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["output_format"]; present {
		t.Fatalf("json_object produced output_format: %s", out.Body)
	}
}

// The chat side reads a folded output format in strict mode: every object
// schema states additionalProperties:false and names all of its properties as
// required, at every nesting depth.
func TestMessagesToChatStrictifiesOutputSchema(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"output_format":{"type":"json_schema","schema":{"type":"object",` +
		`"properties":{"a":{"type":"string"},"b":{"type":"object","properties":{"c":{"type":"number"}}},` +
		`"d":{"type":"array","items":{"type":"object","properties":{"e":{"type":"string"}}}}}}}}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	rf := m["response_format"].(map[string]any)
	js := rf["json_schema"].(map[string]any)
	if js["strict"] != true {
		t.Fatalf("strict = %#v, want true", js["strict"])
	}
	schema := js["schema"].(map[string]any)
	check := func(s map[string]any, want []string) {
		t.Helper()
		if s["additionalProperties"] != false {
			t.Fatalf("additionalProperties = %#v, want false: %#v", s["additionalProperties"], s)
		}
		required := map[string]bool{}
		for _, r := range s["required"].([]any) {
			required[r.(string)] = true
		}
		for _, w := range want {
			if !required[w] {
				t.Fatalf("required = %#v, want %q among them", s["required"], w)
			}
		}
	}
	check(schema, []string{"a", "b", "d"})
	props := schema["properties"].(map[string]any)
	check(props["b"].(map[string]any), []string{"c"})
	items := props["d"].(map[string]any)["items"].(map[string]any)
	check(items, []string{"e"})
}

// The folded response_format carries the conventional name — the source name
// is not a chat-side field — and an output format without a schema carries
// nothing to fold.
func TestMessagesToChatOutputFormatConventionalName(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"output_format":{"type":"json_schema","name":"keepme","schema":{"type":"object"}}}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	rf := m["response_format"].(map[string]any)
	js := rf["json_schema"].(map[string]any)
	if js["name"] != "structured_output" {
		t.Fatalf("name = %#v, want structured_output", js["name"])
	}

	empty := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"output_format":{"type":"json_schema","schema":{}}}`
	out, err = ConvertRequest(Messages, Chat, []byte(empty))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if _, present := m["response_format"]; present {
		t.Fatalf("empty schema folded a response_format: %s", out.Body)
	}
}

// A dict tool_choice is read by its shape: a type-keyed choice names a mode,
// and a choice carrying a function names that function whatever its type
// field says. A type:tool choice with the name at the top level names nothing
// the target reads and is dropped, and a name the request does not declare
// rides through as written — the target is the one to say it is unknown.
func TestChatToMessagesToolChoiceDictShapes(t *testing.T) {
	tools := `"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}},{"type":"function","function":{"name":"bad name!","parameters":{"type":"object"}}}]`
	cases := []struct {
		choice string
		want   string // "" means the key is absent
	}{
		{`{"type":"tool","name":"x"}`, ``},
		{`{"type":"auto","function":{"name":"fn"}}`, `{"type":"tool","name":"fn"}`},
		{`{"function":{"name":"fn"}}`, `{"type":"tool","name":"fn"}`},
		{`{"type":"auto","function":{}}`, ``},
		{`{"type":"function","function":{"name":"not declared!"}}`, `{"type":"tool","name":"not declared!"}`},
		{`{"type":"function","function":{"name":"bad name!"}}`, `{"type":"tool","name":"bad_name_"}`},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],` + tools + `,"tool_choice":` + tc.choice + `}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.choice, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if tc.want == "" {
			if _, present := m["tool_choice"]; present {
				t.Fatalf("%s: tool_choice = %#v, want dropped", tc.choice, m["tool_choice"])
			}
			continue
		}
		var want map[string]any
		if err := json.Unmarshal([]byte(tc.want), &want); err != nil {
			t.Fatal(err)
		}
		got, ok := m["tool_choice"].(map[string]any)
		if !ok || got["type"] != want["type"] || got["name"] != want["name"] {
			t.Fatalf("%s: tool_choice = %#v, want %s", tc.choice, m["tool_choice"], tc.want)
		}
	}
}

// Tool keys the function shape has no field for are provider knobs the caller
// set; they ride inside the parameters schema rather than being dropped.
func TestMessagesToChatToolExtrasRideInParameters(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[` +
		`{"type":"custom","name":"fn","input_schema":{"type":"object","properties":{"a":{"type":"string"}}},"version":2,"max_uses":3}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	fn := m["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	params, ok := fn["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("parameters = %#v", fn["parameters"])
	}
	if params["type"] != "object" || params["version"] != float64(2) || params["max_uses"] != float64(3) {
		t.Fatalf("parameters = %#v, want the schema carrying version and max_uses", params)
	}
	if _, present := params["name"]; present {
		t.Fatalf("tool's own field leaked into parameters: %#v", params)
	}

	// A tool with no schema still folds its extras as the parameters.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[` +
		`{"type":"custom","name":"fn","version":2}]}`
	out, err = ConvertRequest(Messages, Chat, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	fn = m["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	params, ok = fn["parameters"].(map[string]any)
	if !ok || params["version"] != float64(2) || len(params) != 1 {
		t.Fatalf("parameters = %#v, want only the extras", fn["parameters"])
	}
}

// An assistant turn with no content at all and nothing to carry folds to no
// message — the reference drops it — while a turn whose content is an
// explicit empty string still reads as the placeholder.
func TestChatToMessagesDropsContentlessAssistant(t *testing.T) {
	for _, content := range []string{`null`, ``, `[]`} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"a"}`
		if content == `` {
			body += `,{"role":"assistant"}`
		} else {
			body += `,{"role":"assistant","content":` + content + `}`
		}
		body += `,{"role":"user","content":"b"}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("content %s: %v", content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		for _, msg := range m["messages"].([]any) {
			if msg.(map[string]any)["role"] == "assistant" {
				t.Fatalf("content %s: contentless assistant survived: %s", content, out.Body)
			}
		}
	}
	// An explicit empty string is a turn the caller sent; it folds to the
	// placeholder rather than being dropped.
	kept := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"a"},{"role":"assistant","content":""},{"role":"user","content":"b"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(kept))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), "[System: Empty message content sanitised to satisfy protocol]") {
		t.Fatalf("empty-string assistant lost its placeholder: %s", out.Body)
	}
}

// The legacy function protocol — an assistant function_call answered by a
// role:function message — folds to a tool_use/tool_result pair whose ids
// reference each other: the answer's own tool_call_id when it carries one,
// else one assigned per position.
func TestChatToMessagesFoldsLegacyFunctionCall(t *testing.T) {
	// The answer carries the id; the call picks it up.
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"go"},` +
		`{"role":"assistant","content":null,"function_call":{"name":"get.weather","arguments":"{\"city\": \"SF\"}"}},` +
		`{"role":"function","tool_call_id":"call.l.1","content":"sunny"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	assistant := msgs[1].(map[string]any)
	use := assistant["content"].([]any)[0].(map[string]any)
	// The call names no declared tool, so its name rides as written — the
	// target is the one to say it is unknown.
	if use["type"] != "tool_use" || use["name"] != "get.weather" {
		t.Fatalf("tool_use = %#v", use)
	}
	if use["id"] != "call_l_1" {
		t.Fatalf("tool_use id = %#v, want the answer's sanitized id", use["id"])
	}
	input := use["input"].(map[string]any)
	if input["city"] != "SF" {
		t.Fatalf("input = %#v", input)
	}
	result := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if result["tool_use_id"] != "call_l_1" || result["content"] != "sunny" {
		t.Fatalf("tool_result = %#v", result)
	}

	// Neither side names an id: position assigns the pair one.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"go"},` +
		`{"role":"assistant","content":null,"function_call":{"name":"fn","arguments":"{}"}},` +
		`{"role":"function","content":"ok"}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"id":"legacy_call_0"`) || !strings.Contains(string(out.Body), `"tool_use_id":"legacy_call_0"`) {
		t.Fatalf("position-assigned pair missing: %s", out.Body)
	}
}

// A merged assistant run may repeat a tool_call id — history recorded twice —
// and the target requires tool_use ids to be unique within the message, so
// only the first of each id survives.
func TestChatToMessagesDedupesRepeatedToolUseIDs(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call_dup_1","type":"function","function":{"name":"fn","arguments":"{}"}}]},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call_dup_1","type":"function","function":{"name":"fn","arguments":"{}"}}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	blocks := msgs[1].(map[string]any)["content"].([]any)
	uses := 0
	for _, b := range blocks {
		if bm, ok := b.(map[string]any); ok && bm["type"] == "tool_use" {
			uses++
		}
	}
	if uses != 1 {
		t.Fatalf("tool_use blocks = %d, want the duplicate collapsed: %s", uses, out.Body)
	}
}

// Two string turns of one role merged into a single message keep their own
// text blocks: run together they would read as one glued word the caller
// never wrote.
func TestChatToMessagesMergesConsecutiveStringsAsBlocks(t *testing.T) {
	// Three user turns merge into one message, each keeping its own block.
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"user","content":"hi"},{"role":"user","content":"bye"}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want one merged turn", msgs)
	}
	blocks, ok := msgs[0].(map[string]any)["content"].([]any)
	if !ok || len(blocks) != 3 {
		t.Fatalf("content = %#v, want three text blocks", msgs[0].(map[string]any)["content"])
	}
	for i, want := range []string{"q", "hi", "bye"} {
		if blocks[i].(map[string]any)["text"] != want {
			t.Fatalf("block %d = %#v, want %q", i, blocks[i], want)
		}
	}

	// Assistant string turns merge the same way.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"hi"},{"role":"assistant","content":"bye"}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	msgs = m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v, want the merged assistant turn", msgs)
	}
	blocks, ok = msgs[1].(map[string]any)["content"].([]any)
	if !ok || len(blocks) != 2 {
		t.Fatalf("assistant content = %#v, want two text blocks", msgs[1].(map[string]any)["content"])
	}
	if blocks[0].(map[string]any)["text"] != "hi" || blocks[1].(map[string]any)["text"] != "bye" {
		t.Fatalf("blocks = %#v", blocks)
	}
}

// A data URL whose payload is a document (application/pdf, text/plain) is a
// document a tool result carries — wrapping it in an image block is a shape
// the target rejects. Image payloads keep folding as images.
func TestChatToMessagesToolResultDocumentDataURI(t *testing.T) {
	pdf := `data:application/pdf;base64,QUJD`
	txt := `data:text/plain;base64,QUJD`
	png := `data:image/png;base64,QUJD`
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":[` +
		`{"type":"image_url","image_url":{"url":"` + pdf + `"}},` +
		`{"type":"image_url","image_url":{"url":"` + txt + `"}},` +
		`{"type":"image_url","image_url":{"url":"` + png + `"}}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)["content"].([]any)
	wantTypes := []string{"document", "document", "image"}
	for i, want := range wantTypes {
		if got := blocks[i].(map[string]any)["type"]; got != want {
			t.Fatalf("block %d = %v, want %v (%#v)", i, got, want, blocks[i])
		}
	}
	src := blocks[0].(map[string]any)["source"].(map[string]any)
	if src["media_type"] != "application/pdf" || src["data"] != "QUJD" {
		t.Fatalf("document source = %#v", src)
	}
}

// A file reference folds by what it names: a document source for document
// formats, an image source for image formats, a URL document when the id is
// a URL, and otherwise a container upload the target resolves.
func TestChatToMessagesFilePartsByReference(t *testing.T) {
	cases := []struct {
		file map[string]any
		want map[string]any
	}{
		{map[string]any{"file_id": "file-abc123"},
			map[string]any{"type": "container_upload", "file_id": "file-abc123"}},
		{map[string]any{"file_id": "file-abc123", "format": "application/pdf"},
			map[string]any{"type": "document", "source": map[string]any{"type": "file", "file_id": "file-abc123"}}},
		{map[string]any{"file_id": "f", "format": "text/plain"},
			map[string]any{"type": "document", "source": map[string]any{"type": "file", "file_id": "f"}}},
		{map[string]any{"file_id": "f", "format": "image/png"},
			map[string]any{"type": "image", "source": map[string]any{"type": "file", "file_id": "f"}}},
		{map[string]any{"file_id": "f", "format": "application/octet-stream"},
			map[string]any{"type": "container_upload", "file_id": "f"}},
		{map[string]any{"file_id": "https://x/a.pdf"},
			map[string]any{"type": "document", "source": map[string]any{"type": "url", "url": "https://x/a.pdf"}}},
		{map[string]any{"file_id": "https://x/a.pdf", "format": "application/pdf"},
			map[string]any{"type": "document", "source": map[string]any{"type": "file", "file_id": "https://x/a.pdf"}}},
		{map[string]any{"file_data": "data:application/pdf;base64,QUJD"},
			map[string]any{"type": "document", "source": map[string]any{"type": "base64", "media_type": "application/pdf", "data": "QUJD"}}},
	}
	for _, tc := range cases {
		fileJSON, err := json.Marshal(tc.file)
		if err != nil {
			t.Fatal(err)
		}
		wantJSON, err := json.Marshal(tc.want)
		if err != nil {
			t.Fatal(err)
		}
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"file","file":` + string(fileJSON) + `}]}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", fileJSON, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		block := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0]
		got, err := json.Marshal(block)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != string(wantJSON) {
			t.Fatalf("%s: block = %s, want %s", fileJSON, got, wantJSON)
		}
	}
}

// A file part that names nothing foldable fails the request: an inline
// payload must be a data URL, and a part without either field has no block
// to carry the attachment at all.
func TestChatToMessagesFilePartErrors(t *testing.T) {
	for _, file := range []string{`{"file_data":"QUJD"}`, `{"filename":"x.pdf"}`} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"file","file":` + file + `}]}]}`
		_, err := ConvertRequest(Chat, Messages, []byte(body))
		if err == nil {
			t.Fatalf("%s: want an error", file)
		}
		var conv ConversionError
		if !errors.As(err, &conv) {
			t.Fatalf("%s: err = %T(%v), want ConversionError", file, err, err)
		}
	}
}

// A file part's filename is the chat side's own annotation; the Messages
// document block reads only what the attachment is, so no title rides along.
func TestChatToMessagesFilePartCarriesNoTitle(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"data:application/pdf;base64,QUJD","filename":"doc.pdf"}}]}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out.Body), `"title"`) {
		t.Fatalf("title rode along: %s", out.Body)
	}
}

// The caller's format is the authoritative media type for an inline file
// payload: it overrides the payload's own mime when both are present.
func TestChatToMessagesFileFormatOverridesMedia(t *testing.T) {
	cases := []struct {
		file map[string]any
		want string
	}{
		{map[string]any{"file_data": "data:image/png;base64,QUJD", "format": "application/pdf"}, "application/pdf"},
		{map[string]any{"file_data": "data:image/png;base64,QUJD"}, "image/png"},
	}
	for _, tc := range cases {
		fileJSON, _ := json.Marshal(tc.file)
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"file","file":` + string(fileJSON) + `}]}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", fileJSON, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		block := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		media := block["source"].(map[string]any)["media_type"]
		if media != tc.want {
			t.Fatalf("%s: media_type = %v, want %v", fileJSON, media, tc.want)
		}
	}
}

// A file part without its file object names nothing this fold can read: the
// top-level spelling is not the chat vocabulary's, and the part is refused
// rather than read from a shape the dialect does not define.
func TestChatToMessagesFilePartRequiresFileObject(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"file","file_data":"data:application/pdf;base64,QUJD"}]}]}`
	_, err := ConvertRequest(Chat, Messages, []byte(body))
	if err == nil {
		t.Fatal("want an error for a file part without the file object")
	}
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("err = %T(%v), want ConversionError", err, err)
	}
}

// An image part's declared format is the authoritative media type for an
// inline payload, overriding the data URL's own mime.
func TestChatToMessagesImageFormatOverridesMedia(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QUJD","format":"image/webp"}}]}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	block := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if media := block["source"].(map[string]any)["media_type"]; media != "image/webp" {
		t.Fatalf("media_type = %v, want image/webp", media)
	}
}

// An image url this fold cannot carry — empty, or a scheme the target cannot
// fetch — fails the request rather than passing a block the upstream rejects.
// data: and http(s):// URLs ride their own ways.
func TestChatToMessagesBadImageURLsFail(t *testing.T) {
	for _, url := range []string{``, `ftp://x/a.png`, `notaurl`} {
		part := `{"type":"image_url","image_url":{"url":"` + url + `"}}`
		if url == `` {
			part = `{"type":"image_url","image_url":{}}`
		}
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[` + part + `]}]}`
		_, err := ConvertRequest(Chat, Messages, []byte(body))
		if err == nil {
			t.Fatalf("url %q: want an error", url)
		}
		var conv ConversionError
		if !errors.As(err, &conv) {
			t.Fatalf("url %q: err = %T(%v), want ConversionError", url, err, err)
		}
	}
	kept := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://x/a.png"}},{"type":"image_url","image_url":{"url":"http://x/a.png"}}]}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(kept))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"type":"url","url":"https://x/a.png"`) || !strings.Contains(string(out.Body), `"url":"http://x/a.png"`) {
		t.Fatalf("http(s) urls not carried as url sources: %s", out.Body)
	}
}

// A data URL without the ;base64, marker is a spelling the target does not
// read: the request fails rather than guessing at the payload. An empty
// media type stays empty, and a JSON-escaped slash in the media type reads
// as the slash it names.
func TestChatToMessagesDataURLRequiresBase64Marker(t *testing.T) {
	bad := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png,QUJD"}}]}]}`
	_, err := ConvertRequest(Chat, Messages, []byte(bad))
	if err == nil {
		t.Fatal("want an error for a data URL without the base64 marker")
	}
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("err = %T(%v), want ConversionError", err, err)
	}

	cases := []struct{ url, wantMedia, wantData string }{
		{"data:;base64,", "", ""},
		{`data:image\/png;base64,QUJD`, "image/png", "QUJD"},
		{"data:image/png;base64,QUJD", "image/png", "QUJD"},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"` + tc.url + `"}}]}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.url, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		src := m["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["source"].(map[string]any)
		if src["media_type"] != tc.wantMedia || src["data"] != tc.wantData {
			t.Fatalf("%s: source = %#v, want media %q data %q", tc.url, src, tc.wantMedia, tc.wantData)
		}
	}
}

// An inline thinking block rides only when it says something and carries the
// signature the target verifies: unsigned or empty ones are dropped, and a
// redacted block has no inline spelling the target reads.
func TestChatToMessagesDropsUnsignedInlineThinking(t *testing.T) {
	cases := []struct {
		content string
		wantLen int
	}{
		{`[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"t"}]`, 1},
		{`[{"type":"thinking","thinking":"hmm","signature":"s"},{"type":"text","text":"t"}]`, 2},
		{`[{"type":"redacted_thinking","data":"x"},{"type":"text","text":"t"}]`, 1},
		{`[{"type":"thinking","thinking":"","signature":"s"},{"type":"text","text":"t"}]`, 1},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":` + tc.content + `}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		blocks := m["messages"].([]any)[1].(map[string]any)["content"]
		var list []any
		if s, ok := blocks.(string); ok {
			list = []any{s}
		} else {
			list = blocks.([]any)
		}
		if len(list) != tc.wantLen {
			t.Fatalf("%s: blocks = %#v, want %d", tc.content, blocks, tc.wantLen)
		}
	}
}

// An empty or blank assistant text folds to the placeholder wherever the
// turn sits — the target refuses empty text — and a turn whose only words
// are tool calls still carries the placeholder text before them.
func TestChatToMessagesPlaceholderForBlankAssistantText(t *testing.T) {
	placeholder := "[System: Empty message content sanitised to satisfy protocol]"
	cases := []struct {
		content string
		want    string
	}{
		{`[{"type":"text","text":""}]`, placeholder},
		{`   `, placeholder},
	}
	for _, tc := range cases {
		content := tc.content
		if strings.TrimSpace(content) == "" && content != `[{"type":"text","text":""}]` {
			content = `"   "`
		}
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":` +
			content + `},{"role":"user","content":"r"}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%q: %v", tc.content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		msgs := m["messages"].([]any)
		blank := msgs[1].(map[string]any)
		if !strings.Contains(string(out.Body), placeholder) {
			t.Fatalf("%q: placeholder missing: %s", tc.content, out.Body)
		}
		if _, isString := blank["content"].(string); !isString && blank["content"] == nil {
			t.Fatalf("%q: content = %#v", tc.content, blank["content"])
		}
	}

	// A tool-calling turn with empty or blank text carries the placeholder
	// text block before the tool_use.
	for _, text := range []string{``, `   `} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"` + text + `","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]}],"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		blocks := m["messages"].([]any)[1].(map[string]any)["content"].([]any)
		if len(blocks) != 2 || blocks[0].(map[string]any)["type"] != "text" || blocks[0].(map[string]any)["text"] != placeholder {
			t.Fatalf("%q: blocks = %#v, want placeholder text then tool_use", text, blocks)
		}
	}
}

// The thinking_blocks field rides at the head of the turn only when the
// content list carries no thinking of its own inline: both kept, the turn
// would hold the same chain of thought twice.
func TestChatToMessagesSkipsThinkingFieldWhenInline(t *testing.T) {
	// Field alone: its block rides first.
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"text","text":"t"}],"thinking_blocks":[{"type":"thinking","thinking":"outer","signature":"sig"}]}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks := m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if len(blocks) != 2 || blocks[0].(map[string]any)["type"] != "thinking" {
		t.Fatalf("field-only blocks = %#v, want thinking first", blocks)
	}

	// Both spellings: only the inline one rides.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"text","text":"t"},{"type":"thinking","thinking":"inner","signature":"sig"}],"thinking_blocks":[{"type":"thinking","thinking":"outer","signature":"sig"}]}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	blocks = m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("both blocks = %#v, want only the inline pair", blocks)
	}
	for _, b := range blocks {
		bm := b.(map[string]any)
		if bm["type"] == "thinking" && bm["thinking"] != "inner" {
			t.Fatalf("field thinking duplicated the inline one: %#v", blocks)
		}
	}
}

// A history tool_call naming a tool the request does not declare rides as
// written — the target is the one to say it is unknown — while a declared
// tool's call carries the wire name its declaration was given.
func TestChatToMessagesUndeclaredToolNamesRideAsWritten(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"undeclared tool!","arguments":"{}"}}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	use := m["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if use["name"] != "undeclared tool!" {
		t.Fatalf("undeclared name = %#v, want as written", use["name"])
	}

	// A declared tool's call carries the rewritten wire name; the already
	// legal name reserved its slot, so the colliding rewrite takes the
	// disambiguating suffix.
	body = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"q"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"a/b<c>","arguments":"{}"}}]}],` +
		`"tools":[{"type":"function","function":{"name":"a/b<c>","parameters":{"type":"object"}}},{"type":"function","function":{"name":"a_b_c_","parameters":{"type":"object"}}}]}`
	out, err = ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	use = m["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if use["name"] != "a_b_c__2" {
		t.Fatalf("declared name = %#v, want the rewritten wire name", use["name"])
	}
}

// A tool result that is not text or parts — absent, null, a scalar, an
// object — carries nothing: it folds to the empty string rather than a null
// the target reads as a malformed block.
func TestChatToMessagesEmptyToolResultContentFoldsEmptyString(t *testing.T) {
	for _, content := range []string{`null`, `123`, `{}`, ``} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]},`
		if content == `` {
			body += `{"role":"tool","tool_call_id":"c1"}`
		} else {
			body += `{"role":"tool","tool_call_id":"c1","content":` + content + `}`
		}
		body += `],"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("content %s: %v", content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		result := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
		if result["content"] != "" {
			t.Fatalf("content %s: tool_result content = %#v, want \"\"", content, result["content"])
		}
	}
}

// A tool reference part names a declared tool rather than carrying result
// content; it rides as the reference block it already is.
func TestChatToMessagesToolReferencePartFolds(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":[{"type":"tool_reference","tool_name":"web_search"}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	result := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	parts := result["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts = %#v, want the reference kept", parts)
	}
	ref := parts[0].(map[string]any)
	if ref["type"] != "tool_reference" || ref["tool_name"] != "web_search" {
		t.Fatalf("reference = %#v", ref)
	}
}

// A prompt-cache breakpoint on a tool result's text part rides with the
// text, the same as on a user turn's — dropping it would un-cache the very
// prefix the caller marked.
func TestChatToMessagesToolResultTextCarriesCacheControl(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":[` +
		`{"type":"text","text":"ok","cache_control":{"type":"ephemeral"}},{"type":"text","text":"plain"}]}],` +
		`"tools":[{"type":"function","function":{"name":"fn","parameters":{"type":"object"}}}]}`
	out, err := ConvertRequest(Chat, Messages, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	result := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	parts := result["content"].([]any)
	first := parts[0].(map[string]any)
	if cc, ok := first["cache_control"].(map[string]any); !ok || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control = %#v, want carried on the marked part", first)
	}
	if _, present := parts[1].(map[string]any)["cache_control"]; present {
		t.Fatalf("unmarked part grew a cache_control: %#v", parts[1])
	}
}

// A Messages user turn whose content carries nothing the chat dialect can
// read — null, absent, a scalar, an empty string — folds to no message
// rather than a null the target rejects.
func TestMessagesToChatDropsUnreadableUserContent(t *testing.T) {
	for _, content := range []string{`null`, `42`, `""`, ``} {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":` +
			map[bool]string{true: content, false: `null`}[content != ``] +
			`},{"role":"user","content":[{"type":"text","text":"kept"}]}]}`
		if content == `` {
			body = `{"model":"m","max_tokens":10,"messages":[{"role":"user"},{"role":"user","content":[{"type":"text","text":"kept"}]}]}`
		}
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatalf("content %q: %v", content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		msgs := m["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("content %q: messages = %#v, want only the readable turn", content, msgs)
		}
	}
}

// A source with no payload names no image: the block is dropped rather than
// folded to an empty data URL. A declared media type — empty included — is
// the caller's; only a missing one reads as the default.
func TestMessagesToChatImageSourceEdges(t *testing.T) {
	cases := []struct {
		source string
		want   string // "" means the block is dropped
	}{
		{`{"type":"base64","media_type":"image/png","data":""}`, ``},
		{`{"type":"base64","media_type":"","data":"AAA="}`, "data:;base64,AAA="},
		{`{"type":"base64","data":"AAA="}`, "data:image/jpeg;base64,AAA="},
		{`{"type":"url","url":""}`, ``},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":` + tc.source + `},{"type":"text","text":"kept"}]}]}`
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.source, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		content := m["messages"].([]any)[0].(map[string]any)["content"]
		if tc.want == "" {
			// The dropped image leaves the one text part, which collapses to
			// the plain string this library keeps for single-text content.
			if s, ok := content.(string); !ok || s != "kept" {
				t.Fatalf("%s: content = %#v, want only the text", tc.source, content)
			}
			continue
		}
		parts, ok := content.([]any)
		if !ok || len(parts) != 2 || parts[0].(map[string]any)["image_url"].(map[string]any)["url"] != tc.want {
			t.Fatalf("%s: content = %#v, want image url %q", tc.source, content, tc.want)
		}
	}
}

// An in-sequence system entry rides at its position: a string stays a string
// when it says something, a list becomes its non-empty text parts, and an
// entry left with no content is dropped rather than sent empty.
func TestMessagesToChatMidturnSystemShapes(t *testing.T) {
	kept := `{"model":"m","max_tokens":10,"messages":[{"role":"system","content":"abc"},{"role":"user","content":"q"}]}`
	out, err := ConvertRequest(Messages, Chat, []byte(kept))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	if msgs := m["messages"].([]any); len(msgs) != 2 || msgs[0].(map[string]any)["content"] != "abc" {
		t.Fatalf("string system: %#v", m["messages"])
	}

	list := `{"model":"m","max_tokens":10,"messages":[{"role":"system","content":[{"type":"text","text":"A"},{"type":"image"},{"type":"text","text":"B"}]},{"role":"user","content":"q"}]}`
	out, err = ConvertRequest(Messages, Chat, []byte(list))
	if err != nil {
		t.Fatal(err)
	}
	m = map[string]any{}
	if err := json.Unmarshal(out.Body, &m); err != nil {
		t.Fatal(err)
	}
	parts := m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["text"] != "A" || parts[1].(map[string]any)["text"] != "B" {
		t.Fatalf("list system parts = %#v, want the two texts alone", parts)
	}

	for _, empty := range []string{
		`{"role":"system","content":""}`,
		`{"role":"system","content":[{"type":"text","text":""}]}`,
		`{"role":"system","content":null}`,
		`{"role":"system","content":[{"type":"image"}]}`,
	} {
		body := `{"model":"m","max_tokens":10,"messages":[` + empty + `,{"role":"user","content":"q"}]}`
		out, err := ConvertRequest(Messages, Chat, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", empty, err)
		}
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		if msgs := m["messages"].([]any); len(msgs) != 1 {
			t.Fatalf("%s: messages = %#v, want the empty system entry dropped", empty, msgs)
		}
	}
}

// A tool_result whose parts all carry nothing renders as the empty string —
// never an Anthropic-shaped list the chat upstream rejects wholesale. A
// single text part collapses to the flat string; more parts ride as the
// parts array, tool references among them, and bare strings read as text.
func TestMessagesToChatToolResultContentShapes(t *testing.T) {
	cases := []struct {
		content any
		want    string // JSON of the expected chat content
	}{
		{[]any{}, `""`},
		{[]any{map[string]any{"type": "banana"}}, `""`},
		{[]any{"plain"}, `"plain"`},
		{[]any{map[string]any{"type": "text", "text": "only"}}, `"only"`},
		{[]any{map[string]any{"type": "text", "text": ""}}, `""`},
		{[]any{map[string]any{"type": "text", "text": "a"}, map[string]any{"type": "text", "text": "b"}},
			`[{"text":"a","type":"text"},{"text":"b","type":"text"}]`},
		{[]any{map[string]any{"type": "text", "text": "a"}, map[string]any{"type": "tool_reference", "tool_name": "ws"}},
			`[{"text":"a","type":"text"},{"tool_name":"ws","type":"tool_reference"}]`},
		{[]any{map[string]any{"type": "tool_reference", "tool_name": "ws"}},
			`[{"tool_name":"ws","type":"tool_reference"}]`},
		{42, `""`},
	}
	for _, tc := range cases {
		in := map[string]any{
			"model": "m",
			"messages": []any{
				map[string]any{"role": "user", "content": "q"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "tool_use", "id": "c1", "name": "f", "input": map[string]any{}},
				}},
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "c1", "content": tc.content},
				}},
			},
			"max_tokens": 10,
		}
		b, err := json.Marshal(in)
		if err != nil {
			t.Fatal(err)
		}
		out, err := ConvertRequest(Messages, Chat, b)
		if err != nil {
			t.Fatalf("%#v: %v", tc.content, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		var toolMsg map[string]any
		for _, msg := range m["messages"].([]any) {
			if mm := msg.(map[string]any); mm["role"] == "tool" {
				toolMsg = mm
			}
		}
		got, err := json.Marshal(toolMsg["content"])
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != tc.want {
			t.Fatalf("content %#v: got %s, want %s", tc.content, got, tc.want)
		}
	}
}

// An integer enum is written as integers: the schema's declared type is kept,
// not silently dropped by a number-shape check that cannot see the decoded
// literal's own shape.
func TestChatToMessagesKeepsEnumDeclaredType(t *testing.T) {
	cases := []struct {
		schema string
		keeps  bool
	}{
		{`{"type":"integer","enum":[1,2]}`, true},
		{`{"type":"number","enum":[1.5]}`, true},
		{`{"type":"number","enum":[1,2]}`, true},
		{`{"type":"integer","enum":[1.5]}`, false},
		{`{"type":"string","enum":["a"]}`, true},
		{`{"type":"string","enum":[1]}`, false},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
			`"response_format":{"type":"json_schema","json_schema":{"schema":` + tc.schema + `}}}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.schema, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		schema := m["output_format"].(map[string]any)["schema"].(map[string]any)
		if _, hasType := schema["type"]; hasType != tc.keeps {
			t.Fatalf("%s: type present = %v, want %v (%s)", tc.schema, hasType, tc.keeps, out.Body)
		}
	}
}

// A declared type the enum check has no rule for is not in conflict — the
// type rides as written; a type written as a list cannot be checked and is
// the conflict it reads as.
func TestChatToMessagesEnumWithUncheckableType(t *testing.T) {
	cases := []struct {
		schema string
		keeps  bool
	}{
		{`{"type":"custom","enum":["a"]}`, true},
		{`{"type":["string","null"],"enum":["a"]}`, false},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
			`"response_format":{"type":"json_schema","json_schema":{"schema":` + tc.schema + `}}}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.schema, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		schema := m["output_format"].(map[string]any)["schema"].(map[string]any)
		if _, hasType := schema["type"]; hasType != tc.keeps {
			t.Fatalf("%s: type present = %v, want %v", tc.schema, hasType, tc.keeps)
		}
	}
}

// A circular definition terminates: the reference being resolved stays as
// written for the target to judge, and nothing loops or grows.
func TestChatToMessagesCircularDefinitionTerminates(t *testing.T) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/definitions/A"},
		},
		"definitions": map[string]any{
			"A": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"self": map[string]any{"$ref": "#/definitions/A"},
				},
			},
		},
	}
	body, err := json.Marshal(map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"tools":      []any{map[string]any{"type": "function", "function": map[string]any{"name": "fn", "parameters": params}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := ConvertRequest(Chat, Messages, body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"$ref":"#/definitions/A"`) {
		t.Fatalf("circular ref not left in place: %s", string(out.Body)[:min(400, len(out.Body))])
	}
}

// The resolution scope reads what the schema carries: the native $defs joins
// it (and wins a name clash with a legacy definition), definitions embedded
// in a resolved target resolve against the scope it brings, and an empty
// legacy block still leaves $defs refs resolvable.
func TestChatToMessagesDefsResolutionScope(t *testing.T) {
	run := func(t *testing.T, params map[string]any) map[string]any {
		t.Helper()
		body, err := json.Marshal(map[string]any{
			"model":      "m",
			"max_tokens": 10,
			"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
			"tools":      []any{map[string]any{"type": "function", "function": map[string]any{"name": "fn", "parameters": params}}},
		})
		if err != nil {
			t.Fatal(err)
		}
		out, err := ConvertRequest(Chat, Messages, body)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		return m["tools"].([]any)[0].(map[string]any)["input_schema"].(map[string]any)
	}

	// (a) empty legacy block + $defs: the $defs ref resolves, the block stays.
	schema := run(t, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/$defs/A"},
		},
		"definitions": map[string]any{},
		"$defs":       map[string]any{"A": map[string]any{"type": "string"}},
	})
	if p := schema["properties"].(map[string]any)["p"].(map[string]any); p["type"] != "string" {
		t.Fatalf("(a) $defs ref unresolved: %#v", schema)
	}
	if _, present := schema["$defs"]; !present {
		t.Fatalf("(a) $defs block dropped: %#v", schema)
	}

	// (b) name clash: the $defs body wins.
	schema = run(t, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/definitions/A"},
		},
		"definitions": map[string]any{"A": map[string]any{"type": "integer"}},
		"$defs":       map[string]any{"A": map[string]any{"type": "string"}},
	})
	if p := schema["properties"].(map[string]any)["p"].(map[string]any); p["type"] != "string" {
		t.Fatalf("(b) clash resolved to the wrong body: %#v", p)
	}

	// (c) definitions a target carries resolve against the scope it brings.
	schema = run(t, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/definitions/A"},
		},
		"definitions": map[string]any{
			"A": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"q": map[string]any{"$ref": "#/definitions/B"},
				},
				"definitions": map[string]any{"B": map[string]any{"type": "string"}},
			},
		},
	})
	inner := schema["properties"].(map[string]any)["p"].(map[string]any)
	if q := inner["properties"].(map[string]any)["q"].(map[string]any); q["type"] != "string" {
		t.Fatalf("(c) embedded definitions unresolved: %#v", inner)
	}
}

// An expansion past the inlining budget fails the request: a request-supplied
// schema cannot balloon into a schema-bomb on its way to the upstream.
func TestChatToMessagesDefsBudgetFails(t *testing.T) {
	big := map[string]any{"type": "string", "description": strings.Repeat("a", 6_000_000)}
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/definitions/A"},
			"q": map[string]any{"$ref": "#/definitions/B"},
		},
		"definitions": map[string]any{"A": big, "B": big},
	}
	body, err := json.Marshal(map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"tools":      []any{map[string]any{"type": "function", "function": map[string]any{"name": "fn", "parameters": params}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = ConvertRequest(Chat, Messages, body)
	if err == nil {
		t.Fatal("want an error when the inlined schema exceeds the budget")
	}
	var conv ConversionError
	if !errors.As(err, &conv) {
		t.Fatalf("err = %T(%v), want ConversionError", err, err)
	}
}

// The fold never writes through the caller's map: a tool schema passed by
// reference comes back byte-identical, which is the pure-function contract.
func TestChatToMessagesLeavesCallerSchemaUntouched(t *testing.T) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"p": map[string]any{"$ref": "#/definitions/A"},
		},
		"definitions": map[string]any{"A": map[string]any{"type": "string"}},
	}
	before, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	body := map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"tools":      []any{map[string]any{"type": "function", "function": map[string]any{"name": "fn", "parameters": params}}},
	}
	if _, err := chatmessages.ChatToMessages(body, false); err != nil {
		t.Fatal(err)
	}
	after, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatalf("caller schema mutated:\nbefore %s\nafter  %s", before, after)
	}
}

// The two combinator keys merge into anyOf in a fixed order: map iteration
// is not ordered, and the same request must fold the same way every time.
func TestChatToMessagesCombinatorMergeIsDeterministic(t *testing.T) {
	schema := `{"oneOf":[{"type":"string"},{"type":"integer"}],"anyOf":[{"type":"number"}]}`
	want := `"anyOf":[{"type":"string"},{"type":"integer"},{"type":"number"}]`
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"response_format":{"type":"json_schema","json_schema":{"schema":` + schema + `}}}`
	for i := 0; i < 20; i++ {
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out.Body), want) {
			t.Fatalf("run %d: merge order changed: %s", i, out.Body)
		}
	}
}

// A dropped constraint's advisory note spells its value the way the
// reference serializer does: ", " and ": " separators, no HTML escaping,
// non-ASCII escaped, exponent numbers in float repr, whole floats with .0,
// and a capitalised boolean word.
func TestChatToMessagesConstraintNoteSpelling(t *testing.T) {
	cases := []struct {
		schema string
		note   string
	}{
		{`{"type":"object","contains":{"type":"string"}}`,
			`array must contain an item matching: {"type": "string"}`},
		{`{"type":"object","contains":{"type":"string","maxLength":3}}`,
			`array must contain an item matching: {"maxLength": 3, "type": "string"}`},
		{`{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}]}`,
			`leading items must match, in order: [{"type": "string"}, {"type": "integer"}]`},
		{`{"type":"object","then":{"description":"a<b>c"}}`,
			`conditional (then): {"description": "a<b>c"}`},
		{`{"type":"object","then":{"description":"café"}}`,
			`conditional (then): {"description": "caf\u00e9"}`},
		{`{"type":"number","maximum":1e21}`, `maximum value: 1e+21`},
		{`{"type":"number","minimum":2.5e-7}`, `minimum value: 2.5e-07`},
		{`{"type":"number","minimum":1.5}`, `minimum value: 1.5`},
		{`{"type":"number","minimum":5.0}`, `minimum value: 5.0`},
		{`{"type":"object","unevaluatedProperties":true}`, `unevaluated properties must satisfy: True`},
	}
	for _, tc := range cases {
		body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],` +
			`"response_format":{"type":"json_schema","json_schema":{"schema":` + tc.schema + `}}}`
		out, err := ConvertRequest(Chat, Messages, []byte(body))
		if err != nil {
			t.Fatalf("%s: %v", tc.schema, err)
		}
		var m map[string]any
		m = map[string]any{}
		if err := json.Unmarshal(out.Body, &m); err != nil {
			t.Fatal(err)
		}
		schema := m["output_format"].(map[string]any)["schema"].(map[string]any)
		desc, _ := schema["description"].(string)
		want := "Note: " + tc.note + "."
		if desc != want {
			t.Fatalf("%s:\n got %q\nwant %q", tc.schema, desc, want)
		}
	}
}
