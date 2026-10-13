package chatmessages

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

const openaiMaxToolName = 64

// anthropicMaxToolName is Anthropic's tool-name cap, and its allowed character
// class. A chat tool name that violates either is refused by the upstream, so
// it is sanitized on the way in — the same chokepoint the reverse direction
// already truncates at.
const anthropicMaxToolName = 128

// emptyTextPlaceholder stands in for an empty text block. Anthropic rejects
// empty text content with a 400, and there is no way to pass one through, so
// an empty user or assistant turn is rewritten to this.
const emptyTextPlaceholder = "[System: Empty message content sanitised to satisfy protocol]"

// sanitizeToolNameForAnthropic makes a chat tool name legal on the Messages
// wire: every character outside [a-zA-Z0-9_-] becomes '_', and the result is
// capped at 128 characters.
func sanitizeToolNameForAnthropic(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if len(out) > anthropicMaxToolName {
		out = out[:anthropicMaxToolName]
	}
	return out
}

// The Chat dialect's request vocabulary. A parameter in this set that the
// target dialect cannot represent is refused, never silently dropped.
var openaiChatParams = map[string]struct{}{
	"messages": {}, "model": {}, "frequency_penalty": {}, "logit_bias": {}, "logprobs": {},
	"top_logprobs": {}, "max_tokens": {}, "max_completion_tokens": {}, "n": {},
	"presence_penalty": {}, "response_format": {}, "seed": {}, "stop": {}, "stream": {},
	"stream_options": {}, "temperature": {}, "top_p": {}, "tools": {}, "tool_choice": {},
	"parallel_tool_calls": {}, "user": {}, "function_call": {}, "functions": {},
	"reasoning_effort": {}, "thinking": {}, "metadata": {}, "modalities": {},
	"prediction": {}, "service_tier": {}, "store": {}, "web_search_options": {},
	"extra_headers": {}, "timeout": {}, "speed": {}, "context_management": {},
	"cache_control": {}, "audio": {},
}

// Chat parameters a Messages target can represent, plus always-kept keys.
var chatToMessagesSupported = map[string]struct{}{
	"model": {}, "messages": {},
	"stream": {}, "stop": {}, "temperature": {}, "top_p": {},
	"max_tokens": {}, "max_completion_tokens": {},
	"tools": {}, "tool_choice": {}, "extra_headers": {},
	"parallel_tool_calls": {}, "response_format": {}, "user": {},
	"web_search_options": {}, "speed": {}, "context_management": {},
	"cache_control": {}, "thinking": {}, "reasoning_effort": {},
}

func ChatToMessages(body map[string]any, dropParams bool) (map[string]any, error) {
	if err := rejectUnsupported(body, chatToMessagesSupported, dropParams); err != nil {
		return nil, err
	}
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	// Tool names are sanitized once and the same mapping is applied to every
	// tool_use block in the history: a strict target rejects a turn whose
	// tool_use names a tool the request does not declare.
	nameMap := anthropicToolNameMap(body["tools"])
	system, converted, err := openaiMessagesToAnthropic(msgs, nameMap)
	if err != nil {
		return nil, err
	}
	out["messages"] = converted
	if len(system) == 1 {
		if t, ok := system[0]["text"].(string); ok && system[0]["type"] == "text" && len(system[0]) == 2 {
			out["system"] = t
		} else {
			out["system"] = system
		}
	} else if len(system) > 1 {
		out["system"] = system
	}

	if v, ok := body["max_completion_tokens"]; ok {
		out["max_tokens"] = v
	} else if v, ok := body["max_tokens"]; ok {
		out["max_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	if v, ok := body["stop"]; ok {
		if seq := mapStopSequences(v, dropParams); seq != nil {
			out["stop_sequences"] = seq
		}
	}
	if v, ok := body["tools"]; ok {
		out["tools"] = openaiToolsToAnthropic(v, nameMap)
	}
	if tc, ok := mapToolChoiceToAnthropic(body["tool_choice"], body["parallel_tool_calls"], nameMap); ok {
		out["tool_choice"] = tc
	}
	if v, ok := body["user"].(string); ok && v != "" && validUserID(v) {
		out["metadata"] = map[string]any{"user_id": v}
	}
	if v, ok := body["thinking"]; ok {
		out["thinking"] = v
	} else if effort, ok := reasoningEffortString(body["reasoning_effort"]); ok {
		thinking, err := thinkingFromEffort(effort)
		if err != nil {
			return nil, err
		}
		if thinking != nil {
			out["thinking"] = thinking
		}
	}
	if rf, ok := jsonx.AsMap(body["response_format"]); ok {
		if mapped := responseFormatToAnthropic(rf); mapped != nil {
			out["output_format"] = mapped
		}
	}
	copyIf(body, out, "context_management")
	copyIf(body, out, "speed")
	// Parameters outside the Chat vocabulary pass through untranslated.
	for k, v := range body {
		if _, isOA := openaiChatParams[k]; isOA {
			continue
		}
		if _, exists := out[k]; exists {
			continue
		}
		out[k] = v
	}
	return out, nil
}

func MessagesToChat(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	converted := anthropicMessagesToOpenAI(msgs)
	if sys := body["system"]; sys != nil {
		converted = prependSystem(converted, sys)
	}
	out["messages"] = converted
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	if v, ok := body["max_tokens"]; ok {
		out["max_tokens"] = v
	}
	if v, ok := body["stop_sequences"]; ok {
		out["stop"] = v
	}
	if v, ok := body["tools"]; ok {
		out["tools"] = anthropicToolsToOpenAI(v)
	}
	if v, ok := body["tool_choice"]; ok {
		out["tool_choice"] = anthropicToolChoiceToOpenAI(v)
	}
	if thinking, ok := jsonx.AsMap(body["thinking"]); ok {
		if effort := effortFromThinking(thinking); effort != "" {
			out["reasoning_effort"] = effort
		}
	}
	// Structured output arrives under the legacy top-level `output_format` or
	// the newer `output_config.format`; the legacy field wins when both are set.
	of, ok := jsonx.AsMap(body["output_format"])
	if !ok {
		if oc, ok := jsonx.AsMap(body["output_config"]); ok {
			of, ok = jsonx.AsMap(oc["format"])
		}
	}
	if ok {
		if rf := anthropicOutputFormatToOpenAI(of); rf != nil {
			out["response_format"] = rf
		}
	}
	if md, ok := jsonx.AsMap(body["metadata"]); ok {
		if uid := jsonx.GetString(md, "user_id"); uid != "" {
			out["user"] = uid
		}
	}
	translatable := map[string]struct{}{
		"messages": {}, "metadata": {}, "system": {}, "tool_choice": {}, "tools": {},
		"thinking": {}, "output_format": {}, "output_config": {}, "stop_sequences": {},
	}
	for k, v := range body {
		if _, skip := translatable[k]; skip {
			continue
		}
		if _, exists := out[k]; exists {
			continue
		}
		out[k] = v
	}
	return out, nil
}

func rejectUnsupported(body map[string]any, supported map[string]struct{}, drop bool) error {
	for k := range body {
		if _, isOA := openaiChatParams[k]; !isOA {
			continue
		}
		if _, ok := supported[k]; ok {
			continue
		}
		if drop {
			continue
		}
		return cerr.UnsupportedParam{Param: k}
	}
	return nil
}

func copyIf(src, dst map[string]any, key string) {
	if v, ok := src[key]; ok {
		dst[key] = v
	}
}

func mapStopSequences(v any, drop bool) []any {
	switch t := v.(type) {
	case string:
		if drop && strings.TrimSpace(t) == t && strings.TrimSpace(t) == "" {
			return nil
		}
		if drop && strings.TrimSpace(t) != t { // whitespace-only
			if strings.TrimSpace(t) == "" {
				return nil
			}
		}
		return []any{t}
	case []any:
		out := make([]any, 0, len(t))
		for _, item := range t {
			s := jsonx.String(item)
			if drop && strings.TrimSpace(s) == "" {
				continue
			}
			out = append(out, item)
		}
		if len(out) == 0 {
			return nil
		}
		return out
	default:
		return []any{v}
	}
}

// reasoningEffortString reads a reasoning_effort value, which arrives either as
// a bare string or as the Responses-shaped {"effort": ...} dict a bridged
// caller sends.
func reasoningEffortString(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case map[string]any:
		if s := jsonx.GetString(t, "effort"); s != "" {
			return s, true
		}
	}
	return "", false
}

// validUserID reports whether a chat `user` value may ride Messages metadata as
// user_id. Anthropic rejects an email address or a phone number there.
func validUserID(v string) bool {
	if strings.Contains(v, "@") {
		return false
	}
	for _, r := range v {
		switch {
		case r >= '0' && r <= '9', r == '+', r == ' ', r == '(', r == ')', r == '-':
		default:
			return true
		}
	}
	return len(v) < 7
}

func thinkingFromEffort(effort string) (map[string]any, error) {
	if effort == "" || effort == "none" {
		return nil, nil
	}
	budget, ok := types.BudgetFromEffort(effort)
	if !ok {
		return nil, fmt.Errorf("unmapped reasoning effort: %q", effort)
	}
	return map[string]any{"type": "enabled", "budget_tokens": budget}, nil
}

func effortFromThinking(thinking map[string]any) string {
	switch jsonx.GetString(thinking, "type") {
	case "disabled":
		return "none"
	case "adaptive":
		return "medium"
	case "enabled":
		budget, _ := jsonx.Int(thinking["budget_tokens"])
		return types.EffortFromBudget(budget)
	default:
		return ""
	}
}

func responseFormatToAnthropic(rf map[string]any) map[string]any {
	switch jsonx.GetString(rf, "type") {
	case "json_schema":
		schema := rf["json_schema"]
		if sm, ok := jsonx.AsMap(schema); ok {
			out := map[string]any{"type": "json_schema"}
			if s, ok := sm["schema"]; ok {
				out["schema"] = s
			}
			if n := jsonx.GetString(sm, "name"); n != "" {
				out["name"] = n
			}
			if st, ok := sm["strict"]; ok {
				out["strict"] = st
			}
			return out
		}
		if schema != nil {
			return map[string]any{"type": "json_schema", "schema": schema}
		}
	case "json_object":
		return map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}
	}
	return nil
}

func anthropicOutputFormatToOpenAI(of map[string]any) map[string]any {
	if jsonx.GetString(of, "type") != "json_schema" {
		return nil
	}
	js := map[string]any{"name": "structured_output", "schema": of["schema"]}
	if n := jsonx.GetString(of, "name"); n != "" {
		js["name"] = n
	}
	if st, ok := of["strict"]; ok {
		js["strict"] = st
	}
	return map[string]any{"type": "json_schema", "json_schema": js}
}

func truncateToolName(name string) string {
	if len(name) <= openaiMaxToolName {
		return name
	}
	sum := sha256.Sum256([]byte(name))
	h := hex.EncodeToString(sum[:])[:8]
	prefix := name
	if len(prefix) > 55 {
		prefix = prefix[:55]
	}
	return prefix + "_" + h
}

// anthropicToolNameMap maps each declared chat tool name to the name that will
// appear on the Messages wire, so history and tool_choice can reference the
// same spelling. A name that is already legal maps to itself.
func anthropicToolNameMap(v any) map[string]string {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	out := map[string]string{}
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok || jsonx.GetString(m, "type") != "function" {
			continue
		}
		fn, ok := jsonx.AsMap(m["function"])
		if !ok {
			continue
		}
		name := jsonx.GetString(fn, "name")
		if name != "" {
			out[name] = sanitizeToolNameForAnthropic(name)
		}
	}
	return out
}

// anthropicHostedTools are the Anthropic-native tool types. Each owns a
// reserved name the provider defines, so a tool of one of these types is
// forwarded as it stands rather than rewritten into a function tool.
var anthropicHostedTools = []string{
	"web_search", "bash", "text_editor", "code_execution",
	"web_fetch", "memory", "tool_search_tool",
}

func isAnthropicHostedTool(t string) bool {
	for _, hosted := range anthropicHostedTools {
		if strings.HasPrefix(t, hosted) {
			return true
		}
	}
	return false
}

func openaiToolsToAnthropic(v any, nameMap map[string]string) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	out := make([]any, 0, len(list))
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			out = append(out, item)
			continue
		}
		if jsonx.GetString(m, "type") == "function" {
			if fn, ok := jsonx.AsMap(m["function"]); ok {
				tool := map[string]any{"name": sanitizeToolNameForAnthropic(jsonx.GetString(fn, "name"))}
				if d := jsonx.GetString(fn, "description"); d != "" {
					tool["description"] = d
				}
				tool["input_schema"] = sanitizeInputSchemaForAnthropic(fn["parameters"])
				if st, ok := fn["strict"]; ok {
					tool["strict"] = st
				}
				out = append(out, tool)
				continue
			}
		}
		if _, hasSchema := m["input_schema"]; hasSchema {
			out = append(out, m)
			continue
		}
		out = append(out, m)
	}
	return out
}

func anthropicToolsToOpenAI(v any) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	out := make([]any, 0, len(list))
	for i, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			out = append(out, item)
			continue
		}
		t := jsonx.GetString(m, "type")
		if isAnthropicHostedTool(t) {
			// A hosted tool owns a reserved name the provider defines; rewriting
			// it into a function tool would drop the type that names it.
			out = append(out, m)
			continue
		}
		if t == "function" && m["function"] != nil {
			out = append(out, m)
			continue
		}
		name := jsonx.GetString(m, "name")
		if strings.TrimSpace(name) == "" {
			name = fmt.Sprintf("unnamed_tool_%d", i)
		}
		trunc := truncateToolName(name)
		fn := map[string]any{"name": trunc}
		if s, ok := m["input_schema"]; ok {
			fn["parameters"] = s
		}
		if d := jsonx.GetString(m, "description"); d != "" {
			fn["description"] = d
		}
		if st, ok := m["strict"]; ok {
			fn["strict"] = st
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

func mapToolChoiceToAnthropic(toolChoice, parallel any, nameMap map[string]string) (map[string]any, bool) {
	var out map[string]any
	switch t := toolChoice.(type) {
	case nil:
	case string:
		switch t {
		case "auto":
			out = map[string]any{"type": "auto"}
		case "required":
			out = map[string]any{"type": "any"}
		case "none":
			out = map[string]any{"type": "none"}
		}
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto":
			out = map[string]any{"type": "auto"}
		case "required", "any":
			out = map[string]any{"type": "any"}
		case "none":
			out = map[string]any{"type": "none"}
		default:
			name := ""
			if fn, ok := jsonx.AsMap(t["function"]); ok {
				name = jsonx.GetString(fn, "name")
			}
			if name == "" {
				name = jsonx.GetString(t, "name")
			}
			if name != "" {
				// The name must be the one the declared tool now carries, or
				// the choice points at a tool the upstream does not see.
				if mapped, ok := nameMap[name]; ok {
					name = mapped
				} else {
					name = sanitizeToolNameForAnthropic(name)
				}
				out = map[string]any{"type": "tool", "name": name}
			}
		}
	}
	if p, ok := jsonx.Bool(parallel); ok && jsonx.String(toolChoice) != "none" {
		if out == nil {
			out = map[string]any{"type": "auto"}
		}
		out["disable_parallel_tool_use"] = !p
	}
	return out, out != nil
}

func anthropicToolChoiceToOpenAI(v any) any {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return v
	}
	switch jsonx.GetString(m, "type") {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "none":
		return "none"
	case "tool":
		// The name names a tool the request also declares, and that
		// declaration is truncated to OpenAI's 64-char limit above — so this
		// reference must be truncated the same way, or it points at a tool the
		// upstream does not see and the request is rejected.
		name := truncateToolName(jsonx.GetString(m, "name"))
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}
	default:
		return v
	}
}

func openaiMessagesToAnthropic(msgs []any, nameMap map[string]string) (system []map[string]any, out []any, err error) {
	var pendingToolResults []any
	flushTools := func() {
		if len(pendingToolResults) == 0 {
			return
		}
		out = append(out, map[string]any{"role": "user", "content": pendingToolResults})
		pendingToolResults = nil
	}
	// Anthropic requires the wire to alternate user and assistant. Two
	// consecutive turns of one role are merged into a single message with
	// combined content, which is how the caller's intent reads anyway; left
	// apart they are a request the upstream rejects outright.
	appendMessage := func(role string, content any) {
		if last, ok := lastOfRole(out, role); ok {
			last["content"] = mergeContent(last["content"], content)
			return
		}
		out = append(out, map[string]any{"role": role, "content": content})
	}
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		role := jsonx.GetString(m, "role")
		switch role {
		case "system", "developer":
			flushTools()
			system = append(system, openaiSystemToAnthropic(m)...)
		case "tool":
			pendingToolResults = append(pendingToolResults, openaiToolResultToAnthropic(m))
		case "user":
			flushTools()
			appendMessage("user", openaiUserContentToAnthropic(m["content"]))
		case "assistant":
			flushTools()
			msg := openaiAssistantToAnthropic(m, nameMap)
			appendMessage("assistant", msg["content"])
			if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
				if last, ok := lastOfRole(out, "assistant"); ok {
					last["tool_calls"] = append(existingToolCalls(last), tcs...)
				}
			}
		default:
			flushTools()
			out = append(out, m)
		}
	}
	flushTools()
	return system, out, nil
}

// lastOfRole returns the trailing message when it already carries role, which
// is the message a following turn of the same role merges into.
func lastOfRole(out []any, role string) (map[string]any, bool) {
	if len(out) == 0 {
		return nil, false
	}
	last, ok := jsonx.AsMap(out[len(out)-1])
	if !ok || jsonx.GetString(last, "role") != role {
		return nil, false
	}
	return last, true
}

func existingToolCalls(msg map[string]any) []any {
	if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
		return tcs
	}
	return nil
}

// mergeContent combines two content values for one merged message: two strings
// join, and a string with a block list becomes a block list.
func mergeContent(a, b any) any {
	as, aok := a.(string)
	bs, bok := b.(string)
	if aok && bok {
		return as + bs
	}
	var blocks []any
	if aok {
		blocks = append(blocks, map[string]any{"type": "text", "text": as})
	} else if list, ok := jsonx.AsSlice(a); ok {
		blocks = append(blocks, list...)
	}
	if bok {
		blocks = append(blocks, map[string]any{"type": "text", "text": bs})
	} else if list, ok := jsonx.AsSlice(b); ok {
		blocks = append(blocks, list...)
	}
	return blocks
}

func openaiSystemToAnthropic(m map[string]any) []map[string]any {
	switch c := m["content"].(type) {
	case string:
		// Anthropic rejects an empty text block, so an empty system turn
		// contributes nothing rather than a block the upstream will 400 on.
		if c == "" {
			return nil
		}
		return []map[string]any{{"type": "text", "text": c}}
	case []any:
		var out []map[string]any
		for _, p := range c {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			if jsonx.GetString(pm, "type") == "text" || jsonx.GetString(pm, "text") != "" {
				text := jsonx.GetString(pm, "text")
				if text == "" {
					continue
				}
				block := map[string]any{"type": "text", "text": text}
				if cc, ok := pm["cache_control"]; ok {
					block["cache_control"] = cc
				}
				out = append(out, block)
			}
		}
		return out
	default:
		if c != nil {
			if s := jsonx.String(c); s != "" {
				return []map[string]any{{"type": "text", "text": s}}
			}
		}
		return nil
	}
}

func openaiUserContentToAnthropic(content any) any {
	switch c := content.(type) {
	case string:
		// Anthropic rejects an empty text block, and there is no way to pass
		// one through: an empty user turn is rewritten to a placeholder so the
		// request is not refused outright.
		if strings.TrimSpace(c) == "" {
			return emptyTextPlaceholder
		}
		return c
	case []any:
		var blocks []any
		for _, p := range c {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			switch jsonx.GetString(pm, "type") {
			case "text", "":
				text := jsonx.GetString(pm, "text")
				if text == "" && jsonx.GetString(pm, "type") == "" {
					continue
				}
				if strings.TrimSpace(text) == "" {
					text = emptyTextPlaceholder
				}
				block := map[string]any{"type": "text", "text": text}
				// A prompt-cache breakpoint is a request feature, not content;
				// dropping it silently un-caches the prefix the caller marked.
				if cc, ok := pm["cache_control"]; ok {
					block["cache_control"] = cc
				}
				blocks = append(blocks, block)
			case "image_url":
				if img := openaiImageToAnthropic(pm); img != nil {
					blocks = append(blocks, img)
				}
			case "image":
				blocks = append(blocks, pm)
			case "document":
				blocks = append(blocks, pm)
			case "file":
				// A chat `file` part carries a data URL or file id; the
				// Messages wire takes a document block with a source. Passing
				// the chat shape through is a spelling Anthropic rejects.
				if doc := openaiFileToAnthropicDocument(pm); doc != nil {
					blocks = append(blocks, doc)
				}
			}
		}
		if len(blocks) == 1 {
			if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
				// A single text block collapses to a plain string for a stable
				// prompt prefix — unless it carries a cache_control breakpoint,
				// which only a block can hold.
				if _, hasCache := b["cache_control"]; !hasCache {
					return jsonx.GetString(b, "text")
				}
			}
		}
		return blocks
	default:
		return content
	}
}

// openaiFileToAnthropicDocument converts a chat `file` part to an Anthropic
// document block. A data URL becomes base64 source with the media type split
// out; a bare file id has no Messages equivalent and is dropped rather than
// forwarded in a spelling the target rejects.
func openaiFileToAnthropicDocument(pm map[string]any) map[string]any {
	inner, _ := jsonx.AsMap(pm["file"])
	data := ""
	filename := ""
	if inner != nil {
		data = jsonx.GetString(inner, "file_data")
		filename = jsonx.GetString(inner, "filename")
	}
	if data == "" {
		data = jsonx.GetString(pm, "file_data")
	}
	if data == "" {
		return nil
	}
	doc := map[string]any{}
	if strings.HasPrefix(data, "data:") {
		media, payload := parseDataURL(data)
		doc["type"] = "document"
		doc["source"] = map[string]any{
			"type":       "base64",
			"media_type": media,
			"data":       payload,
		}
	} else {
		doc["type"] = "document"
		doc["source"] = map[string]any{"type": "base64", "media_type": "application/pdf", "data": data}
	}
	if filename != "" {
		doc["title"] = filename
	}
	return doc
}

func openaiImageToAnthropic(pm map[string]any) map[string]any {
	var url string
	if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
		url = jsonx.GetString(iu, "url")
	} else if s, ok := pm["image_url"].(string); ok {
		url = s
	}
	if url == "" {
		return nil
	}
	if strings.HasPrefix(url, "data:") {
		media, data := parseDataURL(url)
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": media,
				"data":       data,
			},
		}
	}
	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type": "url",
			"url":  url,
		},
	}
}

func parseDataURL(url string) (media, data string) {
	media = "image/jpeg"
	rest := strings.TrimPrefix(url, "data:")
	if i := strings.Index(rest, ";base64,"); i >= 0 {
		if i > 0 {
			media = rest[:i]
		}
		data = rest[i+len(";base64,"):]
		return media, data
	}
	if i := strings.Index(rest, ","); i >= 0 {
		data = rest[i+1:]
	}
	return media, data
}

func openaiAssistantToAnthropic(m map[string]any, nameMap map[string]string) map[string]any {
	var blocks []any
	if tbs, ok := jsonx.AsSlice(m["thinking_blocks"]); ok {
		for _, tb := range tbs {
			if tm, ok := jsonx.AsMap(tb); ok {
				// Anthropic verifies a thinking block's signature; a block
				// without one is rejected, so an unsigned block is dropped
				// rather than forwarded as a request the upstream will 400.
				if jsonx.GetString(tm, "type") == "thinking" && jsonx.GetString(tm, "signature") == "" {
					continue
				}
				blocks = append(blocks, tm)
			}
		}
	}
	switch c := m["content"].(type) {
	case string:
		if c != "" {
			blocks = append(blocks, map[string]any{"type": "text", "text": c})
		}
	case []any:
		for _, p := range c {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			typ := jsonx.GetString(pm, "type")
			if typ == "text" || typ == "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
			} else if typ == "thinking" || typ == "redacted_thinking" {
				blocks = append(blocks, pm)
			}
		}
	}
	if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok {
		for _, tc := range tcs {
			tm, ok := jsonx.AsMap(tc)
			if !ok {
				continue
			}
			fn, _ := jsonx.AsMap(tm["function"])
			input := any(map[string]any{})
			if fn != nil {
				if args := jsonx.GetString(fn, "arguments"); args != "" {
					var parsed any
					if err := json.Unmarshal([]byte(args), &parsed); err == nil {
						input = parsed
					} else {
						input = map[string]any{}
					}
				}
			}
			name := ""
			if fn != nil {
				name = jsonx.GetString(fn, "name")
				if mapped, ok := nameMap[name]; ok {
					name = mapped
				} else {
					name = sanitizeToolNameForAnthropic(name)
				}
			}
			blocks = append(blocks, map[string]any{
				"type":  "tool_use",
				"id":    jsonx.GetString(tm, "id"),
				"name":  name,
				"input": input,
			})
		}
	}
	out := map[string]any{"role": "assistant"}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
			out["content"] = jsonx.GetString(b, "text")
			return out
		}
	}
	if len(blocks) == 0 {
		// Anthropic rejects an empty text block, and there is no way to pass
		// one through, so an empty assistant turn is rewritten to a placeholder
		// rather than sent as content the upstream will 400 on.
		out["content"] = emptyTextPlaceholder
		return out
	}
	out["content"] = blocks
	return out
}

func openaiToolResultToAnthropic(m map[string]any) map[string]any {
	block := map[string]any{
		"type":        "tool_result",
		"tool_use_id": jsonx.GetString(m, "tool_call_id"),
	}
	switch c := m["content"].(type) {
	case string:
		block["content"] = c
	case []any:
		// A tool result may carry media. Its parts use the chat vocabulary,
		// which is not what a Messages tool_result reads, so each part is
		// converted rather than forwarded verbatim.
		block["content"] = openaiToolResultPartsToAnthropic(c)
	default:
		block["content"] = c
	}
	return block
}

// openaiToolResultPartsToAnthropic converts a chat tool message's content parts
// to the block list an Anthropic tool_result carries.
func openaiToolResultPartsToAnthropic(parts []any) []any {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "text", "":
			out = append(out, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
		case "image_url":
			if img := openaiImageToAnthropic(pm); img != nil {
				out = append(out, img)
			}
		case "image":
			out = append(out, pm)
		case "document":
			out = append(out, pm)
		case "file":
			if doc := openaiFileToAnthropicDocument(pm); doc != nil {
				out = append(out, doc)
			}
		}
	}
	return out
}

func anthropicMessagesToOpenAI(msgs []any) []any {
	var out []any
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		role := jsonx.GetString(m, "role")
		if role == "system" {
			out = append(out, map[string]any{"role": "system", "content": anthropicContentToText(m["content"])})
			continue
		}
		if role == "user" {
			content := m["content"]
			if s, ok := content.(string); ok {
				out = append(out, map[string]any{"role": "user", "content": s})
				continue
			}
			list, ok := jsonx.AsSlice(content)
			if !ok {
				out = append(out, map[string]any{"role": "user", "content": content})
				continue
			}
			var userParts []any
			for _, b := range list {
				bm, ok := jsonx.AsMap(b)
				if !ok {
					continue
				}
				switch jsonx.GetString(bm, "type") {
				case "text":
					userParts = append(userParts, map[string]any{"type": "text", "text": jsonx.GetString(bm, "text")})
				case "image", "document":
					// Both carry a source the chat wire takes as an image_url
					// data URL (a PDF rides the same spelling), and dropping
					// one would lose an attachment the caller sent.
					if img := anthropicImageToOpenAI(bm); img != nil {
						userParts = append(userParts, img)
					}
				case "tool_result":
					out = append(out, map[string]any{
						"role":         "tool",
						"tool_call_id": jsonx.GetString(bm, "tool_use_id"),
						"content":      toolResultContent(bm["content"]),
					})
				}
			}
			if len(userParts) == 1 {
				if p, ok := jsonx.AsMap(userParts[0]); ok && jsonx.GetString(p, "type") == "text" {
					out = append(out, map[string]any{"role": "user", "content": jsonx.GetString(p, "text")})
					continue
				}
			}
			if len(userParts) > 0 {
				out = append(out, map[string]any{"role": "user", "content": userParts})
			}
			continue
		}
		if role == "assistant" {
			out = append(out, anthropicAssistantToOpenAI(m))
		}
	}
	return out
}

func anthropicAssistantToOpenAI(m map[string]any) map[string]any {
	msg := map[string]any{"role": "assistant"}
	content := m["content"]
	if s, ok := content.(string); ok {
		msg["content"] = s
		return msg
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		msg["content"] = content
		return msg
	}
	var texts []string
	var toolCalls []any
	var thinking []any
	for _, b := range list {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "text":
			texts = append(texts, jsonx.GetString(bm, "text"))
		case "tool_use":
			args, _ := json.Marshal(bm["input"])
			if bm["input"] == nil {
				args = []byte("{}")
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   jsonx.GetString(bm, "id"),
				"type": "function",
				"function": map[string]any{
					"name":      truncateToolName(jsonx.GetString(bm, "name")),
					"arguments": string(args),
				},
			})
		case "thinking", "redacted_thinking":
			thinking = append(thinking, bm)
		}
	}
	if len(texts) == 0 {
		msg["content"] = nil
	} else {
		msg["content"] = strings.Join(texts, "")
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	if len(thinking) > 0 {
		msg["thinking_blocks"] = thinking
		var rc []string
		for _, t := range thinking {
			if tm, ok := jsonx.AsMap(t); ok {
				if s := jsonx.GetString(tm, "thinking"); s != "" {
					rc = append(rc, s)
				}
			}
		}
		if len(rc) > 0 {
			msg["reasoning_content"] = strings.Join(rc, "")
		}
	}
	return msg
}

func anthropicImageToOpenAI(bm map[string]any) map[string]any {
	src, ok := jsonx.AsMap(bm["source"])
	if !ok {
		return nil
	}
	var url string
	switch jsonx.GetString(src, "type") {
	case "base64":
		media := jsonx.GetString(src, "media_type")
		if media == "" {
			media = "image/jpeg"
		}
		url = "data:" + media + ";base64," + jsonx.GetString(src, "data")
	case "url":
		url = jsonx.GetString(src, "url")
	}
	if url == "" {
		return nil
	}
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}

func anthropicContentToText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, p := range c {
			if pm, ok := jsonx.AsMap(p); ok {
				parts = append(parts, jsonx.GetString(pm, "text"))
			}
		}
		return strings.Join(parts, "\n")
	default:
		return jsonx.String(c)
	}
}

// toolResultContent renders one Anthropic tool_result's content for the chat
// wire, where a tool message carries the result.
//
// Text-only results keep the legacy flat string (so prompt prefixes stay
// stable), but a result that returned media — a screenshot from a computer-use
// tool, a generated file — cannot be represented that way: the media would be
// flattened to its text and silently lost, and an Anthropic image block passed
// through verbatim is not part of the chat vocabulary, so an upstream parsing
// the chat vocabulary rejects the whole request. Such a result therefore
// becomes a content-part array, the spelling every chat upstream accepts for a
// message that carries more than text.
func toolResultContent(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		parts := make([]any, 0, len(t))
		texts := make([]string, 0, len(t))
		media := false
		for _, p := range t {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			switch jsonx.GetString(pm, "type") {
			case "text":
				texts = append(texts, jsonx.GetString(pm, "text"))
				parts = append(parts, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
			case "image", "document":
				// A block whose source cannot be rendered is dropped rather
				// than forwarded in a spelling the target rejects.
				if img := anthropicImageToOpenAI(pm); img != nil {
					media = true
					parts = append(parts, img)
				}
			}
		}
		if !media {
			if len(texts) > 0 {
				return strings.Join(texts, "\n")
			}
			break
		}
		return parts
	}
	if v == nil {
		return ""
	}
	return v
}

func prependSystem(msgs []any, sys any) []any {
	text := anthropicContentToText(sys)
	if text == "" {
		return msgs
	}
	return append([]any{map[string]any{"role": "system", "content": text}}, msgs...)
}
