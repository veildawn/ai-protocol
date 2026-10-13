package chatmessages

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/dialectvocab"
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

// sanitizeToolUseIDForAnthropic makes a chat tool-call id legal on the Messages
// wire: a tool_use id may carry only [a-zA-Z0-9_-], and Anthropic refuses one
// with any other character — a chat caller's "call.abc" is rewritten rather
// than passed through, both on the call and on its result. An id with no legal
// character at all still needs one, so it is named.
func sanitizeToolUseIDForAnthropic(id string) string {
	var b strings.Builder
	b.Grow(len(id))
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if out := b.String(); out != "" {
		return out
	}
	return "tool_use_id"
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
	// The Messages system prompt is always a block list — one text block per
	// system turn — never the folded string form, which is a spelling the
	// reference fold does not produce.
	if len(system) > 0 {
		out["system"] = system
	}

	// Messages demands an integer of at least 1. A whole-number ceiling rides
	// through as written; a fractional one is rounded (halves to even, the
	// tie-break Python's round applies) and clamped up to 1 rather than sent
	// as a value the upstream rejects.
	if v, ok := body["max_completion_tokens"]; ok {
		out["max_tokens"] = coerceMaxTokens(v)
	} else if v, ok := body["max_tokens"]; ok {
		out["max_tokens"] = coerceMaxTokens(v)
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	// The Messages wire defaults to non-streaming, so an explicit false is the
	// default itself and is omitted; only a request that actually streams
	// says so.
	if s, ok := body["stream"].(bool); ok && s {
		out["stream"] = true
	}
	if v, ok := body["stop"]; ok {
		if seq := mapStopSequences(v, dropParams); seq != nil {
			out["stop_sequences"] = seq
		}
	}
	if v, ok := body["tools"]; ok {
		tools, err := openaiToolsToAnthropic(v, nameMap)
		if err != nil {
			return nil, err
		}
		out["tools"] = tools
	}
	if v, ok := body["web_search_options"]; ok {
		hosted, err := mapWebSearchOptionsToAnthropic(v)
		if err != nil {
			return nil, err
		}
		if hosted != nil {
			tools, _ := jsonx.AsSlice(out["tools"])
			out["tools"] = append(tools, hosted)
		}
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
		mapped, err := responseFormatToAnthropic(rf)
		if err != nil {
			return nil, err
		}
		if mapped != nil {
			out["output_format"] = mapped
		}
	}
	// A Messages turn with thinking enabled needs room for the thinking plus
	// the answer, so a request that named no ceiling gets one derived from the
	// thinking budget rather than none at all.
	if _, hasMax := out["max_tokens"]; !hasMax {
		if budget, ok := dialectvocab.ThinkingBudget(out["thinking"]); ok {
			out["max_tokens"] = budget + dialectvocab.DefaultMaxTokens
		}
	}
	// The chat context_management list folds into the dict form the Messages
	// dialect reads; a value with nothing the target can act on is dropped
	// rather than forwarded in a spelling the upstream rejects.
	if v, ok := body["context_management"]; ok {
		if cm := mapContextManagementToAnthropic(v); cm != nil {
			out["context_management"] = cm
		}
	}
	copyIf(body, out, "speed")
	// A top-level cache_control marks the request for automatic prompt caching.
	copyIf(body, out, "cache_control")
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
		// An empty stop list names nothing to stop on; the chat wire omits
		// the key rather than carrying an empty one.
		if !emptyStopSequences(v) {
			out["stop"] = v
		}
	}
	if v, ok := body["tools"]; ok {
		tools, hadWebSearch := anthropicToolsToOpenAI(v)
		if len(tools) > 0 {
			out["tools"] = tools
		}
		if hadWebSearch {
			// The hosted web-search tool has no chat spelling as a tool; the
			// search capability rides the request as web_search_options.
			out["web_search_options"] = map[string]any{}
		}
	}
	if v, ok := body["tool_choice"]; ok {
		if !emptyToolChoice(v) {
			tc, err := anthropicToolChoiceToOpenAI(v)
			if err != nil {
				return nil, err
			}
			out["tool_choice"] = tc
		}
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

// coerceMaxTokens makes a chat ceiling legal on the Messages wire, which wants
// an integer no smaller than 1. A value written as a whole number passes
// through untouched — including 0 or a negative ceiling, which the reference
// fold also passes through — while a fractional one is rounded (halves to
// even, the tie-break Python's round applies) and clamped up to 1 rather than
// sent as a value the upstream rejects.
func coerceMaxTokens(v any) any {
	switch t := v.(type) {
	case json.Number:
		// An integer literal stays integral however large it is written, the
		// way Python's json reads it back as an int.
		if !strings.ContainsAny(t.String(), ".eE") {
			if i, err := t.Int64(); err == nil {
				return i
			}
			return v
		}
		f, err := t.Float64()
		if err != nil {
			return v
		}
		return clampedTokenRound(f)
	case float64:
		if t == math.Trunc(t) {
			return v
		}
		return clampedTokenRound(t)
	default:
		return v
	}
}

// clampedTokenRound rounds and floors the result at 1; a magnitude no int64
// can hold rides through as it was written rather than wrapping.
func clampedTokenRound(f float64) any {
	if math.IsNaN(f) || math.Abs(f) >= 1<<62 {
		return f
	}
	n := int64(roundHalfEven(f))
	if n < 1 {
		n = 1
	}
	return n
}

// roundHalfEven rounds to the nearest integer with halves going to the even
// neighbour.
func roundHalfEven(f float64) float64 {
	r := math.Round(f)
	if math.Abs(f-math.Trunc(f)) == 0.5 && math.Mod(r, 2) != 0 {
		r -= math.Copysign(1, f)
	}
	return r
}

func mapStopSequences(v any, drop bool) []any {
	switch t := v.(type) {
	case string:
		// An empty stop is a real stop the caller named; only a non-empty
		// whitespace-only one is droppable — the reference reads isspace(),
		// which is false for the empty string.
		if drop && t != "" && strings.TrimSpace(t) == "" {
			return nil
		}
		return []any{t}
	case []any:
		out := make([]any, 0, len(t))
		for _, item := range t {
			s := jsonx.String(item)
			if drop && s != "" && strings.TrimSpace(s) == "" {
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

// webSearchMaxUses is the search-context size to allowed-search-count table
// the hosted web-search tool carries.
var webSearchMaxUses = map[string]int{
	"low": 1, "medium": 5, "high": 10,
}

// mapWebSearchOptionsToAnthropic declares the chat web_search_options param as
// the hosted web-search tool the Messages dialect reads instead: the search
// capability rides the tools list, so the param has no target spelling of its
// own and is consumed here rather than silently dropped. A non-dict value
// names nothing mappable and is dropped.
func mapWebSearchOptionsToAnthropic(v any) (map[string]any, error) {
	wso, ok := jsonx.AsMap(v)
	if !ok {
		return nil, nil
	}
	tool := map[string]any{"type": "web_search_20250305", "name": "web_search"}
	if ul, ok := jsonx.AsMap(wso["user_location"]); ok {
		// Only the approximation fields the target defines ride along; a
		// location that carries no approximation carries nothing.
		if approx, ok := jsonx.AsMap(ul["approximate"]); ok {
			loc := map[string]any{"type": "approximate"}
			for _, k := range []string{"city", "country", "region", "timezone"} {
				if val, ok := approx[k]; ok {
					loc[k] = val
				}
			}
			tool["user_location"] = loc
		}
	}
	if size := jsonx.GetString(wso, "search_context_size"); size != "" {
		n, ok := webSearchMaxUses[size]
		if !ok {
			return nil, fmt.Errorf("web_search_options.search_context_size: unknown value %q", size)
		}
		tool["max_uses"] = n
	}
	return tool, nil
}

// mapContextManagementToAnthropic folds the chat context_management parameter
// into the shape the Messages dialect reads: a chat compaction entry becomes
// the edit the target performs, keyed the way it expects. The Anthropic dict
// form passes through, and a value with no foldable entry is dropped rather
// than forwarded in a spelling the upstream rejects.
func mapContextManagementToAnthropic(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if _, ok := t["edits"]; ok {
			return t
		}
	case []any:
		var edits []any
		for _, item := range t {
			e, ok := jsonx.AsMap(item)
			if !ok || jsonx.GetString(e, "type") != "compaction" {
				continue
			}
			edit := map[string]any{"type": "compact_20260112"}
			if n, ok := jsonx.Int(e["compact_threshold"]); ok {
				edit["trigger"] = map[string]any{"type": "input_tokens", "value": n}
			}
			for k, val := range e {
				if k == "type" || k == "compact_threshold" {
					continue
				}
				edit[k] = val
			}
			edits = append(edits, edit)
		}
		if len(edits) > 0 {
			return map[string]any{"edits": edits}
		}
	}
	return nil
}

// reasoningEffortString reads a reasoning_effort value, which arrives either as
// a bare string or as the Responses-shaped {"effort": ...} dict a bridged
// caller sends. An explicitly sent effort — the empty string included — is
// reported as present; what to do with it is the caller's decision.
func reasoningEffortString(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case map[string]any:
		if e, present := t["effort"]; present {
			if s, ok := e.(string); ok {
				return s, true
			}
		}
	}
	return "", false
}

// validUserID reports whether a chat `user` value may ride Messages metadata
// as user_id: the target refuses a value shaped like a full email address or a
// phone number, and anything else — including a bare word with an @ in it, or
// digits with a + mid-string — is simply an id.
var (
	anthropicUserIDEmail = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	anthropicUserIDPhone = regexp.MustCompile(`^\+?[\d\s()-]{7,}$`)
)

func validUserID(v string) bool {
	return !anthropicUserIDEmail.MatchString(v) && !anthropicUserIDPhone.MatchString(v)
}

func thinkingFromEffort(effort string) (map[string]any, error) {
	if effort == "none" {
		return nil, nil
	}
	// An empty effort is a value the caller sent and no budget maps to; it
	// fails the request rather than silently asking for no thinking.
	if effort == "" {
		return nil, fmt.Errorf("unmapped reasoning effort: %q", effort)
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

func responseFormatToAnthropic(rf map[string]any) (map[string]any, error) {
	switch jsonx.GetString(rf, "type") {
	case "json_schema":
		sm, ok := jsonx.AsMap(rf["json_schema"])
		if !ok {
			return nil, nil
		}
		schema, ok := jsonx.AsMap(sm["schema"])
		if !ok {
			return nil, nil
		}
		schema, _ = deepCopy(schema).(map[string]any)
		// The Messages output format is the schema alone: the target compiles
		// it as written and refuses a $ref it cannot follow, so the
		// definitions the schema carries are inlined into it first, and the
		// chat form's name and strict flag have no target spelling and stay
		// behind.
		if err := inlineOutputSchemaDefs(schema); err != nil {
			return nil, err
		}
		return map[string]any{"type": "json_schema", "schema": filterOutputSchemaForAnthropic(schema)}, nil
	}
	// json_object names no schema to compile, and the Messages dialect has no
	// spelling for "any object will do" — the reference fold sends nothing at
	// all for it (neither an output format nor a simulating tool), so the
	// request carries no output_format rather than one describing a schema the
	// caller never wrote.
	return nil, nil
}

// anthropicOutputFormatToOpenAI folds the Messages output format back to the
// chat response_format. The chat side reads it in strict mode: the schema is
// rewritten to what strictness requires — every object closed with
// additionalProperties:false and every property named as required — the name
// is the conventional one, and an output format without a schema carries
// nothing to fold.
func anthropicOutputFormatToOpenAI(of map[string]any) map[string]any {
	if jsonx.GetString(of, "type") != "json_schema" {
		return nil
	}
	schema, ok := jsonx.AsMap(of["schema"])
	if !ok || len(schema) == 0 {
		return nil
	}
	schema, _ = deepCopy(schema).(map[string]any)
	strictifySchemaForChat(schema)
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "structured_output",
			"schema": schema,
			"strict": true,
		},
	}
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
// appear on the Messages wire, so the declaration, history and tool_choice all
// reference the same spelling. A name that is already legal maps to itself.
//
// Two chat names can sanitize onto the same wire name — foo/bar and foo_bar
// both want foo_bar — and the Messages dialect refuses a request whose declared
// tool names are not unique. A rewritten name that collides therefore carries
// an _2/_3 suffix, and a name that is already legal reserves its slot before
// any rewrite is considered, so which tool keeps the bare name does not depend
// on declaration order.
func anthropicToolNameMap(v any) map[string]string {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var names []string
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok || jsonx.GetString(m, "type") != "function" {
			continue
		}
		fn, ok := jsonx.AsMap(m["function"])
		if !ok {
			continue
		}
		if name := jsonx.GetString(fn, "name"); name != "" {
			names = append(names, name)
		}
	}
	// First pass: a legal name is its own wire name wherever it is declared.
	used := map[string]struct{}{}
	for _, name := range names {
		if sanitizeToolNameForAnthropic(name) == name {
			used[name] = struct{}{}
		}
	}
	out := map[string]string{}
	for _, name := range names {
		candidate := sanitizeToolNameForAnthropic(name)
		if candidate == name {
			out[name] = name
			continue
		}
		// The same chat name declared twice maps to the one rewrite.
		if _, done := out[name]; done {
			continue
		}
		unique := candidate
		for n := 2; ; n++ {
			if _, taken := used[unique]; !taken {
				break
			}
			suffix := "_" + strconv.Itoa(n)
			head := candidate
			if len(head) > anthropicMaxToolName-len(suffix) {
				head = head[:anthropicMaxToolName-len(suffix)]
			}
			unique = head + suffix
		}
		out[name] = unique
		used[unique] = struct{}{}
	}
	return out
}

func openaiToolsToAnthropic(v any, nameMap map[string]string) ([]any, error) {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil, nil
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
				name := jsonx.GetString(fn, "name")
				if mapped, ok := nameMap[name]; ok {
					name = mapped
				} else {
					name = sanitizeToolNameForAnthropic(name)
				}
				// The Messages spelling of a function tool: type "custom",
				// never the chat `strict` flag, and a cache breakpoint
				// lifted from the function level onto the tool where the
				// Messages wire reads it — dropped, it would silently
				// un-cache the prefix the caller marked.
				inputSchema, err := sanitizeInputSchemaForAnthropic(fn["parameters"])
				if err != nil {
					return nil, err
				}
				tool := map[string]any{
					"name":         name,
					"input_schema": inputSchema,
					"type":         "custom",
				}
				if d := jsonx.GetString(fn, "description"); d != "" {
					tool["description"] = d
				}
				cc, hasCC := m["cache_control"]
				if !hasCC {
					if fnCC, ok := jsonx.AsMap(fn["cache_control"]); ok {
						cc, hasCC = fnCC, true
					}
				}
				if hasCC {
					tool["cache_control"] = cc
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
	return out, nil
}

// anthropicToolsToOpenAI also reports whether the list declared the hosted
// web-search tool: that one folds into the request's web_search_options
// instead of a chat tool, where it has no spelling.
func anthropicToolsToOpenAI(v any) ([]any, bool) {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil, false
	}
	out := make([]any, 0, len(list))
	hadWebSearch := false
	for i, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			out = append(out, item)
			continue
		}
		if isAnthropicWebSearchTool(m) {
			hadWebSearch = true
			continue
		}
		t := jsonx.GetString(m, "type")
		if dialectvocab.IsAnthropicHostedTool(t) {
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
		if d := jsonx.GetString(m, "description"); d != "" {
			fn["description"] = d
		}
		if st, ok := m["strict"]; ok {
			fn["strict"] = st
		}
		// The input schema rides as the function's parameters, and tool keys
		// the function shape has no field for are provider knobs the caller
		// set (a version, a use cap) that ride inside that schema rather than
		// being dropped on the floor. The schema is copied before the merge:
		// the request body is the caller's, not ours to mutate.
		var params map[string]any
		hasParams := false
		if s, ok := jsonx.AsMap(m["input_schema"]); ok {
			params = jsonx.CloneMap(s)
			hasParams = true
		} else {
			params = map[string]any{}
		}
		for k, v := range m {
			switch k {
			case "name", "input_schema", "description", "cache_control", "strict", "type":
				continue
			}
			params[k] = v
			hasParams = true
		}
		if hasParams {
			fn["parameters"] = params
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out, hadWebSearch
}

// isAnthropicWebSearchTool reports whether a Messages tool names the hosted
// web search: the type carries the web_search prefix, or the name is the
// reserved one.
func isAnthropicWebSearchTool(m map[string]any) bool {
	return strings.HasPrefix(jsonx.GetString(m, "type"), "web_search") ||
		jsonx.GetString(m, "name") == "web_search"
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
		_, hasFunction := t["function"]
		if _, hasType := t["type"]; hasType && !hasFunction {
			// A type-keyed choice names a mode. A shape outside the modes the
			// target reads — a type:tool choice carrying its name at the top
			// level among them — names nothing it can act on and is dropped
			// rather than forwarded.
			switch jsonx.GetString(t, "type") {
			case "auto":
				out = map[string]any{"type": "auto"}
			case "required", "any":
				out = map[string]any{"type": "any"}
			case "none":
				out = map[string]any{"type": "none"}
			}
		} else if hasFunction {
			// A choice that carries a function names that function whatever
			// its type field says: the function is the stronger signal.
			if fn, ok := jsonx.AsMap(t["function"]); ok {
				if name := jsonx.GetString(fn, "name"); name != "" {
					// The name must be the one the declared tool now carries,
					// or the choice points at a tool the upstream does not
					// see. A name the request does not declare rides through
					// as written — the target is the one to say it is unknown.
					if mapped, ok := nameMap[name]; ok {
						name = mapped
					}
					out = map[string]any{"type": "tool", "name": name}
				}
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

// emptyToolChoice reports a tool_choice value that names no choice at all:
// the chat wire omits the key rather than carrying an empty one.
func emptyToolChoice(v any) bool {
	switch t := v.(type) {
	case nil:
		return true
	case map[string]any:
		return len(t) == 0
	default:
		return false
	}
}

// emptyStopSequences reports a stop list that names nothing to stop on.
func emptyStopSequences(v any) bool {
	switch t := v.(type) {
	case nil:
		return true
	case []any:
		return len(t) == 0
	case string:
		return t == ""
	default:
		return false
	}
}

func anthropicToolChoiceToOpenAI(v any) (any, error) {
	m, ok := jsonx.AsMap(v)
	if !ok {
		return v, nil
	}
	switch jsonx.GetString(m, "type") {
	case "auto":
		return "auto", nil
	case "any":
		return "required", nil
	case "none":
		return "none", nil
	case "tool":
		// The name names a tool the request also declares, and that
		// declaration is truncated to OpenAI's 64-char limit above — so this
		// reference must be truncated the same way, or it points at a tool the
		// upstream does not see and the request is rejected.
		name := truncateToolName(jsonx.GetString(m, "name"))
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}, nil
	default:
		// An Anthropic-shaped choice the chat dialect has no value for is not
		// passable: forwarded verbatim it is a shape the target rejects, so
		// the request fails here instead.
		return nil, fmt.Errorf("incompatible tool choice param submitted: %v", v)
	}
}

func openaiMessagesToAnthropic(msgs []any, nameMap map[string]string) (system []map[string]any, out []any, err error) {
	var pendingToolResults []any
	// The legacy function protocol carries no ids of its own: the assistant
	// names a function to call and a function-role message answers it, in
	// order. The pair is assigned one id per position — the caller's
	// tool_call_id when the answer carries one, else a deterministic name —
	// so the folded tool_use and tool_result actually reference each other.
	legacyCallIDs, legacyResultIDs := legacyFunctionIDs(msgs)
	legacyCallIdx, legacyResultIdx := 0, 0
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
	// A tool result is user-role content on the Messages wire, so a run of
	// them joins the user turn around it rather than forming its own second
	// consecutive user message.
	flushTools := func() {
		if len(pendingToolResults) == 0 {
			return
		}
		appendMessage("user", pendingToolResults)
		pendingToolResults = nil
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
			block, err := openaiToolResultToAnthropic(m, "")
			if err != nil {
				return nil, nil, err
			}
			pendingToolResults = append(pendingToolResults, block)
		case "function":
			// A legacy function answer is a tool result the same way; without
			// an id of its own it carries the one its position was assigned.
			fallback := ""
			if legacyResultIdx < len(legacyResultIDs) {
				fallback = legacyResultIDs[legacyResultIdx]
			}
			legacyResultIdx++
			block, err := openaiToolResultToAnthropic(m, fallback)
			if err != nil {
				return nil, nil, err
			}
			pendingToolResults = append(pendingToolResults, block)
		case "user":
			flushTools()
			content, err := openaiUserContentToAnthropic(m["content"])
			if err != nil {
				return nil, nil, err
			}
			appendMessage("user", content)
		case "assistant":
			flushTools()
			legacyID := ""
			if _, ok := jsonx.AsMap(m["function_call"]); ok {
				if legacyCallIdx < len(legacyCallIDs) {
					legacyID = legacyCallIDs[legacyCallIdx]
				}
				legacyCallIdx++
			}
			msg, err := openaiAssistantToAnthropic(m, nameMap, legacyID)
			if err != nil {
				return nil, nil, err
			}
			if msg == nil {
				continue
			}
			appendMessage("assistant", msg["content"])
			if tcs, ok := jsonx.AsSlice(msg["tool_calls"]); ok {
				if last, ok := lastOfRole(out, "assistant"); ok {
					last["tool_calls"] = append(existingToolCalls(last), tcs...)
				}
			}
			// A merged assistant run may repeat a tool_call id — history that
			// was recorded twice — and the target requires tool_use ids to be
			// unique within the message, so only the first of each survives.
			if last, ok := lastOfRole(out, "assistant"); ok {
				dedupeToolUseIDs(last)
			}
		default:
			flushTools()
			out = append(out, m)
		}
	}
	flushTools()
	rstripFinalAssistant(out)
	return system, out, nil
}

// legacyFunctionIDs assigns the tool_use id each legacy function_call carries
// and the tool_use_id each function-role answer carries, paired by position.
func legacyFunctionIDs(msgs []any) (callIDs, resultIDs []string) {
	answerIDs := collectLegacyAnswerIDs(msgs)
	calls := 0
	for _, item := range msgs {
		if m, ok := jsonx.AsMap(item); ok && jsonx.GetString(m, "role") == "assistant" {
			if _, ok := jsonx.AsMap(m["function_call"]); ok {
				calls++
			}
		}
	}
	for i := 0; i < calls; i++ {
		id := fmt.Sprintf("legacy_call_%d", i)
		if i < len(answerIDs) && answerIDs[i] != "" {
			id = answerIDs[i]
		}
		callIDs = append(callIDs, id)
	}
	for j := range answerIDs {
		id := answerIDs[j]
		if id == "" {
			id = fmt.Sprintf("legacy_call_%d", j)
		}
		resultIDs = append(resultIDs, id)
	}
	return callIDs, resultIDs
}

// collectLegacyAnswerIDs reads each function-role answer's own tool_call_id,
// sanitized for the wire, in order; an answer without one holds an empty slot.
func collectLegacyAnswerIDs(msgs []any) []string {
	var ids []string
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok || jsonx.GetString(m, "role") != "function" {
			continue
		}
		id := ""
		if tcID := jsonx.GetString(m, "tool_call_id"); tcID != "" {
			id = sanitizeToolUseIDForAnthropic(tcID)
		}
		ids = append(ids, id)
	}
	return ids
}

// dedupeToolUseIDs keeps only the first tool_use block for each id in one
// assistant message's merged content.
func dedupeToolUseIDs(msg map[string]any) {
	blocks, ok := jsonx.AsSlice(msg["content"])
	if !ok {
		return
	}
	seen := map[string]struct{}{}
	kept := blocks[:0]
	for _, b := range blocks {
		bm, ok := jsonx.AsMap(b)
		if !ok || jsonx.GetString(bm, "type") != "tool_use" {
			kept = append(kept, b)
			continue
		}
		id := jsonx.GetString(bm, "id")
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		kept = append(kept, b)
	}
	msg["content"] = kept
}

// rstripFinalAssistant trims trailing whitespace from the last assistant turn:
// Anthropic refuses a final assistant message whose text ends in whitespace,
// string or block alike. A text that trims to nothing was whitespace-only,
// and reads as the placeholder rather than as an empty block the upstream
// rejects.
func rstripFinalAssistant(out []any) {
	if len(out) == 0 {
		return
	}
	last, ok := jsonx.AsMap(out[len(out)-1])
	if !ok || jsonx.GetString(last, "role") != "assistant" {
		return
	}
	switch c := last["content"].(type) {
	case string:
		last["content"] = rstripOrPlaceholder(c)
	case []any:
		for _, b := range c {
			bm, ok := jsonx.AsMap(b)
			if !ok || jsonx.GetString(bm, "type") != "text" {
				continue
			}
			if s, ok := bm["text"].(string); ok {
				bm["text"] = rstripOrPlaceholder(s)
			}
		}
	}
}

func rstripOrPlaceholder(s string) string {
	if t := strings.TrimRightFunc(s, unicode.IsSpace); t != "" {
		return t
	}
	return emptyTextPlaceholder
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

// mergeContent combines two content values for one merged message. Two
// strings become two text blocks — run together they would read as one
// glued word the caller never wrote — and a string with a block list
// becomes a block list.
func mergeContent(a, b any) any {
	as, aok := a.(string)
	bs, bok := b.(string)
	if aok && bok {
		return []any{
			map[string]any{"type": "text", "text": as},
			map[string]any{"type": "text", "text": bs},
		}
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

func openaiUserContentToAnthropic(content any) (any, error) {
	switch c := content.(type) {
	case string:
		// Anthropic rejects an empty text block, and there is no way to pass
		// one through: an empty user turn is rewritten to a placeholder so the
		// request is not refused outright.
		if strings.TrimSpace(c) == "" {
			return emptyTextPlaceholder, nil
		}
		return c, nil
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
				img, err := openaiImageToAnthropic(pm)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, img)
			case "image":
				blocks = append(blocks, pm)
			case "document":
				blocks = append(blocks, pm)
			case "file":
				// A chat `file` part carries an inline payload or a file
				// reference; the Messages wire takes the block that shape
				// names. Passing the chat shape through is a spelling the
				// target rejects, and a part that names nothing fails here
				// rather than losing the attachment silently.
				doc, err := openaiFileToAnthropicDocument(pm)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, doc)
			}
		}
		if len(blocks) == 1 {
			if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
				// A single text block collapses to a plain string for a stable
				// prompt prefix — unless it carries a cache_control breakpoint,
				// which only a block can hold.
				if _, hasCache := b["cache_control"]; !hasCache {
					return jsonx.GetString(b, "text"), nil
				}
			}
		}
		return blocks, nil
	default:
		return content, nil
	}
}

// openaiFileToAnthropicDocument converts a chat `file` part to the Messages
// block the attachment rides. An inline payload (file_data) must be a data
// URL and folds to a base64 document whose media type is the caller's
// format when one is given — the caller's word about the payload is the
// authoritative one — and the payload's own mime otherwise; a reference
// (file_id) folds by what it names — a document file source for document
// formats, an image file source for image formats, a URL document when the
// id is a URL, and otherwise a container upload the target resolves. A part
// without the file object, or naming neither payload nor reference, is
// refused: there is no block to fold and no way to pass nothing through.
func openaiFileToAnthropicDocument(pm map[string]any) (map[string]any, error) {
	inner, ok := jsonx.AsMap(pm["file"])
	if !ok {
		return nil, fmt.Errorf("file part is missing the required 'file' field")
	}
	data := jsonx.GetString(inner, "file_data")
	fileID := jsonx.GetString(inner, "file_id")
	format := jsonx.GetString(inner, "format")
	if data != "" {
		if !strings.HasPrefix(data, "data:") {
			return nil, fmt.Errorf("file part carries file_data that is not a data URL: %q", truncateForError(data))
		}
		media, payload, ok := parseDataURL(data)
		if !ok {
			return nil, fmt.Errorf("file part carries file_data that is not a base64 data URL: %q", truncateForError(data))
		}
		if format != "" {
			media = format
		}
		return map[string]any{
			"type": "document",
			"source": map[string]any{
				"type":       "base64",
				"media_type": media,
				"data":       payload,
			},
		}, nil
	}
	if fileID != "" {
		switch {
		case format == "application/pdf" || format == "text/plain":
			return map[string]any{"type": "document", "source": map[string]any{"type": "file", "file_id": fileID}}, nil
		case format == "image/jpeg" || format == "image/png" || format == "image/gif" || format == "image/webp":
			return map[string]any{"type": "image", "source": map[string]any{"type": "file", "file_id": fileID}}, nil
		case format != "":
			// A format the target cannot name keeps the reference whole: the
			// upload the container resolves is the one shape left.
			return map[string]any{"type": "container_upload", "file_id": fileID}, nil
		case strings.HasPrefix(fileID, "http://") || strings.HasPrefix(fileID, "https://"):
			return map[string]any{"type": "document", "source": map[string]any{"type": "url", "url": fileID}}, nil
		default:
			return map[string]any{"type": "container_upload", "file_id": fileID}, nil
		}
	}
	return nil, fmt.Errorf("file part carries neither file_data nor file_id")
}

// truncateForError keeps an error message bounded when the value is long.
func truncateForError(s string) string {
	if len(s) > 40 {
		return s[:40] + "..."
	}
	return s
}

// openaiImageToAnthropic converts a chat image part to a Messages image
// block. A data URL folds to a base64 source whose media type is the
// caller's format when one is given — the caller's word about the payload is
// the authoritative one — and the payload's own mime otherwise; an https
// (or http, which this codec cannot fetch) URL rides as a url source. Any
// other spelling — an empty url, a scheme the target cannot fetch — fails
// the request rather than passing a block the upstream rejects.
func openaiImageToAnthropic(pm map[string]any) (map[string]any, error) {
	url, format := imagePartURLAndFormat(pm)
	if url == "" {
		return nil, fmt.Errorf("image url not in an expected format: no url")
	}
	switch {
	case strings.HasPrefix(url, "data:"):
		media, data, ok := parseDataURL(url)
		if !ok {
			return nil, fmt.Errorf("image url not in an expected format: %q", truncateForError(url))
		}
		if format != "" {
			media = format
		}
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": media,
				"data":       data,
			},
		}, nil
	case strings.HasPrefix(url, "https://"), strings.HasPrefix(url, "http://"):
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type": "url",
				"url":  url,
			},
		}, nil
	default:
		return nil, fmt.Errorf("image url not in an expected format: %q", truncateForError(url))
	}
}

// imagePartURLAndFormat reads a chat image part's url and declared format,
// in either of the part's two spellings; the bare-string spelling carries no
// format of its own.
func imagePartURLAndFormat(pm map[string]any) (url, format string) {
	if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
		return jsonx.GetString(iu, "url"), jsonx.GetString(iu, "format")
	}
	if s, ok := pm["image_url"].(string); ok {
		return s, ""
	}
	return "", ""
}

// parseDataURL splits a base64 data URL into its media type and payload. The
// ;base64, marker is the only spelling the target reads — a data URL without
// it is refused by the caller rather than guessed at — and a JSON-escaped
// slash in the media type (\/) reads as the slash it names.
func parseDataURL(url string) (media, data string, ok bool) {
	rest := strings.TrimPrefix(url, "data:")
	i := strings.Index(rest, ";base64,")
	if i < 0 {
		return "", "", false
	}
	media = strings.ReplaceAll(rest[:i], `\/`, "/")
	return media, rest[i+len(";base64,"):], true
}

func openaiAssistantToAnthropic(m map[string]any, nameMap map[string]string, legacyID string) (map[string]any, error) {
	var blocks []any
	// The thinking_blocks field rides at the head of the turn only when the
	// content list carries no thinking of its own inline: both kept, the
	// turn would hold the same chain of thought twice and break the target's
	// position check on signatures.
	contentList, hasContentList := jsonx.AsSlice(m["content"])
	inlineHasThinking := false
	if hasContentList {
		for _, p := range contentList {
			if pm, ok := jsonx.AsMap(p); ok {
				switch jsonx.GetString(pm, "type") {
				case "thinking", "redacted_thinking":
					inlineHasThinking = true
				}
			}
		}
	}
	if !inlineHasThinking {
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
	}
	switch c := m["content"].(type) {
	case string:
		// Anthropic rejects an empty text block, and an empty or blank turn
		// still reads as a turn: it folds to the placeholder rather than
		// vanishing or passing whitespace the target refuses.
		if strings.TrimSpace(c) == "" {
			blocks = append(blocks, map[string]any{"type": "text", "text": emptyTextPlaceholder})
		} else {
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
				text := jsonx.GetString(pm, "text")
				if strings.TrimSpace(text) == "" {
					text = emptyTextPlaceholder
				}
				blocks = append(blocks, map[string]any{"type": "text", "text": text})
			} else if typ == "thinking" {
				// An inline thinking block rides only when it says something
				// and carries the signature the target verifies; unsigned or
				// empty ones are dropped, and a redacted block has no inline
				// spelling the target reads.
				if jsonx.GetString(pm, "thinking") != "" && jsonx.GetString(pm, "signature") != "" {
					blocks = append(blocks, pm)
				}
			}
		}
	}
	if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok {
		for _, tc := range tcs {
			tm, ok := jsonx.AsMap(tc)
			if !ok {
				continue
			}
			// A chat tool call is a function call or it is nothing the
			// Messages dialect can name: an entry of another type folds to a
			// tool_use with no legal spelling, so it is skipped rather than
			// converted into a block the upstream rejects.
			if jsonx.GetString(tm, "type") != "function" {
				continue
			}
			fn, _ := jsonx.AsMap(tm["function"])
			name := ""
			if fn != nil {
				name = jsonx.GetString(fn, "name")
				// A declared tool carries the wire name the declaration was
				// given; an undeclared one rides as written — the target is
				// the one to say it is unknown.
				if mapped, ok := nameMap[name]; ok {
					name = mapped
				}
			}
			input := any(map[string]any{})
			if fn != nil {
				parsed, err := parseToolCallArguments(jsonx.GetString(fn, "arguments"), name)
				if err != nil {
					return nil, err
				}
				input = parsed
			}
			blocks = append(blocks, map[string]any{
				"type":  "tool_use",
				"id":    sanitizeToolUseIDForAnthropic(jsonx.GetString(tm, "id")),
				"name":  name,
				"input": input,
			})
		}
	}
	// The legacy single function call rides as one tool_use block, its
	// arguments read the same repaired-or-failing way a tool_call's are. The
	// name is the declared wire name when the request declares the function,
	// and as written otherwise.
	if fc, ok := jsonx.AsMap(m["function_call"]); ok {
		name := jsonx.GetString(fc, "name")
		if mapped, ok := nameMap[name]; ok {
			name = mapped
		}
		parsed, err := parseToolCallArguments(jsonx.GetString(fc, "arguments"), name)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, map[string]any{
			"type":  "tool_use",
			"id":    legacyID,
			"name":  name,
			"input": parsed,
		})
	}
	out := map[string]any{"role": "assistant"}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
			out["content"] = jsonx.GetString(b, "text")
			return out, nil
		}
	}
	if len(blocks) == 0 {
		if _, isText := m["content"].(string); isText {
			// Anthropic rejects an empty text block, and there is no way to
			// pass one through, so a turn whose content is an explicit empty
			// string is rewritten to a placeholder rather than sent as
			// content the upstream will 400 on.
			out["content"] = emptyTextPlaceholder
			return out, nil
		}
		// No content at all and nothing else carried — a null-content
		// assistant turn names no turn to fold, and the message is dropped
		// the way the reference reads it.
		return nil, nil
	}
	out["content"] = blocks
	return out, nil
}

// parseToolCallArguments reads a chat tool call's arguments. A truncated
// argument string — the model cut the object mid-flight — is repaired by
// closing what was opened rather than emptied out: silently replacing the
// caller's arguments with {} is a data change nothing downstream can see. A
// string that is neither valid JSON nor repairable fails the request instead
// of masquerading as no arguments.
func parseToolCallArguments(args, toolName string) (any, error) {
	if strings.TrimSpace(args) == "" {
		return map[string]any{}, nil
	}
	var parsed any
	if err := json.Unmarshal([]byte(args), &parsed); err == nil {
		return parsed, nil
	}
	if repaired, ok := attemptJSONRepair(args); ok {
		return repaired, nil
	}
	return nil, fmt.Errorf("failed to parse tool call arguments for tool %q", toolName)
}

// attemptJSONRepair closes the brackets a truncated JSON string left open,
// respecting nesting order and string state. It invents no content: a string
// with unbalanced-but-closable structure repairs, anything else does not.
func attemptJSONRepair(s string) (any, bool) {
	stripped := strings.TrimRightFunc(s, unicode.IsSpace)
	if stripped == "" {
		return nil, false
	}
	var openers []byte
	inString := false
	escapeNext := false
	for _, ch := range stripped {
		if escapeNext {
			escapeNext = false
			continue
		}
		if ch == '\\' {
			if inString {
				escapeNext = true
			}
			continue
		}
		if ch == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		switch ch {
		case '{':
			openers = append(openers, '}')
		case '[':
			openers = append(openers, ']')
		case '}', ']':
			if len(openers) > 0 && openers[len(openers)-1] == byte(ch) {
				openers = openers[:len(openers)-1]
			}
		}
	}
	if len(openers) == 0 {
		return nil, false
	}
	// Drop a trailing comma so the closed structure is well-formed, then
	// close in reverse order of opening.
	candidate := strings.TrimRight(stripped, ",")
	for i := len(openers) - 1; i >= 0; i-- {
		candidate += string(openers[i])
	}
	var parsed any
	if err := json.Unmarshal([]byte(candidate), &parsed); err != nil {
		return nil, false
	}
	return parsed, true
}

// openaiToolResultToAnthropic folds one tool or legacy function answer. An
// answer that names no tool_call_id of its own carries the fallback id its
// position was assigned, so the pair still references each other.
func openaiToolResultToAnthropic(m map[string]any, fallbackID string) (map[string]any, error) {
	id := sanitizeToolUseIDForAnthropic(jsonx.GetString(m, "tool_call_id"))
	if jsonx.GetString(m, "tool_call_id") == "" && fallbackID != "" {
		id = fallbackID
	}
	block := map[string]any{
		"type":        "tool_result",
		"tool_use_id": id,
	}
	switch c := m["content"].(type) {
	case string:
		block["content"] = c
	case []any:
		// A tool result may carry media. Its parts use the chat vocabulary,
		// which is not what a Messages tool_result reads, so each part is
		// converted rather than forwarded verbatim.
		parts, err := openaiToolResultPartsToAnthropic(c)
		if err != nil {
			return nil, err
		}
		block["content"] = parts
	default:
		// A result that is not text or parts — absent, null, a scalar —
		// carries nothing: it folds to the empty string rather than a null
		// the target reads as a malformed block.
		block["content"] = ""
	}
	return block, nil
}

// openaiToolResultPartsToAnthropic converts a chat tool message's content parts
// to the block list an Anthropic tool_result carries.
func openaiToolResultPartsToAnthropic(parts []any) ([]any, error) {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "text", "":
			block := map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")}
			// A prompt-cache breakpoint on a result's text is a request
			// feature the user path already carries; dropping it here would
			// un-cache the very prefix the caller marked.
			if cc, ok := pm["cache_control"]; ok {
				block["cache_control"] = cc
			}
			out = append(out, block)
		case "image_url":
			// A data URL whose payload is a document is a document the target
			// reads — wrapping it in an image block is a shape it rejects.
			if url := imagePartURL(pm); isDocumentDataURL(url) {
				media, data, ok := parseDataURL(url)
				if !ok {
					return nil, fmt.Errorf("image url not in an expected format: %q", truncateForError(url))
				}
				out = append(out, map[string]any{
					"type":   "document",
					"source": map[string]any{"type": "base64", "media_type": media, "data": data},
				})
				continue
			}
			img, err := openaiImageToAnthropic(pm)
			if err != nil {
				return nil, err
			}
			out = append(out, img)
		case "image":
			out = append(out, pm)
		case "document":
			out = append(out, pm)
		case "tool_reference":
			// A tool reference names a declared tool rather than carrying
			// result content; it rides as the reference block it already is.
			out = append(out, map[string]any{"type": "tool_reference", "tool_name": jsonx.GetString(pm, "tool_name")})
		case "file":
			doc, err := openaiFileToAnthropicDocument(pm)
			if err != nil {
				return nil, err
			}
			out = append(out, doc)
		}
	}
	return out, nil
}

// imagePartURL reads a chat image part's URL, in either of its two spellings.
func imagePartURL(pm map[string]any) string {
	url, _ := imagePartURLAndFormat(pm)
	return url
}

// isDocumentDataURL reports a data URL whose payload is a document type the
// Messages dialect reads as a document block — application/pdf and text/plain
// are the two it accepts.
func isDocumentDataURL(url string) bool {
	if !strings.HasPrefix(url, "data:") {
		return false
	}
	rest := strings.TrimPrefix(url, "data:")
	media := rest
	if i := strings.IndexAny(rest, ";,"); i >= 0 {
		media = rest[:i]
	}
	return media == "application/pdf" || media == "text/plain"
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
			// An in-sequence system entry rides at its own position. A string
			// stays a string when it says something; a list becomes its text
			// parts — empty ones carry nothing and are skipped — and an entry
			// left with no content is dropped rather than sent empty.
			switch c := m["content"].(type) {
			case string:
				if c == "" {
					continue
				}
				out = append(out, map[string]any{"role": "system", "content": c})
			case []any:
				var parts []any
				for _, b := range c {
					bm, ok := jsonx.AsMap(b)
					if !ok || jsonx.GetString(bm, "type") != "text" || jsonx.GetString(bm, "text") == "" {
						continue
					}
					part := map[string]any{"type": "text", "text": jsonx.GetString(bm, "text")}
					if cc, ok := bm["cache_control"]; ok {
						part["cache_control"] = cc
					}
					parts = append(parts, part)
				}
				if len(parts) == 0 {
					continue
				}
				out = append(out, map[string]any{"role": "system", "content": parts})
			}
			continue
		}
		if role == "user" {
			content := m["content"]
			if s, ok := content.(string); ok {
				if s == "" {
					// An empty user turn carries nothing the chat wire can
					// read; the turn is dropped rather than forwarded as
					// content the target refuses.
					continue
				}
				out = append(out, map[string]any{"role": "user", "content": s})
				continue
			}
			list, ok := jsonx.AsSlice(content)
			if !ok {
				// Null, absent, or scalar content names no turn either: a
				// user message the chat dialect cannot carry is dropped, not
				// passed through as a null the target rejects.
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
			if msg := anthropicAssistantToOpenAI(m); msg != nil {
				out = append(out, msg)
			}
		}
	}
	return out
}

// anthropicAssistantToOpenAI folds one Anthropic assistant turn. A turn that
// carries nothing — no text, no tool call, no thinking — folds to no message
// at all rather than an assistant turn with null content, which is a shape
// the chat dialect reads as a phantom turn.
func anthropicAssistantToOpenAI(m map[string]any) map[string]any {
	msg := map[string]any{"role": "assistant"}
	content := m["content"]
	if s, ok := content.(string); ok {
		msg["content"] = s
		return msg
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return nil
	}
	var texts []string
	var toolCalls []any
	var thinking []any
	carries := false
	for _, b := range list {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "text":
			texts = append(texts, jsonx.GetString(bm, "text"))
			carries = true
		case "tool_use":
			carries = true
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
			carries = true
			thinking = append(thinking, bm)
		}
	}
	if !carries {
		return nil
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
			// Thinking blocks are separate turns of one chain of thought;
			// they join on newlines, the way the reference fold reads them.
			msg["reasoning_content"] = strings.Join(rc, "\n")
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
		// The declared media type is the caller's — empty included — and only
		// a missing one reads as the default; a source with no payload names
		// no image at all.
		media := "image/jpeg"
		if mt, present := src["media_type"]; present {
			if s, isStr := mt.(string); isStr {
				media = s
			}
		}
		data := jsonx.GetString(src, "data")
		if data == "" {
			return nil
		}
		url = "data:" + media + ";base64," + data
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
// A single text part collapses to the legacy flat string (so prompt prefixes
// stay stable); anything more folds to the content-part array the chat wire
// reads — media as image_url parts, tool references as the reference parts
// they are, bare strings as text. A result whose parts all carry nothing
// renders as the empty string rather than an Anthropic-shaped list the chat
// upstream would reject wholesale.
func toolResultContent(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var parts []any
		for _, p := range t {
			switch item := p.(type) {
			case string:
				parts = append(parts, map[string]any{"type": "text", "text": item})
			case map[string]any:
				switch jsonx.GetString(item, "type") {
				case "text":
					parts = append(parts, map[string]any{"type": "text", "text": jsonx.GetString(item, "text")})
				case "image", "document":
					// A block whose source cannot be rendered is dropped rather
					// than forwarded in a spelling the target rejects.
					if img := anthropicImageToOpenAI(item); img != nil {
						parts = append(parts, img)
					}
				case "tool_reference":
					parts = append(parts, map[string]any{"type": "tool_reference", "tool_name": jsonx.GetString(item, "tool_name")})
				}
			}
		}
		if len(parts) == 0 {
			return ""
		}
		if len(parts) == 1 {
			if pm, ok := parts[0].(map[string]any); ok && pm["type"] == "text" {
				return pm["text"]
			}
		}
		return parts
	}
	return ""
}

// prependSystem places the Messages system prompt at the head of the chat
// message list. A string rides as a string; a block list stays a parts list —
// each text block one part, its cache breakpoint with it — and blocks that
// are not text carry nothing the chat dialect reads and are dropped.
func prependSystem(msgs []any, sys any) []any {
	var content any
	switch c := sys.(type) {
	case string:
		if c == "" {
			return msgs
		}
		content = c
	case []any:
		var parts []any
		for _, b := range c {
			bm, ok := jsonx.AsMap(b)
			if !ok || jsonx.GetString(bm, "type") != "text" {
				continue
			}
			part := map[string]any{"type": "text", "text": jsonx.GetString(bm, "text")}
			if cc, ok := bm["cache_control"]; ok {
				part["cache_control"] = cc
			}
			parts = append(parts, part)
		}
		if len(parts) == 0 {
			return msgs
		}
		content = parts
	default:
		text := anthropicContentToText(sys)
		if text == "" {
			return msgs
		}
		content = text
	}
	return append([]any{map[string]any{"role": "system", "content": content}}, msgs...)
}
