package messagesresponses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/dialectvocab"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

func ResponsesRequestToMessages(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	// The system parameter carries text blocks, instructions first then the
	// blocks lifted from system-role items — the reference's fold never spells
	// it as a bare string.
	var systemBlocks []any
	if inst := jsonx.GetString(body, "instructions"); inst != "" {
		systemBlocks = append(systemBlocks, map[string]any{"type": "text", "text": inst})
	}
	msgs, itemBlocks, err := inputToMessages(body["input"])
	if err != nil {
		return nil, err
	}
	// System-role items are lifted out of the message sequence into the system
	// parameter, their text blocks appended after the instructions — leaving
	// one in place would fold a system prompt into user content.
	systemBlocks = append(systemBlocks, itemBlocks...)
	if len(systemBlocks) > 0 {
		out["system"] = systemBlocks
	}
	out["messages"] = msgs
	if v, ok := body["max_output_tokens"]; ok {
		out["max_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	if tools, ok := body["tools"]; ok {
		mapped, mcpServers := responsesToolsToMessages(tools)
		out["tools"] = mapped
		// An MCP server tool names an out-of-band server, so it rides the
		// mcp_servers list rather than the tools the model calls in-band.
		if len(mcpServers) > 0 {
			out["mcp_servers"] = mcpServers
		}
	}
	if tc, ok := body["tool_choice"]; ok {
		// A choice the reference cannot map is omitted rather than spelled out
		// as auto — the target's default is the same reading.
		if choice := responsesToolChoiceToMessages(tc); choice != nil {
			out["tool_choice"] = choice
		}
	}
	if r, ok := jsonx.AsMap(body["reasoning"]); ok {
		th, err := reasoningToThinking(r)
		if err != nil {
			return nil, err
		}
		if th != nil {
			out["thinking"] = th
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
	if text, ok := jsonx.AsMap(body["text"]); ok {
		if fm, ok := jsonx.AsMap(text["format"]); ok && jsonx.GetString(fm, "type") == "json_schema" {
			// The schema is inlined and filtered before it rides output_format:
			// the target compiles it into a constrained-decoding grammar and
			// rejects the keywords the filter folds into descriptions.
			of := map[string]any{
				"type":   "json_schema",
				"schema": anthropicOutputSchema(fm["schema"]),
			}
			// A missing strict is stated as false rather than left null, which
			// is not a value the field takes.
			if st, ok := jsonx.Bool(fm["strict"]); ok {
				of["strict"] = st
			} else {
				of["strict"] = false
			}
			out["output_format"] = of
		}
	}
	// An email or a phone number is a user value the target's metadata
	// rejects, so such a `user` drops the metadata entirely rather than
	// failing the request — the guard the chat fold also carries.
	if uid := jsonx.GetString(body, "user"); uid != "" && validUserID(uid) {
		out["metadata"] = map[string]any{"user_id": uid}
	}
	return out, nil
}

func inputToMessages(input any) ([]any, []any, error) {
	if s, ok := input.(string); ok {
		return []any{map[string]any{"role": "user", "content": s}}, nil, nil
	}
	list, ok := jsonx.AsSlice(input)
	if !ok {
		return []any{}, nil, nil
	}
	var msgs []any
	var systemBlocks []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "type") {
		case "function_call_output", "custom_tool_call_output":
			// Both output items ride the same tool-result fold: the reference's
			// request chain reads `output` for either kind of call.
			output, err := functionOutputToAnthropic(m["output"])
			if err != nil {
				return nil, nil, err
			}
			msgs = appendUserTurn(msgs, []any{map[string]any{
				"type":        "tool_result",
				"tool_use_id": sanitizeToolUseID(jsonx.GetString(m, "call_id")),
				"content":     output,
			}})
		case "custom_tool_call":
			// A custom call's payload is freeform text rather than JSON, so it
			// reaches the tool_use input wrapped as {"content": ...} — the
			// shape the reference's request chain normalizes it to.
			var input any = map[string]any{}
			if raw := jsonx.GetString(m, "input"); raw != "" {
				input = map[string]any{"content": raw}
			}
			msgs = appendAssistantTurn(msgs, []any{map[string]any{
				"type":  "tool_use",
				"id":    sanitizeToolUseID(first(jsonx.GetString(m, "call_id"), jsonx.GetString(m, "id"))),
				"name":  jsonx.GetString(m, "name"),
				"input": input,
			}})
		case "function_call":
			var parsed any = map[string]any{}
			if args := jsonx.GetString(m, "arguments"); args != "" {
				_ = json.Unmarshal([]byte(args), &parsed)
			}
			msgs = appendAssistantTurn(msgs, []any{map[string]any{
				"type":  "tool_use",
				"id":    sanitizeToolUseID(jsonx.GetString(m, "call_id")),
				"name":  jsonx.GetString(m, "name"),
				"input": parsed,
			}})
		case "reasoning":
			var blocks []any
			if replayed := replayThinkingBlocks(jsonx.GetString(m, "encrypted_content")); len(replayed) > 0 {
				// Blocks the reference's bridge signed and serialized replay
				// verbatim, which is what lets the target verify them.
				blocks = replayed
			} else if sum, ok := jsonx.AsSlice(m["summary"]); ok {
				sig := jsonx.GetString(m, "encrypted_content")
				for _, p := range sum {
					text := jsonx.String(p)
					if pm, ok := jsonx.AsMap(p); ok {
						text = jsonx.GetString(pm, "text")
					}
					if text != "" {
						block := map[string]any{"type": "thinking", "thinking": text}
						if sig != "" {
							block["signature"] = sig
						}
						blocks = append(blocks, block)
					}
				}
			}
			if len(blocks) == 0 {
				continue
			}
			msgs = appendAssistantTurn(msgs, blocks)
		case "message", "":
			role := jsonx.GetString(m, "role")
			if role == "" {
				role = "user"
			}
			if role == "system" || role == "developer" {
				// Lifted to the system parameter by the caller; the reference
				// removes these from the sequence rather than folding them
				// into user content. (A developer item its own template
				// rejects outright; folding it with system is the closest
				// carrying shape.)
				systemBlocks = append(systemBlocks, systemTextBlocks(m["content"])...)
				continue
			}
			if role == "assistant" {
				msgs = appendAssistantTurn(msgs, outputPartsToAnthropic(m["content"]))
				continue
			}
			content, err := inputPartsToAnthropic(m["content"])
			if err != nil {
				return nil, nil, err
			}
			msgs = appendUserTurn(msgs, content)
		}
	}
	return msgs, systemBlocks, nil
}

// appendAssistantTurn adds content to the trailing assistant turn, opening one
// when it is not already open — two consecutive assistant messages are a
// request the target rejects, so a reasoning item and the calls it precedes
// share one turn.
func appendAssistantTurn(msgs []any, content any) []any {
	if last, ok := trailingAssistant(msgs); ok {
		last["content"] = mergeAssistantBlocks(last["content"], content)
		return msgs
	}
	return append(msgs, map[string]any{"role": "assistant", "content": content})
}

// mergeAssistantBlocks joins an assistant turn's blocks in the order the
// reference's template fixes: the thinking first, the turn's text, then the
// calls — a message that follows a call still reads ahead of it.
func mergeAssistantBlocks(content, add any) []any {
	joined := appendBlocks(content, add)
	var thinking, text, calls []any
	for _, b := range joined {
		if bm, ok := jsonx.AsMap(b); ok {
			switch jsonx.GetString(bm, "type") {
			case "thinking", "redacted_thinking":
				thinking = append(thinking, b)
				continue
			case "tool_use":
				calls = append(calls, b)
				continue
			}
		}
		text = append(text, b)
	}
	out := append(thinking, text...)
	return append(out, calls...)
}

// appendUserTurn is its user-side twin: adjacent user content — a tool result
// and the turn that follows it, or two consecutive user messages — joins one
// turn, the way the reference's template merges neighbouring messages.
func appendUserTurn(msgs []any, content any) []any {
	if len(msgs) > 0 {
		if last, ok := jsonx.AsMap(msgs[len(msgs)-1]); ok && jsonx.GetString(last, "role") == "user" {
			last["content"] = appendBlocks(last["content"], content)
			return msgs
		}
	}
	return append(msgs, map[string]any{"role": "user", "content": content})
}

// systemTextBlocks renders a system item's content as the text blocks the
// system parameter carries; empty parts are skipped, as the target rejects
// them.
func systemTextBlocks(content any) []any {
	if s, ok := content.(string); ok {
		if s == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": s}}
	}
	list, _ := jsonx.AsSlice(content)
	var blocks []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		// The part may arrive under either spelling: the reference normalizes
		// input_text to text on its way to the system parameter.
		t := jsonx.GetString(pm, "type")
		if t != "text" && t != "input_text" {
			continue
		}
		if text := jsonx.GetString(pm, "text"); text != "" {
			blocks = append(blocks, map[string]any{"type": "text", "text": text})
		}
	}
	return blocks
}

// validUserID reports whether a `user` value may ride Messages metadata as
// user_id: the target rejects an email address or a phone number there. The
// test mirrors the guard the chat fold carries for the same field.
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

// sanitizeToolUseID rewrites a call id into the shape the target's tool_use_id
// pattern accepts — alphanumeric, underscore and hyphen only — replacing
// anything else with an underscore. Both the call and its result are
// sanitized, so the pair still matches afterwards.
func sanitizeToolUseID(id string) string {
	if id == "" {
		return "tool_use_id"
	}
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

// replayThinkingBlocks decodes the JSON array of thinking blocks the
// reference's chat bridge writes into encrypted_content. Only blocks the
// target can verify replay — a thinking block needs its signature, a redacted
// one its data; an encrypted_content that is not such an array is a blob this
// fold did not write and is skipped rather than forwarded as garbage.
func replayThinkingBlocks(enc string) []any {
	if strings.TrimSpace(enc) == "" {
		return nil
	}
	var decoded []any
	if err := json.Unmarshal([]byte(enc), &decoded); err != nil {
		return nil
	}
	var blocks []any
	for _, b := range decoded {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "thinking":
			if sig := jsonx.GetString(bm, "signature"); sig != "" {
				blocks = append(blocks, map[string]any{
					"type":      "thinking",
					"thinking":  jsonx.GetString(bm, "thinking"),
					"signature": sig,
				})
			}
		case "redacted_thinking":
			if data := jsonx.GetString(bm, "data"); data != "" {
				blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": data})
			}
		}
	}
	return blocks
}

// trailingAssistant returns the trailing message when it is an assistant turn —
// the turn a following reasoning item or function_call belongs to.
func trailingAssistant(msgs []any) (map[string]any, bool) {
	if len(msgs) == 0 {
		return nil, false
	}
	last, ok := jsonx.AsMap(msgs[len(msgs)-1])
	if !ok || jsonx.GetString(last, "role") != "assistant" {
		return nil, false
	}
	return last, true
}

// appendBlocks adds content to an assistant turn, promoting a bare string to a
// text block so blocks have somewhere to go.
func appendBlocks(content, add any) []any {
	var out []any
	switch c := content.(type) {
	case string:
		if c != "" {
			out = append(out, map[string]any{"type": "text", "text": c})
		}
	case []any:
		out = append(out, c...)
	}
	switch a := add.(type) {
	case string:
		if a != "" {
			out = append(out, map[string]any{"type": "text", "text": a})
		}
	case []any:
		out = append(out, a...)
	}
	return out
}

// functionOutputToAnthropic renders a Responses function_call_output for an
// Anthropic tool_result. A plain string stays a string; a part list becomes a
// block list, and anything else is re-encoded as JSON so the bytes stay
// parseable rather than rendering as Go syntax.
func functionOutputToAnthropic(v any) (any, error) {
	if s, ok := v.(string); ok {
		return s, nil
	}
	list, ok := jsonx.AsSlice(v)
	if !ok {
		if v == nil {
			return "", nil
		}
		if b, err := jsonx.Marshal(v); err == nil {
			return string(b), nil
		}
		return "", nil
	}
	var blocks []any
	var texts []string
	hasImage := false
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			// A part that is not an object carries nothing the reference
			// folds; it skips the part outright.
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "output_text", "text":
			if t := jsonx.GetString(pm, "text"); t != "" {
				texts = append(texts, t)
				blocks = append(blocks, map[string]any{"type": "text", "text": t})
			}
		case "input_image", "image_url":
			url := jsonx.GetString(pm, "image_url")
			if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			if url == "" {
				url = jsonx.GetString(pm, "url")
			}
			if url != "" {
				block, err := inputImageToAnthropicBlock(url)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, block)
				hasImage = true
			}
		}
	}
	// Images ride their own blocks with the text parts alongside, in the
	// order the output listed them.
	if hasImage {
		return blocks, nil
	}
	// Text-only output stays the flat string a tool result usually carries.
	// (The reference joins the texts with nothing; this fold joins with a
	// newline — a divergence the root parity suite pins on purpose.)
	if len(texts) > 0 {
		return strings.Join(texts, "\n"), nil
	}
	// Nothing foldable survived: the whole output is serialized so the turn
	// is not left empty.
	if b, err := jsonx.Marshal(v); err == nil {
		return string(b), nil
	}
	return "", nil
}

// inputImageToAnthropicBlock builds an Anthropic image block from a URL or data
// URL.
func inputImageToAnthropicBlock(url string) (map[string]any, error) {
	if !strings.HasPrefix(url, "data:") {
		return map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": url}}, nil
	}
	media, data, ok := splitDataURL(url)
	if !ok {
		// The reference rejects a data URL it cannot parse rather than emit a
		// base64 block with nothing in it.
		return nil, fmt.Errorf("image url %q not in expected format", url)
	}
	return map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": media, "data": data}}, nil
}

func inputPartsToAnthropic(content any) (any, error) {
	if s, ok := content.(string); ok {
		return s, nil
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content, nil
	}
	var blocks []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "text":
			blocks = append(blocks, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
		case "input_image":
			block, err := inputImageToAnthropicBlock(jsonx.GetString(pm, "image_url"))
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		}
	}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok && jsonx.GetString(b, "type") == "text" {
			return jsonx.GetString(b, "text"), nil
		}
	}
	return blocks, nil
}

func outputPartsToAnthropic(content any) any {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content
	}
	var blocks []any
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		if jsonx.GetString(pm, "type") == "output_text" || jsonx.GetString(pm, "type") == "text" {
			blocks = append(blocks, map[string]any{"type": "text", "text": jsonx.GetString(pm, "text")})
		}
	}
	if len(blocks) == 1 {
		if b, ok := jsonx.AsMap(blocks[0]); ok {
			return jsonx.GetString(b, "text")
		}
	}
	return blocks
}

// splitDataURL parses a base64 data URL the way the reference does: the media
// type is whatever sits between "data:" and the one ";base64," separator —
// empty when nothing does, no default — and a URL without exactly one
// separator is not a data URL this fold can read.
func splitDataURL(url string) (media, data string, ok bool) {
	rest := url[len("data:"):]
	i := strings.Index(rest, ";base64,")
	if i < 0 || strings.Contains(rest[i+len(";base64,"):], ";base64,") {
		return "", "", false
	}
	return rest[:i], rest[i+len(";base64,"):], true
}

func responsesToolsToMessages(v any) ([]any, []any) {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil, nil
	}
	var out []any
	var mcpServers []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		typ := jsonx.GetString(m, "type")
		if typ == "mcp" {
			// An MCP server tool names an out-of-band server, not a function
			// the model calls in-band: it rides the mcp_servers list with the
			// entry shape the reference's fold emits.
			server := map[string]any{
				"type": "url",
				"url":  jsonx.GetString(m, "server_url"),
				"name": jsonx.GetString(m, "server_label"),
			}
			if allowed, ok := jsonx.AsSlice(m["allowed_tools"]); ok {
				server["tool_configuration"] = map[string]any{"allowed_tools": allowed}
			}
			if headers, ok := jsonx.AsMap(m["headers"]); ok {
				if bearer := jsonx.GetString(headers, "Authorization"); bearer != "" {
					server["authorization_token"] = strings.ReplaceAll(bearer, "Bearer ", "")
				}
			}
			mcpServers = append(mcpServers, server)
			continue
		}
		if typ == "web_search_preview" || typ == "web_search" {
			tool := map[string]any{"type": "web_search_20250305", "name": "web_search"}
			if ul, ok := jsonx.AsMap(m["user_location"]); ok {
				if appr, ok := jsonx.AsMap(ul["approximate"]); ok {
					location := map[string]any{"type": "approximate"}
					for _, k := range []string{"city", "country", "region", "timezone"} {
						if v, ok := appr[k]; ok {
							location[k] = v
						}
					}
					tool["user_location"] = location
				}
			}
			if scs := jsonx.GetString(m, "search_context_size"); scs != "" {
				if n, ok := webSearchMaxUses[scs]; ok {
					tool["max_uses"] = n
				}
			}
			out = append(out, tool)
			continue
		}
		// A Responses-only tool has no Messages equivalent, and forwarding it
		// yields a tool with no name, which the upstream rejects. It is dropped
		// rather than passed through.
		if dialectvocab.IsResponsesOnlyTool(typ) {
			continue
		}
		// An Anthropic-native tool keeps its own shape; rewriting it into a
		// custom tool would drop the type that names it.
		if dialectvocab.IsAnthropicHostedTool(typ) {
			out = append(out, m)
			continue
		}
		if typ == "custom" {
			// A custom tool's payload is a freeform string, which Messages can
			// only carry as a function tool over a content property — the
			// schema the reference's bridge synthesizes for it.
			name := jsonx.GetString(m, "name")
			tool := map[string]any{
				"type": "custom",
				"name": name,
				// The reference's bridge states the description even when the
				// tool named none; a cache-control marker rides the tool.
				"description": customToolDescription(m),
				"input_schema": sanitizeToolInputSchema(map[string]any{
					"type": "object",
					"properties": map[string]any{
						"content": map[string]any{
							"type":        "string",
							"description": "The " + name + " content following the specified format",
						},
					},
					"required": []any{"content"},
				}),
			}
			if cc, ok := jsonx.AsMap(m["cache_control"]); ok {
				tool["cache_control"] = cc
			}
			out = append(out, tool)
			continue
		}
		tool := map[string]any{
			"type": "custom",
			"name": jsonx.GetString(m, "name"),
			// The reference's bridge defaults a missing description to the
			// empty string and states it, so the field is always present.
			"description": jsonx.GetString(m, "description"),
			// Messages requires input_schema on every tool, and the schema
			// itself has to be the shape the target accepts — the reference
			// sanitizes every folded schema the same way.
		}
		schema := map[string]any{}
		if p, ok := jsonx.AsMap(m["parameters"]); ok {
			schema = p
		}
		tool["input_schema"] = sanitizeToolInputSchema(schema)
		if cc, ok := jsonx.AsMap(m["cache_control"]); ok {
			// A cache breakpoint the caller set on the tool survives the fold,
			// as it does through the reference's bridge.
			tool["cache_control"] = cc
		}
		out = append(out, tool)
	}
	return out, mcpServers
}

// webSearchMaxUses is the search-tuning translation the reference fixes: how
// many searches a low, medium or high context budget buys.
var webSearchMaxUses = map[string]int{
	"low":    1,
	"medium": 5,
	"high":   10,
}

// sanitizeToolInputSchema coerces a folded tool's schema into the shape the
// Messages target accepts: the type stated as object, a properties object
// always present, legacy definitions inlined, and no keys outside the set the
// target recognises ($defs included, $schema and friends dropped).
func sanitizeToolInputSchema(schema map[string]any) map[string]any {
	normalized := jsonx.CloneMap(schema)
	if jsonx.GetString(normalized, "type") != "object" {
		normalized["type"] = "object"
	}
	if _, ok := normalized["properties"]; !ok {
		normalized["properties"] = map[string]any{}
	}
	unpackLegacyDefs(normalized)
	out := make(map[string]any, len(normalized))
	for _, key := range []string{"$defs", "additionalProperties", "properties", "required", "strict", "type"} {
		if v, ok := normalized[key]; ok {
			out[key] = v
		}
	}
	return out
}

// unpackLegacyDefs pops the legacy definitions namespaces — draft-04
// `definitions` and OpenAPI `components.schemas` — and inlines every $ref that
// pointed at them. `$defs` itself stays: the target recognises it.
func unpackLegacyDefs(schema map[string]any) {
	defs := map[string]any{}
	if comps, ok := jsonx.AsMap(schema["components"]); ok {
		if schemas, ok := jsonx.AsMap(comps["schemas"]); ok {
			for k, v := range schemas {
				defs[k] = v
			}
		}
	}
	delete(schema, "components")
	if d, ok := jsonx.AsMap(schema["definitions"]); ok {
		for k, v := range d {
			defs[k] = v
		}
	}
	delete(schema, "definitions")
	if len(defs) == 0 {
		return
	}
	unpackRefs(schema, nil, nil, defs, map[string]bool{})
}

// customToolDescription renders a custom tool's description with its grammar
// appended, the way the reference's bridge embeds the format definition so the
// model can honor it.
func customToolDescription(m map[string]any) string {
	desc := jsonx.GetString(m, "description")
	if fm, ok := jsonx.AsMap(m["format"]); ok {
		definition := jsonx.GetString(fm, "definition")
		if definition != "" {
			desc += "\n\nFormat:\n```" + jsonx.GetString(fm, "syntax") + "\n" + definition + "\n```"
		}
	}
	return desc
}

func responsesToolChoiceToMessages(v any) any {
	switch t := v.(type) {
	case string:
		switch t {
		case "auto":
			return map[string]any{"type": "auto"}
		case "required":
			return map[string]any{"type": "any"}
		case "none":
			return map[string]any{"type": "none"}
		}
		return nil
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		if typ == "function" || typ == "custom" {
			// A custom choice nests its name under `custom`; a named choice
			// means "call this tool", and dropping the name would tell the
			// model to pick any tool instead.
			name := jsonx.GetString(t, "name")
			if name == "" {
				if nested, ok := jsonx.AsMap(t[typ]); ok {
					name = jsonx.GetString(nested, "name")
				}
			}
			if name != "" {
				return map[string]any{"type": "tool", "name": name}
			}
			// A named-choice shape that names nothing means "any tool".
			return map[string]any{"type": "any"}
		}
		switch typ {
		case "auto":
			return map[string]any{"type": "auto"}
		case "none":
			return map[string]any{"type": "none"}
		case "required", "tool", "any":
			// "tool" with no name forces tool use without naming one — the
			// reading Messages spells "any".
			return map[string]any{"type": "any"}
		}
	}
	// Anything else the reference cannot map: omitted, not spelled out as
	// auto.
	return nil
}

func reasoningToThinking(r map[string]any) (map[string]any, error) {
	effort := jsonx.GetString(r, "effort")
	if effort == "" {
		return nil, nil
	}
	switch effort {
	case "minimal", "low", "medium", "high", "xhigh", "max":
	case "none":
		return nil, nil
	default:
		// The reference refuses an effort it cannot map rather than silently
		// dropping or rewriting the caller's reasoning configuration.
		return nil, cerr.UnsupportedParam{Param: "reasoning.effort"}
	}
	budget, ok := types.BudgetFromEffort(effort)
	if !ok {
		return nil, nil
	}
	// Only type and budget_tokens: the Messages thinking object has no
	// summary member, and carrying the Responses-only r.summary here draws
	// "thinking.enabled.summary: Extra inputs are not permitted" from the
	// upstream. Summary is a Responses-API concept with no Messages
	// equivalent, so the faithful fold drops it.
	return map[string]any{"type": "enabled", "budget_tokens": budget}, nil
}

// anthropicOutputSchema readies a Responses text.format schema for the
// Messages output_format: $refs are inlined against the definitions they
// point at — the target accepts no external schema references — and the
// keywords its grammar cannot compile are folded into descriptions. The
// keyword set and note wording mirror the reference's filter, which this
// repository's chat fold also carries; the reference applies it only for
// models with native structured output, and this codec carries no model
// knowledge, so it applies it always.
func anthropicOutputSchema(schema any) any {
	m, ok := jsonx.AsMap(schema)
	if !ok {
		return schema
	}
	clone := jsonx.CloneMap(m)
	unpackSchemaDefs(clone)
	return filterAnthropicOutputSchema(clone)
}

// unpackSchemaDefs pops $defs/definitions off the schema and inlines every
// $ref that points at them. A circular or unknown reference is left in place,
// as the reference leaves it.
func unpackSchemaDefs(schema map[string]any) {
	defs := map[string]any{}
	for _, key := range []string{"$defs", "definitions"} {
		if d, ok := jsonx.AsMap(schema[key]); ok {
			for k, v := range d {
				defs[k] = v
			}
		}
		delete(schema, key)
	}
	if len(defs) == 0 {
		return
	}
	unpackRefs(schema, nil, nil, defs, map[string]bool{})
}

// unpackRefs walks the schema replacing $ref nodes with deep copies of their
// targets; the chain guards against circular references.
func unpackRefs(node any, parent any, key any, defs map[string]any, chain map[string]bool) {
	switch n := node.(type) {
	case map[string]any:
		ref := jsonx.GetString(n, "$ref")
		if ref == "" {
			for k, v := range n {
				unpackRefs(v, n, k, defs, chain)
			}
			return
		}
		name := lastPathSegment(ref)
		if chain[name] {
			return
		}
		target, ok := jsonx.AsMap(defs[name])
		if !ok {
			return
		}
		resolved := deepCopyValue(target)
		walk := resolved
		if parent == nil && key == nil {
			// The schema itself was a ref: it becomes the resolved target in
			// place, so the caller's map keeps the unpacked schema.
			if rm, ok := resolved.(map[string]any); ok {
				for k := range n {
					delete(n, k)
				}
				for k, v := range rm {
					n[k] = v
				}
			}
			walk = n
		} else {
			replaceRefSlot(parent, key, resolved)
		}
		next := make(map[string]bool, len(chain)+1)
		for k := range chain {
			next[k] = true
		}
		next[name] = true
		unpackRefs(walk, parent, key, defs, next)
	case []any:
		for i, v := range n {
			unpackRefs(v, n, i, defs, chain)
		}
	}
}

// replaceRefSlot writes a resolved ref back through the container that held
// it.
func replaceRefSlot(parent any, key any, resolved any) {
	switch p := parent.(type) {
	case map[string]any:
		if k, ok := key.(string); ok {
			p[k] = resolved
		}
	case []any:
		if i, ok := key.(int); ok && i < len(p) {
			p[i] = resolved
		}
	}
}

func lastPathSegment(ref string) string {
	for i := len(ref) - 1; i >= 0; i-- {
		if ref[i] == '/' {
			return ref[i+1:]
		}
	}
	return ref
}

func deepCopyValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopyValue(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = deepCopyValue(val)
		}
		return out
	default:
		return v
	}
}

// outputSchemaConstraints maps each JSON Schema keyword the Messages
// structured output rejects onto the note that replaces it: the target
// compiles the schema into a constrained-decoding grammar and refuses a
// schema carrying these, so the constraint is dropped and described in the
// schema's own description instead of silently losing what the caller asked
// for.
var outputSchemaConstraints = []struct{ key, note string }{
	{"minItems", "minimum number of items: %v"},
	{"maxItems", "maximum number of items: %v"},
	{"uniqueItems", "all array items must be unique"},
	{"contains", "array must contain an item matching: %v"},
	{"minContains", "minimum number of matching items: %v"},
	{"maxContains", "maximum number of matching items: %v"},
	{"prefixItems", "leading items must match, in order: %v"},
	{"minimum", "minimum value: %v"},
	{"maximum", "maximum value: %v"},
	{"exclusiveMinimum", "exclusive minimum value: %v"},
	{"exclusiveMaximum", "exclusive maximum value: %v"},
	{"multipleOf", "must be a multiple of %v"},
	{"minLength", "minimum length: %v"},
	{"maxLength", "maximum length: %v"},
	{"minProperties", "minimum number of properties: %v"},
	{"maxProperties", "maximum number of properties: %v"},
	{"patternProperties", "properties whose names match each pattern must satisfy: %v"},
	{"propertyNames", "property names must satisfy: %v"},
	{"dependentRequired", "dependent required properties: %v"},
	{"dependentSchemas", "dependent schemas: %v"},
	{"unevaluatedProperties", "unevaluated properties must satisfy: %v"},
	{"if", "conditional (if): %v"},
	{"then", "conditional (then): %v"},
	{"else", "conditional (else): %v"},
	{"not", "must not match: %v"},
}

// filterAnthropicOutputSchema strips those keywords, folding each into the
// schema's description, and rewrites `oneOf` to `anyOf` — the one combinator
// the target accepts.
func filterAnthropicOutputSchema(schema any) any {
	m, ok := jsonx.AsMap(schema)
	if !ok {
		return schema
	}
	out := map[string]any{}
	var notes []string
	for _, c := range outputSchemaConstraints {
		value, present := m[c.key]
		if !present {
			continue
		}
		if b, isBool := value.(bool); isBool && !b {
			continue
		}
		if len(c.note) > 0 && strings.Contains(c.note, "%v") {
			notes = append(notes, fmt.Sprintf(c.note, formatConstraintValue(value)))
		} else {
			notes = append(notes, c.note)
		}
	}
	if len(notes) > 0 {
		note := "Note: " + strings.Join(notes, ", ") + "."
		if desc := jsonx.GetString(m, "description"); desc != "" {
			out["description"] = desc + " " + note
		} else {
			out["description"] = note
		}
	}
	dropsType := enumConflictsWithType(m)
	for key, value := range m {
		if isOutputSchemaConstraint(key) {
			continue
		}
		if key == "type" && dropsType {
			continue
		}
		if key == "description" {
			if _, handled := out["description"]; handled {
				continue
			}
		}
		switch key {
		case "properties", "$defs":
			if sub, ok := jsonx.AsMap(value); ok {
				filtered := map[string]any{}
				for k, v := range sub {
					filtered[k] = filterAnthropicOutputSchema(v)
				}
				out[key] = filtered
				continue
			}
		case "items":
			if _, ok := jsonx.AsMap(value); ok {
				out[key] = filterAnthropicOutputSchema(value)
				continue
			}
		case "anyOf", "oneOf":
			if list, ok := jsonx.AsSlice(value); ok {
				var filtered []any
				for _, item := range list {
					filtered = append(filtered, filterAnthropicOutputSchema(item))
				}
				out["anyOf"] = append(existingAnyOfList(out["anyOf"]), filtered...)
				continue
			}
		case "allOf":
			if list, ok := jsonx.AsSlice(value); ok {
				var filtered []any
				for _, item := range list {
					filtered = append(filtered, filterAnthropicOutputSchema(item))
				}
				out[key] = filtered
				continue
			}
		}
		out[key] = value
	}
	// The target requires additionalProperties=false on every object schema.
	if jsonx.GetString(out, "type") == "object" {
		out["additionalProperties"] = false
	}
	return out
}

func existingAnyOfList(v any) []any {
	if list, ok := jsonx.AsSlice(v); ok {
		return list
	}
	return nil
}

func isOutputSchemaConstraint(key string) bool {
	for _, c := range outputSchemaConstraints {
		if c.key == key {
			return true
		}
	}
	return false
}

// formatConstraintValue renders a dropped constraint's value for the advisory
// note, serializing a sub-schema so the note says what was actually required.
func formatConstraintValue(value any) string {
	switch value.(type) {
	case map[string]any, []any:
		if b, err := jsonx.Marshal(value); err == nil {
			return string(b)
		}
	}
	return jsonx.String(value)
}

// enumConflictsWithType reports whether a schema's enum cannot match its
// declared type, in which case the declared type is the wrong half and is
// dropped.
func enumConflictsWithType(schema map[string]any) bool {
	values, ok := jsonx.AsSlice(schema["enum"])
	if !ok {
		return false
	}
	declared := jsonx.GetString(schema, "type")
	if declared == "" {
		return false
	}
	for _, v := range values {
		if !matchesJSONType(declared, v) {
			return true
		}
	}
	return false
}

func matchesJSONType(declared string, v any) bool {
	switch declared {
	case "null":
		return v == nil
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "integer":
		switch n := v.(type) {
		case json.Number:
			f, err := n.Float64()
			return err == nil && f == float64(int64(f))
		case float64:
			return n == float64(int64(n))
		case int, int64:
			return true
		}
		return false
	case "number":
		switch v.(type) {
		case json.Number, float64, int, int64:
			return true
		}
		return false
	case "string":
		_, ok := v.(string)
		return ok
	case "array":
		_, ok := jsonx.AsSlice(v)
		return ok
	case "object":
		_, ok := jsonx.AsMap(v)
		return ok
	}
	return false
}
