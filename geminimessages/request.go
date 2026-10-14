package geminimessages

// Request-direction folds between the Messages dialect and the gemini
// generateContent request body. The semantics are the reference's composed
// two-step fold (Messages to the chat shape, chat to the target), translated
// into one direct pass: no chat body is built and no sibling package is
// consulted.

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/dialectvocab"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/types"
)

// messagesRejectedOnGemini names the source parameters the composed fold lands
// on the target's reject list. The reference's Messages bridge copies every
// parameter it does not translate straight through, so a body carrying one of
// these is refused by the target fold unless the caller asked for drops. The
// Messages `metadata` parameter is deliberately absent: the bridge consumes it
// (only its user_id crosses, as the unvalidated `user` parameter), so it never
// reaches the target as the rejected same-named chat parameter.
var messagesRejectedOnGemini = map[string]struct{}{
	"context_management": {},
	"logit_bias":         {},
	"function_call":      {},
	"prediction":         {},
	"store":              {},
}

// MessagesToGemini folds an Anthropic Messages request into the gemini
// generateContent request body.
//
// The target body carries no model member (the model rides the request URL on
// that wire) and no stream member, so neither is written. Parameters the target
// cannot spell are dropped — the omission is visible through the loss API in
// the root package.
func MessagesToGemini(body map[string]any, dropParams bool) (map[string]any, error) {
	for k := range body {
		if _, bad := messagesRejectedOnGemini[k]; !bad {
			continue
		}
		if dropParams {
			continue
		}
		return nil, cerr.UnsupportedParam{Param: k}
	}

	systemParts, entries, err := foldMessagesTurns(body)
	if err != nil {
		return nil, err
	}
	// A request that named only a system prompt still needs a turn: the
	// reference substitutes a bare "." user message, because the target
	// refuses a body with no contents at all.
	if len(systemParts) > 0 && len(entries) == 0 {
		entries = []foldEntry{{role: "user", parts: []any{map[string]any{"text": "."}}}}
	}
	contents, err := geminiContents(entries)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"contents": contents}
	if len(systemParts) > 0 {
		out["system_instruction"] = map[string]any{"parts": systemParts}
	}
	tools, err := messagesToolsToGemini(body["tools"], includeServerSide(body))
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		out["tools"] = tools
	}
	if tc, err := messagesToolChoice(body["tool_choice"]); err != nil {
		return nil, err
	} else if tc != nil {
		out["toolConfig"] = tc
	} else if includeServerSide(body) {
		out["toolConfig"] = map[string]any{"includeServerSideToolInvocations": true}
	}
	if err := applyMessagesParams(body, out); err != nil {
		return nil, err
	}
	return out, nil
}

func includeServerSide(body map[string]any) bool {
	b, _ := jsonx.Bool(body["include_server_side_tool_invocations"])
	return b
}

// foldEntry is one entry of the work list the fold builds while walking the
// source conversation. It is the fold's own bookkeeping in target terms —
// parts are gemini parts — never a chat-dialect body.
type foldEntry struct {
	role  string // "user" | "model" | "tool"
	parts []any  // user/model: the gemini parts this turn contributes
	calls map[string]string
	// tool entries: the source tool_use_id and the result content, kept in
	// Messages terms until the name is resolved against the matching call.
	toolCallID  string
	toolContent any
}

// foldMessagesTurns walks the Messages conversation once and produces the
// system-instruction parts plus the ordered work list. The order mirrors the
// reference bridge: a user turn's tool results are handed over ahead of the
// turn's own text and media, and an assistant turn's thinking, text and calls
// keep their source order. A media reference this offline codec cannot
// resolve is an error, not a silent drop: the reference would fetch it, and a
// prompt that quietly lost the caller's image answers the wrong question.
func foldMessagesTurns(body map[string]any) (systemParts []any, entries []foldEntry, err error) {
	if sys, ok := body["system"]; ok && truthyJSON(sys) {
		systemParts = append(systemParts, topLevelSystemPart(sys))
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "role") {
		case "system":
			if part, ok := midTurnSystemPart(m["content"]); ok {
				systemParts = append(systemParts, part)
			}
		case "user":
			var tools []foldEntry
			var parts []any
			switch c := m["content"].(type) {
			case string:
				if c != "" {
					parts = []any{map[string]any{"text": c}}
				}
			case []any:
				for _, raw := range c {
					bm, ok := jsonx.AsMap(raw)
					if !ok {
						// The reference fails on a bare string inside a content
						// list; this codec drops the malformed part rather than
						// inventing one.
						continue
					}
					switch jsonx.GetString(bm, "type") {
					case "text":
						if text := jsonx.GetString(bm, "text"); text != "" {
							parts = append(parts, map[string]any{"text": text})
						}
					case "image", "document":
						if url := anthropicSourceURL(bm["source"]); url != "" {
							part, merr := mediaPartForURL(url)
							if merr != nil {
								return nil, nil, merr
							}
							parts = append(parts, part)
						}
					case "tool_result":
						tools = append(tools, foldEntry{
							role:        "tool",
							toolCallID:  jsonx.GetString(bm, "tool_use_id"),
							toolContent: bm["content"],
						})
					}
				}
			}
			entries = append(entries, tools...)
			if len(parts) > 0 {
				entries = append(entries, foldEntry{role: "user", parts: parts})
			}
		case "assistant":
			if entry, ok := assistantFoldEntry(m); ok {
				entries = append(entries, entry)
			}
		default:
			// A role the Messages dialect does not define contributes nothing:
			// the reference's bridge has no branch for it either, so the turn
			// is dropped rather than refused.
		}
	}
	return systemParts, entries, nil
}

// truthyJSON reports the truthiness a decoded JSON value has in the reference:
// nil, false, zero, "", an empty list and an empty object are falsy.
func truthyJSON(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case json.Number:
		f, err := t.Float64()
		return err != nil || f != 0
	case string:
		return t != ""
	case []any:
		return len(t) > 0
	case map[string]any:
		return len(t) > 0
	default:
		return true
	}
}

// topLevelSystemPart renders the request-level system prompt as one
// system-instruction part. A string rides as its own text; a block list is
// concatenated with no separator — the reference reads a system message as one
// part whatever the caller split it into.
func topLevelSystemPart(sys any) map[string]any {
	switch c := sys.(type) {
	case string:
		return map[string]any{"text": c}
	case []any:
		var b strings.Builder
		for _, raw := range c {
			bm, ok := jsonx.AsMap(raw)
			if !ok || jsonx.GetString(bm, "type") != "text" {
				continue
			}
			b.WriteString(jsonx.GetString(bm, "text"))
		}
		return map[string]any{"text": b.String()}
	default:
		return map[string]any{"text": ""}
	}
}

// midTurnSystemPart renders an in-sequence system message. Empty text is
// dropped here (unlike the request-level prompt, which keeps it), so a
// mid-turn message that says nothing contributes no part at all.
func midTurnSystemPart(content any) (map[string]any, bool) {
	switch c := content.(type) {
	case string:
		if c == "" {
			return nil, false
		}
		return map[string]any{"text": c}, true
	case []any:
		var b strings.Builder
		for _, raw := range c {
			bm, ok := jsonx.AsMap(raw)
			if !ok || jsonx.GetString(bm, "type") != "text" {
				continue
			}
			if text := jsonx.GetString(bm, "text"); text != "" {
				b.WriteString(text)
			}
		}
		return map[string]any{"text": b.String()}, true
	default:
		return nil, false
	}
}

// anthropicSourceURL renders an image/document block's source as the URL the
// media fold reads: a base64 source becomes a data URL (an empty payload names
// no media and is dropped), a url source rides as the URL it names, and any
// other source type has no spelling on the target wire.
func anthropicSourceURL(source any) string {
	src, ok := jsonx.AsMap(source)
	if !ok {
		return ""
	}
	switch jsonx.GetString(src, "type") {
	case "base64":
		data := jsonx.GetString(src, "data")
		if data == "" {
			return ""
		}
		media := "image/jpeg"
		if v, present := src["media_type"]; present {
			media = jsonx.String(v)
		}
		return "data:" + media + ";base64," + data
	case "url":
		return jsonx.GetString(src, "url")
	default:
		return ""
	}
}

// mediaPartForURL converts a media URL into the inline or file part the target
// reads, following the reference's own ladder: a data URL inlines; an object
// store URI or an https URL whose extension names a MIME type becomes a file
// reference; the file service's URIs pass through untouched because that
// service refuses direct fetches. Anything else would have the reference fetch
// the bytes — this offline codec cannot, so the request fails rather than
// silently losing the attachment.
func mediaPartForURL(url string) (map[string]any, error) {
	switch {
	case strings.Contains(url, "gs://"):
		ext := urlExtension(url)
		mime := mimeFromURL(url)
		if ext == "" || mime == "" {
			return nil, fmt.Errorf("object-store media URI has no resolvable type: %q", truncateForError(url))
		}
		return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
	case uploadedFileURI(url):
		// The wire's own file storage: the URI rides as-is, no fetch, no type.
		return map[string]any{"file_data": map[string]any{"file_uri": url}}, nil
	case strings.Contains(url, "https://"):
		if mime := mimeFromURL(url); mime != "" {
			return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
		}
		return nil, fmt.Errorf("media url needs fetching, which this codec cannot do: %q", truncateForError(url))
	case strings.Contains(url, "http://"):
		return nil, fmt.Errorf("media url needs fetching, which this codec cannot do: %q", truncateForError(url))
	case strings.Contains(url, "base64"):
		media, data, ok := parseDataURL(url)
		if !ok {
			return nil, fmt.Errorf("image url not in an expected format: %q", truncateForError(url))
		}
		return map[string]any{"inline_data": map[string]any{"data": data, "mime_type": media}}, nil
	default:
		return nil, fmt.Errorf("image url not in an expected format: %q", truncateForError(url))
	}
}

func urlExtension(url string) string {
	path := url
	if i := strings.IndexAny(path, "?#"); i >= 0 {
		path = path[:i]
	}
	if i := strings.LastIndex(path, "."); i >= 0 && i > strings.LastIndex(path, "/") {
		return path[i:]
	}
	return ""
}

// truncateForError keeps an error message bounded when the value is long.
func truncateForError(s string) string {
	if len(s) > 40 {
		return s[:40] + "..."
	}
	return s
}

// assistantFoldEntry folds one assistant turn. The part order is the
// reference's own: each thinking block as a thought part, the turn's text,
// then the calls.
//
// The thought part carries its block's signature itself. This deviates from
// the reference's encoding, which writes the reasoning TWICE — one merged
// thought part plus a plain-text part per block carrying the signature — a
// shape this audit kept until it was measured against the live upstream:
// every replayed turn bills its reasoning text twice, and a signature on a
// plain text part has no spelling on the way back (the reverse fold drops
// it), so the duplicate part bought nothing. On the thought part the
// signature round-trips: the reverse fold reads it back into the thinking
// block it came from.
func assistantFoldEntry(m map[string]any) (foldEntry, bool) {
	var textBlocks []string
	hasStringContent := false
	stringContent := ""
	type thinkBlock struct{ text, sig string }
	var thinks []thinkBlock
	type useBlock struct {
		id      string
		name    string
		args    any
		hasArgs bool
		sig     string
	}
	var uses []useBlock

	switch c := m["content"].(type) {
	case string:
		stringContent = c
		hasStringContent = true
	case []any:
		for _, raw := range c {
			bm, ok := jsonx.AsMap(raw)
			if !ok {
				continue
			}
			switch jsonx.GetString(bm, "type") {
			case "text":
				textBlocks = append(textBlocks, jsonx.GetString(bm, "text"))
			case "thinking":
				thinks = append(thinks, thinkBlock{
					text: jsonx.GetString(bm, "thinking"),
					sig:  jsonx.GetString(bm, "signature"),
				})
			case "redacted_thinking":
				// The target's request fold has no spelling for a redacted
				// block, so the reference drops it; so does this fold.
			case "tool_use":
				args, hasArgs := toolUseArgs(bm)
				uses = append(uses, useBlock{
					id:      jsonx.GetString(bm, "id"),
					name:    truncateToolName(jsonx.GetString(bm, "name")),
					args:    args,
					hasArgs: hasArgs,
					sig:     toolUseSignature(bm),
				})
			}
		}
	}

	// The turn must carry something to exist at all: a turn with no text, no
	// thinking and no calls folds to nothing, exactly as the reference's
	// bridge leaves it out.
	joined := strings.Join(textBlocks, "")
	carries := hasStringContent || len(textBlocks) > 0 || len(thinks) > 0 || len(uses) > 0
	if !carries {
		return foldEntry{}, false
	}

	var parts []any
	for _, t := range thinks {
		part := map[string]any{"thought": true, "text": t.text}
		if t.sig != "" {
			part["thoughtSignature"] = t.sig
		}
		parts = append(parts, part)
	}
	if len(textBlocks) > 0 {
		parts = append(parts, map[string]any{"text": joined})
	} else if hasStringContent {
		parts = append(parts, map[string]any{"text": stringContent})
	}
	calls := map[string]string{}
	for _, u := range uses {
		if u.id == "" {
			continue
		}
		calls[u.id] = u.name
		part := map[string]any{"function_call": map[string]any{"name": u.name}}
		if u.hasArgs {
			part["function_call"].(map[string]any)["args"] = u.args
		}
		if u.sig == "" {
			// The target rejects a replayed functionCall part without a
			// thoughtSignature (INVALID_ARGUMENT, verified against the live
			// upstream), and it VALIDATES the value: a real signature from
			// another part of the turn does not clear it, only the call's own
			// does. A client that dropped the tool_use's
			// provider_specific_fields (the common client behavior this fold
			// was measured against) has nothing valid left to send, so the
			// part carries the upstream's own skip marker — the sentinel it
			// accepts in place of a signature — rather than a turn that
			// cannot be replayed at all.
			u.sig = skipThoughtSignature
		}
		part["thoughtSignature"] = u.sig
		// The reference skips a call whose name and arguments it has already
		// emitted in this turn: a client reading the parts would otherwise
		// count one call twice.
		if !partExistsIgnoringSignature(parts, part) {
			parts = append(parts, part)
		}
	}
	return foldEntry{role: "model", parts: parts, calls: calls}, true
}

// toolUseArgs reads a tool_use block's input as the arguments the target
// reads. The reference serializes the input and parses it back on the other
// side, so an absent input folds to {} — an empty arguments object — while an
// explicit null stays null, the same values that round trip there.
func toolUseArgs(block map[string]any) (any, bool) {
	input, present := block["input"]
	if !present {
		return map[string]any{}, true
	}
	if input == nil {
		return nil, true
	}
	return roundTripJSON(input), true
}

// toolUseSignature reads the signature a tool_use block carries in its
// provider_specific_fields, the one spelling the reference's bridge lifts onto
// the call.
func toolUseSignature(bm map[string]any) string {
	psf, ok := jsonx.AsMap(bm["provider_specific_fields"])
	if !ok {
		return ""
	}
	return jsonx.GetString(psf, "signature")
}

// partExistsIgnoringSignature reports whether an equivalent part is already in
// the list, comparing every key except thoughtSignature.
func partExistsIgnoringSignature(parts []any, part map[string]any) bool {
	for _, raw := range parts {
		existing, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		if sameIgnoringSignature(existing, part) {
			return true
		}
	}
	return false
}

func sameIgnoringSignature(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}
	for k, av := range a {
		if k == "thoughtSignature" {
			continue
		}
		bv, ok := b[k]
		if !ok || !jsonEqual(av, bv) {
			return false
		}
	}
	return true
}

// jsonEqual compares two decoded JSON values structurally, treating numbers by
// their literal so 1 and 1.0 stay the distinct documents they were written as.
func jsonEqual(a, b any) bool {
	ab, aerr := json.Marshal(a)
	bb, berr := json.Marshal(b)
	if aerr != nil || berr != nil {
		return false
	}
	return string(ab) == string(bb)
}

// geminiContents groups the work list into the target's contents: consecutive
// user turns merge into one content, consecutive assistant turns into one
// model content, and a run of tool results becomes its own user content right
// after the model turn that made the calls. A user content that ends up with
// no text at all gains a blank one — the target refuses a content with no text
// parameter, so the reference appends a single space.
func geminiContents(entries []foldEntry) ([]any, error) {
	var contents []any
	var toolResponses []any
	lastCalls := map[string]string{}
	i := 0
	for i < len(entries) {
		var userParts []any
		for i < len(entries) && entries[i].role == "user" {
			userParts = append(userParts, entries[i].parts...)
			i++
		}
		if len(userParts) > 0 {
			if !hasTextPart(userParts) {
				userParts = append(userParts, map[string]any{"text": " "})
			}
			contents = append(contents, map[string]any{"role": "user", "parts": userParts})
		}
		var modelParts []any
		for i < len(entries) && entries[i].role == "model" {
			modelParts = append(modelParts, entries[i].parts...)
			if entries[i].calls != nil {
				lastCalls = entries[i].calls
			}
			i++
		}
		if len(modelParts) > 0 {
			contents = append(contents, map[string]any{"role": "model", "parts": modelParts})
		}
		if i < len(entries) && entries[i].role == "tool" {
			part, err := functionResponsePart(entries[i], lastCalls)
			if err != nil {
				return nil, err
			}
			toolResponses = append(toolResponses, part)
			i++
		}
		if i < len(entries) && entries[i].role != "tool" && len(toolResponses) > 0 {
			contents = append(contents, map[string]any{"role": "user", "parts": toolResponses})
			toolResponses = nil
		}
	}
	if len(toolResponses) > 0 {
		contents = append(contents, map[string]any{"role": "user", "parts": toolResponses})
	}
	if len(contents) == 0 {
		contents = []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": " "}}}}
	}
	return contents, nil
}

func hasTextPart(parts []any) bool {
	for _, raw := range parts {
		if m, ok := jsonx.AsMap(raw); ok {
			if v, ok := m["text"]; ok && v != nil {
				return true
			}
		}
	}
	return false
}

// functionResponsePart folds one tool result. The function name comes from the
// call this result answers — matched by id against the most recent assistant
// turn that made calls — because a result on its own names nothing. A result
// that matches no call is a request the target would reject, so it fails here
// instead.
func functionResponsePart(entry foldEntry, lastCalls map[string]string) (map[string]any, error) {
	name := ""
	if entry.toolCallID != "" {
		name = lastCalls[entry.toolCallID]
	}
	if name == "" {
		return nil, fmt.Errorf("missing corresponding tool call for tool response message (tool_use_id %q)", entry.toolCallID)
	}
	response, inline := toolResultResponse(entry.toolContent)
	fr := map[string]any{"name": name, "response": response}
	if len(inline) > 0 {
		fr["parts"] = inline
	}
	return map[string]any{"function_response": fr}, nil
}

// toolResultResponse renders a tool result's content as the response object
// the target reads, plus any media it carries. Text joins with no separator;
// media that can be inlined rides the function response's parts. A result
// whose joined text parses as a JSON object is that object; anything else is
// wrapped as {"content": <text>} — the spelling the reference's forward fold
// produces for a plain string.
func toolResultResponse(content any) (map[string]any, []any) {
	var b strings.Builder
	var inline []any
	switch c := content.(type) {
	case string:
		s := c
		if media, data, ok := parseDataURL(s); ok && strings.HasPrefix(media, "image/") {
			inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": media}})
			s = ""
		}
		b.WriteString(s)
	case []any:
		// The reference's bridge collapses a single text part to a bare string
		// before the target fold reads it; a multi-part result keeps its list
		// shape with text joined and media inlined.
		var chatParts []any
		for _, raw := range c {
			switch p := raw.(type) {
			case string:
				chatParts = append(chatParts, map[string]any{"type": "text", "text": p})
			case map[string]any:
				switch jsonx.GetString(p, "type") {
				case "text":
					chatParts = append(chatParts, map[string]any{"type": "text", "text": jsonx.GetString(p, "text")})
				case "image", "document":
					if url := anthropicSourceURL(p["source"]); url != "" {
						if media, data, ok := parseDataURL(url); ok {
							inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": media}})
						}
						// A URL the data-URL reader cannot inline would have the
						// reference fetch it; its tool-result path catches that
						// failure and drops the media, keeping the text.
					}
				case "tool_reference":
					// A tool reference names a tool rather than carrying
					// result content; the target fold has no spelling for it.
				}
			}
		}
		switch {
		case len(chatParts) == 0:
			// nothing to say
		case len(chatParts) == 1:
			if pm, ok := jsonx.AsMap(chatParts[0]); ok && jsonx.GetString(pm, "type") == "text" {
				s := jsonx.GetString(pm, "text")
				if media, data, ok := parseDataURL(s); ok && strings.HasPrefix(media, "image/") {
					inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": media}})
					s = ""
				}
				b.WriteString(s)
				break
			}
			b.WriteString(chatPartsText(chatParts))
		default:
			b.WriteString(chatPartsText(chatParts))
		}
	default:
		// Absent, null or scalar content carries nothing; the reference folds
		// it to the empty string rather than a null the target rejects.
	}
	contentStr := b.String()
	trimmed := strings.TrimSpace(contentStr)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		dec := json.NewDecoder(strings.NewReader(contentStr))
		dec.UseNumber()
		var parsed any
		if err := dec.Decode(&parsed); err == nil {
			if pm, ok := parsed.(map[string]any); ok {
				return pm, inline
			}
		}
	}
	return map[string]any{"content": contentStr}, inline
}

func chatPartsText(parts []any) string {
	var b strings.Builder
	for _, raw := range parts {
		if pm, ok := jsonx.AsMap(raw); ok && jsonx.GetString(pm, "type") == "text" {
			b.WriteString(jsonx.GetString(pm, "text"))
		}
	}
	return b.String()
}

// messagesToolsToGemini folds the tools list. Function tools merge into one
// declarations block; the hosted web-search tool becomes the target's search
// tool — and is dropped when function declarations are also present, because
// that wire refuses the mixture (unless server-side tool invocations were
// asked for, which is the target's own escape hatch for combining them).
func messagesToolsToGemini(v any, includeServerSide bool) ([]any, error) {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil, nil
	}
	var decls []any
	var special []any
	webSearch := false
	for i, raw := range list {
		tool, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		if isWebSearchTool(tool) {
			webSearch = true
			continue
		}
		if specialTool, ok := providerNativeTool(tool); ok {
			special = append(special, specialTool)
			continue
		}
		if fn, ok := jsonx.AsMap(tool["function"]); ok {
			// Already in the chat function spelling; the name crossed the
			// bridge as-is, so it is not rewritten here.
			decls = append(decls, functionDeclaration(fn["name"], fn["description"], fn["parameters"])...)
			continue
		}
		if dialectvocab.IsAnthropicHostedTool(jsonx.GetString(tool, "type")) {
			// A hosted Messages tool keeps its own shape across the bridge;
			// the target fold then reads it as a declaration carrying its name
			// (plus a description, when it stated one).
			decls = append(decls, functionDeclaration(tool["name"], tool["description"], tool["input_schema"])...)
			continue
		}
		decls = append(decls, messagesToolDeclaration(tool, i)...)
	}
	if len(decls) == 0 && len(special) == 0 && !webSearch {
		return nil, nil
	}
	var out []any
	if len(decls) > 0 {
		out = append(out, map[string]any{"function_declarations": decls})
	}
	if webSearch && (len(decls) == 0 || includeServerSide) {
		out = append(out, map[string]any{"googleSearch": map[string]any{}})
	}
	out = append(out, special...)
	return out, nil
}

// isWebSearchTool reports whether a Messages tool names the hosted web search,
// by type prefix or reserved name.
func isWebSearchTool(tool map[string]any) bool {
	t := jsonx.GetString(tool, "type")
	return strings.HasPrefix(t, "web_search") || jsonx.GetString(tool, "name") == "web_search"
}

// geminiToolKeys are the tool keys only the target wire defines. A Messages
// tool that is exactly one of these passes through: the caller is speaking the
// target's own vocabulary, and rewriting it would drop the configuration.
var geminiToolKeys = []string{
	"googleSearch", "google_search", "googleSearchRetrieval", "google_search_retrieval",
	"enterpriseWebSearch", "enterprise_web_search", "urlContext", "url_context",
	"codeExecution", "code_execution", "googleMaps", "google_maps",
	"computerUse", "computer_use",
}

// camelToolKey maps a tool key onto the target's own camelCase spelling.
var camelToolKey = map[string]string{
	"googleSearch": "googleSearch", "google_search": "googleSearch",
	"googleSearchRetrieval": "googleSearchRetrieval", "google_search_retrieval": "googleSearchRetrieval",
	"enterpriseWebSearch": "enterpriseWebSearch", "enterprise_web_search": "enterpriseWebSearch",
	"urlContext": "urlContext", "url_context": "urlContext",
	"codeExecution": "codeExecution", "code_execution": "codeExecution",
	"googleMaps": "googleMaps", "google_maps": "googleMaps",
	"computerUse": "computerUse", "computer_use": "computerUse",
}

// providerNativeTool recognizes a single-key tool whose key is one of the
// target's own tool names and whose value is a configuration object.
func providerNativeTool(tool map[string]any) (map[string]any, bool) {
	if len(tool) != 1 {
		return nil, false
	}
	for rawKey, val := range tool {
		cfg, ok := jsonx.AsMap(val)
		if !ok {
			return nil, false
		}
		camel, known := camelToolKey[rawKey]
		if !known {
			return nil, false
		}
		return map[string]any{camel: cfg}, true
	}
	return nil, false
}

// functionDeclaration builds the declaration from a name, description and
// parameter schema, each taken only when the source stated it. The name is
// used as the bridge left it — the Messages-only truncation happens in the
// declaration builder below. The schema is copied and normalized into the
// vocabulary the target accepts.
func functionDeclaration(name, description, parameters any) []any {
	decl := map[string]any{"name": jsonx.String(name)}
	if d, ok := description.(string); ok {
		decl["description"] = d
	} else if description != nil {
		decl["description"] = description
	}
	switch p := parameters.(type) {
	case nil:
		// No schema stated: no parameters member.
	case string:
		if p == "" {
			decl["parameters"] = map[string]any{"type": "object"}
		} else {
			decl["parameters"] = p
		}
	default:
		decl["parameters"] = normalizedParameterSchema(p)
	}
	return []any{decl}
}

// normalizedParameterSchema copies a parameter schema, strips the
// additionalProperties:false the target schema rejects, and normalizes it
// into the target's vocabulary. A schema the target refuses outright rides as
// written: the caller's request is not silently replaced with a different
// schema, and the upstream is the one to refuse it.
func normalizedParameterSchema(params any) any {
	pm, ok := jsonx.AsMap(params)
	if !ok {
		return params
	}
	stripFalseAdditionalProperties(pm)
	if schema, err := buildTargetSchema(pm); err == nil {
		return schema
	}
	return params
}

// messagesToolDeclaration folds one Messages function tool. Keys the Messages
// tool vocabulary does not define ride inside the parameter schema — the
// reference merges them there — and the schema builder then keeps whichever of
// them the target schema vocabulary accepts.
func messagesToolDeclaration(tool map[string]any, index int) []any {
	name := jsonx.GetString(tool, "name")
	if strings.TrimSpace(name) == "" {
		name = fmt.Sprintf("unnamed_tool_%d", index)
	}
	var params any
	hasParams := false
	if s, ok := tool["input_schema"]; ok {
		if s != nil {
			params = deepCopyValue(s)
			hasParams = true
		}
	}
	extra := map[string]any{}
	for k, v := range tool {
		switch k {
		case "name", "type", "input_schema", "description", "cache_control", "strict":
			continue
		}
		extra[k] = v
	}
	if len(extra) > 0 {
		if !hasParams {
			params = map[string]any{}
		}
		if pm, ok := params.(map[string]any); ok {
			for k, v := range extra {
				pm[k] = v
			}
		}
		hasParams = true
	}
	decl := map[string]any{"name": truncateToolName(name)}
	if d, ok := tool["description"]; ok && d != nil {
		decl["description"] = d
	}
	if hasParams {
		decl["parameters"] = normalizedParameterSchema(params)
	}
	return []any{decl}
}

// messagesToolChoice folds the Messages tool_choice onto the target's
// functionCallingConfig. A named choice names the function the request must be
// called with; a name the request does not declare rides as written — the
// target is the one to say it is unknown.
func messagesToolChoice(v any) (map[string]any, error) {
	tc, ok := jsonx.AsMap(v)
	if !ok || len(tc) == 0 {
		return nil, nil
	}
	switch jsonx.GetString(tc, "type") {
	case "any":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY"}}, nil
	case "auto":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "AUTO"}}, nil
	case "none":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}, nil
	case "tool":
		name := truncateToolName(jsonx.GetString(tc, "name"))
		return map[string]any{"functionCallingConfig": map[string]any{
			"mode": "ANY", "allowed_function_names": []any{name},
		}}, nil
	default:
		return nil, fmt.Errorf("incompatible tool choice param submitted: %v", v)
	}
}

// applyMessagesParams folds every request parameter that is not the
// conversation or the tools into the body: sampling knobs into
// generationConfig, thinking into thinkingConfig, structured output into the
// mime type and JSON schema, and the transport-level settings into their own
// members.
func applyMessagesParams(body map[string]any, out map[string]any) error {
	gc := map[string]any{}
	copyIfPresent(body, gc, "temperature")
	copyIfPresent(body, gc, "top_p")
	copyIfPresent(body, gc, "top_k")
	copyIfPresent(body, gc, "seed")
	copyIfPresent(body, gc, "frequency_penalty")
	copyIfPresent(body, gc, "presence_penalty")
	if v, ok := body["max_tokens"]; ok && v != nil {
		gc["max_output_tokens"] = v
	}
	if n, ok := body["n"]; ok {
		gc["candidate_count"] = n
	}
	if v, ok := body["logprobs"]; ok {
		gc["responseLogprobs"] = v
	}
	if v, ok := body["top_logprobs"]; ok {
		gc["logprobs"] = v
	}
	if v, ok := body["stop_sequences"]; ok && truthyJSON(v) {
		gc["stop_sequences"] = stopSequenceList(v)
	}
	if list, ok := jsonx.AsSlice(body["modalities"]); ok {
		gc["responseModalities"] = mapModalities(list)
	}
	if audio, ok := jsonx.AsMap(body["audio"]); ok {
		speech, err := speechConfig(audio)
		if err != nil {
			return err
		}
		gc["speechConfig"] = speech
		if _, has := gc["responseModalities"]; !has {
			gc["responseModalities"] = []any{"AUDIO"}
		} else if !containsString(gc["responseModalities"], "AUDIO") {
			gc["responseModalities"] = append(asAnySlice(gc["responseModalities"]), "AUDIO")
		}
	}
	if err := applyThinking(body, gc); err != nil {
		return err
	}
	if err := applyOutputFormat(body, gc); err != nil {
		return err
	}
	if len(gc) > 0 {
		out["generationConfig"] = gc
	}
	if v, ok := body["safety_settings"]; ok {
		out["safetySettings"] = v
	}
	if v, ok := body["cached_content"]; ok {
		out["cachedContent"] = v
	}
	if v, ok := body["service_tier"]; ok {
		if s, ok := v.(string); ok {
			out["serviceTier"] = serviceTierToGemini(s)
		} else {
			out["serviceTier"] = v
		}
	}
	// Dropped with no target spelling: user (arrives unvalidated but rides
	// nothing), stream (the target body has no stream member), stream_options,
	// parallel_tool_calls, timeout, extra_headers, speed, cache_control and
	// metadata (only its user_id crossed, as user). The omissions are visible
	// through the loss API in the root package.
	return nil
}

func copyIfPresent(src, dst map[string]any, key string) {
	if v, ok := src[key]; ok {
		dst[key] = v
	}
}

func stopSequenceList(v any) []any {
	switch t := v.(type) {
	case string:
		return []any{t}
	case []any:
		return t
	default:
		return []any{v}
	}
}

func mapModalities(list []any) []any {
	out := make([]any, 0, len(list))
	for _, m := range list {
		switch m {
		case "text":
			out = append(out, "TEXT")
		case "image":
			out = append(out, "IMAGE")
		case "audio":
			out = append(out, "AUDIO")
		case "video":
			out = append(out, "VIDEO")
		default:
			out = append(out, "MODALITY_UNSPECIFIED")
		}
	}
	return out
}

// speechConfig folds the audio settings into the target's speech config. The
// target's synthesis only accepts one payload format, so a caller naming
// another is told rather than silently resampled.
func speechConfig(audio map[string]any) (map[string]any, error) {
	if f, ok := audio["format"]; ok && f != nil && f != "pcm16" {
		return nil, fmt.Errorf("unsupported audio format for the target's speech output: %v", f)
	}
	speech := map[string]any{}
	if v, ok := audio["voice"]; ok {
		speech["voiceConfig"] = map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": v}}
	}
	if v, ok := audio["language_code"]; ok {
		speech["languageCode"] = v
	}
	return speech, nil
}

func serviceTierToGemini(s string) string {
	switch strings.ToLower(s) {
	case "auto":
		return "priority"
	case "default":
		return "standard"
	default:
		return strings.ToLower(s)
	}
}

func containsString(v any, want string) bool {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return false
	}
	for _, item := range list {
		if s, ok := item.(string); ok && s == want {
			return true
		}
	}
	return false
}

func asAnySlice(v any) []any {
	list, _ := jsonx.AsSlice(v)
	return list
}

// applyThinking folds the thinking parameter into thinkingConfig through the
// effort ladder: the Messages budget is bucketed into an effort, and the
// effort names the budget the target uses. That two-step fold is the
// reference's own behavior — a budget of 5000 lands on 4096 because the
// effort bucket for it says so, and a disabled request turns the thoughts off
// entirely.
func applyThinking(body, gc map[string]any) error {
	effort, ok := "", false
	if re, present := body["reasoning_effort"]; present {
		// The bridge copies an effort the caller stated verbatim, and it lands
		// after the thinking fold — so the caller's word wins.
		effort, ok = effortString(re)
		if !ok {
			return nil
		}
	} else if thinking, present := body["thinking"]; present && truthyJSON(thinking) {
		effort, ok = thinkingEffort(thinking, body["output_config"])
		if !ok {
			return nil
		}
	} else {
		return nil
	}
	budget, include, err := thinkingBudgetForEffort(effort)
	if err != nil {
		return err
	}
	gc["thinkingConfig"] = map[string]any{"thinkingBudget": budget, "includeThoughts": include}
	return nil
}

// effortString reads an effort that arrives as a bare string or as the dict
// shape a bridged caller sends.
func effortString(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case map[string]any:
		if e, ok := t["effort"]; ok {
			if s, ok := e.(string); ok {
				return s, true
			}
		}
	}
	return "", false
}

// thinkingEffort buckets a Messages thinking parameter into an effort label.
// An adaptive request takes the effort its output config declared, defaulting
// to medium; an enabled request buckets its budget; a disabled one is "none";
// a type the bridge does not know names no effort at all.
func thinkingEffort(thinking, outputConfig any) (string, bool) {
	m, ok := jsonx.AsMap(thinking)
	if !ok {
		return "", false
	}
	switch jsonx.GetString(m, "type") {
	case "disabled":
		return "none", true
	case "adaptive":
		if oc, ok := jsonx.AsMap(outputConfig); ok {
			if e := jsonx.GetString(oc, "effort"); e != "" {
				return e, true
			}
		}
		return "medium", true
	case "enabled":
		budget := 0
		if b, ok := jsonx.Int(m["budget_tokens"]); ok {
			budget = b
		}
		return types.EffortFromBudget(budget), true
	default:
		return "", false
	}
}

// thinkingBudgetForEffort maps an effort label onto the target's thinking
// budget. The values are the reference's own for this model family; an effort
// outside the ladder is a request the target refuses, so it fails here.
func thinkingBudgetForEffort(effort string) (int, bool, error) {
	switch effort {
	case "minimal":
		return 128, true, nil
	case "low":
		return 1024, true, nil
	case "medium":
		return 2048, true, nil
	case "high":
		return 4096, true, nil
	case "none", "disable":
		return 0, false, nil
	default:
		return 0, false, fmt.Errorf("unmapped reasoning effort: %q", effort)
	}
}

// applyOutputFormat folds the structured-output configuration — the legacy
// output_format field or the newer output_config.format — into the response
// mime type and JSON schema. Only a json_schema format with a schema crosses:
// the reference's bridge has no spelling for any other shape, so an
// output_format that names none is dropped whole.
func applyOutputFormat(body, gc map[string]any) error {
	format := body["output_format"]
	if !truthyJSON(format) {
		if oc, ok := jsonx.AsMap(body["output_config"]); ok {
			format = oc["format"]
		}
	}
	fm, ok := jsonx.AsMap(format)
	if !ok || jsonx.GetString(fm, "type") != "json_schema" {
		return nil
	}
	schema, ok := jsonx.AsMap(fm["schema"])
	if !ok || len(schema) == 0 {
		return nil
	}
	schema, _ = deepCopyValue(schema).(map[string]any)
	strictifyOutputSchema(schema)
	gc["response_mime_type"] = "application/json"
	gc["response_json_schema"] = strictFreeCopy(schema)
	return nil
}

// GeminiToMessages folds a gemini generateContent request body into an
// Anthropic Messages request.
//
// The source body names no model and no stream, so neither is written; a
// Messages request needs a max_tokens ceiling, which is derived when the
// source carries one and otherwise left for the caller's fallback (see the
// root package's ConvertOptions).
func GeminiToMessages(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	if si, ok := jsonx.AsMap(body["system_instruction"]); ok {
		if parts, ok := jsonx.AsSlice(si["parts"]); ok {
			var blocks []any
			for _, raw := range parts {
				pm, ok := jsonx.AsMap(raw)
				if !ok {
					continue
				}
				if text, present := pm["text"]; present {
					blocks = append(blocks, map[string]any{"type": "text", "text": text})
				}
			}
			if len(blocks) > 0 {
				out["system"] = blocks
			}
		}
	}
	msgs := geminiContentsToMessages(body["contents"])
	if msgs != nil {
		out["messages"] = msgs
	}
	tools := geminiToolsToMessages(body["tools"])
	if len(tools) > 0 {
		out["tools"] = tools
	}
	if tc := geminiToolChoiceToMessages(body["toolConfig"]); tc != nil {
		out["tool_choice"] = tc
	}
	if err := applyGeminiParams(body, out); err != nil {
		return nil, err
	}
	return out, nil
}

// geminiContentsToMessages folds the contents list into Messages messages.
// The target wire requires user and assistant to alternate — the same law the
// folds from chat and Responses already merge runs under — and the forward
// geminiContents joins consecutive same-role turns into one content per run,
// so the reverse folds each same-role run back into ONE message, that run's
// inverse: the message's blocks follow the part order exactly, the parts of
// the run's contents joined without reordering or dedupe. A content that
// contributes no blocks (empty or unspellable parts) folds to no message at
// all — an empty content array is itself a body the target rejects, and
// skipping the content lets the role run on either side of it close.
//
// A function_call part carries its own id when the source stated one, else a
// positional id counted over the calls of this fold; a function_response
// without an id of its own pairs with the nearest preceding call of the same
// name, and one that matches nothing takes a fresh positional id rather than
// being dropped — the source still names a tool result, and losing it would
// leave the conversation without the answer the call was waiting for.
func geminiContentsToMessages(v any) []any {
	contents, _ := jsonx.AsSlice(v)
	var msgs []any
	callCount := 0
	lastNameID := map[string]string{}
	for _, raw := range contents {
		c, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		role := jsonx.GetString(c, "role")
		if role != "user" && role != "model" {
			// The Messages dialect has no spelling for another role; the
			// forward fold never writes one, and forwarding it would produce a
			// body the upstream rejects.
			continue
		}
		parts, _ := jsonx.AsSlice(c["parts"])
		var blocks []any
		for _, p := range parts {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			blocks = append(blocks, geminiPartToBlocks(pm, lastNameID, &callCount)...)
		}
		if len(blocks) == 0 {
			// Nothing foldable rode this content: no message, and no empty
			// content array the target would reject.
			continue
		}
		msgRole := "user"
		if role == "model" {
			msgRole = "assistant"
		}
		if len(msgs) > 0 {
			if last, ok := jsonx.AsMap(msgs[len(msgs)-1]); ok && jsonx.GetString(last, "role") == msgRole {
				// The role run continues: this content's blocks join the
				// trailing same-role message's content, in part order.
				last["content"] = append(last["content"].([]any), blocks...)
				continue
			}
		}
		msgs = append(msgs, map[string]any{"role": msgRole, "content": blocks})
	}
	return msgs
}

// geminiPartToBlocks folds one part into Messages blocks. A part with no
// Messages spelling contributes nothing.
func geminiPartToBlocks(pm map[string]any, lastNameID map[string]string, callCount *int) []any {
	if fc, ok := jsonx.AsMap(pm["function_call"]); ok {
		id := jsonx.GetString(fc, "id")
		if id == "" {
			id = fmt.Sprintf("call_%d", *callCount)
		}
		*callCount++
		lastNameID[jsonx.GetString(fc, "name")] = id
		input := any(map[string]any{})
		if args, ok := fc["args"]; ok && args != nil {
			input = args
		}
		return []any{map[string]any{
			"type": "tool_use", "id": id, "name": jsonx.GetString(fc, "name"), "input": input,
		}}
	}
	if fr, ok := jsonx.AsMap(pm["function_response"]); ok {
		id := jsonx.GetString(fr, "id")
		if id == "" {
			if matched, ok := lastNameID[jsonx.GetString(fr, "name")]; ok {
				id = matched
			} else {
				id = fmt.Sprintf("call_%d", *callCount)
				*callCount++
			}
		}
		return []any{map[string]any{
			"type": "tool_result", "tool_use_id": id, "content": functionResponseContent(fr["response"]),
		}}
	}
	if inline, ok := jsonx.AsMap(pm["inline_data"]); ok {
		mime := jsonx.GetString(inline, "mime_type")
		blockType := "document"
		if strings.HasPrefix(mime, "image/") {
			blockType = "image"
		}
		return []any{map[string]any{
			"type":   blockType,
			"source": map[string]any{"type": "base64", "media_type": mime, "data": jsonx.GetString(inline, "data")},
		}}
	}
	if fd, ok := jsonx.AsMap(pm["file_data"]); ok {
		return []any{map[string]any{
			"type":   "document",
			"source": map[string]any{"type": "url", "url": jsonx.GetString(fd, "file_uri")},
		}}
	}
	if text, present := pm["text"]; present {
		if thought, _ := jsonx.Bool(pm["thought"]); thought {
			sig := jsonx.GetString(pm, "thoughtSignature")
			thinking := jsonx.GetString(pm, "text")
			if thinking == "" && sig == "" {
				// A thought part with neither text nor signature holds nothing
				// a client could replay, and the Messages dialect rejects an
				// empty thinking block outright.
				return nil
			}
			// The signature key rides only a real signature: an absent one
			// folds back to a block with no signature field, the shape the
			// block arrived in, rather than an explicit empty string the next
			// forward fold would have to special-case.
			block := map[string]any{"type": "thinking", "thinking": thinking}
			if sig != "" {
				block["signature"] = sig
			}
			return []any{block}
		}
		// A signature on a plain text part has no Messages spelling — the
		// forward fold reads it, but the Messages wire cannot carry it back.
		return []any{map[string]any{"type": "text", "text": text}}
	}
	// video_metadata, media_resolution and the server-side invocation channel
	// have no Messages spelling; they are dropped.
	return nil
}

// functionResponseContent renders a function response for the Messages wire:
// a response that is exactly {"content": <string>} is that string (the shape
// the forward fold writes for a plain result), and any other response is its
// compact JSON.
func functionResponseContent(response any) any {
	if m, ok := jsonx.AsMap(response); ok {
		if len(m) == 1 {
			if v, ok := m["content"]; ok {
				if _, isStr := v.(string); isStr {
					return v
				}
			}
		}
		b, err := json.Marshal(response)
		if err != nil {
			return ""
		}
		return string(b)
	}
	if response == nil {
		return ""
	}
	b, err := json.Marshal(response)
	if err != nil {
		return ""
	}
	return string(b)
}

// geminiToolsToMessages folds the target's tools list into Messages tools.
// Function declarations merge into one list; the search tool becomes the
// hosted web-search tool the Messages dialect reads; every other tool key
// exists only on the source wire and is dropped.
func geminiToolsToMessages(v any) []any {
	list, _ := jsonx.AsSlice(v)
	var out []any
	for _, raw := range list {
		tool, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		if decls, ok := jsonx.AsSlice(tool["function_declarations"]); ok {
			for _, d := range decls {
				dm, ok := jsonx.AsMap(d)
				if !ok {
					continue
				}
				t := map[string]any{"name": jsonx.GetString(dm, "name")}
				if desc, ok := dm["description"]; ok && desc != nil {
					t["description"] = desc
				}
				if params, ok := dm["parameters"]; ok {
					t["input_schema"] = params
				} else {
					// The Messages dialect requires a schema on every tool; a
					// declaration that stated none gets the permissive object.
					t["input_schema"] = map[string]any{"type": "object"}
				}
				out = append(out, t)
			}
			continue
		}
		if _, ok := tool["googleSearch"]; ok {
			out = append(out, map[string]any{"type": "web_search_20250305", "name": "web_search"})
			continue
		}
		// googleSearchRetrieval, enterpriseWebSearch, urlContext,
		// codeExecution, googleMaps and computerUse have no Messages
		// spelling; they are dropped and reported through the loss API.
	}
	return out
}

// geminiToolChoiceToMessages folds the functionCallingConfig into a Messages
// tool_choice. An allowed-function list naming exactly one function pins the
// choice to it; otherwise the mode maps directly.
func geminiToolChoiceToMessages(v any) map[string]any {
	tc, ok := jsonx.AsMap(v)
	if !ok {
		return nil
	}
	fcc, ok := jsonx.AsMap(tc["functionCallingConfig"])
	if !ok {
		return nil
	}
	if allowed, ok := jsonx.AsSlice(fcc["allowed_function_names"]); ok && len(allowed) == 1 {
		if name, ok := allowed[0].(string); ok && name != "" {
			return map[string]any{"type": "tool", "name": name}
		}
	}
	switch jsonx.GetString(fcc, "mode") {
	case "NONE":
		return map[string]any{"type": "none"}
	case "ANY":
		return map[string]any{"type": "any"}
	case "AUTO":
		return map[string]any{"type": "auto"}
	default:
		return nil
	}
}

// applyGeminiParams folds the generation config and transport settings into
// the Messages request.
func applyGeminiParams(body, out map[string]any) error {
	gc, _ := jsonx.AsMap(body["generationConfig"])
	copyIfPresent(gc, out, "temperature")
	copyIfPresent(gc, out, "top_p")
	// The Messages wire reads the sampling knobs the target's config names:
	// top_k and the seed and penalty controls ride as their own parameters,
	// the same names the composed fold's first hop passes through untouched.
	copyIfPresent(gc, out, "top_k")
	copyIfPresent(gc, out, "seed")
	copyIfPresent(gc, out, "frequency_penalty")
	copyIfPresent(gc, out, "presence_penalty")
	if v, ok := gc["stop_sequences"]; ok {
		out["stop_sequences"] = v
	}
	if v, ok := gc["max_output_tokens"]; ok {
		out["max_tokens"] = v
	} else if tc, ok := jsonx.AsMap(gc["thinkingConfig"]); ok {
		if budget, ok := jsonx.Int(tc["thinkingBudget"]); ok && budget > 0 {
			out["max_tokens"] = budget + dialectvocab.DefaultMaxTokens
		}
	}
	if tc, ok := jsonx.AsMap(gc["thinkingConfig"]); ok {
		if include, ok := jsonx.Bool(tc["includeThoughts"]); ok && !include {
			out["thinking"] = map[string]any{"type": "disabled"}
		} else {
			thinking := map[string]any{"type": "enabled"}
			// The budget is bucketed back through the effort ladder the
			// forward fold uses, so the value written is the one the ladder
			// reproduces — not the raw source number.
			if budget, ok := jsonx.Int(tc["thinkingBudget"]); ok {
				effort := types.EffortFromBudget(budget)
				if canonical, _, err := thinkingBudgetForEffort(effort); err == nil {
					thinking["budget_tokens"] = canonical
				}
			}
			out["thinking"] = thinking
		}
	}
	if jsonx.GetString(gc, "response_mime_type") == "application/json" {
		schema, ok := jsonx.AsMap(gc["response_json_schema"])
		if !ok {
			schema, ok = jsonx.AsMap(gc["response_schema"])
		}
		if ok && len(schema) > 0 {
			out["output_format"] = map[string]any{
				"type": "json_schema", "schema": schema, "strict": false,
			}
		}
	}
	if v, ok := body["serviceTier"]; ok {
		if s, ok := v.(string); ok {
			out["service_tier"] = serviceTierFromGemini(s)
		} else {
			out["service_tier"] = v
		}
	}
	// Dropped with no Messages spelling: candidate_count, responseLogprobs,
	// logprobs, responseModalities, the speech/image/response format configs,
	// mediaResolution, safetySettings, cachedContent and
	// includeServerSideToolInvocations. Each is visible through the loss API
	// in the root package.
	return nil
}

func serviceTierFromGemini(s string) string {
	switch s {
	case "priority":
		return "auto"
	case "standard":
		return "default"
	default:
		return s
	}
}

// skipThoughtSignature is the marker the upstream's own validator accepts in
// place of a thought signature a client could not preserve — the spelling the
// official hub client sends for a replayed call whose signature it never saw.
const skipThoughtSignature = "skip_thought_signature_validator"
