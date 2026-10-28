package geminiresponses

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/internal/schemanorm"
	"github.com/veildawn/ai-protocol/internal/thinkingcfg"
)

// thoughtSeparator is the marker the reference embeds in a tool call id to
// carry a thought signature across dialects that have no field for it.
const thoughtSeparator = "__thought__"

// skipThoughtSignature is the marker the upstream's own validator accepts in
// place of a thought signature a client could not preserve — the spelling the
// official hub client sends for a replayed call whose signature it never saw.
// A missing signature and a borrowed one are both refused (INVALID_ARGUMENT,
// verified against the live upstream), so the fallback is the sentinel or
// nothing.
const skipThoughtSignature = "skip_thought_signature_validator"

// partThoughtSignature reads a part's thought signature in either spelling.
// Google's own wire spells the member BOTH ways — the plugin's reader notes
// "thoughtSignature comes and goes" — and the packer keeps both keys in its
// map, so a reader that consults one spelling alone silently drops the
// signature half the time. A dropped signature is not a cosmetic loss: it
// rides the call_id the client replays, and an unsigned replayed call is
// refused by the upstream on the NEXT request. Mirrors the plugin's own
// part_thought_signature, which checks the same two keys.
func partThoughtSignature(pm map[string]any) string {
	if s := jsonx.GetString(pm, "thoughtSignature"); s != "" {
		return s
	}
	return jsonx.GetString(pm, "thought_signature")
}

// uploadedFileURI reports whether uri names a file the target wire already
// holds: an https URI whose path addresses the wire's uploaded-files
// namespace. The reference hands such a URI over instead of fetching it — the
// file answers direct reads with a refusal, so passing it through is the only
// way to use it. The check is on the path's shape rather than on any host:
// this codec carries no host knowledge, and every URI shaped like an upload
// reads the same way here — the same rule the chat-side fold of this wire
// applies.
func uploadedFileURI(uri string) bool {
	rest, ok := strings.CutPrefix(uri, "https://")
	if !ok {
		return false
	}
	if i := strings.IndexAny(rest, "?#"); i >= 0 {
		rest = rest[:i]
	}
	slash := strings.Index(rest, "/")
	if slash < 0 {
		return false
	}
	return strings.HasPrefix(rest[slash:], "/v1beta/files/")
}

// ResponsesToGemini folds a Responses request body onto the target wire.
//
// The reference reaches this pair in two hops — Responses onto the chat
// dialect, then chat onto this wire — and this fold reproduces the composed
// result directly: instructions and system-role items become the system
// instruction block, input items become contents with consecutive same-role
// turns merged, function calls and their outputs pair up as function_call and
// function_response parts, and the sampling/structured-output/reasoning
// parameters land in the generation config.
func ResponsesToGemini(body map[string]any, dropParams bool) (map[string]any, error) {
	// context_management is the one source parameter the reference forwards
	// and then refuses: the target wire has no spelling for it, so it errors
	// unless the caller asked for silent dropping.
	if v, ok := body["context_management"]; ok && v != nil && !dropParams {
		return nil, cerr.UnsupportedParam{Param: "context_management"}
	}
	f := &requestFold{lastCalls: map[string]string{}, groupCalls: map[string]string{}}
	f.addInstructions(body["instructions"])
	input := body["input"]
	if s, ok := input.(string); ok {
		f.beginUser()
		f.parts = append(f.parts, map[string]any{"text": s})
	} else if list, ok := jsonx.AsSlice(input); ok {
		for _, raw := range list {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if err := f.item(m); err != nil {
				return nil, err
			}
		}
	}
	f.flushPendingReasoning()
	f.flushGroup()
	if len(f.contents) == 0 {
		// Contents are required on this wire. A system instruction with nothing
		// else gets the reference's single-dot user turn; anything else empty
		// gets the blank-text turn the same fallback produces.
		fallback := " "
		if len(f.sysParts) > 0 {
			fallback = "."
		}
		f.contents = []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": fallback}}}}
	}
	out := map[string]any{"contents": f.contents}
	if len(f.sysParts) > 0 {
		out["system_instruction"] = map[string]any{"parts": f.sysParts}
	}
	gc, err := responsesGenerationConfig(body)
	if err != nil {
		return nil, err
	}
	if len(gc) > 0 {
		out["generationConfig"] = gc
	}
	tools, toolConfig, err := responsesTools(body, dropParams)
	if err != nil {
		return nil, err
	}
	if tools != nil {
		out["tools"] = tools
	}
	if toolConfig != nil {
		out["toolConfig"] = toolConfig
	}
	return out, nil
}

// GeminiToResponses folds a target-wire request body onto the Responses
// dialect. This is the reverse of ResponsesToGemini, derived rule by rule
// from the forward mapping: contents become input items (thought parts fold
// into reasoning items, function calls and responses into their own items),
// the system instruction becomes instructions, tool declarations become
// function tools, and the generation config maps back onto the Responses
// sampling, structured-output and reasoning parameters.
func GeminiToResponses(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if inst := systemInstructionText(body["system_instruction"]); inst != "" {
		out["instructions"] = inst
	}
	input := geminiContentsToInput(body["contents"])
	if len(input) == 0 {
		// The Responses upstream rejects an empty input, and a system-only
		// request would otherwise serialize `"input": null`. The instructions
		// ride a system-role input item instead, the same move the chat fold
		// onto this dialect makes.
		if inst, ok := out["instructions"].(string); ok && inst != "" {
			input = []any{map[string]any{
				"type":    "message",
				"role":    "system",
				"content": []any{map[string]any{"type": "input_text", "text": inst}},
			}}
			delete(out, "instructions")
		}
	}
	out["input"] = input
	gc, _ := jsonx.AsMap(body["generationConfig"])
	if v, ok := gc["max_output_tokens"]; ok && v != nil {
		out["max_output_tokens"] = v
	}
	if v, ok := gc["temperature"]; ok && v != nil {
		out["temperature"] = v
	}
	if v, ok := gc["top_p"]; ok && v != nil {
		out["top_p"] = v
	}
	if effort, ok := thinkingConfigToEffort(gc["thinkingConfig"]); ok {
		out["reasoning"] = map[string]any{"effort": effort}
	}
	if text := responseMimeToText(gc); text != nil {
		out["text"] = text
	}
	if tools := geminiToolsToResponses(body["tools"]); len(tools) > 0 {
		out["tools"] = tools
	}
	if choice, ok := geminiToolConfigToChoice(body["toolConfig"]); ok {
		out["tool_choice"] = choice
	}
	switch st := jsonx.GetString(body, "serviceTier"); st {
	case "priority":
		out["service_tier"] = "auto"
	case "standard":
		out["service_tier"] = "default"
	case "":
	default:
		out["service_tier"] = st
	}
	return out, nil
}

// requestFold accumulates one open turn at a time, mirroring the way the
// reference's message fold groups consecutive same-role messages into a
// single content: a run of user items shares one user content, a run of
// model fragments shares one model content, and a run of tool results
// becomes its own user content.
type requestFold struct {
	sysParts  []any
	contents  []any
	group     string // "" | "user" | "model" | "tool"
	parts     []any
	lastCalls map[string]string
	// groupCalls collects the calls of the model turn being built; they become
	// the lookup table for tool results once the turn closes.
	groupCalls    map[string]string
	pendingReason []*pendingReasoning
}

type pendingReasoning struct {
	text   string
	blocks []any
}

func (f *requestFold) addInstructions(v any) {
	switch inst := v.(type) {
	case string:
		if inst != "" {
			f.sysParts = append(f.sysParts, map[string]any{"text": inst})
		}
	case []any:
		if len(inst) == 0 {
			return
		}
		var b strings.Builder
		for _, blk := range inst {
			if bm, ok := blk.(map[string]any); ok {
				b.WriteString(jsonx.GetString(bm, "text"))
			}
		}
		f.sysParts = append(f.sysParts, map[string]any{"text": b.String()})
	}
}

func (f *requestFold) item(m map[string]any) error {
	switch jsonx.GetString(m, "type") {
	case "function_call_output", "custom_tool_call_output", "web_search_call",
		"computer_call_output", "tool_result":
		return f.toolOutputItem(m)
	case "function_call", "custom_tool_call":
		return f.functionCallItem(m)
	case "reasoning":
		return f.reasoningItem(m)
	default:
		return f.messageItem(m)
	}
}

// openGroup starts a turn of the given kind, flushing whatever turn was open
// first. A user turn following a tool-result turn does NOT merge with it: the
// reference emits the collected tool results as their own content before the
// next message is read, and the two stay separate contents on the wire.
func (f *requestFold) openGroup(kind string) {
	if f.group == kind {
		return
	}
	f.flushGroup()
	f.group = kind
	f.parts = nil
}

func (f *requestFold) flushGroup() {
	switch f.group {
	case "user":
		if len(f.parts) > 0 {
			// A user content with no text part at all is rejected upstream, so
			// the reference appends a blank one — an image-only turn still
			// carries a text part.
			if !hasTextPart(f.parts) {
				f.parts = append(f.parts, map[string]any{"text": " "})
			}
			f.contents = append(f.contents, map[string]any{"role": "user", "parts": f.parts})
		}
	case "model":
		if len(f.parts) > 0 {
			f.contents = append(f.contents, map[string]any{"role": "model", "parts": f.parts})
			if len(f.groupCalls) > 0 {
				f.lastCalls = f.groupCalls
				f.groupCalls = map[string]string{}
			}
		}
	case "tool":
		if len(f.parts) > 0 {
			f.contents = append(f.contents, map[string]any{"role": "user", "parts": f.parts})
		}
	}
	f.group = ""
	f.parts = nil
}

// hasTextPart reports whether any part carries a text key with a value, the
// check the reference runs over a merged user content before appending the
// blank fallback.
func hasTextPart(parts []any) bool {
	for _, p := range parts {
		if m, ok := p.(map[string]any); ok {
			if v, ok := m["text"]; ok && v != nil {
				return true
			}
		}
	}
	return false
}

func (f *requestFold) beginUser() {
	f.flushPendingReasoning()
	f.openGroup("user")
}

// beginModel opens a model turn, attaching any reasoning items waiting to
// merge forward. The reference folds a standalone reasoning message into the
// assistant message that follows it — joining the reasoning texts and putting
// the waiting blocks first — so replayed reasoning rides the turn it belongs
// to instead of a turn of its own.
func (f *requestFold) beginModel() {
	f.openGroup("model")
	if len(f.pendingReason) == 0 {
		return
	}
	var texts []string
	var blocks []any
	for _, p := range f.pendingReason {
		if p.text != "" {
			texts = append(texts, p.text)
		}
		blocks = append(blocks, p.blocks...)
	}
	f.pendingReason = nil
	if len(texts) > 0 {
		f.parts = append(f.parts, map[string]any{"text": strings.Join(texts, "\n"), "thought": true})
	}
	for _, b := range blocks {
		if part := thinkingBlockPart(b); part != nil {
			f.parts = append(f.parts, part)
		}
	}
}

// flushPendingReasoning turns waiting reasoning items into their own model
// turns. They still merge with a following model turn at the content level
// (consecutive model turns share one content), which is exactly what the
// reference's grouping does with the standalone messages it leaves behind.
func (f *requestFold) flushPendingReasoning() {
	if len(f.pendingReason) == 0 {
		return
	}
	f.openGroup("model")
	for _, p := range f.pendingReason {
		if p.text != "" {
			f.parts = append(f.parts, map[string]any{"text": p.text, "thought": true})
		}
		for _, b := range p.blocks {
			if part := thinkingBlockPart(b); part != nil {
				f.parts = append(f.parts, part)
			}
		}
	}
	f.pendingReason = nil
}

func (f *requestFold) messageItem(m map[string]any) error {
	content, present := m["content"]
	if !present || content == nil {
		// The reference drops an input item whose content is absent entirely.
		return nil
	}
	role := jsonx.GetString(m, "role")
	if role == "" {
		role = "user"
	}
	switch role {
	case "system", "developer":
		f.addInstructions(content)
		return nil
	case "user":
		f.beginUser()
		return f.userContent(content)
	case "assistant":
		f.beginModel()
		return f.assistantContent(content)
	default:
		// Any other role rides the reference's fold unchanged and is then
		// refused there — a tool-role item carries no call to answer, and the
		// rest have no place in this wire's two-role contents.
		return fmt.Errorf("message role %q has no place on the target wire", role)
	}
}

func (f *requestFold) userContent(content any) error {
	switch c := content.(type) {
	case string:
		f.parts = append(f.parts, map[string]any{"text": c})
	case []any:
		for _, p := range c {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch jsonx.GetString(pm, "type") {
			case "input_file":
				part, err := inputFilePart(pm)
				if err != nil {
					return err
				}
				f.parts = append(f.parts, part)
			case "input_image":
				url := partImageURL(pm)
				part, err := mediaPart(url)
				if err != nil {
					return err
				}
				f.parts = append(f.parts, part)
			case "encrypted_content":
				if v, ok := pm["encrypted_content"]; ok && v != nil {
					if s := jsonx.String(v); s != "" {
						f.parts = append(f.parts, map[string]any{"text": s})
					}
				}
			default:
				// Every other part lands on the chat wire as a text block; on
				// this side only a non-empty text block produces a part.
				if s, ok := pm["text"].(string); ok && s != "" {
					f.parts = append(f.parts, map[string]any{"text": s})
				}
			}
		}
	}
	return nil
}

func (f *requestFold) assistantContent(content any) error {
	switch c := content.(type) {
	case string:
		f.parts = append(f.parts, map[string]any{"text": c})
	case []any:
		// The reference's assistant branch reads only text blocks from a list
		// content — media an assistant turn carried is dropped there, and this
		// fold does not invent a place for it.
		for _, p := range c {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if s, ok := pm["text"].(string); ok {
				f.parts = append(f.parts, map[string]any{"text": s})
			}
		}
	}
	return nil
}

func (f *requestFold) reasoningItem(m map[string]any) error {
	text := reasoningItemText(m)
	blocks := decodeThinkingBlocks(m["encrypted_content"])
	if text == "" && len(blocks) == 0 {
		return nil
	}
	f.pendingReason = append(f.pendingReason, &pendingReasoning{text: text, blocks: blocks})
	return nil
}

func (f *requestFold) functionCallItem(m map[string]any) error {
	f.beginModel()
	callID := jsonx.GetString(m, "call_id")
	if callID == "" {
		callID = jsonx.GetString(m, "id")
	}
	name := jsonx.GetString(m, "name")
	if ns := jsonx.GetString(m, "namespace"); ns != "" && jsonx.GetString(m, "type") != "custom_tool_call" {
		name = ns + "__" + name
	}
	args := jsonx.GetString(m, "arguments")
	if args == "" && jsonx.GetString(m, "type") == "custom_tool_call" {
		// A custom call carries its freeform payload in `input`; the chat wire
		// reads it as arguments, wrapped the way the reference wraps it.
		if in := jsonx.GetString(m, "input"); in != "" {
			if b, err := json.Marshal(map[string]any{"content": in}); err == nil {
				args = string(b)
			}
		}
	}
	sig := ""
	if i := strings.Index(callID, thoughtSeparator); i >= 0 {
		sig = callID[i+len(thoughtSeparator):]
	}
	if sig == "" {
		// A client that preserved the signature on the item itself — the
		// reverse fold writes it to provider_specific_fields as well as the
		// call id — must have it read back, or the turn loses the signature it
		// was handed and the upstream sees an unsigned replay.
		if psf, ok := jsonx.AsMap(firstNonNil(m["provider_specific_fields"], m["providerSpecificFields"])); ok {
			sig = jsonx.GetString(psf, "thought_signature")
			if sig == "" {
				sig = jsonx.GetString(psf, "thoughtSignature")
			}
		}
	}
	// An object's keys ride the string's own order into the target's args —
	// a decoded map would alphabetize them — re-rendered with the
	// reference's separators (jsonx.ArgsValue). Anything else keeps the
	// value-level rule below.
	fc := map[string]any{"name": name}
	if ordered, ok := jsonx.ArgsValue(args); ok {
		fc["args"] = ordered
	} else {
		parsed, err := parseCallArgs(args)
		if err != nil {
			// The reference refuses a call whose arguments are not one JSON value.
			return fmt.Errorf("tool call %q: %w", name, err)
		}
		fc["args"] = parsed
	}
	part := map[string]any{"function_call": fc}
	if sig == "" {
		// The target rejects a replayed functionCall part that carries no
		// thoughtSignature (INVALID_ARGUMENT, "Function call is missing a
		// thought_signature in functionCall parts"), and it VALIDATES the
		// value: one borrowed from another part of the turn does not clear
		// it either. A client that round-trips no signature — the common
		// behavior for a Responses client replaying a call it received as a
		// bare call_id — has nothing valid left, so the part rides the
		// upstream's own skip marker rather than a turn that cannot be
		// replayed at all. A preserved signature always wins.
		sig = skipThoughtSignature
	}
	part["thoughtSignature"] = sig
	if !functionCallExists(f.parts, part) {
		f.parts = append(f.parts, part)
	}
	// The id-to-name table registers every call, deduplicated ones included: a
	// later output may answer the twin the wire never saw, and the name lookup
	// it needs is the call the caller made, not the part this fold kept.
	f.groupCalls[callID] = name
	return nil
}

func (f *requestFold) toolOutputItem(m map[string]any) error {
	callID := jsonx.GetString(m, "call_id")
	if callID == "" {
		// Without a call id the reference contributes nothing for the item.
		return nil
	}
	// Opening the tool group closes the model turn the call belongs to, which
	// is what publishes the id-to-name table the lookup below reads.
	f.flushPendingReasoning()
	f.openGroup("tool")
	name, ok := f.lastCalls[callID]
	if !ok || name == "" {
		return fmt.Errorf("tool output %q answers no tool call in the conversation", callID)
	}
	content := normalizeToolOutput(m["output"])
	contentStr := ""
	var inline []any
	switch c := content.(type) {
	case string:
		if mime, data, ok := splitDataURL(c); ok && strings.HasPrefix(mime, "image/") {
			// A tool result that serialised as a bare image data URL is
			// promoted back to inline data, the way the reference reads it.
			inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": mime}})
		} else {
			contentStr = c
		}
	case []any:
		for _, blk := range c {
			bm, ok := blk.(map[string]any)
			if !ok {
				continue
			}
			switch jsonx.GetString(bm, "type") {
			case "text":
				contentStr += jsonx.GetString(bm, "text")
			case "image", "input_image", "image_url":
				// The reference tries to read each image as a data URL and
				// merely warns past the ones it cannot; unreadable references
				// are skipped here too.
				if mime, data, ok := splitDataURL(partImageURL(bm)); ok {
					inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": mime}})
				}
			case "file", "input_file":
				if mime, data, ok := splitDataURL(fileBlockData(bm)); ok {
					inline = append(inline, map[string]any{"inline_data": map[string]any{"data": data, "mime_type": mime}})
				}
			}
		}
	}
	response := map[string]any{"content": contentStr}
	if trimmed := strings.TrimSpace(contentStr); strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		var parsed any
		if err := json.Unmarshal([]byte(contentStr), &parsed); err == nil {
			// A `$ref` key anywhere in the parsed object keeps the string
			// wrapped: the gemini wire reserves `$ref` inside a function
			// response as a reference to a media part's display_name, so a
			// tool result that happens to print a JSON Schema is refused
			// outright upstream when its `$ref`s ride as structure (400,
			// INVALID_ARGUMENT, "The referenced name ... does not match to a
			// display_name in the function_response.parts"). As text the
			// schema reaches the model verbatim.
			if pm, ok := parsed.(map[string]any); ok && !hasReservedRefKey(pm) {
				response = pm
			}
		}
	}
	fr := map[string]any{"name": name, "response": response}
	if len(inline) > 0 {
		// Multimodal tool results nest their media inside the function
		// response, not beside it.
		fr["parts"] = inline
	}
	f.parts = append(f.parts, map[string]any{"function_response": fr})
	return nil
}

// hasReservedRefKey reports whether the media-reference key `$ref` occurs as
// an object key anywhere in the parsed tool-result object, at any depth. The
// gemini wire reserves it inside function_response.response — it names a part
// by display_name — so a result that happens to print a JSON Schema must ride
// as text rather than structure.
func hasReservedRefKey(v any) bool {
	switch t := v.(type) {
	case map[string]any:
		for k, val := range t {
			if k == "$ref" || hasReservedRefKey(val) {
				return true
			}
		}
	case []any:
		for _, el := range t {
			if hasReservedRefKey(el) {
				return true
			}
		}
	}
	return false
}

// parseCallArgs turns a call's arguments string into the object the wire
// takes. An empty string is not {} but a stated object type — the reference's
// own placeholder — and anything that is not one JSON value is an error.
func parseCallArgs(args string) (any, error) {
	if args == "" {
		return map[string]any{"type": "object"}, nil
	}
	var parsed any
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		return nil, fmt.Errorf("arguments are not one JSON value: %w", err)
	}
	return parsed, nil
}

// functionCallExists reports whether an equal function_call part is already
// in the turn. The reference de-duplicates identical calls within one
// assistant turn, ignoring the signature each carries.
func functionCallExists(parts []any, part map[string]any) bool {
	want, _ := part["function_call"].(map[string]any)
	for _, p := range parts {
		m, ok := p.(map[string]any)
		if !ok {
			continue
		}
		got, ok := m["function_call"].(map[string]any)
		if !ok || len(got) != len(want) {
			continue
		}
		same := true
		for k, v := range got {
			if wv, ok := want[k]; !ok || !equalArgs(v, wv) {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

// equalArgs compares two args members. The ordered render carries them as
// raw bytes, so two spellings of one object compare by their decoded value —
// the reference de-duplicates on the decoded dict, where key order is not
// part of equality.
func equalArgs(a, b any) bool {
	ra, aok := a.(json.RawMessage)
	rb, bok := b.(json.RawMessage)
	if aok && bok {
		var va, vb any
		if json.Unmarshal(ra, &va) != nil || json.Unmarshal(rb, &vb) != nil {
			return false
		}
		return reflect.DeepEqual(va, vb)
	}
	return reflect.DeepEqual(a, b)
}

// thinkingBlockPart renders one replayable thinking block as a part: a
// signature rides the part, and a block whose text is itself a JSON object is
// spread into the part rather than quoted — the two shapes the reference
// emits when it replays signed reasoning.
func thinkingBlockPart(block any) map[string]any {
	m, ok := block.(map[string]any)
	if !ok || jsonx.GetString(m, "type") != "thinking" {
		// Redacted blocks have no readable text to replay.
		return nil
	}
	text, hasText := m["thinking"]
	if !hasText || text == nil {
		return nil
	}
	sig, hasSig := m["signature"]
	if !hasSig || sig == nil {
		return nil
	}
	if s, ok := text.(string); ok {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(s), &parsed); err == nil {
			out := deepCopyValue(parsed).(map[string]any)
			out["thoughtSignature"] = sig
			return out
		}
	}
	return map[string]any{"text": text, "thoughtSignature": sig}
}

// reasoningItemText reads the plaintext a reasoning item carries: its content
// first, its summary as the fallback — the order the reference reads them in.
func reasoningItemText(m map[string]any) string {
	if s, ok := m["content"].(string); ok && strings.TrimSpace(s) != "" {
		return s
	}
	if list, ok := jsonx.AsSlice(m["content"]); ok {
		var parts []string
		for _, b := range list {
			bm, ok := b.(map[string]any)
			if !ok {
				continue
			}
			switch jsonx.GetString(bm, "type") {
			case "encrypted_content", "redacted_thinking":
				continue
			}
			if t, ok := bm["text"].(string); ok && strings.TrimSpace(t) != "" {
				parts = append(parts, strings.TrimSpace(t))
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	if list, ok := jsonx.AsSlice(m["summary"]); ok {
		var parts []string
		for _, b := range list {
			if bm, ok := b.(map[string]any); ok {
				if t := strings.TrimSpace(jsonx.GetString(bm, "text")); t != "" {
					parts = append(parts, t)
				}
			} else if s, ok := b.(string); ok && strings.TrimSpace(s) != "" {
				parts = append(parts, strings.TrimSpace(s))
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// decodeThinkingBlocks reads the encrypted content the reference itself wrote
// — a JSON array of signed thinking blocks — back into the blocks a replay
// needs. Only blocks the provider can verify survive: a thinking block needs
// its signature, a redacted one its opaque data.
func decodeThinkingBlocks(v any) []any {
	s, isString := v.(string)
	if !isString {
		return nil
	}
	ec := s
	if strings.TrimSpace(ec) == "" {
		return nil
	}
	var decoded []any
	if err := json.Unmarshal([]byte(ec), &decoded); err != nil {
		return nil
	}
	var blocks []any
	for _, b := range decoded {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		switch jsonx.GetString(bm, "type") {
		case "thinking":
			if jsonx.GetString(bm, "signature") != "" {
				blocks = append(blocks, bm)
			}
		case "redacted_thinking":
			if jsonx.GetString(bm, "data") != "" {
				blocks = append(blocks, bm)
			}
		}
	}
	return blocks
}

// normalizeToolOutput renders a tool result's output as the tool-message
// content the fold reads: a plain string stays, a part list becomes text and
// image blocks (images only when present, else the joined text), and any
// other value is re-encoded so the bytes stay parseable.
func normalizeToolOutput(output any) any {
	switch t := output.(type) {
	case nil:
		return ""
	case string:
		return t
	case []any:
		var texts []string
		var blocks []any
		hasImage := false
		for _, p := range t {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch jsonx.GetString(pm, "type") {
			case "input_text", "output_text", "text":
				if s, ok := pm["text"].(string); ok && s != "" {
					texts = append(texts, s)
					blocks = append(blocks, map[string]any{"type": "text", "text": s})
				}
			case "input_image", "image_url":
				if url := partImageURL(pm); url != "" {
					hasImage = true
					blocks = append(blocks, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
				}
			}
		}
		if hasImage {
			return blocks
		}
		if len(texts) > 0 {
			return strings.Join(texts, "")
		}
		if b, err := jsonx.Marshal(t); err == nil {
			return string(b)
		}
		return ""
	default:
		if b, err := jsonx.Marshal(t); err == nil {
			return string(b)
		}
		return ""
	}
}

// partImageURL reads the URL off a content part that carries one, in either
// the string or the object spelling.
func partImageURL(pm map[string]any) string {
	if s, ok := pm["image_url"].(string); ok {
		return s
	}
	if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
		return jsonx.GetString(iu, "url")
	}
	return ""
}

// fileBlockData reads the bytes off a file block, in either spelling.
func fileBlockData(pm map[string]any) string {
	if s, ok := pm["file_data"].(string); ok {
		return s
	}
	if fm, ok := jsonx.AsMap(pm["file"]); ok {
		return jsonx.GetString(fm, "file_data")
	}
	return jsonx.GetString(pm, "file_data")
}

// inputFilePart folds an input_file part: the id (or the URL falling back to
// it) names the file, the bytes ride file_data, and the media fold decides
// which spelling the wire takes.
func inputFilePart(pm map[string]any) (map[string]any, error) {
	fid := jsonx.GetString(pm, "file_id")
	if fid == "" {
		fid = jsonx.GetString(pm, "file_url")
	}
	data := jsonx.GetString(pm, "file_data")
	if fid != "" {
		return mediaPart(fid)
	}
	if data != "" {
		return mediaPart(data)
	}
	return nil, fmt.Errorf("file part carries neither a file id nor file data")
}

// mediaPart resolves one media reference onto the wire's two spellings: bytes
// the reference could read inline become inline data, and a URI the wire can
// fetch itself becomes a file reference. A URI whose type cannot be told from
// its name is an error — the reference would have to fetch it, and this
// library has no network to do that with.
func mediaPart(url string) (map[string]any, error) {
	if url == "" {
		return nil, fmt.Errorf("media reference is empty")
	}
	if strings.Contains(url, "gs://") {
		mime := mimeFromExtension(url)
		if mime == "" {
			return nil, fmt.Errorf("unable to determine the media type of %q", url)
		}
		return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
	}
	if uploadedFileURI(url) {
		// The wire's own file storage: the URI rides as-is, no fetch, no type.
		return map[string]any{"file_data": map[string]any{"file_uri": url}}, nil
	}
	if mime := mimeFromExtension(url); mime != "" && strings.Contains(url, "https://") {
		return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
	}
	if strings.Contains(url, "http://") || strings.Contains(url, "https://") || strings.Contains(url, "base64") {
		mime, data, ok := splitDataURL(url)
		if !ok {
			return nil, fmt.Errorf("media reference %q is not a data URL and cannot be read offline", url)
		}
		return map[string]any{"inline_data": map[string]any{"data": data, "mime_type": mime}}, nil
	}
	return nil, fmt.Errorf("unable to determine the media type for %q", url)
}

// splitDataURL parses `data:<mime>;base64,<bytes>`; the mime keeps any
// parameters it carries, matching the reference's parser.
//
// The scheme must sit at the START of the string, the way the reference reads
// a data URL (factory.py:1369 `content_str[:5].lower() == "data:"`). An
// unanchored search is not a cosmetic difference: a tool result that merely
// MENTIONS a data URL — a Read of a file whose icon field holds one — has
// everything from that mention onward sliced out as the "payload". The text
// is then gone AND the slice is not base64, so the upstream refuses the whole
// turn (400 INVALID_ARGUMENT, "Base64 decoding failed for ..."), permanently:
// every replay of the conversation carries the same broken part. Observed
// live 2026-10-16 on this arm — five requests, two payload shapes.
//
// The payload must also read as base64 under an alphabet and padding the
// upstream's own JSON parser takes — see looksBase64. This part is
// deliberately stricter than the reference's tool-result path, which
// promotes a junk payload straight to inline_data only for the upstream to
// refuse the request; the same gate is what the reference runs before it
// treats a data-URL text part as audio (utils.py:7664 is_base64_encoded).
// A payload that fails it is not media: the value stays text, which is a
// turn that works.
func splitDataURL(url string) (mime string, data string, ok bool) {
	if len(url) < 5 || !strings.EqualFold(url[:5], "data:") {
		return "", "", false
	}
	rest := url[5:]
	j := strings.Index(rest, ";base64,")
	if j < 0 {
		return "", "", false
	}
	data = rest[j+len(";base64,"):]
	if !looksBase64(data) {
		return "", "", false
	}
	return strings.ReplaceAll(rest[:j], `\/`, `/`), data, true
}

// looksBase64 reports whether s is a payload the upstream's base64 reader
// takes: the standard or URL alphabet, padded or unpadded (protobuf's JSON
// mapping accepts all four). Empty is not valid — it names no bytes to send.
// The byte sweep is what keeps whitespace and prose out: Go's decoders SKIP
// \r\n rather than reject them, so a file's trailing lines after a real
// payload would otherwise still "decode" into a request the upstream refuses.
func looksBase64(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '+', c == '/', c == '=', c == '-', c == '_':
		default:
			return false
		}
	}
	for _, enc := range []*base64.Encoding{
		base64.StdEncoding, base64.RawStdEncoding,
		base64.URLEncoding, base64.RawURLEncoding,
	} {
		if _, err := enc.DecodeString(s); err == nil {
			return true
		}
	}
	return false
}

// urlMimeTable maps a file extension onto the wire's accepted media types.
var urlMimeTable = map[string]string{
	".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
	".mp4": "video/mp4", ".mov": "video/mov", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",
	".avi": "video/avi", ".wmv": "video/wmv", ".mpegps": "video/mpegps", ".flv": "video/flv",
	".mp3": "audio/mp3", ".wav": "audio/wav", ".ogg": "audio/ogg",
	".pdf": "application/pdf", ".txt": "text/plain",
}

// mimeFromExtension tells a reference's media type from its file extension,
// ignoring any query or fragment — the only offline signal available.
func mimeFromExtension(raw string) string {
	u := strings.ToLower(raw)
	if i := strings.IndexByte(u, '?'); i >= 0 {
		u = u[:i]
	}
	if i := strings.IndexByte(u, '#'); i >= 0 {
		u = u[:i]
	}
	for ext, mime := range urlMimeTable {
		if strings.HasSuffix(u, ext) {
			return mime
		}
	}
	return ""
}

// responsesGenerationConfig maps the Responses sampling, structured-output
// and reasoning parameters onto the generation config. Everything else the
// Responses dialect carries (metadata, user, store, parallel tool calls,
// truncation and friends) has no landing here and is silently dropped — the
// reference's bridge never forwards it onto the wire.
func responsesGenerationConfig(body map[string]any) (map[string]any, error) {
	gc := map[string]any{}
	if v, ok := body["max_output_tokens"]; ok && v != nil {
		gc["max_output_tokens"] = v
	}
	if v, ok := body["temperature"]; ok && v != nil {
		gc["temperature"] = v
	}
	if v, ok := body["top_p"]; ok && v != nil {
		gc["top_p"] = v
	}
	if text, ok := jsonx.AsMap(body["text"]); ok {
		if format, ok := jsonx.AsMap(text["format"]); ok {
			switch jsonx.GetString(format, "type") {
			case "json_schema":
				gc["response_mime_type"] = "application/json"
				if schema, ok := jsonx.AsMap(format["schema"]); ok && len(schema) > 0 {
					cp := deepCopyValue(schema)
					// The wire takes a standard schema but rejects `strict`
					// anywhere in it; everything else (additionalProperties
					// included) rides as written.
					removeStrictKeys(cp)
					gc["response_json_schema"] = cp
				}
			case "json_object":
				gc["response_mime_type"] = "application/json"
			}
		}
	}
	switch r := body["reasoning"].(type) {
	case string:
		if r != "" {
			if err := applyReasoningEffort(gc, r); err != nil {
				return nil, err
			}
		}
	case map[string]any:
		if len(r) == 0 {
			break
		}
		// A reasoning object that also states a summary rides onto the chat
		// wire whole, and the effort is read out of it there — the summary
		// itself has no landing and drops.
		if e, ok := r["effort"]; ok && e != nil {
			s, ok := e.(string)
			if !ok {
				return nil, fmt.Errorf("reasoning effort %v is not one of the accepted levels", e)
			}
			if err := applyReasoningEffort(gc, s); err != nil {
				return nil, err
			}
		}
	}
	return gc, nil
}

// applyReasoningEffort maps an effort level onto the thinking config: the
// four levels carry fixed budgets, and none (or its synonym) switches
// thinking off outright. Any other level is refused — the reference rejects
// it rather than guess a budget.
func applyReasoningEffort(gc map[string]any, effort string) error {
	var tc map[string]any
	switch effort {
	case "minimal":
		tc = map[string]any{"thinkingBudget": 128, "includeThoughts": true}
	case "low":
		tc = map[string]any{"thinkingBudget": 1024, "includeThoughts": true}
	case "medium":
		tc = map[string]any{"thinkingBudget": 2048, "includeThoughts": true}
	case "high":
		tc = map[string]any{"thinkingBudget": 4096, "includeThoughts": true}
	case "none", "disable":
		tc = map[string]any{"thinkingBudget": 0, "includeThoughts": false}
	default:
		return fmt.Errorf("reasoning effort %q is not one of the accepted levels", effort)
	}
	gc["thinkingConfig"] = tc
	return nil
}

// responsesTools folds the Responses tool list onto the wire. Function tools
// (custom and namespaced ones included) become one declarations block; a
// search tool becomes the wire's own search tool, dropped when declarations
// are also present because the wire rejects the mix; and the tool types the
// chat hop drops never arrive, while the ones it forwards but the wire cannot
// use (server-tool descriptors among them) leave the key present but empty —
// the shape the reference's own plumbing produces.
func responsesTools(body map[string]any, dropParams bool) ([]any, map[string]any, error) {
	list, ok := jsonx.AsSlice(body["tools"])
	if !ok {
		return nil, nil, nil
	}
	var decls []any
	hasSearch := false
	chatSurvivors := 0
	for _, raw := range list {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch jsonx.GetString(m, "type") {
		case "function":
			chatSurvivors++
			decl, err := functionToolDeclaration(m)
			if err != nil {
				return nil, nil, err
			}
			decls = append(decls, decl)
		case "custom":
			chatSurvivors++
			decls = append(decls, customToolDeclaration(m))
		case "namespace":
			chatSurvivors++
			decls = append(decls, namespaceToolDeclarations(m)...)
		case "web_search", "web_search_preview":
			hasSearch = true
		case "computer_use", "image_generation", "shell":
			// Dropped on the chat hop; nothing reaches the wire.
		default:
			// Forwarded verbatim by the chat hop and then ignored by the
			// wire's tool builder — the key survives with no entries.
			chatSurvivors++
		}
	}
	var tools []any
	if len(decls) > 0 {
		tools = []any{map[string]any{"function_declarations": decls}}
	} else if hasSearch {
		tools = []any{map[string]any{"googleSearch": map[string]any{}}}
	} else if chatSurvivors > 0 {
		tools = []any{}
	}
	var toolConfig map[string]any
	if chatSurvivors > 0 {
		if tc, ok := body["tool_choice"]; ok && tc != nil {
			config, err := responsesToolChoice(tc, dropParams)
			if err != nil {
				return nil, nil, err
			}
			toolConfig = config
		}
	}
	return tools, toolConfig, nil
}

// functionToolDeclaration builds one declaration from a Responses function
// tool: the name rides, the description is always stated (blank when absent),
// and the parameters are normalised into the schema vocabulary the wire
// accepts.
func functionToolDeclaration(m map[string]any) (map[string]any, error) {
	decl := map[string]any{"name": jsonx.GetString(m, "name")}
	if d, ok := m["description"]; ok && d != nil {
		decl["description"] = d
	} else {
		decl["description"] = ""
	}
	params, err := toolParameters(m["parameters"])
	if err != nil {
		return nil, err
	}
	decl["parameters"] = params
	return decl, nil
}

// customToolDeclaration renders a custom tool as a function declaration: the
// grammar rides the description so the model can honour it, and the one
// freeform string the format takes becomes the parameters.
func customToolDeclaration(m map[string]any) map[string]any {
	name := jsonx.GetString(m, "name")
	desc := jsonx.GetString(m, "description") + grammarSuffix(m["format"])
	return map[string]any{
		"name":        name,
		"description": desc,
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{"content": map[string]any{
				"type":        "string",
				"description": "The " + name + " content following the specified format",
			}},
			"required": []any{"content"},
		},
	}
}

// grammarSuffix renders a custom tool's grammar as the Format block appended
// to its description, so the model can produce format-shaped output. A format
// with no definition contributes nothing.
func grammarSuffix(format any) string {
	fm, ok := jsonx.AsMap(format)
	if !ok {
		return ""
	}
	def := jsonx.GetString(fm, "definition")
	if def == "" {
		return ""
	}
	return "\n\nFormat:\n```" + jsonx.GetString(fm, "syntax") + "\n" + def + "\n```"
}

// namespaceToolDeclarations flattens a namespaced tool group into plain
// declarations named `namespace__tool`, the spelling the wire's function
// vocabulary accepts.
func namespaceToolDeclarations(m map[string]any) []any {
	ns := jsonx.GetString(m, "name")
	nsDesc := jsonx.GetString(m, "description")
	nested, ok := jsonx.AsSlice(m["tools"])
	if !ok {
		// A flat namespace tool is itself the function.
		return []any{namespaceDeclaration(ns, nsDesc, m, false)}
	}
	var out []any
	for _, raw := range nested {
		tm, ok := raw.(map[string]any)
		if !ok || jsonx.GetString(tm, "type") != "function" {
			continue
		}
		out = append(out, namespaceDeclaration(ns, nsDesc, tm, true))
	}
	return out
}

func namespaceDeclaration(ns, nsDesc string, tool map[string]any, nested bool) map[string]any {
	name := jsonx.GetString(tool, "name")
	desc := jsonx.GetString(tool, "description")
	if nested {
		if nsDesc != "" && desc != "" {
			desc = nsDesc + "\n\n" + desc
		} else if nsDesc != "" {
			desc = nsDesc
		}
		name = ns + "__" + name
	}
	decl := map[string]any{"name": name, "description": desc}
	params, ok := jsonx.AsMap(tool["parameters"])
	if !ok {
		params = map[string]any{}
	}
	cp, _ := deepCopyValue(params).(map[string]any)
	if _, has := cp["type"]; !has {
		cp["type"] = "object"
	}
	decl["parameters"] = cp
	return decl
}

// responsesToolChoice maps a Responses tool choice onto the wire's calling
// config. The chat hop normalises the Responses spellings first, so only the
// three mode strings and the named-function shape reach the mapping; a value
// none of them can carry is refused unless the caller asked for dropping.
func responsesToolChoice(v any, dropParams bool) (map[string]any, error) {
	unsupported := func() (map[string]any, error) {
		if dropParams {
			return nil, nil
		}
		return nil, cerr.UnsupportedParam{Param: "tool_choice"}
	}
	anyMode := func(names []any) map[string]any {
		cfg := map[string]any{"mode": "ANY"}
		if names != nil {
			cfg["allowed_function_names"] = names
		}
		return map[string]any{"functionCallingConfig": cfg}
	}
	switch t := v.(type) {
	case string:
		switch t {
		case "none":
			return map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}, nil
		case "auto":
			return map[string]any{"functionCallingConfig": map[string]any{"mode": "AUTO"}}, nil
		case "required":
			return anyMode(nil), nil
		default:
			return unsupported()
		}
	case map[string]any:
		if fn, ok := jsonx.AsMap(t["function"]); ok {
			if name := jsonx.GetString(fn, "name"); name != "" {
				return anyMode([]any{name}), nil
			}
		}
		switch jsonx.GetString(t, "type") {
		case "auto":
			return map[string]any{"functionCallingConfig": map[string]any{"mode": "AUTO"}}, nil
		case "none":
			return map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}, nil
		case "required", "tool", "any":
			return anyMode(nil), nil
		case "function", "custom":
			name := jsonx.GetString(t, "name")
			if name == "" {
				if cm, ok := jsonx.AsMap(t["custom"]); ok {
					name = jsonx.GetString(cm, "name")
				}
			}
			if name != "" {
				return anyMode([]any{name}), nil
			}
			return anyMode(nil), nil
		default:
			// A shape the chat hop forwards as-is; the wire reads whatever
			// function name it names — none, so the blank entry it produces.
			return anyMode([]any{""}), nil
		}
	default:
		// Values that are neither a string nor an object are not read by the
		// wire's choice mapping at all.
		return nil, nil
	}
}

// systemInstructionText joins the system instruction's part texts with
// newlines — the fold the reverse direction applies to reconstruct the
// instructions string.
func systemInstructionText(v any) string {
	sys, ok := jsonx.AsMap(v)
	if !ok {
		return ""
	}
	parts, ok := jsonx.AsSlice(sys["parts"])
	if !ok {
		return ""
	}
	var texts []string
	for _, p := range parts {
		if pm, ok := p.(map[string]any); ok {
			if s, ok := pm["text"].(string); ok {
				texts = append(texts, s)
			}
		}
	}
	return strings.Join(texts, "\n")
}

// geminiContentsToInput folds the wire's contents onto the Responses input
// list: each content walks its parts in order, text and media parts group
// into one message item, thought parts group into a reasoning item ahead of
// it, and function calls and responses become their own items.
func geminiContentsToInput(v any) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return []any{}
	}
	var out []any
	callSeq := 0
	byName := map[string]string{}
	for _, raw := range list {
		cm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role := "user"
		textType := "input_text"
		if jsonx.GetString(cm, "role") == "model" {
			role = "assistant"
			textType = "output_text"
		}
		parts, _ := jsonx.AsSlice(cm["parts"])
		var msgParts []any
		var summary []any
		flushSummary := func() {
			if len(summary) > 0 {
				out = append(out, map[string]any{"type": "reasoning", "summary": summary})
				summary = nil
			}
		}
		flushMessage := func() {
			flushSummary()
			if len(msgParts) > 0 {
				out = append(out, map[string]any{
					"type": "message", "role": role, "content": msgParts,
				})
				msgParts = nil
			}
		}
		for _, p := range parts {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch {
			case pm["function_call"] != nil:
				flushMessage()
				fc, _ := jsonx.AsMap(pm["function_call"])
				id := jsonx.GetString(fc, "id")
				if id == "" {
					id = fmt.Sprintf("call_%d", callSeq)
					callSeq++
				}
				name := jsonx.GetString(fc, "name")
				byName[name] = id
				out = append(out, map[string]any{
					"type": "function_call", "call_id": id, "name": name,
					"arguments": compactJSON(fc["args"]),
				})
			case pm["function_response"] != nil:
				flushMessage()
				fr, _ := jsonx.AsMap(pm["function_response"])
				id := jsonx.GetString(fr, "id")
				if id == "" {
					// Pair the result with the call it answers by name, the
					// only label a result carries; an unpaired one still needs
					// an id, so the sequence mints it.
					id = byName[jsonx.GetString(fr, "name")]
				}
				if id == "" {
					id = fmt.Sprintf("call_%d", callSeq)
					callSeq++
				}
				out = append(out, map[string]any{
					"type": "function_call_output", "call_id": id,
					"output": functionResponseOutput(fr["response"]),
				})
			case isThought(pm):
				flushMessage()
				if s, ok := pm["text"].(string); ok && s != "" {
					summary = append(summary, map[string]any{"type": "summary_text", "text": s})
				}
			case pm["inline_data"] != nil || pm["inlineData"] != nil:
				idata, _ := jsonx.AsMap(firstNonNil(pm["inline_data"], pm["inlineData"]))
				mime := jsonx.GetString(idata, "mime_type")
				if mime == "" {
					mime = jsonx.GetString(idata, "mimeType")
				}
				msgParts = append(msgParts, map[string]any{
					"type":      "input_image",
					"image_url": "data:" + mime + ";base64," + jsonx.GetString(idata, "data"),
				})
			case pm["file_data"] != nil || pm["fileData"] != nil:
				fdata, _ := jsonx.AsMap(firstNonNil(pm["file_data"], pm["fileData"]))
				msgParts = append(msgParts, map[string]any{
					"type":     "input_file",
					"file_url": firstNonEmpty(jsonx.GetString(fdata, "file_uri"), jsonx.GetString(fdata, "fileUri")),
				})
			default:
				// A bare thoughtSignature rides nothing on the Responses
				// wire; the reference's fold drops it there.
				if s, ok := pm["text"].(string); ok {
					msgParts = append(msgParts, map[string]any{"type": textType, "text": s})
				}
			}
		}
		flushMessage()
	}
	return out
}

// isThought reports whether a part is a thought part.
func isThought(pm map[string]any) bool {
	b, _ := jsonx.Bool(pm["thought"])
	return b
}

// functionResponseOutput renders a function response's payload as the
// Responses output: the folded {"content": string} shape reads back as the
// string, anything else as its compact JSON.
func functionResponseOutput(resp any) any {
	if rm, ok := resp.(map[string]any); ok && len(rm) == 1 {
		if s, ok := rm["content"].(string); ok {
			return s
		}
	}
	return compactJSON(resp)
}

// compactJSON renders a decoded value as compact JSON; nil renders as an
// empty object, the arguments a call with no parameters carries.
func compactJSON(v any) string {
	if v == nil {
		return "{}"
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// thinkingConfigToEffort buckets a thinking config back into an effort level.
// The reading lives in internal/thinkingcfg — the one place that knows both
// spellings a gemini wire states depth in (budget and thinkingLevel), so this
// fold and the chat one cannot disagree about what a config means.
func thinkingConfigToEffort(v any) (string, bool) {
	tc, ok := jsonx.AsMap(v)
	if !ok {
		return "", false
	}
	return thinkingcfg.Effort(tc)
}

// responseMimeToText rebuilds the Responses text.format from the generation
// config: a JSON schema rides with a fixed name and strict off (the wire has
// no strict), JSON without a schema is the plain object request, and the text
// mime is an explicit no-structured-output request.
func responseMimeToText(gc map[string]any) map[string]any {
	switch jsonx.GetString(gc, "response_mime_type") {
	case "application/json":
		// The schema crosses enum-normalized — see ResponseSchemaOf for why a
		// fold must not relay the gemini wire's uppercase spellings.
		if sm, ok := schemanorm.ResponseSchemaOf(gc); ok {
			return map[string]any{"format": map[string]any{
				"type": "json_schema", "name": "response_schema", "schema": sm, "strict": false,
			}}
		}
		return map[string]any{"format": map[string]any{"type": "json_object"}}
	case "text/plain":
		return map[string]any{"format": map[string]any{"type": "text"}}
	default:
		return nil
	}
}

// geminiToolsToResponses folds the wire's tools onto the Responses tool list:
// every declaration becomes a function tool (description and parameters
// always stated, strict off), and the wire's own search tool becomes the
// Responses search spelling. Tool keys with no Responses equivalent drop.
func geminiToolsToResponses(v any) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var out []any
	for _, raw := range list {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if decls, ok := jsonx.AsSlice(m["function_declarations"]); ok {
			// Same enum re-spelling as the chat fold: the Responses dialect's
			// parameters validate the lowercase vocabulary.
			schemanorm.NormalizeTree(decls)
			for _, d := range decls {
				dm, ok := d.(map[string]any)
				if !ok {
					continue
				}
				tool := map[string]any{
					"type":   "function",
					"name":   jsonx.GetString(dm, "name"),
					"strict": false,
				}
				if desc, ok := dm["description"]; ok && desc != nil {
					tool["description"] = desc
				} else {
					tool["description"] = ""
				}
				if params, ok := schemanorm.SchemaOf(dm); ok {
					tool["parameters"] = params
				} else {
					tool["parameters"] = map[string]any{"type": "object"}
				}
				out = append(out, tool)
			}
		}
		if _, ok := m["googleSearch"]; ok {
			out = append(out, map[string]any{"type": "web_search_preview"})
		}
	}
	return out
}

// geminiToolConfigToChoice maps the calling config back onto a Responses tool
// choice: the three modes are the three choice strings, and a config naming
// exactly one function asks for that function.
func geminiToolConfigToChoice(v any) (any, bool) {
	tc, ok := jsonx.AsMap(v)
	if !ok {
		return nil, false
	}
	fcc, ok := jsonx.AsMap(tc["functionCallingConfig"])
	if !ok {
		return nil, false
	}
	if names, ok := jsonx.AsSlice(fcc["allowed_function_names"]); ok && len(names) == 1 {
		if n, ok := names[0].(string); ok && n != "" {
			return map[string]any{"type": "function", "name": n}, true
		}
	}
	switch jsonx.GetString(fcc, "mode") {
	case "NONE":
		return "none", true
	case "AUTO":
		return "auto", true
	case "ANY":
		return "required", true
	default:
		return nil, false
	}
}

func firstNonNil(vs ...any) any {
	for _, v := range vs {
		if v != nil {
			return v
		}
	}
	return nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
