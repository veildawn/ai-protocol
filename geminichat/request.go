package geminichat

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/veildawn/ai-protocol/internal/cerr"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/internal/schemanorm"
	"github.com/veildawn/ai-protocol/internal/thinkingcfg"
)

// chatVocabulary is the Chat dialect's request vocabulary: the parameters a
// chat body may carry that name something at all. Keys outside it are neither
// mapped nor refused — the reference's request builder reads a fixed set of
// fields, so an unknown key rides nothing onto the wire.
var chatVocabulary = map[string]struct{}{
	"messages": {}, "model": {}, "frequency_penalty": {}, "logit_bias": {},
	"logprobs": {}, "top_logprobs": {}, "max_tokens": {}, "max_completion_tokens": {},
	"n": {}, "presence_penalty": {}, "response_format": {}, "seed": {}, "stop": {},
	"stream": {}, "stream_options": {}, "temperature": {}, "top_p": {}, "top_k": {},
	"tools": {}, "tool_choice": {}, "parallel_tool_calls": {}, "user": {},
	"function_call": {}, "functions": {}, "reasoning_effort": {}, "thinking": {},
	"metadata": {}, "modalities": {}, "prediction": {}, "service_tier": {},
	"store": {}, "web_search_options": {}, "extra_headers": {}, "timeout": {},
	"speed": {}, "context_management": {}, "cache_control": {}, "audio": {},
	"safety_settings": {}, "cached_content": {}, "include_server_side_tool_invocations": {},
}

// geminiRefusedParams are the chat parameters the target has no spelling for at
// all. The reference refuses them from its parameter check — dropping them is
// a caller's explicit choice, not a fold's silent one.
var geminiRefusedParams = map[string]struct{}{
	"logit_bias": {}, "function_call": {}, "metadata": {}, "prediction": {},
	"store": {}, "context_management": {},
}

// thoughtSignatureSeparator marks a thought-signature suffix some chat clients
// append to a tool call id. The signature is not part of the id on the target
// wire, so the fold lifts it off onto the part it signs.
const thoughtSignatureSeparator = "__thought__"

// skipThoughtSignature is the marker the upstream's own validator accepts in
// place of a thought signature a client could not preserve — the spelling the
// official hub client sends for a replayed call whose signature it never saw.
// A missing signature and a borrowed one are both refused (INVALID_ARGUMENT,
// verified against the live upstream), so the fallback is the sentinel or
// nothing.
const skipThoughtSignature = "skip_thought_signature_validator"

// partThoughtSignature reads a part's thought signature in either spelling.
// Google's own wire spells the member BOTH ways — the plugin's reader notes
// "thoughtSignature comes and goes" — so a reader that consults one spelling
// alone silently drops the signature half the time. A dropped signature is
// not a cosmetic loss: it rides the tool call id the client replays, and an
// unsigned replayed call is refused by the upstream on the NEXT request.
// Mirrors the plugin's own part_thought_signature, which checks the same two
// keys.
func partThoughtSignature(pm map[string]any) string {
	if s := jsonx.GetString(pm, "thoughtSignature"); s != "" {
		return s
	}
	return jsonx.GetString(pm, "thought_signature")
}

// ChatToGemini folds a chat request body into the target dialect's
// generateContent request body.
//
// The body has no `model` member (the model rides the URL path, not the
// payload) and no `stream` member (streaming is a query flag on the same path),
// so both are consumed rather than carried. Parameters the target cannot
// represent are refused unless dropParams is set; the ones it accepts but never
// reads (`user`, `stream_options`, `timeout`, `extra_headers`, `speed`,
// `cache_control`, `parallel_tool_calls`) ride nothing, exactly as in the
// reference.
func ChatToGemini(body map[string]any, dropParams bool) (map[string]any, error) {
	if err := rejectChatParams(body, dropParams); err != nil {
		return nil, err
	}
	out := map[string]any{}
	msgs, _ := jsonx.AsSlice(body["messages"])
	systemParts, rest := extractSystemParts(msgs)
	contents, err := chatMessagesToContents(rest)
	if err != nil {
		return nil, err
	}
	out["contents"] = contents
	if len(systemParts) > 0 {
		out["system_instruction"] = map[string]any{"parts": systemParts}
	}
	tools, err := chatToolsToWire(body)
	if err != nil {
		return nil, err
	}
	if tools != nil {
		out["tools"] = tools
	}
	toolConfig, err := chatToolConfigToWire(body)
	if err != nil {
		return nil, err
	}
	if toolConfig != nil {
		out["toolConfig"] = toolConfig
	}
	if v, ok := body["safety_settings"]; ok && v != nil {
		out["safetySettings"] = v
	}
	if v, ok := body["cached_content"]; ok && v != nil {
		out["cachedContent"] = v
	}
	gc, err := chatGenerationConfig(body)
	if err != nil {
		return nil, err
	}
	if len(gc) > 0 {
		out["generationConfig"] = gc
	}
	if st, ok := body["service_tier"].(string); ok && st != "" {
		out["serviceTier"] = serviceTierToWire(st)
	}
	return out, nil
}

func rejectChatParams(body map[string]any, drop bool) error {
	for k := range body {
		if _, refused := geminiRefusedParams[k]; !refused {
			continue
		}
		if _, isChat := chatVocabulary[k]; !isChat {
			continue
		}
		if drop {
			continue
		}
		return cerr.UnsupportedParam{Param: k}
	}
	return nil
}

// extractSystemParts lifts the system turns out of the conversation: every
// system message becomes one instruction part, its list content joined with no
// separator. A system message whose content is neither string nor list stays
// in the conversation, where it contributes no part at all — the reference
// leaves it there too. When nothing else survives, a single "." user turn
// keeps the request legal: the target refuses a body with no contents.
func extractSystemParts(msgs []any) (system []any, rest []any) {
	rest = make([]any, 0, len(msgs))
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok || jsonx.GetString(m, "role") != "system" {
			rest = append(rest, item)
			continue
		}
		switch c := m["content"].(type) {
		case string:
			system = append(system, map[string]any{"text": c})
		case []any:
			var b strings.Builder
			for _, el := range c {
				if em, ok := jsonx.AsMap(el); ok {
					if t, ok := em["text"].(string); ok {
						b.WriteString(t)
					}
				}
			}
			system = append(system, map[string]any{"text": b.String()})
		default:
			rest = append(rest, item)
		}
	}
	if len(system) > 0 && len(rest) == 0 {
		rest = append(rest, map[string]any{"role": "user", "content": "."})
	}
	return system, rest
}

func messageRole(item any) string {
	m, ok := jsonx.AsMap(item)
	if !ok {
		return ""
	}
	return jsonx.GetString(m, "role")
}

// chatMessagesToContents folds the conversation into the target's contents:
// consecutive user and system turns merge into one user content, consecutive
// assistant turns into one model content, and a run of tool results flushes as
// one user content carrying every function response. A turn whose role names
// none of the five the target reads is refused — the reference raises on it
// rather than guessing where the turn belongs.
func chatMessagesToContents(msgs []any) ([]any, error) {
	var contents []any
	var pendingToolResponses []any
	var lastToolCalls map[string]any
	i := 0
	for i < len(msgs) {
		start := i
		var userParts []any
		for i < len(msgs) {
			role := messageRole(msgs[i])
			if role != "user" && role != "system" {
				break
			}
			m, ok := jsonx.AsMap(msgs[i])
			if !ok {
				return nil, fmt.Errorf("message %d is not an object", i)
			}
			parts, err := userMessageParts(m)
			if err != nil {
				return nil, err
			}
			userParts = append(userParts, parts...)
			i++
		}
		if len(userParts) > 0 {
			if !hasTextPart(userParts) {
				// A user turn with no text part at all is refused upstream
				// ("must have a text parameter"); one blank cures it.
				userParts = append(userParts, map[string]any{"text": " "})
			}
			contents = append(contents, map[string]any{"role": "user", "parts": userParts})
		}
		var modelParts []any
		for i < len(msgs) && messageRole(msgs[i]) == "assistant" {
			m, ok := jsonx.AsMap(msgs[i])
			if !ok {
				return nil, fmt.Errorf("message %d is not an object", i)
			}
			parts, carriesCalls, err := assistantMessageParts(m)
			if err != nil {
				return nil, err
			}
			modelParts = append(modelParts, parts...)
			if carriesCalls {
				lastToolCalls = m
			}
			i++
		}
		if len(modelParts) > 0 {
			contents = append(contents, map[string]any{"role": "model", "parts": modelParts})
		}
		if i < len(msgs) && isToolRole(messageRole(msgs[i])) {
			m, ok := jsonx.AsMap(msgs[i])
			if !ok {
				return nil, fmt.Errorf("message %d is not an object", i)
			}
			parts, err := toolResponseParts(m, lastToolCalls)
			if err != nil {
				return nil, err
			}
			pendingToolResponses = append(pendingToolResponses, parts...)
			i++
		}
		if i < len(msgs) && !isToolRole(messageRole(msgs[i])) && len(pendingToolResponses) > 0 {
			contents = append(contents, map[string]any{"role": "user", "parts": pendingToolResponses})
			pendingToolResponses = nil
		}
		if i == start {
			return nil, fmt.Errorf("message %d has role %q: not a role the target reads", i, messageRole(msgs[i]))
		}
	}
	if len(pendingToolResponses) > 0 {
		contents = append(contents, map[string]any{"role": "user", "parts": pendingToolResponses})
	}
	if len(contents) == 0 {
		contents = []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": " "}}}}
	}
	return contents, nil
}

func isToolRole(role string) bool {
	return role == "tool" || role == "function"
}

// hasTextPart reports whether any part carries a `text` key with a value: the
// target's own blank-text cure is only for a turn that has no text at all —
// an empty string is a text parameter the upstream accepts.
func hasTextPart(parts []any) bool {
	for _, p := range parts {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		if v, ok := pm["text"]; ok && v != nil {
			return true
		}
	}
	return false
}

// userMessageParts folds one user (or leftover system) turn's content. A string
// content is one text part whatever it says; a list content contributes a part
// per element the target reads — text (non-empty), an image, audio, or a file —
// and every other element type names nothing the target can read.
func userMessageParts(m map[string]any) ([]any, error) {
	content := m["content"]
	list, isList := jsonx.AsSlice(content)
	if !isList {
		if s, isStr := content.(string); isStr {
			return []any{map[string]any{"text": s}}, nil
		}
		return nil, nil
	}
	var parts []any
	for _, el := range list {
		em, ok := jsonx.AsMap(el)
		if !ok {
			continue
		}
		switch jsonx.GetString(em, "type") {
		case "text":
			if t, ok := em["text"].(string); ok && len(t) > 0 {
				parts = append(parts, map[string]any{"text": t})
			}
		case "image_url":
			part, err := imageURLElementPart(em)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case "input_audio":
			part, err := inputAudioElementPart(em)
			if err != nil {
				return nil, err
			}
			if part != nil {
				parts = append(parts, part)
			}
		case "file":
			part, err := fileElementPart(em)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
	}
	return parts, nil
}

func imageURLElementPart(em map[string]any) (map[string]any, error) {
	raw := em["image_url"]
	if raw == nil {
		return nil, fmt.Errorf("image content part carries no image_url: the target needs the URL to read the media")
	}
	url := ""
	format := ""
	if iu, ok := jsonx.AsMap(raw); ok {
		u, has := iu["url"]
		if !has || u == nil {
			return nil, fmt.Errorf("image_url object carries no url: the target needs the URL to read the media")
		}
		url = jsonx.String(u)
		format = firstTruthyString(iu, "format", "mime_type", "content_type")
	} else {
		url = jsonx.String(raw)
	}
	return processMedia(url, format)
}

// inputAudioElementPart folds an input_audio element. Both the data and the
// format must be stated — one without the other names audio the target cannot
// type, and the reference drops the element rather than sending it untyped.
func inputAudioElementPart(em map[string]any) (map[string]any, error) {
	ia, ok := jsonx.AsMap(em["input_audio"])
	if !ok {
		return nil, nil
	}
	data, hasData := ia["data"]
	format, hasFormat := ia["format"]
	if !hasData || data == nil || !hasFormat || format == nil {
		return nil, nil
	}
	f := jsonx.String(format)
	if !strings.HasPrefix(f, "audio/") {
		f = "audio/" + f
	}
	return processMedia("data:"+f+";base64,"+jsonx.String(data), f)
}

func fileElementPart(em map[string]any) (map[string]any, error) {
	f, ok := jsonx.AsMap(em["file"])
	if !ok {
		return nil, fmt.Errorf("file content part carries no file object: the target needs it to read the bytes")
	}
	format := firstTruthyString(f, "format", "mime_type", "content_type")
	var passed any
	if id, ok := f["file_id"]; ok && id != nil {
		passed = id
	} else if fd, ok := f["file_data"]; ok && fd != nil {
		passed = fd
	} else {
		return nil, fmt.Errorf("file content part names neither file_id nor file_data")
	}
	part, err := processMedia(jsonx.String(passed), format)
	if err != nil {
		return nil, fmt.Errorf("unable to type the file part's media: %w", err)
	}
	return part, nil
}

// firstTruthyString returns the first key whose value is present and truthy,
// mirroring the reference's `or`-chained field reads.
func firstTruthyString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		v, ok := m[k]
		if !ok || v == nil || v == false {
			continue
		}
		if s := jsonx.String(v); s != "" {
			return s
		}
	}
	return ""
}

// fileExtensionMIMEs maps a storage-object extension onto the media type the
// target accepts for it. Anything outside this table is refused by the target,
// so the fold refuses it here rather than sending a type the upstream rejects.
var fileExtensionMIMEs = map[string]string{
	"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp",
	"aac": "audio/aac", "flac": "audio/flac", "mp3": "audio/mpeg", "mpa": "audio/m4a",
	"mpeg": "video/mpeg", "mpga": "audio/mpga", "opus": "audio/opus", "pcm": "audio/pcm",
	"wav": "audio/wav", "webm": "video/webm", "flv": "video/x-flv", "mov": "video/quicktime",
	"mpegps": "video/mpegps", "mpg": "video/mpg", "mp4": "video/mp4", "wmv": "video/wmv",
	"3gpp": "video/3gpp", "pdf": "application/pdf", "txt": "text/plain",
}

// urlExtensionMIMEs infers a media type from an https URL's extension, in the
// order the reference's table checks them.
var urlExtensionMIMEs = []struct{ ext, mime string }{
	{".jpg", "image/jpeg"}, {".jpeg", "image/jpeg"}, {".png", "image/png"},
	{".webp", "image/webp"}, {".mp4", "video/mp4"}, {".mov", "video/mov"},
	{".mpeg", "video/mpeg"}, {".mpg", "video/mpeg"}, {".avi", "video/avi"},
	{".wmv", "video/wmv"}, {".mpegps", "video/mpegps"}, {".flv", "video/flv"},
	{".mp3", "audio/mp3"}, {".wav", "audio/wav"}, {".ogg", "audio/ogg"},
	{".pdf", "application/pdf"}, {".txt", "text/plain"},
}

func urlExtensionMIME(url string) string {
	u := strings.ToLower(url)
	if i := strings.IndexAny(u, "?#"); i >= 0 {
		u = u[:i]
	}
	for _, e := range urlExtensionMIMEs {
		if strings.HasSuffix(u, e.ext) {
			return e.mime
		}
	}
	return ""
}

// normalizeMimeAlias lowercases a stated type, strips its parameters, and maps
// the one alias the target spells differently.
func normalizeMimeAlias(mime string) string {
	m := strings.ToLower(strings.TrimSpace(strings.SplitN(mime, ";", 2)[0]))
	if m == "image/jpg" {
		return "image/jpeg"
	}
	return m
}

// uploadedFileURI reports whether uri names a file the target already holds:
// an https URI whose path addresses the uploaded-files namespace. The reference
// hands such a URI over instead of fetching it — the file answers direct reads
// with a refusal, so passing it through is the only way to use it. The check is
// on the path's shape rather than the host: this codec carries no host
// knowledge, and every URI shaped like an upload reads the same way here.
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

func pathExtension(uri string) string {
	base := uri
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	dot := strings.LastIndex(base, ".")
	if dot <= 0 {
		return ""
	}
	return base[dot+1:]
}

// processMedia resolves one media reference onto the part spelling the target
// reads: bytes it must be handed inline, or a URI it fetches itself. This
// library has no network, so a reference the reference would fetch — an http
// URL, an https URL whose type no extension names — is refused here rather than
// silently dropped: the caller asked for media, and losing it would hand the
// target a prompt that no longer says what the caller said.
func processMedia(url string, format string) (map[string]any, error) {
	if strings.Contains(url, "gs://") {
		mime := ""
		if format != "" {
			mime = normalizeMimeAlias(format)
		} else {
			ext := strings.ToLower(pathExtension(url))
			m, ok := fileExtensionMIMEs[ext]
			if !ok {
				return nil, fmt.Errorf("storage URI %q names no media type the target accepts", url)
			}
			mime = m
		}
		return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
	}
	if uploadedFileURI(url) {
		fd := map[string]any{"file_uri": url}
		if format != "" {
			fd["mime_type"] = format
		}
		return map[string]any{"file_data": fd}, nil
	}
	if strings.Contains(url, "https://") {
		mime := format
		if mime == "" {
			mime = urlExtensionMIME(url)
		}
		if mime != "" {
			return map[string]any{"file_data": map[string]any{"mime_type": mime, "file_uri": url}}, nil
		}
	}
	if strings.Contains(url, "http://") || strings.Contains(url, "https://") || strings.Contains(url, "base64") {
		media, data, err := splitDataURL(url)
		if err != nil {
			return nil, err
		}
		if format != "" {
			media = format
		}
		return map[string]any{"inline_data": map[string]any{"data": data, "mime_type": media}}, nil
	}
	return nil, fmt.Errorf("media reference %q names neither bytes the target reads inline nor a URI it fetches", url)
}

func splitDataURL(url string) (media, data string, err error) {
	i := strings.Index(url, "data:")
	if i < 0 {
		return "", "", fmt.Errorf("media reference %q is not a data URL the target reads inline", url)
	}
	rest := url[i+len("data:"):]
	before, after, found := strings.Cut(rest, ";base64,")
	if !found {
		return "", "", fmt.Errorf("media reference %q carries no base64 payload the target reads inline", url)
	}
	return strings.ReplaceAll(before, `\/`, "/"), after, nil
}

// assistantMessageParts folds one assistant turn. The part order is the
// reference's: the reasoning summary first, then signed thinking blocks, the
// text, generated images, and the calls. carriesCalls reports whether the turn
// actually names a call — the tool results that follow pair against it.
func assistantMessageParts(m map[string]any) (parts []any, carriesCalls bool, err error) {
	if rc, ok := m["reasoning_content"]; ok && rc != nil {
		parts = append(parts, map[string]any{"thought": true, "text": jsonx.String(rc)})
	}
	if raw, ok := m["thinking_blocks"]; ok && raw != nil {
		if tbs, isList := jsonx.AsSlice(raw); isList {
			for _, b := range tbs {
				bm, isMap := jsonx.AsMap(b)
				if !isMap || jsonx.GetString(bm, "type") != "thinking" {
					// A redacted block carries opaque bytes no fold can read;
					// the reference drops it too.
					continue
				}
				text, hasText := bm["thinking"]
				sig, hasSig := bm["signature"]
				if !hasText || text == nil || !hasSig || sig == nil {
					// An unsigned block names reasoning the target cannot
					// verify; the reference drops it rather than replay it.
					continue
				}
				parts = append(parts, thinkingBlockPart(jsonx.String(text), jsonx.String(sig)))
			}
		}
	}
	content := m["content"]
	if list, isList := jsonx.AsSlice(content); isList {
		// Only text parts of a list content ride; an image the caller put in an
		// assistant list has no target spelling there (images ride the images
		// field below), and the reference drops it.
		for _, el := range list {
			em, ok := jsonx.AsMap(el)
			if !ok {
				continue
			}
			if jsonx.GetString(em, "type") == "text" {
				parts = append(parts, map[string]any{"text": em["text"]})
			}
		}
	} else if s, isStr := content.(string); isStr {
		if sig := assistantTextSignature(m); sig != "" {
			parts = append(parts, map[string]any{"text": s, "thoughtSignature": sig})
		} else {
			parts = append(parts, map[string]any{"text": s})
		}
	}
	if imgs, ok := jsonx.AsSlice(m["images"]); ok && m["images"] != nil {
		for _, item := range imgs {
			im, ok := jsonx.AsMap(item)
			if !ok {
				continue
			}
			iu, ok := jsonx.AsMap(im["image_url"])
			if !ok {
				continue
			}
			url := jsonx.GetString(iu, "url")
			if url == "" {
				continue
			}
			part, err := processMedia(url, firstTruthyString(iu, "format", "mime_type", "content_type"))
			if err != nil {
				return nil, false, err
			}
			parts = append(parts, part)
		}
	}
	if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok && m["tool_calls"] != nil {
		for _, tc := range tcs {
			tm, isMap := jsonx.AsMap(tc)
			if !isMap {
				continue
			}
			fn, hasFn := jsonx.AsMap(tm["function"])
			if !hasFn {
				return nil, false, fmt.Errorf("tool call carries no function object: the target cannot send a call it cannot name")
			}
			part, err := functionCallPart(fn, tm)
			if err != nil {
				return nil, false, err
			}
			if !partExistsInParts(parts, part) {
				parts = append(parts, part)
			}
		}
		if len(tcs) > 0 {
			carriesCalls = true
		}
		return parts, carriesCalls, nil
	}
	if fc, ok := jsonx.AsMap(m["function_call"]); ok {
		part, err := functionCallPart(fc, nil)
		if err != nil {
			return nil, false, err
		}
		if !partExistsInParts(parts, part) {
			parts = append(parts, part)
		}
		carriesCalls = true
	}
	return parts, carriesCalls, nil
}

// thinkingBlockPart renders one signed thinking block. The reference first
// tries to read the block's text as a JSON object and spread its keys onto the
// part; only when the text is not an object does it send the text verbatim.
// thinkingBlockPart builds the part a signed thinking block becomes: a
// thought part carrying the signature itself, so the reverse fold can read
// the signature back off it. (The reference's encoding — a plain text part
// with the signature beside it — loses the signature on the way back, since
// the chat wire's reverse reads signatures only off thought parts, and a
// block whose text parses as a JSON object used to be spread as part keys;
// both dropped in this fold's audit.)
func thinkingBlockPart(text, sig string) map[string]any {
	part := map[string]any{"thought": true, "text": text, "thoughtSignature": sig}
	return part
}

// assistantTextSignature reads the signature a replay attaches to the turn's
// text — but only when no tool-call part of the same message already carries
// it: the target bills every replayed copy as the previous turn's reasoning.
func assistantTextSignature(m map[string]any) string {
	psf, ok := jsonx.AsMap(m["provider_specific_fields"])
	if !ok {
		return ""
	}
	sigs, ok := jsonx.AsSlice(psf["thought_signatures"])
	if !ok || len(sigs) == 0 {
		return ""
	}
	first := jsonx.String(sigs[0])
	if first == "" {
		return ""
	}
	for _, carried := range toolCallSignatures(m) {
		if carried == first {
			return ""
		}
	}
	return first
}

func toolCallSignatures(m map[string]any) []string {
	var out []string
	if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok {
		for _, tc := range tcs {
			if tm, ok := jsonx.AsMap(tc); ok {
				if s := toolCallThoughtSignature(tm); s != "" {
					out = append(out, s)
				}
			}
		}
	}
	if fc, ok := jsonx.AsMap(m["function_call"]); ok {
		if s := functionCallThoughtSignature(fc); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// toolCallThoughtSignature reads a call's signature from the field a replay
// writes it in — on the call, on its function, or as the suffix the chat wire
// allows on the call id.
func toolCallThoughtSignature(tm map[string]any) string {
	if psf, ok := jsonx.AsMap(tm["provider_specific_fields"]); ok {
		if s := jsonx.GetString(psf, "thought_signature"); s != "" {
			return s
		}
	}
	if fn, ok := jsonx.AsMap(tm["function"]); ok {
		if s := functionCallThoughtSignature(fn); s != "" {
			return s
		}
	}
	if id := jsonx.GetString(tm, "id"); id != "" {
		if _, sig, found := strings.Cut(id, thoughtSignatureSeparator); found && sig != "" {
			return sig
		}
	}
	return ""
}

func functionCallThoughtSignature(fn map[string]any) string {
	psf, ok := jsonx.AsMap(fn["provider_specific_fields"])
	if !ok {
		return ""
	}
	return jsonx.GetString(psf, "thought_signature")
}

// functionCallPart renders one call. An empty arguments string is not a shape
// the target accepts, so the reference substitutes a placeholder object; a
// value that is not JSON at all is refused, the same refusal the reference
// raises. The call's id is deliberately not written: only the newest models
// read it, and the older ones reject the field outright.
func functionCallPart(fn map[string]any, tm map[string]any) (map[string]any, error) {
	name := jsonx.GetString(fn, "name")
	raw, hasArgs := fn["arguments"]
	if !hasArgs {
		raw = ""
	}
	var args any
	switch t := raw.(type) {
	case string:
		if len(t) == 0 {
			args = map[string]any{"type": "object"}
			break
		}
		// An object rides the string's own key order, re-rendered with the
		// reference's separators: a decoded map would alphabetize the keys.
		if ordered, ok := jsonx.ArgsValue(t); ok {
			args = ordered
			break
		}
		parsed, err := parseJSONValue(t)
		if err != nil {
			return nil, fmt.Errorf("unable to convert the tool call %q: its arguments are not JSON: %w", name, err)
		}
		args = parsed
	default:
		return nil, fmt.Errorf("unable to convert the tool call %q: its arguments are not a JSON string", name)
	}
	part := map[string]any{"function_call": map[string]any{"name": name, "args": args}}
	sig := ""
	if tm != nil {
		sig = toolCallThoughtSignature(tm)
	}
	if sig == "" {
		sig = functionCallThoughtSignature(fn)
	}
	if sig == "" {
		// The target refuses a replayed functionCall part without a
		// thoughtSignature (INVALID_ARGUMENT, "Function call is missing a
		// thought_signature in functionCall parts") and validates the value,
		// so a client that preserved none has nothing valid to send: the part
		// rides the upstream's own skip marker. A preserved signature wins.
		sig = skipThoughtSignature
	}
	part["thoughtSignature"] = sig
	return part, nil
}

// partExistsInParts reports whether an equivalent part is already in the list,
// ignoring the signature member: a replayed turn can carry the same call twice
// with different signatures, and the target would bill both.
func partExistsInParts(parts []any, part map[string]any) bool {
	keys := make([]string, 0, len(part))
	for k := range part {
		if k != "thoughtSignature" {
			keys = append(keys, k)
		}
	}
	for _, raw := range parts {
		pm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		found := true
		for _, k := range keys {
			got, has := equivalentField(pm, k)
			if !has || !reflect.DeepEqual(got, part[k]) {
				found = false
				break
			}
		}
		if found {
			return true
		}
	}
	return false
}

// equivalentField reads a key from a part under either of the two spellings
// the wire accepts for it.
func equivalentField(m map[string]any, key string) (any, bool) {
	if v, ok := m[key]; ok {
		return v, true
	}
	for _, alt := range [2]string{snakeToCamel(key), camelToSnake(key)} {
		if alt == key || alt == "" {
			continue
		}
		if v, ok := m[alt]; ok {
			return v, true
		}
	}
	return nil, false
}

func snakeToCamel(s string) string {
	parts := strings.Split(s, "_")
	if len(parts) == 1 {
		return s
	}
	out := parts[0]
	for _, p := range parts[1:] {
		if p == "" {
			continue
		}
		out += strings.ToUpper(p[:1]) + p[1:]
	}
	return out
}

func camelToSnake(s string) string {
	var b strings.Builder
	for i, r := range s {
		if r >= 'A' && r <= 'Z' {
			if i > 0 {
				b.WriteByte('_')
			}
			b.WriteRune(r + ('a' - 'A'))
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// toolResponseParts folds one tool result. The function's name comes from the
// message itself or from the call it answers — matched by call id against the
// last assistant turn that carried calls. A result whose text parses as a JSON
// object is handed over as the response itself; anything else is wrapped under
// a `content` key, the shape the target reads as a plain result.
func toolResponseParts(m map[string]any, lastToolCalls map[string]any) ([]any, error) {
	contentStr := ""
	var inline []any
	if content, ok := m["content"]; ok && content != nil {
		switch c := content.(type) {
		case string:
			contentStr = c
			if len(c) >= 5 && strings.EqualFold(c[:5], "data:") {
				if before, after, found := strings.Cut(c[5:], ";base64,"); found && strings.HasPrefix(before, "image/") {
					// A data-URL image rides as bytes; sending the URI as the
					// result text would hand the target a string it cannot see.
					clean := strings.SplitN(before, ";", 2)[0]
					inline = append(inline, map[string]any{"data": after, "mime_type": clean})
					contentStr = ""
				}
			}
		case []any:
			for _, el := range c {
				em, ok := jsonx.AsMap(el)
				if !ok {
					continue
				}
				switch jsonx.GetString(em, "type") {
				case "text":
					if t, ok := em["text"].(string); ok {
						contentStr += t
					}
				case "image":
					if src, ok := jsonx.AsMap(em["source"]); ok && jsonx.GetString(src, "type") == "base64" {
						mt := jsonx.GetString(src, "media_type")
						if mt == "" {
							mt = "image/jpeg"
						}
						if d := jsonx.GetString(src, "data"); d != "" {
							inline = append(inline, map[string]any{"data": d, "mime_type": mt})
						}
					}
				case "input_image", "image_url":
					url := ""
					if iu, ok := jsonx.AsMap(em["image_url"]); ok {
						url = jsonx.GetString(iu, "url")
					} else if s, ok := em["image_url"].(string); ok {
						url = s
					}
					if url != "" {
						// Only inline bytes ride here: the reference drops a
						// result image it cannot read offline rather than
						// failing the whole request over it.
						if b, ok := inlineFromDataURL(url); ok {
							inline = append(inline, b)
						}
					}
				case "file", "input_file":
					fd := jsonx.GetString(em, "file_data")
					if fd == "" {
						if f, ok := jsonx.AsMap(em["file"]); ok {
							fd = jsonx.GetString(f, "file_data")
						} else if s, ok := em["file"].(string); ok {
							fd = s
						}
					}
					if fd != "" {
						if b, ok := inlineFromDataURL(fd); ok {
							inline = append(inline, b)
						}
					}
				}
			}
		}
	}
	name := jsonx.GetString(m, "name")
	if lastToolCalls != nil {
		if callID := jsonx.GetString(m, "tool_call_id"); callID != "" {
			if tcs, ok := jsonx.AsSlice(lastToolCalls["tool_calls"]); ok {
				for _, tc := range tcs {
					tm, isMap := jsonx.AsMap(tc)
					if !isMap || jsonx.GetString(tm, "id") != callID {
						continue
					}
					if fn, ok := jsonx.AsMap(tm["function"]); ok {
						name = jsonx.GetString(fn, "name")
					}
				}
			}
		}
	}
	if name == "" {
		return nil, fmt.Errorf("missing corresponding tool call for the tool response: the target pairs a result with the function's name")
	}
	response := map[string]any{"content": contentStr}
	trimmed := strings.TrimSpace(contentStr)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		if parsed, err := parseJSONValue(trimmed); err == nil {
			// A `$ref` key anywhere in the parsed object keeps the string
			// wrapped: the gemini wire reserves `$ref` inside a function
			// response as a reference to a media part's display_name, so a
			// tool result that happens to print a JSON Schema is refused
			// outright upstream when its `$ref`s ride as structure (400,
			// INVALID_ARGUMENT, "The referenced name ... does not match to a
			// display_name in the function_response.parts"). As text the
			// schema reaches the model verbatim.
			if pm, isMap := parsed.(map[string]any); isMap && !hasReservedRefKey(pm) {
				response = pm
			}
		}
	}
	fr := map[string]any{"name": name, "response": response}
	if len(inline) > 0 {
		nested := make([]any, 0, len(inline))
		for _, b := range inline {
			nested = append(nested, map[string]any{"inline_data": b})
		}
		// Media in a result rides inside the function response, not beside it.
		fr["parts"] = nested
	}
	return []any{map[string]any{"function_response": fr}}, nil
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

// inlineFromDataURL reads a data URL into the inline-blob shape, for the media
// a tool result carries.
func inlineFromDataURL(url string) (map[string]any, bool) {
	media, data, err := splitDataURL(url)
	if err != nil {
		return nil, false
	}
	return map[string]any{"data": data, "mime_type": media}, true
}

func parseJSONValue(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return v, nil
}

// chatToolsToWire folds the tools list (and the legacy functions list) plus
// web_search_options into the target's tools array. Function declarations from
// every source merge into a single entry; each native tool keeps an entry of
// its own. A search tool mixed with function declarations is dropped — the
// target refuses the combination — unless server-side tool invocations were
// asked for, which is the one setting that allows it.
func chatToolsToWire(body map[string]any) ([]any, error) {
	var mapped []any
	declared := false
	for _, key := range []string{"tools", "functions"} {
		list, ok := jsonx.AsSlice(body[key])
		if !ok || len(list) == 0 {
			continue
		}
		declared = true
		out, err := mapToolList(list)
		if err != nil {
			return nil, err
		}
		mapped = append(mapped, out...)
	}
	if _, ok := jsonx.AsMap(body["web_search_options"]); ok {
		declared = true
		mapped = append(mapped, map[string]any{"googleSearch": map[string]any{}})
	}
	if !declared {
		return nil, nil
	}
	if !includeServerSideToolInvocations(body) {
		hasDeclarations := false
		hasSearch := false
		for _, raw := range mapped {
			tm, ok := jsonx.AsMap(raw)
			if !ok {
				continue
			}
			if _, ok := tm["function_declarations"]; ok {
				hasDeclarations = true
			}
			for _, k := range []string{"googleSearch", "googleSearchRetrieval", "enterpriseWebSearch", "url_context"} {
				if _, ok := tm[k]; ok {
					hasSearch = true
				}
			}
		}
		if hasDeclarations && hasSearch {
			kept := make([]any, 0, len(mapped))
			for _, raw := range mapped {
				tm, ok := jsonx.AsMap(raw)
				if !ok {
					kept = append(kept, raw)
					continue
				}
				drop := false
				for _, k := range []string{"googleSearch", "googleSearchRetrieval", "enterpriseWebSearch", "url_context"} {
					if _, ok := tm[k]; ok {
						drop = true
					}
				}
				if !drop {
					kept = append(kept, raw)
				}
			}
			mapped = kept
		}
	}
	if mapped == nil {
		mapped = []any{}
	}
	return mapped, nil
}

func includeServerSideToolInvocations(body map[string]any) bool {
	for _, key := range []string{"include_server_side_tool_invocations", "includeServerSideToolInvocations"} {
		if b, ok := body[key].(bool); ok && b {
			return true
		}
	}
	return false
}

// specialToolWireKeys maps both spellings of a native tool's key onto the wire
// key the target reads.
var specialToolWireKeys = map[string]string{
	"codeExecution": "code_execution", "code_execution": "code_execution",
	"googleSearch": "googleSearch", "google_search": "googleSearch",
	"googleSearchRetrieval": "googleSearchRetrieval", "google_search_retrieval": "googleSearchRetrieval",
	"enterpriseWebSearch": "enterpriseWebSearch", "enterprise_web_search": "enterpriseWebSearch",
	"urlContext": "urlContext", "url_context": "urlContext",
	"googleMaps": "googleMaps", "google_maps": "googleMaps",
	"computerUse": "computerUse", "computer_use": "computerUse",
}

// mapToolList maps one tools or functions list. The reference strips a false
// `additionalProperties` and every `strict` off the whole list before it reads
// a declaration, and narrows a `function`-shaped tool's parameters through the
// schema builder — a legacy `functions` entry's parameters ride as they stand.
func mapToolList(list []any) ([]any, error) {
	removeAdditionalProperties(list)
	removeStrict(list)
	var declarations []any
	special := map[string]any{}
	for _, item := range list {
		tool, ok := jsonx.AsMap(item)
		if !ok {
			return nil, fmt.Errorf("tools entry is not an object: the target cannot read a declaration from it")
		}
		var fnObj map[string]any
		if raw, hasFn := tool["function"]; hasFn {
			fm, isMap := jsonx.AsMap(raw)
			if !isMap {
				return nil, fmt.Errorf("tools entry carries a function that is not an object")
			}
			fnObj = fm
			if params, ok := jsonx.AsMap(fm["parameters"]); ok && fm["parameters"] != nil {
				built, err := buildTargetSchema(params)
				if err != nil {
					return nil, err
				}
				fm["parameters"] = built
			}
		} else if _, hasName := tool["name"]; hasName {
			fnObj = tool
		}
		switch jsonx.GetString(tool, "type") {
		case "computer_use":
			cfg := map[string]any{}
			for k, v := range tool {
				if k != "type" {
					cfg[k] = v
				}
			}
			tool = map[string]any{"computerUse": cfg}
		case "web_search", "web_search_preview":
			tool = map[string]any{"googleSearch": map[string]any{}}
		case "":
			// No type key: the entry stands as written.
		default:
			stripped := map[string]any{}
			for k, v := range tool {
				if k != "type" {
					stripped[k] = v
				}
			}
			tool = stripped
		}
		if len(tool) == 1 {
			for k, v := range tool {
				if wire, isSpecial := specialToolWireKeys[k]; isSpecial {
					special[wire] = v
					fnObj = nil
					break
				}
			}
		}
		if fnObj == nil {
			continue
		}
		decl := map[string]any{"name": jsonx.GetString(fnObj, "name")}
		if d, ok := fnObj["description"]; ok && d != nil {
			decl["description"] = d
		}
		if params, ok := fnObj["parameters"]; ok {
			if s, isStr := params.(string); isStr && len(s) == 0 {
				params = map[string]any{"type": "object"}
			}
			if params != nil {
				decl["parameters"] = params
			}
		}
		declarations = append(declarations, decl)
	}
	var out []any
	if len(declarations) > 0 {
		out = append(out, map[string]any{"function_declarations": declarations})
	}
	// Each native tool keeps an entry of its own, in the fixed order the
	// reference assembles them.
	for _, slot := range []string{"googleSearch", "googleSearchRetrieval", "enterpriseWebSearch", "code_execution", "urlContext", "googleMaps", "computerUse"} {
		if v, ok := special[slot]; ok && v != nil {
			out = append(out, map[string]any{slot: v})
		}
	}
	return out, nil
}

// chatToolConfigToWire folds tool_choice (and the server-side invocation flag)
// into the target's toolConfig.
func chatToolConfigToWire(body map[string]any) (map[string]any, error) {
	var toolConfig map[string]any
	if raw, ok := body["tool_choice"]; ok {
		switch v := raw.(type) {
		case string:
			var mode string
			switch v {
			case "none":
				mode = "NONE"
			case "required":
				mode = "ANY"
			case "auto":
				mode = "AUTO"
			default:
				// The value error is not a droppable parameter: the reference
				// raises it even when drops were asked for.
				return nil, cerr.UnsupportedParam{Param: "tool_choice"}
			}
			toolConfig = map[string]any{"functionCallingConfig": map[string]any{"mode": mode}}
		case map[string]any:
			name := ""
			if fn, ok := jsonx.AsMap(v["function"]); ok {
				name = jsonx.GetString(fn, "name")
			}
			toolConfig = map[string]any{"functionCallingConfig": map[string]any{
				"mode":                   "ANY",
				"allowed_function_names": []any{name},
			}}
		}
	}
	if includeServerSideToolInvocations(body) {
		if toolConfig == nil {
			toolConfig = map[string]any{}
		}
		toolConfig["includeServerSideToolInvocations"] = true
	}
	return toolConfig, nil
}

// chatGenerationConfig folds the sampling and output parameters into the
// target's generationConfig. Only the members the target names survive; the
// rest of the chat vocabulary was consumed or dropped elsewhere.
func chatGenerationConfig(body map[string]any) (map[string]any, error) {
	gc := map[string]any{}
	for _, k := range []string{"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "seed"} {
		if v, ok := body[k]; ok && v != nil {
			gc[k] = v
		}
	}
	if v, ok := body["n"]; ok && v != nil {
		gc["candidate_count"] = v
	}
	if v, ok := body["max_completion_tokens"]; ok && v != nil {
		gc["max_output_tokens"] = v
	} else if v, ok := body["max_tokens"]; ok && v != nil {
		gc["max_output_tokens"] = v
	}
	if v, ok := body["stop"]; ok {
		switch t := v.(type) {
		case string:
			gc["stop_sequences"] = []any{t}
		case []any:
			gc["stop_sequences"] = t
		}
	}
	if v, ok := body["logprobs"]; ok && v != nil {
		gc["responseLogprobs"] = v
	}
	if v, ok := body["top_logprobs"]; ok && v != nil {
		gc["logprobs"] = v
	}
	if rf, ok := jsonx.AsMap(body["response_format"]); ok {
		mime, schema := responseFormatToWire(rf)
		if mime != "" {
			gc["response_mime_type"] = mime
		}
		if schema != nil {
			gc["response_json_schema"] = schema
		}
	}
	tc, err := thinkingConfigFromBody(body)
	if err != nil {
		return nil, err
	}
	if tc != nil {
		gc["thinkingConfig"] = tc
	}
	if list, ok := jsonx.AsSlice(body["modalities"]); ok {
		mapped := make([]any, 0, len(list))
		for _, m := range list {
			mapped = append(mapped, wireModality(jsonx.String(m)))
		}
		gc["responseModalities"] = mapped
	}
	if audio, ok := jsonx.AsMap(body["audio"]); ok {
		sc, err := speechConfigFromAudio(audio)
		if err != nil {
			return nil, err
		}
		gc["speechConfig"] = sc
		// Asking for speech output implies the audio modality; the reference
		// appends it when the request did not name its modalities itself.
		if mods, ok := jsonx.AsSlice(gc["responseModalities"]); ok {
			hasAudio := false
			for _, m := range mods {
				if m == "AUDIO" {
					hasAudio = true
				}
			}
			if !hasAudio {
				gc["responseModalities"] = append(mods, any("AUDIO"))
			}
		} else {
			gc["responseModalities"] = []any{"AUDIO"}
		}
	}
	return gc, nil
}

func wireModality(m string) string {
	switch m {
	case "text":
		return "TEXT"
	case "image":
		return "IMAGE"
	case "audio":
		return "AUDIO"
	case "video":
		return "VIDEO"
	default:
		return "MODALITY_UNSPECIFIED"
	}
}

// responseFormatToWire folds a response_format into the mime type (and the
// standard JSON Schema) the target reads. `strict` is removed recursively: the
// target has no such spelling, and a schema carrying it is refused outright.
func responseFormatToWire(rf map[string]any) (string, map[string]any) {
	value, _ := deepCopy(rf).(map[string]any)
	removeStrict(value)
	mime := ""
	switch jsonx.GetString(value, "type") {
	case "json_object":
		mime = "application/json"
	case "text":
		mime = "text/plain"
	}
	var schema any
	if v, ok := value["response_schema"]; ok {
		mime = "application/json"
		schema = v
	} else if jsonx.GetString(value, "type") == "json_schema" {
		if js, ok := jsonx.AsMap(value["json_schema"]); ok {
			if raw, has := js["schema"]; has {
				mime = "application/json"
				schema = raw
			}
		}
	}
	if sm, ok := schema.(map[string]any); ok && len(sm) > 0 {
		return mime, sm
	}
	return mime, nil
}

// thinkingConfigFromBody folds the two thinking spellings the chat wire
// carries. An explicit `thinking` object wins over `reasoning_effort` — the
// reference applies whichever it reads last, and a fold with no key order to
// read takes the more specific one.
func thinkingConfigFromBody(body map[string]any) (map[string]any, error) {
	if raw, ok := body["thinking"]; ok {
		t, isMap := jsonx.AsMap(raw)
		if !isMap {
			if raw == nil {
				return nil, nil
			}
			return nil, fmt.Errorf("thinking parameter is not an object: the target reads type and budget_tokens from it")
		}
		return thinkingParamToWire(t), nil
	}
	if raw, ok := body["reasoning_effort"]; ok {
		switch v := raw.(type) {
		case string:
			return effortToThinkingConfig(v)
		case map[string]any:
			if e, ok := v["effort"]; ok {
				if s, isStr := e.(string); isStr {
					return effortToThinkingConfig(s)
				}
				return nil, fmt.Errorf("invalid reasoning effort %v: the target has no thinking budget for it", e)
			}
			return nil, nil
		}
		return nil, nil
	}
	return nil, nil
}

// effortToThinkingConfig maps an effort label onto the thinking budget the
// reference's default table names for it. The library carries no model
// knowledge, so the general table is the one used — the per-model minimums the
// reference consults are the same value on every model this fold accepts.
func effortToThinkingConfig(effort string) (map[string]any, error) {
	switch effort {
	case "minimal":
		return map[string]any{"thinkingBudget": 128, "includeThoughts": true}, nil
	case "low":
		return map[string]any{"thinkingBudget": 1024, "includeThoughts": true}, nil
	case "medium":
		return map[string]any{"thinkingBudget": 2048, "includeThoughts": true}, nil
	case "high":
		return map[string]any{"thinkingBudget": 4096, "includeThoughts": true}, nil
	case "none", "disable":
		return map[string]any{"thinkingBudget": 0, "includeThoughts": false}, nil
	}
	return nil, fmt.Errorf("invalid reasoning effort %q: the target has no thinking budget for it", effort)
}

// thinkingParamToWire folds the object spelling: an enabled turn with a
// non-zero budget asks for its thoughts, and the budget rides when it is an
// integer at all. Anything else — disabled, adaptive, a budget of zero —
// yields the empty config, which the target reads as "no thinking asked for"
// and the reference still sends.
func thinkingParamToWire(t map[string]any) map[string]any {
	out := map[string]any{}
	enabled := jsonx.GetString(t, "type") == "enabled"
	budget, hasBudget := t["budget_tokens"]
	budgetZero := false
	if hasBudget {
		if n, ok := jsonx.Int(budget); ok {
			budgetZero = n == 0
		}
	}
	if enabled && !budgetZero {
		out["includeThoughts"] = true
	}
	if hasBudget && budget != nil {
		if _, isInt := jsonx.Int(budget); isInt {
			out["thinkingBudget"] = budget
		}
	}
	return out
}

// speechConfigFromAudio folds the audio parameter into the target's speech
// config. Only pcm output exists on this line, so any other stated format is
// refused rather than silently re-encoded.
func speechConfigFromAudio(audio map[string]any) (map[string]any, error) {
	if f, ok := audio["format"]; ok && f != nil {
		if s, isStr := f.(string); isStr && s != "pcm16" {
			return nil, fmt.Errorf("audio format %q: the target's speech output is pcm only", s)
		}
	}
	sc := map[string]any{}
	if v, ok := audio["voice"]; ok {
		sc["voiceConfig"] = map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": v}}
	}
	if v, ok := audio["language_code"]; ok {
		sc["languageCode"] = v
	}
	return sc, nil
}

func serviceTierToWire(v string) string {
	switch l := strings.ToLower(v); l {
	case "auto":
		return "priority"
	case "default":
		return "standard"
	default:
		return l
	}
}

// GeminiToChat folds a generateContent request body into a chat request body.
// The source names no model and no stream flag (both ride the URL, not the
// payload), so the fold carries neither. Fields the chat wire has no spelling
// for — safety settings, a cache reference, the speech and image configs — are
// dropped; the loss API names them for the host that wants to know.
func GeminiToChat(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	var msgs []any
	if sys, ok := jsonx.AsMap(body["system_instruction"]); ok {
		if parts, ok := jsonx.AsSlice(sys["parts"]); ok {
			for _, p := range parts {
				pm, ok := jsonx.AsMap(p)
				if !ok {
					continue
				}
				if t, ok := pm["text"].(string); ok {
					msgs = append(msgs, map[string]any{"role": "system", "content": t})
				}
			}
		}
	}
	contents, _ := jsonx.AsSlice(body["contents"])
	msgs = append(msgs, geminiContentsToChatMessages(contents)...)
	if msgs == nil {
		msgs = []any{}
	}
	out["messages"] = msgs
	gc, _ := jsonx.AsMap(body["generationConfig"])
	for _, k := range []string{"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "seed"} {
		if v, ok := gc[k]; ok && v != nil {
			out[k] = v
		}
	}
	if v, ok := gc["candidate_count"]; ok && v != nil {
		out["n"] = v
	}
	if v, ok := gc["max_output_tokens"]; ok && v != nil {
		out["max_tokens"] = v
	}
	if v, ok := gc["stop_sequences"]; ok && v != nil {
		out["stop"] = v
	}
	if v, ok := gc["responseLogprobs"]; ok && v != nil {
		out["logprobs"] = v
	}
	if v, ok := gc["logprobs"]; ok && v != nil {
		out["top_logprobs"] = v
	}
	if list, ok := jsonx.AsSlice(gc["responseModalities"]); ok {
		mods := []any{}
		for _, m := range list {
			switch jsonx.String(m) {
			case "TEXT":
				mods = append(mods, "text")
			case "IMAGE":
				mods = append(mods, "image")
			case "AUDIO":
				mods = append(mods, "audio")
			case "VIDEO":
				mods = append(mods, "video")
			}
		}
		out["modalities"] = mods
	}
	if tc, ok := jsonx.AsMap(gc["thinkingConfig"]); ok {
		if e, ok := thinkingConfigToEffort(tc); ok {
			out["reasoning_effort"] = e
		}
	}
	if rf := wireResponseFormat(gc); rf != nil {
		out["response_format"] = rf
	}
	toolsList := []any{}
	hasTools := false
	hadSearch := false
	if list, ok := jsonx.AsSlice(body["tools"]); ok {
		for _, t := range list {
			tm, ok := jsonx.AsMap(t)
			if !ok {
				continue
			}
			if decls, ok := jsonx.AsSlice(tm["function_declarations"]); ok {
				// The gemini wire spells its type enums in SCREAMING_SNAKE; the
				// chat wire validates the lowercase JSON Schema vocabulary and
				// refuses the uppercase spelling, so the relayed schema is
				// normalized in place first.
				schemanorm.NormalizeTree(decls)
				for _, d := range decls {
					dm, ok := jsonx.AsMap(d)
					if !ok {
						continue
					}
					fn := map[string]any{"name": jsonx.GetString(dm, "name")}
					if v, ok := dm["description"]; ok && v != nil {
						fn["description"] = v
					}
					if v, ok := schemanorm.SchemaOf(dm); ok {
						fn["parameters"] = v
					}
					toolsList = append(toolsList, map[string]any{"type": "function", "function": fn})
					hasTools = true
				}
			}
			if _, ok := tm["googleSearch"]; ok {
				hadSearch = true
			}
		}
	}
	if hasTools {
		out["tools"] = toolsList
	}
	if hadSearch {
		out["web_search_options"] = map[string]any{}
	}
	if tcfg, ok := jsonx.AsMap(body["toolConfig"]); ok {
		if fcc, ok := jsonx.AsMap(tcfg["functionCallingConfig"]); ok {
			var nonEmpty []string
			if names, ok := jsonx.AsSlice(fcc["allowed_function_names"]); ok {
				for _, n := range names {
					if s := jsonx.String(n); s != "" {
						nonEmpty = append(nonEmpty, s)
					}
				}
			}
			mode := jsonx.GetString(fcc, "mode")
			switch {
			case len(nonEmpty) == 1:
				out["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": nonEmpty[0]}}
			case mode == "NONE":
				out["tool_choice"] = "none"
			case mode == "AUTO":
				out["tool_choice"] = "auto"
			case mode == "ANY":
				out["tool_choice"] = "required"
			}
		}
	}
	if st := jsonx.GetString(body, "serviceTier"); st != "" {
		out["service_tier"] = serviceTierFromWire(st)
	}
	return out, nil
}

func serviceTierFromWire(v string) string {
	switch v {
	case "priority":
		return "auto"
	case "standard":
		return "default"
	default:
		return v
	}
}

// thinkingConfigToEffort reads a thinkingConfig back into the effort label a
// chat request asks with. The reading itself lives in internal/thinkingcfg,
// which is the one place that knows both spellings a gemini wire states depth
// in — see it for why the answer is false when the config names no depth.
func thinkingConfigToEffort(tc map[string]any) (string, bool) {
	return thinkingcfg.Effort(tc)
}

// wireResponseFormat reads the structured-output config back into a chat
// response_format. A json mime with no schema can only have come from a bare
// json_object request, so it folds back to that.
func wireResponseFormat(gc map[string]any) map[string]any {
	switch jsonx.GetString(gc, "response_mime_type") {
	case "application/json":
		schema := gc["response_json_schema"]
		if schema == nil {
			schema = gc["response_schema"]
		}
		if sm, ok := schema.(map[string]any); ok && schema != nil {
			return map[string]any{"type": "json_schema", "json_schema": map[string]any{
				"name": "response_schema", "schema": sm, "strict": false,
			}}
		}
		return map[string]any{"type": "json_object"}
	case "text/plain":
		return map[string]any{"type": "text"}
	}
	return nil
}

// geminiContentsToChatMessages folds contents into chat messages: one message
// per content, its parts carried in order. A model content's thought parts
// become the assistant's reasoning (and, when signed, its thinking blocks); a
// user content's thought parts are plain text there, the way the forward fold
// reads them. Media rides an assistant's images field (where the forward fold
// reads it back) and a user's content parts. A function response becomes its
// own tool message, paired to the call it answers.
func geminiContentsToChatMessages(contents []any) []any {
	var msgs []any
	callSeq := 0
	var calls []callRef
	for _, c := range contents {
		cm, ok := jsonx.AsMap(c)
		if !ok {
			continue
		}
		isModel := jsonx.GetString(cm, "role") == "model"
		role := "user"
		if isModel {
			role = "assistant"
		}
		parts, _ := jsonx.AsSlice(cm["parts"])
		var contentParts []any
		textCount := 0
		var mediaParts []string
		var thoughtTexts []string
		var thinking []any
		var sigs []string
		var toolCalls []any
		var toolMsgs []any
		for _, p := range parts {
			pm, ok := jsonx.AsMap(p)
			if !ok {
				continue
			}
			sig := partThoughtSignature(pm)
			if sig != "" {
				sigs = append(sigs, sig)
			}
			if fc, ok := jsonx.AsMap(pm["function_call"]); ok {
				id := jsonx.GetString(fc, "id")
				if id == "" {
					id = fmt.Sprintf("call_%d", callSeq)
					callSeq++
				}
				args := fc["args"]
				if args == nil {
					args = map[string]any{}
				}
				call := map[string]any{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      jsonx.GetString(fc, "name"),
						"arguments": mustArgs(args),
					},
				}
				if sig != "" {
					call["provider_specific_fields"] = map[string]any{"thought_signature": sig}
					call["id"] = id + thoughtSignatureSeparator + sig
				}
				toolCalls = append(toolCalls, call)
				// The result that answers this call pairs against the id the
				// chat wire carries, signature suffix included.
				calls = append(calls, callRef{name: jsonx.GetString(fc, "name"), id: jsonx.GetString(call, "id")})
				continue
			}
			if fr, ok := jsonx.AsMap(pm["function_response"]); ok {
				id := jsonx.GetString(fr, "id")
				if id == "" {
					name := jsonx.GetString(fr, "name")
					for i := len(calls) - 1; i >= 0; i-- {
						if calls[i].name == name {
							id = calls[i].id
							break
						}
					}
				}
				// A response that names an id the client was handed must pair
				// with the id the wire actually carries for that call, which
				// gains the signature suffix above. See signedCallID.
				id = signedCallID(calls, id)
				if id == "" {
					id = fmt.Sprintf("call_%d", callSeq)
					callSeq++
				}
				toolMsgs = append(toolMsgs, map[string]any{
					"role":         "tool",
					"tool_call_id": id,
					"content":      functionResponseContent(fr["response"]),
				})
				continue
			}
			if inline, ok := jsonx.AsMap(pm["inline_data"]); ok {
				mime := firstNonEmpty(jsonx.GetString(inline, "mime_type"), jsonx.GetString(inline, "mimeType"))
				url := "data:" + mime + ";base64," + jsonx.GetString(inline, "data")
				if isModel {
					mediaParts = append(mediaParts, url)
				} else {
					contentParts = append(contentParts, imageURLPart(url))
				}
				continue
			}
			if fd, ok := jsonx.AsMap(pm["file_data"]); ok {
				url := jsonx.GetString(fd, "file_uri")
				if isModel {
					mediaParts = append(mediaParts, url)
				} else {
					contentParts = append(contentParts, imageURLPart(url))
				}
				continue
			}
			if raw, ok := pm["text"]; ok {
				s := jsonx.String(raw)
				if b, isThought := pm["thought"].(bool); isThought && b && isModel {
					thoughtTexts = append(thoughtTexts, s)
					block := map[string]any{"type": "thinking", "thinking": s}
					if sig != "" {
						block["signature"] = sig
					}
					thinking = append(thinking, block)
					continue
				}
				textCount++
				contentParts = append(contentParts, map[string]any{"type": "text", "text": s})
			}
		}
		msg := map[string]any{"role": role}
		if !isModel {
			// A content that was nothing but function responses needs no
			// wrapper: the tool messages carry the whole turn, and an empty
			// user message is a shape the chat upstream rejects.
			if len(toolMsgs) > 0 && len(contentParts) == 0 && len(toolCalls) == 0 {
				msgs = append(msgs, toolMsgs...)
				continue
			}
			switch {
			case len(contentParts) == 1 && textCount == 1:
				msg["content"] = contentParts[0].(map[string]any)["text"]
			case len(contentParts) == 0:
				msg["content"] = nil
			default:
				msg["content"] = contentParts
			}
			if len(toolCalls) > 0 {
				msg["tool_calls"] = toolCalls
			}
			msgs = append(msgs, msg)
			msgs = append(msgs, toolMsgs...)
			continue
		}
		switch {
		case len(contentParts) == 1 && textCount == 1:
			msg["content"] = contentParts[0].(map[string]any)["text"]
		case len(contentParts) == 0:
			msg["content"] = nil
		default:
			msg["content"] = contentParts
		}
		if len(mediaParts) > 0 {
			// The forward fold reads an assistant's media from the images
			// field, so folding it into content would drop it on replay.
			imgs := make([]any, 0, len(mediaParts))
			for _, u := range mediaParts {
				imgs = append(imgs, map[string]any{
					"image_url": map[string]any{"url": u, "detail": "auto"},
					"index":     0,
					"type":      "image_url",
				})
			}
			msg["images"] = imgs
		}
		if len(thoughtTexts) > 0 {
			msg["reasoning_content"] = strings.Join(thoughtTexts, "")
		}
		if len(thinking) > 0 {
			msg["thinking_blocks"] = thinking
		}
		if len(toolCalls) > 0 {
			msg["tool_calls"] = toolCalls
		}
		if len(sigs) > 0 {
			msg["provider_specific_fields"] = map[string]any{"thought_signatures": sigs}
		}
		msgs = append(msgs, msg)
		msgs = append(msgs, toolMsgs...)
	}
	return msgs
}

// functionResponseContent renders a function response for the chat wire: a
// response that is nothing but a wrapped content string folds back to the
// string itself, and anything else rides as its compact JSON.
func functionResponseContent(response any) any {
	if m, ok := jsonx.AsMap(response); ok {
		if len(m) == 1 {
			if v, has := m["content"]; has {
				if s, isStr := v.(string); isStr {
					return s
				}
			}
		}
	}
	if response == nil {
		return ""
	}
	b, err := jsonx.Marshal(response)
	if err != nil {
		return ""
	}
	return string(b)
}

// mustArgs renders decoded arguments as the JSON string the chat wire carries
// them in. The reference renders the args object back to text with the default
// separators of its own encoder — a space after every comma and colon — so the
// fold spells them the same way. Map keys come out sorted: the decode the fold
// starts from keeps no insertion order to restore.
func mustArgs(v any) string {
	var b strings.Builder
	writeDumpJSON(&b, v)
	if b.Len() == 0 {
		return "{}"
	}
	return b.String()
}

// writeDumpJSON renders one decoded-JSON value with the spacing described
// above, escaping nothing an ASCII payload does not need.
func writeDumpJSON(b *strings.Builder, v any) {
	switch t := v.(type) {
	case map[string]any:
		if len(t) == 0 {
			b.WriteString("{}")
			return
		}
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writeDumpJSONString(b, k)
			b.WriteString(": ")
			writeDumpJSON(b, t[k])
		}
		b.WriteByte('}')
	case []any:
		if len(t) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		for i, item := range t {
			if i > 0 {
				b.WriteString(", ")
			}
			writeDumpJSON(b, item)
		}
		b.WriteByte(']')
	case json.Number:
		if t.String() == "" {
			b.WriteString("0")
			return
		}
		b.WriteString(t.String())
	case string:
		writeDumpJSONString(b, t)
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	default:
		// A value the decode never produces (e.g. a raw float): let the
		// standard encoder spell it rather than guess.
		raw, err := json.Marshal(v)
		if err != nil {
			b.WriteString("null")
			return
		}
		b.Write(raw)
	}
}

// writeDumpJSONString quotes one key or string value, leaving non-ASCII as the
// bytes it came in and escaping exactly the characters a JSON string must.
func writeDumpJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
				continue
			}
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// callRef remembers one folded call — the name a result pairs against and the
// id the chat wire carries for it.
type callRef struct{ name, id string }

// signedCallID maps the id a functionResponse names onto the id the chat wire
// carries for the call it answers, resolving the thought-signature suffix.
//
// A replayed call whose part carried a thoughtSignature rides the chat wire as
// "<id>__thought__<sig>" — that suffix is how a LATER fold recovers the
// signature when the client dropped provider_specific_fields. The
// functionResponse answering the call, however, names the bare "<id>": on the
// gemini wire the signature rides the call, never the result, so the bare id is
// the only one the caller was ever handed. Writing it verbatim leaves a tool
// message that answers no call, and the pair is lost:
//
//	assistant tool_calls[0].id = "call_2138938__thought__<sig>"
//	tool.tool_call_id          = "call_2138938"        ← answers nothing
//
// An exact id still wins, so a history carrying both "call_1" and
// "call_1__thought__x" keeps them apart; the suffix scan is the fallback, most
// recent call first, because that is the exchange being replayed.
func signedCallID(calls []callRef, id string) string {
	if id == "" {
		return id
	}
	for i := len(calls) - 1; i >= 0; i-- {
		if calls[i].id == id {
			return id
		}
	}
	prefix := id + thoughtSignatureSeparator
	for i := len(calls) - 1; i >= 0; i-- {
		if strings.HasPrefix(calls[i].id, prefix) {
			return calls[i].id
		}
	}
	return id
}

// imageURLPart is the chat content spelling of a media reference.
func imageURLPart(url string) map[string]any {
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}
