// Package geminimessages converts directly between the Messages dialect and
// the gemini dialect's generateContent line format, on the request, response
// and SSE stream faces.
//
// The two dialects have no direct bridge in the reference: it composes a
// Messages fold onto the chat shape and folds that onto the target. This
// package translates in one pass — no chat body is constructed and no sibling
// package is imported — while keeping the composed semantics, so a converted
// body matches what the reference's two-step fold would have produced. Where
// the reference drops or degrades content it cannot map (model-made images on
// a response, a signature on a streamed thought), this package documents the
// same behavior on the same code path rather than inventing a target spelling.
package geminimessages

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/stream"
)

// StreamOpts controls SSE conversion.
type StreamOpts struct {
	Flush func()
	Model string
	// OnEvent, when set, sees every target-dialect event just before it is
	// written and returns the event to write instead. See protocol.StreamOpts.
	OnEvent func(stream.Event) (stream.Event, error)
}

// PipeMessagesToGemini folds an Anthropic Messages SSE stream into the
// target's SSE line format: one data frame per emitted part, no event names
// and no sentinel — that wire has none.
//
// A tool call's arguments cannot ride the wire as fragments, because the
// target reads each function call as one JSON object. Fragments accumulate
// and the call is emitted once, on the last parseable snapshot before the
// block closes; a stream whose arguments never become an object emits no call
// rather than half of one.
//
// The terminal frame carries the finish reason and the usage, and is written
// only when the SOURCE closed the turn (message_stop, or the [DONE] sentinel
// a relay dialect appends). A stream that ends with neither was cut — the
// target dialect has no other legal ending — and inventing a finish would
// tell the client it holds a finished answer while the host records a failed
// delivery for the same turn.
func PipeMessagesToGemini(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &messagesToGeminiState{}
	var n int64
	write := func(evs []stream.Event) error {
		for _, out := range evs {
			nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if werr != nil {
				return werr
			}
		}
		return nil
	}
	err := stream.Scan(src, func(ev stream.Event) error {
		if stream.IsDone(ev.Data) {
			// [DONE] is the one close a relay dialect writes instead of the
			// typed terminal, so it closes the turn here. An EOF with no close
			// at all deliberately does not — see the doc comment.
			return write(st.closeFromSource())
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		return write(st.feed(ev.Event, obj))
	})
	if err != nil {
		return n, err
	}
	// EOF with no close of any kind is a truncated turn: nothing further is
	// written for the client.
	return n, nil
}

// messagesToGeminiState folds Messages events into target frames.
type messagesToGeminiState struct {
	usage  map[string]any
	stop   string
	closed bool
	// tool is the open tool_use block: its name from the block start and the
	// arguments accumulated from the json deltas.
	tool *openToolCall
	// thinking accumulates the open thinking block's text so a signature
	// delta can carry the block whole, the shape the reference emits when a
	// chat delta hands over a thinking block.
	thinking strings.Builder
}

type openToolCall struct {
	name string
	args strings.Builder
}

func (s *messagesToGeminiState) feed(event string, obj map[string]any) []stream.Event {
	if s.closed {
		return nil
	}
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	switch typ {
	case "message_start":
		if msg, ok := jsonx.AsMap(obj["message"]); ok {
			if u, ok := jsonx.AsMap(msg["usage"]); ok {
				s.usage = copyUsage(s.usage, u)
			}
		}
	case "content_block_start":
		cb, _ := jsonx.AsMap(obj["content_block"])
		switch jsonx.GetString(cb, "type") {
		case "tool_use", "server_tool_use":
			s.tool = &openToolCall{name: jsonx.GetString(cb, "name")}
		case "thinking":
			s.thinking.Reset()
		}
	case "content_block_delta":
		delta, _ := jsonx.AsMap(obj["delta"])
		if delta == nil {
			break
		}
		switch jsonx.GetString(delta, "type") {
		case "text_delta":
			if text := jsonx.GetString(delta, "text"); text != "" {
				return s.partFrame(map[string]any{"text": text})
			}
		case "thinking_delta":
			thinking := jsonx.GetString(delta, "thinking")
			s.thinking.WriteString(thinking)
			if thinking != "" {
				return s.partFrame(map[string]any{"text": thinking, "thought": true})
			}
		case "signature_delta":
			// The thinking text already went out with its own delta; the
			// signature rides a bare part — the reader collects a signature
			// from any part, and a second copy of the text would double the
			// reasoning the reader reconstructs (verified against the
			// reference's reader during this audit).
			if sig := jsonx.GetString(delta, "signature"); sig != "" {
				return s.partFrame(map[string]any{"thoughtSignature": sig})
			}
		case "input_json_delta":
			if s.tool != nil {
				s.tool.args.WriteString(jsonx.GetString(delta, "partial_json"))
			}
		}
	case "content_block_stop":
		if s.tool != nil {
			if frame, ok := s.toolFrame(); ok {
				s.tool = nil
				return frame
			}
			s.tool = nil
		}
	case "message_delta":
		if d, ok := jsonx.AsMap(obj["delta"]); ok {
			if sr := jsonx.GetString(d, "stop_reason"); sr != "" {
				s.stop = sr
			}
		}
		if u, ok := jsonx.AsMap(obj["usage"]); ok {
			s.usage = copyUsage(s.usage, u)
		}
	case "message_stop":
		return s.terminal()
	case "error":
		// The upstream declared the turn failed. Emit nothing: the frame the
		// client has to see is the host's error frame for the failed delivery,
		// and a synthesized terminal here would dress that failure as a
		// finished answer.
		s.closed = true
	}
	return nil
}

func (s *messagesToGeminiState) partFrame(part map[string]any) []stream.Event {
	return []stream.Event{{Data: mustJSON(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{part}},
			"index":   0,
		}},
	})}}
}

// toolFrame emits the open tool call once, with the arguments as one JSON
// object. Arguments that never became an object emit no call at all — the
// wire cannot carry half of one — and an empty accumulation is the empty
// object, the same value the request-side fold writes for a call that stated
// no arguments.
func (s *messagesToGeminiState) toolFrame() ([]stream.Event, bool) {
	if s.tool == nil {
		return nil, false
	}
	args := s.tool.args.String()
	var parsed any
	if args == "" {
		parsed = map[string]any{}
	} else {
		dec := json.NewDecoder(strings.NewReader(args))
		dec.UseNumber()
		if err := dec.Decode(&parsed); err != nil {
			return nil, false
		}
		if _, isObject := parsed.(map[string]any); !isObject {
			// The target reads args as a JSON object; anything else cannot
			// ride the wire.
			return nil, false
		}
	}
	return s.partFrame(map[string]any{"function_call": map[string]any{
		"name": s.tool.name,
		"args": parsed,
	}}), true
}

// terminal writes the frame that closes the turn: the finish reason mapped
// back from the stop reason, and the usage on the same frame — the only frame
// the target's usage ever rides.
func (s *messagesToGeminiState) terminal() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	if s.tool != nil {
		if frame, ok := s.toolFrame(); ok {
			evs = append(evs, frame...)
		}
		s.tool = nil
	}
	candidate := map[string]any{
		"content":      map[string]any{"role": "model", "parts": []any{}},
		"finishReason": messagesStopToFinishReason(s.stop),
		"index":        0,
	}
	body := map[string]any{"candidates": []any{candidate}}
	if s.usage != nil {
		body["usageMetadata"] = messagesUsageToGemini(s.usage)
	}
	return append(evs, stream.Event{Data: mustJSON(body)})
}

// closeFromSource closes the turn for a source that ended with the [DONE]
// sentinel rather than message_stop. Idempotent, and reached only from the
// source's own close — never from EOF, where the missing terminal IS the
// upstream's failure.
func (s *messagesToGeminiState) closeFromSource() []stream.Event {
	if s.closed {
		return nil
	}
	return s.terminal()
}

func copyUsage(base, delta map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range base {
		out[k] = v
	}
	for k, v := range delta {
		out[k] = v
	}
	return out
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// PipeGeminiToMessages folds the target's SSE line format — one data frame per
// response chunk, no event names, no sentinel — into an Anthropic Messages
// SSE stream.
//
// The Messages client is told the turn finished only when the SOURCE said so:
// a frame carrying a finishReason (or a blocked-prompt frame, which the
// reference folds into a filtered close). A stream that ends with neither was
// cut, and synthesizing message_stop would tell the client it holds a
// finished answer while the host records a failed delivery for the same turn.
// The [DONE] sentinel a relay dialect appends does close the turn, and is
// closed without a stop reason — nothing named one.
func PipeGeminiToMessages(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &geminiToMessagesState{model: opts.Model}
	var n int64
	write := func(evs []stream.Event) error {
		for _, out := range evs {
			nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if werr != nil {
				return werr
			}
		}
		return nil
	}
	err := stream.Scan(src, func(ev stream.Event) error {
		if stream.IsDone(ev.Data) {
			return write(st.closeFromSource())
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		evs := st.feed(obj)
		return write(evs)
	})
	if err != nil {
		return n, err
	}
	// A finishReason already seen is the source's own close: the held terminal
	// goes out even though the stream ended without a sentinel. No finish and
	// no sentinel is a truncated turn and writes nothing further.
	if err := write(st.flushHeldTerminal()); err != nil {
		return n, err
	}
	return n, nil
}

// geminiToMessagesState folds target frames into Messages events.
type geminiToMessagesState struct {
	started   bool
	closed    bool
	id        string
	idMinted  bool
	model     string
	blockOpen bool
	blockType string // "" | "text" | "thinking" | "tool_use"
	blockIdx  int
	// sawToolCalls latches the stream's tool calls: the reference rewrites a
	// later clean stop into a tool close once a call has been seen, because
	// the calls and the reason arrive in separate frames.
	sawToolCalls bool
	callSeq      int
	// heldFinish is a finish the source named on a frame that carried no
	// usage; the terminal waits for the usage the next frame may bring, the
	// hold-and-merge the reference performs.
	heldFinish  string
	heldBlocked bool
	usage       map[string]any
}

func (s *geminiToMessagesState) feed(obj map[string]any) []stream.Event {
	if s.closed {
		return nil
	}
	var evs []stream.Event
	// The frame's own id is captured before the opening message names one, so
	// a stream that opens with its id keeps it (the reference mints a fresh
	// one; this codec keeps the identity the source stated).
	if id := jsonx.GetString(obj, "responseId"); id != "" && s.id == "" {
		s.id = id
	}
	evs = append(evs, s.openMessageStart()...)

	if blocked, ok := jsonx.AsMap(obj["promptFeedback"]); ok && jsonx.GetString(blocked, "blockReason") != "" {
		// A blocked prompt is the source's own close, filtered: the reference
		// folds it into a content-filter close, which the Messages dialect
		// spells end_turn.
		s.heldFinish = "end_turn"
		s.heldBlocked = true
		if u, ok := jsonx.AsMap(obj["usageMetadata"]); ok {
			evs = append(evs, s.terminalWith(geminiUsageToMessages(u, obj))...)
			return evs
		}
		return evs
	}

	candidates, _ := jsonx.AsSlice(obj["candidates"])
	if len(candidates) == 0 {
		if u, ok := jsonx.AsMap(obj["usageMetadata"]); ok {
			s.usage = geminiUsageToMessages(u, obj)
			if s.heldFinish != "" {
				evs = append(evs, s.terminalWith(s.usage)...)
			}
		}
		return evs
	}

	c, _ := jsonx.AsMap(candidates[0])
	finish := jsonx.GetString(c, "finishReason")
	content, _ := jsonx.AsMap(c["content"])
	parts, _ := jsonx.AsSlice(content["parts"])

	var text strings.Builder
	hasText := false
	var thoughts []string
	type streamedCall struct {
		block map[string]any
		args  any
	}
	var calls []streamedCall
	for _, raw := range parts {
		p, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		// The streamed response wire spells these camelCase; the request wire
		// is the snake_case one.
		if fc, ok := jsonx.AsMap(p["functionCall"]); ok {
			s.sawToolCalls = true
			args := any(map[string]any{})
			if a, ok := fc["args"]; ok && a != nil {
				args = a
			}
			calls = append(calls, streamedCall{
				block: s.toolCallBlock(fc, jsonx.GetString(p, "thoughtSignature")),
				args:  args,
			})
			continue
		}
		if inline, ok := jsonx.AsMap(p["inlineData"]); ok {
			mime := jsonx.GetString(inline, "mimeType")
			if strings.HasPrefix(mime, "image/") || strings.HasPrefix(mime, "audio/") {
				// Images and audio have no Messages spelling in a streamed
				// response — the composed fold delivers them nowhere.
				continue
			}
			text.WriteString("data:" + mime + ";base64," + jsonx.GetString(inline, "data"))
			hasText = true
			continue
		}
		if raw, present := p["text"]; present {
			body := jsonx.String(raw)
			if thought, _ := jsonx.Bool(p["thought"]); thought {
				if body != "" {
					thoughts = append(thoughts, body)
				}
				continue
			}
			if isAudioDataURL(body) {
				continue
			}
			if body != "" {
				text.WriteString(body)
				hasText = true
			}
		}
	}

	for _, th := range thoughts {
		evs = append(evs, s.ensureBlock("thinking", map[string]any{"type": "thinking", "thinking": "", "signature": ""})...)
		// A signature on a streamed thought rides the chat delta's provider
		// fields, which the composed fold never reads back — the block opens
		// unsigned and stays that way.
		evs = append(evs, s.delta(map[string]any{"type": "thinking_delta", "thinking": th}))
	}
	if hasText {
		evs = append(evs, s.ensureBlock("text", map[string]any{"type": "text", "text": ""})...)
		evs = append(evs, s.delta(map[string]any{"type": "text_delta", "text": text.String()}))
	}
	for _, call := range calls {
		// A tool block opens with an empty input; the arguments ride the
		// json delta that follows, the way the reference streams them.
		evs = append(evs, s.ensureBlock("tool_use", call.block)...)
		evs = append(evs, s.delta(map[string]any{
			"type": "input_json_delta", "partial_json": pythonCompactJSON(call.args),
		}))
	}

	if finish != "" {
		if finish == "STOP" && s.sawToolCalls {
			finish = "tool_calls"
		}
		s.heldFinish = geminiFinishToStopReason(finish, false)
		if u, ok := jsonx.AsMap(obj["usageMetadata"]); ok {
			evs = append(evs, s.terminalWith(geminiUsageToMessages(u, obj))...)
		}
	}
	return evs
}

// toolCallBlock builds a tool_use block start. The signature a part carries
// moves off the id (the target embeds it there for chat clients) onto the
// block's provider fields, keeping the turn replayable.
func (s *geminiToMessagesState) toolCallBlock(fc map[string]any, signature string) map[string]any {
	rawID := jsonx.GetString(fc, "id")
	sig := signature
	if rawID == "" {
		rawID = "call_" + itoa(s.callSeq)
	}
	s.callSeq++
	// The input stated on a block start is always empty: the arguments ride
	// the json delta that follows, never the opening frame.
	block := map[string]any{
		"type": "tool_use", "id": normalizeToolUseID(rawID),
		"name": jsonx.GetString(fc, "name"), "input": map[string]any{},
	}
	if sig != "" {
		block["provider_specific_fields"] = map[string]any{"signature": sig}
	}
	return block
}

func (s *geminiToMessagesState) openMessageStart() []stream.Event {
	if s.started {
		return nil
	}
	s.started = true
	id := s.id
	if id == "" {
		id = "msg_0"
	}
	msg := map[string]any{
		"id": id, "type": "message", "role": "assistant",
		"model": s.model, "content": []any{},
		"stop_reason": nil, "stop_sequence": nil,
		// The opening usage states the cache halves as zero: clients probe
		// these fields to decide whether the target supports prompt caching at
		// all. The real counts ride the closing message_delta.
		"usage": map[string]any{
			"input_tokens": 0, "output_tokens": 0,
			"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
		},
	}
	return []stream.Event{{Event: "message_start", Data: mustJSON(map[string]any{"type": "message_start", "message": msg})}}
}

// ensureBlock opens the block the current content needs, closing the one
// before it. Block indexes count up densely from zero, the way a Messages
// client accumulates them.
func (s *geminiToMessagesState) ensureBlock(typ string, block map[string]any) []stream.Event {
	if s.blockOpen && s.blockType == typ && typ != "tool_use" {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.closeBlock()...)
	if s.blockType != "" {
		s.blockIdx++
	}
	s.blockOpen = true
	s.blockType = typ
	return append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
		"type": "content_block_start", "index": s.blockIdx, "content_block": block,
	})})
}

func (s *geminiToMessagesState) closeBlock() []stream.Event {
	if !s.blockOpen {
		return nil
	}
	s.blockOpen = false
	return []stream.Event{{Event: "content_block_stop", Data: mustJSON(map[string]any{
		"type": "content_block_stop", "index": s.blockIdx,
	})}}
}

func (s *geminiToMessagesState) delta(d map[string]any) stream.Event {
	return stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
		"type": "content_block_delta", "index": s.blockIdx, "delta": d,
	})}
}

// terminalWith closes the turn: the open block stops, one message_delta names
// the stop reason and carries the usage, then message_stop. The Messages
// dialect allows exactly one message_delta per stream.
func (s *geminiToMessagesState) terminalWith(usage map[string]any) []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	stop := s.stopReasonForTerminal()
	s.heldFinish = ""
	var evs []stream.Event
	if !s.blockOpen && s.blockType == "" && !s.sawToolCalls {
		// A close on a stream that never opened a block still opens one: the
		// reference's wrapper opens an empty text block for a close-only
		// frame, and a Messages client reading content[index] expects it.
		evs = append(evs, s.ensureBlock("text", map[string]any{"type": "text", "text": ""})...)
	}
	evs = append(evs, s.closeBlock()...)
	if usage == nil {
		usage = map[string]any{"input_tokens": 0, "output_tokens": 0}
	}
	evs = append(evs, stream.Event{Event: "message_delta", Data: mustJSON(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop},
		"usage": usage,
	})})
	return append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
}

func (s *geminiToMessagesState) stopReasonForTerminal() string {
	if s.heldFinish != "" {
		return s.heldFinish
	}
	return "end_turn"
}

// flushHeldTerminal emits a terminal the source named but never delivered
// usage for. It is reached only when the source already closed itself with a
// finishReason; a truncated stream holds nothing and emits nothing.
func (s *geminiToMessagesState) flushHeldTerminal() []stream.Event {
	if s.closed || s.heldFinish == "" {
		return nil
	}
	return s.terminalWith(s.usage)
}

// closeFromSource closes the turn for a source that ended with the [DONE]
// sentinel. A finish the source already named closes with it; a close without
// one wraps the stream up without inventing a reason — the block closes and
// the message stops, and no stop_reason is put in the client's mouth.
func (s *geminiToMessagesState) closeFromSource() []stream.Event {
	if s.closed {
		return nil
	}
	if s.heldFinish != "" {
		return s.terminalWith(s.usage)
	}
	s.closed = true
	var evs []stream.Event
	evs = append(evs, s.closeBlock()...)
	return append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
}
