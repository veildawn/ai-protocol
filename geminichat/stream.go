// Package geminichat folds the chat dialect and the gemini dialect into each
// other — request bodies, response bodies and SSE streams. The mappings are the
// reference's own rules for this pair, applied directly: where the reference
// reaches the other dialects by composing through chat, this package speaks
// both sides natively and never builds an intermediate body.
package geminichat

import (
	"encoding/json"
	"io"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/stream"
)

type StreamOpts struct {
	Flush func()
	Model string
	// OnEvent, when set, sees every target-dialect event just before it is
	// written and returns the event to write instead. See protocol.StreamOpts.
	OnEvent func(stream.Event) (stream.Event, error)
}

// PipeChatToGemini folds a chat chunk stream into the target dialect's frames.
//
// The target wire has no sentinel and no separate usage frame: the one frame
// that says how the turn ended carries the usage with it. So a finish_reason
// whose counts have not arrived yet is held for the choices-less chunk that
// carries them (the shape a chat source that asked for usage counts sends),
// and the terminal frame goes out carrying both.
//
// The client is told the turn ended only when the source said so itself — a
// finish_reason frame, the [DONE] sentinel a relay writes, or the bare usage
// tail several Chat vendors close on instead of either. A source that simply
// stops has not said, and this wire has no reason-free close to offer in its
// place, so the fold leaves the stream open rather than dress a cut delivery
// as a clean stop.
func PipeChatToGemini(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &chatToGeminiState{}
	var n int64
	write := func(evs []stream.Event) error {
		for _, out := range evs {
			nn, err := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if err != nil {
				return err
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
			return nil // skip comments / non-json
		}
		return write(st.feed(obj))
	})
	if err != nil {
		return n, err
	}
	// A finish the source stated but whose counts never came still closes the
	// turn — the counts it would have carried are simply absent.
	for _, out := range st.finish() {
		nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
		n += int64(nn)
		if werr != nil {
			return n, werr
		}
	}
	return n, nil
}

// chatToolSlot accumulates one call's argument fragments. The wire a call
// rides has no fragment spelling — its args member is a whole JSON object — so
// a fragment that does not parse yet rides nothing and the completed object
// goes out as one part.
type chatToolSlot struct {
	name    string
	args    string
	emitted string
}

func (s *chatToolSlot) part() (map[string]any, bool) {
	if s.args == s.emitted {
		return nil, false
	}
	parsed, err := parseJSONValue(s.args)
	if err != nil {
		return nil, false
	}
	if _, isObject := parsed.(map[string]any); !isObject {
		return nil, false
	}
	s.emitted = s.args
	// A streamed frame is a response body, whose part vocabulary is the
	// camel-case spelling; the snake-case one belongs to the request wire.
	return map[string]any{"functionCall": map[string]any{"name": s.name, "args": parsed}}, true
}

type chatToGeminiState struct {
	closed        bool
	pendingFinish string
	// usageClosed records that the source ended on a bare usage tail — the
	// close several Chat vendors send instead of a finish_reason or [DONE].
	// The counts are the turn's own settlement, so the fold closes for the
	// client at the end of the source, the same ruling the other chat folds
	// apply to this ending.
	usageClosed bool
	usage       any
	slots       []*chatToolSlot
	slotByIndex map[int]*chatToolSlot
}

func (s *chatToGeminiState) slotFor(tm map[string]any) *chatToolSlot {
	idx := 0
	if v, ok := jsonx.Int(tm["index"]); ok {
		idx = v
	}
	if s.slotByIndex == nil {
		s.slotByIndex = map[int]*chatToolSlot{}
	}
	if slot, ok := s.slotByIndex[idx]; ok {
		return slot
	}
	slot := &chatToolSlot{}
	s.slotByIndex[idx] = slot
	s.slots = append(s.slots, slot)
	return slot
}

func (s *chatToGeminiState) feed(obj map[string]any) []stream.Event {
	if _, failed := obj["error"]; failed {
		// The upstream declared the turn failed. Emit nothing: the frame the
		// client has to see is the host's error frame for the failed delivery,
		// and a synthesized terminal here would dress that failure as a
		// finished answer.
		s.closed = true
		return nil
	}
	choices, _ := jsonx.AsSlice(obj["choices"])
	if len(choices) == 0 {
		// A choices-less chunk is the usage tail; it produces no frame of its
		// own, but it completes a held finish — and, when no finish ever
		// comes, it is itself the source's close.
		if obj["usage"] != nil {
			s.usage = obj["usage"]
			s.usageClosed = true
		}
		if s.pendingFinish != "" {
			return s.terminal(s.pendingFinish)
		}
		return nil
	}
	ch, _ := jsonx.AsMap(choices[0])
	if ch == nil {
		return nil
	}
	var parts []any
	if delta, ok := jsonx.AsMap(ch["delta"]); ok {
		if tbs, ok := jsonx.AsSlice(delta["thinking_blocks"]); ok {
			for _, b := range tbs {
				bm, isMap := jsonx.AsMap(b)
				if !isMap || jsonx.GetString(bm, "type") != "thinking" {
					continue
				}
				text, hasText := bm["thinking"]
				sig, hasSig := bm["signature"]
				if !hasText || text == nil || !hasSig || sig == nil {
					// An unsigned block is the same fact as the reasoning text
					// the delta also carries; sending both would replay the
					// reasoning twice.
					continue
				}
				// A signed block rides a flagged thought part here — a
				// response body's only consumer is the reader, which folds
				// unflagged text into content (see replayedThoughtPart).
				parts = append(parts, replayedThoughtPart(jsonx.String(text), jsonx.String(sig)))
			}
		}
		if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
			parts = append(parts, map[string]any{"text": rc, "thought": true})
		}
		if c, ok := delta["content"].(string); ok && c != "" {
			parts = append(parts, map[string]any{"text": c})
		}
		if tcs, ok := jsonx.AsSlice(delta["tool_calls"]); ok {
			for _, tc := range tcs {
				tm, isMap := jsonx.AsMap(tc)
				if !isMap {
					continue
				}
				slot := s.slotFor(tm)
				if fn, ok := jsonx.AsMap(tm["function"]); ok {
					if name := jsonx.GetString(fn, "name"); name != "" {
						slot.name = name
					}
					slot.args += jsonx.GetString(fn, "arguments")
				}
				if part, ok := slot.part(); ok {
					parts = append(parts, part)
				}
			}
		}
	}
	var evs []stream.Event
	if len(parts) > 0 {
		evs = append(evs, geminiContentFrame(parts))
	}
	if obj["usage"] != nil {
		s.usage = obj["usage"]
	}
	if fr := jsonx.GetString(ch, "finish_reason"); fr != "" && !s.closed {
		if obj["usage"] == nil {
			// The counts a chat source reports after the finish ride the
			// terminal frame here, so the frame waits for them rather than
			// report zeros for a turn that has a real count coming.
			s.pendingFinish = fr
			return evs
		}
		evs = append(evs, s.terminal(fr)...)
	}
	return evs
}

// terminal emits the frame that closes the turn: any call whose arguments only
// completed with the stream's last bytes goes out whole ahead of it, then the
// finish reason and the usage together on one frame.
func (s *chatToGeminiState) terminal(finish string) []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	var parts []any
	for _, slot := range s.slots {
		if part, ok := slot.part(); ok {
			parts = append(parts, part)
		}
	}
	if len(parts) > 0 {
		evs = append(evs, geminiContentFrame(parts))
	}
	return append(evs, stream.Event{Data: mustJSON(map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": []any{}},
			"finishReason": finishReasonFromChat(finish),
			"index":        0,
		}},
		"usageMetadata": chatUsageToMetadata(s.usage),
	})})
}

// closeFromSource closes the turn for a source that ended with the [DONE]
// sentinel — a close the source itself stated, so the fold names the finish it
// held, or the natural stop of a turn that never said.
func (s *chatToGeminiState) closeFromSource() []stream.Event {
	if s.closed {
		return nil
	}
	finish := s.pendingFinish
	if finish == "" {
		finish = "stop"
	}
	return s.terminal(finish)
}

// finish closes out a source that ended without a sentinel. A held finish
// closes with its own reason — the counts it would have carried are simply
// absent. A bare usage tail is the source's own close too (several Chat
// vendors end the turn there instead of [DONE]): the fold closes for the
// client with the wire's natural stop, exactly as the other chat folds close
// this ending for their clients. A source that said none of those things was
// cut, and the fold invents nothing for it.
func (s *chatToGeminiState) finish() []stream.Event {
	if s.closed {
		return nil
	}
	if s.pendingFinish != "" {
		return s.terminal(s.pendingFinish)
	}
	if s.usageClosed {
		return s.terminal("stop")
	}
	return nil
}

func geminiContentFrame(parts []any) stream.Event {
	return stream.Event{Data: mustJSON(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": parts},
			"index":   0,
		}},
	})}
}

// PipeGeminiToChat folds the target dialect's SSE frames into chat chunks: one
// chunk per frame, the way the reference's chunk parser yields them.
//
// The client is told the turn ended only when the source's own close arrived —
// a frame whose candidates carry a finish reason (or a prompt the safety layer
// refused). A source that ends with neither was cut, and the fold leaves the
// chat stream open rather than synthesize an ending for it: the host records
// the cut, and the client must not read a finished answer out of it.
func PipeGeminiToChat(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &geminiToChatState{id: "chatcmpl-unknown", model: opts.Model}
	var n int64
	write := func(evs []stream.Event) error {
		for _, out := range evs {
			nn, err := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if err != nil {
				return err
			}
		}
		return nil
	}
	err := stream.Scan(src, func(ev stream.Event) error {
		if stream.IsDone(ev.Data) {
			// This wire has no sentinel of its own; one arriving means a relay
			// closed the source. The close is relayed to the chat client as
			// the sentinel alone — no finish frame is invented for a source
			// that never said how the turn ended.
			return write(st.relayDone())
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		return write(st.feed(obj))
	})
	if err != nil {
		return n, err
	}
	// A source that stops without a finish reason was cut; nothing is closed
	// for the client here.
	return n, nil
}

type geminiToChatState struct {
	id               string
	model            string
	closed           bool
	hasSeenToolCalls bool
	toolIndex        int
	callSeq          int
}

func (s *geminiToChatState) relayDone() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	return []stream.Event{{Data: "[DONE]"}}
}

func (s *geminiToChatState) feed(obj map[string]any) []stream.Event {
	if _, failed := obj["error"]; failed {
		// An in-band failure frame ends the stream without an answer; the
		// frame the client has to see is the host's error for the failed
		// delivery, not a synthesized finish.
		s.closed = true
		return nil
	}
	if s.closed {
		return nil
	}
	if id := jsonx.GetString(obj, "responseId"); id != "" {
		s.id = id
	}
	candidates, _ := jsonx.AsSlice(obj["candidates"])
	// A usage-only frame reports its counts with an empty choice list, the
	// shape the reference's parser yields for it.
	choices := []any{}
	blocked := false
	if pf, ok := jsonx.AsMap(obj["promptFeedback"]); ok {
		if _, has := pf["blockReason"]; has {
			// A refused prompt ends the turn before any content: one filtered
			// choice, the role the client needs to see, nothing else.
			blocked = true
			choices = append(choices, map[string]any{
				"finish_reason": "content_filter",
				"index":         0,
				"delta":         map[string]any{"content": nil, "role": "assistant"},
			})
		}
	}
	sawFinish := ""
	for idx, raw := range candidates {
		cm, ok := jsonx.AsMap(raw)
		if !ok {
			continue
		}
		content, hasContent := jsonx.AsMap(cm["content"])
		if !hasContent {
			continue
		}
		var f geminiPartsFold
		parts, _ := jsonx.AsSlice(content["parts"])
		for _, p := range parts {
			if pm, ok := jsonx.AsMap(p); ok {
				f.feed(pm, &s.toolIndex, &s.callSeq)
			}
		}
		if len(f.toolCalls) > 0 {
			s.hasSeenToolCalls = true
		}
		choice := map[string]any{
			"index":         candidateIndex(cm, idx),
			"delta":         f.applyDelta(),
			"finish_reason": nil,
		}
		// Only the frame that carries the wire's own close names a finish
		// reason: a chunk that merely carries content or a call leaves the
		// turn open, whatever the reference's raw parser would stamp on it.
		if fr := jsonx.GetString(cm, "finishReason"); fr != "" {
			sawFinish = fr
			choice["finish_reason"] = f.finishReason(fr)
		}
		choices = append(choices, choice)
	}
	if len(choices) == 0 && len(candidates) > 0 && !blocked {
		// A candidate that carried nothing but its ending still closes the
		// turn: an empty-delta choice keeps the finish from being lost with
		// the frame that carried it.
		for idx, raw := range candidates {
			cm, ok := jsonx.AsMap(raw)
			if !ok {
				continue
			}
			var finish any
			if fr := jsonx.GetString(cm, "finishReason"); fr != "" {
				sawFinish = fr
				finish = finishReasonToChat[fr]
				if finish == "" {
					finish = "stop"
				}
			}
			choices = append(choices, map[string]any{
				"index":         candidateIndex(cm, idx),
				"delta":         map[string]any{},
				"finish_reason": finish,
			})
		}
	}
	if s.hasSeenToolCalls {
		// A stream that made a call closes as a tool-call turn: a plain stop
		// would tell the client the model merely finished speaking.
		for _, raw := range choices {
			if cm, ok := jsonx.AsMap(raw); ok && cm["finish_reason"] == "stop" {
				cm["finish_reason"] = "tool_calls"
			}
		}
	}
	var usage map[string]any
	if obj["usageMetadata"] != nil {
		usage = geminiUsageToChat(obj)
	}
	if len(choices) == 0 && usage == nil && !blocked {
		// A frame with nothing to deliver produces no chunk; the reference's
		// own wrapper drops such chunks rather than feed a client empty frames.
		return nil
	}
	terminal := sawFinish != "" || blocked
	var evs []stream.Event
	evs = append(evs, s.chunk(choices, usage))
	if terminal {
		s.closed = true
		evs = append(evs, stream.Event{Data: "[DONE]"})
	}
	return evs
}

func candidateIndex(cm map[string]any, position int) any {
	if v, ok := jsonx.Int(cm["index"]); ok {
		return v
	}
	return position
}

func (s *geminiToChatState) chunk(choices []any, usage map[string]any) stream.Event {
	full := map[string]any{
		"id":      s.id,
		"object":  "chat.completion.chunk",
		"created": 0,
		"model":   s.model,
		"choices": choices,
	}
	if usage != nil {
		full["usage"] = usage
	}
	return stream.Event{Data: mustJSON(full)}
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
