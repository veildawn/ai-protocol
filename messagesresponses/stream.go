package messagesresponses

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

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

func PipeMessagesToResponses(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &msgToResp{model: opts.Model}
	var n int64
	err := stream.Scan(src, func(ev stream.Event) error {
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		for _, out := range st.feed(ev.Event, obj) {
			nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if werr != nil {
				return werr
			}
		}
		return nil
	})
	if err != nil {
		return n, err
	}
	for _, out := range st.finish() {
		nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
		n += int64(nn)
		if werr != nil {
			return n, werr
		}
	}
	return n, nil
}

func PipeResponsesToMessages(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &respToMsg{model: opts.Model}
	var n int64
	err := stream.Scan(src, func(ev stream.Event) error {
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		for _, out := range st.feed(ev.Event, obj) {
			nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
			n += int64(nn)
			if werr != nil {
				return werr
			}
		}
		return nil
	})
	if err != nil {
		return n, err
	}
	for _, out := range st.finish() {
		nn, werr := stream.WriteEventWith(dst, out, opts.Flush, opts.OnEvent)
		n += int64(nn)
		if werr != nil {
			return n, werr
		}
	}
	return n, nil
}

// msgRespItem is one Responses output item backed by one Anthropic content
// block. text accumulates message text, reasoning summary, or call arguments
// depending on kind.
type msgRespItem struct {
	kind      string // "message" | "reasoning" | "function_call"
	id        string
	index     int
	text      strings.Builder
	callID    string
	name      string
	encrypted string
	noSummary bool // redacted_thinking: a reasoning item with no summary part
	closed    bool
}

type msgToResp struct {
	started bool
	closed  bool
	id      string
	model   string

	items  []*msgRespItem
	blocks map[int]*msgRespItem // content-block index → item
	next   int
	seq    int

	inTok     int
	cachedTok int
	outTok    int
	stop      string
}

func respEvent(typ string, fields map[string]any) stream.Event {
	fields["type"] = typ
	return stream.Event{Event: typ, Data: mustJSON(fields)}
}

func (s *msgToResp) responseShell() map[string]any {
	return map[string]any{"id": s.id, "object": "response", "model": s.model, "status": "in_progress"}
}

func (s *msgToResp) openBlock(idx int, cb map[string]any) []stream.Event {
	kind := ""
	var it *msgRespItem
	switch jsonx.GetString(cb, "type") {
	case "text":
		kind = "message"
	case "thinking":
		kind = "reasoning"
	case "redacted_thinking":
		kind = "reasoning"
	case "tool_use":
		kind = "function_call"
	default:
		return nil
	}
	it = &msgRespItem{kind: kind, index: s.next}
	s.next++
	switch kind {
	case "message":
		it.id = fmt.Sprintf("msg_%d", s.seq)
	case "reasoning":
		it.id = fmt.Sprintf("rs_%d", s.seq)
	default:
		it.id = fmt.Sprintf("fc_%d", s.seq)
	}
	s.seq++
	if kind == "function_call" {
		it.callID = jsonx.GetString(cb, "id")
		it.name = jsonx.GetString(cb, "name")
	}
	if kind == "reasoning" && jsonx.GetString(cb, "type") == "redacted_thinking" {
		it.encrypted = jsonx.GetString(cb, "data")
		it.noSummary = true
	}
	if s.blocks == nil {
		s.blocks = map[int]*msgRespItem{}
	}
	s.blocks[idx] = it
	s.items = append(s.items, it)
	return s.openEvents(it)
}

func (s *msgToResp) openEvents(it *msgRespItem) []stream.Event {
	var item map[string]any
	var part stream.Event
	switch it.kind {
	case "reasoning":
		item = map[string]any{"id": it.id, "type": "reasoning", "summary": []any{}}
		if it.encrypted != "" {
			item["encrypted_content"] = it.encrypted
		}
		if !it.noSummary {
			part = respEvent("response.reasoning_summary_part.added", map[string]any{
				"item_id": it.id, "output_index": it.index, "summary_index": 0,
				"part": map[string]any{"type": "summary_text", "text": ""},
			})
		}
	case "function_call":
		item = map[string]any{
			"id": it.id, "type": "function_call", "status": "in_progress",
			"call_id": it.callID, "name": it.name, "arguments": "",
		}
	default:
		item = map[string]any{
			"id": it.id, "type": "message", "role": "assistant",
			"status": "in_progress", "content": []any{},
		}
		part = respEvent("response.content_part.added", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": ""},
		})
	}
	evs := []stream.Event{respEvent("response.output_item.added", map[string]any{
		"output_index": it.index, "item": item,
	})}
	if part.Event != "" {
		evs = append(evs, part)
	}
	return evs
}

func (s *msgToResp) doneEvents(it *msgRespItem) []stream.Event {
	var evs []stream.Event
	switch it.kind {
	case "function_call":
		evs = append(evs, respEvent("response.function_call_arguments.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "arguments": it.text.String(),
		}))
	case "reasoning":
		if !it.noSummary {
			evs = append(evs,
				respEvent("response.reasoning_summary_text.done", map[string]any{
					"item_id": it.id, "output_index": it.index, "summary_index": 0,
					"text": it.text.String(),
				}),
				respEvent("response.reasoning_summary_part.done", map[string]any{
					"item_id": it.id, "output_index": it.index, "summary_index": 0,
					"part": map[string]any{"type": "summary_text", "text": it.text.String()},
				}))
		}
	default:
		evs = append(evs,
			respEvent("response.output_text.done", map[string]any{
				"item_id": it.id, "output_index": it.index, "content_index": 0,
				"text": it.text.String(),
			}),
			respEvent("response.content_part.done", map[string]any{
				"item_id": it.id, "output_index": it.index, "content_index": 0,
				"part": map[string]any{"type": "output_text", "text": it.text.String()},
			}))
	}
	return append(evs, respEvent("response.output_item.done", map[string]any{
		"output_index": it.index, "item": it.snapshot(),
	}))
}

func (it *msgRespItem) snapshot() map[string]any {
	switch it.kind {
	case "function_call":
		return map[string]any{
			"id": it.id, "type": "function_call", "status": "completed",
			"call_id": it.callID, "name": it.name, "arguments": it.text.String(),
		}
	case "reasoning":
		item := map[string]any{
			"id": it.id, "type": "reasoning",
			"summary": []any{map[string]any{"type": "summary_text", "text": it.text.String()}},
		}
		if it.encrypted != "" {
			item["encrypted_content"] = it.encrypted
		}
		return item
	default:
		return map[string]any{
			"id": it.id, "type": "message", "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": it.text.String()}},
		}
	}
}

func (s *msgToResp) closeBlock(idx int) []stream.Event {
	it := s.blocks[idx]
	if it == nil || it.closed {
		return nil
	}
	it.closed = true
	return s.doneEvents(it)
}

func (s *msgToResp) feed(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	var evs []stream.Event
	switch typ {
	case "message_start":
		if msg, ok := jsonx.AsMap(obj["message"]); ok {
			s.id = jsonx.GetString(msg, "id")
			if m := jsonx.GetString(msg, "model"); m != "" {
				s.model = m
			}
			if u, ok := jsonx.AsMap(msg["usage"]); ok {
				in, _ := jsonx.Int(u["input_tokens"])
				cr, _ := jsonx.Int(u["cache_read_input_tokens"])
				cw, _ := jsonx.Int(u["cache_creation_input_tokens"])
				s.inTok = in + cr + cw
				s.cachedTok = cr
			}
		}
		s.started = true
		evs = append(evs,
			respEvent("response.created", map[string]any{"response": s.responseShell()}),
			respEvent("response.in_progress", map[string]any{"response": s.responseShell()}),
		)
	case "content_block_start":
		idx, _ := jsonx.Int(obj["index"])
		cb, _ := jsonx.AsMap(obj["content_block"])
		if cb != nil {
			evs = append(evs, s.openBlock(idx, cb)...)
		}
	case "content_block_delta":
		idx, _ := jsonx.Int(obj["index"])
		it := s.blocks[idx]
		delta, _ := jsonx.AsMap(obj["delta"])
		if it == nil || delta == nil {
			return evs
		}
		switch jsonx.GetString(delta, "type") {
		case "text_delta":
			text := jsonx.GetString(delta, "text")
			it.text.WriteString(text)
			evs = append(evs, respEvent("response.output_text.delta", map[string]any{
				"item_id": it.id, "output_index": it.index, "content_index": 0, "delta": text,
			}))
		case "input_json_delta":
			part := jsonx.GetString(delta, "partial_json")
			it.text.WriteString(part)
			evs = append(evs, respEvent("response.function_call_arguments.delta", map[string]any{
				"item_id": it.id, "output_index": it.index, "delta": part,
			}))
		case "thinking_delta":
			thinking := jsonx.GetString(delta, "thinking")
			it.text.WriteString(thinking)
			evs = append(evs, respEvent("response.reasoning_summary_text.delta", map[string]any{
				"item_id": it.id, "output_index": it.index, "summary_index": 0, "delta": thinking,
			}))
		case "signature_delta":
			it.encrypted += jsonx.GetString(delta, "signature")
		}
	case "content_block_stop":
		idx, _ := jsonx.Int(obj["index"])
		evs = append(evs, s.closeBlock(idx)...)
	case "message_delta":
		if d, ok := jsonx.AsMap(obj["delta"]); ok {
			if sr := jsonx.GetString(d, "stop_reason"); sr != "" {
				s.stop = sr
			}
		}
		// Anthropic reports the running output count here, not on
		// message_stop — reading usage off the terminal frame is what used
		// to zero out billing on this fold.
		if u, ok := jsonx.AsMap(obj["usage"]); ok {
			if out, ok := jsonx.Int(u["output_tokens"]); ok {
				s.outTok = out
			}
		}
	case "message_stop":
		evs = append(evs, s.complete()...)
	}
	return evs
}

func (s *msgToResp) complete() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	output := make([]any, 0, len(s.items))
	for _, it := range s.items {
		if !it.closed {
			it.closed = true
			evs = append(evs, s.doneEvents(it)...)
		}
		output = append(output, it.snapshot())
	}
	status := "completed"
	if s.stop == "max_tokens" {
		status = "incomplete"
	}
	usage := map[string]any{
		"input_tokens":  s.inTok,
		"output_tokens": s.outTok,
		"total_tokens":  s.inTok + s.outTok,
	}
	if s.cachedTok > 0 {
		usage["input_tokens_details"] = map[string]any{"cached_tokens": s.cachedTok}
	}
	resp := map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": status,
		"output": output,
		"usage":  usage,
	}
	return append(evs, respEvent("response.completed", map[string]any{"response": resp}))
}

func (s *msgToResp) finish() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	return s.complete()
}

type respToMsg struct {
	started    bool
	closed     bool
	id         string
	model      string
	blockOpen  bool
	blockIndex int
	blockType  string
}

func (s *respToMsg) feed(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	if resp, ok := jsonx.AsMap(obj["response"]); ok {
		if id := jsonx.GetString(resp, "id"); id != "" {
			s.id = id
		}
		if m := jsonx.GetString(resp, "model"); m != "" {
			s.model = m
		}
	}
	var evs []stream.Event
	switch typ {
	case "response.created":
		s.started = true
		msg := map[string]any{"id": s.id, "type": "message", "role": "assistant", "model": s.model, "content": []any{}}
		evs = append(evs, stream.Event{Event: "message_start", Data: mustJSON(map[string]any{"type": "message_start", "message": msg})})
	case "response.output_text.delta":
		text := jsonx.String(obj["delta"])
		evs = append(evs, s.ensure("text", map[string]any{"type": "text", "text": ""})...)
		evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})})
	case "response.function_call_arguments.delta":
		evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": jsonx.String(obj["delta"])},
		})})
	case "response.output_item.added":
		item, _ := jsonx.AsMap(obj["item"])
		if item != nil && jsonx.GetString(item, "type") == "function_call" {
			evs = append(evs, s.close()...)
			s.blockOpen = true
			s.blockType = "tool_use"
			if s.blockType != "" {
				s.blockIndex++
			}
			evs = append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
				"type": "content_block_start", "index": s.blockIndex,
				"content_block": map[string]any{
					"type": "tool_use", "id": jsonx.GetString(item, "call_id"),
					"name": jsonx.GetString(item, "name"), "input": map[string]any{},
				},
			})})
		}
	case "response.reasoning_summary_text.delta":
		evs = append(evs, s.ensure("thinking", map[string]any{"type": "thinking", "thinking": ""})...)
		evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": jsonx.String(obj["delta"])},
		})})
	case "response.completed":
		evs = append(evs, s.close()...)
		stop := "end_turn"
		resp, _ := jsonx.AsMap(obj["response"])
		usage := any(nil)
		if resp != nil {
			if jsonx.GetString(resp, "status") == "incomplete" {
				stop = "max_tokens"
			}
			if out, ok := jsonx.AsSlice(resp["output"]); ok {
				for _, it := range out {
					if im, ok := jsonx.AsMap(it); ok && jsonx.GetString(im, "type") == "function_call" {
						stop = "tool_use"
					}
				}
			}
			usage = usageToAnthropic(resp["usage"])
		}
		evs = append(evs, stream.Event{Event: "message_delta", Data: mustJSON(map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
			"usage": usage,
		})})
		evs = append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
		s.closed = true
	}
	return evs
}

func (s *respToMsg) ensure(typ string, block map[string]any) []stream.Event {
	if s.blockOpen && s.blockType == typ {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.close()...)
	idx := 0
	if s.blockType != "" {
		idx = s.blockIndex + 1
	}
	s.blockIndex = idx
	s.blockOpen = true
	s.blockType = typ
	evs = append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
		"type": "content_block_start", "index": idx, "content_block": block,
	})})
	return evs
}

func (s *respToMsg) close() []stream.Event {
	if !s.blockOpen {
		return nil
	}
	s.blockOpen = false
	return []stream.Event{{Event: "content_block_stop", Data: mustJSON(map[string]any{
		"type": "content_block_stop", "index": s.blockIndex,
	})}}
}

func (s *respToMsg) finish() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.close()...)
	evs = append(evs, stream.Event{Event: "message_delta", Data: mustJSON(map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
	})})
	evs = append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
	s.closed = true
	return evs
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
