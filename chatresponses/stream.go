package chatresponses

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
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

func PipeChatToResponses(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &chatToRespState{model: opts.Model}
	var n int64
	err := stream.Scan(src, func(ev stream.Event) error {
		if stream.IsDone(ev.Data) {
			return nil
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		for _, out := range st.feed(obj) {
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

func PipeResponsesToChat(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &respToChatState{model: opts.Model, id: "chatcmpl-unknown"}
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

// chatRespItem is one Responses output item being assembled from chat deltas.
// Text accumulates in text regardless of kind: message output_text, reasoning
// summary_text, or function_call arguments.
type chatRespItem struct {
	kind   string // "message" | "reasoning" | "function_call"
	id     string
	index  int
	text   strings.Builder
	callID string
	name   string
}

type chatToRespState struct {
	started bool
	closed  bool
	id      string
	model   string

	items   []*chatRespItem
	nextIdx int
	seq     int

	msg    *chatRespItem            // the one message item, opened lazily
	rs     *chatRespItem            // the one reasoning item, opened lazily
	fcs    map[string]*chatRespItem // tool-call key → item
	lastFC *chatRespItem            // fallback target for keyless arg fragments
}

func respEvent(typ string, fields map[string]any) stream.Event {
	fields["type"] = typ
	return stream.Event{Event: typ, Data: mustJSON(fields)}
}

func (s *chatToRespState) responseShell() map[string]any {
	return map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": "in_progress",
	}
}

func (s *chatToRespState) newItem(kind, prefix string) *chatRespItem {
	it := &chatRespItem{kind: kind, id: fmt.Sprintf("%s_%d", prefix, s.seq), index: s.nextIdx}
	s.seq++
	s.nextIdx++
	s.items = append(s.items, it)
	return it
}

// openEvents emits the item's opening frame pair: output_item.added, then the
// part frame text kinds need before their first delta.
func (s *chatToRespState) openEvents(it *chatRespItem) []stream.Event {
	var item map[string]any
	var part stream.Event
	switch it.kind {
	case "reasoning":
		item = map[string]any{"id": it.id, "type": "reasoning", "summary": []any{}}
		part = respEvent("response.reasoning_summary_part.added", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": ""},
		})
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

// doneEvents emits the item's closing frames; the item snapshot they carry is
// also what lands in response.completed's output array.
func (s *chatToRespState) doneEvents(it *chatRespItem) []stream.Event {
	var first stream.Event
	switch it.kind {
	case "function_call":
		first = respEvent("response.function_call_arguments.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "arguments": it.text.String(),
		})
	case "reasoning":
		first = respEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"text": it.text.String(),
		})
	default:
		first = respEvent("response.output_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			"text": it.text.String(),
		})
	}
	evs := []stream.Event{first}
	switch it.kind {
	case "reasoning":
		evs = append(evs, respEvent("response.reasoning_summary_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": it.text.String()},
		}))
	case "message":
		evs = append(evs, respEvent("response.content_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": it.text.String()},
		}))
	}
	return append(evs, respEvent("response.output_item.done", map[string]any{
		"output_index": it.index, "item": it.snapshot(),
	}))
}

func (it *chatRespItem) snapshot() map[string]any {
	switch it.kind {
	case "function_call":
		return map[string]any{
			"id": it.id, "type": "function_call", "status": "completed",
			"call_id": it.callID, "name": it.name, "arguments": it.text.String(),
		}
	case "reasoning":
		return map[string]any{
			"id": it.id, "type": "reasoning",
			"summary": []any{map[string]any{"type": "summary_text", "text": it.text.String()}},
		}
	default:
		return map[string]any{
			"id": it.id, "type": "message", "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": it.text.String()}},
		}
	}
}

func (s *chatToRespState) ensure(kind string) (*chatRespItem, []stream.Event) {
	slot := &s.msg
	prefix := "msg"
	if kind == "reasoning" {
		slot = &s.rs
		prefix = "rs"
	}
	if *slot != nil {
		return *slot, nil
	}
	it := s.newItem(kind, prefix)
	*slot = it
	return it, s.openEvents(it)
}

// fcFor resolves a tool_calls delta fragment to its item. Fragments key by
// `index` when present (the parallel-call slot) else by `id`; a fragment with
// neither continues whichever call is open. `id` alone is never the trigger
// for a new item — real upstreams repeat it on every fragment, which is what
// used to mint a fresh output_item.added per chunk.
func (s *chatToRespState) fcFor(tm map[string]any) (*chatRespItem, []stream.Event) {
	key := ""
	if idx, ok := jsonx.Int(tm["index"]); ok {
		key = "i" + strconv.Itoa(idx)
	} else if cid := jsonx.GetString(tm, "id"); cid != "" {
		key = "c" + cid
	} else if s.lastFC != nil {
		return s.lastFC, nil
	} else {
		key = "i0"
	}
	if s.fcs == nil {
		s.fcs = map[string]*chatRespItem{}
	}
	if it := s.fcs[key]; it != nil {
		s.lastFC = it
		return it, nil
	}
	it := s.newItem("function_call", "fc")
	it.callID = jsonx.GetString(tm, "id")
	if fn, _ := jsonx.AsMap(tm["function"]); fn != nil {
		it.name = jsonx.GetString(fn, "name")
	}
	s.fcs[key] = it
	s.lastFC = it
	return it, s.openEvents(it)
}

func (s *chatToRespState) feed(obj map[string]any) []stream.Event {
	if id := jsonx.GetString(obj, "id"); id != "" {
		s.id = id
	}
	if m := jsonx.GetString(obj, "model"); m != "" {
		s.model = m
	}
	var evs []stream.Event
	if !s.started {
		s.started = true
		evs = append(evs,
			respEvent("response.created", map[string]any{"response": s.responseShell()}),
			respEvent("response.in_progress", map[string]any{"response": s.responseShell()}),
		)
	}
	choices, _ := jsonx.AsSlice(obj["choices"])
	if len(choices) > 0 {
		ch, _ := jsonx.AsMap(choices[0])
		delta, _ := jsonx.AsMap(ch["delta"])
		if delta != nil {
			if rc := jsonx.String(delta["reasoning_content"]); rc != "" {
				it, open := s.ensure("reasoning")
				evs = append(evs, open...)
				it.text.WriteString(rc)
				evs = append(evs, respEvent("response.reasoning_summary_text.delta", map[string]any{
					"item_id": it.id, "output_index": it.index, "summary_index": 0, "delta": rc,
				}))
			}
			if c := delta["content"]; c != nil {
				if text := jsonx.String(c); text != "" {
					it, open := s.ensure("message")
					evs = append(evs, open...)
					it.text.WriteString(text)
					evs = append(evs, respEvent("response.output_text.delta", map[string]any{
						"item_id": it.id, "output_index": it.index, "content_index": 0, "delta": text,
					}))
				}
			}
			if tcs, ok := jsonx.AsSlice(delta["tool_calls"]); ok {
				for _, tc := range tcs {
					tm, ok := jsonx.AsMap(tc)
					if !ok {
						continue
					}
					fn, _ := jsonx.AsMap(tm["function"])
					args, name := "", ""
					if fn != nil {
						args = jsonx.GetString(fn, "arguments")
						name = jsonx.GetString(fn, "name")
					}
					it, open := s.fcFor(tm)
					evs = append(evs, open...)
					if it.name == "" {
						it.name = name
					}
					if args != "" {
						it.text.WriteString(args)
						evs = append(evs, respEvent("response.function_call_arguments.delta", map[string]any{
							"item_id": it.id, "output_index": it.index, "delta": args,
						}))
					}
				}
			}
		}
		if fr := jsonx.GetString(ch, "finish_reason"); fr != "" {
			evs = append(evs, s.complete(obj["usage"], fr)...)
		}
	}
	return evs
}

func (s *chatToRespState) complete(usage any, finish string) []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	output := make([]any, 0, len(s.items))
	for _, it := range s.items {
		evs = append(evs, s.doneEvents(it)...)
		output = append(output, it.snapshot())
	}
	resp := map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": finishToStatus(finish),
		"output": output,
		"usage":  chatUsageToResponses(usage),
	}
	return append(evs, respEvent("response.completed", map[string]any{"response": resp}))
}

func (s *chatToRespState) finish() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	return s.complete(nil, "stop")
}

type respToChatState struct {
	id    string
	model string
	done  bool
}

func (s *respToChatState) feed(event string, obj map[string]any) []stream.Event {
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
	switch typ {
	case "response.created":
		return []stream.Event{s.chunk(map[string]any{"role": "assistant"}, nil)}
	case "response.output_text.delta":
		delta := obj["delta"]
		if delta == nil {
			delta = jsonx.GetString(obj, "text")
		}
		return []stream.Event{s.chunk(map[string]any{"content": jsonx.String(delta)}, nil)}
	case "response.function_call_arguments.delta":
		return []stream.Event{s.chunk(map[string]any{
			"tool_calls": []any{map[string]any{
				"index":    0,
				"function": map[string]any{"arguments": jsonx.String(obj["delta"])},
			}},
		}, nil)}
	case "response.output_item.added":
		item, _ := jsonx.AsMap(obj["item"])
		if item != nil && jsonx.GetString(item, "type") == "function_call" {
			return []stream.Event{s.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": 0,
					"id":    jsonx.GetString(item, "call_id"),
					"type":  "function",
					"function": map[string]any{
						"name":      jsonx.GetString(item, "name"),
						"arguments": "",
					},
				}},
			}, nil)}
		}
		return nil
	case "response.completed":
		s.done = true
		resp, _ := jsonx.AsMap(obj["response"])
		finish := "stop"
		if resp != nil {
			finish = statusToFinish(jsonx.GetString(resp, "status"), false)
			if out, ok := jsonx.AsSlice(resp["output"]); ok {
				for _, it := range out {
					if im, ok := jsonx.AsMap(it); ok && jsonx.GetString(im, "type") == "function_call" {
						finish = "tool_calls"
					}
				}
			}
		}
		chunk := s.full(map[string]any{}, finish)
		if resp != nil {
			chunk["usage"] = responsesUsageToChat(resp["usage"])
		}
		return []stream.Event{{Data: mustJSON(chunk)}, {Data: "[DONE]"}}
	default:
		return nil
	}
}

func (s *respToChatState) chunk(delta map[string]any, finish any) stream.Event {
	return stream.Event{Data: mustJSON(s.full(delta, finish))}
}

func (s *respToChatState) full(delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id":     s.id,
		"object": "chat.completion.chunk",
		"model":  s.model,
		"choices": []any{
			map[string]any{"index": 0, "delta": delta, "finish_reason": finish},
		},
	}
}

func (s *respToChatState) finish() []stream.Event {
	if s.done {
		return nil
	}
	s.done = true
	return []stream.Event{s.chunk(map[string]any{}, "stop"), {Data: "[DONE]"}}
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
