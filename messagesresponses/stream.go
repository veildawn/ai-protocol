package messagesresponses

import (
	"encoding/json"
	"io"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/stream"
)

type StreamOpts struct {
	Flush func()
	Model string
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
			nn, werr := stream.WriteEvent(dst, out, opts.Flush)
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
		nn, werr := stream.WriteEvent(dst, out, opts.Flush)
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
			nn, werr := stream.WriteEvent(dst, out, opts.Flush)
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
		nn, werr := stream.WriteEvent(dst, out, opts.Flush)
		n += int64(nn)
		if werr != nil {
			return n, werr
		}
	}
	return n, nil
}

type msgToResp struct {
	started bool
	closed  bool
	id      string
	model   string
	text    string
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
		}
		s.started = true
		evs = append(evs, stream.Event{Event: "response.created", Data: mustJSON(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": s.id, "object": "response", "model": s.model, "status": "in_progress"},
		})})
	case "content_block_delta":
		delta, _ := jsonx.AsMap(obj["delta"])
		if delta == nil {
			return evs
		}
		switch jsonx.GetString(delta, "type") {
		case "text_delta":
			text := jsonx.GetString(delta, "text")
			s.text += text
			evs = append(evs, stream.Event{Event: "response.output_text.delta", Data: mustJSON(map[string]any{
				"type": "response.output_text.delta", "delta": text,
			})})
		case "input_json_delta":
			evs = append(evs, stream.Event{Event: "response.function_call_arguments.delta", Data: mustJSON(map[string]any{
				"type": "response.function_call_arguments.delta", "delta": jsonx.GetString(delta, "partial_json"),
			})})
		case "thinking_delta":
			evs = append(evs, stream.Event{Event: "response.reasoning_summary_text.delta", Data: mustJSON(map[string]any{
				"type": "response.reasoning_summary_text.delta", "delta": jsonx.GetString(delta, "thinking"),
			})})
		}
	case "content_block_start":
		cb, _ := jsonx.AsMap(obj["content_block"])
		if cb != nil && jsonx.GetString(cb, "type") == "tool_use" {
			evs = append(evs, stream.Event{Event: "response.output_item.added", Data: mustJSON(map[string]any{
				"type": "response.output_item.added",
				"item": map[string]any{
					"type":    "function_call",
					"call_id": jsonx.GetString(cb, "id"),
					"name":    jsonx.GetString(cb, "name"),
				},
			})})
		}
	case "message_delta":
		// wait for message_stop to emit completed
	case "message_stop":
		evs = append(evs, s.complete(obj["usage"])...)
	}
	return evs
}

func (s *msgToResp) complete(usage any) []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	u := usageToAnthropic(usage)
	in, _ := jsonx.Int(u["input_tokens"])
	out, _ := jsonx.Int(u["output_tokens"])
	resp := map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": "completed",
		"output": []any{map[string]any{
			"type":    "message",
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": s.text}},
		}},
		"usage": map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": in + out},
	}
	return []stream.Event{{Event: "response.completed", Data: mustJSON(map[string]any{"type": "response.completed", "response": resp})}}
}

func (s *msgToResp) finish() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	return s.complete(nil)
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
