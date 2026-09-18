package chatresponses

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

type chatToRespState struct {
	started bool
	closed  bool
	id      string
	model   string
	text    string
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
		evs = append(evs, stream.Event{Event: "response.created", Data: mustJSON(map[string]any{
			"type": "response.created",
			"response": map[string]any{
				"id":     s.id,
				"object": "response",
				"model":  s.model,
				"status": "in_progress",
			},
		})})
		evs = append(evs, stream.Event{Event: "response.output_item.added", Data: mustJSON(map[string]any{
			"type":         "response.output_item.added",
			"output_index": 0,
			"item":         map[string]any{"type": "message", "role": "assistant", "content": []any{}},
		})})
	}
	choices, _ := jsonx.AsSlice(obj["choices"])
	if len(choices) > 0 {
		ch, _ := jsonx.AsMap(choices[0])
		delta, _ := jsonx.AsMap(ch["delta"])
		if delta != nil {
			if c := delta["content"]; c != nil {
				text := jsonx.String(c)
				if text != "" {
					s.text += text
					evs = append(evs, stream.Event{Event: "response.output_text.delta", Data: mustJSON(map[string]any{
						"type":          "response.output_text.delta",
						"delta":         text,
						"output_index":  0,
						"content_index": 0,
					})})
				}
			}
			if tcs, ok := jsonx.AsSlice(delta["tool_calls"]); ok {
				for _, tc := range tcs {
					tm, ok := jsonx.AsMap(tc)
					if !ok {
						continue
					}
					fn, _ := jsonx.AsMap(tm["function"])
					args := ""
					name := ""
					if fn != nil {
						args = jsonx.GetString(fn, "arguments")
						name = jsonx.GetString(fn, "name")
					}
					if name != "" || jsonx.GetString(tm, "id") != "" {
						evs = append(evs, stream.Event{Event: "response.output_item.added", Data: mustJSON(map[string]any{
							"type": "response.output_item.added",
							"item": map[string]any{
								"type":    "function_call",
								"call_id": jsonx.GetString(tm, "id"),
								"name":    name,
							},
						})})
					}
					if args != "" {
						evs = append(evs, stream.Event{Event: "response.function_call_arguments.delta", Data: mustJSON(map[string]any{
							"type":  "response.function_call_arguments.delta",
							"delta": args,
						})})
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
	resp := map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": finishToStatus(finish),
		"output": []any{
			map[string]any{
				"type":    "message",
				"role":    "assistant",
				"content": []any{map[string]any{"type": "output_text", "text": s.text}},
			},
		},
		"usage": chatUsageToResponses(usage),
	}
	return []stream.Event{
		{Event: "response.completed", Data: mustJSON(map[string]any{"type": "response.completed", "response": resp})},
	}
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
