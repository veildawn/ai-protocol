package chatmessages

import (
	"encoding/json"
	"io"
	"strconv"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/stream"
)

type StreamOpts struct {
	Flush func()
	Model string
}

func PipeChatToMessages(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &chatToMsgState{model: opts.Model}
	var n int64
	err := stream.Scan(src, func(ev stream.Event) error {
		if stream.IsDone(ev.Data) {
			return nil
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil // skip comments / non-json
		}
		for _, out := range st.feedChatChunk(obj) {
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

func PipeMessagesToChat(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &msgToChatState{model: opts.Model, id: "chatcmpl-unknown"}
	var n int64
	err := stream.Scan(src, func(ev stream.Event) error {
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		for _, out := range st.feedAnthropic(ev.Event, obj) {
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

type chatToMsgState struct {
	started    bool
	blockOpen  bool
	blockType  string // text, tool_use, thinking
	blockIndex int
	id         string
	model      string
	toolIndex  map[int]int // openai tool index -> anthropic block index
	closed     bool
}

func (s *chatToMsgState) feedChatChunk(obj map[string]any) []stream.Event {
	if jsonx.GetString(obj, "id") != "" {
		s.id = jsonx.GetString(obj, "id")
	}
	if m := jsonx.GetString(obj, "model"); m != "" {
		s.model = m
	}
	var evs []stream.Event
	if !s.started {
		s.started = true
		msg := map[string]any{
			"id":      s.id,
			"type":    "message",
			"role":    "assistant",
			"model":   s.model,
			"content": []any{},
		}
		evs = append(evs,
			stream.Event{Event: "message_start", Data: mustJSON(map[string]any{"type": "message_start", "message": msg})},
		)
	}
	choices, _ := jsonx.AsSlice(obj["choices"])
	if len(choices) == 0 {
		return evs
	}
	ch, _ := jsonx.AsMap(choices[0])
	delta, _ := jsonx.AsMap(ch["delta"])
	if delta == nil {
		delta = map[string]any{}
	}
	if rc := jsonx.GetString(delta, "reasoning_content"); rc != "" {
		evs = append(evs, s.ensureBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})...)
		evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": rc},
		})})
	}
	if content := delta["content"]; content != nil {
		text := jsonx.String(content)
		if text != "" {
			evs = append(evs, s.ensureBlock("text", map[string]any{"type": "text", "text": ""})...)
			evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
				"type": "content_block_delta", "index": s.blockIndex,
				"delta": map[string]any{"type": "text_delta", "text": text},
			})})
		}
	}
	if tcs, ok := jsonx.AsSlice(delta["tool_calls"]); ok {
		for _, tc := range tcs {
			tm, ok := jsonx.AsMap(tc)
			if !ok {
				continue
			}
			idx, _ := jsonx.Int(tm["index"])
			if s.toolIndex == nil {
				s.toolIndex = map[int]int{}
			}
			fn, _ := jsonx.AsMap(tm["function"])
			name := ""
			args := ""
			if fn != nil {
				name = jsonx.GetString(fn, "name")
				args = jsonx.GetString(fn, "arguments")
			}
			if _, seen := s.toolIndex[idx]; !seen {
				id := jsonx.GetString(tm, "id")
				evs = append(evs, s.closeBlock()...)
				s.blockIndex++
				if s.blockOpen {
					s.blockIndex--
				}
				s.blockOpen = true
				s.blockType = "tool_use"
				s.toolIndex[idx] = s.blockIndex
				evs = append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
					"type": "content_block_start", "index": s.blockIndex,
					"content_block": map[string]any{"type": "tool_use", "id": id, "name": name, "input": map[string]any{}},
				})})
			}
			if args != "" {
				bi := s.toolIndex[idx]
				evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
					"type": "content_block_delta", "index": bi,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
				})})
			}
		}
	}
	if fr := jsonx.GetString(ch, "finish_reason"); fr != "" {
		evs = append(evs, s.closeBlock()...)
		usage := openaiUsageToAnthropic(obj["usage"])
		evs = append(evs, stream.Event{Event: "message_delta", Data: mustJSON(map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": finishToStopReason(fr), "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": usage["output_tokens"]},
		})})
		evs = append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
		s.closed = true
	}
	return evs
}

func (s *chatToMsgState) ensureBlock(typ string, block map[string]any) []stream.Event {
	if s.blockOpen && s.blockType == typ {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.closeBlock()...)
	if s.blockOpen {
		// closed
	}
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

func (s *chatToMsgState) closeBlock() []stream.Event {
	if !s.blockOpen {
		return nil
	}
	s.blockOpen = false
	return []stream.Event{{Event: "content_block_stop", Data: mustJSON(map[string]any{
		"type": "content_block_stop", "index": s.blockIndex,
	})}}
}

func (s *chatToMsgState) finish() []stream.Event {
	if s.closed {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.closeBlock()...)
	if s.started {
		evs = append(evs, stream.Event{Event: "message_delta", Data: mustJSON(map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 0},
		})})
		evs = append(evs, stream.Event{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})})
		s.closed = true
	}
	return evs
}

type msgToChatState struct {
	id        string
	model     string
	toolID    string
	toolName  string
	toolIndex int
	nextTool  int
	emitted   bool
	done      bool
	blockType string
	usage     map[string]any
}

func (s *msgToChatState) feedAnthropic(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	switch typ {
	case "message_start":
		if msg, ok := jsonx.AsMap(obj["message"]); ok {
			if id := jsonx.GetString(msg, "id"); id != "" {
				s.id = id
			}
			if m := jsonx.GetString(msg, "model"); m != "" {
				s.model = m
			}
			if u, ok := jsonx.AsMap(msg["usage"]); ok {
				s.usage = copyUsage(u)
			}
		}
		return []stream.Event{s.chunk(map[string]any{"role": "assistant"}, "")}
	case "content_block_start":
		cb, _ := jsonx.AsMap(obj["content_block"])
		s.blockType = jsonx.GetString(cb, "type")
		if s.blockType == "tool_use" {
			s.toolID = jsonx.GetString(cb, "id")
			s.toolName = jsonx.GetString(cb, "name")
			s.toolIndex = s.nextTool
			s.nextTool++
			return []stream.Event{s.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": s.toolIndex,
					"id":    s.toolID,
					"type":  "function",
					"function": map[string]any{
						"name":      s.toolName,
						"arguments": "",
					},
				}},
			}, "")}
		}
		return nil
	case "content_block_delta":
		delta, _ := jsonx.AsMap(obj["delta"])
		dt := jsonx.GetString(delta, "type")
		switch dt {
		case "text_delta":
			return []stream.Event{s.chunk(map[string]any{"content": jsonx.GetString(delta, "text")}, "")}
		case "thinking_delta":
			return []stream.Event{s.chunk(map[string]any{"reasoning_content": jsonx.GetString(delta, "thinking")}, "")}
		case "input_json_delta":
			return []stream.Event{s.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": s.toolIndex,
					"function": map[string]any{
						"arguments": jsonx.GetString(delta, "partial_json"),
					},
				}},
			}, "")}
		}
		return nil
	case "message_delta":
		d, _ := jsonx.AsMap(obj["delta"])
		fr := ""
		if d != nil {
			fr = stopReasonToFinish(jsonx.GetString(d, "stop_reason"))
		}
		if um, ok := jsonx.AsMap(obj["usage"]); ok {
			s.usage = mergeUsage(s.usage, um)
		}
		chunk := s.fullChunk(map[string]any{}, fr)
		if s.usage != nil {
			chunk["usage"] = anthropicUsageToOpenAI(s.usage)
		}
		return []stream.Event{{Data: mustJSON(chunk)}}
	case "message_stop":
		s.done = true
		return []stream.Event{{Data: "[DONE]"}}
	default:
		return nil
	}
}

func (s *msgToChatState) chunk(delta map[string]any, finish string) stream.Event {
	return stream.Event{Data: mustJSON(s.fullChunk(delta, finish))}
}

func (s *msgToChatState) fullChunk(delta map[string]any, finish any) map[string]any {
	var fr any
	if finish != "" {
		fr = finish
	}
	return map[string]any{
		"id":     s.id,
		"object": "chat.completion.chunk",
		"model":  s.model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"delta":         delta,
				"finish_reason": fr,
			},
		},
	}
}

func (s *msgToChatState) finish() []stream.Event {
	if s.done {
		return nil
	}
	s.done = true
	return []stream.Event{
		s.chunk(map[string]any{}, "stop"),
		{Data: "[DONE]"},
	}
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func itoa(i int) string { return strconv.Itoa(i) }

func copyUsage(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func mergeUsage(base, delta map[string]any) map[string]any {
	if base == nil {
		return copyUsage(delta)
	}
	out := copyUsage(base)
	for k, v := range delta {
		out[k] = v
	}
	return out
}
