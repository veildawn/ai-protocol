package chatmessages

import (
	"encoding/json"
	"io"

	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/internal/thinksig"
	"github.com/veildawn/ai-protocol/stream"
)

type StreamOpts struct {
	Flush func()
	Model string
	// OnEvent, when set, sees every target-dialect event just before it is
	// written and returns the event to write instead. See protocol.StreamOpts.
	OnEvent func(stream.Event) (stream.Event, error)
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

// PipeMessagesToChat folds an Anthropic Messages SSE stream into
// chat.completion.chunk events.
//
// The chat client is told the turn finished only when the SOURCE said so:
// message_stop, or the [DONE] sentinel a relay dialect appends instead. A
// Messages source that ends with neither was cut — the dialect has no other
// legal ending — and inventing a "stop" for it would tell the client it holds a
// finished answer while the host records a failed delivery for the same turn
// (StreamResult carries it as Truncated). See PipeResponsesToChat for the worked
// case that made this the rule.
func PipeMessagesToChat(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &msgToChatState{model: opts.Model, id: "chatcmpl-unknown"}
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
		// [DONE] is the one close a relay dialect writes instead of the typed
		// terminal, so it closes the chat turn here. An EOF with no close at all
		// deliberately does not — see the doc comment.
		if stream.IsDone(ev.Data) {
			return write(st.finishFromSource())
		}
		obj, err := jsonx.UnmarshalMap([]byte(ev.Data))
		if err != nil {
			return nil
		}
		return write(st.feedAnthropic(ev.Event, obj))
	})
	if err != nil {
		return n, err
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
	// pendingStop holds a finish_reason seen on a chunk that carried no usage.
	// Under stream_options.include_usage the usage arrives in a later
	// choices-less chunk, so the terminal frames wait for it rather than
	// reporting zero tokens.
	pendingStop string
}

func (s *chatToMsgState) feedChatChunk(obj map[string]any) []stream.Event {
	if jsonx.GetString(obj, "id") != "" {
		s.id = jsonx.GetString(obj, "id")
	}
	if m := jsonx.GetString(obj, "model"); m != "" {
		s.model = m
	}
	var evs []stream.Event
	// A usage-bearing chunk closes a turn whose finish_reason already arrived:
	// the terminal frames were held for exactly this, so emit them now with
	// the real counts rather than the zero the earlier chunk would have forced.
	// The usage chunk may carry no choices at all, or one empty choice — the
	// include_usage tail shape — so usage alone is the signal, as it is for
	// the reference's hold-and-merge.
	if choices, _ := jsonx.AsSlice(obj["choices"]); s.pendingStop != "" && (len(choices) == 0 || obj["usage"] != nil) {
		return s.terminal(s.pendingStop, openaiUsageToAnthropic(obj["usage"]))
	}
	if !s.started {
		s.started = true
		// The opening message carries the fields an Anthropic client reads
		// before any content arrives: a null stop_reason/stop_sequence, and a
		// usage block whose input_tokens is 0 because the chat source has not
		// reported it yet (the terminal message_delta carries the real counts).
		// The cache halves are stated as 0 for the same reason the reference
		// adapter states them: clients probe these fields to decide whether
		// the target supports prompt caching at all.
		msg := map[string]any{
			"id":            s.id,
			"type":          "message",
			"role":          "assistant",
			"model":         s.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens": 0, "output_tokens": 0,
				"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
			},
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
	// delta.thinking_blocks carries a thinking block whole — the shape this
	// package's own Messages→chat fold spells its terminal in, and the shape
	// chat dialects stream on the open wire. Ignoring it silently drops the
	// signature the client needs to replay the turn. The key is read ahead of
	// reasoning_content, as the reference reads it, and folds as a thinking
	// content_block_start carrying the chunk's text and signature, plus a
	// signature_delta when signed (a thinking_delta when not); a redacted
	// block has no readable fold and drops, as it does there.
	thinkingBlocks := false
	if tbs, ok := jsonx.AsSlice(delta["thinking_blocks"]); ok {
		thinkingBlocks = true
		var aggText, aggSig, firstText, firstSig string
		sawThinking := false
		for _, tb := range tbs {
			tm, ok := jsonx.AsMap(tb)
			if !ok || jsonx.GetString(tm, "type") != "thinking" {
				continue
			}
			text := jsonx.GetString(tm, "thinking")
			sig := jsonx.GetString(tm, "signature")
			if !sawThinking {
				sawThinking, firstText, firstSig = true, text, sig
			}
			aggText += text
			aggSig += sig
		}
		if sawThinking {
			if !(s.blockOpen && s.blockType == "thinking") {
				// An unsigned chunk (no signature on the wire) opens with the
				// fold's stand-in rather than "": clients read an unsigned
				// block as "no signature" and downgrade the thinking to text on
				// replay — see internal/thinksig.
				evs = append(evs, s.ensureBlock("thinking", map[string]any{
					"type": "thinking", "thinking": firstText, "signature": thinksig.Or(firstSig),
				})...)
			}
			d := map[string]any{"type": "thinking_delta", "thinking": aggText}
			if aggSig != "" {
				d = map[string]any{"type": "signature_delta", "signature": aggSig}
			}
			if aggSig != "" || aggText != "" {
				evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
					"type": "content_block_delta", "index": s.blockIndex, "delta": d,
				})})
			}
		}
	}
	if !thinkingBlocks {
		if rc := jsonx.GetString(delta, "reasoning_content"); rc != "" {
			// reasoning_content carries no signature at all, so the block opens
			// with the fold's stand-in — see internal/thinksig.
			evs = append(evs, s.ensureBlock("thinking", map[string]any{
				"type": "thinking", "thinking": "", "signature": thinksig.Synthetic,
			})...)
			evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
				"type": "content_block_delta", "index": s.blockIndex,
				"delta": map[string]any{"type": "thinking_delta", "thinking": rc},
			})})
		}
	}
	if content := delta["content"]; content != nil {
		// A part list must be read part by part — the non-streaming fold's
		// rule: rendering it with a generic formatter produces Go syntax the
		// model cannot read.
		if text := chatContentToText(content); text != "" {
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
				evs = append(evs, s.closeBlock()...)
				// Anthropic numbers content blocks densely from 0, so this
				// call takes the slot after whatever block was open — the
				// first one lands on 0. Advancing unconditionally would start
				// the stream at index 1 and leave a hole a client accumulating
				// content[index] cannot fill.
				if s.blockType != "" {
					s.blockIndex++
				}
				s.blockOpen = true
				s.blockType = "tool_use"
				s.toolIndex[idx] = s.blockIndex
				// The id is normalized to the Messages character class; a
				// thought-signature suffix riding it cannot, so it moves onto
				// the block, the shape the reference emits, keeping the turn
				// replayable.
				base, thoughtSig := splitThoughtSignature(jsonx.GetString(tm, "id"))
				block := map[string]any{
					"type": "tool_use", "id": normalizeToolUseID(base),
					"name": name, "input": map[string]any{},
				}
				if thoughtSig != "" {
					block["provider_specific_fields"] = map[string]any{"signature": thoughtSig}
				}
				evs = append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
					"type": "content_block_start", "index": s.blockIndex, "content_block": block,
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
	// A relay may repeat finish_reason on a trailing usage-only chunk. The
	// terminal frames go out once: a second message_stop is a protocol
	// violation a client reading the stream cannot reconcile.
	if fr := jsonx.GetString(ch, "finish_reason"); fr != "" && !s.closed {
		evs = append(evs, s.closeBlock()...)
		// Under stream_options.include_usage the usage rides a later
		// choices-less chunk, so the terminal frames wait for it: emitting
		// them now would report zero tokens for a turn that has a real count.
		if len(choices) == 0 || obj["usage"] == nil {
			s.pendingStop = fr
			return evs
		}
		evs = append(evs, s.terminal(fr, openaiUsageToAnthropic(obj["usage"]))...)
	}
	return evs
}

// terminal emits the frames that close an Anthropic turn. The closing delta
// names only the stop_reason — the reference's message_delta carries no
// stop_sequence key.
func (s *chatToMsgState) terminal(finish string, usage map[string]any) []stream.Event {
	s.closed = true
	return []stream.Event{
		{Event: "message_delta", Data: mustJSON(map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": finishToStopReason(finish)},
			"usage": usage,
		})},
		{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})},
	}
}

func (s *chatToMsgState) ensureBlock(typ string, block map[string]any) []stream.Event {
	if s.blockOpen && s.blockType == typ {
		return nil
	}
	var evs []stream.Event
	evs = append(evs, s.closeBlock()...)
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

// finish closes out a chat source that ended without a finish_reason frame.
// A held stop_reason whose usage never arrived still closes the turn with the
// real reason — the counts it would have carried are simply absent. A stream
// that never said how it ended is closed, not answered: the client gets a bare
// message_stop — the one frame the reference emits there — and no synthesized
// stop_reason or zero-usage message_delta that would dress a cut stream as a
// clean turn. The turn still closes for the client: a Chat source's folds
// close the turn.
func (s *chatToMsgState) finish() []stream.Event {
	if s.closed {
		return nil
	}
	if s.pendingStop != "" {
		var evs []stream.Event
		evs = append(evs, s.closeBlock()...)
		evs = append(evs, s.terminal(s.pendingStop, map[string]any{"input_tokens": 0, "output_tokens": 0})...)
		return evs
	}
	if s.started {
		s.closed = true
		return []stream.Event{{Event: "message_stop", Data: mustJSON(map[string]any{"type": "message_stop"})}}
	}
	return nil
}

type msgToChatState struct {
	id        string
	model     string
	toolID    string
	toolName  string
	toolIndex int
	nextTool  int
	sentRole  bool
	done      bool
	blockType string
	usage     map[string]any
	// thinkingText accumulates the current thinking block's content, which
	// arrives on deltas ahead of its signature; the signature_delta chunk
	// replays the block whole — text plus signature — the moment it lands.
	thinkingText string
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
		// No chunk of its own: the role rides the first chunk that carries
		// content, the shape the reference emits.
		return nil
	case "content_block_start":
		cb, _ := jsonx.AsMap(obj["content_block"])
		s.blockType = jsonx.GetString(cb, "type")
		// A server tool call is a tool call on the chat wire too: the same
		// id/name open chunk, the same index accounting. Dropping it loses
		// the call's identity and collides the next tool on its index.
		if s.blockType == "tool_use" || s.blockType == "server_tool_use" {
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
		if s.blockType == "thinking" {
			// The signature chunk replays the block it closes, so a second
			// thinking block must not inherit the first one's text.
			s.thinkingText = ""
		}
		if s.blockType == "redacted_thinking" {
			// A redacted block carries opaque bytes and no readable text. The
			// chat wire spells it as a thinking block, and dropping it leaves
			// the turn unreplayable against a target that verifies it. It goes
			// out the moment the block opens — the reference emits it there,
			// and a stream cut before the terminal would otherwise lose it.
			return []stream.Event{s.chunk(map[string]any{
				"thinking_blocks": []any{map[string]any{
					"type": "redacted_thinking",
					"data": jsonx.GetString(cb, "data"),
				}},
			}, "")}
		}
		return nil
	case "content_block_stop":
		// The block's deltas are over; a stray json delta after this point
		// belongs to no open tool.
		s.blockType = ""
		return nil
	case "content_block_delta":
		delta, _ := jsonx.AsMap(obj["delta"])
		dt := jsonx.GetString(delta, "type")
		switch dt {
		case "text_delta":
			return []stream.Event{s.chunk(map[string]any{"content": jsonx.GetString(delta, "text")}, "")}
		case "thinking_delta":
			text := jsonx.GetString(delta, "thinking")
			s.thinkingText += text
			// The incremental text rides both spellings, as the reference
			// emits it: reasoning_content for clients that read it, and an
			// incremental thinking block for those that replay the turn from
			// blocks.
			return []stream.Event{s.chunk(map[string]any{
				"reasoning_content": text,
				"thinking_blocks":   []any{map[string]any{"type": "thinking", "thinking": text}},
			}, "")}
		case "signature_delta":
			// The signature rides a delta after the thinking text; the block it
			// signs goes out with it, whole, rather than waiting for the
			// terminal chunk — a stream cut in between would otherwise lose it.
			return []stream.Event{s.chunk(map[string]any{
				"reasoning_content": "",
				"thinking_blocks": []any{map[string]any{
					"type":      "thinking",
					"thinking":  s.thinkingText,
					"signature": jsonx.GetString(delta, "signature"),
				}},
			}, "")}
		case "input_json_delta":
			// Only a tool block's arguments ride tool_calls: a server-tool
			// result's partial_json would append to the previous call's
			// arguments and corrupt them.
			if s.blockType != "tool_use" && s.blockType != "server_tool_use" {
				return nil
			}
			return []stream.Event{s.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": s.toolIndex,
					// A continuation entry names its kind too — the
					// reference spells type on every chunk of the call.
					"type": "function",
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
		// The thinking block has already gone out — incrementally on each
		// thinking_delta and whole on the signature_delta — so the terminal
		// chunk carries only the close, as the reference's does. Usage rides
		// its own trailing chunk, the shape the reference spells under
		// stream_options.include_usage: one choice with an empty delta, the
		// usage beside it.
		evs := []stream.Event{{Data: mustJSON(s.fullChunk(map[string]any{}, fr))}}
		if s.usage != nil {
			evs = append(evs, stream.Event{Data: mustJSON(map[string]any{
				"id":     s.id,
				"object": "chat.completion.chunk",
				"model":  s.model,
				"choices": []any{map[string]any{
					"index": 0, "delta": map[string]any{},
				}},
				"usage": anthropicUsageToOpenAI(s.usage),
			})})
		}
		return evs
	case "message_stop":
		s.done = true
		return []stream.Event{{Data: "[DONE]"}}
	case "error":
		// The upstream declared the turn failed. Emit nothing: the frame the
		// client has to see is the host's error frame for the failed delivery,
		// and a synthesized "stop" here would dress that failure as a finished
		// answer. done also keeps a trailing [DONE] from closing the turn.
		s.done = true
		return nil
	default:
		return nil
	}
}

func (s *msgToChatState) chunk(delta map[string]any, finish string) stream.Event {
	return stream.Event{Data: mustJSON(s.fullChunk(delta, finish))}
}

func (s *msgToChatState) fullChunk(delta map[string]any, finish any) map[string]any {
	if !s.sentRole {
		// The chat wire names the turn's role on the first chunk that goes
		// out, whatever that chunk carries.
		delta["role"] = "assistant"
		s.sentRole = true
	}
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

// finishFromSource emits the chat terminal for a source that closed itself with
// the [DONE] sentinel rather than message_stop. Idempotent, and reached only
// from the source's own close — never from EOF, where the missing terminal IS
// the upstream's failure (see PipeMessagesToChat).
func (s *msgToChatState) finishFromSource() []stream.Event {
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
