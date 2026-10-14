package messagesresponses

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

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

// PipeMessagesToResponses folds an Anthropic Messages SSE stream into
// Responses events.
//
// The Responses client is told the turn finished only when the SOURCE said so:
// message_stop, or the [DONE] sentinel a relay dialect appends instead. A
// Messages source that ends with neither was cut — the dialect has no other
// legal ending — and synthesizing response.completed for it would tell the
// client it holds a finished turn while the host records a failed delivery for
// the same one (StreamResult carries it as Truncated). See PipeResponsesToChat
// for the worked case that made this the rule.
func PipeMessagesToResponses(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	// output_index 0 belongs to the merged message/reasoning item; tools are
	// announced from 1.
	st := &msgToResp{model: opts.Model, nextTool: 1}
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
		// terminal, so it closes the turn here. An EOF with no close at all
		// deliberately does not — see the doc comment.
		if stream.IsDone(ev.Data) {
			return write(st.finishFromSource())
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
	return n, nil
}

// PipeResponsesToMessages folds a Responses SSE stream into Anthropic Messages
// events.
//
// The Messages client is told the turn finished only when the SOURCE said so:
// response.completed, response.incomplete, response.failed, or the [DONE]
// sentinel a relay dialect appends instead. A Responses source that ends with
// none of those was cut — the dialect has no other legal ending — and
// synthesizing message_stop for it would tell the client it holds a finished
// turn while the host records a failed delivery for the same one (StreamResult
// carries it as Truncated). See PipeResponsesToChat for the worked case that
// made this the rule.
//
// The reference's own Responses iterator emits nothing further on [DONE], but
// the relay sentinel is still the source's close: this library's settlement
// facts count it as a terminal (stream.ClassifyEvent reports [DONE] as a
// completion fact, so StreamResult.Terminal is true for it), and the
// Responses→Chat fold closes on it too. Swallowing it here would leave the
// host booking a delivered turn while the Messages client's stream never
// closes — the exact host/client divergence the truncation rule exists to
// prevent. A typed response.failed closes the turn for the client the way the
// reference's adapter does — the failure itself still surfaces as
// StreamResult.InBandErr.
func PipeResponsesToMessages(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &respToMsg{model: opts.Model}
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
		// terminal, so it closes the turn here. An EOF with no close at all
		// deliberately does not — see the doc comment.
		if stream.IsDone(ev.Data) {
			return write(st.finishFromSource())
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
	return n, nil
}

// msgThinkingBlock mirrors one Anthropic thinking/redacted_thinking block for
// the completed snapshot's merged reasoning item: its share of the summary
// text plus the opaque bytes the next turn replays.
type msgThinkingBlock struct {
	redacted bool
	text     strings.Builder
	sig      string
	data     string
}

// blockMaps renders the accumulated thinking blocks the way the source spoke
// them, which is the array encrypted_content re-encodes.
func blockMaps(blocks []*msgThinkingBlock) []any {
	var out []any
	for _, b := range blocks {
		if b.redacted {
			m := map[string]any{"type": "redacted_thinking", "data": b.data}
			out = append(out, m)
			continue
		}
		m := map[string]any{"type": "thinking", "thinking": b.text.String()}
		if b.sig != "" {
			m["signature"] = b.sig
		}
		out = append(out, m)
	}
	return out
}

// msgToolItem is one streamed tool call, opened when its Anthropic block opens.
type msgToolItem struct {
	callID string
	name   string
	args   strings.Builder
	index  int // output_index; 0 is reserved for the merged message/reasoning item
}

type msgToResp struct {
	started bool
	closed  bool
	id      string
	model   string

	msgID string // minted with the prelude or at the first text delta; every message event references it
	rsID  string // minted at the first thinking delta; the reasoning events reference it

	// announced records that the stream's one index-0 item was decided — the
	// reference's bridge announces exactly one item off the first content
	// chunk (reasoning for a thinking delta, nothing for a tool chunk, the
	// message item otherwise) and never announces a second, so a thinking- or
	// tool-first stream's message item is only ever named by its own events.
	announced bool
	// reasoningActive/reasoningDone track the reasoning item's lifecycle: the
	// reference closes a reasoning stretch with its done trio at the first
	// following non-reasoning chunk, not at the end of the stream.
	reasoningActive bool
	reasoningDone   bool

	text     strings.Builder
	thinking []*msgThinkingBlock
	tools    []*msgToolItem
	blocks   map[int]*msgToolItem // content-block index → tool item
	nextTool int                  // next tool output_index, from 1

	seq      int
	eventSeq int

	inTok        int
	inRawTok     int
	cachedTok    int
	cacheWriteTk int
	outTok       int
	stop         string
}

// respEvent builds one Responses event. Every event carries a sequence_number
// that counts up across the stream, as the Responses SSE protocol requires.
func (s *msgToResp) respEvent(typ string, fields map[string]any) stream.Event {
	fields["type"] = typ
	fields["sequence_number"] = s.eventSeq
	s.eventSeq++
	return stream.Event{Event: typ, Data: mustJSON(fields)}
}

func (s *msgToResp) responseShell() map[string]any {
	// The shell the reference's prelude carries. Everything here is derivable
	// without the request or a clock; created_at is the one field the
	// reference fills from the wall clock, and this library's contract has
	// none (README: offline, no clock).
	return map[string]any{
		"id": s.id, "object": "response", "model": s.model, "status": "in_progress",
		"output":              []any{},
		"parallel_tool_calls": true,
		"tool_choice":         "auto",
		"tools":               []any{},
		"top_p":               1.0,
		"reasoning":           map[string]any{"effort": nil, "summary": nil},
		"store":               true,
	}
}

// openShells emits the stream's opening frame pair exactly once. The reference
// opens the Responses stream before any content is known; which index-0 item it
// announces is decided later, off the first content chunk.
func (s *msgToResp) openShells() []stream.Event {
	if s.started {
		return nil
	}
	s.started = true
	return []stream.Event{
		s.respEvent("response.created", map[string]any{"response": s.responseShell()}),
		s.respEvent("response.in_progress", map[string]any{"response": s.responseShell()}),
	}
}

// announce opens the stream's one index-0 item the way the reference's bridge
// does off the first content chunk: a thinking delta announces the merged
// reasoning item (its deltas then reference the rs id), anything else announces
// the message item together with its content part. A tool chunk announces
// nothing — markAnnounced covers that case: the tools announce themselves from
// index 1 and the message item is never announced after them.
func (s *msgToResp) announce(reasoningFirst bool) []stream.Event {
	if s.markAnnounced() {
		return nil
	}
	if reasoningFirst {
		s.rsID = fmt.Sprintf("rs_%d", s.seq)
		s.seq++
		// No summary member on the wire: the reference's event serializes it
		// as null, which the Responses wire drops for this item.
		return []stream.Event{s.respEvent("response.output_item.added", map[string]any{
			"output_index": 0,
			"item":         map[string]any{"id": s.rsID, "type": "reasoning", "status": "in_progress"},
		})}
	}
	return s.announceMessage()
}

// announceMessage announces the merged message item and its content part; used
// for a content-first stream and, at the close, for a stream that never carried
// content the bridge could announce off (an empty turn still gets its item).
func (s *msgToResp) announceMessage() []stream.Event {
	s.msgID = fmt.Sprintf("msg_%d", s.seq)
	s.seq++
	return []stream.Event{
		s.respEvent("response.output_item.added", map[string]any{
			"output_index": 0,
			"item": map[string]any{
				"id": s.msgID, "type": "message", "role": "assistant",
				"status": "in_progress", "content": []any{},
			},
		}),
		s.respEvent("response.content_part.added", map[string]any{
			"item_id": s.msgID, "output_index": 0, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		}),
	}
}

// markAnnounced latches the one announcement this stream gets and reports
// whether it had already been spent.
func (s *msgToResp) markAnnounced() bool {
	if s.announced {
		return true
	}
	s.announced = true
	return false
}

// reasoningText concatenates the turn's thinking text — the summary the
// reference's reasoning done events carry.
func (s *msgToResp) reasoningText() string {
	var b strings.Builder
	for _, t := range s.thinking {
		if !t.redacted {
			b.WriteString(t.text.String())
		}
	}
	return b.String()
}

// maybeReasoningDone closes an active reasoning stretch with the reference's
// done trio — summary text done, summary part done, the reasoning item done —
// at the first chunk that is not reasoning (text, a tool call, or the finish).
// The reference fires these mid-stream, before whatever event ended the
// reasoning; deferring them to the close would misorder them after later deltas.
func (s *msgToResp) maybeReasoningDone() []stream.Event {
	if !s.reasoningActive || s.reasoningDone {
		return nil
	}
	s.reasoningDone = true
	if s.rsID == "" {
		s.rsID = fmt.Sprintf("rs_%d", s.seq)
		s.seq++
	}
	text := s.reasoningText()
	return []stream.Event{
		s.respEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id": s.rsID, "output_index": 0, "summary_index": 0, "text": text,
		}),
		s.respEvent("response.reasoning_summary_part.done", map[string]any{
			"item_id": s.rsID, "output_index": 0, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": text},
		}),
		s.respEvent("response.output_item.done", map[string]any{
			"output_index": 0,
			"item": map[string]any{
				"id": s.rsID, "type": "reasoning",
				"summary": []any{map[string]any{"type": "summary_text", "text": text}},
			},
		}),
	}
}

// openTool announces a tool call the moment its Anthropic block opens, the way
// the reference's bridge does off the block's first chunk.
func (s *msgToResp) openTool(idx int, cb map[string]any) []stream.Event {
	t := &msgToolItem{
		callID: jsonx.GetString(cb, "id"),
		name:   jsonx.GetString(cb, "name"),
		index:  s.nextTool,
	}
	s.nextTool++
	if s.blocks == nil {
		s.blocks = map[int]*msgToolItem{}
	}
	s.blocks[idx] = t
	s.tools = append(s.tools, t)
	return []stream.Event{s.respEvent("response.output_item.added", map[string]any{
		"output_index": t.index,
		"item": map[string]any{
			"type": "function_call", "id": t.callID, "call_id": t.callID,
			"name": t.name, "status": "in_progress", "arguments": "",
		},
	})}
}

// finalArgs is the arguments a call closes with. Anthropic sends no
// input_json_delta for an empty input, and the reference's parser synthesizes
// "{}" there so the wire value stays parseable JSON.
func (t *msgToolItem) finalArgs() string {
	if a := t.args.String(); a != "" {
		return a
	}
	return "{}"
}

// chunkArgs splits one arguments delta into the ~10-character pieces the
// reference re-chunks tool deltas to (runes, so a piece never splits a
// multi-byte character).
func chunkArgs(part string) []string {
	runes := []rune(part)
	if len(runes) == 0 {
		return nil
	}
	var out []string
	for i := 0; i < len(runes); i += 10 {
		end := i + 10
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}

func (s *msgToResp) feed(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	var evs []stream.Event
	// The shells open on the first frame, whatever it is — which index-0 item
	// the stream announces is decided by the first content chunk below, not by
	// the opening frame. message_start defers its shells to the end of its own
	// case so they can carry the id and model it names, the way the reference
	// read-aheads one chunk before opening the Responses stream.
	if typ != "message_start" {
		evs = append(evs, s.openShells()...)
	}
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
				s.inRawTok = in
				s.cachedTok = cr
				s.cacheWriteTk = cw
			}
		}
		evs = append(evs, s.openShells()...)
	case "content_block_start":
		idx, _ := jsonx.Int(obj["index"])
		cb, _ := jsonx.AsMap(obj["content_block"])
		if cb == nil {
			break
		}
		switch jsonx.GetString(cb, "type") {
		case "thinking":
			s.thinking = append(s.thinking, &msgThinkingBlock{})
		case "redacted_thinking":
			s.thinking = append(s.thinking, &msgThinkingBlock{
				redacted: true, data: jsonx.GetString(cb, "data"),
			})
		case "tool_use", "server_tool_use":
			// A tool chunk is a reasoning-end trigger, and it announces nothing
			// at index 0: the reference's bridge spends its one announcement
			// without emitting an event, so the message item of a tool-first
			// stream is never announced.
			evs = append(evs, s.maybeReasoningDone()...)
			s.markAnnounced()
			// A server-side call the model made is still a call: the reference
			// folds server_tool_use through the same tool_call path as
			// tool_use, and dropping its block would lose the call.
			evs = append(evs, s.openTool(idx, cb)...)
		}
	case "content_block_delta":
		idx, _ := jsonx.Int(obj["index"])
		delta, _ := jsonx.AsMap(obj["delta"])
		if delta == nil {
			break
		}
		switch jsonx.GetString(delta, "type") {
		case "text_delta":
			evs = append(evs, s.maybeReasoningDone()...)
			evs = append(evs, s.announce(false)...)
			if s.msgID == "" {
				s.msgID = fmt.Sprintf("msg_%d", s.seq)
				s.seq++
			}
			text := jsonx.GetString(delta, "text")
			s.text.WriteString(text)
			evs = append(evs, s.respEvent("response.output_text.delta", map[string]any{
				"item_id": s.msgID, "output_index": 0, "content_index": 0, "delta": text,
			}))
		case "thinking_delta":
			thinking := jsonx.GetString(delta, "thinking")
			if len(s.thinking) == 0 {
				s.thinking = append(s.thinking, &msgThinkingBlock{})
			}
			s.thinking[len(s.thinking)-1].text.WriteString(thinking)
			s.reasoningActive = true
			evs = append(evs, s.announce(true)...)
			if s.rsID == "" {
				s.rsID = fmt.Sprintf("rs_%d", s.seq)
				s.seq++
			}
			evs = append(evs, s.respEvent("response.reasoning_summary_text.delta", map[string]any{
				"item_id": s.rsID, "output_index": 0, "summary_index": 0, "delta": thinking,
			}))
		case "signature_delta":
			if len(s.thinking) > 0 {
				s.thinking[len(s.thinking)-1].sig += jsonx.GetString(delta, "signature")
			}
		case "input_json_delta":
			t := s.blocks[idx]
			if t == nil {
				break
			}
			evs = append(evs, s.maybeReasoningDone()...)
			part := jsonx.GetString(delta, "partial_json")
			t.args.WriteString(part)
			for _, piece := range chunkArgs(part) {
				evs = append(evs, s.respEvent("response.function_call_arguments.delta", map[string]any{
					"item_id": t.callID, "output_index": t.index, "delta": piece,
				}))
			}
		}
	case "content_block_stop":
		// The reference defers every done event to the end of the stream, so a
		// block closing mid-stream produces nothing here.
	case "message_delta":
		// The finish chunk ends an active reasoning stretch too.
		evs = append(evs, s.maybeReasoningDone()...)
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
		evs = append(evs, s.maybeReasoningDone()...)
		evs = append(evs, s.complete()...)
	case "error":
		// The upstream declared the turn failed. Emit nothing: the frame the
		// client has to see is the host's error frame for the failed delivery,
		// and a synthesized terminal here would dress that failure as a
		// finished turn.
		s.closed = true
	}
	return evs
}

func (s *msgToResp) complete() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	// A stream that never carried content the bridge could announce off still
	// announces its message item — an empty turn's item arrives with its done
	// events, the shape the reference's default branch produces at the finish
	// chunk.
	if !s.announced {
		evs = append(evs, s.announceMessage()...)
		s.announced = true
	}
	status := stopReasonToStatus(s.stop)
	text := s.text.String()
	reasoningText, preserved := mergeThinking(blockMaps(s.thinking))
	if s.msgID == "" {
		s.msgID = fmt.Sprintf("msg_%d", s.seq)
		s.seq++
	}

	// Done events are deferred to the end of the stream, tools before the
	// message regardless of source order — the order the reference's bridge
	// emits them in.
	for _, t := range s.tools {
		// A call that streamed nothing still closes with "{}" — and the
		// reference streams that synthesized value as one last delta before
		// its done event.
		if t.args.String() == "" {
			evs = append(evs, s.respEvent("response.function_call_arguments.delta", map[string]any{
				"item_id": t.callID, "output_index": t.index, "delta": "{}",
			}))
		}
		args := t.finalArgs()
		evs = append(evs,
			s.respEvent("response.function_call_arguments.done", map[string]any{
				"item_id": t.callID, "output_index": t.index, "arguments": args,
			}),
			s.respEvent("response.output_item.done", map[string]any{
				"output_index": t.index,
				"item": map[string]any{
					"type": "function_call", "id": t.callID, "call_id": t.callID,
					"name": t.name, "status": "completed", "arguments": args,
				},
			}))
	}
	// A turn that reasoned folds its reasoning into the message part's done
	// event as a reasoning_text part; the item itself stays an output_text
	// message. Reasoning gets no done events of its own on this fold.
	part := any(map[string]any{"type": "output_text", "text": text, "annotations": []any{}})
	if reasoningText != "" {
		part = map[string]any{"type": "reasoning_text", "reasoning": reasoningText}
	}
	evs = append(evs,
		s.respEvent("response.output_text.done", map[string]any{
			"item_id": s.msgID, "output_index": 0, "content_index": 0, "text": text,
		}),
		s.respEvent("response.content_part.done", map[string]any{
			"item_id": s.msgID, "output_index": 0, "content_index": 0, "part": part,
		}),
		s.respEvent("response.output_item.done", map[string]any{
			"output_index": 0,
			"item": map[string]any{
				"id": s.msgID, "status": "completed", "type": "message", "role": "assistant",
				"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
			},
		}))

	// The completed snapshot carries the same ids the streamed events used.
	var output []any
	if reasoningText != "" || len(preserved) > 0 {
		if s.rsID == "" {
			s.rsID = fmt.Sprintf("rs_%d", s.seq)
			s.seq++
		}
		output = append(output, reasoningItem(reasoningText, preserved, s.rsID, status))
	}
	// The streaming snapshot keeps the message part's text even when empty:
	// the reference's chunk assembler holds content "" (not null), so the
	// empty string survives onto the wire here — unlike the non-streaming
	// fold, whose transform nulls empty content out.
	msgPart := map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
	output = append(output, map[string]any{
		"id": s.msgID, "type": "message", "role": "assistant", "status": status,
		"content": []any{msgPart},
	})
	for _, t := range s.tools {
		output = append(output, map[string]any{
			"type": "function_call", "id": t.callID, "call_id": t.callID,
			"name": t.name, "arguments": t.finalArgs(), "status": "completed",
		})
	}
	// The reference's calculate_usage folds cache_creation and cache_read into
	// prompt_tokens before the responses usage is built, and the terminal
	// event's usage always carries input_tokens_details — cached_tokens is
	// simply 0 without a cache.
	usage := map[string]any{
		"input_tokens":  s.inTok,
		"output_tokens": s.outTok,
		"total_tokens":  s.inTok + s.outTok,
		"input_tokens_details": map[string]any{
			"cached_tokens":      s.cachedTok,
			"text_tokens":        s.inRawTok,
			"cache_write_tokens": s.cacheWriteTk,
		},
	}
	resp := map[string]any{
		"id":     s.id,
		"object": "response",
		"model":  s.model,
		"status": status,
		"output": output,
		"usage":  usage,
	}
	// The reference's streaming bridge closes every turn with
	// response.completed — an unfinished one says so in response.status, not
	// in the event type.
	return append(evs, s.respEvent("response.completed", map[string]any{"response": resp}))
}

// finishFromSource completes the turn for a source that closed itself with the
// [DONE] sentinel rather than message_stop. Idempotent, and reached only from
// the source's own close — never from EOF, where the missing terminal IS the
// upstream's failure (see PipeMessagesToResponses).
func (s *msgToResp) finishFromSource() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	return s.complete()
}

type respToMsg struct {
	started    bool
	closed     bool
	model      string
	seq        int
	blockOpen  bool
	blockIndex int
	blockType  string
	blockItem  string // the Responses item id that owns the open block
}

func (s *respToMsg) feed(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	if resp, ok := jsonx.AsMap(obj["response"]); ok {
		// The model the wrapper was constructed with (the host's routed model)
		// wins; the response's own model is only a fallback when no host model
		// was supplied, which is the reference's adapter behavior.
		if s.model == "" {
			if m := jsonx.GetString(resp, "model"); m != "" {
				s.model = m
			}
		}
	}
	var evs []stream.Event
	// message_start rides the first frame whatever it is: the reference's
	// adapter opens lazily, so a source that skips response.created still
	// produces a legal Messages stream instead of one that starts mid-block.
	evs = append(evs, s.openMessageStart()...)
	switch typ {
	case "response.created":
		// The opening frame was emitted above; nothing else to do.
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
		if item == nil {
			break
		}
		if isToolCallItem(item) {
			evs = append(evs, s.close()...)
			// Anthropic numbers content blocks densely from 0: this call takes
			// the slot after the block that was open, and the first one lands
			// on 0. Checking blockType after assigning it would start every
			// stream at index 1.
			if s.blockType != "" {
				s.blockIndex++
			}
			s.blockOpen = true
			s.blockType = "tool_use"
			evs = append(evs, stream.Event{Event: "content_block_start", Data: mustJSON(map[string]any{
				"type": "content_block_start", "index": s.blockIndex,
				"content_block": map[string]any{
					"type": "tool_use", "id": jsonx.GetString(item, "call_id"),
					"name": jsonx.GetString(item, "name"), "input": map[string]any{},
				},
			})})
		} else if jsonx.GetString(item, "type") == "message" {
			// Each message item is its own text block — the text block opens
			// the moment the item is announced, and a second item's text must
			// not land in the first one's block, so an item boundary closes
			// the previous block and opens a fresh one.
			itemID := jsonx.GetString(item, "id")
			if s.blockOpen && s.blockType == "text" && s.blockItem == itemID {
				break
			}
			evs = append(evs, s.ensure("text", map[string]any{"type": "text", "text": ""})...)
			s.blockItem = itemID
		}
	case "response.output_item.done":
		// A message item's block ends at its own done event, the way the
		// reference closes it there — so even a stream cut before the
		// terminal leaves no block hanging open.
		item, _ := jsonx.AsMap(obj["item"])
		if item == nil {
			break
		}
		if jsonx.GetString(item, "type") == "message" &&
			s.blockOpen && s.blockType == "text" &&
			s.blockItem != "" && s.blockItem == jsonx.GetString(item, "id") {
			evs = append(evs, s.close()...)
		}
	case "response.reasoning_summary_text.delta":
		text := jsonx.String(obj["delta"])
		// The reference opens a thinking block only for a non-empty delta: an
		// empty delta with no block open produces nothing at all (an empty
		// delta on an already-open block still relays).
		if text == "" && !(s.blockOpen && s.blockType == "thinking") {
			break
		}
		// The thinking block opens with the fold's stand-in signature: an
		// unsigned block ("" or null) reads as "no signature" to Messages
		// clients, which then downgrade the thinking to text on replay and
		// break the next request's tool pairing — see internal/thinksig. The
		// value rides the content_block_start frame; no signature_delta
		// follows, and clients that seed the block from the start frame (pi,
		// the Anthropic SDKs) keep it as-is.
		evs = append(evs, s.ensure("thinking", map[string]any{"type": "thinking", "thinking": "", "signature": thinksig.Synthetic})...)
		evs = append(evs, stream.Event{Event: "content_block_delta", Data: mustJSON(map[string]any{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": text},
		})})
	case "response.completed", "response.incomplete", "response.failed":
		// A typed failure is still the source closing its own turn, and the
		// reference's adapter closes it exactly like a completed one:
		// message_delta + message_stop. The host still sees the failure —
		// StreamResult.InBandErr comes from the source frame, not from the
		// terminal the fold writes.
		evs = append(evs, s.close()...)
		stop := "end_turn"
		if typ == "response.incomplete" {
			stop = "max_tokens"
		}
		resp, _ := jsonx.AsMap(obj["response"])
		usage := any(nil)
		if resp != nil {
			if jsonx.GetString(resp, "status") == "incomplete" {
				stop = "max_tokens"
			}
			if out, ok := jsonx.AsSlice(resp["output"]); ok {
				for _, it := range out {
					if im, ok := jsonx.AsMap(it); ok && isToolCallItem(im) {
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
	case "error":
		// An untyped error frame is not a terminal the Responses dialect
		// defines, so it closes nothing here: the frame the client has to see
		// is the host's error frame for the failed delivery, and a synthesized
		// terminal would dress that failure as a finished turn.
		s.closed = true
	}
	return evs
}

// openMessageStart emits the Anthropic opening frame exactly once. The
// reference's adapter mints a fresh msg id and opens with the fields an
// Anthropic message carries before they are known: null stop_reason and
// stop_sequence, and an all-zero usage.
func (s *respToMsg) openMessageStart() []stream.Event {
	if s.started {
		return nil
	}
	s.started = true
	msg := map[string]any{
		"id": fmt.Sprintf("msg_%d", s.seq), "type": "message", "role": "assistant",
		"model": s.model, "content": []any{},
		"stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens": 0, "output_tokens": 0,
			"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
		},
	}
	s.seq++
	return []stream.Event{{Event: "message_start", Data: mustJSON(map[string]any{"type": "message_start", "message": msg})}}
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

// finishFromSource closes the turn for a source that closed itself with the
// [DONE] sentinel rather than response.completed. Idempotent, and reached only
// from the source's own close — never from EOF, where the missing terminal IS
// the upstream's failure (see PipeResponsesToMessages).
func (s *respToMsg) finishFromSource() []stream.Event {
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

// isToolCallItem reports whether a Responses output item names a call this
// fold relays. Only function_call does: the reference's Messages adapter
// ignores custom_tool_call items entirely, so folding one would invent a call
// the source never delivered to a Messages client.
func isToolCallItem(item map[string]any) bool {
	return jsonx.GetString(item, "type") == "function_call"
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
