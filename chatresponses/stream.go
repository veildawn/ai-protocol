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

// PipeResponsesToChat folds a Responses SSE stream into chat.completion.chunk
// events.
//
// The chat client is told the turn finished only when the SOURCE said so:
// response.completed, response.incomplete (a ceiling cut, reported as
// finish_reason "length"), or the [DONE] sentinel a relay dialect appends after
// the typed events. A Responses source that ends with none of those is a
// truncated turn — the dialect has no other legal ending — and inventing a
// "stop" for it would tell the client it holds a finished answer while the host
// running this pipe records a failed delivery for the same turn (StreamResult
// carries it as Truncated, and the host shows the client what it recorded).
// That disagreement is how a turn the client rendered as complete ended up
// logged as a 502 with the account cooled for it.
func PipeResponsesToChat(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &respToChatState{model: opts.Model, id: "chatcmpl-unknown"}
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
		return write(st.feed(ev.Event, obj))
	})
	if err != nil {
		return n, err
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
	// created carries the chat chunks' created timestamp onto the Responses
	// events, where created_at is required — a strict client (the OpenAI SDK's
	// Response model) rejects a payload without one. The reference stamps its
	// opening shell from the wall clock; this library has no clock by
	// contract, so the shell takes the source's timestamp too.
	created int

	items    []*chatRespItem
	nextIdx  int
	seq      int
	eventSeq int

	msg    *chatRespItem            // the one message item, opened lazily
	rs     *chatRespItem            // the one reasoning item, opened lazily
	fcs    map[string]*chatRespItem // tool-call key → item
	lastFC *chatRespItem            // fallback target for keyless arg fragments
	// pendingStop holds a finish_reason seen on a chunk that carried no usage.
	// Under stream_options.include_usage the usage arrives in a later
	// choices-less chunk, so the terminal event waits for it.
	pendingStop string
}

// respEvent builds one Responses event. Every event carries a sequence_number
// that counts up across the stream, as the Responses SSE protocol requires.
func (s *chatToRespState) respEvent(typ string, fields map[string]any) stream.Event {
	fields["type"] = typ
	fields["sequence_number"] = s.eventSeq
	s.eventSeq++
	return stream.Event{Event: typ, Data: mustJSON(fields)}
}

func (s *chatToRespState) responseShell() map[string]any {
	// output/parallel_tool_calls/tool_choice/tools are required members of the
	// shell a strict Responses client validates (the OpenAI SDK's
	// ResponseCreatedEvent rejects a shell without them); the defaults are the
	// reference's own opening-shell values.
	return map[string]any{
		"id":                  s.id,
		"object":              "response",
		"created_at":          s.created,
		"model":               s.model,
		"output":              []any{},
		"parallel_tool_calls": true,
		"status":              "in_progress",
		"tool_choice":         "auto",
		"tools":               []any{},
	}
}

func (s *chatToRespState) newItem(kind, prefix string) *chatRespItem {
	// Every item takes the next output_index in open order — the shape the
	// Responses wire itself uses, each item announced as its own output item
	// the moment it opens. The text kinds no longer share a reserved slot 0
	// with a one-shot announcement: a tool-first chunk used to burn that
	// announcement, and the message item's text then streamed as deltas
	// addressing an item no output_item.added had ever declared — dropped by
	// protocol-abiding clients, the model's text lost while usage reported.
	index := s.nextIdx
	s.nextIdx++
	it := &chatRespItem{kind: kind, id: fmt.Sprintf("%s_%d", prefix, s.seq), index: index}
	s.seq++
	s.items = append(s.items, it)
	return it
}

// openEvents emits the item's opening frame pair: output_item.added, then the
// part frame a message needs before its first delta. Each item announces
// itself exactly once, when it opens — whichever chunk opens it, in whatever
// order the source chose. (A reasoning item never carries a part frame at all;
// the reference's protocol emits none for it.)
func (s *chatToRespState) openEvents(it *chatRespItem) []stream.Event {
	var item map[string]any
	var partType string
	var partFields map[string]any
	switch it.kind {
	case "reasoning":
		// The added frame declares the summary slot null — the reference's own
		// opening shape — and fills it only on the closing frames.
		item = map[string]any{"id": it.id, "type": "reasoning", "status": "in_progress", "summary": nil}
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
		partType = "response.content_part.added"
		partFields = map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			// The part carries its annotations member even when empty — the
			// reference shapes it so, and a strict client (the OpenAI SDK's
			// ResponseOutputText) rejects the part without one.
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		}
	}
	// The events are built in emission order so their sequence numbers do too.
	evs := []stream.Event{s.respEvent("response.output_item.added", map[string]any{
		"output_index": it.index, "item": item,
	})}
	if partType != "" {
		evs = append(evs, s.respEvent(partType, partFields))
	}
	return evs
}

// doneEvents emits the item's closing frames; the item snapshot they carry is
// also what lands in response.completed's output array.
func (s *chatToRespState) doneEvents(it *chatRespItem) []stream.Event {
	var first stream.Event
	switch it.kind {
	case "function_call":
		first = s.respEvent("response.function_call_arguments.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "arguments": it.text.String(),
		})
	case "reasoning":
		first = s.respEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"text": it.text.String(),
		})
	default:
		first = s.respEvent("response.output_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			"text": it.text.String(),
		})
	}
	evs := []stream.Event{first}
	switch it.kind {
	case "reasoning":
		evs = append(evs, s.respEvent("response.reasoning_summary_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": it.text.String()},
		}))
	case "message":
		evs = append(evs, s.respEvent("response.content_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0,
			// Same annotations member as the part frames: the reference closes
			// the part with it, and a strict client rejects the done part
			// without one.
			"part": map[string]any{"type": "output_text", "text": it.text.String(), "annotations": []any{}},
		}))
	}
	return append(evs, s.respEvent("response.output_item.done", map[string]any{
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
			"content": []any{map[string]any{
				"type": "output_text", "text": it.text.String(),
				// The annotations member rides the part even when empty — the
				// reference shapes it so, and a strict client requires it.
				"annotations": []any{},
			}},
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
	idxKey, idKey := "", ""
	if idx, ok := jsonx.Int(tm["index"]); ok {
		idxKey = "i" + strconv.Itoa(idx)
	}
	if cid := jsonx.GetString(tm, "id"); cid != "" {
		idKey = "c" + cid
	}
	if idxKey == "" && idKey == "" {
		// A fragment carrying neither key continues whichever call is open —
		// the arguments are the call's tail, and dropping them the way the
		// reference does would lose it.
		if s.lastFC != nil {
			return s.lastFC, nil
		}
		idxKey = "i0"
	}
	if s.fcs == nil {
		s.fcs = map[string]*chatRespItem{}
	}
	// One call is addressable by either its chat slot index or its call id —
	// upstreams mix the two spellings across fragments of the same call, and
	// the reference resolves them through one index-to-id map. Both keys
	// therefore resolve (and register) onto the same item, rather than
	// splitting the arguments across two items.
	var it *chatRespItem
	for _, key := range []string{idxKey, idKey} {
		if key == "" {
			continue
		}
		if found := s.fcs[key]; found != nil {
			it = found
			break
		}
	}
	if it != nil {
		s.lastFC = it
		return it, nil
	}
	it = s.newItem("function_call", "fc")
	it.callID = jsonx.GetString(tm, "id")
	// The item id IS the call id on the reference's wire (it puts the chat
	// tool_call id in both fields); a minted id is the fallback for a fragment
	// that carries none, since the added frame still needs one.
	if it.callID != "" {
		it.id = it.callID
	}
	if fn, _ := jsonx.AsMap(tm["function"]); fn != nil {
		it.name = jsonx.GetString(fn, "name")
	}
	for _, key := range []string{idxKey, idKey} {
		if key != "" {
			s.fcs[key] = it
		}
	}
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
	if c, ok := jsonx.Int(obj["created"]); ok {
		s.created = c
	}
	var evs []stream.Event
	if !s.started {
		s.started = true
		evs = append(evs,
			s.respEvent("response.created", map[string]any{"response": s.responseShell()}),
			s.respEvent("response.in_progress", map[string]any{"response": s.responseShell()}),
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
				evs = append(evs, s.respEvent("response.reasoning_summary_text.delta", map[string]any{
					"item_id": it.id, "output_index": it.index, "summary_index": 0, "delta": rc,
				}))
			}
			if c := delta["content"]; c != nil {
				if text := jsonx.String(c); text != "" {
					it, open := s.ensure("message")
					evs = append(evs, open...)
					it.text.WriteString(text)
					evs = append(evs, s.respEvent("response.output_text.delta", map[string]any{
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
						// An argument fragment is re-sliced into ~10-character
						// deltas — the size the reference emits to match OpenAI's
						// own streaming (streaming_iterator chunk_size = 10).
						// Concatenation is unchanged; only the frame sizes differ.
						for _, piece := range chunkRunes(args, fcDeltaChunkRunes) {
							evs = append(evs, s.respEvent("response.function_call_arguments.delta", map[string]any{
								"item_id": it.id, "output_index": it.index, "delta": piece,
							}))
						}
					}
				}
			}
		}
		if fr := jsonx.GetString(ch, "finish_reason"); fr != "" && !s.closed {
			// Under stream_options.include_usage the usage rides a later
			// choices-less chunk, so the terminal event waits for it rather
			// than reporting zero tokens for a turn that has a real count.
			if obj["usage"] == nil {
				s.pendingStop = fr
				return evs
			}
			evs = append(evs, s.complete(obj["usage"], fr)...)
		}
		return evs
	}
	// A usage-only chunk closes the turn whose finish_reason was held for it.
	if choices, _ := jsonx.AsSlice(obj["choices"]); s.pendingStop != "" && len(choices) == 0 {
		return s.complete(obj["usage"], s.pendingStop)
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
		snap := it.snapshot()
		// The completed payload's items report the finish the chunks reported:
		// a ceiling cut leaves the message item incomplete in the snapshot the
		// reference assembles for response.completed, while the per-item done
		// events above keep "completed" as the reference's own streaming path
		// does. Function-call items stay completed either way.
		if it.kind == "message" {
			if st := finishToStatus(finish); st != "completed" {
				clone := jsonx.CloneMap(snap)
				clone["status"] = st
				snap = clone
			}
		}
		if it.kind == "reasoning" {
			// The completed snapshot's reasoning item takes the reference's
			// non-streaming shape — role, finish-derived status, the reasoning
			// text as an output_text part — while the per-item done event above
			// keeps the summary shape the reference's own streaming path emits.
			snap = map[string]any{
				"id":     it.id,
				"type":   "reasoning",
				"status": finishToStatus(finish),
				"role":   "assistant",
				"content": []any{map[string]any{
					"type": "output_text", "text": it.text.String(), "annotations": []any{},
				}},
			}
		}
		output = append(output, snap)
	}
	status := finishToStatus(finish)
	resp := map[string]any{
		"id":                  s.id,
		"object":              "response",
		"created_at":          s.created,
		"model":               s.model,
		"status":              status,
		"output":              output,
		"parallel_tool_calls": false,
		"text":                map[string]any{},
		"tool_choice":         "auto",
		"tools":               []any{},
		"usage":               chatUsageToResponses(usage),
	}
	typ := "response.completed"
	if status == "incomplete" {
		typ = "response.incomplete"
		resp["incomplete_details"] = map[string]any{"reason": incompleteReason(finish)}
	}
	return append(evs, s.respEvent(typ, map[string]any{"response": resp}))
}

func (s *chatToRespState) finish() []stream.Event {
	if s.closed || !s.started {
		return nil
	}
	// A held stop_reason whose usage never arrived still closes the turn.
	finish := "stop"
	if s.pendingStop != "" {
		finish = s.pendingStop
	}
	return s.complete(nil, finish)
}

type respToChatState struct {
	id    string
	model string
	done  bool
	// created carries the Responses response's created_at onto every chat
	// chunk, where `created` is required — a strict chat client (the OpenAI
	// SDK's ChatCompletionChunk) rejects a chunk without one. The reference
	// stamps its own chunks from the wall clock; this library has no clock by
	// contract, so the source's timestamp is mapped instead.
	created int
	// fcSlots maps a Responses output_index to the chat tool_calls slot that
	// item occupies. Chat addresses parallel calls by a dense 0..n-1 index, so
	// two function_call items must land on distinct slots — reusing one slot
	// makes a client merge them into a single call and lose all but the last
	// name. The mapping is per item, and its key falls back to the call id when
	// an event omits output_index.
	fcSlots map[string]int
}

// fcSlot resolves the chat tool_calls index for one Responses output item,
// assigning the next free slot the first time the item is seen.
func (s *respToChatState) fcSlot(obj map[string]any, item map[string]any) int {
	key := ""
	if idx, ok := jsonx.Int(obj["output_index"]); ok {
		key = "i" + strconv.Itoa(idx)
	} else if id := jsonx.GetString(obj, "item_id"); id != "" {
		key = "t" + id
	} else if item != nil {
		if id := jsonx.GetString(item, "id"); id != "" {
			key = "t" + id
		}
	}
	if key == "" {
		return 0
	}
	if s.fcSlots == nil {
		s.fcSlots = map[string]int{}
	}
	if slot, ok := s.fcSlots[key]; ok {
		return slot
	}
	slot := len(s.fcSlots)
	s.fcSlots[key] = slot
	return slot
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
		if c, ok := jsonx.Int(resp["created_at"]); ok {
			s.created = c
		}
	}
	switch typ {
	case "response.created":
		// The reference opens with a bare empty-content delta — no role — and
		// the chat role is implied by the turn itself.
		return []stream.Event{s.chunk(map[string]any{"content": ""}, nil)}
	case "response.output_text.delta":
		delta := obj["delta"]
		if delta == nil {
			delta = jsonx.GetString(obj, "text")
		}
		return []stream.Event{s.chunk(map[string]any{"content": jsonx.String(delta)}, nil)}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary.delta", "response.reasoning.delta":
		// Reasoning is model output the chat wire carries in its own field.
		// Dropping it leaves the client with an answer and no reasoning.
		delta := obj["delta"]
		if delta == nil {
			delta = jsonx.GetString(obj, "text")
		}
		// A multi-segment summary mirrors its summary_index into the chunk's
		// choice index, the way the reference folds those events.
		idx := 0
		if v, ok := jsonx.Int(obj["summary_index"]); ok {
			idx = v
		}
		return []stream.Event{s.chunkAt(idx, map[string]any{"reasoning_content": jsonx.String(delta)}, nil)}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		return []stream.Event{s.chunk(map[string]any{
			"tool_calls": []any{map[string]any{
				// The reference types every tool_calls entry, argument deltas
				// included, and a strict chat client reads the type off each
				// entry rather than remembering the opening one.
				"index":    s.fcSlot(obj, nil),
				"type":     "function",
				"function": map[string]any{"arguments": jsonx.String(obj["delta"])},
			}},
		}, nil)}
	case "response.output_item.added":
		item, _ := jsonx.AsMap(obj["item"])
		if item != nil && isToolCallItem(item) {
			return []stream.Event{s.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": s.fcSlot(obj, item),
					// The call id is the item's call_id, falling back to the
					// item id when the event carries none — an empty id breaks
					// the pairing the tool result will answer to.
					"id":   first(jsonx.GetString(item, "call_id"), jsonx.GetString(item, "id")),
					"type": "function",
					"function": map[string]any{
						"name":      jsonx.GetString(item, "name"),
						"arguments": "",
					},
				}},
			}, nil)}
		}
		return s.emptyChunk()
	case "response.completed", "response.incomplete":
		s.done = true
		resp, _ := jsonx.AsMap(obj["response"])
		// The event name is authoritative for a ceiling cut: a backend that
		// omits response.status would otherwise report "stop" for a turn the
		// request's max_output_tokens cut short.
		status := "completed"
		if typ == "response.incomplete" {
			status = "incomplete"
		} else if resp != nil {
			if got := jsonx.GetString(resp, "status"); got != "" {
				status = got
			}
		}
		detailReason := ""
		if resp != nil {
			if d, ok := jsonx.AsMap(resp["incomplete_details"]); ok {
				detailReason = jsonx.GetString(d, "reason")
			}
		}
		finish := statusToFinish(status, false, detailReason)
		// An incomplete turn keeps its own reason. A ceiling cut that happened
		// to contain a function_call is still a ceiling cut, and reporting it as
		// "tool_calls" tells the client the model chose to call a tool when it
		// actually ran out of room.
		if status != "incomplete" && resp != nil {
			if out, ok := jsonx.AsSlice(resp["output"]); ok {
				for _, it := range out {
					if im, ok := jsonx.AsMap(it); ok && isToolCallItem(im) {
						finish = "tool_calls"
					}
				}
			}
		}
		// The terminal delta is the reference's bare empty-content shape; the
		// finish_reason on the choice is what closes the turn.
		chunk := s.full(map[string]any{"content": ""}, finish)
		// Usage rides the terminal only when the source reported one: the
		// reference leaves it null for a terminal that carries none, rather
		// than inventing a zeros object.
		chunk["usage"] = nil
		if resp != nil {
			if u, ok := jsonx.AsMap(resp["usage"]); ok && len(u) > 0 {
				chunk["usage"] = responsesUsageToChat(u)
			}
		}
		return []stream.Event{{Data: mustJSON(chunk)}, {Data: "[DONE]"}}
	case "response.failed", "error":
		// The upstream declared the turn failed. Emit nothing: the frame the
		// client has to see is the host's error frame for the failed delivery,
		// and a synthesized "stop" here would dress that failure as a finished
		// answer. done also keeps a trailing [DONE] from closing the turn.
		s.done = true
		return nil
	default:
		// An event the fold has no mapping for still advances the stream the
		// way the reference does — a minimal empty-content chunk, so a chat
		// client tracking chunk cadence sees the same frame count. The
		// deliberate exceptions are a declared failure (silent, above) and the
		// reasoning aliases (folded, above): the reference discards the
		// reasoning text those carry, and this library does not drop model
		// output to match.
		return s.emptyChunk()
	}
}

// emptyChunk is the minimal delta chunk the reference emits for an event it
// maps no content onto.
func (s *respToChatState) emptyChunk() []stream.Event {
	return []stream.Event{s.chunk(map[string]any{"content": ""}, nil)}
}

// isToolCallItem reports whether a Responses output item is a tool call. A
// custom call carries its payload in `input` rather than `arguments`, but on
// the chat wire both are a tool_call.
func isToolCallItem(item map[string]any) bool {
	switch jsonx.GetString(item, "type") {
	case "function_call", "custom_tool_call":
		return true
	}
	return false
}

func (s *respToChatState) chunk(delta map[string]any, finish any) stream.Event {
	return stream.Event{Data: mustJSON(s.full(delta, finish))}
}

// chunkAt is chunk with an explicit choice index, for the frames whose index
// mirrors a source field rather than the single choice slot.
func (s *respToChatState) chunkAt(index int, delta map[string]any, finish any) stream.Event {
	full := s.full(delta, finish)
	choice := full["choices"].([]any)[0].(map[string]any)
	choice["index"] = index
	return stream.Event{Data: mustJSON(full)}
}

func (s *respToChatState) full(delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id":      s.id,
		"object":  "chat.completion.chunk",
		"created": s.created,
		"model":   s.model,
		"choices": []any{
			map[string]any{"index": 0, "delta": delta, "finish_reason": finish},
		},
	}
}

// finishFromSource emits the chat terminal for a source that closed itself
// with the [DONE] sentinel rather than a typed terminal frame. It is idempotent,
// and it is reached only from the source's own close — never from EOF, where the
// missing terminal IS the upstream's failure (see PipeResponsesToChat).
func (s *respToChatState) finishFromSource() []stream.Event {
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

// fcDeltaChunkRunes is the per-delta size the reference slices function-call
// argument fragments into.
const fcDeltaChunkRunes = 10

// chunkRunes splits s into consecutive pieces of at most `size` runes. Runes,
// not bytes: the reference slices by character, and a byte split could cut a
// multi-byte character in half and corrupt the delta.
func chunkRunes(s string, size int) []string {
	if size <= 0 || len(s) <= size {
		return []string{s}
	}
	runes := []rune(s)
	var out []string
	for i := 0; i < len(runes); i += size {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}
