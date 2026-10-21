// Package geminiresponses converts between the Responses dialect and the
// generateContent wire format. The reference has no direct path between the
// two — it composes the chat dialect in two hops — and this package
// translates directly, reproducing the composed semantics: request bodies
// fold input items and parameters onto the wire's contents and generation
// config, response bodies fold candidates back onto Responses output items,
// and streams fold frames both ways. The design rules live in
// the reference specification.
package geminiresponses

import (
	"encoding/json"
	"io"
	"sort"
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

// PipeGeminiToResponses folds a generateContent SSE stream into Responses
// events.
//
// The reference reaches this pair through its chat hop, and its bridge owns
// one quirk this fold used to reproduce and now deliberately does not: it
// announces exactly one output item off the first content frame (a tool frame
// spends the announcement without emitting one, and the frame that spends it
// loses its own delta), so the message item of a tool-first turn streamed its
// text as deltas addressing an item no output_item.added had ever declared —
// dropped by protocol-abiding clients — and a frame carrying both text and a
// call had to choose which content survived. This fold instead announces every
// item when it opens and lets a frame contribute all the content it carried.
//
// One reference quirk is kept: every turn closes with response.completed — an
// unfinished one says so in the response status rather than the event name.
//
// The turn is only closed when the source said so: a frame carrying a finish
// reason (or a blocked prompt) closes it with a terminal that states how the
// turn ended, a relay sentinel closes it with a plain completion (the source
// ended the turn and never said it failed), and an EOF that arrived with
// neither is a truncation this fold refuses to dress up as a completion.
func PipeGeminiToResponses(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &gemToResp{model: opts.Model}
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
			// The wire has no sentinel of its own, so this is a relay's close.
			// It is still the SOURCE's close, and the Responses dialect has
			// exactly one legal ending — closing the items while withholding
			// response.completed would leave the host booking a delivered turn
			// (StreamResult.Terminal) while the client's stream never closes.
			// So the terminal is written, stating a plain completion: the
			// source said the turn ended and never said why it failed.
			return write(st.close())
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
	return n, nil
}

// PipeResponsesToGemini folds a Responses SSE stream into generateContent
// frames: one bare data frame per content delta, no event names and no
// sentinel, because this wire has neither.
//
// Tool arguments are the one place the two dialects genuinely differ in kind:
// the wire takes a whole JSON object where the Responses stream sends string
// fragments. Fragments that do not yet parse are skipped — half an object is
// not a frame — and the last parseable snapshot is emitted before the
// terminal so the complete call always arrives.
//
// The terminal frame (finishReason with the usage riding the same frame) is
// written when the source closed itself: at its completed/incomplete event,
// or at a relay sentinel. An EOF with neither is a truncation, and no frame
// is invented for it.
func PipeResponsesToGemini(dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	st := &respToGem{}
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
			return write(st.closeFromSentinel())
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

// frameEvent renders one wire frame: a bare data line, no event name.
func frameEvent(body map[string]any) stream.Event {
	b, err := json.Marshal(body)
	if err != nil {
		b = []byte("{}")
	}
	return stream.Event{Data: string(b)}
}

// ---------------------------------------------------------------------------
// gemini -> responses
// ---------------------------------------------------------------------------

// streamTool is one call being assembled from the wire's frames.
type streamTool struct {
	id       string
	name     string
	args     string
	index    int
	declared bool
	streamed int
}

type gemToResp struct {
	started  bool
	closed   bool
	id       string
	model    string
	seq      int
	eventSeq int

	// msgID/msgIdx name the message item, minted when it opens — its first
	// text frame, or the close of a turn that never carried text. rsID/rsIdx
	// name the reasoning item, minted at its first thought frame. Every event
	// the item generates references the pair.
	msgID  string
	msgIdx int
	rsID   string
	rsIdx  int
	// nextIdx allocates the output_index every item takes when it opens, in
	// open order, so each item is announced as its own output item the moment
	// it opens — the shape the Responses wire itself uses.
	nextIdx int

	text      strings.Builder
	reasoning strings.Builder
	blocks    []any
	images    []string
	tools     []*streamTool
	byCallID  map[string]*streamTool

	finish        string
	usageMetadata any
	// grounded latches that some frame carried search grounding, whose
	// per-request fee replaces the tool-use tokens the usage fold would
	// otherwise add to the input count.
	grounded bool
	// lastSigs is the signature list of the last frame that carried any — the
	// value the reference's stream accumulator keeps (each frame states its
	// own list, and the last one stated wins).
	lastSigs []string
}

// respEvent builds one Responses event; every event carries a sequence number
// counting up across the stream, as the protocol requires.
func (s *gemToResp) respEvent(typ string, fields map[string]any) stream.Event {
	fields["type"] = typ
	fields["sequence_number"] = s.eventSeq
	s.eventSeq++
	return stream.Event{Event: typ, Data: marshalOrEmpty(fields)}
}

func (s *gemToResp) mint(prefix string) string {
	id := prefix + "_" + itoa(s.seq)
	s.seq++
	return id
}

// responseShell is the opening shell every Responses stream starts with. The
// members a strict client validates are always present; the defaults are the
// reference's own.
func (s *gemToResp) responseShell() map[string]any {
	return map[string]any{
		"id": s.id, "object": "response", "created_at": 0, "model": s.model,
		"output": []any{}, "parallel_tool_calls": true, "status": "in_progress",
		"tool_choice": "auto", "tools": []any{},
	}
}

// frameFacts is one frame's contribution in the vocabulary the reference's
// chat hop produces: the visible text, the reasoning text, the signed
// thinking blocks, the calls, the images and every part signature the frame
// carried (the reference reads them off any part, thought or not).
type frameFacts struct {
	text      string
	reasoning string
	blocks    []any
	calls     []parsedCall
	images    []string
	sigs      []string
}

func (s *gemToResp) feed(obj map[string]any) []stream.Event {
	if s.closed {
		return nil
	}
	if _, failed := obj["error"]; failed {
		// An in-band failure frame: the fold stays silent so the host's error
		// frame is the one the client sees, rather than dressing the failure
		// up as a finished turn.
		s.closed = true
		return nil
	}
	if id := jsonx.GetString(obj, "responseId"); id != "" && s.id == "" {
		s.id = id
	}
	var evs []stream.Event
	if !s.started {
		s.started = true
		evs = append(evs,
			s.respEvent("response.created", map[string]any{"response": s.responseShell()}),
			s.respEvent("response.in_progress", map[string]any{"response": s.responseShell()}),
		)
	}
	if um, ok := jsonx.AsMap(obj["usageMetadata"]); ok {
		s.usageMetadata = um
	}
	if pf, ok := jsonx.AsMap(obj["promptFeedback"]); ok && jsonx.GetString(pf, "blockReason") != "" {
		// A blocked prompt is the source's own close: the reference folds it
		// into one content-filtered turn, and so does this fold. Its blocked
		// frame carries an assistant-role delta, so the message item is
		// announced before the turn closes.
		if s.msgID == "" {
			s.msgID = s.mint("msg")
			s.msgIdx = s.nextIdx
			s.nextIdx++
			evs = append(evs, s.openMessageItem()...)
		}
		s.finish = "content_filter"
		return append(evs, s.close()...)
	}
	candidates, ok := jsonx.AsSlice(obj["candidates"])
	if !ok || len(candidates) == 0 {
		// A usage-only frame carries no content. The reference's bridge
		// mishandles it; this fold reads the usage it came for and emits
		// nothing, which is what the frame has to say.
		return evs
	}
	cand, _ := jsonx.AsMap(candidates[0])
	if !s.grounded && candidateHasGrounding(cand) {
		s.grounded = true
	}
	facts := parseFrame(cand)

	// Each item announces itself when it opens — the first frame carrying that
	// kind of content, whichever order the source chose. The frame's own deltas
	// still go out below; opening an item never costs its content.
	if facts.reasoning != "" && s.rsID == "" {
		s.rsID = s.mint("rs")
		s.rsIdx = s.nextIdx
		s.nextIdx++
		evs = append(evs, s.respEvent("response.output_item.added", map[string]any{
			"output_index": s.rsIdx,
			"item":         map[string]any{"id": s.rsID, "type": "reasoning", "status": "in_progress", "summary": nil},
		}))
	}
	if facts.text != "" && s.msgID == "" {
		s.msgID = s.mint("msg")
		s.msgIdx = s.nextIdx
		s.nextIdx++
		evs = append(evs, s.openMessageItem()...)
	}
	s.text.WriteString(facts.text)
	s.reasoning.WriteString(facts.reasoning)
	s.blocks = append(s.blocks, facts.blocks...)
	s.images = append(s.images, facts.images...)
	if len(facts.sigs) > 0 {
		s.lastSigs = facts.sigs
	}
	// Calls go out before the finish check: a frame that carries a functionCall
	// and its finishReason together is the wire's normal way to end a tool
	// turn, and returning on the finish used to drop such a call entirely —
	// as did a text delta arriving ahead of one in the same frame.
	if len(facts.calls) > 0 {
		for _, call := range facts.calls {
			evs = append(evs, s.toolEvents(call)...)
		}
	}
	if fr := jsonx.GetString(cand, "finishReason"); fr != "" {
		s.finish = mapFinishReason(fr)
		return append(evs, s.close()...)
	}
	if facts.reasoning != "" {
		evs = append(evs, s.respEvent("response.reasoning_summary_text.delta", map[string]any{
			"item_id": s.rsID, "output_index": s.rsIdx, "summary_index": 0, "delta": facts.reasoning,
		}))
	}
	if facts.text != "" {
		evs = append(evs, s.respEvent("response.output_text.delta", map[string]any{
			"item_id": s.msgID, "output_index": s.msgIdx, "content_index": 0, "delta": facts.text,
		}))
	}
	return evs
}

// openMessageItem announces the message item with its content part; the id and
// output_index are minted by the caller.
func (s *gemToResp) openMessageItem() []stream.Event {
	return []stream.Event{
		s.respEvent("response.output_item.added", map[string]any{
			"output_index": s.msgIdx,
			"item":         map[string]any{"id": s.msgID, "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}},
		}),
		s.respEvent("response.content_part.added", map[string]any{
			"item_id": s.msgID, "output_index": s.msgIdx, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		}),
	}
}

// toolEvents announces a call the moment its frame carries it and streams the
// arguments in the small pieces the reference re-chunks them to.
func (s *gemToResp) toolEvents(call parsedCall) []stream.Event {
	id := call.id
	if id == "" {
		id = s.mint("call")
	}
	if call.signature != "" {
		id += thoughtSeparator + call.signature
	}
	if s.byCallID == nil {
		s.byCallID = map[string]*streamTool{}
	}
	tool := s.byCallID[id]
	if tool == nil {
		tool = &streamTool{id: id, name: call.name, index: s.nextIdx}
		s.nextIdx++
		s.byCallID[id] = tool
		s.tools = append(s.tools, tool)
	}
	var evs []stream.Event
	if !tool.declared {
		tool.declared = true
		evs = append(evs, s.respEvent("response.output_item.added", map[string]any{
			"output_index": tool.index,
			"item": map[string]any{
				"id": tool.id, "type": "function_call", "status": "in_progress",
				"call_id": tool.id, "name": tool.name, "arguments": "",
			},
		}))
	}
	fresh := call.arguments
	if len(fresh) > 0 && tool.streamed < len(tool.args) {
		// A second frame for the same call continues its arguments.
		fresh = ""
	}
	tool.args += call.arguments
	for _, piece := range chunkRunes(fresh, 10) {
		tool.streamed += len(piece)
		evs = append(evs, s.respEvent("response.function_call_arguments.delta", map[string]any{
			"item_id": tool.id, "output_index": tool.index, "delta": piece,
		}))
	}
	return evs
}

// close winds the turn up: every tool closes, the message item closes with
// the whole text (the reasoning text rides its part when the turn reasoned),
// and the terminal event follows with the assembled response. It runs when
// the source stated its close — its own finishReason frame, or a relay
// sentinel — never for a bare EOF, which is a truncation the host owns.
func (s *gemToResp) close() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	// A turn that never opened an item still gets it — announced ahead of the
	// done events that close it, so those reference items the client has seen
	// declared.
	if s.msgID == "" {
		s.msgID = s.mint("msg")
		s.msgIdx = s.nextIdx
		s.nextIdx++
		evs = append(evs, s.openMessageItem()...)
	}
	if s.rsID == "" && s.reasoning.Len() > 0 {
		s.rsID = s.mint("rs")
		s.rsIdx = s.nextIdx
		s.nextIdx++
		evs = append(evs, s.respEvent("response.output_item.added", map[string]any{
			"output_index": s.rsIdx,
			"item":         map[string]any{"id": s.rsID, "type": "reasoning", "status": "in_progress", "summary": nil},
		}))
	}
	for _, tool := range s.tools {
		if !tool.declared {
			tool.declared = true
			evs = append(evs, s.respEvent("response.output_item.added", map[string]any{
				"output_index": tool.index,
				"item": map[string]any{
					"id": tool.id, "type": "function_call", "status": "in_progress",
					"call_id": tool.id, "name": tool.name, "arguments": "",
				},
			}))
		}
		if rest := tool.args[tool.streamed:]; rest != "" {
			tool.streamed = len(tool.args)
			for _, piece := range chunkRunes(rest, 10) {
				evs = append(evs, s.respEvent("response.function_call_arguments.delta", map[string]any{
					"item_id": tool.id, "output_index": tool.index, "delta": piece,
				}))
			}
		}
		evs = append(evs,
			s.respEvent("response.function_call_arguments.done", map[string]any{
				"item_id": tool.id, "output_index": tool.index, "arguments": tool.args,
			}),
			s.respEvent("response.output_item.done", map[string]any{
				"output_index": tool.index,
				"item": map[string]any{
					"id": tool.id, "type": "function_call", "status": "completed",
					"call_id": tool.id, "name": tool.name, "arguments": tool.args,
				},
			}))
	}
	text := s.text.String()
	reasoning := s.reasoning.String()
	part := any(map[string]any{"type": "output_text", "text": text, "annotations": []any{}})
	if reasoning != "" {
		// The reasoning text rides the message part's done event — the shape
		// the reference's bridge closes a reasoned turn with.
		part = map[string]any{"type": "reasoning_text", "reasoning": reasoning}
	}
	evs = append(evs,
		s.respEvent("response.output_text.done", map[string]any{
			"item_id": s.msgID, "output_index": s.msgIdx, "content_index": 0, "text": text,
		}),
		s.respEvent("response.content_part.done", map[string]any{
			"item_id": s.msgID, "output_index": s.msgIdx, "content_index": 0, "part": part,
		}),
		s.respEvent("response.output_item.done", map[string]any{
			"output_index": s.msgIdx,
			"item": map[string]any{
				"id": s.msgID, "type": "message", "role": "assistant", "status": "completed",
				"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
			},
		}))
	status := finishToStatus(s.finish)
	// The completed snapshot's output array is assembled in output_index order,
	// so the payload agrees with the per-item events about the turn's shape.
	type indexedItem struct {
		idx  int
		item any
	}
	var indexed []indexedItem
	// The reasoning item the stream terminal carries states its text alone:
	// the stream side of the reference's bridge never encodes the signed
	// thinking blocks into the item — that spelling belongs to the non-stream
	// fold. What a streamed signature becomes instead is the response-level
	// thought_signatures list below.
	if reasoning != "" {
		item := map[string]any{
			"id": s.rsID, "type": "reasoning", "role": "assistant", "status": status,
		}
		content := []any{}
		if reasoning != "" {
			content = append(content, map[string]any{"type": "output_text", "text": reasoning, "annotations": []any{}})
		}
		item["content"] = content
		indexed = append(indexed, indexedItem{s.rsIdx, item})
	}
	if len(s.images) > 0 {
		for _, img := range s.images {
			idx := s.nextIdx
			s.nextIdx++
			indexed = append(indexed, indexedItem{idx, map[string]any{
				"id": s.mint("ig"), "type": "image_generation_call",
				"status": imageGenerationStatus(s.finish), "result": img,
			}})
		}
	} else {
		// The terminal response's message part states the absence itself: the
		// reference's assembled item carries text null where its own
		// output_text.done event said "" — the event's value is an
		// accumulator artifact, the assembled item's is the fact (reproduced
		// from the reference's stream bridge during this audit).
		msgText := any(text)
		if s.text.Len() == 0 {
			msgText = nil
		}
		indexed = append(indexed, indexedItem{s.msgIdx, map[string]any{
			"id": s.msgID, "type": "message", "role": "assistant", "status": status,
			"content": []any{map[string]any{"type": "output_text", "text": msgText, "annotations": []any{}}},
		}})
	}
	for _, tool := range s.tools {
		indexed = append(indexed, indexedItem{tool.index, map[string]any{
			"id": tool.id, "type": "function_call", "status": "completed",
			"call_id": tool.id, "name": tool.name, "arguments": tool.args,
		}})
	}
	sort.Slice(indexed, func(i, j int) bool { return indexed[i].idx < indexed[j].idx })
	output := make([]any, len(indexed))
	for i, it := range indexed {
		output[i] = it.item
	}
	resp := map[string]any{
		"id": s.id, "object": "response", "created_at": 0, "model": s.model,
		"status": status, "output": output, "parallel_tool_calls": false,
		"text": map[string]any{}, "tool_choice": "auto", "tools": []any{},
		"usage": geminiUsageToResponses(s.usageMetadata, s.grounded),
	}
	if s.lastSigs != nil {
		// The accumulator the reference runs over its chat hop keeps the last
		// frame's signature list and states it on the response itself.
		resp["provider_specific_fields"] = map[string]any{"thought_signatures": s.lastSigs}
	}
	return append(evs, s.respEvent("response.completed", map[string]any{"response": resp}))
}

// parseFrame reads one candidate's parts the way the reference's chunk parser
// does: thought text accumulates as reasoning, plain text (and inline media
// other than images and audio, which fold in as data URLs) as content, calls
// and images each into their own lists.
func parseFrame(cand map[string]any) frameFacts {
	var f frameFacts
	content, _ := jsonx.AsMap(cand["content"])
	parts, _ := jsonx.AsSlice(content["parts"])
	for _, raw := range parts {
		pm, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if sig := partThoughtSignature(pm); sig != "" {
			f.sigs = append(f.sigs, sig)
		}
		text, hasText := pm["text"].(string)
		thought, _ := jsonx.Bool(pm["thought"])
		if hasText && thought {
			f.reasoning += text
			block := map[string]any{"type": "thinking", "thinking": text}
			if sig := partThoughtSignature(pm); sig != "" {
				block["signature"] = sig
			}
			f.blocks = append(f.blocks, block)
			continue
		}
		if hasText {
			if mime, _, ok := splitDataURL(text); ok && strings.HasPrefix(mime, "audio/") {
				continue
			}
			f.text += text
			continue
		}
		if fc, ok := jsonx.AsMap(firstNonNil(pm["functionCall"], pm["function_call"])); ok {
			f.calls = append(f.calls, parsedCall{
				id:        jsonx.GetString(fc, "id"),
				name:      jsonx.GetString(fc, "name"),
				arguments: dumpArgsJSON(fc["args"]),
				signature: partThoughtSignature(pm),
			})
			continue
		}
		idata, ok := jsonx.AsMap(firstNonNil(pm["inlineData"], pm["inline_data"]))
		if !ok {
			continue
		}
		mime := firstNonEmpty(jsonx.GetString(idata, "mimeType"), jsonx.GetString(idata, "mime_type"))
		data := jsonx.GetString(idata, "data")
		switch {
		case strings.HasPrefix(mime, "image/"):
			f.images = append(f.images, data)
		case strings.HasPrefix(mime, "audio/"):
			// Audio rides the audio object, which the Responses fold drops.
		default:
			f.text += "data:" + mime + ";base64," + data
		}
	}
	return f
}

// chunkRunes splits s into consecutive pieces of at most size runes. Runes,
// not bytes: a byte split could cut a multi-byte character in half.
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

func marshalOrEmpty(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// responses -> gemini
// ---------------------------------------------------------------------------

// respCall is one tool call whose argument fragments are being concatenated
// into the whole object the wire needs.
type respCall struct {
	name    string
	args    strings.Builder
	emitted string
}

type respToGem struct {
	closed bool
	calls  []*respCall
	byKey  map[string]*respCall
	last   *respCall
	usage  map[string]any
	// textDelivered is the visible text already emitted as text frames —
	// deltas plus whatever a done event had to make up. Done events state the
	// WHOLE text of their content part, so the fold needs the running total
	// to know what portion (if any) no delta ever carried.
	textDelivered string
}

// makeUpText returns the portion of a done event's full text that no frame
// delivered yet. A done event is not a delta: it states the whole text of
// its content part, so the reverse emits only the undelivered remainder —
// all of it when every delta was swallowed (a single-frame answer, the real
// endpoint's common short-answer shape), the suffix when deltas stopped
// early, and nothing when the text already went out. A full text the
// delivered one is not a prefix of does not go out at all: an unexplained
// mismatch is not something to compound.
func (s *respToGem) makeUpText(full string) string {
	if full == "" || full == s.textDelivered {
		return ""
	}
	if strings.HasPrefix(full, s.textDelivered) {
		return full[len(s.textDelivered):]
	}
	return ""
}

// messageItemText reads the visible text an assembled message item carries.
func messageItemText(item map[string]any) string {
	content, _ := jsonx.AsSlice(item["content"])
	var b strings.Builder
	for _, raw := range content {
		if pm, ok := jsonx.AsMap(raw); ok && jsonx.GetString(pm, "type") == "output_text" {
			b.WriteString(jsonx.GetString(pm, "text"))
		}
	}
	return b.String()
}

// assembledText reads the visible text a whole assembled response carries.
func assembledText(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	items, _ := jsonx.AsSlice(resp["output"])
	var b strings.Builder
	for _, raw := range items {
		if item, ok := jsonx.AsMap(raw); ok && jsonx.GetString(item, "type") == "message" {
			b.WriteString(messageItemText(item))
		}
	}
	return b.String()
}

// responseThoughtSignature reads the streamed signature the forward fold
// parks on the response itself — the only place a streamed signature lives.
func responseThoughtSignature(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	psf, ok := jsonx.AsMap(resp["provider_specific_fields"])
	if !ok {
		return ""
	}
	sigs, ok := jsonx.AsSlice(psf["thought_signatures"])
	if !ok || len(sigs) == 0 {
		return ""
	}
	return jsonx.String(sigs[0])
}

func (s *respToGem) feed(event string, obj map[string]any) []stream.Event {
	typ := event
	if typ == "" {
		typ = jsonx.GetString(obj, "type")
	}
	if resp, ok := jsonx.AsMap(obj["response"]); ok {
		if u, ok := jsonx.AsMap(resp["usage"]); ok && len(u) > 0 {
			s.usage = u
		}
	}
	switch typ {
	case "response.output_item.added":
		if item, ok := jsonx.AsMap(obj["item"]); ok {
			if jsonx.GetString(item, "type") == "function_call" {
				call := &respCall{name: jsonx.GetString(item, "name")}
				s.calls = append(s.calls, call)
				if s.byKey == nil {
					s.byKey = map[string]*respCall{}
				}
				if id := jsonx.GetString(obj, "item_id"); id != "" {
					s.byKey[id] = call
				}
				s.last = call
			}
		}
		return nil
	case "response.function_call_arguments.delta":
		if call := s.callFor(obj); call != nil {
			call.args.WriteString(jsonx.GetString(obj, "delta"))
			return s.maybeEmitCall(call)
		}
		return nil
	case "response.function_call_arguments.done":
		if call := s.callFor(obj); call != nil {
			if args := jsonx.GetString(obj, "arguments"); args != "" {
				call.args.Reset()
				call.args.WriteString(args)
			}
			return s.maybeEmitCall(call)
		}
		return nil
	case "response.output_item.done":
		if item, ok := jsonx.AsMap(obj["item"]); ok {
			switch jsonx.GetString(item, "type") {
			case "function_call":
				if call := s.callFor(obj); call != nil {
					if name := jsonx.GetString(item, "name"); name != "" {
						call.name = name
					}
					if args := jsonx.GetString(item, "arguments"); args != "" {
						call.args.Reset()
						call.args.WriteString(args)
					}
					return s.maybeEmitCall(call)
				}
			case "message":
				// A stream may close its message item without ever stating
				// the text in a done event; the assembled item is another
				// chance to make the text up.
				if missing := s.makeUpText(messageItemText(item)); missing != "" {
					s.textDelivered += missing
					return []stream.Event{s.contentFrame(map[string]any{"text": missing})}
				}
			}
		}
		return nil
	case "response.output_text.delta":
		delta := obj["delta"]
		if delta == nil {
			delta = obj["text"]
		}
		if text := jsonx.String(delta); text != "" {
			s.textDelivered += text
			return []stream.Event{s.contentFrame(map[string]any{"text": text})}
		}
		return nil
	case "response.output_text.done":
		// The done event states the whole text of its content part; the
		// undelivered remainder goes out here (see makeUpText).
		if missing := s.makeUpText(jsonx.GetString(obj, "text")); missing != "" {
			s.textDelivered += missing
			return []stream.Event{s.contentFrame(map[string]any{"text": missing})}
		}
		return nil
	case "response.content_part.done":
		// The closed part states the whole text too — the same make-up, so a
		// stream that skips the text done event still lands its text.
		if part, ok := jsonx.AsMap(obj["part"]); ok {
			if missing := s.makeUpText(jsonx.GetString(part, "text")); missing != "" {
				s.textDelivered += missing
				return []stream.Event{s.contentFrame(map[string]any{"text": missing})}
			}
		}
		return nil
	case "response.reasoning_summary_text.delta", "response.reasoning_summary.delta", "response.reasoning.delta":
		delta := obj["delta"]
		if delta == nil {
			delta = obj["text"]
		}
		if text := jsonx.String(delta); text != "" {
			return []stream.Event{s.contentFrame(map[string]any{"text": text, "thought": true})}
		}
		return nil
	case "response.completed", "response.incomplete":
		s.closed = true
		var evs []stream.Event
		for _, call := range s.calls {
			evs = append(evs, s.maybeEmitCall(call)...)
		}
		finishReason := "STOP"
		status := "completed"
		if typ == "response.incomplete" {
			status = "incomplete"
		}
		resp, _ := jsonx.AsMap(obj["response"])
		if resp != nil {
			if got := jsonx.GetString(resp, "status"); got != "" {
				status = got
			}
			if d, ok := jsonx.AsMap(resp["incomplete_details"]); ok && jsonx.GetString(d, "reason") == "content_filter" {
				status = "incomplete"
				finishReason = "SAFETY"
			}
		}
		if status == "incomplete" && finishReason == "STOP" {
			finishReason = "MAX_TOKENS"
		}
		// The assembled response is the last chance to recover text no delta
		// or done event carried — and the only place a streamed signature
		// lives: the forward fold parks it on the response itself, so the
		// reverse reads it back onto the terminal (the text part when one
		// closes the turn — the wire's own replay spelling — a bare part
		// otherwise; the reader collects a signature from any part).
		parts := []any{}
		if missing := s.makeUpText(assembledText(resp)); missing != "" {
			s.textDelivered += missing
			part := map[string]any{"text": missing}
			if sig := responseThoughtSignature(resp); sig != "" {
				part["thoughtSignature"] = sig
			}
			parts = append(parts, part)
		} else if sig := responseThoughtSignature(resp); sig != "" {
			parts = append(parts, map[string]any{"thoughtSignature": sig})
		}
		candidate := map[string]any{"content": map[string]any{"role": "model", "parts": parts}, "finishReason": finishReason, "index": 0}
		body := map[string]any{"candidates": []any{candidate}}
		if um := responsesUsageToMetadata(s.usage); len(um) > 0 {
			body["usageMetadata"] = um
		}
		return append(evs, frameEvent(body))
	case "response.failed", "error":
		// The source declared the turn failed; no frame is written for it, and
		// the host's error frame is what the client must see.
		s.closed = true
		return nil
	default:
		return nil
	}
}

// closeFromSentinel writes the terminal for a source that closed itself with
// a relay sentinel instead of a typed terminal — the only close this wire
// never states itself.
func (s *respToGem) closeFromSentinel() []stream.Event {
	if s.closed {
		return nil
	}
	s.closed = true
	var evs []stream.Event
	for _, call := range s.calls {
		evs = append(evs, s.maybeEmitCall(call)...)
	}
	return append(evs, frameEvent(map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": []any{}},
			"finishReason": "STOP", "index": 0,
		}},
	}))
}

// contentFrame wraps one part into a bare wire frame.
func (s *respToGem) contentFrame(part map[string]any) stream.Event {
	return frameEvent(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": []any{part}},
			"index":   0,
		}},
	})
}

// callFor resolves the call an arguments event belongs to, by item id, by
// output index, else whichever call is open.
func (s *respToGem) callFor(obj map[string]any) *respCall {
	if s.byKey != nil {
		if id := jsonx.GetString(obj, "item_id"); id != "" {
			if call := s.byKey[id]; call != nil {
				return call
			}
		}
		if idx, ok := jsonx.Int(obj["output_index"]); ok {
			if call := s.byKey["i"+itoa(idx)]; call != nil {
				return call
			}
		}
	}
	return s.last
}

// maybeEmitCall writes a function_call frame when the call's arguments have
// grown into a whole JSON object the fold has not sent yet. Fragments that do
// not parse are held back — half an object is not a frame.
func (s *respToGem) maybeEmitCall(call *respCall) []stream.Event {
	args := call.args.String()
	if args == "" {
		return nil
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		return nil
	}
	compact := compactJSON(parsed)
	if compact == call.emitted {
		return nil
	}
	call.emitted = compact
	// A frame is one response body, and the response side of this wire spells
	// its part keys in camelCase.
	return []stream.Event{s.contentFrame(map[string]any{
		"functionCall": map[string]any{"name": call.name, "args": parsed},
	})}
}
