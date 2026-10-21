package chatresponses

import (
	"encoding/json"
	"strings"

	"github.com/veildawn/ai-protocol/internal/jsonx"
)

// reasoningEfforts is the effort vocabulary the reference accepts on a chat
// reasoning_effort; a string outside it is dropped rather than forwarded for
// the upstream to reject the whole request over.
var reasoningEfforts = map[string]bool{
	"none": true, "minimal": true, "low": true, "medium": true,
	"high": true, "xhigh": true, "max": true,
}

func ChatToResponses(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	msgs, _ := jsonx.AsSlice(body["messages"])
	instructions, systemSeen, input := chatMessagesToResponsesInput(msgs)
	if len(input) == 0 {
		// The Responses API rejects an empty input, and a system-only request
		// would otherwise serialize `"input": null` — a body the upstream
		// refuses. Carry the instructions as a system-role input item instead.
		// The fold triggers on a system turn having been SEEN, not on the text
		// being non-empty: the reference folds on "instructions is not None",
		// and a system turn with empty content still sets instructions to a
		// present-but-empty string.
		if systemSeen {
			input = []any{map[string]any{
				"type":    "message",
				"role":    "system",
				"content": []any{map[string]any{"type": "input_text", "text": instructions}},
			}}
			instructions = ""
		} else {
			input = []any{}
		}
	}
	out["input"] = input
	if instructions != "" {
		out["instructions"] = instructions
	}
	if v, ok := body["max_completion_tokens"]; ok {
		out["max_output_tokens"] = v
	} else if v, ok := body["max_tokens"]; ok {
		out["max_output_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	copyIf(body, out, "user")
	copyIf(body, out, "parallel_tool_calls")
	copyIf(body, out, "metadata")
	if v, ok := body["tools"]; ok {
		out["tools"] = chatToolsToResponses(v)
	}
	if v, ok := body["tool_choice"]; ok {
		out["tool_choice"] = chatToolChoiceToResponses(v)
	}
	// The chat wire carries web search as a request param; the Responses wire
	// carries it as a built-in member of tools. The options fold into the
	// tool's own fields, and the tool joins whatever tools the request
	// already declared.
	if wso, ok := body["web_search_options"]; ok && wso != nil {
		tool := map[string]any{"type": "web_search"}
		if opts, ok := jsonx.AsMap(wso); ok {
			for k, v := range opts {
				tool[k] = v
			}
		}
		tools, _ := jsonx.AsSlice(out["tools"])
		out["tools"] = append(tools, tool)
	}
	if rf, ok := jsonx.AsMap(body["response_format"]); ok {
		if text := responseFormatToText(rf); text != nil {
			out["text"] = text
		}
	}
	if effort, ok := body["reasoning_effort"]; ok {
		switch t := effort.(type) {
		case string:
			// Only the effort vocabulary folds; a string outside it rides
			// nothing rather than a value the upstream rejects the request
			// for. "none" is a member — it asks for no reasoning, which is a
			// request, not an absence of one.
			if reasoningEfforts[t] {
				out["reasoning"] = map[string]any{"effort": t}
			}
		case map[string]any:
			out["reasoning"] = t
		}
	}
	if _, ok := body["stream"]; ok {
		if b, ok := jsonx.Bool(body["stream"]); ok && b {
			out["stream"] = true
		}
	}
	return out, nil
}

func ResponsesToChat(body map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if model := jsonx.GetString(body, "model"); model != "" {
		out["model"] = model
	}
	input := body["input"]
	msgs := responsesInputToChatMessages(input)
	// Top-level instructions become their own leading system message. A
	// system-role input item stays where the caller put it: hoisting it to the
	// front reorders the conversation the caller wrote.
	if inst := jsonx.GetString(body, "instructions"); strings.TrimSpace(inst) != "" {
		msgs = append([]any{map[string]any{"role": "system", "content": inst}}, msgs...)
	}
	out["messages"] = msgs
	if v, ok := body["max_output_tokens"]; ok {
		out["max_tokens"] = v
	}
	copyIf(body, out, "temperature")
	copyIf(body, out, "top_p")
	copyIf(body, out, "stream")
	copyIf(body, out, "user")
	copyIf(body, out, "parallel_tool_calls")
	// metadata is deliberately NOT forwarded: the reference's responses->chat
	// bridge takes metadata from its own call kwargs, never from the request
	// body, so a body-level metadata rides nothing onto the chat wire.
	if v, ok := body["tools"]; ok {
		tools, webSearchOptions := responsesToolsToChat(v)
		out["tools"] = tools
		if webSearchOptions != nil {
			out["web_search_options"] = webSearchOptions
		}
	}
	if v, ok := body["tool_choice"]; ok {
		out["tool_choice"] = responsesToolChoiceToChat(v)
	}
	// A tool_choice that names no tool is a pairing the chat wire rejects, so
	// when conversion left no tools — none declared, an empty list, or every
	// one dropped as Responses-only — both fold away together rather than ride
	// along. The emptiness test runs on the CONVERTED list, the list the
	// upstream would read.
	if ts, ok := out["tools"].([]any); !ok || len(ts) == 0 {
		delete(out, "tools")
		delete(out, "tool_choice")
	}
	if text, ok := jsonx.AsMap(body["text"]); ok {
		if rf := textFormatToResponseFormat(text); rf != nil {
			out["response_format"] = rf
		}
	}
	// reasoning_effort is a scalar string field on the chat wire: only the
	// effort value carries over. reasoning.summary (OpenAI Responses clients
	// always set one, e.g. "auto") and every other reasoning.* key have no
	// chat equivalent, so they drop here.
	if r, ok := jsonx.AsMap(body["reasoning"]); ok {
		if e, ok := r["effort"]; ok {
			out["reasoning_effort"] = e
		}
	}
	if b, ok := jsonx.Bool(body["stream"]); ok && b {
		out["stream"] = true
		out["stream_options"] = map[string]any{"include_usage": true}
	}
	// drop nil-ish
	for k, v := range out {
		if v == nil {
			delete(out, k)
		}
	}
	return out, nil
}

func copyIf(src, dst map[string]any, key string) {
	if v, ok := src[key]; ok && v != nil {
		dst[key] = v
	}
}

func chatMessagesToResponsesInput(msgs []any) (instructions string, systemSeen bool, input []any) {
	// A system turn's TEXT accumulates onto instructions the way the reference
	// does: appended with a space when what is held so far is truthy, and
	// REPLACING it otherwise. That makes an empty system turn still mark
	// instructions as present (it assigns the empty string), keeps a turn's
	// padding verbatim, and leaves the accumulated text falsy only while
	// nothing non-empty has arrived.
	inst := ""
	seen := false
	// A tool reply to a custom tool call is a custom_tool_call_output, so the
	// custom call ids are collected up front.
	customCalls := customToolCallIDs(msgs)
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		role := jsonx.GetString(m, "role")
		if role == "system" {
			if s, ok := m["content"].(string); ok {
				if seen && inst != "" {
					inst += " " + s
				} else {
					inst = s
				}
				seen = true
				continue
			}
			// Non-string system content is not instruction text: it stays a
			// system input item, parts and all — flattening it would drop an
			// image the prompt carried.
			input = append(input, map[string]any{
				"type":    "message",
				"role":    "system",
				"content": chatContentToResponsesParts(m["content"], "input_text"),
			})
			continue
		}
		if role == "developer" {
			// Only a system turn folds into instructions; a developer turn
			// rides the generic message branch — role preserved, text reading
			// as output_text. The item rides as long as content was stated:
			// a missing key reads as the empty string, and only an explicit
			// null carries nothing.
			if content, ok := m["content"]; !ok || content != nil {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "developer",
					"content": chatContentToResponsesParts(content, "output_text"),
				})
			}
			continue
		}
		if role == "tool" {
			callID := jsonx.GetString(m, "tool_call_id")
			if customCalls[callID] {
				// A reply to a custom call keeps its freeform payload: string
				// content rides as the string it is, anything else as parts.
				output := any(chatToolOutputParts(m))
				if content, ok := m["content"]; !ok {
					output = ""
				} else if s, ok := content.(string); ok {
					output = s
				}
				input = append(input, map[string]any{
					"type":    "custom_tool_call_output",
					"call_id": callID,
					"output":  output,
				})
				continue
			}
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  chatToolOutputParts(m),
			})
			continue
		}
		if role == "assistant" {
			// Order matters: a Responses replay expects the reasoning items
			// before the message and the calls they produced, so the assistant
			// turn is emitted reasoning -> message -> calls.
			input = append(input, reasoningInputItems(m)...)
			content, contentPresent := m["content"]
			if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok && len(tcs) > 0 {
				if chatContentTruthy(content, contentPresent) {
					input = append(input, map[string]any{
						"type":    "message",
						"role":    "assistant",
						"content": chatContentToResponsesParts(content, "output_text"),
					})
				}
				for _, tc := range tcs {
					tm, ok := jsonx.AsMap(tc)
					if !ok {
						continue
					}
					callID := jsonx.GetString(tm, "id")
					if fn, ok := jsonx.AsMap(tm["function"]); ok && len(fn) > 0 {
						call := map[string]any{"type": "function_call", "call_id": callID}
						// name and arguments ride only when the call states
						// them: a missing arguments is not "{}", and a missing
						// name is not "".
						if v, ok := fn["name"]; ok {
							call["name"] = v
						}
						if v, ok := fn["arguments"]; ok {
							call["arguments"] = v
						}
						input = append(input, call)
						continue
					}
					if custom, ok := jsonx.AsMap(tm["custom"]); ok {
						input = append(input, map[string]any{
							"type":    "custom_tool_call",
							"call_id": callID,
							"name":    jsonx.GetString(custom, "name"),
							"input":   jsonx.GetString(custom, "input"),
						})
						continue
					}
					input = append(input, map[string]any{"type": "function_call", "call_id": callID})
				}
				continue
			}
			// Without calls, the message rides as long as content was stated —
			// the empty string included (a missing key reads as ""), an image
			// list especially; only an explicit null carries nothing.
			if content, ok := m["content"]; !ok || content != nil {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": chatContentToResponsesParts(content, "output_text"),
				})
			}
			continue
		}
		// user (and any other role): same gate — content stated rides, an
		// explicit null rides nothing. The parts fold through the shared
		// converter, so a file part, a bare string item and an already-
		// Responses part all keep their shape instead of being dropped for
		// not being text or image_url.
		if content, ok := m["content"]; !ok || content != nil {
			input = append(input, map[string]any{
				"type":    "message",
				"role":    "user",
				"content": chatContentToResponsesParts(content, "input_text"),
			})
		}
	}
	// systemSeen reports whether any system turn supplied text — the
	// reference's "instructions is not None".
	return inst, seen, input
}

// customToolCallIDs collects the call ids of the conversation's custom tool
// calls: a tool message answering one is a custom_tool_call_output, not a
// function_call_output.
func customToolCallIDs(msgs []any) map[string]bool {
	ids := map[string]bool{}
	for _, item := range msgs {
		m, ok := jsonx.AsMap(item)
		if !ok || jsonx.GetString(m, "role") != "assistant" {
			continue
		}
		tcs, ok := jsonx.AsSlice(m["tool_calls"])
		if !ok {
			continue
		}
		for _, tc := range tcs {
			tm, ok := jsonx.AsMap(tc)
			if !ok {
				continue
			}
			if fn, ok := jsonx.AsMap(tm["function"]); ok && len(fn) > 0 {
				continue
			}
			if _, ok := jsonx.AsMap(tm["custom"]); ok {
				ids[jsonx.GetString(tm, "id")] = true
			}
		}
	}
	return ids
}

// reasoningInputItems replays an assistant turn's reasoning onto the Responses
// wire. Stored reasoning_items win — each carries the id the Responses API
// minted, which a replay needs; an item without one stays id-less rather than
// being given an invented id (the reference mints one from a memory address,
// which is not reproducible). thinking_blocks is the fallback for a turn that
// arrived over a surface carrying signed blocks. reasoning_content is
// deliberately not read: free text carries no id to replay.
func reasoningInputItems(m map[string]any) []any {
	if items, ok := jsonx.AsSlice(m["reasoning_items"]); ok && len(items) > 0 {
		var out []any
		for _, raw := range items {
			im, ok := jsonx.AsMap(raw)
			if !ok {
				continue
			}
			item := map[string]any{"type": "reasoning", "summary": []any{}}
			if id := jsonx.GetString(im, "id"); id != "" {
				item["id"] = id
			}
			if s, ok := jsonx.AsSlice(im["summary"]); ok && len(s) > 0 {
				item["summary"] = s
			}
			if ec := jsonx.GetString(im, "encrypted_content"); ec != "" {
				item["encrypted_content"] = ec
			}
			out = append(out, item)
		}
		if len(out) > 0 {
			return out
		}
	}
	if item := reasoningItemFromThinkingBlocks(m["thinking_blocks"]); item != nil {
		return []any{item}
	}
	return nil
}

// chatContentTruthy reports the truthiness the reference's `if content:` gate
// reads: a missing key defaults to the empty string, and empty strings, empty
// lists, null and zero are falsy.
func chatContentTruthy(v any, present bool) bool {
	if !present {
		return false
	}
	switch t := v.(type) {
	case nil:
		return false
	case string:
		return t != ""
	case []any:
		return len(t) > 0
	case map[string]any:
		return len(t) > 0
	case bool:
		return t
	case int:
		return t != 0
	case float64:
		return t != 0
	case json.Number:
		n, err := t.Float64()
		return err == nil && n != 0
	default:
		return true
	}
}

func appendInstruction(instructions []string, text string) []string {
	if text = strings.TrimSpace(text); text != "" {
		return append(instructions, text)
	}
	return instructions
}

// chatToolOutputParts folds a chat tool message's content into the
// function_call_output the Responses wire takes: always a part list — a bare
// string is one input_text part, an image or file part survives as its own
// part, and an explicit null content is an empty list rather than a part with
// nothing in it. A MISSING content key reads as the empty string, the way the
// reference's msg.get("content", "") default does.
func chatToolOutputParts(m map[string]any) []any {
	content, present := m["content"]
	if !present {
		return []any{inputTextPart("")}
	}
	if content == nil {
		return []any{}
	}
	return chatContentToResponsesParts(content, "input_text")
}

// chatContentToResponsesParts converts a chat content value into the part list
// the Responses wire takes: a bare string is one text part, a list converts
// part by part, and anything else stringifies into a text part. textType is
// the type a text piece takes — input_text where the conversation reads it,
// output_text on an assistant or developer turn.
func chatContentToResponsesParts(content any, textType string) []any {
	switch c := content.(type) {
	case nil:
		return []any{map[string]any{"type": textType, "text": ""}}
	case string:
		return []any{map[string]any{"type": textType, "text": c}}
	case []any:
		// An empty part list serializes as [], not null.
		parts := []any{}
		for _, p := range c {
			switch part := p.(type) {
			case string:
				parts = append(parts, map[string]any{"type": textType, "text": part})
			case map[string]any:
				if converted, ok := chatContentPartToResponses(part, textType); ok {
					parts = append(parts, converted)
				}
			}
		}
		return parts
	default:
		return []any{map[string]any{"type": textType, "text": jsonx.String(content)}}
	}
}

// chatContentPartToResponses maps one chat content part onto its Responses
// spelling, the way the reference's content converter does: text becomes a
// text part of textType, image_url becomes input_image with the detail the
// reference defaults to "auto", a file part keeps the bytes the input_file
// part takes, a part already in Responses form passes through verbatim, and
// anything else folds to text. A tool_reference part has no Responses
// equivalent and contributes nothing.
func chatContentPartToResponses(part map[string]any, textType string) (map[string]any, bool) {
	switch jsonx.GetString(part, "type") {
	case "text":
		return map[string]any{"type": textType, "text": jsonx.GetString(part, "text")}, true
	case "image_url":
		url := ""
		detail := "auto"
		if iu, ok := jsonx.AsMap(part["image_url"]); ok {
			url = jsonx.GetString(iu, "url")
			if d := jsonx.GetString(iu, "detail"); d != "" {
				detail = d
			}
		} else if s, ok := part["image_url"].(string); ok {
			url = s
		}
		return map[string]any{"type": "input_image", "image_url": url, "detail": detail}, true
	case "image":
		// The reference spreads the part over an input_image, so the part's
		// own keys win — its own type included.
		converted := map[string]any{"type": "input_image"}
		for k, v := range part {
			converted[k] = v
		}
		return converted, true
	case "file":
		converted := map[string]any{"type": "input_file"}
		if fv, ok := jsonx.AsMap(part["file"]); ok {
			for _, k := range []string{"file_id", "file_data", "filename"} {
				if v, ok := fv[k]; ok {
					converted[k] = v
				}
			}
		}
		return converted, true
	case "tool_reference":
		return nil, false
	case "", "input_text", "input_image", "output_text", "refusal", "input_file",
		"computer_screenshot", "summary_text":
		// A part with no type reads as input_text — already in Responses form,
		// so it passes through verbatim like the typed natives do.
		return part, true
	default:
		if t, ok := part["text"]; ok {
			return map[string]any{"type": textType, "text": jsonx.String(t)}, true
		}
		return map[string]any{"type": textType, "text": jsonx.String(part)}, true
	}
}

func inputTextPart(text string) map[string]any {
	return map[string]any{"type": "input_text", "text": text}
}

func responsesInputToChatMessages(input any) []any {
	if s, ok := input.(string); ok {
		return []any{map[string]any{"role": "user", "content": s}}
	}
	list, ok := jsonx.AsSlice(input)
	if !ok {
		return []any{}
	}
	var msgs []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		typ := jsonx.GetString(m, "type")
		switch typ {
		case "function_call_output", "custom_tool_call_output", "web_search_call",
			"computer_call_output", "tool_result":
			// A tool message must name the call it answers; without one the
			// upstream rejects the whole request, so an item with no call id
			// contributes nothing rather than an invalid turn. Every output
			// shape folds the same way — a web_search_call carries no output
			// and reads as an empty tool message, a tool_result is the
			// Anthropic/MCP spelling of the same answer.
			callID := jsonx.GetString(m, "call_id")
			if callID == "" {
				continue
			}
			msgs = append(msgs, map[string]any{
				"role":         "tool",
				"tool_call_id": callID,
				"content":      functionCallOutputContent(m["output"]),
			})
		case "function_call", "custom_tool_call":
			// A custom tool call stores its freeform payload in `input`; the
			// chat wire reads it as arguments, wrapped the way the reference
			// wraps it (input "boston" -> {"content": "boston"}).
			callID := jsonx.GetString(m, "call_id")
			if callID == "" {
				callID = jsonx.GetString(m, "id")
			}
			name := jsonx.GetString(m, "name")
			args := jsonx.GetString(m, "arguments")
			if typ == "custom_tool_call" {
				args = ""
				if in := jsonx.GetString(m, "input"); in != "" {
					if b, err := json.Marshal(in); err == nil {
						args = `{"content": ` + string(b) + `}`
					}
				}
			}
			call := map[string]any{
				"id":   callID,
				"type": "function",
				"function": map[string]any{
					"name":      name,
					"arguments": args,
				},
			}
			// The call belongs to the assistant turn it was emitted in. Folding
			// it into a separate assistant message splits one turn in two, and
			// a strict upstream rejects a replayed turn whose reasoning and
			// tool_calls are not carried by the same assistant message.
			if last, ok := lastAssistant(msgs); ok {
				if existing, ok := jsonx.AsSlice(last["tool_calls"]); ok {
					last["tool_calls"] = append(existing, call)
				} else {
					last["tool_calls"] = []any{call}
				}
				continue
			}
			msgs = append(msgs, map[string]any{
				"role":       "assistant",
				"tool_calls": []any{call},
			})
		case "reasoning":
			// The item replays as reasoning_content plus, when the reference
			// itself serialized signed blocks into encrypted_content, the
			// decoded thinking_blocks. An item with neither rides nothing.
			text := reasoningText(m)
			blocks := thinkingBlocksFromEncrypted(m)
			if text == "" && blocks == nil {
				continue
			}
			// A reasoning item joins the assistant turn it PRECEDES: it merges
			// onto a trailing empty shell (an earlier reasoning of the same
			// turn), but one that lands after a turn that already spoke stays
			// its own trailing assistant — the reference merges forward only.
			if last, ok := lastAssistant(msgs); ok && isEmptyAssistantShell(last) {
				if text != "" {
					last["reasoning_content"] = joinReasoning(jsonx.GetString(last, "reasoning_content"), text)
				}
				if blocks != nil {
					if existing, ok := jsonx.AsSlice(last["thinking_blocks"]); ok {
						last["thinking_blocks"] = append(blocks, existing...)
					} else {
						last["thinking_blocks"] = blocks
					}
				}
				continue
			}
			msg := map[string]any{"role": "assistant", "content": nil}
			if text != "" {
				msg["reasoning_content"] = text
			}
			if blocks != nil {
				msg["thinking_blocks"] = blocks
			}
			msgs = append(msgs, msg)
		case "message", "":
			role := jsonx.GetString(m, "role")
			if role == "" {
				role = "user"
			}
			// A system or developer item keeps its place, and its own role —
			// the reference forwards the role verbatim rather than rewriting
			// developer to system — and its content folds through the same
			// per-part conversion any message item takes: blocks stay blocks,
			// never flattened into one string. An item with no content at all
			// contributes nothing.
			if role == "system" || role == "developer" {
				if content, ok := m["content"]; ok && content != nil {
					msgs = append(msgs, map[string]any{"role": role, "content": partsToChatContent(content, role)})
				}
				continue
			}
			content := partsToChatContent(m["content"], role)
			// An assistant message whose text was split from its own reasoning
			// rejoins it. Only a message the previous turn left EMPTY is
			// absorbed: a second message with text of its own is a distinct
			// turn and dropping it would lose content.
			if role == "assistant" {
				if last, ok := lastAssistant(msgs); ok && isEmptyAssistantContent(last["content"]) {
					last["content"] = content
					continue
				}
			}
			msg := map[string]any{"role": role, "content": content}
			msgs = append(msgs, msg)
		}
	}
	return msgs
}

// lastAssistant reports the trailing message when it is an assistant message —
// the message a following function_call, reasoning item or split text belongs to.
func lastAssistant(msgs []any) (map[string]any, bool) {
	if len(msgs) == 0 {
		return nil, false
	}
	last, ok := msgs[len(msgs)-1].(map[string]any)
	if !ok || jsonx.GetString(last, "role") != "assistant" {
		return nil, false
	}
	return last, true
}

// joinReasoning appends a later reasoning item to the text already carried, so
// several items in one turn read as one block rather than being concatenated
// without a separator.
func joinReasoning(prev, next string) string {
	if strings.TrimSpace(prev) == "" {
		return next
	}
	return prev + "\n" + next
}

// isEmptyAssistantShell reports whether a trailing assistant message is an
// empty turn a following reasoning item belongs to: no stated content and no
// tool calls of its own.
func isEmptyAssistantShell(m map[string]any) bool {
	if tcs, ok := jsonx.AsSlice(m["tool_calls"]); ok && len(tcs) > 0 {
		return false
	}
	return isEmptyAssistantContent(m["content"])
}

// isEmptyAssistantContent reports content that carries nothing: null, an
// empty string, or an empty part list.
func isEmptyAssistantContent(v any) bool {
	switch c := v.(type) {
	case nil:
		return true
	case string:
		return c == ""
	case []any:
		return len(c) == 0
	default:
		return false
	}
}

// partsToChatContent converts a Responses message item's content onto the chat
// wire the way the reference does, part by part: a bare string stays a string,
// a text part becomes a chat text block (never joined into one string), an
// image part carries the detail it stated (defaulting to auto), a file part
// rides the chat file object, and an encrypted_content part reads as its text.
func partsToChatContent(content any, role string) any {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return content
	}
	var parts []any
	for _, p := range list {
		switch part := p.(type) {
		case string:
			parts = append(parts, part)
		case map[string]any:
			parts = append(parts, responsesPartToChatPart(part)...)
		}
	}
	return parts
}

// responsesPartToChatPart maps one Responses content part onto its chat
// spelling, the way the reference's content converter does. A part with a null
// text contributes nothing.
func responsesPartToChatPart(part map[string]any) []any {
	switch jsonx.GetString(part, "type") {
	case "input_file":
		// A file rides the chat file object: the id (the url falling back to
		// it) and the bytes it carries, nothing else.
		file := map[string]any{}
		if fid := first(jsonx.GetString(part, "file_id"), jsonx.GetString(part, "file_url")); fid != "" {
			file["file_id"] = fid
		}
		if fd, ok := part["file_data"]; ok && fd != nil && fd != "" {
			file["file_data"] = fd
		}
		out := map[string]any{"type": "file", "file": file}
		if cc, ok := part["cache_control"]; ok {
			out["cache_control"] = cc
		}
		return []any{out}
	case "input_image":
		url := jsonx.GetString(part, "image_url")
		if iu, ok := jsonx.AsMap(part["image_url"]); ok {
			url = jsonx.GetString(iu, "url")
		}
		detail := jsonx.GetString(part, "detail")
		if iu, ok := jsonx.AsMap(part["image_url"]); ok && detail == "" {
			detail = jsonx.GetString(iu, "detail")
		}
		if detail == "" {
			detail = "auto"
		}
		out := map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": url, "detail": detail},
		}
		if cc, ok := part["cache_control"]; ok {
			out["cache_control"] = cc
		}
		return []any{out}
	case "encrypted_content":
		if v, ok := part["encrypted_content"]; ok && v != nil {
			return []any{map[string]any{"type": "text", "text": jsonx.String(v)}}
		}
		return nil
	default:
		text, ok := part["text"]
		if !ok || text == nil {
			return nil
		}
		out := map[string]any{"type": chatContentPartType(jsonx.GetString(part, "type")), "text": text}
		if cc, ok := part["cache_control"]; ok {
			out["cache_control"] = cc
		}
		return []any{out}
	}
}

// chatContentPartType maps a Responses content part type onto the chat
// vocabulary: the input_ prefix drops (input_text -> text), an input_audio
// keeps its wire name, and anything the chat side has no spelling for reads
// as plain text.
func chatContentPartType(t string) string {
	valid := map[string]bool{
		"text": true, "image_url": true, "input_audio": true, "audio_url": true,
		"document": true, "guarded_text": true, "video_url": true, "file": true,
	}
	if rest, ok := strings.CutPrefix(t, "input_"); ok {
		if valid[rest] {
			return rest
		}
		if rest == "audio" {
			return "input_audio"
		}
		return "text"
	}
	if t == "tool_result" || t == "output_text" {
		return "text"
	}
	if valid[t] {
		return t
	}
	return "text"
}

func partsToText(content any, types ...string) string {
	if s, ok := content.(string); ok {
		return s
	}
	list, ok := jsonx.AsSlice(content)
	if !ok {
		return jsonx.String(content)
	}
	want := map[string]struct{}{}
	for _, t := range types {
		want[t] = struct{}{}
	}
	var parts []string
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		if _, ok := want[jsonx.GetString(pm, "type")]; ok || jsonx.GetString(pm, "type") == "" {
			parts = appendInstruction(parts, jsonx.GetString(pm, "text"))
		}
	}
	return strings.Join(parts, "\n")
}

// functionCallOutputContent renders a Responses function_call_output for the
// chat wire. A plain string output stays a string, and any other JSON value is
// re-encoded so the bytes stay parseable rather than rendering as Go syntax. A
// part list becomes a content-part array when it carries media — flattening an
// image to its JSON would lose it — and a joined string when it is text only.
// functionCallOutputContent renders a Responses function_call_output for the
// chat wire the way the reference normalizes it: a plain string output stays
// a string, and a part list becomes text/image chat blocks — texts joined with
// no separator, images kept as blocks — while non-dict parts are skipped
// rather than stringified. A list nothing matched falls back to the JSON of
// the whole value, so an empty list reads as "[]" rather than nothing.
func functionCallOutputContent(v any) any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		if s, ok := v.(string); ok {
			return s
		}
		if v == nil {
			return ""
		}
		if b, err := jsonx.Marshal(v); err == nil {
			return string(b)
		}
		return ""
	}
	var texts []string
	var parts []any
	hasMedia := false
	for _, p := range list {
		pm, ok := jsonx.AsMap(p)
		if !ok {
			continue
		}
		switch jsonx.GetString(pm, "type") {
		case "input_text", "output_text", "text":
			if t := jsonx.GetString(pm, "text"); t != "" {
				texts = append(texts, t)
				parts = append(parts, map[string]any{"type": "text", "text": t})
			}
		case "input_image", "image_url":
			url := jsonx.GetString(pm, "image_url")
			if iu, ok := jsonx.AsMap(pm["image_url"]); ok {
				url = jsonx.GetString(iu, "url")
			}
			if url != "" {
				hasMedia = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
		case "input_file":
			// The chat wire has no file part on a tool message; the bytes ride
			// a data URL, which is what a chat upstream reads for a PDF.
			if url := inputFileToDataURL(pm); url != "" {
				hasMedia = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
		}
	}
	if hasMedia {
		return parts
	}
	if len(texts) > 0 {
		return strings.Join(texts, "")
	}
	if b, err := jsonx.Marshal(v); err == nil {
		return string(b)
	}
	return ""
}

// inputFileToDataURL renders a Responses input_file part as a data URL for the
// chat wire, which carries a file's bytes as an image_url data URL.
func inputFileToDataURL(pm map[string]any) string {
	if data := jsonx.GetString(pm, "file_data"); data != "" {
		return data
	}
	return jsonx.GetString(pm, "file_url")
}

// reasoningItemFromThinkingBlocks builds a Responses reasoning item from a chat
// message's thinking_blocks, the fallback when reasoning_content is absent. A
// redacted block carries no readable text, so it contributes no summary part.
func reasoningItemFromThinkingBlocks(v any) map[string]any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var summary []any
	for _, raw := range list {
		bm, ok := jsonx.AsMap(raw)
		if !ok || jsonx.GetString(bm, "type") != "thinking" {
			continue
		}
		if t := jsonx.GetString(bm, "thinking"); t != "" {
			summary = append(summary, map[string]any{"type": "summary_text", "text": t})
		}
	}
	if len(summary) == 0 {
		return nil
	}
	return map[string]any{"type": "reasoning", "summary": summary}
}

// reasoningText reads the plaintext a Responses reasoning item carries, joining
// its summary parts the way the reference does.
// reasoningText reads the plaintext a Responses reasoning item carries:
// string content verbatim (a whitespace-only one reads as absent), a block
// content joined with newlines (opaque blocks contribute nothing), and the
// summary parts as the fallback — the order the reference reads them in.
func reasoningText(m map[string]any) string {
	if s, ok := m["content"].(string); ok {
		if strings.TrimSpace(s) != "" {
			return s
		}
	} else if list, ok := jsonx.AsSlice(m["content"]); ok {
		var parts []string
		for _, b := range list {
			bm, ok := jsonx.AsMap(b)
			if !ok {
				continue
			}
			switch jsonx.GetString(bm, "type") {
			case "encrypted_content", "redacted_thinking":
				continue
			}
			if t, ok := bm["text"].(string); ok && strings.TrimSpace(t) != "" {
				parts = append(parts, strings.TrimSpace(t))
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	if list, ok := jsonx.AsSlice(m["summary"]); ok {
		var parts []string
		for _, p := range list {
			if pm, ok := jsonx.AsMap(p); ok {
				if t := strings.TrimSpace(jsonx.GetString(pm, "text")); t != "" {
					parts = append(parts, t)
				}
			} else if s, ok := p.(string); ok && strings.TrimSpace(s) != "" {
				parts = append(parts, strings.TrimSpace(s))
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// thinkingBlocksFromEncrypted decodes an encrypted_content the reference
// itself wrote — a JSON array of the provider's signed thinking blocks — back
// into the blocks a replay needs. Anything that is not that (a genuinely
// opaque OpenAI blob, a non-array, invalid JSON) decodes to nothing rather
// than being forwarded as garbage.
func thinkingBlocksFromEncrypted(m map[string]any) []any {
	ec := jsonx.GetString(m, "encrypted_content")
	if strings.TrimSpace(ec) == "" {
		return nil
	}
	var decoded []any
	if err := json.Unmarshal([]byte(ec), &decoded); err != nil {
		return nil
	}
	var blocks []any
	for _, b := range decoded {
		bm, ok := jsonx.AsMap(b)
		if !ok {
			continue
		}
		// A block is only worth replaying when the provider can verify it: a
		// thinking block needs its signature, a redacted one its opaque data.
		switch jsonx.GetString(bm, "type") {
		case "thinking":
			if jsonx.GetString(bm, "signature") != "" {
				blocks = append(blocks, bm)
			}
		case "redacted_thinking":
			if jsonx.GetString(bm, "data") != "" {
				blocks = append(blocks, bm)
			}
		}
	}
	return blocks
}

func contentToPlain(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, p := range c {
			if pm, ok := jsonx.AsMap(p); ok {
				parts = append(parts, jsonx.GetString(pm, "text"))
			}
		}
		return strings.Join(parts, "")
	default:
		if c == nil {
			return ""
		}
		return jsonx.String(c)
	}
}

func chatToolsToResponses(v any) []any {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil
	}
	var out []any
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		if jsonx.GetString(m, "type") == "function" {
			fn, _ := jsonx.AsMap(m["function"])
			tool := map[string]any{"type": "function"}
			if fn != nil {
				tool["name"] = jsonx.GetString(fn, "name")
				// The reference bridge always carries these three — a chat
				// tool that stated none sends them as null rather than
				// leaving the keys out.
				tool["parameters"] = fn["parameters"]
				tool["strict"] = fn["strict"]
				tool["description"] = fn["description"]
			}
			out = append(out, tool)
			continue
		}
		if jsonx.GetString(m, "type") == "custom" {
			if custom, ok := jsonx.AsMap(m["custom"]); ok {
				tool := map[string]any{"type": "custom", "name": jsonx.GetString(custom, "name")}
				if d := jsonx.GetString(custom, "description"); d != "" {
					tool["description"] = d
				}
				if f, ok := jsonx.AsMap(custom["format"]); ok {
					tool["format"] = customToolFormatToResponses(f)
				}
				out = append(out, tool)
				continue
			}
		}
		out = append(out, m)
	}
	return out
}

// customToolFormatToResponses unwraps a chat grammar object into the flat
// grammar shape the Responses API takes.
func customToolFormatToResponses(format map[string]any) map[string]any {
	if jsonx.GetString(format, "type") != "grammar" {
		return format
	}
	grammar, ok := jsonx.AsMap(format["grammar"])
	if !ok {
		return format
	}
	out := map[string]any{"type": "grammar"}
	if v, ok := grammar["definition"]; ok {
		out["definition"] = v
	}
	if v, ok := grammar["syntax"]; ok {
		out["syntax"] = v
	}
	return out
}

// responsesToolsToChat folds a Responses tools list onto the chat wire and,
// when the list named a web search tool, derives the web_search_options the
// chat request carries instead of a tool entry.
func responsesToolsToChat(v any) (tools []any, webSearchOptions map[string]any) {
	list, ok := jsonx.AsSlice(v)
	if !ok {
		return nil, nil
	}
	for _, item := range list {
		m, ok := jsonx.AsMap(item)
		if !ok {
			continue
		}
		typ := jsonx.GetString(m, "type")
		if typ == "function" {
			// description and strict are always stated — absent reads as the
			// empty string and false, the values the reference always sends.
			fn := map[string]any{
				"name":        jsonx.GetString(m, "name"),
				"description": jsonx.GetString(m, "description"),
			}
			// A function tool needs an object schema; a provider rejects one
			// that carries no `type`, so an absent or empty parameters object
			// becomes `{"type":"object"}`.
			params, ok := jsonx.AsMap(m["parameters"])
			if !ok || params["type"] == nil {
				if !ok {
					params = map[string]any{}
				}
				params["type"] = "object"
			}
			fn["parameters"] = params
			strict, _ := jsonx.Bool(m["strict"])
			fn["strict"] = strict
			tools = append(tools, map[string]any{"type": "function", "function": fn})
			continue
		}
		if typ == "custom" {
			// A custom tool becomes a function tool: the grammar rides the
			// description as a Format block, and the parameters fix the one
			// string the format-shaped call takes.
			name := jsonx.GetString(m, "name")
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{
				"name":        name,
				"description": jsonx.GetString(m, "description") + grammarSuffix(m["format"]),
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{"content": map[string]any{
						"type":        "string",
						"description": "The " + name + " content following the specified format",
					}},
					"required": []any{"content"},
				},
			}})
			continue
		}
		if typ == "web_search" || typ == "web_search_preview" {
			// Web search is a request param on the chat wire, not a tool the
			// model could call; the options it stated ride the param, and a
			// later search tool's options win over an earlier one's.
			webSearchOptions = map[string]any{
				"search_context_size": nil,
				"user_location":       nil,
			}
			if v, ok := m["search_context_size"]; ok {
				webSearchOptions["search_context_size"] = v
			}
			if v, ok := m["user_location"]; ok && v != nil {
				webSearchOptions["user_location"] = v
			}
			continue
		}
		// The reference's explicit drop list: these tool types have no chat
		// spelling and the target rejects them. Everything else — mcp, dated
		// hosted tools, previews — passes through as it stands.
		if typ == "computer_use" || typ == "image_generation" || typ == "shell" {
			continue
		}
		tools = append(tools, m)
	}
	return tools, webSearchOptions
}

// grammarSuffix renders a custom tool's grammar as the Format block appended
// to its description, so the model can produce format-shaped output. A format
// with no definition contributes nothing.
func grammarSuffix(format any) string {
	fm, ok := jsonx.AsMap(format)
	if !ok {
		return ""
	}
	def := jsonx.GetString(fm, "definition")
	if def == "" {
		return ""
	}
	return "\n\nFormat:\n```" + jsonx.GetString(fm, "syntax") + "\n" + def + "\n```"
}

// chatToolChoiceToResponses maps a chat tool_choice to the Responses shape.
// Chat nests a named choice under `function`/`custom`; Responses takes the name
// at the top level, so the nested object is flattened rather than forwarded.
func chatToolChoiceToResponses(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto", "none", "required":
			return typ
		case "function", "custom":
			if name := jsonx.GetString(t, "name"); name != "" {
				return map[string]any{"type": typ, "name": name}
			}
			if nested, ok := jsonx.AsMap(t[typ]); ok {
				if name := jsonx.GetString(nested, "name"); name != "" {
					return map[string]any{"type": typ, "name": name}
				}
			}
		}
		return t
	default:
		return v
	}
}

// responsesToolChoiceToChat maps a Responses tool_choice to the chat shape.
// Chat nests a named choice under `function`, and a choice that names no tool
// means "any tool" — the chat spelling for that is "required".
func responsesToolChoiceToChat(v any) any {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		// A choice that already carries a chat-shaped function name is
		// forwarded untouched.
		if fn, ok := jsonx.AsMap(t["function"]); ok && jsonx.GetString(fn, "name") != "" {
			return t
		}
		typ := jsonx.GetString(t, "type")
		switch typ {
		case "auto", "none", "required":
			return typ
		case "tool", "any":
			return "required"
		case "function", "custom":
			name := jsonx.GetString(t, "name")
			if name == "" {
				if nested, ok := jsonx.AsMap(t[typ]); ok {
					name = jsonx.GetString(nested, "name")
				}
			}
			if name != "" {
				return map[string]any{"type": "function", "function": map[string]any{"name": name}}
			}
			return "required"
		default:
			return t
		}
	default:
		return v
	}
}

func responseFormatToText(rf map[string]any) map[string]any {
	switch jsonx.GetString(rf, "type") {
	case "json_schema":
		// A json_schema format states its name, schema and strict whatever the
		// chat side supplied: missing members read as the defaults the
		// reference fills in, so the format never rides with a null schema or
		// no name for the Responses upstream to reject.
		js, _ := jsonx.AsMap(rf["json_schema"])
		if js == nil {
			js = map[string]any{}
		}
		format := map[string]any{"type": "json_schema"}
		if n, ok := js["name"]; ok {
			format["name"] = n
		} else {
			format["name"] = "response_schema"
		}
		if s, ok := js["schema"]; ok {
			format["schema"] = s
		} else {
			format["schema"] = map[string]any{}
		}
		if st, ok := js["strict"]; ok {
			format["strict"] = st
		} else {
			format["strict"] = false
		}
		return map[string]any{"format": format}
	case "json_object":
		return map[string]any{"format": map[string]any{"type": "json_object"}}
	case "text":
		// An explicit "no structured output" request; the Responses spelling is
		// a text format rather than an absent one.
		return map[string]any{"format": map[string]any{"type": "text"}}
	default:
		return nil
	}
}

func textFormatToResponseFormat(text map[string]any) map[string]any {
	format, ok := jsonx.AsMap(text["format"])
	if !ok {
		return nil
	}
	switch jsonx.GetString(format, "type") {
	case "json_schema":
		js := map[string]any{}
		if n, ok := format["name"]; ok {
			js["name"] = n
		} else {
			js["name"] = "response_schema"
		}
		if s, ok := format["schema"]; ok {
			js["schema"] = s
		} else {
			js["schema"] = map[string]any{}
		}
		// A missing strict is stated as false rather than left null, which is
		// not a value the schema takes.
		if st, ok := jsonx.Bool(format["strict"]); ok {
			js["strict"] = st
		} else {
			js["strict"] = false
		}
		return map[string]any{"type": "json_schema", "json_schema": js}
	case "json_object":
		return map[string]any{"type": "json_object"}
	case "text":
		return map[string]any{"type": "text"}
	default:
		return nil
	}
}

func first(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
