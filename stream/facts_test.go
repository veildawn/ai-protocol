package stream

import "testing"

func TestClassifyResponseCreatedAsPrelude(t *testing.T) {
	facts := ClassifyEvent(Responses, Event{Data: `{"type":"response.created","response":{"id":"resp_1"}}`})
	if facts.Kind != EventPrelude {
		t.Fatalf("response.created kind = %v, want prelude", facts.Kind)
	}
}

func TestClassifyNonEmptyResponsesTextDeltaAsContent(t *testing.T) {
	facts := ClassifyEvent(Responses, Event{Data: `{"type":"response.output_text.delta","delta":"hello"}`})
	if facts.Kind != EventContent {
		t.Fatalf("non-empty output_text delta kind = %v, want content", facts.Kind)
	}
}

func TestClassifyStreamEventFacts(t *testing.T) {
	tests := []struct {
		name         string
		dialect      Dialect
		event        Event
		want         EventKind
		wantTerminal bool
	}{
		{
			name:    "response metadata is prelude",
			dialect: Responses,
			event:   Event{Data: `{"type":"response.created","response":{"id":"r1"}}`},
			want:    EventPrelude,
		},
		{
			name:    "response text is content",
			dialect: Responses,
			event:   Event{Data: `{"type":"response.output_text.delta","delta":"hello"}`},
			want:    EventContent,
		},
		{
			name:    "empty response text is prelude",
			dialect: Responses,
			event:   Event{Data: `{"type":"response.output_text.delta","delta":""}`},
			want:    EventPrelude,
		},
		{
			name:    "empty chat role is prelude",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`},
			want:    EventPrelude,
		},
		{
			name:    "empty chat tool index is prelude",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}`},
			want:    EventPrelude,
		},
		{
			name:    "chat tool id alone is prelude",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1"}]}}]}`},
			want:    EventPrelude,
		},
		{
			name:    "chat tool name is content",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup"}}]}}]}`},
			want:    EventContent,
		},
		{
			name:    "chat reasoning is distinct from output text",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{"reasoning_content":"think"}}]}`},
			want:    EventReasoning,
		},
		{
			name:    "chat finish reason is explicit completion",
			dialect: Chat,
			event:   Event{Data: `{"choices":[{"delta":{},"finish_reason":"stop"}]}`},
			want:    EventComplete,
		},
		{
			name:    "messages empty text block is prelude",
			dialect: Messages,
			event:   Event{Event: "content_block_start", Data: `{"type":"content_block_start","content_block":{"type":"text","text":""}}`},
			want:    EventPrelude,
		},
		{
			name:    "messages reasoning delta is reasoning",
			dialect: Messages,
			event:   Event{Event: "content_block_delta", Data: `{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"private thought"}}`},
			want:    EventReasoning,
		},
		{
			name:    "messages tool declaration is actionable content",
			dialect: Messages,
			event:   Event{Event: "content_block_start", Data: `{"type":"content_block_start","content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}`},
			want:    EventContent,
		},
		{
			name:         "responses terminal succeeds",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.completed","response":{"output":[]}}`},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:         "responses incomplete remains distinct",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.incomplete","response":{"output":[]}}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			// A relay may fold a ceiling cut into the Responses shape and close
			// it under response.completed while status still says incomplete.
			// The name alone would call that a success and hide the truncation.
			name:         "responses completed with incomplete status is not a success",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.completed","response":{"status":"incomplete","output":[]}}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			name:         "responses completed with top-level incomplete status is not a success",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.completed","status":"incomplete","response":{"output":[]}}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			// The common case must not regress: an explicit completed status
			// still classifies as a completion.
			name:         "responses completed with completed status stays complete",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.completed","response":{"status":"completed","output":[]}}`},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:         "responses failure is explicit",
			dialect:      Responses,
			event:        Event{Data: `{"type":"response.failed","response":{"error":{"message":"denied"}}}`},
			want:         EventFailure,
			wantTerminal: true,
		},
		{
			name:         "chat done sentinel is terminal fact",
			dialect:      Chat,
			event:        Event{Data: "[DONE]"},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:    "usage only is not content",
			dialect: Chat,
			event:   Event{Data: `{"choices":[],"usage":{"prompt_tokens":42}}`},
			want:    EventPrelude,
		},
		{
			name:    "malformed event is unknown",
			dialect: Responses,
			event:   Event{Data: `{"type":"response.created"`},
			want:    EventUnknown,
		},
		{
			name:  "ping is prelude even without a dialect",
			event: Event{Event: "ping"},
			want:  EventPrelude,
		},
		// The generateContent wire: no sentinel, no separate terminal frame —
		// a candidate's finishReason (or a blocked promptFeedback) IS the
		// source's own close, so those frames carry Terminal here.
		{
			name:    "gemini text frame is content",
			dialect: Gemini,
			event:   Event{Data: `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"index":0}]}`},
			want:    EventContent,
		},
		{
			name:    "gemini thought frame is reasoning",
			dialect: Gemini,
			event:   Event{Data: `{"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]}}]}`},
			want:    EventReasoning,
		},
		{
			name:    "gemini function call frame is content",
			dialect: Gemini,
			event:   Event{Data: `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]}}]}`},
			want:    EventContent,
		},
		{
			name:    "gemini empty candidate is prelude",
			dialect: Gemini,
			event:   Event{Data: `{"candidates":[{"content":{"role":"model","parts":[]},"index":0}]}`},
			want:    EventPrelude,
		},
		{
			// A signature alone says nothing was delivered: the thought text
			// it signs rides other frames.
			name:    "gemini lone thought signature is prelude",
			dialect: Gemini,
			event:   Event{Data: `{"candidates":[{"content":{"role":"model","parts":[{"thoughtSignature":"sig"}]}}]}`},
			want:    EventPrelude,
		},
		{
			name:         "gemini finishReason STOP is a terminal completion",
			dialect:      Gemini,
			event:        Event{Data: `{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1}}`},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:         "gemini finishReason MAX_TOKENS is an incomplete terminal",
			dialect:      Gemini,
			event:        Event{Data: `{"candidates":[{"finishReason":"MAX_TOKENS","index":0}]}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			name:         "gemini finishReason SAFETY is an incomplete terminal",
			dialect:      Gemini,
			event:        Event{Data: `{"candidates":[{"finishReason":"SAFETY","index":0}]}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			// Values the finish mapping does not name settle on stop, the way
			// the reference's mapper treats an unknown reason.
			name:         "gemini unknown finishReason is still a completion",
			dialect:      Gemini,
			event:        Event{Data: `{"candidates":[{"finishReason":"WAT","index":0}]}`},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:         "gemini blocked prompt is an incomplete terminal",
			dialect:      Gemini,
			event:        Event{Data: `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5}}`},
			want:         EventIncomplete,
			wantTerminal: true,
		},
		{
			name:    "gemini usage-only frame is prelude",
			dialect: Gemini,
			event:   Event{Data: `{"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5}}`},
			want:    EventPrelude,
		},
		{
			// This wire has no sentinel of its own; one arriving names no fact
			// of it. (A relay's sentinel is still a source close for the folds.)
			name:    "gemini does not claim the done sentinel",
			dialect: Gemini,
			event:   Event{Data: "[DONE]"},
			want:    EventUnknown,
		},
		{
			// The empty-dialect inference must reach the gemini classifier on
			// its own framing keys — candidates, promptFeedback, usageMetadata —
			// because the observer classifies with no dialect at all.
			name:         "untyped finishReason frame infers gemini",
			event:        Event{Data: `{"candidates":[{"finishReason":"STOP","index":0}]}`},
			want:         EventComplete,
			wantTerminal: true,
		},
		{
			name:  "untyped usageMetadata frame infers gemini",
			event: Event{Data: `{"usageMetadata":{"promptTokenCount":5}}`},
			want:  EventPrelude,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			facts := ClassifyEvent(tt.dialect, tt.event)
			if facts.Kind != tt.want {
				t.Fatalf("ClassifyEvent(%q, %+v).Kind = %v, want %v", tt.dialect, tt.event, facts.Kind, tt.want)
			}
			if facts.Terminal != tt.wantTerminal {
				t.Fatalf("ClassifyEvent(%q, %+v).Terminal = %v, want %v", tt.dialect, tt.event, facts.Terminal, tt.wantTerminal)
			}
		})
	}
}

func TestClassifyEmptyShellsRemainPrelude(t *testing.T) {
	cases := []struct {
		name    string
		dialect Dialect
		event   Event
	}{
		{"responses empty message content", Responses, Event{Data: `{"type":"response.output_item.added","item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":""}]}}`}},
		{"responses tool id alone", Responses, Event{Data: `{"type":"response.output_item.added","item":{"type":"function_call","id":"call_1"}}`}},
		{"messages signature without reasoning", Messages, Event{Data: `{"type":"content_block_delta","delta":{"type":"signature_delta","signature":"abc"}}`}},
		{"chat function call index alone", Chat, Event{Data: `{"choices":[{"delta":{"function_call":{"index":0}}}]}`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyEvent(tc.dialect, tc.event).Kind; got != EventPrelude {
				t.Fatalf("kind = %v, want prelude", got)
			}
		})
	}
}

func TestClassifyFailurePayloadIsDialectScoped(t *testing.T) {
	tests := []struct {
		name    string
		dialect Dialect
		event   Event
		want    EventKind
	}{
		{"chat error envelope", Chat, Event{Data: `{"error":{"message":"bad"}}`}, EventFailure},
		{"responses failed event", Responses, Event{Data: `{"type":"response.failed","response":{"error":{"message":"bad"}}}`}, EventFailure},
		{"message error event", Messages, Event{Data: `{"type":"error","error":{"message":"bad"}}`}, EventFailure},
		{"gemini error envelope", Gemini, Event{Data: `{"error":{"code":429,"message":"quota"}}`}, EventFailure},
		{"unknown messages error-shaped extension", Messages, Event{Data: `{"type":"message","error":{"message":"extension"}}`}, EventUnknown},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyEvent(tc.dialect, tc.event).Kind; got != tc.want {
				t.Fatalf("kind = %v, want %v", got, tc.want)
			}
		})
	}
}

// The gemini wire's failure frame must classify as a failure with the dialect
// stated EXPLICITLY, not only through the empty-dialect path that happens to
// infer Chat for a bare error key first.
func TestClassifyGeminiErrorIsTerminalFailure(t *testing.T) {
	facts := ClassifyEvent(Gemini, Event{Data: `{"error":{"code":429,"message":"quota"}}`})
	if facts.Kind != EventFailure || !facts.Terminal {
		t.Fatalf("facts = %+v, want a terminal failure", facts)
	}
}

// Usage rides the wire's own usageMetadata member, on content frames and the
// usage tail alike; HasUsage must state it either way.
func TestClassifyGeminiUsage(t *testing.T) {
	tail := ClassifyEvent(Gemini, Event{Data: `{"usageMetadata":{"promptTokenCount":5}}`})
	if !tail.HasUsage {
		t.Fatalf("usage tail HasUsage = false")
	}
	terminal := ClassifyEvent(Gemini, Event{Data: `{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"totalTokenCount":5}}`})
	if !terminal.HasUsage {
		t.Fatalf("terminal frame HasUsage = false")
	}
}

// A Chat frame's keys (choices, usage) must keep their Chat reading when the
// dialect is inferred, and the gemini keys must not reach for a typed frame:
// the two vocabularies never share a key name, so the order only has to keep
// them apart deterministically. Bare Chat chunks have never claimed a dialect
// here (only the sentinel does); what matters is that they are not claimed BY
// the gemini inference.
func TestEventDialectInferenceKeepsChatAndGeminiApart(t *testing.T) {
	if got := (Event{Data: `[DONE]`}).Dialect(); got != Chat {
		t.Fatalf("sentinel dialect = %q, want chat", got)
	}
	if got := (Event{Data: `{"choices":[{"delta":{"content":"hi"}}]}`}).Dialect(); got != "" {
		t.Fatalf("choices frame dialect = %q, want unresolved", got)
	}
	if got := (Event{Data: `{"usage":{"prompt_tokens":1}}`}).Dialect(); got != "" {
		t.Fatalf("usage frame dialect = %q, want unresolved", got)
	}
	if got := (Event{Data: `{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`}).Dialect(); got != Gemini {
		t.Fatalf("candidates frame dialect = %q, want gemini", got)
	}
	if got := (Event{Data: `{"usageMetadata":{"promptTokenCount":1}}`}).Dialect(); got != Gemini {
		t.Fatalf("usageMetadata frame dialect = %q, want gemini", got)
	}
	if got := (Event{Data: `{"promptFeedback":{"blockReason":"SAFETY"}}`}).Dialect(); got != Gemini {
		t.Fatalf("promptFeedback frame dialect = %q, want gemini", got)
	}
}

func FuzzClassifyFramedSSEDoesNotPanic(f *testing.F) {
	f.Add("data: {\"type\":\"response.created\"}\n\n")
	f.Add("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
	f.Add("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"x\"}}\n\n")
	f.Add("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1}}\n\n")
	f.Fuzz(func(t *testing.T, wire string) {
		if len(wire) > maxEventBytes*2 {
			t.Skip()
		}
		var framer Framer
		consume := func(ev Event) error {
			ClassifyEvent(Responses, ev)
			ClassifyEvent(Messages, ev)
			ClassifyEvent(Chat, ev)
			ClassifyEvent(Gemini, ev)
			return nil
		}
		if err := framer.Feed([]byte(wire), consume); err == nil {
			_ = framer.Finish(consume)
		}
	})
}
