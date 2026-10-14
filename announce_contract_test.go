package protocol_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/veildawn/ai-protocol/chatresponses"
	"github.com/veildawn/ai-protocol/geminiresponses"
	"github.com/veildawn/ai-protocol/messagesresponses"
)

// The announcement contract every →Responses fold owes its clients: no event
// may reference an output item the stream never announced with
// response.output_item.added, and sequence numbers strictly increase. A strict
// client (Codex, the OpenAI SDK) reads the stream by item declarations; a delta
// addressing an undeclared item is dropped there — the model's text silently
// lost while usage still reports. These tests pin the mixed-content turns that
// used to break it: a tool call and text in the same response, in either order.

type contractEvent struct {
	Type      string `json:"type"`
	ItemID    string `json:"item_id"`
	Seq       *int64 `json:"sequence_number"`
	Item      *struct {
		ID   string `json:"id"`
		Type string `json:"type"`
	} `json:"item"`
}

func parseContractEvents(t *testing.T, body string) []contractEvent {
	t.Helper()
	var evs []contractEvent
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var ev contractEvent
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			t.Fatalf("unparsable event %q: %v", payload, err)
		}
		evs = append(evs, ev)
	}
	return evs
}

// assertAnnouncementContract fails the test at the first event that references
// an item no output_item.added declared, or at the first sequence number that
// does not strictly increase.
func assertAnnouncementContract(t *testing.T, body string) {
	t.Helper()
	announced := map[string]bool{}
	var lastSeq int64 = -1
	for i, ev := range parseContractEvents(t, body) {
		if ev.Seq != nil {
			if *ev.Seq <= lastSeq {
				t.Fatalf("event %d (%s): sequence_number %d not increasing (previous %d)", i, ev.Type, *ev.Seq, lastSeq)
			}
			lastSeq = *ev.Seq
		}
		if ev.Type == "response.output_item.added" {
			if ev.Item != nil && ev.Item.ID != "" {
				announced[ev.Item.ID] = true
			}
			continue
		}
		id := ev.ItemID
		if id == "" && ev.Item != nil {
			id = ev.Item.ID
		}
		if id != "" && !announced[id] {
			t.Fatalf("event %d (%s) references item %q that was never announced", i, ev.Type, id)
		}
	}
}

// collectText joins every output_text delta the client would see.
func collectText(t *testing.T, body string) string {
	t.Helper()
	var b strings.Builder
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		var ev struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
		}
		if json.Unmarshal([]byte(payload), &ev) == nil && ev.Type == "response.output_text.delta" {
			b.WriteString(ev.Delta)
		}
	}
	return b.String()
}

// callNames lists the function calls the stream announced.
func callNames(t *testing.T, body string) []string {
	t.Helper()
	var names []string
	for _, ev := range parseContractEvents(t, body) {
		if ev.Type == "response.output_item.added" && ev.Item != nil && ev.Item.Type == "function_call" {
			names = append(names, ev.Item.ID)
		}
	}
	return names
}

// --- chat → Responses ---

func chatToolFirstStream() string {
	return "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Let me check the weather for you.\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20}}\n\n" +
		"data: [DONE]\n\n"
}

func TestChatToolFirstKeepsAnnouncedText(t *testing.T) {
	var out bytes.Buffer
	if _, err := chatresponses.PipeChatToResponses(&out, strings.NewReader(chatToolFirstStream()), chatresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "Let me check the weather for you." {
		t.Fatalf("text after a tool call lost: got delta text %q", got)
	}
	if calls := callNames(t, out.String()); len(calls) != 1 || calls[0] != "call_1" {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// A chunk carrying content AND tool_calls together must keep both.
func TestChatSameChunkKeepsTextAndCall(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Checking the weather now.\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20}}\n\n" +
		"data: [DONE]\n\n"
	var out bytes.Buffer
	if _, err := chatresponses.PipeChatToResponses(&out, strings.NewReader(src), chatresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "Checking the weather now." {
		t.Fatalf("text lost: got %q", got)
	}
	if calls := callNames(t, out.String()); len(calls) != 1 {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// Text that follows a reasoning stretch rides its own announced message item.
func TestChatReasoningThenTextAnnouncesMessage(t *testing.T) {
	src := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking...\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer.\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
	var out bytes.Buffer
	if _, err := chatresponses.PipeChatToResponses(&out, strings.NewReader(src), chatresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "The answer." {
		t.Fatalf("text after reasoning lost: got %q", got)
	}
}

// --- messages → Responses ---

func messagesToolFirstStream() string {
	return "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check the weather.\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":20}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

func TestMessagesToolFirstKeepsAnnouncedText(t *testing.T) {
	var out bytes.Buffer
	if _, err := messagesresponses.PipeMessagesToResponses(&out, strings.NewReader(messagesToolFirstStream()), messagesresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "Let me check the weather." {
		t.Fatalf("text after a tool block lost: got %q", got)
	}
	if calls := callNames(t, out.String()); len(calls) != 1 || calls[0] != "toolu_1" {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// Text that follows a thinking stretch rides its own announced message item.
func TestMessagesThinkingThenTextAnnouncesMessage(t *testing.T) {
	src := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"pondering\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"The answer.\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	var out bytes.Buffer
	if _, err := messagesresponses.PipeMessagesToResponses(&out, strings.NewReader(src), messagesresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "The answer." {
		t.Fatalf("text after thinking lost: got %q", got)
	}
}

// --- gemini → Responses ---

func TestGeminiCallThenTextKeepsAnnouncedText(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Let me check the weather.\"}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":20}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "Let me check the weather." {
		t.Fatalf("text after a call lost: got %q", got)
	}
	if calls := callNames(t, out.String()); len(calls) != 1 {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// A frame carrying text AND a functionCall together — the wire's common
// "narrate, then call" shape — must keep both.
func TestGeminiTextAndCallSameFrameKeepsBoth(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"I'll check the weather.\"},{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":20}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "I'll check the weather." {
		t.Fatalf("text lost: got %q", got)
	}
	if calls := callNames(t, out.String()); len(calls) != 1 {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// The wire's normal tool-turn ending — the functionCall and its finishReason in
// the same frame — must deliver the call.
func TestGeminiCallAndFinishSameFrameKeepsCall(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":20}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if calls := callNames(t, out.String()); len(calls) != 1 {
		t.Fatalf("tool call lost: %v", calls)
	}
}

// A first text frame keeps its own delta now: the announcement no longer
// replaces the frame's content.
func TestGeminiFirstTextFrameKeepsDelta(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello there.\"}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":2}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
	if got := collectText(t, out.String()); got != "Hello there." {
		t.Fatalf("first frame's delta lost: got %q", got)
	}
}

// A tool-only turn still closes with an announced message item, matching the
// non-streaming fold's always-present message item.
func TestGeminiToolOnlyTurnAnnouncesMessageItem(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":20}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
}

// Silence: events referencing nothing are fine, and an empty turn still
// announces its message item.
func TestGeminiEmptyTurnAnnouncesMessageItem(t *testing.T) {
	src := "data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}\n\n"
	var out bytes.Buffer
	if _, err := geminiresponses.PipeGeminiToResponses(&out, strings.NewReader(src), geminiresponses.StreamOpts{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	assertAnnouncementContract(t, out.String())
}
