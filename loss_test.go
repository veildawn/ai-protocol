package protocol

import (
	"encoding/json"
	"testing"
)

func TestBodyLossesChatToMessages(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7,"frequency_penalty":0.5}`)
	got := BodyLosses(Chat, Messages, body, ConvertOptions{})
	if !got[LossSeed] || !got[LossPenaltyControls] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesMissingBodyIsUnknownContent(t *testing.T) {
	got := BodyLosses(Chat, Messages, []byte(`{`), ConvertOptions{})
	if !got[LossUnknownContent] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesFileReference(t *testing.T) {
	chat := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"file","file":{"file_id":"f"}}]}]}`)
	got := BodyLosses(Chat, Messages, chat, ConvertOptions{})
	if !got[LossFileReference] {
		t.Fatalf("losses=%v", got)
	}
	responses := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"x"},{"type":"input_file","file_id":"f"}]}]}`)
	got = BodyLosses(Responses, Messages, responses, ConvertOptions{})
	if !got[LossFileReference] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesInvalidToolArguments(t *testing.T) {
	chat := []byte(`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"["}}]}]}`)
	got := BodyLosses(Chat, Responses, chat, ConvertOptions{})
	if !got[LossToolArguments] {
		t.Fatalf("losses=%v", got)
	}
	responses := []byte(`{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":"[]"}]}`)
	got = BodyLosses(Responses, Messages, responses, ConvertOptions{})
	if !got[LossToolArguments] {
		t.Fatalf("losses=%v", got)
	}
}

func TestBodyLossesMatchConverterRejection(t *testing.T) {
	bodies := map[string][]byte{
		"seed":       []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7}`),
		"logit_bias": []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"logit_bias":{"x":1}}`),
		"logprobs":   []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"logprobs":true}`),
	}
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			// A fallback keeps the required-field rule out of this test's way:
			// the assertion is about LOSSES, not about max_tokens.
			losses := BodyLosses(Chat, Messages, body, ConvertOptions{MaxTokensFallback: 8192})
			_, err := ConvertRequestWith(Chat, Messages, body, ConvertOptions{MaxTokensFallback: 8192})
			if losses != nil && err == nil {
				t.Fatalf("loss=%v but conversion succeeded", losses)
			}
			if losses == nil && err != nil {
				t.Fatalf("no loss but conversion failed: %v", err)
			}
		})
	}
}

func TestConvertRequestWithDropsUnsupportedParams(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"seed":7}`)
	got, err := ConvertRequestWith(Chat, Messages, body, ConvertOptions{DropParams: true, MaxTokensFallback: 8192})
	if err != nil {
		t.Fatalf("drop params conversion failed: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(got.Body, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["seed"]; ok {
		t.Fatalf("seed was not dropped: %s", got.Body)
	}
}
