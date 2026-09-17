package stream

import (
	"encoding/json"
	"errors"
)

// Result describes how one SSE stream ended. A pipe may still return an error
// alongside a result; readers should prefer Result for settlement once the
// transport itself succeeded.
type Result struct {
	// Terminal reports that the source dialect's own terminal frame arrived.
	Terminal bool
	// InBandErr is a failure the provider framed inside a successful stream.
	InBandErr error
}

// Observer tracks source-dialect terminal and in-band error frames.
type Observer struct {
	Result Result
}

// Feed inspects one decoded source event and returns an error when the caller
// should stop reading after this event.
func (o *Observer) Feed(ev Event) error {
	if o.Result.Terminal {
		return nil
	}
	d := ev.Dialect()
	if d == Chat && IsDone(ev.Data) {
		o.Result.Terminal = true
		return nil
	}
	var obj map[string]any
	if json.Unmarshal([]byte(ev.Data), &obj) != nil {
		return nil
	}
	if d == "" {
		if _, ok := obj["error"]; ok {
			d = Chat
		}
	}
	if typ, _ := obj["type"].(string); typ == "error" || typ == "response.failed" {
		o.Result.Terminal = true
		o.Result.InBandErr = InBandError(obj)
		return o.Result.InBandErr
	}
	if _, ok := obj["error"]; ok && d == Chat {
		o.Result.Terminal = true
		o.Result.InBandErr = InBandError(obj)
		return o.Result.InBandErr
	}
	if typ, _ := obj["type"].(string); d == Messages && typ == "message_stop" {
		o.Result.Terminal = true
		return nil
	}
	if typ, _ := obj["type"].(string); d == Responses &&
		(typ == "response.completed" || typ == "response.incomplete") {
		o.Result.Terminal = true
		return nil
	}
	return nil
}

// InBandError extracts a stable message from any public dialect's error frame.
func InBandError(obj map[string]any) error {
	if e, ok := obj["error"].(map[string]any); ok {
		if s, ok := e["message"].(string); ok && s != "" {
			return errors.New(s)
		}
		if s, ok := e["code"].(string); ok && s != "" {
			return errors.New(s)
		}
	}
	if resp, ok := obj["response"].(map[string]any); ok {
		if e, ok := resp["error"].(map[string]any); ok {
			if s, ok := e["message"].(string); ok && s != "" {
				return errors.New(s)
			}
			if s, ok := e["code"].(string); ok && s != "" {
				return errors.New(s)
			}
		}
	}
	if s, ok := obj["message"].(string); ok && s != "" {
		return errors.New(s)
	}
	return errors.New("upstream error")
}
