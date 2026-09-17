package protocol

import (
	"fmt"
	"io"
	"strings"

	"github.com/veildawn/ai-protocol/chatmessages"
	"github.com/veildawn/ai-protocol/chatresponses"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/messagesresponses"
)

// Converted is a converted request body.
type Converted struct {
	Body  []byte
	Model string
}

// StreamOpts controls SSE conversion.
type StreamOpts struct {
	Flush func()
	Model string
}

// ConvertRequest maps a request body from dialect `from` to dialect `to`.
// Same-dialect calls return the original bytes.
func ConvertRequest(from, to Dialect, body []byte) (Converted, error) {
	return convertRequest(from, to, body, false)
}

func convertRequest(from, to Dialect, body []byte, dropParams bool) (Converted, error) {
	src, ok := Normalize(string(from))
	if !ok {
		return Converted{}, fmt.Errorf("unknown dialect %q", from)
	}
	dst, ok := Normalize(string(to))
	if !ok {
		return Converted{}, fmt.Errorf("unknown dialect %q", to)
	}
	if src == dst {
		return Converted{Body: body, Model: peekModel(body)}, nil
	}
	obj, err := jsonx.UnmarshalMap(body)
	if err != nil {
		return Converted{}, ConversionError{Op: "request", Err: err}
	}
	var out map[string]any
	switch {
	case src == Chat && dst == Messages:
		out, err = chatmessages.ChatToMessages(obj, dropParams)
	case src == Messages && dst == Chat:
		out, err = chatmessages.MessagesToChat(obj)
	case src == Chat && dst == Responses:
		out, err = chatresponses.ChatToResponses(obj)
	case src == Responses && dst == Chat:
		out, err = chatresponses.ResponsesToChat(obj)
	case src == Messages && dst == Responses:
		out, err = messagesresponses.MessagesToResponses(obj)
	case src == Responses && dst == Messages:
		out, err = messagesresponses.ResponsesRequestToMessages(obj)
	default:
		return Converted{}, fmt.Errorf("unsupported conversion %s -> %s", src, dst)
	}
	if err != nil {
		if strings.HasPrefix(err.Error(), "unsupported openai param ") {
			p := strings.Trim(strings.TrimPrefix(err.Error(), "unsupported openai param "), `"`)
			return Converted{}, UnsupportedParamError{Param: p, From: src, To: dst}
		}
		return Converted{}, ConversionError{Op: "request", Err: err}
	}
	b, err := jsonx.Marshal(out)
	if err != nil {
		return Converted{}, ConversionError{Op: "request", Err: err}
	}
	return Converted{Body: b, Model: jsonx.GetString(out, "model")}, nil
}

// ConvertResponse maps a non-stream response body from dialect `from` to `to`.
func ConvertResponse(from, to Dialect, body []byte) ([]byte, error) {
	src, ok := Normalize(string(from))
	if !ok {
		return nil, fmt.Errorf("unknown dialect %q", from)
	}
	dst, ok := Normalize(string(to))
	if !ok {
		return nil, fmt.Errorf("unknown dialect %q", to)
	}
	if src == dst {
		return body, nil
	}
	obj, err := jsonx.UnmarshalMap(body)
	if err != nil {
		return nil, ConversionError{Op: "response", Err: err}
	}
	var out map[string]any
	switch {
	case src == Chat && dst == Messages:
		out = chatmessages.ChatResponseToMessages(obj)
	case src == Messages && dst == Chat:
		out = chatmessages.MessagesResponseToChat(obj)
	case src == Chat && dst == Responses:
		out = chatresponses.ChatResponseToResponses(obj)
	case src == Responses && dst == Chat:
		out = chatresponses.ResponsesResponseToChat(obj)
	case src == Messages && dst == Responses:
		out = messagesresponses.MessagesResponseToResponses(obj)
	case src == Responses && dst == Messages:
		out, err = messagesresponses.ResponsesToMessages(obj)
	default:
		return nil, fmt.Errorf("unsupported conversion %s -> %s", src, dst)
	}
	if err != nil {
		return nil, ConversionError{Op: "response", Err: err}
	}
	return jsonx.Marshal(out)
}

// PipeStream rewrites SSE from dialect `from` into dialect `to`.
func PipeStream(from, to Dialect, dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	s, ok := Normalize(string(from))
	if !ok {
		return 0, fmt.Errorf("unknown dialect %q", from)
	}
	d, ok := Normalize(string(to))
	if !ok {
		return 0, fmt.Errorf("unknown dialect %q", to)
	}
	if s == d {
		return io.Copy(dst, src)
	}
	switch {
	case s == Chat && d == Messages:
		return chatmessages.PipeChatToMessages(dst, src, chatmessages.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	case s == Messages && d == Chat:
		return chatmessages.PipeMessagesToChat(dst, src, chatmessages.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	case s == Chat && d == Responses:
		return chatresponses.PipeChatToResponses(dst, src, chatresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	case s == Responses && d == Chat:
		return chatresponses.PipeResponsesToChat(dst, src, chatresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	case s == Messages && d == Responses:
		return messagesresponses.PipeMessagesToResponses(dst, src, messagesresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	case s == Responses && d == Messages:
		return messagesresponses.PipeResponsesToMessages(dst, src, messagesresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model})
	default:
		return 0, fmt.Errorf("unsupported stream conversion %s -> %s", s, d)
	}
}

func peekModel(body []byte) string {
	var p struct {
		Model string `json:"model"`
	}
	_ = jsonxUnmarshalModel(body, &p)
	return p.Model
}
