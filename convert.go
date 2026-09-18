package protocol

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/veildawn/ai-protocol/chatmessages"
	"github.com/veildawn/ai-protocol/chatresponses"
	"github.com/veildawn/ai-protocol/internal/jsonx"
	"github.com/veildawn/ai-protocol/messagesresponses"
	"github.com/veildawn/ai-protocol/stream"
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
	// OnEvent, when set, sees every TARGET-dialect event immediately before it
	// is written and returns the event to write instead. It is the host's seam
	// for policy this library has no business knowing — a tool-naming
	// convention, a redaction, a metric — applied to decoded events rather than
	// by re-parsing serialized SSE. Nil is the identity.
	//
	// It is NOT called on the same-dialect path (from == to), which is a raw
	// copy: there is no decoding there to hang a hook on. A caller that needs
	// the hook is by construction on a translating pair.
	OnEvent func(stream.Event) (stream.Event, error)
}

// ConvertOptions controls parameter handling during conversion.
type ConvertOptions struct {
	// DropParams omits source parameters the target dialect cannot represent
	// instead of returning UnsupportedParamError. The omission is still visible
	// through the loss API.
	DropParams bool
	// MaxTokensFallback is the value inserted when a conversion INTO Messages
	// would otherwise leave that dialect's required max_tokens field absent.
	// Neither source dialect carries an equivalent, so the value cannot be
	// derived from the body — it is host policy (typically the target model's
	// published output ceiling), injected here rather than guessed, because
	// guessing is vendor knowledge and the codec carries none. Zero leaves the
	// field absent and the conversion fails with MissingRequiredFieldError:
	// a body that would 400 at the upstream is a conversion bug reported late.
	MaxTokensFallback int
}

// ConvertRequest maps a request body from dialect `from` to dialect `to`.
// Same-dialect calls return the original bytes.
func ConvertRequest(from, to Dialect, body []byte) (Converted, error) {
	return ConvertRequestWith(from, to, body, ConvertOptions{})
}

// ConvertRequestWith is ConvertRequest with explicit conversion options.
func ConvertRequestWith(from, to Dialect, body []byte, opts ConvertOptions) (Converted, error) {
	return convertRequest(from, to, body, opts)
}

func convertRequest(from, to Dialect, body []byte, opts ConvertOptions) (Converted, error) {
	dropParams := opts.DropParams
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
	// Messages is the one dialect with a required field neither other dialect
	// carries: a fold into it has to invent max_tokens, and inventing a value
	// is host policy, not conversion mechanics.
	if dst == Messages {
		if _, ok := out["max_tokens"]; !ok {
			if opts.MaxTokensFallback <= 0 {
				return Converted{}, MissingRequiredFieldError{Field: "max_tokens", To: dst}
			}
			out["max_tokens"] = opts.MaxTokensFallback
		}
	}
	b, err := jsonx.Marshal(out)
	if err != nil {
		return Converted{}, ConversionError{Op: "request", Err: err}
	}
	return Converted{Body: b, Model: jsonx.GetString(out, "model")}, nil
}

// ConvertResponse maps a non-stream response body from dialect `from` to `to`.
func ConvertResponse(from, to Dialect, body []byte) ([]byte, error) {
	return ConvertResponseWith(from, to, body, ConvertOptions{})
}

// ConvertResponseWith is ConvertResponse with explicit conversion options. The
// option set is accepted now so callers can pass one policy through request and
// response conversion without a future breaking change.
func ConvertResponseWith(from, to Dialect, body []byte, opts ConvertOptions) ([]byte, error) {
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

// StreamResult describes how the SOURCE stream ended in addition to the bytes
// written to dst. Terminal is true only when the upstream emitted its own
// terminal frame; synthetic terminators generated by a converter do not count.
type StreamResult struct {
	Written   int64
	Terminal  bool
	Truncated bool
	InBandErr error
}

// PipeStream rewrites SSE from dialect `from` into dialect `to`.
func PipeStream(from, to Dialect, dst io.Writer, src io.Reader, opts StreamOpts) (int64, error) {
	result, err := PipeStreamWith(from, to, dst, src, opts)
	return result.Written, err
}

// PipeStreamWith is PipeStream with source-stream settlement metadata.
func PipeStreamWith(from, to Dialect, dst io.Writer, source io.Reader, opts StreamOpts) (StreamResult, error) {
	src, ok := Normalize(string(from))
	if !ok {
		return StreamResult{}, fmt.Errorf("unknown dialect %q", from)
	}
	dstDialect, ok := Normalize(string(to))
	if !ok {
		return StreamResult{}, fmt.Errorf("unknown dialect %q", to)
	}

	var pipe func(io.Writer, io.Reader) (int64, error)
	switch {
	case src == dstDialect:
		pipe = func(w io.Writer, r io.Reader) (int64, error) { return io.Copy(w, r) }
	case src == Chat && dstDialect == Messages:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return chatmessages.PipeChatToMessages(w, r, chatmessages.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	case src == Messages && dstDialect == Chat:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return chatmessages.PipeMessagesToChat(w, r, chatmessages.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	case src == Chat && dstDialect == Responses:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return chatresponses.PipeChatToResponses(w, r, chatresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	case src == Responses && dstDialect == Chat:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return chatresponses.PipeResponsesToChat(w, r, chatresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	case src == Messages && dstDialect == Responses:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return messagesresponses.PipeMessagesToResponses(w, r, messagesresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	case src == Responses && dstDialect == Messages:
		pipe = func(w io.Writer, r io.Reader) (int64, error) {
			return messagesresponses.PipeResponsesToMessages(w, r, messagesresponses.StreamOpts{Flush: opts.Flush, Model: opts.Model, OnEvent: opts.OnEvent})
		}
	default:
		return StreamResult{}, fmt.Errorf("unsupported stream conversion %s -> %s", src, dstDialect)
	}

	observer := &stream.Observer{}
	tee := observeReader{r: source, observer: observer}
	n, err := pipe(dst, tee)
	result := StreamResult{Written: n}
	// A source in-band error is returned to stop scanning; retain it as metadata
	// rather than surfacing it as a transport failure.
	if errors.Is(err, observer.Result.InBandErr) {
		err = nil
	}
	result.Terminal = observer.Result.Terminal
	result.InBandErr = observer.Result.InBandErr
	result.Truncated = err == nil && !result.Terminal && result.InBandErr == nil
	return result, err
}

type observeReader struct {
	r        io.Reader
	observer *stream.Observer
}

func (o observeReader) Read(p []byte) (int, error) {
	n, err := o.r.Read(p)
	if n > 0 {
		// Feed each SSE-sized read through the observer. Scanner returns the
		// same buffer repeatedly, so Scan must copy; this wrapper intentionally
		// works one read at a time.
		if scanErr := stream.Scan(bytes.NewReader(p[:n]), o.observer.Feed); scanErr != nil {
			if o.observer.Result.InBandErr != nil && errors.Is(scanErr, o.observer.Result.InBandErr) {
				return n, err
			}
			if err == nil {
				err = scanErr
			}
		}
	}
	return n, err
}

func peekModel(body []byte) string {
	var p struct {
		Model string `json:"model"`
	}
	_ = jsonxUnmarshalModel(body, &p)
	return p.Model
}
