# ai-protocol

[English](README.md) | [简体中文](README.zh-CN.md)

[![Go Reference](https://pkg.go.dev/badge/github.com/veildawn/ai-protocol.svg)](https://pkg.go.dev/github.com/veildawn/ai-protocol)
[![CI](https://github.com/veildawn/ai-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/veildawn/ai-protocol/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/veildawn/ai-protocol)](https://goreportcard.com/report/github.com/veildawn/ai-protocol)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Go library that converts request bodies, response bodies and SSE streams
between the public LLM wire formats. It is a set of pure functions with no
dependencies outside the standard library, no I/O, and no network.

| Dialect     | Wire                                                              |
| ----------- | ----------------------------------------------------------------- |
| `chat`      | OpenAI Chat Completions (`/v1/chat/completions`)                  |
| `messages`  | Anthropic Messages (`/v1/messages`)                               |
| `responses` | OpenAI Responses (`/v1/responses`)                                |
| `gemini`    | The `generateContent` wire (`contents`/`parts`, `candidates`)     |
| `images`    | The image-generation wire (`prompt`/`n`/`size`, `b64_json`)       |

## Why this exists

An API gateway, proxy or SDK that accepts more than one public API shape has to
translate between them. Written per pair, that is N×M bodies of mapping code
that drift apart, disagree about edge cases, and bury protocol facts inside
HTTP handlers.

This library takes the translation out of the transport: every mapping is a
pure function over bytes or `map[string]any`, conversions name the protocol
facts they cannot preserve, and the package does not open connections, read
clocks, or know which upstream it is talking to. The caller keeps routing,
retries, credentials and vendor-specific behavior.

## Supported conversions

All four text dialects convert between any pair. `images` pairs with `gemini`
only, on request and response bodies — image generation does not stream, so
that pair has no SSE fold.

| Source ↓ / Target → | `chat` | `messages` | `responses` | `gemini` | `images` |
| ------------------- | ------ | ---------- | ----------- | -------- | -------- |
| `chat`              | copy   | req/resp/SSE | req/resp/SSE | req/resp/SSE | — |
| `messages`          | req/resp/SSE | copy | req/resp/SSE | req/resp/SSE | — |
| `responses`         | req/resp/SSE | req/resp/SSE | copy | req/resp/SSE | — |
| `gemini`            | req/resp/SSE | req/resp/SSE | req/resp/SSE | copy | req/resp |
| `images`            | — | — | — | req/resp | copy |

A same-dialect call returns the original bytes unchanged.

## Install

```sh
go get github.com/veildawn/ai-protocol@latest
```

Requires Go 1.23 or newer. The module has no third-party dependencies.

## Quick start

```go
package main

import (
	"fmt"

	protocol "github.com/veildawn/ai-protocol"
)

func main() {
	// A client sent a chat-completions body; the upstream speaks messages.
	body := []byte(`{"model":"example-model","messages":[{"role":"user","content":"hi"}]}`)

	req, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{
		// The messages wire requires max_tokens. A fold that cannot derive it
		// from a thinking budget asks for the value rather than inventing one.
		MaxTokensFallback: 8192,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("upstream body: %s\n", req.Body)

	// ... send req.Body to the upstream, read its response ...

	upstream := []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}]}`)
	out, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, upstream)
	if err != nil {
		panic(err)
	}
	fmt.Printf("client body: %s\n", out)
}
```

Both entry points return typed errors — `UnsupportedParamError`,
`MissingRequiredFieldError`, `ConversionError` — so a host can match them with
`errors.As` instead of string comparison.

Runnable programs live in [examples/](examples): one converts a request and a
response, one pipes an SSE turn.

`ConvertRequest` also returns the model name it read from the source body, so a
router can resolve an upstream before any conversion runs. `PeekModel` does the
same without a full decode, for hosts that route first.

## Streaming

```go
// w is the http.ResponseWriter; upstream is the upstream response body.
result, err := protocol.PipeStreamWith(protocol.Messages, protocol.Chat, w, upstream, protocol.StreamOpts{
	Flush: func() { w.(http.Flusher).Flush() },
})
if err != nil {
	// Transport or decode failure.
}
switch {
case result.InBandErr != nil:
	// The source framed a failure inside a successful stream.
case result.Truncated:
	// The source ended without its own terminal frame. The turn was left
	// open on purpose: only the source gets to close it.
case result.Terminal:
	// The source closed the turn itself.
}
```

`StreamResult` reports how the source stream ended, separately from the bytes
written to the destination (`result.Written`). A converter never closes a turn
the source did not close.

For policy the library should not know about — redaction, tool-name rewriting,
metrics — pass `StreamOpts.OnEvent`. It sees every target-dialect event
immediately before it is written, and is not called on the same-dialect path,
which is a raw copy.

`stream.Framer` yields complete SSE events from split reads.
`stream.ClassifyEvent` reports per-event protocol facts (prelude, content,
reasoning, completion, incomplete, failure, unknown) without deciding retry or
billing policy.

## Reporting lost parameters

A conversion can drop or reshape what the target dialect cannot carry. Losses
name those facts explicitly, so a caller can decide whether an omission is
acceptable before or alongside the conversion:

```go
losses := protocol.BodyLosses(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{})
if losses[protocol.LossLogprobs] {
	// The messages wire has no logprobs; decide whether that is acceptable.
}
```

`StaticLosses(from, to)` returns every loss possible for a directed pair,
independent of one body. `Loss` values are descriptive protocol facts,
never routing policy. The full list is in [loss.go](loss.go).

## Errors

| Error                      | Meaning                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `UnsupportedParamError`    | The target dialect cannot carry a parameter; set `DropParams` to omit |
| `MissingRequiredFieldError`| The target requires a field the caller declined to supply            |
| `ConversionError`          | The body could not be read or rebuilt                                |

Errors are typed and wrapped where useful; match them with `errors.As`.

## Design guarantees

These are enforced by `boundary_test.go`, which fails the build when one is
broken:

- **Standard library only.** No `require` directives, no third-party imports.
- **Offline.** No `net`, no HTTP, no clock, no filesystem.
- **Pure conversions.** Same input, same output; no global state.
- **Five dialects, closed set.** A new wire format is a deliberate boundary
  decision, not an implementation detail.
- **No vendor behavior.** Vendor-specific parameter repair, hostnames,
  credentials and quirks belong to the caller, not to the codec.
- **No policy.** Routing, billing, quota and retry decisions stay outside.

The reasoning behind each of these — and the trade-offs accepted — is in
[docs/design.md](docs/design.md).

## Compatibility

Mapping semantics are validated against [LiteLLM](https://github.com/BerriAI/litellm)
v1.100.1, whose MIT-licensed protocol mapping this project derives from. The
attribution is in [NOTICE](NOTICE); the compatibility fixtures that pin the
behavior are the `litellm_parity_*_test.go` files.

## Testing

```sh
make test        # unit, compatibility, boundary and property tests
make lint        # staticcheck
make cover-check # statement coverage against the floor CI enforces
make fuzz        # 10 seconds per target: conversions, SSE pipe, framer
make bench       # conversion, pipe and framer benchmarks
```

The fuzz targets treat the input as hostile: no request body, response body,
stream or chunking a client or an upstream can produce may panic the codec or
make a same-dialect conversion change the bytes.

Conversions are allocation-bound rather than I/O-bound — a request fold takes a
few microseconds on a laptop-class CPU — so `make bench` reports throughput in
MB/s alongside `ns/op` and allocations.

## Status

Pre-1.0 (`v0.8.x`). The public surface is small and pinned by tests;
versioning follows [SemVer](https://semver.org/), so a minor release may still
change it and a patch release does not.

## Contributing

Bug reports, mapping fixes and new parity fixtures are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) ([简体中文](CONTRIBUTING.zh-CN.md)) for the
build, test and boundary rules. Please read
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

MIT — see [LICENSE](LICENSE).
