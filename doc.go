// Package protocol converts request bodies, response bodies and server-sent
// event streams between the public LLM wire formats.
//
// Five public dialects are supported: Chat (the chat-completions wire),
// Messages (the messages wire), Responses (the responses wire), Gemini (the
// generateContent wire) and Images (the image-generation wire). The four text
// dialects all convert between each other, in both directions, for requests,
// responses and SSE. Images pairs with Gemini on request and
// response bodies only: image generation does not stream, so that pair has no
// SSE fold. A same-dialect conversion returns the original bytes unchanged.
//
// # Design
//
// Every conversion is a pure function. The package depends only on the
// standard library, performs no I/O, opens no connection and reads no clock.
// Routing, retries, credentials and vendor-specific behavior belong to the
// caller.
//
// A conversion into Messages needs that wire's required max_tokens. The fold
// derives it from a thinking budget when the source carries one, and otherwise
// asks for [ConvertOptions.MaxTokensFallback] rather than inventing a value.
// Guessing a ceiling would be vendor knowledge the codec does not claim to
// have; omitting the field would produce a body the upstream rejects.
//
// # Reporting what was lost
//
// [StaticLosses] and [BodyLosses] name the request features a directed
// conversion cannot preserve. A [Loss] is a protocol fact, not a policy
// decision: the caller decides whether a given loss is acceptable, and
// [ConvertOptions.DropParams] controls whether an unrepresentable parameter is
// omitted or reported as an [UnsupportedParamError].
//
// # Streams
//
// [PipeStream] copies a server-sent event stream from one dialect to another.
// [PipeStreamWith] additionally reports how the source ended — terminal,
// truncated, or an in-band failure — so the caller can settle the turn without
// re-reading the stream. A converter never closes a turn the source did not
// close. The stream subpackage exposes incremental framing and event
// classification for hosts that need to observe a stream while forwarding it;
// [StreamOpts.OnEvent] is the seam for rewriting decoded events before they
// are written.
//
// # Boundary
//
// The public surface is deliberately small and guarded by tests in the
// repository. Adding a dialect, a dependency or a public declaration is a
// boundary decision, not an implementation detail: it changes what every
// consumer inherits.
package protocol
