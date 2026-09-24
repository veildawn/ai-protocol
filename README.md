# ai-protocol

A **public dialect codec**: it converts JSON bodies and SSE streams between the
three public LLM dialects, and it does nothing else.

- `chat` — OpenAI Chat Completions (`/v1/chat/completions`)
- `messages` — Anthropic Messages (`/v1/messages`)
- `responses` — OpenAI Responses (`/v1/responses`)

Mappings follow [LiteLLM](https://github.com/BerriAI/litellm) **v1.100.1** (see
[NOTICE](NOTICE)).

## The contract

This library is a leaf. Everything a host needs to know about the boundary is in
this list, and `boundary_test.go` fails the build when a change breaks one of
them.

**What it is**

- Exactly three public dialects, request/response/SSE conversion between any
  pair of them, as pure functions.
- Explicit error semantics: `UnsupportedParamError` for a parameter the target
  dialect cannot carry, `MissingRequiredFieldError` for a field the target
  dialect requires that the caller declined to supply (Messages folds and
  `max_tokens`), `ConversionError` for a body it cannot read, and
  `StreamResult` (`Terminal`/`Truncated`/`InBandErr`) for how a stream ended.
- Offline and standard-library only. No module dependencies, no I/O, no clock.

**What it is not**

- **No HTTP.** It never opens a connection and never sees a header.
- **No provider routing.** It does not choose an upstream, and it knows nothing
  about accounts, quota, billing or policy.
- **No vendor knowledge.** No brand names, no vendor hosts, no vendor-specific
  parameter repair. Those live in a provider plugin, which is also where the
  private wire format lives.

**Where it sits**

```
client public dialect        openai / anthropic / responses, as the client speaks it
        ↓
ai-protocol                  public ↔ public conversion, pure functions
        ↓
public provider dialect      the arm the provider plugin declares it exposes
        ↓
provider plugin              public → vendor wire, vendor quirks, credentials
        ↓
vendor wire
```

A host supplies **policy** through the seams on the options types — `Flush` for
write-through, `OnEvent` for rewriting decoded stream events,
`MaxTokensFallback` for the value of a required field a fold cannot derive —
and the policy itself stays in the host. Nothing in this library has an opinion
about what a hook does or what a supplied value should be.

## Usage

```go
import "github.com/veildawn/ai-protocol"

out, err := protocol.ConvertRequest(protocol.Chat, protocol.Messages, body)
resp, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, upstream)
n, err := protocol.PipeStream(protocol.Messages, protocol.Chat, dst, src, protocol.StreamOpts{Flush: flush})
```

Same-dialect calls return the original bytes unchanged.

Streaming takes an optional per-event hook, applied to every target-dialect
event just before it is written — including the terminal frames a fold
synthesizes. A fold closes the turn for the client only when the **source**
closed it too: Responses and Messages have exactly one legal ending, so a stream
that stops without it is a truncation, the fold leaves the turn open, and
`StreamResult.Truncated` tells the host (which is where the failure the host
records and the failure the client sees become the same failure). A Chat source
is the exception, because several vendors close on a usage chunk instead of
`[DONE]`: its folds still close the turn for the client. It is how a host applies
a rewriting this library does not know about without re-parsing serialized SSE:

```go
result, err := protocol.PipeStreamWith(protocol.Chat, protocol.Responses, dst, src, protocol.StreamOpts{
    Flush: flush,
    OnEvent: func(ev stream.Event) (stream.Event, error) {
        // rewrite ev.Event / ev.Data, or return ev untouched
        return ev, nil
    },
})
```

A nil hook is the identity: the bytes written are exactly what the converter
produces on its own. Hooks are **not** called on the same-dialect path, which is
a raw copy with nothing decoded to hand a hook.

## Changing the boundary

Adding public API, a dependency, a dialect or an import is a boundary decision,
not an implementation detail. `boundary_test.go` will fail and point here: either
update the allowlist in that file *and* this contract, or move the change to the
consumer where the policy belongs.