# Design notes

This document explains why the library is shaped the way it is. It is not API
documentation — that lives in the package comments and on pkg.go.dev.

## The problem

An API gateway that accepts several public API shapes has an N×M problem. Every
client dialect that must reach every upstream dialect is another pair of
mappings, and written inside HTTP handlers those mappings drift: each one grows
its own edge cases, its own idea of what a missing field means, its own silent
drops. The transport code and the protocol code become impossible to test
separately.

## What this library owns

- Request body → request body.
- Response body → response body.
- SSE stream → SSE stream.
- The protocol facts a conversion cannot preserve, named.

## What it deliberately does not own

- HTTP: no connection, no header, no status code.
- Routing: which upstream, which credential, which model.
- Policy: retries, billing, quota, redaction.
- Vendor behavior: a hostname, a brand-specific parameter repair, a quirk of
  one deployment. Those belong to the caller, which is the only party that
  knows which upstream it is talking to.

Keeping those out is what makes the conversions pure, and pure is what makes
them testable: the same input produces the same output on every machine.

## The fold model

A conversion is a fold from one wire's semantics onto another's.

**Same dialect is the identity.** `ConvertRequest(Chat, Chat, body)` returns
`body` unchanged, even when `body` is not valid JSON. There is nothing to
translate, and a passthrough that can fail is a liability in a proxy.

**Different dialects are total, but not invertible.** A fold answers every
input with either a target body or a typed error; it never invents a value the
target requires and the source cannot supply. `Messages` requires `max_tokens`;
a source without one gets `MissingRequiredFieldError` unless the caller passes
`ConvertOptions.MaxTokensFallback`. Guessing a ceiling would be vendor
knowledge, and a guessed ceiling is a request that silently truncates.

**The two directions are separate mappings.** `A → B` and `B → A` are written
and tested independently. Treating one as the inverse of the other is how
lossy conversions become silently lossy: the inverse of a drop is an invention.

## Losses are data

Some target wires cannot express what a source expresses — logprobs, a seed, a
native parameter, a file reference. Rather than dropping them silently:

- `StaticLosses(from, to)` says what a pair can lose, independent of a body.
- `BodyLosses(from, to, body, opts)` says what *this* body loses.
- `ConvertOptions.DropParams` decides whether an unrepresentable parameter is
  omitted or reported as `UnsupportedParamError`.

A `Loss` is a protocol fact. It is not a routing signal, not a billing signal,
and not advice — the caller decides what an acceptable loss is for its use case.

## Streams settle; they do not get closed

An SSE stream can end in more than one way, and a proxy must distinguish them:

| Outcome | Meaning |
| ------- | ------- |
| `Terminal` | The source emitted its own terminal frame. |
| `Truncated` | The source stopped without one. The turn is left open on purpose. |
| `InBandErr` | The source framed a failure inside a successful stream. |

The rule is that **a converter never closes a turn the source did not close**.
A client that sees a normal close when the upstream actually died will treat a
partial answer as complete. The exception is a Chat source, because several
vendors close on a usage chunk rather than `[DONE]`.

When a host aborts a delivery itself — timeout, cancellation, a policy
decision — `WriteTerminalError` mints a terminal error frame in the client's own
wire, so the client can place the failure instead of watching a connection go
quiet.

## The boundary is a test

Architecture decays unless something fails when it is violated. `boundary_test.go`
turns the design into executable rules:

- no `require` directives and no third-party imports;
- no `net` or `net/http` import;
- no vendor vocabulary in non-test sources — a denylist of brand names, hosts
  and vendor-only spellings;
- a closed set of five dialects;
- an allowlist of exported declarations, per package.

Adding public API is allowed. It just has to be a deliberate act: the allowlist
and the contract in the README change in the same commit, which is the review
the guard exists to force.

## Compatibility

The mappings are pinned against LiteLLM v1.100.1, whose MIT-licensed protocol
mapping this project derives from (see [NOTICE](../NOTICE)). Two mechanisms keep
that honest:

- `litellm_parity_*_test.go` — end-to-end fixtures per direction and shape.
- `testdata/goldens/` — bodies generated from the reference itself, reviewed
  when the pin moves.

The reference is a specification, not a dependency: nothing from it is imported
or shipped.

## Testing strategy

| Layer | What it proves |
| ----- | -------------- |
| Package tests | Behavior, error paths, edge cases. |
| Parity fixtures | Agreement with the pinned reference. |
| Golden files | Agreement on full bodies, regenerated deliberately. |
| Fuzz targets | No input a client or upstream controls panics the codec. |
| Benchmarks | The conversions stay allocation-bound, not I/O-bound. |

## Trade-offs accepted

- **Bytes in, bytes out.** A conversion decodes to `map[string]any` and
  re-marshals. That costs allocations, but it matches what a gateway already
  has (a body) and keeps the API free of struct types that would freeze one
  vendor's schema.
- **Explicit per-pair code.** No reflection, no schema-driven mapping. Verbose,
  but greppable and reviewable — which matters more than line count for a
  protocol codec.
- **Vendor quirks stay outside.** Callers repeat some work that a vendored
  adapter would hide. In exchange the codec has no opinion about where it runs
  or who it talks to.
