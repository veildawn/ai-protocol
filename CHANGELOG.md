# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.3] - 2026-10-29

### Fixed

- Gemini folds: a structured-output schema is re-spelled before it leaves the
  gemini wire, the way a declaration's parameters already were. The official
  SDKs state the type enums in the wire's SCREAMING spelling on
  `responseSchema` too, and the out-of-gemini folds relayed the config's schema
  verbatim into the target's `json_schema` — a strict chat upstream validates
  the lowercase JSON Schema vocabulary and answers the uppercase spelling with
  a hard failure that reads as its own outage (measured live as a 5xx on every
  schema-carrying structured-output request against one upstream, with each
  failure also cooling the gateway's account breaker for that provider).
  A stated-but-empty schema now degrades to the plain `json_object` request on
  all three folds, the reading two of them already applied.

## [0.9.2] - 2026-10-28

### Fixed

- Gemini folds: a declaration's schema is read whichever member carries it.
  The wire's original `parameters` (an OpenAPI-subset Schema) gained a
  standard-JSON-Schema sibling the official SDKs state — `parametersJsonSchema`
  lowerCamel, `parameters_json_schema` snake — and the out-of-gemini folds read
  only the legacy member, so a client stating the newer one handed the target a
  schema-less declaration: the model was told a tool existed but not what it
  takes, and answered with a tool call written as plain text instead of a
  structured call the client could execute. Observed 2026-10-28 as 21 tools
  folded with 22KB of schemas silently gone on a DeepSeek upstream. The legacy
  member wins when a declaration states both.

## [0.9.1] - 2026-10-28

### Fixed

- Gemini/Chat folds: a `functionResponse` that names its id now pairs with the
  call it answers. A replayed call carrying a thoughtSignature rides the chat
  wire as `<id>__thought__<sig>` while the result names the bare `<id>` — the
  only one the client was handed, since the signature rides the call and not the
  result on the gemini wire. Writing the bare id left a tool message that
  answered nothing: every signed call in a replayed history read as unanswered
  and the whole turn was refused. Observed 2026-10-28 as thirteen dropped calls
  on a 101,780-token request. An exact id still outranks the suffix scan.

- Gemini folds: `thinkingLevel` is read. Gemini 3 states depth as a named level
  where the codec knew only `thinkingBudget`; the absent budget bucketed as the
  shallowest rung, so a client asking for the deepest reasoning received the
  shallowest and the request SUCCEEDED — no error, and the usage row recorded
  `minimal` as if that had been the ask.

- Gemini folds: a `thinkingConfig` that states no depth no longer answers the
  lowest rung. "Thinking on, depth unspecified" is a request for the target's
  default; turning it into `minimal` substituted this codec's choice for the
  client's. The chat, messages and responses folds now share one reader
  (`internal/thinkingcfg`) instead of three copies that had already drifted.

## [0.8.6] - 2026-10-22

### Fixed

- Gemini/Messages folds: a thinking block the client replays with the fold's
  stand-in signature is folded as an unsigned block (a bare thought part)
  rather than sent to the target as a signature it rejects — the 400 a gateway
  reports as a rejected model request. The same stand-in on a replayed call
  folds to the skip marker.

## [0.8.5] - 2026-10-17

### Fixed

- Gemini/Responses folds: developer-role messages fold into system instructions.

## [0.8.4] - 2026-10-16

### Fixed

- Gemini folds: a `$ref`-bearing tool result stays wrapped as text.

## [0.8.3] - 2026-10-16

### Fixed

- Gemini/Responses folds: anchor the tool-result data URL at the scheme.

## [0.8.2] - 2026-10-16

### Fixed

- Gemini folds: read the thought signature in either spelling.

## [0.8.1] - 2026-10-16

### Fixed

- Gemini folds: stamp the skip marker on every replayed call, in either
  spelling.

## [0.8.0] - 2026-10-15

### Added

- `WriteTerminalError` mints a dialect's terminal error frame, so a host that
  aborts a delivery mid-stream can end it in the client's own wire.

## [0.7.0] - 2026-10-15

### Added

- The `images` dialect (image-generation wire).
- The `images`↔`gemini` fold pair, request and response bodies only.

## [0.6.6] - 2026-10-15

### Fixed

- Responses folds: announce every output item when it opens, and keep
  same-frame tool calls.

## [0.6.5] - 2026-10-14

### Fixed

- Gemini/Messages folds: send the skip marker for a call whose signature was
  lost.

## [0.6.4] - 2026-10-14

### Fixed

- Gemini/Messages folds: lift the turn's thinking signature onto unsigned
  calls.

## [0.6.3] - 2026-10-14

### Fixed

- Gemini folds: fold each thinking block to one signed thought part.

## [0.6.2] - 2026-10-14

### Fixed

- Gemini/Messages folds: merge same-role runs on the reverse fold and skip
  empty-part contents.

## [0.6.1] - 2026-10-14

### Fixed

- Messages folds: sign rebuilt thinking blocks and fold assistant text before
  its calls.

## [0.6.0] - 2026-10-14

### Added

- The `gemini` dialect (`generateContent` wire) and its six conversions.

## [0.5.1] - 2026-10-13

### Added

- `PeekModel` reads the top-level model member without a full decode, for hosts
  that route before converting.

## [0.5.0] - 2026-10-13

### Fixed

- Aligned every dialect fold with the pinned reference version (LiteLLM
  v1.100.1), including cache tokens, refusals, custom tools, reasoning joins and
  structured output.

## [0.4.11] - 2026-10-12

### Fixed

- Streaming: terminate once, and sign a thinking block before it closes.

## [0.4.10] - 2026-10-10

### Fixed

- `chatresponses`: keep one assistant turn on one message.

## [0.4.9] - 2026-10-09

### Fixed

- Length-truncated responses carry `incomplete_details.reason`.
- `messagesresponses`: drop `reasoning.summary` when folding into Messages.

## [0.4.8] - 2026-10-09

### Fixed

- `messagesresponses`: drop `reasoning.summary` when folding Responses into
  Messages.

## [0.4.7] - 2026-10-08

### Added

- `stream.ClassifyEvent` event facts and incremental SSE framing (`stream.Framer`).

## [0.4.6] - 2026-10-08

### Fixed

- `chatresponses`: fold only the effort scalar into chat `reasoning_effort`.

## [0.4.5] - 2026-09-26

### Fixed

- `chatresponses`: consolidate instructions and input system messages into a
  single leading system message.

## [0.4.4] - 2026-09-24

### Fixed

- Observe the source stream across reads, not one read at a time.

## [0.4.3] - 2026-09-24

### Fixed

- Never close a fold the source did not close.

## [0.4.2] - 2026-09-22

### Fixed

- Emit a real item lifecycle in the Responses stream folds.

## [0.4.1] - 2026-09-18

### Changed

- Carry unsupported-parameter failures as a typed error (`UnsupportedParamError`),
  not a string prefix.

## [0.4.0] - 2026-09-18

### Added

- Host-injected `MaxTokensFallback` for conversions into Messages.

## [0.3.0] - 2026-09-18

### Added

- `boundary_test.go`, which pins the public surface, the standard-library-only
  rule, the vendor-vocabulary rule and the closed dialect set.

## [0.2.0] - 2026-09-18

### Added

- `ConvertOptions` (`DropParams`, `MaxTokensFallback`).
- Stream settlement metadata (`StreamResult`).
- Protocol losses (`StaticLosses`, `BodyLosses`, `LossSet`).
- A per-event hook on the stream options (`StreamOpts.OnEvent`).

## [0.1.0] - 2026-09-18

### Added

- Initial release: JSON and SSE conversion between the `chat`, `messages` and
  `responses` dialects.

[Unreleased]: https://github.com/veildawn/ai-protocol/compare/v0.8.5...HEAD
[0.8.5]: https://github.com/veildawn/ai-protocol/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/veildawn/ai-protocol/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/veildawn/ai-protocol/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/veildawn/ai-protocol/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/veildawn/ai-protocol/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/veildawn/ai-protocol/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/veildawn/ai-protocol/compare/v0.6.6...v0.7.0
[0.6.6]: https://github.com/veildawn/ai-protocol/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/veildawn/ai-protocol/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/veildawn/ai-protocol/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/veildawn/ai-protocol/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/veildawn/ai-protocol/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/veildawn/ai-protocol/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/veildawn/ai-protocol/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/veildawn/ai-protocol/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/veildawn/ai-protocol/compare/v0.4.11...v0.5.0
[0.4.11]: https://github.com/veildawn/ai-protocol/compare/v0.4.10...v0.4.11
[0.4.10]: https://github.com/veildawn/ai-protocol/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/veildawn/ai-protocol/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/veildawn/ai-protocol/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/veildawn/ai-protocol/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/veildawn/ai-protocol/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/veildawn/ai-protocol/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/veildawn/ai-protocol/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/veildawn/ai-protocol/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/veildawn/ai-protocol/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/veildawn/ai-protocol/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/veildawn/ai-protocol/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/veildawn/ai-protocol/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/veildawn/ai-protocol/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/veildawn/ai-protocol/releases/tag/v0.1.0
