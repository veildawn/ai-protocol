# Sentinel Security Journal

## 2025-11-04 - Unbounded Memory Growth in Multi-line SSE Event Framing
**Vulnerability:** In `stream.Framer`, individual lines were checked against `maxEventBytes` (8 MiB), but multi-line events (multiple `data:` lines within one SSE event) could accumulate payload data without limit until flushed or OOM crash.
**Learning:** SSE parsers often apply buffer/scanner limits on per-line basis (such as `bufio.Scanner.Buffer`), leaving an accumulation gap when combining multi-line fields (`data:` lines joined by `\n`).
**Prevention:** Track total accumulated event payload size (`dataBytes`) across multi-line chunks in the framer and enforce `maxEventBytes` on each addition before appending to the buffer.
