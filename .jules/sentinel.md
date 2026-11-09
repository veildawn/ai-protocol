# Sentinel Security Journal

## 2025-11-04 - Unbounded Memory Growth in Multi-line SSE Event Framing
**Vulnerability:** In `stream.Framer`, individual lines were checked against `maxEventBytes` (8 MiB), but multi-line events (multiple `data:` lines within one SSE event) could accumulate payload data without limit until flushed or OOM crash.
**Learning:** SSE parsers often apply buffer/scanner limits on per-line basis (such as `bufio.Scanner.Buffer`), leaving an accumulation gap when combining multi-line fields (`data:` lines joined by `\n`).
**Prevention:** Retain one event's data fields in a single buffer capped at `maxEventBytes`, counting the newline that joins fields. A run of empty `data:` fields is part of that total; an empty field still dispatches.

## 2026-11-08 - Unbounded Buffer Accumulation in SSE Stream Observer
**Vulnerability:** In `observeReader`, incoming stream chunks were appended to `tail` until an event delimiter (`\n\n` or `\r\n\r\n`) was observed. When streaming streams without blank-line delimiters (or on same-dialect pass-through where `io.Copy` does not frame events), `tail` could grow indefinitely, leading to memory exhaustion and denial of service (DoS).
**Learning:** Tee-readers and cross-read SSE observers buffering between event boundaries cannot rely on downstream consumers or codecs to limit buffer sizes, especially during passthrough paths (`io.Copy`) or streams with endless comments/events without double newlines.
**Prevention:** Impose an explicit size ceiling (`maxObserverTailBytes`) on unterminated tail buffers in tee/observer readers, aborting with an error if incoming data exceeds the threshold without a boundary delimiter.
