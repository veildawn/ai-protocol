# Bolt Performance Journal

## 2026-11-08 - Zero-Copy SSE Line Processing on Feed Fast Path
**Learning:** `stream.Framer` reassembled SSE lines by appending incoming bytes into `f.line` buffer before processing them, even when the entire line arrived within a single chunk (the typical streaming fast path). This caused unnecessary heap buffer growth and allocations on every line during SSE streaming.
**Action:** Fast-path complete lines by directly delegating to `processLine` without copying into `f.line` when `len(f.line) == 0`. Combine with `bytes.IndexByte` to avoid allocations for separator slices when parsing field names.
