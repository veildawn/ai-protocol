# Examples

Runnable programs that use the library the way a gateway or proxy would.

| Directory            | Shows                                                        |
| -------------------- | ------------------------------------------------------------ |
| [`convert`](convert)   | A request fold and a response fold between two public dialects |
| [`streaming`](streaming) | An SSE turn piped from one dialect into another, with settlement |

Run one from the repository root:

```sh
go run ./examples/convert
go run ./examples/streaming
```

The examples keep the transport out of the picture: they convert in-memory
bytes so the conversion itself is the only thing on screen. A real host reads
the body from an HTTP request, resolves the upstream, calls the codec, and
writes the result back.
