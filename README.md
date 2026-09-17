# ai-protocol

Go SDK that converts JSON bodies and SSE streams between three public LLM dialects:

- `chat` — OpenAI Chat Completions (`/v1/chat/completions`)
- `messages` — Anthropic Messages (`/v1/messages`)
- `responses` — OpenAI Responses (`/v1/responses`)

Mappings follow [LiteLLM](https://github.com/BerriAI/litellm) **v1.100.1**. The library does not send HTTP and does not implement provider-specific dialects.

```go
import "github.com/veildawn/ai-protocol"

out, err := protocol.ConvertRequest(protocol.Chat, protocol.Messages, body)
resp, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, upstream)
n, err := protocol.PipeStream(protocol.Messages, protocol.Chat, dst, src, protocol.StreamOpts{Flush: flush})
```

Same-dialect calls return the original bytes unchanged.
