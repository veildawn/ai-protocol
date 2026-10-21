# ai-protocol

[English](README.md) | [简体中文](README.zh-CN.md)

[![Go Reference](https://pkg.go.dev/badge/github.com/veildawn/ai-protocol.svg)](https://pkg.go.dev/github.com/veildawn/ai-protocol)
[![CI](https://github.com/veildawn/ai-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/veildawn/ai-protocol/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/veildawn/ai-protocol)](https://goreportcard.com/report/github.com/veildawn/ai-protocol)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个 Go 库，在公开的 LLM 线路格式之间转换请求体、响应体和 SSE 流。它由纯函数
组成，除标准库外没有任何依赖，不做 I/O，也不访问网络。

| 方言        | 线路格式                                                          |
| ----------- | ----------------------------------------------------------------- |
| `chat`      | OpenAI Chat Completions（`/v1/chat/completions`）                  |
| `messages`  | Anthropic Messages（`/v1/messages`）                               |
| `responses` | OpenAI Responses（`/v1/responses`）                                |
| `gemini`    | `generateContent` 线路（`contents`/`parts`、`candidates`）          |
| `images`    | 图像生成线路（`prompt`/`n`/`size`，响应为 `b64_json`）              |

## 为什么需要它

一个 API 网关、代理或 SDK，只要接受不止一种公开 API 形态，就必须在它们之间做
转换。按"对"实现的话，就是 N×M 份映射代码：彼此逐渐偏离、对边界情况的处理互相
矛盾，还把协议事实埋进 HTTP 处理函数里。

这个库把转换从传输层里拿出来：每个映射都是作用于字节或 `map[string]any` 的纯
函数，转换会明确命名它无法保留的协议事实，而这个包不建立连接、不读时钟、也不知
道自己在跟哪个上游对话。路由、重试、凭证和厂商特定行为都留给调用方。

## 支持的转换

四种文本方言之间两两互转。`images` 只与 `gemini` 配对，且仅限请求体和响应体
—— 图像生成不流式，因此这一对没有 SSE 折叠。

| 源 ↓ / 目标 → | `chat` | `messages` | `responses` | `gemini` | `images` |
| ------------- | ------ | ---------- | ----------- | -------- | -------- |
| `chat`        | 原样   | 请求/响应/SSE | 请求/响应/SSE | 请求/响应/SSE | — |
| `messages`    | 请求/响应/SSE | 原样 | 请求/响应/SSE | 请求/响应/SSE | — |
| `responses`   | 请求/响应/SSE | 请求/响应/SSE | 原样 | 请求/响应/SSE | — |
| `gemini`      | 请求/响应/SSE | 请求/响应/SSE | 请求/响应/SSE | 原样 | 请求/响应 |
| `images`      | — | — | — | 请求/响应 | 原样 |

同方言调用原样返回输入字节。

## 安装

```sh
go get github.com/veildawn/ai-protocol@latest
```

需要 Go 1.23 或更新版本。模块没有第三方依赖。

## 快速开始

```go
package main

import (
	"fmt"

	protocol "github.com/veildawn/ai-protocol"
)

func main() {
	// 客户端发来的是 chat-completions 请求体，上游说 messages 方言。
	body := []byte(`{"model":"example-model","messages":[{"role":"user","content":"hi"}]}`)

	req, err := protocol.ConvertRequestWith(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{
		// messages 线路要求 max_tokens。无法从思考预算推导出的折叠会向调用方
		// 索要这个值，而不是编造一个。
		MaxTokensFallback: 8192,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("upstream body: %s\n", req.Body)

	// ... 把 req.Body 发给上游，读取它的响应 ...

	upstream := []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}]}`)
	out, err := protocol.ConvertResponse(protocol.Messages, protocol.Chat, upstream)
	if err != nil {
		panic(err)
	}
	fmt.Printf("client body: %s\n", out)
}
```

两个入口都返回类型化错误 —— `UnsupportedParamError`、
`MissingRequiredFieldError`、`ConversionError` —— 因此宿主可以用 `errors.As`
匹配，而不是比较错误字符串。

可直接运行的程序在 [examples/](examples)：一个做请求与响应的转换，一个把一轮
SSE 对话透传过去。

`ConvertRequest` 还会返回它从源请求体里读到的模型名，让路由可以在任何转换发生
之前就确定上游。`PeekModel` 做同样的事但不做完整解码，适合先路由的宿主。

## 流式转换

```go
// w 是 http.ResponseWriter；upstream 是上游的响应体。
result, err := protocol.PipeStreamWith(protocol.Messages, protocol.Chat, w, upstream, protocol.StreamOpts{
	Flush: func() { w.(http.Flusher).Flush() },
})
if err != nil {
	// 传输或解码失败。
}
switch {
case result.InBandErr != nil:
	// 源在一条成功的流里框入了一个失败。
case result.Truncated:
	// 源没有发出自己的终止帧就结束了。这一轮对话是故意保持打开的：
	// 只有源有权关闭它。
case result.Terminal:
	// 源自己关闭了这一轮对话。
}
```

`StreamResult` 报告源流是如何结束的，与写入目标的字节数（`result.Written`）分
开。转换器绝不会关闭源没有关闭的一轮对话。

对于库不该知道的策略 —— 脱敏、工具名改写、埋点 —— 传
`StreamOpts.OnEvent`。它会看到每一个目标方言事件，就在写出之前；同方言路径是原
样拷贝，不会调用它。

`stream.Framer` 从被拆分的读取中逐个产出完整的 SSE 事件。
`stream.ClassifyEvent` 报告单个事件的协议事实（前奏、内容、推理、完成、未完成、
失败、未知），但不决定重试或计费策略。

## 报告丢失的参数

转换可能丢弃或重塑目标方言无法承载的内容。loss 会明确命名这些事实，调用方可以
在转换之前或同时决定某个省略是否可接受：

```go
losses := protocol.BodyLosses(protocol.Chat, protocol.Messages, body, protocol.ConvertOptions{})
if losses[protocol.LossLogprobs] {
	// messages 线路没有 logprobs；判断这是否可以接受。
}
```

`StaticLosses(from, to)` 返回某个有向配对可能发生的全部丢失项，与具体请求体无
关。`Loss` 值是描述性的协议事实，永远不是路由策略。完整列表见
[loss.go](loss.go)。

## 错误

| 错误                       | 含义                                                       |
| -------------------------- | ---------------------------------------------------------- |
| `UnsupportedParamError`    | 目标方言无法承载某个参数；设置 `DropParams` 可省略           |
| `MissingRequiredFieldError`| 目标方言要求某个字段，而调用方没有提供                       |
| `ConversionError`          | 请求体无法被读取或重建                                       |

错误是类型化的，需要时带包装；用 `errors.As` 匹配。

## 设计保证

以下保证由 `boundary_test.go` 强制，违反任何一条都会让构建失败：

- **只用标准库。** 没有 `require` 指令，没有第三方 import。
- **离线。** 不引入 `net`，不碰 HTTP，不读时钟，不碰文件系统。
- **纯转换。** 同样的输入得到同样的输出；没有全局状态。
- **五种方言，封闭集合。** 新增一种线路格式是一次有意的边界决定，而不是实现细节。
- **没有厂商行为。** 厂商特定的参数修补、主机名、凭证和怪癖属于调用方，不属于
  codec。
- **没有策略。** 路由、计费、配额和重试的决定都留在外面。

## 测试

```sh
make test    # 单元、兼容性、边界和属性测试
make fuzz    # 每个目标模糊测试 10 秒：转换、SSE 管道、分帧器
make bench   # 转换、管道和分帧器基准测试
```

模糊测试把输入当成敌意数据：客户端或上游能产生的任何请求体、响应体、流或分块方
式，都不应让 codec panic，也不应让同方言转换改变字节。

转换的瓶颈在内存分配而不在 I/O —— 一次请求折叠在笔记本级 CPU 上是几微秒 —— 因
此 `make bench` 除了 `ns/op` 和分配次数，还会报告 MB/s 吞吐。

## 状态

1.0 之前（`v0.8.x`）。公开接口很小，并由测试固定；版本遵循
[SemVer](https://semver.org/)，因此次要版本仍可能改变它，补丁版本不会。

## 兼容性

映射语义按 [LiteLLM](https://github.com/BerriAI/litellm) v1.100.1 验证，本项目
的协议映射派生自其 MIT 许可的实现。归属声明见 [NOTICE](NOTICE)；固定这些行为的
兼容性夹具是 `litellm_parity_*_test.go`。

## 参与贡献

欢迎提交缺陷报告、映射修正和新的兼容性夹具。构建、测试和边界规则见
[CONTRIBUTING.md](CONTRIBUTING.md)。请先阅读 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
安全问题请按 [SECURITY.md](SECURITY.md) 处理，不要开公开 issue。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
