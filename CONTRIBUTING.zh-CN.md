# 参与 ai-protocol 贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

感谢你抽出时间参与贡献。这个项目有意保持精简，下面的规则就是让它保持精简的原
因。

## 开始之前

- 阅读 [README.zh-CN.md](README.zh-CN.md) 中的契约。它描述了这个库被允许做什么。
- 对于比缺陷修复更大的改动，请先开 issue，让设计在写代码之前达成一致。
- 安全问题请走 [SECURITY.md](SECURITY.md)，不要开公开 issue。

## 开发环境

需要 Go 1.23 或更新版本。没有其他东西要装：模块没有第三方依赖。

```sh
git clone git@github.com:veildawn/ai-protocol.git
cd ai-protocol
go test ./...
```

## 检查

在开 pull request 之前运行：

```sh
make fmt-check   # gofmt
make lint        # staticcheck
make vet         # go vet
make test        # go test ./...
make cover-check # 覆盖率门限
make fuzz        # 每个目标短时模糊测试
make bench       # 基准测试
```

CI 会在 Linux 上、针对支持的最旧 Go 版本和最新稳定版运行同样的命令，另外还有
`govulncheck` 和 `staticcheck`。

## 边界

[`boundary_test.go`](boundary_test.go) 是写成测试的库边界。以下改动会让构建失败：

- 新增未列入 `boundarySurfaceAllowlist` 的公开声明，
- 新增第三方或 `net/` import，
- 在 `go.mod` 里新增 `require` 指令，
- 在非测试源码中提到厂商词汇，
- 新增或重命名公开方言。

如果你的改动确实是公开方言之间的转换，请在同一次提交里更新白名单和
`README.md` 里的契约。这个配对正是该守卫要强制的评审。如果不是，这个改动属于调
用方应用，不属于这里。

## 添加映射修复

映射行为由两层测试固定：

1. `litellm_parity_*_test.go` —— 对应方向和形态（请求、响应、SSE）的兼容性夹具。
2. `convert_test.go` 和各方言包里的 `*_test.go` —— 包级行为、错误路径和边界情况。

映射修复应当附带一个在修复前失败、修复后通过的测试。用它保护的协议事实命名夹
具，而不是用报告它的厂商命名。

## 黄金夹具

`testdata/goldens/` 由固定版本的参考实现生成，不手工编辑：

```sh
pip install litellm==1.100.1
make goldens
```

请在 pull request 中一并审阅重新生成的文件。`testdata/` 有 diff 却没有对应测试改
动，是可疑的。

## 模糊测试

`fuzz_test.go` 把每个输入都当作敌意数据。当模糊测试目标发现崩溃时，Go 会把失败
的输入写到 `testdata/fuzz/<Target>/`，它就成为永久的回归用例。请把该文件留在
pull request 里 —— 它才是证明修复的测试。

## 提交信息

使用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
feat(stream): expose event facts and incremental SSE framing
fix(gemini folds): read the thought signature in either spelling
test: pin the folds a reference audit found diverging
docs: describe the loss API
```

Scope 是包或方言对：`chat`、`messages`、`responses`、`gemini`、`images`、
`stream`、`boundary`、`docs`。

## Pull request

- 保持改动聚焦。无关的重构请另开 pull request。
- 描述改动修复的协议事实，而不只是症状。
- 确保 `make fmt-check lint vet test` 通过。
- 公开行为变化时更新 `README.md`、`README.zh-CN.md` 或 `doc.go`。

## 许可证

参与贡献即表示你同意你的贡献按 [MIT 许可证](LICENSE) 授权，不附加任何额外条款或
条件。
