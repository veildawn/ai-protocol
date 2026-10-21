# Contributing to ai-protocol

Thanks for taking the time to contribute. This project is small on purpose, and
the rules below are what keeps it small.

## Before you start

- Read the contract in [README.md](README.md#design-guarantees). It describes
  what the library is allowed to do.
- For anything larger than a bug fix, open an issue first so the design can be
  agreed on before code is written.
- Security issues go to [SECURITY.md](SECURITY.md), not to a public issue.

## Development setup

You need Go 1.23 or newer. There is nothing else to install: the module has no
third-party dependencies.

```sh
git clone git@github.com:veildawn/ai-protocol.git
cd ai-protocol
go test ./...
```

## Checks

Run these before opening a pull request:

```sh
make fmt-check   # gofmt
make vet         # go vet
make test        # go test ./...
make cover       # coverage summary
```

CI runs the same commands on Linux, against the oldest supported Go version and
the latest stable one.

## The boundary

[`boundary_test.go`](boundary_test.go) is the border of the library, written as
a test. It fails the build when a change:

- adds a public declaration that is not in `boundarySurfaceAllowlist`,
- adds a third-party or `net/` import,
- adds a `require` directive to `go.mod`,
- mentions vendor vocabulary in non-test sources,
- adds or renames a public dialect.

If your change is genuinely public-dialect conversion, update the allowlist and
the contract in `README.md` in the same commit. That pairing is the review the
guard exists to force. If it is not, the change belongs in the calling
application, not here.

## Adding a mapping fix

Mapping behavior is pinned by tests in two layers:

1. `litellm_parity_*_test.go` — compatibility fixtures for the corresponding
   direction and shape (request, response, SSE).
2. `convert_test.go`, `*_test.go` in each dialect package — package-level
   behavior, error paths and edge cases.

A mapping fix should come with a test that fails before the fix and passes
after it. Name the fixture after the protocol fact it protects, not after the
provider that reported it.

## Golden fixtures

`testdata/goldens/` is generated from the pinned reference version, not edited
by hand:

```sh
pip install litellm==1.100.1
make goldens
```

Review the regenerated files as part of the pull request. A diff in `testdata/`
without a corresponding test change is suspicious.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(stream): expose event facts and incremental SSE framing
fix(gemini folds): read the thought signature in either spelling
test: pin the folds a reference audit found diverging
docs: describe the loss API
```

Scope is the package or dialect pair: `chat`, `messages`, `responses`, `gemini`,
`images`, `stream`, `boundary`, `docs`.

## Pull requests

- Keep the change focused. Unrelated refactors belong in a separate pull
  request.
- Describe the protocol fact the change fixes, not just the symptom.
- Make sure `make fmt-check vet test` passes.
- Update `README.md` or `doc.go` when public behavior changes.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE), without any additional terms or conditions.
