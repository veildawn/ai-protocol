# Security Policy

## Supported versions

Security fixes land on the latest minor release line. The project is pre-1.0,
so the supported line is the current `v0.x` release; older minors receive no
backports.

| Version  | Supported |
| -------- | --------- |
| `v0.8.x` | ✅        |
| `v0.7.x` | ❌        |
| earlier  | ❌        |

## Reporting a vulnerability

Do not open a public issue for a security problem.

Use GitHub's private vulnerability reporting: open
[Security → Report a vulnerability](https://github.com/veildawn/ai-protocol/security/advisories/new)
on this repository. The report stays private until an advisory is published.

Please include:

- the version or commit affected,
- the input that triggers the problem (a request or SSE body is usually enough),
- what the impact is, and
- whether you would like to be credited in the advisory.

## Scope

The library converts bytes; it does not open connections or handle
credentials. The most relevant classes of issue are therefore:

- a conversion that panics on attacker-controlled input (a malformed body or a
  truncated SSE stream),
- unbounded memory growth while folding a stream,
- a fold that lets content cross a trust boundary it should not (for example,
  a system-level instruction reconstructed from user-controlled parts).

Transport, authentication, quota and routing issues belong to the application
that embeds the library, not to this project.

## Disclosure

We aim to acknowledge a report within a few business days, agree on a fix
timeline with you, and publish an advisory once a patched release is out.
