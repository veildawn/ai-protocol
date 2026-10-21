## What changed

<!-- The protocol fact this fixes, not just the symptom. -->

## Direction and shape

<!-- e.g. messages -> chat, SSE; or images -> gemini, request body -->

## Tests

<!-- Which fixture or test pins this? A fix should come with one that failed before it. -->

## Checklist

- [ ] `make fmt-check vet test` passes
- [ ] `boundary_test.go` passes, or the allowlist and README contract were updated together
- [ ] No new dependency, no `net/` import, no vendor-specific behavior
- [ ] Public behavior changes are reflected in `README.md` or `doc.go`
- [ ] `testdata/goldens/` was regenerated only if the reference mapping changed
