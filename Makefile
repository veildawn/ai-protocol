.PHONY: test vet lint fmt fmt-check cover cover-check bench fuzz goldens

STATICCHECK ?= honnef.co/go/tools/cmd/staticcheck@v0.8.2
COVERAGE_FLOOR ?= 78

test:
	go test ./...

vet:
	go vet ./...

lint:
	go run $(STATICCHECK) ./...

fmt:
	go fmt ./...

fmt-check:
	@out="$$(gofmt -l .)"; \
	if [ -n "$$out" ]; then \
		echo "unformatted files:"; echo "$$out"; exit 1; \
	fi

cover:
	go test -coverpkg=./... -coverprofile=coverage.out ./... && go tool cover -func=coverage.out | tail -1

cover-check:
	@go test -coverpkg=./... -coverprofile=coverage.out ./... >/dev/null
	@total="$$(go tool cover -func=coverage.out | awk '/^total:/ {sub(/%/, "", $$3); print $$3}')"; \
	awk -v t="$$total" 'BEGIN { \
		printf "statement coverage: %.1f%%\n", t; \
		if (t + 0 < $(COVERAGE_FLOOR)) { printf "below the %s%% floor\n", "$(COVERAGE_FLOOR)"; exit 1 } \
	}'

bench:
	go test -run '^$$' -bench . -benchmem ./...

fuzz:
	go test -run '^$$' -fuzz '^FuzzConvertRequest$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzConvertResponse$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzPipeStream$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzFramer$$' -fuzztime 10s .

goldens:
	python3 scripts/gen_goldens.py
