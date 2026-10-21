.PHONY: test vet fmt fmt-check cover bench fuzz goldens

test:
	go test ./...

vet:
	go vet ./...

fmt:
	go fmt ./...

fmt-check:
	@out="$$(gofmt -l .)"; \
	if [ -n "$$out" ]; then \
		echo "unformatted files:"; echo "$$out"; exit 1; \
	fi

cover:
	go test -coverprofile=coverage.out ./... && go tool cover -func=coverage.out | tail -1

bench:
	go test -run '^$$' -bench . -benchmem ./...

fuzz:
	go test -run '^$$' -fuzz '^FuzzConvertRequest$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzConvertResponse$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzPipeStream$$' -fuzztime 10s .
	go test -run '^$$' -fuzz '^FuzzFramer$$' -fuzztime 10s .

goldens:
	python3 scripts/gen_goldens.py
