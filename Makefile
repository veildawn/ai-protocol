.PHONY: test vet fmt fmt-check cover goldens

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

goldens:
	python3 scripts/gen_goldens.py
