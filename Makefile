.PHONY: test goldens
test:
	go test ./...
goldens:
	python scripts/gen_goldens.py
