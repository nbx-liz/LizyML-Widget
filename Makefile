.PHONY: install test lint format format-check typecheck build build-js dev ci clean

PACKAGE = lizyml_widget

install:
	uv sync --frozen --dev

test:
	uv run pytest --cov=$(PACKAGE) --cov-fail-under=80 -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src/$(PACKAGE)/

build-js:
	cd js && pnpm install && pnpm build

build:
	uv build

dev:
	uv run jupyter lab

ci: lint format-check typecheck test build-js build
	@echo "All CI checks passed"

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache coverage.json
