.PHONY: all install test lint format check website clean dist

all: check test

install:
	uv sync

test:
	uv run python -m pytest tests/ -v

lint:
	uv run ruff check swival/ tests/

format:
	uv run ruff format swival/ tests/

check: lint
	uv run ruff format --check swival/ tests/

website:
	uv run --group website python build.py

clean:
	rm -rf dist/ __pycache__ swival/__pycache__ tests/__pycache__ .pytest_cache
	find . -name '*.pyc' -delete

dist: clean
	uv build
	uv run python scripts/generate_formula.py
