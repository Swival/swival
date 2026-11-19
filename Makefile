.PHONY: all install test lint format check website clean dist sign release formula build-constraints

# Release builds run on a pinned uv-managed interpreter, because the Python
# version leaks into the sdist bytes (3.15 lowered gzip's default level).
BUILD_PYTHON = 3.15

all: check

install:
	uv sync

test:
	uv run python -m pytest tests/ -v -m 'not stress' --durations=25

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

# Built from a clean export of HEAD, so uncommitted or ignored local files
# can't end up in the archives.
# SOURCE_DATE_EPOCH is cleared because it changes every archive timestamp.
dist: clean
	mkdir -p dist/src
	git archive HEAD | tar -x -C dist/src
	cd dist/src && env -u SOURCE_DATE_EPOCH uv build --python $(BUILD_PYTHON) --managed-python \
		--build-constraints build-constraints.txt --require-hashes --out-dir ..
	rm -rf dist/src

sign:
	minisign -Sm dist/*.tar.gz dist/*.whl

release:
	git diff --quiet HEAD || { echo "Commit local changes first" >&2; exit 1; }
	$(MAKE) check
	$(MAKE) test
	$(MAKE) dist
	$(MAKE) sign

formula:
	uv run python scripts/generate_formula.py

build-constraints:
	uv run --no-project --python $(BUILD_PYTHON) python -I -c \
		'import tomllib; print(*tomllib.load(open("pyproject.toml", "rb"))["build-system"]["requires"], sep="\n")' \
		| uv pip compile - --upgrade --generate-hashes --universal --python-version $(BUILD_PYTHON) \
		--custom-compile-command 'make build-constraints' -o build-constraints.txt
