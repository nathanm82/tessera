# Contributing

Thanks for taking the time to contribute! This project is small and the bar is simple:
keep it dependency-light, typed, and tested.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you push

The CI runs exactly these three checks, so run them locally first:

```bash
ruff check . && ruff format --check .
mypy tessera
pytest
```

`scripts/check.sh` runs all three in one go.

## Guidelines

- **One concern per pull request.** Small, focused changes are easier to review.
- **Add a test** with any behavior change. The suite must stay fast and offline — do
  not add a test that downloads a model. Use `HashingEncoder` for fixtures.
- **Type everything.** Public functions carry annotations; `mypy tessera` must pass.
- **New backend?** Register it in the appropriate registry and gate heavy imports
  behind an optional extra, imported lazily (see `encoders/clip.py`).
- **Docs.** Update `docs/` or the README when you change user-facing behavior, and add
  a `CHANGELOG.md` entry under *Unreleased*.

## Reporting bugs

Open an issue with a minimal reproduction. The smaller the snippet, the faster the
fix.
