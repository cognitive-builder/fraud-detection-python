# Contributing

1. Create a feature branch.
2. Install with `pip install -e .[dev,docs]`.
3. Add or update tests for behavioral changes.
4. Run `pytest -q`, `ruff check .`, and `mkdocs build --strict`.
5. Describe the BIAN mapping, verifier implications, and any changed acceptance gate.

New fraud scenarios must remain defensive and synthetic. Every scenario needs a deterministic
evidence check and at least one hard legitimate counterexample.
