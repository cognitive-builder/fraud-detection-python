# Implementation guide

## Repository structure

```text
fraudforge/
  config.py          recipe and acceptance policy
  generator.py       synthetic challenger and evidence flags
  verifier.py        deterministic quality oracle
  solvers.py         weak rules and strong ensemble
  evaluator.py       batch metrics and bounded gates
  orchestrator.py    inner loop and held-out meta-search
  utility.py         downstream before/after experiment
  bian.py            BIAN request normalization and response mapping
  artifacts.py       RL/SFT/evaluation export contract
app.py               Gradio Space
api.py               FastAPI service
scripts/             sample and Space publishing utilities
tests/               unit and integration tests
docs/                GitHub Pages source
social/              launch-ready content
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]
pytest -q
python -m fraudforge --output artifacts/latest
```

## Gradio demo

```bash
python app.py
```

The interface exposes initial recipe controls and returns:

- an acceptance summary;
- round-by-round trajectory;
- family-level weak/strong behavior;
- verified transaction preview;
- BIAN request and response examples; and
- a complete artifact ZIP.

## FastAPI service

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `/docs` while the service is running.

## Adding a fraud family

1. Add the family to `FRAUD_SCENARIOS` and its default probability.
2. Implement a generator branch that plants a multi-feature signature.
3. Add an independent `evidence_flags()` definition.
4. Add a reason code.
5. Decide whether the weak rules should detect it fully, partially, or not at all.
6. Add a legitimate hard-negative variant that contains only a partial clue.
7. Extend tests for signature validity and family coverage.
8. Document the defensive rationale and governance owner.

## Optional LLM policy layer

A future version can replace deterministic recipe revision with a learned or prompted data-scientist
policy. The safe integration boundary is narrow:

```text
metrics + family diagnostics + prior recipe
                 ↓
          policy proposal
                 ↓
       Recipe schema validation
                 ↓
 held-out deterministic candidate evaluation
```

The policy must never write oracle labels, alter verifier definitions, or see held-out seeds. Its
proposal remains advisory until deterministic gates and change controls pass.
