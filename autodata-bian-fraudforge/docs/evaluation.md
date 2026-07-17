# Evaluation and experiment design

## Evaluation levels

FraudForge separates four questions that are often conflated:

1. **Is each row valid?** Planted signatures, labels, and numeric constraints.
2. **Is the batch useful?** Weak/strong separation, variance, false positives, coverage, duplicates.
3. **Did the recipe improve?** Round trajectory and held-out candidate score.
4. **Can the data improve a learner?** Fixed before/after model on a shifted holdout.

## Classification metrics

Both solvers are measured against the deterministic oracle:

- precision;
- recall;
- F1;
- false-positive rate; and
- score standard deviation.

The strong-minus-weak recall gap is the main discrimination measure, but it is not sufficient by
itself. A large gap can arise because the weak solver fails on everything, which may provide poor
learning signal.

## Dataset-quality gates

| Metric | Reason |
|---|---|
| verifier validity | prevents invalid or label-inconsistent examples |
| family coverage | avoids a high-scoring but narrow curriculum |
| duplicate rate | protects effective sample size and preference quality |
| legitimate false positives | checks specificity and hard-negative quality |
| weak-score variance | rejects all-zero or all-one weak behavior |
| preference-pair rate | quantifies strong-only successes useful for DPO/RL |

## Outer-loop validation

Each recipe proposal is evaluated on two seeds that are not used for the current training batch.
The selection score is the mean quality score minus a small penalty for every failed gate. This
reduces the chance that one favorable batch drives recipe selection.

For a production program, extend this to:

- multiple temporal and geographic splits;
- scenario-family holdouts;
- institution-specific cost-weighted metrics;
- bootstrap confidence intervals;
- champion/challenger tracking; and
- frozen external evaluation sets unavailable to the generator and policy.

## Downstream utility experiment

The included test uses a fixed logistic learner:

1. Train a baseline on a narrower synthetic distribution with limited representation of selected
   fraud families.
2. Train an identical learner on baseline plus the accepted FraudForge curriculum.
3. Evaluate both on a more ambiguous, noisier, shifted distribution.
4. Report ROC-AUC, precision, recall, F1, and deltas.

This is a smoke test of learning utility, not proof of production performance. Its purpose is to
catch a common failure mode: a dataset can score well on internal heuristics yet fail to improve a
separate learner.

## Reproducibility

- Every generator and outer-loop decision is seed-controlled.
- Calibration data uses a separate fixed seed.
- Recipes, target bands, round diagnostics, and selected outputs are serialized.
- The CI workflow runs unit tests and a smoke Autodata experiment.
