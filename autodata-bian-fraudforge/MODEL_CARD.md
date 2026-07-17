# Model Card: FraudForge Strong Ensemble

## Summary

The Strong Ensemble is a CPU-friendly `ExtraTreesClassifier` combined with a small deterministic
signature score. It is used as the **strong solver** inside FraudForge and as one constituent of the
BIAN-aligned Fraud Evaluation endpoint.

## Intended use

- Evaluate whether a generated synthetic curriculum contains cases that broader contextual
  modeling can solve while bounded rules struggle.
- Demonstrate the BIAN distinction between rule/decision-tree tests and model-based tests.
- Provide reproducible offline behavior without an external model endpoint.

## Training data

The model is calibrated at runtime on 7,000 fictional transaction records generated from a fixed,
separate recipe and seed. No real customer or payment data is used.

## Inputs

Transaction-level behavioral and contextual features such as amount-to-balance ratio, velocity,
distance from home, device/beneficiary age, counterparty fan-in/fan-out, merchant risk, identity
consistency, channel, category, and country.

## Outputs

A fraud-suspicion score in `[0, 1]` and a binary decision at `0.50`. In the public API, the score is
combined with the bounded rule score using a maximum ensemble for demonstration purposes.

## Evaluation

The repository reports recall, precision, false-positive rate, weak/strong gap, score variance,
scenario-family metrics, deterministic-verifier validity, duplicates, and a shifted-holdout
before/after learner test. Results are specific to the synthetic distributions and are not claims
about production fraud performance.

## Limitations

The planted scenarios are stylized, not exhaustive; calibration and test data share a generator
family; the classifier has no temporal graph, causal, or institution-specific features; and the
threshold is not cost-optimized. It must not be used to approve, reject, or block real payments.
