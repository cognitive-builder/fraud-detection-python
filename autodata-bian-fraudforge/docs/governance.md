# Governance and responsible use

## Control objectives

A bank adopting agentic synthetic data should govern the **data-generation system** as a model and
as a data-processing pipeline. The relevant control surface includes the generator, solver pair,
verifier, acceptance policy, source grounding, exported labels, and any learned recipe policy.

## Required production controls

| Area | Minimum control |
|---|---|
| data privacy | no raw PII in prompts, logs, exports, or public demos |
| provenance | recipe, source, seed, verifier version, and solver version per batch |
| model risk | independent validation of labels, target bands, and downstream claims |
| fairness | subgroup performance and error-cost review on legitimate activity |
| security | authentication, rate limits, secret management, audit logs, dependency scanning |
| human oversight | fraud-domain approval for new scenarios and material recipe changes |
| change management | versioned gates, rollback, champion/challenger, reproducible release bundles |
| monitoring | drift, family coverage, duplicate yield, false-positive rate, verifier failures |
| retention | explicit storage, deletion, and access policies for generated and source data |

## Avoiding reward hacking

The generator and policy must not be able to:

- edit the weak solver to force failure;
- alter oracle labels after generation;
- weaken the verifier or acceptance threshold;
- see frozen holdout labels or seeds;
- hide invalid rows through aggregation; or
- increase reward by producing nonsensical but mechanically difficult examples.

FraudForge enforces these boundaries structurally in the demo. Production controls should add
separate repositories/permissions for generation and validation, signed release artifacts, and
independent review.

## Synthetic-data risk

Synthetic data reduces direct exposure to customer records, but it is not automatically private,
fair, representative, or safe. A model can reproduce biases from its grounding data, overfit to
stylized scenarios, or create unrealistic correlations. Generated labels remain hypotheses until
validated against institution-specific knowledge and controlled real-data evaluation.

## Human-in-the-loop operating model

A practical target state is co-improvement:

1. the agent identifies weak coverage and proposes scenario changes;
2. fraud specialists review assumptions and label semantics;
3. deterministic and statistical checks run automatically;
4. model-risk staff validate downstream evidence;
5. approved batches enter a controlled training/evaluation registry; and
6. production outcomes feed back only after privacy, quality, and governance checks.
