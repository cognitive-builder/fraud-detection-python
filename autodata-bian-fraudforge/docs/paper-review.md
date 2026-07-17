# Paper and supplied-asset review

## Executive reading of Autodata

Autodata reframes synthetic-data generation as an iterative data-science activity rather than a
one-shot prompting task. The agent first creates data, then inspects examples and dataset-level
metrics, derives lessons, revises the generation recipe, and repeats until a stopping condition is
satisfied. An outer loop can then optimize the data-scientist harness itself using held-out data.

The paper’s practical implementation, Agentic Self-Instruct, has four operative roles:

| Role | Paper function | FraudForge implementation |
|---|---|---|
| Challenger | proposes candidate training/evaluation examples | parameterized fraud-scenario generator |
| Weak solver | target model or bounded-capability solver | transparent rule engine |
| Strong solver | higher-capability reference solver | calibrated contextual ensemble |
| Verifier/Judge | checks solution and example quality | deterministic evidence/schema verifier |

The orchestrator reads the weak/strong outputs and judge feedback, changes the challenger recipe,
and tries again. FraudForge preserves this control loop while replacing expensive LLM calls with
inspectable, reproducible components suitable for a public CPU demo.

## The most important result is not “make it harder”

The computer-science experiments mostly needed to lower weak-solver performance while keeping the
strong solver successful. The legal experiment exposed the opposite failure mode: initial tasks
were so hard that weak rollouts clustered near zero, producing poor GRPO variance. The agent made
the questions more learnable, increasing weak-solver mean and variance while retaining a useful
strong/weak gap.

That result drives FraudForge’s acceptance policy. It uses a **band**, not a monotonic hardness
score. Weak recall must be neither too high nor too low; the strong/weak gap has both lower and
upper bounds; and weak-score standard deviation must remain non-trivial.

## Meta-optimization lesson

The paper’s outer-loop experiment treats the data-scientist prompt/scaffold as code. Changes are
accepted only when held-out weak/strong separation improves. FraudForge implements a lightweight
analogue: after deterministic feedback proposes a revised recipe, an evolutionary search perturbs
the recipe and evaluates candidates on two held-out seeds. The winning candidate becomes the next
round’s policy.

This is intentionally modest. It demonstrates the architecture without claiming to reproduce the
paper’s LLM-scale meta-training.

## Limitations translated into engineering controls

| Paper risk | FraudForge response |
|---|---|
| agent can game the weak solver | weak rules are fixed and hidden from recipe revision details |
| invalid or meaningless “hard” examples | deterministic planted-signature and numeric checks |
| example-level quality misses corpus issues | coverage, duplication, prevalence, and family metrics |
| judge/reward can be brittle | primary verifier is programmatic; reward components are exported |
| outer-loop overfitting | recipe candidates are evaluated on held-out seeds |
| removing humans is undesirable | production roadmap includes expert approval and change control |

## Review of `autodata-policy-cs`

The supplied Hugging Face artifact is a LoRA adapter on a small Qwen base model. Its strongest value
for this project is its candid model card: the infrastructure ran, but reward stayed approximately
flat; all generations hit the completion cap; the reward was too weak; and only five accepted
prompts were available. The card explicitly distinguishes a completed training job from measurable
policy improvement.

FraudForge therefore does **not** load this adapter in the default runtime. It adopts four lessons:

1. Keep the default loop deterministic and independently testable.
2. Expose reward components and termination conditions.
3. Require hundreds of bank-specific trajectories before policy training.
4. Validate any learned recipe policy against a deterministic baseline and held-out seeds.

## Review of `verified-analytics-tasks`

This dataset makes deterministic verification the product. Each task has a seeded fixture,
reference function, and checker; plausible traps are explicitly tested; the reference answer must
pass its own checker; and the release is compiled into separate experience, fixture-generator, SFT,
rollout, preference, verifier, and rubric tables.

FraudForge mirrors that artifact contract. A bank team can use the exported tables for supervised
training, preference optimization, RL with verifiable rewards, regression testing, or audit review
without reconstructing provenance from a single flat CSV.

## Review of `vi-gsm8k-agentic`

This dataset is a compact applied pattern for Agentic Self-Instruct. Its orchestrator coordinates a
Challenger, Strong Solver, Weak Solver, and Verifier, then keeps examples only when the strong
solver passes, the weak solver fails, verification passes, and the item is not a duplicate. It also
uses code execution as the primary answer oracle and evaluates downstream fine-tuning on both
in-distribution and out-of-distribution sets.

FraudForge preserves the four-role gate but broadens it in two ways:

- It accepts a bounded range rather than requiring weak failure for every item.
- It adds dataset-level and banking-governance metrics, because production fraud curricula need
  legitimate hard negatives, false-positive controls, and scenario coverage.

## What is reproduced—and what is not

**Reproduced conceptually:** iterative recipe learning, weak/strong solver feedback, deterministic
verification, bounded acceptance, outer-loop held-out selection, exportable training/evaluation
artifacts, and downstream utility testing.

**Not claimed:** reproduction of the paper’s exact models, prompts, compute scale, GRPO training,
reported benchmark gains, or a production-grade fraud model. The demo is an architecture and
engineering proof, not an empirical replication study.
