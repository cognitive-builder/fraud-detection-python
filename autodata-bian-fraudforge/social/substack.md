# From Synthetic Data Generation to an Agentic Data Scientist for Banking

*Building FraudForge: an Autodata-inspired laboratory for BIAN Fraud Evaluation*

Synthetic data has become a standard answer to several hard problems in financial services: fraud
labels are sparse and delayed, real transaction data is sensitive, edge cases are long-tailed, and
manual scenario design is expensive. Yet most synthetic-data programs still behave like content
factories. A generator is prompted or configured, a large batch is created, a few quality checks
run, and the dataset is handed to a downstream team.

The new **Autodata** paper from Meta FAIR proposes a more ambitious abstraction: make the agent act
like a data scientist. It should create data, inspect examples, measure how solvers behave, analyze
failures, revise its recipe, and repeat. An outer loop can even optimize the data-scientist agent
itself.

I built **FraudForge** to explore what that pattern looks like in a banking architecture.

## The use case: BIAN Fraud Evaluation

The BIAN Service Landscape provides a disciplined boundary. Fraud Evaluation executes behavioral
pattern tests against production activity and identifies out-of-pattern transactions. Downstream
Fraud Diagnosis and Fraud Resolution handle deeper analysis and case outcomes.

The domain is an excellent match for Autodata because BIAN distinguishes rule-set/decision-tree
tests from model-based tests. FraudForge maps those to a weak and strong solver:

- a transparent, coverage-limited rules engine;
- a stronger contextual ensemble;
- a deterministic verifier that checks planted evidence independently of both; and
- an orchestrator that changes the synthetic-data recipe based on the results.

## Why “harder” is the wrong objective

Autodata’s computer-science experiments needed harder questions: the weak solver was succeeding too
often. Its legal experiment encountered the opposite problem. Initial questions were so hard that
weak rollouts clustered near zero, which reduced the variance needed for effective GRPO learning.
The agent improved the data by making it more learnable, not more difficult.

That finding shapes FraudForge’s objective. A batch is accepted only inside a bounded region:

- weak recall must be neither too low nor too high;
- strong recall must remain high;
- the strong-minus-weak gap has lower and upper bounds;
- strong false positives must remain controlled;
- weak scores need non-trivial variance;
- every planted scenario must pass deterministic verification;
- all fraud families must be represented; and
- duplicates must remain rare.

This creates a curriculum rather than an adversarial benchmark.

## The agentic loop

FraudForge generates seven defensive scenario families: card-not-present velocity, impossible
travel, account takeover, mule fan-out, synthetic identity, merchant-context mismatch, and repeated
threshold-pattern activity. Legitimate hard negatives contain suspicious clues without a complete
fraud signature.

For each round, the system:

1. generates a seeded batch;
2. recomputes observable evidence without consulting the label;
3. scores the batch with weak rules and a strong ensemble;
4. measures row-level and corpus-level quality;
5. accepts the batch or diagnoses why it failed;
6. revises ambiguity, signal strength, noise, hard-negative rate, and family weights; and
7. compares recipe candidates on two held-out seeds.

The outer search is deliberately lightweight, but it demonstrates the key architectural point: the
generation strategy is itself an optimizable artifact.

## Verification before model judging

Two supplied Hugging Face datasets reinforced this design. **Verified Analytics Tasks** treats the
checker as the product: fixtures are seeded, gold answers must pass their own deterministic checks,
and the release separates experiences, rollouts, preference pairs, verifiers, and rubrics.
**Vietnamese GSM8K Agentic** coordinates a Challenger, Strong Solver, Weak Solver, and Verifier,
retaining only code-verified, deduplicated examples with the desired capability gap.

FraudForge adopts both patterns. Its strong model is not the oracle. Labels come from planted
scenario semantics that are independently recomputed. Every run exports enough provenance for an
engineer or model-risk reviewer to trace the row back to its recipe, verifier result, solver scores,
and preference representation.

## An honest lesson from the supplied policy model

The `autodata-policy-cs` model card is unusually candid. The training infrastructure completed, but
mean reward did not improve. Every completion hit the token cap, the reward stayed nearly constant,
and only five useful prompts were available.

That is a valuable result. Agentic-data systems can easily produce impressive activity logs without
learning anything. FraudForge therefore keeps the default revision policy deterministic, exposes
reward components, validates across held-out seeds, and defers learned policy training until a
substantial banking trajectory corpus exists.

## What is deployed

The repository includes:

- a Gradio application suitable for a Hugging Face CPU Space;
- a BIAN-aligned FastAPI endpoint at `/FraudEvaluation/Evaluate`;
- Docker packaging;
- unit and integration tests;
- GitHub Actions for CI, Pages, and Space publishing;
- MkDocs architecture and governance documentation; and
- a social launch pack and reproducible sample artifact bundle.

The public version uses fictional data and no external model API. That makes it cheap to inspect,
fork, and challenge.

## Where this could go next

The real opportunity is not to replace fraud experts. It is to build a co-improvement loop:

- ground the generator in approved typologies and de-identified feature definitions;
- let the agent identify coverage gaps and propose scenarios;
- require domain review for label semantics and hard negatives;
- validate candidates on frozen real-data benchmarks inside the bank;
- train a compact recipe policy only after enough accepted/rejected trajectories exist; and
- integrate the resulting curriculum registry with model-risk and BIAN-aligned delivery processes.

A mature system would treat data recipes with the same seriousness as model code: versioned,
validated, approved, monitored, and reversible.

## Explore the project

- Live demo: {{SPACE_URL}}
- Architecture and paper review: {{DOCS_URL}}
- Source code: {{GITHUB_URL}}

The most useful feedback will be on the verifier: which bank-specific constraints and decision-cost
bands would make this synthetic curriculum genuinely valuable to a fraud team?
