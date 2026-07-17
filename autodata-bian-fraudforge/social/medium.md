# I Built an Agentic Synthetic-Data Scientist for Banking Fraud Evaluation

**Subtitle:** What Meta FAIR’s Autodata idea looks like when mapped to the BIAN Service Landscape.

> Cross-posted from the original article at {{DOCS_URL}}. Set that page as the canonical source
> when publishing on Medium.

Synthetic fraud data is often generated in a single pass: configure a simulator, create millions of
rows, and measure statistical similarity. That approach can miss the question that matters most:
**does the data provide useful learning signal for the system we are trying to improve?**

Meta FAIR’s Autodata paper proposes an agentic alternative. An AI system acts as a data scientist:
it creates examples, evaluates solver behavior, analyzes failures, changes the recipe, and repeats.
The data-scientist agent can itself be optimized using held-out results.

I implemented that pattern in **FraudForge**, a public laboratory for the BIAN **Fraud Evaluation**
service domain.

## A banking-native weak/strong architecture

BIAN represents rule/decision-tree tests and model-based tests as distinct Fraud Evaluation
behavior qualifiers. FraudForge uses that boundary directly:

- **Challenger:** a seeded synthetic transaction generator;
- **Weak solver:** transparent fraud rules;
- **Strong solver:** a contextual Extra Trees ensemble;
- **Verifier:** deterministic evidence and constraint checks; and
- **Data-scientist orchestrator:** a feedback loop that revises the generation recipe.

The generator creates fictional variants of seven defensive patterns and legitimate hard negatives.
The verifier recomputes the evidence without reading the planted label, so the model does not grade
its own output.

## A curriculum, not a difficulty contest

The most important Autodata insight is that useful data is not always harder data. In the paper’s
legal task, questions were initially too difficult: weak rollouts were often all zero, creating
poor reward variance. The agent made the task more learnable.

FraudForge therefore requires weak recall to stay inside a band. It also constrains strong recall,
weak/strong gap, false-positive rate, score variance, deterministic validity, fraud-family coverage,
duplicate rate, and strong-win preference yield.

When a batch fails, the agent changes ambiguity, planted-signal strength, noise, legitimate hard
negatives, and scenario weights. Candidate recipes are compared on held-out seeds before the next
round.

## Verification and export are first-class

The project borrows the artifact discipline of two supplied Hugging Face datasets. Every run can
export:

- transaction fixtures;
- experiences and SFT messages;
- weak and strong rollouts;
- strong-win preference pairs;
- deterministic verifier records;
- target-band rubric items;
- recipe and trajectory history;
- shifted-holdout utility metrics; and
- BIAN request/response examples.

This is more useful than a flat CSV because it preserves why each row exists and how it behaved.

## What the policy-model review changed

The supplied `autodata-policy-cs` adapter reports that training completed but did not measurably
improve reward. The dataset had only five useful prompts, generations never terminated before the
completion cap, and the reward was nearly constant.

FraudForge treats that as a design requirement: never equate infrastructure success with policy
learning. The default recipe agent is deterministic; rewards are visible; candidate policies are
held-out tested; and learned policy training is deferred until sufficient bank-specific trajectory
data exists.

## Try it

The public demo contains no real banking data and needs no external LLM key.

- Demo: {{SPACE_URL}}
- Documentation: {{DOCS_URL}}
- Code: {{GITHUB_URL}}

The next step is to ground the system in institution-approved fraud typologies and validate its
curricula on frozen internal benchmarks under model-risk governance.
