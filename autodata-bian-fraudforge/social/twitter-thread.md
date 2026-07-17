# X / Twitter thread

**1/** I built **FraudForge**: an agentic synthetic-data lab for the BIAN Fraud Evaluation service
domain, inspired by Meta FAIR’s new Autodata paper. 🏦🤖

Code: {{GITHUB_URL}}  
Demo: {{SPACE_URL}}

**2/** The central idea: synthetic data should not be a one-shot prompt.

Generate → evaluate → analyze failures → revise the recipe → repeat.

Then optimize the data-scientist policy itself on held-out examples.

**3/** FraudForge maps the paper’s four roles into a banking system:

- Challenger: synthetic transaction generator
- Weak solver: transparent fraud rules
- Strong solver: contextual ensemble
- Verifier: deterministic planted-evidence checks

**4/** Why BIAN Fraud Evaluation?

BIAN separates rule/decision-tree tests from model-based tests. That maps almost perfectly to the
weak-vs-strong solver design—and keeps the demo bounded before Fraud Diagnosis/Resolution.

**5/** The objective is not “make fraud examples as hard as possible.”

Autodata shows a task can be too hard, producing near-zero weak rollouts and poor learning signal.
FraudForge uses bounded recall, gap, FPR, variance, coverage, and validity gates.

**6/** The agent changes:

- ambiguity
- planted signal strength
- noise
- legitimate hard-negative rate
- fraud-family mixture

A held-out outer loop tests recipe candidates across separate seeds.

**7/** Every run exports more than a CSV:

experiences, SFT examples, rollouts, strong-win preference pairs, deterministic verifier logs,
rubric items, recipe trajectory, downstream utility, and BIAN request/response payloads.

**8/** The public demo is deliberately boring in the best way:

- fictional data only
- no API key
- deterministic primary oracle
- CPU-friendly
- inspectable reward components
- tests + Docker + CI/CD

**9/** The supplied HF policy adapter’s honest model card was an important reminder: a completed
training job is not evidence that a policy improved. Reward stayed flat because the dataset and
termination/reward design were insufficient.

**10/** Full paper/asset review, architecture, governance, and deployment guide:

{{DOCS_URL}}

I’d love critiques from fraud, model risk, BIAN, synthetic-data, and agentic-systems practitioners.
