# Roadmap

## Phase 1 — Public architecture proof

- deterministic synthetic transaction generator;
- weak rules and strong ensemble;
- BIAN-aligned API;
- Gradio demo and complete artifact export;
- held-out recipe search;
- GitHub Pages and CI/CD packaging.

## Phase 2 — Bank-specific grounding

- connect approved rule catalogs, typology documents, and de-identified feature definitions;
- add institution-specific cost matrices and decision bands;
- incorporate temporal sequences and graph-derived features;
- register scenario owners and model-risk approvals;
- add privacy, bias, and realism audits.

## Phase 3 — Learned data-scientist policy

- collect hundreds or thousands of accepted/rejected recipe trajectories;
- create SFT and preference datasets from deterministic feedback;
- train a compact policy adapter;
- compare learned policy against deterministic revision baseline;
- accept policy changes only on frozen multi-seed and family-holdout suites.

## Phase 4 — Co-improvement platform

- human review interface for scenario and recipe proposals;
- production outcome feedback with delayed-label controls;
- active learning around false positives and misses;
- continuous model/data lineage and release governance;
- integration with BIAN Fraud Diagnosis and Fraud Resolution workflows.
