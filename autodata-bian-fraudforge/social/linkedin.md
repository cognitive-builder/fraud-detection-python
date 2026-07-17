# LinkedIn launch post

I built **FraudForge**, an agentic synthetic-data lab for banking fraud evaluation. 🏦🤖

It is a practical implementation of the core idea in Meta FAIR’s new **Autodata** paper: treat data
creation as an iterative data-science process rather than a one-shot generation prompt.

The banking use case is the BIAN **Fraud Evaluation** service domain. The loop:

→ generates fictional transaction scenarios  
→ challenges a bounded rule engine and a stronger contextual model  
→ deterministically verifies planted fraud evidence  
→ measures weak/strong separation, false positives, coverage, duplicates, and score variance  
→ changes the generation recipe when the data is too easy, too hard, noisy, or repetitive  
→ validates recipe candidates on held-out seeds  
→ exports SFT, rollout, preference, verifier, rubric, and BIAN API artifacts

The most interesting design choice is that the objective is **not maximum difficulty**. Autodata’s
legal-reasoning results show that data can be too hard to provide useful RL variance. FraudForge
therefore optimizes for a bounded “just-right” learning-signal region.

The demo is CPU-friendly, requires no external LLM or API key, contains no real customer data, and
includes a Gradio app, FastAPI endpoint, Docker packaging, tests, CI/CD, GitHub Pages documentation,
and a reproducible sample run.

🔬 Paper review and architecture: {{DOCS_URL}}  
🧪 Live demo: {{SPACE_URL}}  
💻 Code: {{GITHUB_URL}}

I would value feedback from banking architects, fraud practitioners, model-risk leaders, and
synthetic-data researchers—especially on how you would extend the verifier and target bands for an
institution-specific curriculum.

#AgenticAI #SyntheticData #Banking #FinancialServices #FraudDetection #BIAN #MachineLearning
#EnterpriseArchitecture #ResponsibleAI
