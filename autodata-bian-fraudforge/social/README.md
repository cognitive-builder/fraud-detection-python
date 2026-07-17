# FraudForge social launch pack

This folder contains platform-specific launch drafts for the GitHub repository, Hugging Face Space,
and GitHub Pages article.

## Replace before publishing

- `{{GITHUB_URL}}` — repository or merged project URL
- `{{PR_URL}}` — pull request during preview period
- `{{SPACE_URL}}` — live Hugging Face Space
- `{{DOCS_URL}}` — live GitHub Pages site
- `{{DEMO_METRICS}}` — verified metrics from the canonical sample run

## Recommended sequence

1. Merge the GitHub pull request and confirm CI.
2. Deploy the Space and run the canonical seed once.
3. Enable/deploy GitHub Pages.
4. Publish the long-form Substack article.
5. Cross-post an adapted version to Medium with canonical attribution.
6. Publish LinkedIn and X/Twitter posts linking to the article and live demo.
7. Respond with architecture, governance, and reproducibility details—not performance hype.
