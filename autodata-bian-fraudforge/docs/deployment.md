# Deployment

## GitHub repository layout

The project is designed to live under:

```text
autodata-bian-fraudforge/
```

inside `cognitive-builder/fraud-detection-python`. Workflows at repository root are path-scoped so
they do not interfere with the existing Hazelcast demo.

## GitHub Pages

The `fraudforge-pages.yml` workflow builds MkDocs Material and deploys the `site/` artifact using
the official GitHub Pages Actions flow.

After merge:

1. Open **Settings → Pages** for the repository.
2. Set **Source** to **GitHub Actions** if it is not already selected.
3. Run **Deploy FraudForge documentation** or push a documentation change to `main`.
4. The expected site is `https://cognitive-builder.github.io/fraud-detection-python/`.

The workflow requires `pages: write` and `id-token: write`; both are declared explicitly.

## Hugging Face Space

The root `README.md` contains Space configuration frontmatter with `sdk: gradio`, Python 3.11,
Gradio 6.5.1, and `app_file: app.py`.

### Manual deployment

```bash
pip install -e .[hub]
export HF_TOKEN=hf_...
export HF_SPACE_ID=<your-hf-user>/fraudforge-autodata-lab
python scripts/publish_hf_space.py
```

### GitHub Actions deployment

Configure in the GitHub repository:

- secret: `HF_TOKEN` with write access to the target Hugging Face namespace;
- variable: `HF_SPACE_ID`, for example `your-user/fraudforge-autodata-lab`.

Then run **Publish FraudForge to Hugging Face** from the Actions tab. The workflow also runs when a
GitHub release is published.

## Docker

```bash
docker build -t fraudforge-autodata .
docker run --rm -p 7860:7860 fraudforge-autodata
```

## Production hardening

The public Space is intentionally stateless and fictional. A bank-hosted version should separate
the generation lab from online transaction scoring, add identity and access management, use a
managed feature store and model registry, sign datasets, stream audit events, and keep real data
inside approved network and retention boundaries.
