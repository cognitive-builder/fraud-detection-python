from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_ID")
    if not token or not repo_id:
        raise SystemExit("HF_TOKEN and HF_SPACE_ID are required")

    root = Path(__file__).resolve().parents[1]
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/**",
            ".github/**",
            ".venv/**",
            "site/**",
            "artifacts/**",
            "tests/**",
            "social/**",
            "examples/**",
            "*.zip",
        ],
        commit_message="Publish FraudForge Autodata Lab",
    )
    print(f"Published https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
