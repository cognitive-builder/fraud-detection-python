import json
from pathlib import Path

from fraudforge.artifacts import export_run
from fraudforge.config import RunConfig
from fraudforge.orchestrator import AutodataOrchestrator


def test_artifact_bundle_contract(tmp_path: Path) -> None:
    result = AutodataOrchestrator(
        RunConfig(
            batch_size=220,
            max_rounds=3,
            seed=9,
            meta_search_candidates=0,
            validation_batch_size=120,
        )
    ).run()
    paths = export_run(result, tmp_path / "run")

    expected = {
        "transactions.csv",
        "experience.jsonl",
        "sft_examples.jsonl",
        "rollouts.jsonl",
        "preference_pairs.jsonl",
        "verifiers.jsonl",
        "metrics.json",
        "downstream_utility.json",
        "bian_request.json",
        "bian_response.json",
        "manifest.json",
    }
    assert expected.issubset({path.name for path in (tmp_path / "run").iterdir()})
    assert Path(paths["archive"]).exists()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["rows"] == 220
