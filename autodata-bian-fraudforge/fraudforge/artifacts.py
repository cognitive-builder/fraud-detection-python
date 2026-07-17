from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bian import evaluate_bian_request, sample_bian_request
from .config import FRAUD_SCENARIOS
from .generator import evidence_flags
from .orchestrator import RunResult
from .utility import downstream_utility


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            )


def _decision_text(row: pd.Series, solver: str) -> str:
    prediction = int(row[f"{solver}_prediction"])
    score = float(row[f"{solver}_score"])
    label = "suspected fraud" if prediction else "no fraud alert"
    return f"{label}; score={score:.4f}; reason={row['reason_code']}"


def export_run(result: RunResult, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = result.transactions.copy()
    frame.to_csv(output / "transactions.csv", index=False)
    try:
        frame.to_parquet(output / "transactions.parquet", index=False)
    except (ImportError, ValueError):
        pass

    result.trajectory.to_csv(output / "recipe_trajectory.csv", index=False)
    result.family_metrics.to_csv(output / "family_metrics.csv", index=False)
    _write_json(output / "recipe.json", result.final_recipe.to_dict())
    _write_json(output / "target_band.json", result.target.to_dict())
    _write_json(output / "metrics.json", result.metrics.to_dict())
    _write_json(
        output / "rounds.json",
        [
            {
                "round": item.round,
                "recipe": item.recipe,
                "metrics": item.metrics,
                "decision": item.decision,
                "feedback": item.feedback,
                "meta_validation_score": item.meta_validation_score,
                "selected_by_meta_search": item.selected_by_meta_search,
            }
            for item in result.rounds
        ],
    )

    flags = evidence_flags(frame)
    experiences: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    verifiers: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        transaction_id = str(row["transaction_id"])
        feature_payload = {
            key: row[key]
            for key in (
                "amount",
                "channel",
                "merchant_category",
                "home_country",
                "transaction_country",
                "distance_from_home_km",
                "minutes_since_previous_txn",
                "prior_txn_1h",
                "prior_txn_24h",
                "is_new_device",
                "is_new_beneficiary",
                "merchant_risk_score",
                "identity_consistency_score",
            )
        }
        gold = {
            "is_fraud": bool(row["oracle_label"]),
            "scenario_type": row["scenario_type"],
            "reason_code": row["reason_code"],
        }
        prompt = {
            "instruction": "Evaluate the transaction for out-of-pattern behavior.",
            "transaction": feature_payload,
        }
        experiences.append(
            {
                "id": transaction_id,
                "prompt": prompt,
                "fixture_reference": "transactions.csv",
                "template": row["scenario_type"],
                "gold_answer": gold,
                "recipe_round": result.selected_round,
            }
        )
        sft.append(
            {
                "id": transaction_id,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(prompt, default=_json_default),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(gold, default=_json_default),
                    },
                ],
            }
        )
        for solver in ("weak", "strong"):
            prediction = int(row[f"{solver}_prediction"])
            rollouts.append(
                {
                    "id": f"{transaction_id}:{solver}",
                    "experience_id": transaction_id,
                    "solver": solver,
                    "answer": {
                        "is_fraud": bool(prediction),
                        "score": float(row[f"{solver}_score"]),
                    },
                    "reward": float(prediction == int(row["oracle_label"])),
                    "verifier_pass": bool(row["verifier_pass"]),
                }
            )
        if bool(row["preference_pair"]):
            preferences.append(
                {
                    "id": f"PREF-{transaction_id}",
                    "experience_id": transaction_id,
                    "chosen": _decision_text(row, "strong"),
                    "chosen_reward": 1.0,
                    "rejected": _decision_text(row, "weak"),
                    "rejected_reward": 0.0,
                    "rejection_reason": (
                        "strong solver found planted evidence while weak rules missed it"
                    ),
                    "signal": "strong",
                }
            )
        verifiers.append(
            {
                "id": f"VER-{transaction_id}",
                "experience_id": transaction_id,
                "gate_passed": bool(row["verifier_pass"]),
                "score": float(bool(row["verifier_pass"])),
                "reason": row["verifier_reason"],
                "evidence": {
                    name: int(flags.loc[idx, name]) for name in FRAUD_SCENARIOS
                },
                "duplicate_flag": bool(row["duplicate_flag"]),
            }
        )

    _write_jsonl(output / "experience.jsonl", experiences)
    _write_jsonl(output / "sft_examples.jsonl", sft)
    _write_jsonl(output / "rollouts.jsonl", rollouts)
    _write_jsonl(output / "preference_pairs.jsonl", preferences)
    _write_jsonl(output / "verifiers.jsonl", verifiers)
    _write_jsonl(
        output / "fixture_generators.jsonl",
        [
            {
                "id": "fraudforge-generator",
                "recipe": result.final_recipe.to_dict(),
                "implementation": (
                    "fraudforge.generator.SyntheticTransactionGenerator"
                ),
            }
        ],
    )
    _write_jsonl(
        output / "rubric_items.jsonl",
        [
            {
                "criterion": key,
                "target": value,
                "scoring": "deterministic bounded gate",
            }
            for key, value in result.target.to_dict().items()
        ],
    )

    utility = downstream_utility(frame)
    _write_json(output / "downstream_utility.json", utility)
    bian_request = sample_bian_request(frame)
    bian_response = evaluate_bian_request(bian_request)
    _write_json(output / "bian_request.json", bian_request)
    _write_json(output / "bian_response.json", bian_response)
    _write_json(
        output / "manifest.json",
        {
            "name": "FraudForge Autodata run",
            "accepted": result.accepted,
            "selected_round": result.selected_round,
            "rows": len(frame),
            "preference_pairs": len(preferences),
            "files": sorted(path.name for path in output.iterdir() if path.is_file()),
        },
    )

    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    return {
        "output_dir": str(output),
        "archive": archive,
        "metrics": str(output / "metrics.json"),
        "transactions": str(output / "transactions.csv"),
        "bian_request": str(output / "bian_request.json"),
        "bian_response": str(output / "bian_response.json"),
    }
