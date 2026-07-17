from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FRAUD_SCENARIOS, TargetBand
from .solvers import SolverOutput
from .verifier import VerificationResult


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass(slots=True)
class BatchMetrics:
    row_count: int
    fraud_count: int
    fraud_prevalence: float
    weak_precision: float
    weak_recall: float
    weak_f1: float
    weak_false_positive_rate: float
    strong_precision: float
    strong_recall: float
    strong_f1: float
    strong_false_positive_rate: float
    recall_gap: float
    weak_score_std: float
    strong_score_std: float
    validity_rate: float
    duplicate_rate: float
    family_coverage: float
    preference_pair_rate: float
    accepted: bool
    quality_score: float
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classification(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def _range_score(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return max(0.0, 1.0 - (lower - value) / max(lower, 0.01))
    return max(0.0, 1.0 - (value - upper) / max(1.0 - upper, 0.01))


def evaluate_batch(
    verified: VerificationResult,
    weak: SolverOutput,
    strong: SolverOutput,
    target: TargetBand,
) -> tuple[pd.DataFrame, BatchMetrics, pd.DataFrame]:
    frame = verified.rows.copy()
    y = frame["oracle_label"].to_numpy(dtype=int)
    frame["weak_score"] = weak.score
    frame["weak_prediction"] = weak.prediction
    frame["strong_score"] = strong.score
    frame["strong_prediction"] = strong.prediction
    frame["preference_pair"] = (
        (frame["oracle_label"] == 1)
        & (frame["strong_prediction"] == 1)
        & (frame["weak_prediction"] == 0)
        & frame["verifier_pass"]
    )

    weak_stats = _classification(y, weak.prediction)
    strong_stats = _classification(y, strong.prediction)
    gap = strong_stats["recall"] - weak_stats["recall"]
    preference_rate = float(frame["preference_pair"].mean())

    reasons: list[str] = []
    if not target.weak_recall_min <= weak_stats["recall"] <= target.weak_recall_max:
        reasons.append("WEAK_RECALL_OUTSIDE_TARGET_BAND")
    if strong_stats["recall"] < target.strong_recall_min:
        reasons.append("STRONG_RECALL_TOO_LOW")
    if not target.gap_min <= gap <= target.gap_max:
        reasons.append("WEAK_STRONG_GAP_OUTSIDE_TARGET_BAND")
    if strong_stats["fpr"] > target.strong_false_positive_rate_max:
        reasons.append("STRONG_FALSE_POSITIVE_RATE_TOO_HIGH")
    if verified.validity_rate < target.validity_rate_min:
        reasons.append("VERIFIER_VALIDITY_TOO_LOW")
    if verified.family_coverage < target.family_coverage_min:
        reasons.append("FRAUD_FAMILY_COVERAGE_TOO_LOW")
    if verified.duplicate_rate > target.duplicate_rate_max:
        reasons.append("DUPLICATE_RATE_TOO_HIGH")
    weak_std = float(np.std(weak.score))
    if weak_std < target.weak_score_std_min:
        reasons.append("WEAK_SCORE_VARIANCE_TOO_LOW")
    if preference_rate < target.preference_pair_rate_min:
        reasons.append("TOO_FEW_STRONG_WIN_PREFERENCE_PAIRS")

    component_scores = [
        _range_score(weak_stats["recall"], target.weak_recall_min, target.weak_recall_max),
        min(1.0, strong_stats["recall"] / target.strong_recall_min),
        _range_score(gap, target.gap_min, target.gap_max),
        min(1.0, target.strong_false_positive_rate_max / max(strong_stats["fpr"], 1e-6)),
        min(1.0, verified.validity_rate / target.validity_rate_min),
        min(1.0, verified.family_coverage / target.family_coverage_min),
        min(1.0, target.duplicate_rate_max / max(verified.duplicate_rate, 1e-6)),
        min(1.0, weak_std / target.weak_score_std_min),
        min(1.0, preference_rate / target.preference_pair_rate_min),
    ]

    metrics = BatchMetrics(
        row_count=len(frame),
        fraud_count=int(y.sum()),
        fraud_prevalence=float(y.mean()),
        weak_precision=weak_stats["precision"],
        weak_recall=weak_stats["recall"],
        weak_f1=weak_stats["f1"],
        weak_false_positive_rate=weak_stats["fpr"],
        strong_precision=strong_stats["precision"],
        strong_recall=strong_stats["recall"],
        strong_f1=strong_stats["f1"],
        strong_false_positive_rate=strong_stats["fpr"],
        recall_gap=gap,
        weak_score_std=weak_std,
        strong_score_std=float(np.std(strong.score)),
        validity_rate=verified.validity_rate,
        duplicate_rate=verified.duplicate_rate,
        family_coverage=verified.family_coverage,
        preference_pair_rate=preference_rate,
        accepted=not reasons,
        quality_score=float(np.mean(component_scores)),
        rejection_reasons=reasons,
    )

    family_rows: list[dict[str, Any]] = []
    for scenario in ("legitimate", *FRAUD_SCENARIOS):
        subset = frame[frame["scenario_type"] == scenario]
        if subset.empty:
            continue
        family_rows.append(
            {
                "scenario_type": scenario,
                "rows": len(subset),
                "weak_positive_rate": float(subset["weak_prediction"].mean()),
                "strong_positive_rate": float(subset["strong_prediction"].mean()),
                "verifier_pass_rate": float(subset["verifier_pass"].mean()),
                "preference_pair_rate": (
                    0.0 if scenario == "legitimate" else float(subset["preference_pair"].mean())
                ),
            }
        )
    return frame, metrics, pd.DataFrame(family_rows)
