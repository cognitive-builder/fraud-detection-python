from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

FRAUD_SCENARIOS: tuple[str, ...] = (
    "cnp_velocity",
    "impossible_travel",
    "account_takeover",
    "mule_fanout",
    "synthetic_identity",
    "merchant_mismatch",
    "threshold_pattern",
)

DEFAULT_SCENARIO_WEIGHTS: dict[str, float] = {
    "cnp_velocity": 0.18,
    "impossible_travel": 0.14,
    "account_takeover": 0.19,
    "mule_fanout": 0.14,
    "synthetic_identity": 0.11,
    "merchant_mismatch": 0.12,
    "threshold_pattern": 0.12,
}


@dataclass(slots=True)
class Recipe:
    """Mutable generation policy optimized by the Autodata loop."""

    fraud_prevalence: float = 0.30
    ambiguous_rate: float = 0.35
    signal_strength: float = 1.20
    noise: float = 0.12
    legitimate_hard_negative_rate: float = 0.18
    scenario_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_SCENARIO_WEIGHTS.copy()
    )

    def normalized_weights(self) -> dict[str, float]:
        cleaned = {
            name: max(0.001, float(self.scenario_weights.get(name, 0.0)))
            for name in FRAUD_SCENARIOS
        }
        total = sum(cleaned.values())
        return {name: weight / total for name, weight in cleaned.items()}

    def clipped(self) -> Recipe:
        return replace(
            self,
            fraud_prevalence=min(0.55, max(0.08, self.fraud_prevalence)),
            ambiguous_rate=min(0.75, max(0.0, self.ambiguous_rate)),
            signal_strength=min(1.60, max(0.45, self.signal_strength)),
            noise=min(0.45, max(0.0, self.noise)),
            legitimate_hard_negative_rate=min(
                0.50, max(0.0, self.legitimate_hard_negative_rate)
            ),
            scenario_weights=self.normalized_weights(),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenario_weights"] = self.normalized_weights()
        return data


@dataclass(slots=True)
class TargetBand:
    """Bounded learning-signal policy—not a maximum-difficulty target."""

    weak_recall_min: float = 0.25
    weak_recall_max: float = 0.60
    strong_recall_min: float = 0.82
    gap_min: float = 0.16
    gap_max: float = 0.58
    strong_false_positive_rate_max: float = 0.12
    validity_rate_min: float = 0.985
    family_coverage_min: float = 0.95
    duplicate_rate_max: float = 0.015
    weak_score_std_min: float = 0.08
    preference_pair_rate_min: float = 0.06

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class RunConfig:
    batch_size: int = 900
    max_rounds: int = 7
    seed: int = 42
    meta_search_candidates: int = 3
    meta_search_every: int = 1
    validation_batch_size: int = 280
    export_preview_rows: int = 250

    def validate(self) -> None:
        if not 100 <= self.batch_size <= 10_000:
            raise ValueError("batch_size must be between 100 and 10,000")
        if not 1 <= self.max_rounds <= 20:
            raise ValueError("max_rounds must be between 1 and 20")
        if not 0 <= self.meta_search_candidates <= 12:
            raise ValueError("meta_search_candidates must be between 0 and 12")
        if self.validation_batch_size < 100:
            raise ValueError("validation_batch_size must be at least 100")
        if self.meta_search_every < 1:
            raise ValueError("meta_search_every must be at least 1")
