from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .config import FRAUD_SCENARIOS, Recipe, RunConfig, TargetBand
from .evaluator import BatchMetrics, evaluate_batch
from .generator import SyntheticTransactionGenerator
from .solvers import StrongEnsembleSolver, WeakRuleSolver
from .verifier import DeterministicVerifier


@dataclass(slots=True)
class RoundRecord:
    round: int
    recipe: dict[str, Any]
    metrics: dict[str, Any]
    decision: str
    feedback: str
    meta_validation_score: float | None = None
    selected_by_meta_search: bool = False


@dataclass(slots=True)
class RunResult:
    accepted: bool
    selected_round: int
    final_recipe: Recipe
    target: TargetBand
    transactions: pd.DataFrame
    family_metrics: pd.DataFrame
    metrics: BatchMetrics
    rounds: list[RoundRecord]

    @property
    def trajectory(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in self.rounds:
            recipe = item.recipe
            metrics = item.metrics
            rows.append(
                {
                    "round": item.round,
                    "decision": item.decision,
                    "quality_score": metrics["quality_score"],
                    "weak_recall": metrics["weak_recall"],
                    "strong_recall": metrics["strong_recall"],
                    "recall_gap": metrics["recall_gap"],
                    "strong_false_positive_rate": metrics["strong_false_positive_rate"],
                    "validity_rate": metrics["validity_rate"],
                    "duplicate_rate": metrics["duplicate_rate"],
                    "preference_pair_rate": metrics["preference_pair_rate"],
                    "weak_score_std": metrics["weak_score_std"],
                    "fraud_prevalence": recipe["fraud_prevalence"],
                    "ambiguous_rate": recipe["ambiguous_rate"],
                    "signal_strength": recipe["signal_strength"],
                    "noise": recipe["noise"],
                    "legitimate_hard_negative_rate": recipe[
                        "legitimate_hard_negative_rate"
                    ],
                    "meta_validation_score": item.meta_validation_score,
                    "feedback": item.feedback,
                }
            )
        return pd.DataFrame(rows)


class AutodataOrchestrator:
    """Inner data-scientist loop plus a held-out outer recipe search."""

    def __init__(
        self,
        config: RunConfig | None = None,
        target: TargetBand | None = None,
        initial_recipe: Recipe | None = None,
    ) -> None:
        self.config = config or RunConfig()
        self.config.validate()
        self.target = target or TargetBand()
        self.initial_recipe = (initial_recipe or Recipe()).clipped()
        self.weak_solver = WeakRuleSolver()
        self.strong_solver = StrongEnsembleSolver()
        self.verifier = DeterministicVerifier()

    def run(self) -> RunResult:
        recipe = self.initial_recipe
        rounds: list[RoundRecord] = []
        best: tuple[float, int, Recipe, pd.DataFrame, BatchMetrics, pd.DataFrame] | None = None

        for round_number in range(1, self.config.max_rounds + 1):
            transactions, metrics, family_metrics = self._evaluate_recipe(
                recipe,
                n_rows=self.config.batch_size,
                seed=self.config.seed + (round_number - 1) * 101,
            )
            if best is None or metrics.quality_score > best[0]:
                best = (
                    metrics.quality_score,
                    round_number,
                    recipe,
                    transactions,
                    metrics,
                    family_metrics,
                )

            feedback = self._analyze(metrics, family_metrics)
            if metrics.accepted:
                rounds.append(
                    RoundRecord(
                        round=round_number,
                        recipe=recipe.to_dict(),
                        metrics=metrics.to_dict(),
                        decision="ACCEPT",
                        feedback=feedback,
                    )
                )
                return RunResult(
                    accepted=True,
                    selected_round=round_number,
                    final_recipe=recipe,
                    target=self.target,
                    transactions=transactions,
                    family_metrics=family_metrics,
                    metrics=metrics,
                    rounds=rounds,
                )

            revised = self._revise(recipe, metrics, family_metrics)
            meta_score: float | None = None
            selected_by_meta = False
            if (
                self.config.meta_search_candidates > 0
                and round_number % self.config.meta_search_every == 0
            ):
                revised, meta_score = self._meta_search(revised, round_number)
                selected_by_meta = True

            rounds.append(
                RoundRecord(
                    round=round_number,
                    recipe=recipe.to_dict(),
                    metrics=metrics.to_dict(),
                    decision="REVISE",
                    feedback=feedback,
                    meta_validation_score=meta_score,
                    selected_by_meta_search=selected_by_meta,
                )
            )
            recipe = revised

        assert best is not None
        _, selected_round, selected_recipe, frame, metrics, family_metrics = best
        return RunResult(
            accepted=False,
            selected_round=selected_round,
            final_recipe=selected_recipe,
            target=self.target,
            transactions=frame,
            family_metrics=family_metrics,
            metrics=metrics,
            rounds=rounds,
        )

    def _evaluate_recipe(
        self, recipe: Recipe, n_rows: int, seed: int
    ) -> tuple[pd.DataFrame, BatchMetrics, pd.DataFrame]:
        raw = SyntheticTransactionGenerator(recipe, seed=seed).generate(n_rows)
        verified = self.verifier.verify(raw)
        weak = self.weak_solver.predict(verified.rows)
        strong = self.strong_solver.predict(verified.rows)
        return evaluate_batch(verified, weak, strong, self.target)

    def _analyze(self, metrics: BatchMetrics, family: pd.DataFrame) -> str:
        observations: list[str] = []
        if metrics.weak_recall > self.target.weak_recall_max:
            observations.append("weak rules solve too many cases; add boundary ambiguity")
        elif metrics.weak_recall < self.target.weak_recall_min:
            observations.append("weak rules have too little success; strengthen observable signals")
        if metrics.strong_recall < self.target.strong_recall_min:
            observations.append("strong model misses planted cases; reduce noise or ambiguity")
        if metrics.recall_gap < self.target.gap_min:
            observations.append("weak/strong separation is too small")
        elif metrics.recall_gap > self.target.gap_max:
            observations.append("examples are too degenerate for the weak solver")
        if metrics.preference_pair_rate < self.target.preference_pair_rate_min:
            observations.append("too few strong-win preference pairs are available")
        if metrics.validity_rate < self.target.validity_rate_min:
            observations.append("planted evidence fails deterministic verification")
        if metrics.duplicate_rate > self.target.duplicate_rate_max:
            observations.append("batch contains too many duplicate fingerprints")
        if not family.empty:
            fraud_family = family[family["scenario_type"] != "legitimate"]
            if not fraud_family.empty:
                richest = fraud_family.sort_values(
                    "preference_pair_rate", ascending=False
                ).iloc[0]
                observations.append(
                    f"{richest['scenario_type']} contributes the richest strong-win signal"
                )
        return "; ".join(observations) if observations else "all bounded quality gates passed"

    def _revise(
        self, recipe: Recipe, metrics: BatchMetrics, family: pd.DataFrame
    ) -> Recipe:
        updated = recipe
        if (
            metrics.weak_recall > self.target.weak_recall_max
            or metrics.recall_gap < self.target.gap_min
        ):
            updated = replace(
                updated,
                ambiguous_rate=updated.ambiguous_rate + 0.10,
                signal_strength=updated.signal_strength - 0.08,
                noise=updated.noise + 0.02,
                legitimate_hard_negative_rate=updated.legitimate_hard_negative_rate + 0.035,
            )
        elif (
            metrics.weak_recall < self.target.weak_recall_min
            or metrics.recall_gap > self.target.gap_max
        ):
            updated = replace(
                updated,
                ambiguous_rate=updated.ambiguous_rate - 0.08,
                signal_strength=updated.signal_strength + 0.09,
                noise=updated.noise - 0.02,
            )

        if (
            metrics.strong_recall < self.target.strong_recall_min
            or metrics.validity_rate < self.target.validity_rate_min
        ):
            updated = replace(
                updated,
                signal_strength=updated.signal_strength + 0.07,
                ambiguous_rate=updated.ambiguous_rate - 0.035,
                noise=updated.noise - 0.025,
            )

        if metrics.preference_pair_rate < self.target.preference_pair_rate_min and not family.empty:
            weights = updated.normalized_weights()
            fraud_family = family[family["scenario_type"] != "legitimate"]
            for _, row in fraud_family.iterrows():
                name = str(row["scenario_type"])
                strong_win = float(row["preference_pair_rate"])
                weights[name] *= 1.0 + min(0.45, 1.6 * strong_win)
            updated = replace(updated, scenario_weights=weights)

        return updated.clipped()

    def _meta_search(self, center: Recipe, round_number: int) -> tuple[Recipe, float]:
        rng = np.random.default_rng(self.config.seed + 10_000 + round_number)
        candidates = [center]
        for _ in range(max(0, self.config.meta_search_candidates - 1)):
            weights = center.normalized_weights()
            focus = str(rng.choice(FRAUD_SCENARIOS))
            weights[focus] *= float(rng.uniform(1.08, 1.35))
            candidate = replace(
                center,
                ambiguous_rate=center.ambiguous_rate + float(rng.normal(0, 0.035)),
                signal_strength=center.signal_strength + float(rng.normal(0, 0.04)),
                noise=center.noise + float(rng.normal(0, 0.012)),
                legitimate_hard_negative_rate=center.legitimate_hard_negative_rate
                + float(rng.normal(0, 0.025)),
                scenario_weights=weights,
            ).clipped()
            candidates.append(candidate)

        scored: list[tuple[float, Recipe]] = []
        for index, candidate in enumerate(candidates):
            validation_scores: list[float] = []
            for seed_offset in (0, 37):
                _, metrics, _ = self._evaluate_recipe(
                    candidate,
                    n_rows=self.config.validation_batch_size,
                    seed=self.config.seed
                    + 50_000
                    + round_number * 1000
                    + index * 100
                    + seed_offset,
                )
                penalty = 0.012 * len(metrics.rejection_reasons)
                validation_scores.append(metrics.quality_score - penalty)
            scored.append((float(np.mean(validation_scores)), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1], scored[0][0]
