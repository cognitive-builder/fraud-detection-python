from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_SCENARIO_WEIGHTS, Recipe
from .generator import SyntheticTransactionGenerator, evidence_flags

NUMERIC_FEATURES = [
    "amount",
    "available_balance",
    "amount_to_balance_ratio",
    "cross_border",
    "distance_from_home_km",
    "minutes_since_previous_txn",
    "account_age_days",
    "device_age_days",
    "prior_txn_1h",
    "prior_txn_24h",
    "prior_declines_24h",
    "beneficiary_age_hours",
    "inbound_counterparties_24h",
    "outbound_counterparties_24h",
    "distinct_beneficiaries_24h",
    "is_new_device",
    "is_new_beneficiary",
    "card_present",
    "cvv_match",
    "billing_match",
    "login_failures_24h",
    "merchant_risk_score",
    "identity_consistency_score",
    "shared_device_accounts",
    "policy_threshold_ratio",
    "round_amount_indicator",
    "hour_of_day",
    "is_weekend",
]
CATEGORICAL_FEATURES = [
    "channel",
    "merchant_category",
    "home_country",
    "transaction_country",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(slots=True)
class SolverOutput:
    score: np.ndarray
    prediction: np.ndarray


class WeakRuleSolver:
    """A deliberately bounded rules/decision-tree proxy for the weak solver."""

    threshold: float = 0.25

    def predict(self, frame: pd.DataFrame) -> SolverOutput:
        score = np.full(len(frame), 0.035, dtype=float)
        score += 0.13 * frame["amount_to_balance_ratio"].gt(0.72).to_numpy()
        score += 0.20 * (
            frame["channel"].eq("card_not_present") & frame["prior_txn_1h"].ge(7)
        ).to_numpy()
        score += 0.28 * (
            frame["distance_from_home_km"].gt(1750)
            & frame["minutes_since_previous_txn"].lt(32)
            & frame["cross_border"].eq(1)
        ).to_numpy()
        score += 0.31 * (
            frame["is_new_device"].eq(1)
            & frame["login_failures_24h"].ge(4)
            & frame["is_new_beneficiary"].eq(1)
            & frame["beneficiary_age_hours"].lt(14)
        ).to_numpy()
        score += 0.28 * (
            frame["inbound_counterparties_24h"].ge(9)
            & frame["outbound_counterparties_24h"].ge(9)
        ).to_numpy()
        score += 0.23 * (
            frame["account_age_days"].lt(52)
            & frame["identity_consistency_score"].lt(0.34)
            & frame["shared_device_accounts"].ge(7)
        ).to_numpy()
        score += 0.22 * (
            frame["merchant_risk_score"].gt(0.90)
            & frame["billing_match"].eq(0)
        ).to_numpy()
        score += 0.18 * (
            frame["prior_txn_24h"].ge(11)
            & frame["policy_threshold_ratio"].between(0.91, 1.0)
            & frame["distinct_beneficiaries_24h"].ge(7)
        ).to_numpy()
        score += 0.08 * frame["prior_declines_24h"].ge(3).to_numpy()
        score = np.clip(score, 0.0, 1.0)
        return SolverOutput(score=score, prediction=(score >= self.threshold).astype(int))


class StrongEnsembleSolver:
    """A stronger model-based qualifier using broader contextual features."""

    def __init__(self, calibration_seed: int = 17) -> None:
        self.calibration_seed = calibration_seed
        self.pipeline = _build_calibrated_pipeline(calibration_seed)

    def predict(self, frame: pd.DataFrame) -> SolverOutput:
        model_score = self.pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]
        signatures = evidence_flags(frame).sum(axis=1).clip(upper=2).to_numpy() / 2.0
        score = np.clip(0.88 * model_score + 0.12 * signatures, 0.0, 1.0)
        return SolverOutput(score=score, prediction=(score >= 0.50).astype(int))


@lru_cache(maxsize=4)
def _build_calibrated_pipeline(seed: int) -> Pipeline:
    calibration_recipe = Recipe(
        fraud_prevalence=0.38,
        ambiguous_rate=0.38,
        signal_strength=0.93,
        noise=0.18,
        legitimate_hard_negative_rate=0.24,
        scenario_weights=DEFAULT_SCENARIO_WEIGHTS.copy(),
    )
    train = SyntheticTransactionGenerator(calibration_recipe, seed=seed).generate(7000)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    classifier = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=13,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
    )
    pipeline = Pipeline([("features", preprocessor), ("model", classifier)])
    pipeline.fit(train[MODEL_FEATURES], train["oracle_label"])
    return pipeline
