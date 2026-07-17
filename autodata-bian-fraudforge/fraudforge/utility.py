from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_SCENARIO_WEIGHTS, Recipe
from .generator import SyntheticTransactionGenerator
from .solvers import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def _learner(seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    C=0.35,
                    random_state=seed,
                ),
            ),
        ]
    )


def _score(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    y = frame["oracle_label"].to_numpy()
    probability = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def downstream_utility(accepted: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Fixed before/after learner on a shifted holdout; never an oracle label."""

    baseline_weights = DEFAULT_SCENARIO_WEIGHTS.copy()
    for name in ("mule_fanout", "synthetic_identity", "threshold_pattern"):
        baseline_weights[name] = 0.008
    baseline = SyntheticTransactionGenerator(
        Recipe(
            fraud_prevalence=0.24,
            ambiguous_rate=0.12,
            signal_strength=1.24,
            noise=0.06,
            legitimate_hard_negative_rate=0.08,
            scenario_weights=baseline_weights,
        ),
        seed=seed + 700,
    ).generate(1800)

    shifted = SyntheticTransactionGenerator(
        Recipe(
            fraud_prevalence=0.34,
            ambiguous_rate=0.43,
            signal_strength=0.88,
            noise=0.20,
            legitimate_hard_negative_rate=0.26,
        ),
        seed=seed + 900,
    ).generate(2200)

    enrichment = accepted[accepted["verifier_pass"]].copy()
    if len(enrichment) > 1200:
        enrichment = enrichment.sample(1200, random_state=seed)
    enriched_train = pd.concat([baseline, enrichment], ignore_index=True)

    before_model = _learner(seed)
    after_model = _learner(seed)
    before_model.fit(baseline[MODEL_FEATURES], baseline["oracle_label"])
    after_model.fit(enriched_train[MODEL_FEATURES], enriched_train["oracle_label"])
    before = _score(before_model, shifted)
    after = _score(after_model, shifted)
    return {
        "protocol": "fixed logistic learner; shifted synthetic holdout; identical seed and threshold",
        "baseline_training_rows": len(baseline),
        "agentic_enrichment_rows": len(enrichment),
        "holdout_rows": len(shifted),
        "before": before,
        "after": after,
        "delta": {name: after[name] - before[name] for name in before},
    }
