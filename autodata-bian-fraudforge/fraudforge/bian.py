from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from .generator import evidence_flags
from .solvers import MODEL_FEATURES, StrongEnsembleSolver, WeakRuleSolver

DEFAULTS: dict[str, Any] = {
    "amount": 50.0,
    "available_balance": 2500.0,
    "amount_to_balance_ratio": 0.02,
    "cross_border": 0,
    "distance_from_home_km": 5.0,
    "minutes_since_previous_txn": 240.0,
    "account_age_days": 900,
    "device_age_days": 365,
    "prior_txn_1h": 0,
    "prior_txn_24h": 3,
    "prior_declines_24h": 0,
    "beneficiary_age_hours": 800.0,
    "inbound_counterparties_24h": 1,
    "outbound_counterparties_24h": 1,
    "distinct_beneficiaries_24h": 1,
    "is_new_device": 0,
    "is_new_beneficiary": 0,
    "card_present": 1,
    "cvv_match": 1,
    "billing_match": 1,
    "login_failures_24h": 0,
    "merchant_risk_score": 0.08,
    "identity_consistency_score": 0.94,
    "shared_device_accounts": 1,
    "policy_threshold_ratio": 0.2,
    "round_amount_indicator": 0,
    "hour_of_day": 12,
    "is_weekend": 0,
    "channel": "card_present",
    "merchant_category": "grocery",
    "home_country": "US",
    "transaction_country": "US",
}


def _records_from_request(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("FraudEvaluationTransactionConsolidationRecord", [])
    if isinstance(records, dict):
        records = records.get("transactions", records.get("records", [records]))
    if not isinstance(records, list):
        raise ValueError(
            "FraudEvaluationTransactionConsolidationRecord must be a list or object"
        )
    return [dict(record) for record in records]


def normalize_transactions(records: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        row = DEFAULTS.copy()
        row.update(source)
        row.setdefault("transaction_id", f"EXT-{index:06d}")
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if "amount_to_balance_ratio" not in source:
            row["amount_to_balance_ratio"] = float(row["amount"]) / max(
                float(row["available_balance"]), 1.0
            )
        if "cross_border" not in source:
            row["cross_border"] = int(
                row["home_country"] != row["transaction_country"]
            )
        normalized.append(row)
    frame = pd.DataFrame(normalized)
    for feature in MODEL_FEATURES:
        if feature not in frame:
            frame[feature] = DEFAULTS[feature]
    return frame


def evaluate_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        empty = frame.copy()
        for column in ("weak_score", "strong_score", "ensemble_score", "fraud_suspected"):
            empty[column] = []
        return empty, {
            "transaction_count": 0,
            "anomaly_count": 0,
            "ensemble": "max(strong model, bounded rules)",
        }

    weak = WeakRuleSolver().predict(frame)
    strong = StrongEnsembleSolver().predict(frame)
    flags = evidence_flags(frame)
    evaluated = frame.copy()
    evaluated["weak_score"] = weak.score
    evaluated["weak_prediction"] = weak.prediction
    evaluated["strong_score"] = strong.score
    evaluated["strong_prediction"] = strong.prediction
    evaluated["ensemble_score"] = evaluated[["weak_score", "strong_score"]].max(axis=1)
    evaluated["fraud_suspected"] = evaluated["ensemble_score"].ge(0.50)
    evaluated["evidence_families"] = [
        [name for name, value in row.items() if int(value) == 1]
        for row in flags.to_dict(orient="records")
    ]
    summary = {
        "transaction_count": len(evaluated),
        "anomaly_count": int(evaluated["fraud_suspected"].sum()),
        "weak_rule_positive_count": int(evaluated["weak_prediction"].sum()),
        "strong_model_positive_count": int(evaluated["strong_prediction"].sum()),
        "ensemble": "max(strong model, bounded rules)",
    }
    return evaluated, summary


def evaluate_bian_request(payload: dict[str, Any]) -> dict[str, Any]:
    records = _records_from_request(payload)
    frame = normalize_transactions(records)
    evaluated, summary = evaluate_transactions(frame)
    anomalies: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for _, row in evaluated[evaluated["fraud_suspected"]].iterrows():
        transaction_id = str(row["transaction_id"])
        anomalies.append(
            {
                "transactionReference": transaction_id,
                "ensembleScore": round(float(row["ensemble_score"]), 6),
                "ruleScore": round(float(row["weak_score"]), 6),
                "modelScore": round(float(row["strong_score"]), 6),
                "evidenceFamilies": list(row["evidence_families"]),
                "recommendedDisposition": "REFER_FOR_INVESTIGATION",
            }
        )
        references.append(
            {"TransactionIdentification": {"IdentifierValue": transaction_id}}
        )

    return {
        "FraudEvaluationAssessmentReference": f"FE-{uuid4().hex[:12].upper()}",
        "ProductProductionSessionReference": payload.get(
            "ProductProductionSessionReference",
            {"SessionIdentification": {"IdentifierValue": "fraudforge-demo"}},
        ),
        "FraudEvaluationTestProfile": payload.get(
            "FraudEvaluationTestProfile",
            {"Profile": "FraudForge synthetic behavioral-pattern profile"},
        ),
        "FraudEvaluationEnsembleTechniqueType": payload.get(
            "FraudEvaluationEnsembleTechniqueType", "RULE_MODEL_MAX"
        ),
        "FraudEvaluationEnsembleTechniqueDefinition": payload.get(
            "FraudEvaluationEnsembleTechniqueDefinition",
            "Maximum of bounded rule score and calibrated model score; threshold 0.50",
        ),
        "FraudEvaluationProductionAnomalyRecord": anomalies,
        "FraudEvaluationProductionAnomalyProductionTransactionReference": references,
        "RuleSetsandDecisionTrees": {
            "name": "FraudForge WeakRuleSolver",
            "positiveCount": summary.get("weak_rule_positive_count", 0),
        },
        "Models": {
            "name": "FraudForge StrongEnsembleSolver",
            "positiveCount": summary.get("strong_model_positive_count", 0),
        },
        "assessmentSummary": summary,
        "assessedAt": datetime.now(timezone.utc).isoformat(),
    }


def sample_bian_request(frame: pd.DataFrame, limit: int = 12) -> dict[str, Any]:
    columns = ["transaction_id", *MODEL_FEATURES]
    records = frame[columns].head(limit).to_dict(orient="records")
    return {
        "ProductProductionSessionReference": {
            "SessionIdentification": {"IdentifierValue": "fraudforge-synthetic-batch"}
        },
        "FraudEvaluationTestProfile": {
            "Profile": "Agentic synthetic edge-case evaluation"
        },
        "FraudEvaluationEnsembleTechniqueType": "RULE_MODEL_MAX",
        "FraudEvaluationEnsembleTechniqueDefinition": (
            "Transparent bounded rule score combined with a stronger contextual ensemble"
        ),
        "FraudEvaluationTransactionConsolidationRecord": records,
    }
