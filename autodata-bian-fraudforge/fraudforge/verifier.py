from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FRAUD_SCENARIOS
from .generator import evidence_flags


@dataclass(slots=True)
class VerificationResult:
    rows: pd.DataFrame
    validity_rate: float
    duplicate_rate: float
    family_coverage: float


class DeterministicVerifier:
    """Primary oracle for schema, planted signatures, constraints, and duplicates."""

    required_columns = {
        "transaction_id",
        "amount",
        "available_balance",
        "scenario_type",
        "oracle_label",
        "reason_code",
    }

    def verify(self, frame: pd.DataFrame) -> VerificationResult:
        missing = self.required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        flags = evidence_flags(frame)
        signature_ok: list[bool] = []
        reasons: list[str] = []
        evidence_counts: list[int] = []
        for idx, row in frame.iterrows():
            scenario = str(row["scenario_type"])
            label_ok = int(row["oracle_label"]) == int(scenario != "legitimate")
            evidence_count = int(flags.loc[idx].sum())
            evidence_counts.append(evidence_count)
            if scenario == "legitimate":
                scenario_ok = evidence_count <= 1
            elif scenario in FRAUD_SCENARIOS:
                scenario_ok = bool(flags.loc[idx, scenario])
            else:
                scenario_ok = False
            numeric_ok = (
                float(row["amount"]) > 0
                and float(row["available_balance"]) > 0
                and 0 <= float(row["merchant_risk_score"]) <= 1
                and 0 <= float(row["identity_consistency_score"]) <= 1
            )
            valid = bool(label_ok and scenario_ok and numeric_ok)
            signature_ok.append(valid)
            if not label_ok:
                reasons.append("LABEL_SCENARIO_MISMATCH")
            elif not scenario_ok:
                reasons.append("PLANTED_SIGNATURE_MISSING_OR_CONFLICTING")
            elif not numeric_ok:
                reasons.append("NUMERIC_CONSTRAINT_FAILURE")
            else:
                reasons.append("PASS")

        checked = frame.copy()
        checked["verifier_pass"] = signature_ok
        checked["verifier_reason"] = reasons
        checked["evidence_family_count"] = evidence_counts
        checked["row_fingerprint"] = self._fingerprint(checked)
        duplicate_mask = checked.duplicated("row_fingerprint", keep=False)
        checked["duplicate_flag"] = duplicate_mask

        observed = set(checked.loc[checked["oracle_label"].eq(1), "scenario_type"])
        coverage = len(observed.intersection(FRAUD_SCENARIOS)) / len(FRAUD_SCENARIOS)
        return VerificationResult(
            rows=checked,
            validity_rate=float(checked["verifier_pass"].mean()),
            duplicate_rate=float(duplicate_mask.mean()),
            family_coverage=float(coverage),
        )

    @staticmethod
    def _fingerprint(frame: pd.DataFrame) -> pd.Series:
        columns = [
            "scenario_type",
            "amount",
            "channel",
            "transaction_country",
            "distance_from_home_km",
            "prior_txn_1h",
            "prior_txn_24h",
            "is_new_device",
            "is_new_beneficiary",
            "merchant_risk_score",
            "identity_consistency_score",
            "shared_device_accounts",
        ]
        normalized = frame[columns].copy()
        for column in normalized.select_dtypes(include="number").columns:
            normalized[column] = normalized[column].round(2)
        return pd.util.hash_pandas_object(normalized, index=False).astype(str)
