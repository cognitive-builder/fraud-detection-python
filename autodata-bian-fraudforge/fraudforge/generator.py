from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import FRAUD_SCENARIOS, Recipe

HOME_COUNTRIES = ("US", "CA", "GB", "DE", "FR", "IN", "SG", "AU")
MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "restaurant",
    "travel",
    "electronics",
    "digital_goods",
    "professional_services",
    "cash_equivalent",
)
CHANNELS = ("card_present", "card_not_present", "atm", "transfer")
REASON_CODES = {
    "legitimate": "NORMAL_BEHAVIOR",
    "cnp_velocity": "CNP_VELOCITY_CLUSTER",
    "impossible_travel": "IMPOSSIBLE_TRAVEL_SEQUENCE",
    "account_takeover": "ACCOUNT_TAKEOVER_SIGNATURE",
    "mule_fanout": "RAPID_FAN_IN_FAN_OUT",
    "synthetic_identity": "IDENTITY_CONSISTENCY_ANOMALY",
    "merchant_mismatch": "MERCHANT_CONTEXT_MISMATCH",
    "threshold_pattern": "REPEATED_POLICY_THRESHOLD_PATTERN",
}


@dataclass(slots=True)
class SyntheticTransactionGenerator:
    recipe: Recipe
    seed: int = 42

    def generate(self, n_rows: int) -> pd.DataFrame:
        if n_rows <= 0:
            raise ValueError("n_rows must be positive")
        recipe = self.recipe.clipped()
        rng = np.random.default_rng(self.seed)
        weights = recipe.normalized_weights()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows: list[dict[str, Any]] = []

        for index in range(n_rows):
            is_fraud = bool(rng.random() < recipe.fraud_prevalence)
            scenario = (
                str(rng.choice(FRAUD_SCENARIOS, p=list(weights.values())))
                if is_fraud
                else "legitimate"
            )
            row = self._base_row(rng, index, start)
            if scenario == "legitimate":
                self._make_legitimate(row, rng, recipe)
            else:
                self._apply_scenario(row, scenario, rng, recipe)
            row["scenario_type"] = scenario
            row["oracle_label"] = int(scenario != "legitimate")
            row["reason_code"] = REASON_CODES[scenario]
            rows.append(row)

        return self._finalize(pd.DataFrame(rows))

    @staticmethod
    def _base_row(
        rng: np.random.Generator,
        index: int,
        start: datetime,
    ) -> dict[str, Any]:
        home_country = str(rng.choice(HOME_COUNTRIES))
        channel = str(rng.choice(CHANNELS, p=(0.42, 0.27, 0.09, 0.22)))
        amount = float(np.clip(rng.lognormal(mean=3.75, sigma=0.92), 1.0, 5000.0))
        balance = float(np.clip(rng.lognormal(mean=7.55, sigma=0.82), 100.0, 100_000.0))
        timestamp = start + timedelta(minutes=int(rng.integers(0, 365 * 24 * 60)))
        return {
            "transaction_id": f"TX-{index:07d}-{rng.integers(1000, 9999)}",
            "customer_id": f"CUS-{rng.integers(1, max(50, index // 3 + 50)):06d}",
            "account_id": f"ACC-{rng.integers(1, max(50, index // 4 + 50)):06d}",
            "merchant_id": f"MER-{rng.integers(1, 800):05d}",
            "timestamp": timestamp.isoformat(),
            "amount": amount,
            "available_balance": balance,
            "amount_to_balance_ratio": min(2.0, amount / max(balance, 1.0)),
            "channel": channel,
            "merchant_category": str(rng.choice(MERCHANT_CATEGORIES)),
            "home_country": home_country,
            "transaction_country": home_country,
            "cross_border": 0,
            "distance_from_home_km": float(np.clip(rng.gamma(1.8, 7.5), 0.0, 600.0)),
            "minutes_since_previous_txn": float(
                np.clip(rng.gamma(2.0, 170.0), 1.0, 1440.0)
            ),
            "account_age_days": int(rng.integers(120, 5000)),
            "device_age_days": int(rng.integers(15, 1600)),
            "prior_txn_1h": int(rng.poisson(0.65)),
            "prior_txn_24h": int(rng.poisson(3.2)),
            "prior_declines_24h": int(rng.poisson(0.12)),
            "beneficiary_age_hours": float(
                np.clip(rng.lognormal(6.1, 1.0), 24.0, 20_000.0)
            ),
            "inbound_counterparties_24h": int(rng.poisson(0.8)),
            "outbound_counterparties_24h": int(rng.poisson(0.7)),
            "distinct_beneficiaries_24h": int(rng.poisson(1.0)),
            "is_new_device": int(rng.random() < 0.035),
            "is_new_beneficiary": int(rng.random() < 0.04),
            "card_present": int(channel == "card_present"),
            "cvv_match": int(rng.random() >= 0.012),
            "billing_match": int(rng.random() >= 0.018),
            "login_failures_24h": int(rng.poisson(0.15)),
            "merchant_risk_score": float(rng.beta(1.25, 8.5)),
            "identity_consistency_score": float(rng.beta(9.0, 1.2)),
            "shared_device_accounts": int(1 + rng.poisson(0.22)),
            "policy_threshold_ratio": float(
                np.clip(rng.beta(1.2, 4.5) * 1.15, 0.02, 1.20)
            ),
            "round_amount_indicator": int(amount >= 100 and abs(amount % 100) < 3),
            "hour_of_day": timestamp.hour,
            "is_weekend": int(timestamp.weekday() >= 5),
            "scenario_ambiguity": 0.0,
        }

    @staticmethod
    def _make_legitimate(
        row: dict[str, Any], rng: np.random.Generator, recipe: Recipe
    ) -> None:
        if rng.random() >= recipe.legitimate_hard_negative_rate:
            return
        variant = int(rng.integers(0, 7))
        if variant == 0:
            row["amount"] = float(rng.uniform(800, 3500))
            row["amount_to_balance_ratio"] = float(rng.uniform(0.35, 0.72))
        elif variant == 1:
            row["is_new_device"] = 1
            row["device_age_days"] = 0
        elif variant == 2:
            row["distance_from_home_km"] = float(rng.uniform(600, 1800))
            row["minutes_since_previous_txn"] = float(rng.uniform(120, 900))
        elif variant == 3:
            row["prior_txn_1h"] = int(rng.integers(3, 6))
            row["channel"] = "card_not_present"
            row["card_present"] = 0
        elif variant == 4:
            row["merchant_risk_score"] = float(rng.uniform(0.65, 0.88))
            row["billing_match"] = 1
        elif variant == 5:
            row["inbound_counterparties_24h"] = int(rng.integers(4, 8))
            row["outbound_counterparties_24h"] = int(rng.integers(0, 3))
        else:
            row["account_age_days"] = int(rng.integers(40, 120))
            row["identity_consistency_score"] = float(rng.uniform(0.68, 0.88))

    def _apply_scenario(
        self,
        row: dict[str, Any],
        scenario: str,
        rng: np.random.Generator,
        recipe: Recipe,
    ) -> None:
        ambiguous = bool(rng.random() < recipe.ambiguous_rate)
        strength = recipe.signal_strength * (0.57 if ambiguous else 1.05)
        jitter = max(0.02, recipe.noise)
        row["scenario_ambiguity"] = float(ambiguous)

        if scenario == "cnp_velocity":
            row["channel"] = "card_not_present"
            row["card_present"] = 0
            row["prior_txn_1h"] = int(
                max(4, rng.normal(4.4 + 3.4 * strength, 1.3 + 2 * jitter))
            )
            row["prior_txn_24h"] = max(
                row["prior_txn_24h"], row["prior_txn_1h"] + int(rng.integers(2, 8))
            )
            row["is_new_device"] = 1
            row["device_age_days"] = int(rng.integers(0, 8 if ambiguous else 3))
            row["cvv_match"] = int(rng.random() > 0.55 * min(strength, 1.0))
            row["amount"] = float(rng.uniform(90, 450 + 650 * strength))
        elif scenario == "impossible_travel":
            row["transaction_country"] = str(
                rng.choice([c for c in HOME_COUNTRIES if c != row["home_country"]])
            )
            row["cross_border"] = 1
            row["distance_from_home_km"] = float(rng.uniform(820, 1250 + 1600 * strength))
            row["minutes_since_previous_txn"] = float(
                rng.uniform(8, 88 if ambiguous else 38)
            )
            row["is_new_device"] = int(rng.random() < 0.22 + 0.36 * min(strength, 1.0))
        elif scenario == "account_takeover":
            row["is_new_device"] = 1
            row["device_age_days"] = int(rng.integers(0, 3))
            row["login_failures_24h"] = int(max(2, rng.normal(2.2 + 3.0 * strength, 1.1)))
            row["is_new_beneficiary"] = 1
            row["beneficiary_age_hours"] = float(rng.uniform(0.2, 46 if ambiguous else 11))
            row["prior_declines_24h"] = int(max(1, rng.normal(1.2 + 1.8 * strength, 0.9)))
            row["amount_to_balance_ratio"] = float(
                rng.uniform(0.44, min(1.55, 0.65 + 0.55 * strength))
            )
            row["amount"] = min(
                row["available_balance"] * row["amount_to_balance_ratio"], 12_000.0
            )
        elif scenario == "mule_fanout":
            row["inbound_counterparties_24h"] = int(
                max(5, rng.normal(5.3 + 5.0 * strength, 1.5))
            )
            row["outbound_counterparties_24h"] = int(
                max(5, rng.normal(5.2 + 5.4 * strength, 1.6))
            )
            row["prior_txn_24h"] = int(max(8, rng.normal(9 + 8 * strength, 2.0)))
            row["distinct_beneficiaries_24h"] = int(
                max(4, rng.normal(4.2 + 4.0 * strength, 1.2))
            )
            row["beneficiary_age_hours"] = float(rng.uniform(1, 60 if ambiguous else 18))
            row["amount_to_balance_ratio"] = float(rng.uniform(0.28, 0.82))
        elif scenario == "synthetic_identity":
            row["account_age_days"] = int(rng.integers(18, 118))
            row["identity_consistency_score"] = float(rng.uniform(0.18, 0.54))
            row["shared_device_accounts"] = int(
                max(4, rng.normal(4.2 + 4.0 * strength, 1.4))
            )
            row["device_age_days"] = int(rng.integers(0, 20))
            row["merchant_risk_score"] = float(rng.uniform(0.42, 0.80))
        elif scenario == "merchant_mismatch":
            row["merchant_risk_score"] = float(
                rng.uniform(0.73, 0.91 + 0.07 * min(strength, 1.0))
            )
            row["billing_match"] = 0
            row["merchant_category"] = str(
                rng.choice(("digital_goods", "cash_equivalent", "electronics"))
            )
            row["amount"] = float(rng.uniform(260, 700 + 900 * strength))
            row["amount_to_balance_ratio"] = min(
                1.5, row["amount"] / max(row["available_balance"], 1.0)
            )
            row["transaction_country"] = str(
                rng.choice([c for c in HOME_COUNTRIES if c != row["home_country"]])
            )
            row["cross_border"] = 1
        elif scenario == "threshold_pattern":
            row["channel"] = "transfer"
            row["card_present"] = 0
            row["policy_threshold_ratio"] = float(rng.uniform(0.73, 1.01))
            row["prior_txn_24h"] = int(max(7, rng.normal(7.5 + 5.5 * strength, 1.7)))
            row["distinct_beneficiaries_24h"] = int(
                max(4, rng.normal(4.2 + 3.2 * strength, 1.1))
            )
            row["round_amount_indicator"] = int(
                rng.random() < 0.54 + 0.25 * min(strength, 1.0)
            )
            row["is_new_beneficiary"] = int(rng.random() < 0.58)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        self._apply_noise(row, rng, recipe.noise)

    @staticmethod
    def _apply_noise(row: dict[str, Any], rng: np.random.Generator, noise: float) -> None:
        if noise <= 0:
            return
        row["amount"] = float(max(1.0, row["amount"] * rng.lognormal(0.0, 0.18 * noise)))
        row["distance_from_home_km"] = float(
            max(0.0, row["distance_from_home_km"] + rng.normal(0, 95 * noise))
        )
        if rng.random() < noise * 0.20:
            row["is_new_device"] = 1 - int(row["is_new_device"])
        if rng.random() < noise * 0.12:
            row["billing_match"] = 1 - int(row["billing_match"])

    @staticmethod
    def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
        frame["amount"] = frame["amount"].clip(1.0, 25_000.0).round(2)
        frame["available_balance"] = frame["available_balance"].round(2)
        frame["amount_to_balance_ratio"] = (
            frame["amount"] / frame["available_balance"].clip(lower=1.0)
        ).clip(0.0, 2.0).round(4)
        frame["distance_from_home_km"] = frame["distance_from_home_km"].round(2)
        frame["minutes_since_previous_txn"] = frame["minutes_since_previous_txn"].round(2)
        frame["beneficiary_age_hours"] = frame["beneficiary_age_hours"].round(2)
        frame["merchant_risk_score"] = frame["merchant_risk_score"].clip(0, 1).round(4)
        frame["identity_consistency_score"] = (
            frame["identity_consistency_score"].clip(0, 1).round(4)
        )
        frame["policy_threshold_ratio"] = (
            frame["policy_threshold_ratio"].clip(0, 1.25).round(4)
        )
        return frame


def evidence_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute observable signatures without consulting the oracle label."""

    flags = pd.DataFrame(index=frame.index)
    flags["cnp_velocity"] = (
        frame["channel"].eq("card_not_present")
        & frame["prior_txn_1h"].ge(4)
        & frame["is_new_device"].eq(1)
    )
    flags["impossible_travel"] = (
        frame["distance_from_home_km"].gt(800)
        & frame["minutes_since_previous_txn"].lt(90)
        & frame["cross_border"].eq(1)
    )
    flags["account_takeover"] = (
        frame["is_new_device"].eq(1)
        & frame["login_failures_24h"].ge(2)
        & frame["is_new_beneficiary"].eq(1)
        & frame["beneficiary_age_hours"].lt(48)
    )
    flags["mule_fanout"] = (
        frame["inbound_counterparties_24h"].ge(5)
        & frame["outbound_counterparties_24h"].ge(5)
        & frame["prior_txn_24h"].ge(8)
    )
    flags["synthetic_identity"] = (
        frame["account_age_days"].lt(120)
        & frame["identity_consistency_score"].lt(0.56)
        & frame["shared_device_accounts"].ge(4)
    )
    flags["merchant_mismatch"] = (
        frame["merchant_risk_score"].gt(0.72)
        & frame["billing_match"].eq(0)
        & frame["amount"].gt(250)
    )
    flags["threshold_pattern"] = (
        frame["prior_txn_24h"].ge(7)
        & frame["policy_threshold_ratio"].between(0.72, 1.02)
        & frame["distinct_beneficiaries_24h"].ge(4)
    )
    return flags.astype(int)
