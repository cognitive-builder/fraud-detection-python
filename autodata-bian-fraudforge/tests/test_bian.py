from fastapi.testclient import TestClient

from api import app
from fraudforge.bian import evaluate_bian_request


def _suspicious_record() -> dict:
    return {
        "transaction_id": "TX-DEMO-1",
        "amount": 4200,
        "available_balance": 5000,
        "channel": "transfer",
        "home_country": "US",
        "transaction_country": "GB",
        "distance_from_home_km": 5200,
        "minutes_since_previous_txn": 12,
        "is_new_device": 1,
        "device_age_days": 0,
        "login_failures_24h": 6,
        "is_new_beneficiary": 1,
        "beneficiary_age_hours": 2,
        "prior_declines_24h": 4,
        "prior_txn_1h": 8,
        "prior_txn_24h": 15,
        "inbound_counterparties_24h": 10,
        "outbound_counterparties_24h": 11,
        "distinct_beneficiaries_24h": 8,
        "merchant_risk_score": 0.96,
        "billing_match": 0,
        "identity_consistency_score": 0.3,
        "shared_device_accounts": 8,
        "account_age_days": 45,
        "policy_threshold_ratio": 0.94,
    }


def test_bian_function_and_api_contract() -> None:
    payload = {"FraudEvaluationTransactionConsolidationRecord": [_suspicious_record()]}
    result = evaluate_bian_request(payload)
    assert result["assessmentSummary"]["transaction_count"] == 1
    assert result["assessmentSummary"]["anomaly_count"] == 1
    assert result["FraudEvaluationProductionAnomalyRecord"][0][
        "recommendedDisposition"
    ] == "REFER_FOR_INVESTIGATION"

    client = TestClient(app)
    response = client.post("/FraudEvaluation/Evaluate", json=payload)
    assert response.status_code == 200
    assert response.json()["assessmentSummary"]["anomaly_count"] == 1
