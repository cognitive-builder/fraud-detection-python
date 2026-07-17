# BIAN use case: Fraud Evaluation

## Service-domain choice

BIAN’s fraud landscape separates the online evaluation of production activity from downstream
Fraud Diagnosis and Fraud Resolution. Fraud Evaluation runs behavioral-pattern tests and isolates
out-of-pattern transactions. This is the right boundary for a synthetic-data demonstration because
it has a clear input (transaction features), multiple test mechanisms (rules and models), and a
clear output (anomaly records) without pretending to resolve or investigate a fraud case.

## Demo business scenario

A bank is modernizing its real-time transaction-monitoring stack. Existing rules perform well on
known patterns but miss combinations involving device novelty, beneficiary age, merchant context,
identity consistency, geographic velocity, and network fan-in/fan-out. Real fraud labels are sparse,
delayed, sensitive, and biased toward previously deployed controls.

FraudForge creates a verified synthetic curriculum around the decision boundary. The bank uses it
to:

1. regression-test rule changes;
2. benchmark a contextual model against installed controls;
3. produce strong-win examples for training or preference optimization;
4. exercise BIAN-shaped service contracts before production data is connected; and
5. identify scenario families with weak coverage or excessive false positives.

## BIAN mapping

| BIAN concept | FraudForge mapping |
|---|---|
| `ProductProductionSessionReference` | fictional synthetic batch/session identifier |
| `FraudEvaluationTestProfile` | selected scenario catalog and acceptance policy |
| `FraudEvaluationEnsembleTechniqueType` | `RULE_MODEL_MAX` |
| `FraudEvaluationEnsembleTechniqueDefinition` | transparent rule/model combination description |
| `FraudEvaluationTransactionConsolidationRecord` | list of normalized transaction feature records |
| `FraudEvaluationProductionAnomalyRecord` | scored suspected-fraud records with evidence families |
| `FraudEvaluationProductionAnomalyProductionTransactionReference` | transaction identifiers |
| Rule Sets and Decision Trees behavior qualifier | `WeakRuleSolver` |
| Models behavior qualifier | `StrongEnsembleSolver` |

## Endpoint

```text
POST /FraudEvaluation/Evaluate
```

Example request:

```json
{
  "ProductProductionSessionReference": {
    "SessionIdentification": {"IdentifierValue": "demo-session-001"}
  },
  "FraudEvaluationTestProfile": {
    "Profile": "Agentic synthetic edge-case evaluation"
  },
  "FraudEvaluationEnsembleTechniqueType": "RULE_MODEL_MAX",
  "FraudEvaluationTransactionConsolidationRecord": [
    {
      "transaction_id": "TX-DEMO-1",
      "amount": 4200,
      "available_balance": 5000,
      "channel": "transfer",
      "home_country": "US",
      "transaction_country": "GB",
      "distance_from_home_km": 5200,
      "minutes_since_previous_txn": 12,
      "is_new_device": 1,
      "login_failures_24h": 6,
      "is_new_beneficiary": 1,
      "beneficiary_age_hours": 2
    }
  ]
}
```

The response includes an assessment reference, anomaly records, transaction references, rule/model
positive counts, and assessment summary.

## Fraud families

| Family | Defensive pattern represented |
|---|---|
| CNP velocity | burst of card-not-present activity plus device novelty |
| impossible travel | high geographic distance in an infeasible time window |
| account takeover | new device, login failures, and newly added beneficiary |
| mule fan-out | rapid multi-counterparty inflow/outflow behavior |
| synthetic identity | young account, inconsistent identity features, shared devices |
| merchant mismatch | risky merchant context, billing mismatch, cross-border activity |
| threshold pattern | repeated activity clustered around a policy boundary |

The patterns are intentionally descriptive rather than operational instructions for evasion.
