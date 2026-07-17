from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraudforge.bian import evaluate_bian_request


class FraudEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    ProductProductionSessionReference: dict[str, Any] | str | None = None
    FraudEvaluationTestProfile: dict[str, Any] | str | None = None
    FraudEvaluationEnsembleTechniqueType: str = "RULE_MODEL_MAX"
    FraudEvaluationEnsembleTechniqueDefinition: str = (
        "Maximum of bounded rule score and calibrated model score; threshold 0.50"
    )
    FraudEvaluationTransactionConsolidationRecord: list[dict[str, Any]] | dict[str, Any] = Field(
        default_factory=list
    )


app = FastAPI(
    title="FraudForge BIAN Fraud Evaluation API",
    version="0.1.0",
    description=(
        "BIAN-aligned demonstration endpoint for evaluating fictional or pre-tokenized "
        "transaction features with transparent rules and a stronger ensemble."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service_domain": "Fraud Evaluation"}


@app.post("/FraudEvaluation/Evaluate")
def evaluate(payload: FraudEvaluationRequest) -> dict[str, Any]:
    try:
        return evaluate_bian_request(payload.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
