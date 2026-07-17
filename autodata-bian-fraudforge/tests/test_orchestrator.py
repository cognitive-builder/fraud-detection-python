from fraudforge.config import RunConfig
from fraudforge.orchestrator import AutodataOrchestrator


def test_agentic_loop_returns_high_quality_batch() -> None:
    result = AutodataOrchestrator(
        RunConfig(
            batch_size=350,
            max_rounds=6,
            seed=42,
            meta_search_candidates=1,
            validation_batch_size=160,
        )
    ).run()

    assert len(result.transactions) == 350
    assert len(result.rounds) >= 1
    assert result.metrics.validity_rate >= 0.97
    assert result.metrics.family_coverage == 1.0
    assert result.metrics.strong_recall >= 0.80
    assert result.metrics.strong_recall >= result.metrics.weak_recall
    assert result.trajectory["quality_score"].between(0, 1).all()
