from pathlib import Path

from fraudforge.artifacts import export_run
from fraudforge.config import RunConfig
from fraudforge.orchestrator import AutodataOrchestrator

if __name__ == "__main__":
    result = AutodataOrchestrator(
        RunConfig(batch_size=700, max_rounds=7, seed=42, meta_search_candidates=3)
    ).run()
    export_run(result, Path("examples/sample_run"))
    print(result.metrics.to_dict())
