from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import export_run
from .config import Recipe, RunConfig, TargetBand
from .orchestrator import AutodataOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FraudForge agentic synthetic-data loop."
    )
    parser.add_argument("--output", default="artifacts/latest")
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-candidates", type=int, default=3)
    parser.add_argument("--fraud-prevalence", type=float, default=0.30)
    parser.add_argument("--ambiguity", type=float, default=0.35)
    parser.add_argument("--signal-strength", type=float, default=1.20)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--hard-negative-rate", type=float, default=0.18)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RunConfig(
        batch_size=args.rows,
        max_rounds=args.rounds,
        seed=args.seed,
        meta_search_candidates=args.meta_candidates,
    )
    recipe = Recipe(
        fraud_prevalence=args.fraud_prevalence,
        ambiguous_rate=args.ambiguity,
        signal_strength=args.signal_strength,
        noise=args.noise,
        legitimate_hard_negative_rate=args.hard_negative_rate,
    )
    result = AutodataOrchestrator(config, TargetBand(), recipe).run()
    paths = export_run(result, Path(args.output))
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "selected_round": result.selected_round,
                "metrics": result.metrics.to_dict(),
                "artifacts": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
