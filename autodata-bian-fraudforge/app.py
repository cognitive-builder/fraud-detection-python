from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import plotly.express as px

from fraudforge.artifacts import export_run
from fraudforge.bian import evaluate_bian_request, sample_bian_request
from fraudforge.config import Recipe, RunConfig, TargetBand
from fraudforge.orchestrator import AutodataOrchestrator


def _summary(result, utility: dict) -> str:
    status = "✅ ACCEPTED" if result.accepted else "🟠 BEST AVAILABLE BATCH"
    metrics = result.metrics
    delta = utility.get("delta", {})
    return f"""
## {status}

**Selected round:** {result.selected_round}  
**Rows:** {metrics.row_count:,} · **Fraud prevalence:** {metrics.fraud_prevalence:.1%}  
**Weak recall:** {metrics.weak_recall:.1%} · **Strong recall:** {metrics.strong_recall:.1%} · **Recall gap:** {metrics.recall_gap:.1%}  
**Strong false-positive rate:** {metrics.strong_false_positive_rate:.1%}  
**Verifier validity:** {metrics.validity_rate:.1%} · **Family coverage:** {metrics.family_coverage:.1%} · **Duplicates:** {metrics.duplicate_rate:.2%}  
**Strong-win preference pairs:** {metrics.preference_pair_rate:.1%} · **Quality score:** {metrics.quality_score:.3f}  
**Shifted-holdout F1 uplift:** {delta.get('f1', 0.0):+.3f} · **ROC-AUC uplift:** {delta.get('roc_auc', 0.0):+.3f}

The batch is selected by bounded learning-signal gates—not by maximizing difficulty alone.
"""


def run_lab(
    rows: int,
    rounds: int,
    seed: int,
    fraud_prevalence: float,
    ambiguity: float,
    signal_strength: float,
    noise: float,
    hard_negative_rate: float,
    meta_candidates: int,
    progress=gr.Progress(track_tqdm=False),  # noqa: B008
):
    progress(0.05, desc="Initializing fraud evaluators")
    config = RunConfig(
        batch_size=int(rows),
        max_rounds=int(rounds),
        seed=int(seed),
        meta_search_candidates=int(meta_candidates),
        validation_batch_size=max(160, min(400, int(rows) // 3)),
    )
    recipe = Recipe(
        fraud_prevalence=float(fraud_prevalence),
        ambiguous_rate=float(ambiguity),
        signal_strength=float(signal_strength),
        noise=float(noise),
        legitimate_hard_negative_rate=float(hard_negative_rate),
    )
    progress(0.20, desc="Running inner loop and held-out recipe search")
    result = AutodataOrchestrator(config, TargetBand(), recipe).run()
    progress(0.78, desc="Exporting verified artifacts and BIAN payloads")
    output = Path(tempfile.mkdtemp(prefix="fraudforge-")) / "run"
    paths = export_run(result, output)

    utility = json.loads((output / "downstream_utility.json").read_text(encoding="utf-8"))
    bian_request = sample_bian_request(result.transactions, limit=8)
    bian_response = evaluate_bian_request(bian_request)
    trajectory = result.trajectory.copy()
    chart_data = trajectory.melt(
        id_vars="round",
        value_vars=["weak_recall", "strong_recall", "recall_gap", "quality_score"],
        var_name="metric",
        value_name="value",
    )
    figure = px.line(
        chart_data,
        x="round",
        y="value",
        color="metric",
        markers=True,
        title="Agentic recipe trajectory",
    )
    preview_columns = [
        "transaction_id",
        "scenario_type",
        "amount",
        "channel",
        "weak_score",
        "strong_score",
        "weak_prediction",
        "strong_prediction",
        "verifier_pass",
        "preference_pair",
    ]
    preview = result.transactions[preview_columns].head(120)
    progress(1.0, desc="Complete")
    return (
        _summary(result, utility),
        result.metrics.to_dict(),
        trajectory,
        result.family_metrics,
        preview,
        figure,
        bian_request,
        bian_response,
        paths["archive"],
    )


CSS = """
#hero {text-align: center; margin-bottom: 0.8rem}
#hero h1 {font-size: 2.2rem; margin-bottom: 0.2rem}
.metric-note {font-size: 0.92rem; opacity: 0.86}
"""

with gr.Blocks(title="FraudForge Autodata Lab") as demo:
    gr.Markdown(
        """
# 🏦 FraudForge Autodata Lab
### Agentic synthetic data for the BIAN **Fraud Evaluation** service domain

Generate a fictional transaction curriculum, measure a bounded weak rule solver against a stronger
contextual model, verify every planted pattern deterministically, revise the generation recipe, and
export RL/SFT/evaluation-ready artifacts plus a BIAN-aligned API payload.
""",
        elem_id="hero",
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Experiment controls")
            rows = gr.Slider(300, 3000, value=900, step=100, label="Transactions per round")
            rounds = gr.Slider(1, 12, value=7, step=1, label="Maximum agentic rounds")
            seed = gr.Number(value=42, precision=0, label="Seed")
            meta_candidates = gr.Slider(
                0, 7, value=3, step=1, label="Held-out recipe candidates"
            )
            fraud_prevalence = gr.Slider(
                0.10, 0.50, value=0.30, step=0.01, label="Fraud prevalence"
            )
            ambiguity = gr.Slider(
                0.0, 0.65, value=0.35, step=0.01, label="Initial ambiguity"
            )
            signal_strength = gr.Slider(
                0.50, 1.60, value=1.20, step=0.01, label="Initial signal strength"
            )
            noise = gr.Slider(
                0.0, 0.40, value=0.12, step=0.01, label="Initial feature noise"
            )
            hard_negative_rate = gr.Slider(
                0.0,
                0.45,
                value=0.18,
                step=0.01,
                label="Legitimate hard negatives",
            )
            run_button = gr.Button("Run agentic data scientist", variant="primary")
            gr.Markdown(
                "Deterministic verifier gates are the primary oracle. No API key or external LLM is required.",
                elem_classes="metric-note",
            )
        with gr.Column(scale=2):
            summary = gr.Markdown("Run the lab to inspect the selected curriculum.")
            trajectory_plot = gr.Plot(label="Trajectory")

    with gr.Tabs():
        with gr.Tab("Quality metrics"):
            metrics = gr.JSON(label="Accepted-batch metrics")
            trajectory = gr.Dataframe(
                label="Round-by-round recipe trajectory", interactive=False
            )
        with gr.Tab("Fraud-family diagnostics"):
            family = gr.Dataframe(label="Per-family solver behavior", interactive=False)
        with gr.Tab("Verified transaction preview"):
            preview = gr.Dataframe(label="Synthetic records", interactive=False)
        with gr.Tab("BIAN request"):
            bian_request = gr.JSON(label="POST /FraudEvaluation/Evaluate")
        with gr.Tab("BIAN response"):
            bian_response = gr.JSON(label="FraudEvaluation assessment")
        with gr.Tab("Export"):
            archive = gr.File(label="Complete run bundle")
            gr.Markdown(
                "The ZIP contains transactions, recipes, trajectories, verifier logs, "
                "rollouts, preference pairs, SFT examples, rubric items, utility metrics, and BIAN payloads."
            )

    run_button.click(
        fn=run_lab,
        inputs=[
            rows,
            rounds,
            seed,
            fraud_prevalence,
            ambiguity,
            signal_strength,
            noise,
            hard_negative_rate,
            meta_candidates,
        ],
        outputs=[
            summary,
            metrics,
            trajectory,
            family,
            preview,
            trajectory_plot,
            bian_request,
            bian_response,
            archive,
        ],
        api_name="run_autodata_lab",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )
