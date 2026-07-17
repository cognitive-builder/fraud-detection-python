"""FraudForge: agentic synthetic data for BIAN Fraud Evaluation."""

from .config import Recipe, RunConfig, TargetBand
from .orchestrator import AutodataOrchestrator, RunResult

__all__ = ["AutodataOrchestrator", "Recipe", "RunConfig", "RunResult", "TargetBand"]
__version__ = "0.1.0"
