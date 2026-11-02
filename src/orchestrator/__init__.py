"""Coordinate clinical guidelines and catalog answers; see README.md for the flow.

Callers import ClinicalDrugOrchestrator, OrchestratorAnswer and OrchestratorUpdate here.
"""

# ---------------------------------------------------------------------------
# START HERE — Stable public imports; read README.md, then orchestrator.py
# ---------------------------------------------------------------------------
from .orchestrator import ClinicalDrugOrchestrator
from .schema import OrchestratorAnswer, OrchestratorUpdate

__all__ = ["ClinicalDrugOrchestrator", "OrchestratorAnswer", "OrchestratorUpdate"]
