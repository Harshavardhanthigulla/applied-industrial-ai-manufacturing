"""Load config.yaml and resolve every path relative to the repository root."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"

# Evidence tables (sheets of reports/evidence_tables.xlsx), grouped by the notebook that explains them.
TABLE_GROUPS = {
    "02_data_understanding": ["signal_dictionary", "sampling", "gaps", "sync_check", "shortest_pulses"],
    "03_data_preparation": ["resample_loss", "resample_event_sensitivity", "stops", "stop_halt_check", "cycles",
                            "drilling_cycles", "robot_moves"],
    "04_model_1": ["normal_durations", "events_rule", "threshold_sweep", "slow_steps", "step_dispersion",
                   "warehouse_travel", "attribution", "paired_output_windows", "encoder_silence", "belt_arrivals",
                   "barrier_changes_belt_off", "small_conveyor_arrivals", "barrier_edges_without_drive",
                   "equipment_quantification"],
    "05_model_2": ["ml_isolation_forest", "ml_contamination_sweep", "ml_kmeans", "ml_vs_rule", "ml_quantification"],
    "06_evaluation": ["line_quiet_periods", "cycle_reference", "cycle_handover", "performance", "events_final",
                      "event_evidence_register", "candidate_register", "incident_quantification",
                      "cycle_quantification", "results_by_level", "method_comparison"],
}


def load_config(path: Path | None = None) -> dict:
    """Return the configuration with all entries of ``paths`` as absolute Paths."""
    path = Path(path) if path else CONFIG_FILE
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("reports_dir", "figures_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
