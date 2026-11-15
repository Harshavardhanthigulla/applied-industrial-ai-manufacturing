"""Optional headless re-run: the figures and reports/evidence_tables.xlsx from the raw data.

    python run_mp1.py

The analysis itself is developed and explained in notebooks/01-06.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mp1.run import main  # noqa: E402

if __name__ == "__main__":
    main()
