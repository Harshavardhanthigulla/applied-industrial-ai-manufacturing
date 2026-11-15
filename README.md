# Applied Industrial AI - MP1 Performance Loss Detection

Find the performance losses in one 8-minute recording of a small production line controlled by a Siemens S7 and a
Beckhoff PLC, quantify them and allocate them to physical equipment.

**Result**
- 8 performance-loss events: 4 line-wide stops (3 safety-curtain stops, 1 HMI stop) and 4 step overruns.
- The complete cycles with events took 33.7 s longer than their position-adjusted normal reference (20.9 %);
  performance is 91.8 % of demonstrated normal operation (upper bound). Official OEE performance needs the ideal cycle
  time, which is not given.
- Machine learning (Isolation Forest, K-Means) re-finds the rule's overruns but adds no verified loss.

Repository: https://github.com/Harshavardhanthigulla/applied-industrial-ai-manufacturing

## How to run

```bash
pip install -r requirements.txt
# run notebooks/01 ... 06 in order (each hands its results to the next through data/processed/)
python run_mp1.py          # optional: the same figures and evidence workbook without Jupyter
```

## Files

| Path | Content |
|---|---|
| `notebooks/01_business_understanding.ipynb` | task, course OEE definitions, hypotheses H1-H10, evidence rule |
| `notebooks/02_data_understanding_and_cleaning.ipynb` | quality checks of both PLCs, detection limit, signal meanings |
| `notebooks/03_data_preparation.ipynb` | time grid, clock check, stop windows, pallet cycles, activation table |
| `notebooks/04_model_1_engineering_detection.ipynb` | Model 1 rule and PLC evidence, attribution, reduced speed, robustness |
| `notebooks/05_model_2_machine_learning.ipynb` | feature table, Isolation Forest, K-Means, candidate review |
| `notebooks/06_evaluation_and_results.ipynb` | cycle-level effect, performance, registers, results by level, evidence workbook |
| `src/mp1/`, `run_mp1.py` | optional headless re-run of notebooks 02-06 |
| `config.yaml` | every path and threshold (the same values as the notebooks' parameter cells) |
| `data/raw/` | the original recording and variable file, never modified |
| `docs/methodology.md` | loss definition, assumptions, method choices, signal meanings, limitations |
| `docs/report.md` | technical report |
| `reports/evidence_tables.xlsx` | every evidence table, one sheet each (written by notebook 06) |
| `reports/figures/` | the figures of the report (written by notebooks 02, 04 and 06) |
| `reports/presentation.pptx` | slides |
| `requirements.txt`, `LICENSE` | dependencies, licence |
