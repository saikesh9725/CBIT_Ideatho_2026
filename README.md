# 🧬 Bio Trust OS
### AI-Powered Biological Data Trust & Readiness Platform

Bio Trust OS ingests biological tabular datasets (gene expression, proteomics,
metabolomics, or clinical cohort CSVs), runs deterministic QC/anomaly/batch-effect
engines, computes a **Biological Trust Score (0–100)**, and issues a **Go / Conditional
/ No-Go** decision gate — with interactive Plotly visualizations and an exportable
executive QC report.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key or internet connection is required — the app works fully out of the box
using two built-in synthetic benchmark cohorts (generated in-memory by
`data_generator.py`), and remediation narratives are produced by an offline,
deterministic rule-based agent (`qc_engines.RuleBasedRemediationAgent`).

## Project Structure

```
app.py             Streamlit UI: layout, charts, export, orchestration
qc_engines.py       Deterministic QC engines: completeness, anomaly/outlier
                    (Isolation Forest + Z-score), batch effect (PCA + ANOVA +
                    silhouette), cohort balance, composite trust scoring,
                    and the offline remediation agent
data_generator.py   Synthetic "Benchmarked Clean Cohort" and
                    "Compromised / Batch-Corrupted Cohort" generators
requirements.txt    Python dependencies
```

## Input Format

CSV/TSV with:
- Rows = samples
- Numeric columns = features (genes/proteins/metabolites)
- Optional metadata columns (auto-detected, case-insensitive aliases supported):
  `SampleID`, `Batch` (or `Plate`), `Condition` (or `Phenotype`/`Label`/`Target`/`Class`), `Center` (or `Site`/`Lab`)

Missing metadata columns degrade gracefully — each engine returns a neutral
score and an explanatory note rather than failing.

## Scoring

```
Biological Trust Score =
      0.25 * Completeness
    + 0.35 * Batch Consistency
    + 0.25 * Outlier Integrity
    + 0.15 * Balance
```

| Trust Score | Gate |
|---|---|
| ≥ 75 | 🟢 GO (Ready for AI/ML Training) |
| 50–74 | 🟠 CONDITIONAL / NEEDS CORRECTION |
| < 50 | 🔴 NO-GO (High Risk of Bias / Failed Reproducibility) |
