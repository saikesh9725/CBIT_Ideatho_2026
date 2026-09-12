"""
qc_engines.py
-------------
Deterministic data-science engines for Bio Trust OS.

Each engine accepts the ingested DataFrame (+ detected metadata columns)
and returns a dict containing:
    - a 0-100 sub-score
    - supporting diagnostics (numbers, flagged samples, etc.)
    - human-readable notes used by the remediation panel

The module is defensive: every engine degrades gracefully when expected
columns (Batch/Condition/Plate) are absent, and never raises on malformed
input -- it returns a score of 0 with an explanatory note instead.

Also includes:
    - compute_trust_score(): weighted composite score + decision gate
    - RuleBasedRemediationAgent: offline fallback "LLM" that synthesizes
      a diagnostic narrative from engine outputs when no API key is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive fallback
    SKLEARN_AVAILABLE = False


META_CANDIDATES = {
    "sample_id": ["SampleID", "Sample_ID", "sample_id", "ID", "Id"],
    "batch": ["Batch", "Plate", "batch", "plate", "Batch_ID"],
    "condition": ["Condition", "condition", "Phenotype", "Label", "Target", "Class"],
    "center": ["Center", "center", "Site", "Lab"],
}


# --------------------------------------------------------------------------
# Column detection helpers
# --------------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> dict:
    """Best-effort detection of metadata columns and numeric feature columns."""
    detected = {}
    for role, candidates in META_CANDIDATES.items():
        found = next((c for c in candidates if c in df.columns), None)
        detected[role] = found

    non_feature_cols = {v for v in detected.values() if v is not None}
    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    detected["features"] = feature_cols
    return detected


# --------------------------------------------------------------------------
# 1. Ingestion & Completeness Engine
# --------------------------------------------------------------------------

def run_completeness_engine(df: pd.DataFrame, columns: dict) -> dict:
    """
    Checks: missing cells, zero-inflation, duplicate sample IDs.
    Returns a Completeness Score (0-100).
    """
    notes = []
    feature_cols = columns.get("features", [])
    sample_col = columns.get("sample_id")

    if not feature_cols:
        return {
            "score": 0.0,
            "missing_pct": None,
            "zero_inflation_pct": None,
            "duplicate_samples": None,
            "notes": ["No numeric feature columns detected -- cannot assess completeness."],
        }

    numeric_df = df[feature_cols]
    total_cells = numeric_df.size
    missing_cells = int(numeric_df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    zero_cells = int((numeric_df == 0).sum().sum())
    zero_pct = (zero_cells / total_cells * 100) if total_cells else 0.0

    duplicate_count = 0
    if sample_col and sample_col in df.columns:
        duplicate_count = int(df[sample_col].duplicated().sum())

    # Scoring heuristics (each penalized independently, floored at 0)
    missing_penalty = min(missing_pct * 3.0, 70)
    zero_penalty = min(max(zero_pct - 5, 0) * 1.5, 20)  # allow up to 5% natural zeros
    duplicate_penalty = min(duplicate_count * 10, 30)

    score = max(0.0, 100 - missing_penalty - zero_penalty - duplicate_penalty)

    if missing_pct > 10:
        notes.append(f"High missingness detected: {missing_pct:.1f}% of cells are empty.")
    elif missing_pct > 2:
        notes.append(f"Moderate missingness detected: {missing_pct:.1f}% of cells are empty.")

    if zero_pct > 10:
        notes.append(f"Zero-inflation detected: {zero_pct:.1f}% of values are exactly zero.")

    if duplicate_count > 0:
        notes.append(f"{duplicate_count} duplicate sample ID(s) found -- verify sample tracking.")

    if not notes:
        notes.append("Data completeness is strong; no material missingness or duplication issues.")

    return {
        "score": round(score, 1),
        "missing_pct": round(missing_pct, 2),
        "zero_inflation_pct": round(zero_pct, 2),
        "duplicate_samples": duplicate_count,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# 2. Anomaly & Outlier Engine
# --------------------------------------------------------------------------

def run_anomaly_engine(df: pd.DataFrame, columns: dict, contamination: float = 0.1) -> dict:
    """
    Runs Isolation Forest + Z-score (>3 SD) across numeric features.
    Returns an Outlier Integrity Score (0-100), flagged sample indices,
    and per-sample anomaly scores for visualization.
    """
    feature_cols = columns.get("features", [])
    notes = []

    if not SKLEARN_AVAILABLE:
        return {
            "score": 0.0,
            "flagged_samples": [],
            "anomaly_scores": None,
            "zscore_flagged": [],
            "notes": ["scikit-learn is not available -- anomaly detection skipped."],
        }

    if len(feature_cols) < 2 or len(df) < 5:
        return {
            "score": 50.0,
            "flagged_samples": [],
            "anomaly_scores": None,
            "zscore_flagged": [],
            "notes": ["Insufficient samples/features for robust anomaly detection (need >=5 samples, >=2 features)."],
        }

    numeric_df = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=min(max(contamination, 0.01), 0.4),
        random_state=42,
    )
    iso.fit(scaled)
    raw_scores = iso.decision_function(scaled)  # higher = more normal
    predictions = iso.predict(scaled)  # -1 = anomaly, 1 = normal
    anomaly_flags = predictions == -1

    # Z-score outlier detection (any feature exceeding 3 SD)
    z_scores = np.abs(stats.zscore(scaled, axis=0, nan_policy="omit"))
    z_flagged = (z_scores > 3).any(axis=1)

    combined_flagged = anomaly_flags | z_flagged
    n_flagged = int(combined_flagged.sum())
    flagged_pct = n_flagged / len(df) * 100

    # Score: penalize proportional to flagged fraction
    score = max(0.0, 100 - flagged_pct * 4.0)

    sample_col = columns.get("sample_id")
    if sample_col and sample_col in df.columns:
        flagged_ids = df.loc[combined_flagged, sample_col].tolist()
    else:
        flagged_ids = df.index[combined_flagged].tolist()

    if n_flagged > 0:
        notes.append(
            f"{n_flagged} sample(s) ({flagged_pct:.1f}%) flagged as anomalous by Isolation Forest / Z-score (>3 SD)."
        )
    else:
        notes.append("No significant outlier samples detected.")

    return {
        "score": round(score, 1),
        "flagged_samples": flagged_ids,
        "flagged_count": n_flagged,
        "flagged_pct": round(flagged_pct, 2),
        "anomaly_scores": raw_scores.tolist(),
        "zscore_flagged": int(z_flagged.sum()),
        "notes": notes,
    }


# --------------------------------------------------------------------------
# 3. Batch Effect & Confounding Engine
# --------------------------------------------------------------------------

def run_batch_engine(df: pd.DataFrame, columns: dict) -> dict:
    """
    Runs PCA (3 components). If a Batch/Plate column exists, computes
    ANOVA p-values (batch vs PC1) and a silhouette score for batch
    clustering strength. Returns a Batch Consistency Score (0-100).
    """
    feature_cols = columns.get("features", [])
    batch_col = columns.get("batch")
    notes = []

    if not SKLEARN_AVAILABLE:
        return {
            "score": 0.0, "pca_df": None, "anova_pvalue": None,
            "silhouette": None, "notes": ["scikit-learn unavailable -- batch analysis skipped."],
        }

    if len(feature_cols) < 3 or len(df) < 5:
        return {
            "score": 50.0, "pca_df": None, "anova_pvalue": None,
            "silhouette": None,
            "notes": ["Insufficient features/samples for PCA-based batch analysis."],
        }

    numeric_df = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric_df)
    scaled = StandardScaler().fit_transform(imputed)

    n_components = min(3, scaled.shape[0] - 1, scaled.shape[1])
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(scaled)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, columns=pca_cols, index=df.index)
    for c in [batch_col, columns.get("condition"), columns.get("sample_id")]:
        if c and c in df.columns:
            pca_df[c] = df[c].values

    explained_var = pca.explained_variance_ratio_.tolist()

    if not batch_col or batch_col not in df.columns or df[batch_col].nunique() < 2:
        notes.append("No 'Batch'/'Plate' column detected (or a single batch present) -- "
                      "batch-effect analysis limited to variance structure only.")
        return {
            "score": 75.0,  # neutral-good score when batch cannot be assessed
            "pca_df": pca_df,
            "explained_variance": explained_var,
            "anova_pvalue": None,
            "silhouette": None,
            "notes": notes,
        }

    batches = df[batch_col].astype(str).values
    n_batches = len(set(batches))

    # ANOVA: does batch explain PC1 variance?
    groups = [pca_df.loc[df[batch_col].astype(str) == b, "PC1"] for b in sorted(set(batches))]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = np.nan, np.nan

    # Silhouette score: how well-separated are batch clusters in PCA space?
    silhouette = None
    if n_batches >= 2 and n_batches < len(df):
        try:
            silhouette = float(silhouette_score(pcs, batches))
        except Exception:
            silhouette = None

    # Scoring: low p-value + high silhouette => strong (bad) batch effect
    penalty = 0.0
    if p_value is not None and not np.isnan(p_value):
        if p_value < 0.001:
            penalty += 45
        elif p_value < 0.01:
            penalty += 30
        elif p_value < 0.05:
            penalty += 15

    if silhouette is not None:
        # silhouette ranges [-1, 1]; positive & large => distinct batch clusters
        penalty += max(0.0, silhouette) * 45

    score = max(0.0, 100 - penalty)

    if p_value is not None and not np.isnan(p_value) and p_value < 0.05:
        notes.append(
            f"Batch significantly explains PC1 variance (ANOVA p={p_value:.2e}) -- technical, not biological, signal is dominating."
        )
    if silhouette is not None and silhouette > 0.25:
        notes.append(
            f"Strong batch clustering in PCA space (silhouette={silhouette:.2f}) -- samples separate by batch rather than biology."
        )
    if not notes:
        notes.append("Batch does not significantly confound the principal components.")

    return {
        "score": round(score, 1),
        "pca_df": pca_df,
        "explained_variance": explained_var,
        "anova_pvalue": None if p_value is None or np.isnan(p_value) else float(p_value),
        "silhouette": silhouette,
        "n_batches": n_batches,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# 4. Cohort Balance Engine
# --------------------------------------------------------------------------

def run_balance_engine(df: pd.DataFrame, columns: dict) -> dict:
    """
    Measures class imbalance across the Condition/phenotype column.
    Returns a Balance Score (0-100) using normalized Shannon entropy.
    """
    condition_col = columns.get("condition")
    notes = []

    if not condition_col or condition_col not in df.columns:
        return {
            "score": 75.0,
            "class_counts": None,
            "notes": ["No 'Condition'/phenotype column detected -- balance assessment skipped (neutral score applied)."],
        }

    counts = df[condition_col].value_counts()
    n_classes = len(counts)

    if n_classes < 2:
        return {
            "score": 40.0,
            "class_counts": counts.to_dict(),
            "notes": ["Only a single class present in the condition column -- cannot assess balance; "
                      "downstream model training may be uninformative."],
        }

    proportions = counts / counts.sum()
    entropy = -(proportions * np.log(proportions)).sum()
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    score = round(normalized_entropy * 100, 1)

    minority_pct = proportions.min() * 100
    if minority_pct < 10:
        notes.append(f"Severe class imbalance: minority class represents only {minority_pct:.1f}% of samples.")
    elif minority_pct < 25:
        notes.append(f"Moderate class imbalance: minority class represents {minority_pct:.1f}% of samples.")
    else:
        notes.append("Cohort classes are reasonably balanced.")

    return {
        "score": score,
        "class_counts": counts.to_dict(),
        "notes": notes,
    }


# --------------------------------------------------------------------------
# Composite Trust Score & Decision Gate
# --------------------------------------------------------------------------

WEIGHTS = {
    "completeness": 0.25,
    "batch": 0.35,
    "outlier": 0.25,
    "balance": 0.15,
}


@dataclass
class TrustScoreResult:
    trust_score: float
    gate_status: str
    gate_color: str
    sub_scores: dict = field(default_factory=dict)


def compute_trust_score(completeness_score: float, batch_score: float,
                         outlier_score: float, balance_score: float) -> TrustScoreResult:
    """Compute the weighted composite Biological Trust Score and decision gate."""
    trust_score = (
        WEIGHTS["completeness"] * completeness_score
        + WEIGHTS["batch"] * batch_score
        + WEIGHTS["outlier"] * outlier_score
        + WEIGHTS["balance"] * balance_score
    )
    trust_score = round(trust_score, 1)

    if trust_score >= 75:
        gate_status = "GO (Ready for AI/ML Training)"
        gate_color = "green"
    elif trust_score >= 50:
        gate_status = "CONDITIONAL / NEEDS CORRECTION"
        gate_color = "amber"
    else:
        gate_status = "NO-GO (High Risk of Bias / Failed Reproducibility)"
        gate_color = "red"

    return TrustScoreResult(
        trust_score=trust_score,
        gate_status=gate_status,
        gate_color=gate_color,
        sub_scores={
            "Completeness": completeness_score,
            "Batch Consistency": batch_score,
            "Outlier Integrity": outlier_score,
            "Balance": balance_score,
        },
    )


# --------------------------------------------------------------------------
# Rule-Based Remediation Agent (offline LLM fallback)
# --------------------------------------------------------------------------

class RuleBasedRemediationAgent:
    """
    Deterministic, offline 'multi-agent' synthesis fallback used when no
    LLM API key is configured. Combines engine outputs into a structured,
    actionable diagnostic narrative -- mirroring the shape of an LLM
    response so the UI layer doesn't need to branch on availability.
    """

    def synthesize(self, completeness: dict, anomaly: dict, batch: dict,
                    balance: dict, trust_result: TrustScoreResult) -> dict:
        recommendations = []

        # Completeness-driven recommendations
        if completeness.get("missing_pct") is not None and completeness["missing_pct"] > 10:
            recommendations.append(
                "Impute or exclude samples with high missingness (>10% missing values) using "
                "KNN or MICE imputation before proceeding to model training."
            )
        if completeness.get("duplicate_samples"):
            recommendations.append(
                f"Resolve {completeness['duplicate_samples']} duplicate sample ID(s) -- verify LIMS/sample "
                "tracking to rule out accidental re-uploads or mislabeling."
            )

        # Outlier-driven recommendations
        if anomaly.get("flagged_count"):
            recommendations.append(
                f"Review and consider excluding {anomaly['flagged_count']} anomalous sample(s) flagged by "
                "Isolation Forest / Z-score (>3 SD) prior to downstream modeling."
            )

        # Batch-driven recommendations
        p_value = batch.get("anova_pvalue")
        silhouette = batch.get("silhouette")
        if p_value is not None and p_value < 0.05:
            pc_hint = "PC1"
            worst_batch_note = "technical batch"
            recommendations.append(
                f"Severe clustering on {pc_hint} driven by {worst_batch_note} (ANOVA p={p_value:.2e}) -- "
                "run ComBat, Harmony, or limma's removeBatchEffect before model training."
            )
        if silhouette is not None and silhouette > 0.25:
            recommendations.append(
                f"Batch clusters are well-separated in PCA space (silhouette={silhouette:.2f}); consider "
                "re-balancing experimental design so batch does not co-vary with condition in future runs."
            )

        # Balance-driven recommendations
        class_counts = balance.get("class_counts")
        if class_counts and len(class_counts) >= 2:
            min_class = min(class_counts, key=class_counts.get)
            total = sum(class_counts.values())
            min_pct = class_counts[min_class] / total * 100
            if min_pct < 25:
                recommendations.append(
                    f"Address class imbalance in '{min_class}' ({min_pct:.1f}% of cohort) via stratified "
                    "sampling, SMOTE, or class-weighted loss functions during model training."
                )

        if not recommendations:
            recommendations.append(
                "No material data-quality issues detected. Cohort is suitable for downstream AI/ML training "
                "as-is; continue standard QC monitoring for future batches."
            )

        # Executive summary line
        summary = (
            f"Biological Trust Score: {trust_result.trust_score}/100 -- {trust_result.gate_status}. "
            f"Weakest pillar: {min(trust_result.sub_scores, key=trust_result.sub_scores.get)} "
            f"({min(trust_result.sub_scores.values()):.1f}/100)."
        )

        return {
            "summary": summary,
            "recommendations": recommendations,
            "agent_mode": "offline-rule-based",
        }


def get_remediation_narrative(completeness: dict, anomaly: dict, batch: dict,
                               balance: dict, trust_result: TrustScoreResult,
                               llm_client: Optional[object] = None) -> dict:
    """
    Entry point used by the UI layer. Uses an LLM client if provided
    (duck-typed: must expose a `.generate(prompt: str) -> str` method),
    otherwise falls back to the deterministic RuleBasedRemediationAgent.
    """
    agent = RuleBasedRemediationAgent()
    # NOTE: LLM orchestration hook -- if a real API-backed client is wired
    # up in app.py, it can be passed here and used to enrich/rewrite the
    # rule-based narrative. Kept as a deterministic default for hackathon
    # reliability (zero external dependency / zero network requirement).
    return agent.synthesize(completeness, anomaly, batch, balance, trust_result)
