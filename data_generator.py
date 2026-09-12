"""
data_generator.py
------------------
Synthetic biological dataset generators for Bio Trust OS.

Provides two in-memory benchmark cohorts so the application works
out-of-the-box without requiring any file upload:

    1. generate_clean_cohort()      -> High-trust, well-balanced cohort
    2. generate_corrupted_cohort()  -> Low-trust, batch-confounded cohort

Both return a pandas DataFrame shaped like a typical omics matrix:
rows = samples, columns = numeric features (genes/proteins/metabolites)
plus metadata columns: 'SampleID', 'Batch', 'Condition', 'Center'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def _make_feature_names(n_features: int, prefix: str = "GENE") -> list:
    """Generate deterministic, readable feature (gene) names."""
    return [f"{prefix}_{i:04d}" for i in range(1, n_features + 1)]


def generate_clean_cohort(
    n_samples: int = 120,
    n_features: int = 60,
    n_batches: int = 3,
    n_conditions: int = 2,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a 'Benchmarked Clean Cohort':
      - Minimal batch effect (batches share the same distribution)
      - No meaningful outliers
      - Balanced condition classes
      - Negligible missingness
    Expected downstream Biological Trust Score: > 85.
    """
    rng = np.random.default_rng(seed)

    feature_names = _make_feature_names(n_features)

    # True biological signal: condition explains a modest amount of variance
    condition_labels = np.array(
        [f"Condition_{c}" for c in (np.arange(n_samples) % n_conditions)]
    )
    rng.shuffle(condition_labels)
    condition_effect = np.where(condition_labels == "Condition_0", 0.0, 0.6)

    # Base expression matrix: standard normal, no batch shift
    base = rng.normal(loc=10.0, scale=1.0, size=(n_samples, n_features))

    # Add a small, consistent biological signal on ~10% of features
    signal_features = rng.choice(
        n_features, size=max(1, n_features // 10), replace=False
    )
    for f_idx in signal_features:
        base[:, f_idx] += condition_effect

    # Balanced batches, evenly interleaved (no confounding with condition)
    batches = np.array([f"Batch_{b}" for b in (np.arange(n_samples) % n_batches)])
    rng.shuffle(batches)
    # Only a tiny, negligible batch shift (well within noise)
    for b in range(n_batches):
        mask = batches == f"Batch_{b}"
        base[mask, :] += rng.normal(0, 0.05, size=n_features)

    centers = np.array([f"Center_{c}" for c in (np.arange(n_samples) % 2)])
    rng.shuffle(centers)

    df = pd.DataFrame(base, columns=feature_names)
    df.insert(0, "SampleID", [f"S{str(i).zfill(4)}" for i in range(1, n_samples + 1)])
    df["Batch"] = batches
    df["Condition"] = condition_labels
    df["Center"] = centers

    # Negligible missingness (<0.5%)
    n_missing = int(0.003 * n_samples * n_features)
    if n_missing > 0:
        rows = rng.integers(0, n_samples, n_missing)
        cols = rng.choice(feature_names, n_missing)
        for r, c in zip(rows, cols):
            df.loc[r, c] = np.nan

    return df


def generate_corrupted_cohort(
    n_samples: int = 120,
    n_features: int = 60,
    n_batches: int = 3,
    n_conditions: int = 2,
    seed: int = RANDOM_SEED + 1,
) -> pd.DataFrame:
    """
    Generate a 'Compromised / Batch-Corrupted Cohort':
      - Severe batch effect (batches strongly separated in PCA space)
      - High missingness (~15%)
      - Extreme outlier samples injected
      - Confounded / imbalanced condition classes (batch correlates with condition)
    Expected downstream Biological Trust Score: < 45.
    """
    rng = np.random.default_rng(seed)

    feature_names = _make_feature_names(n_features)

    # Imbalanced condition: ~85% Condition_0, 15% Condition_1
    condition_labels = rng.choice(
        ["Condition_0", "Condition_1"],
        size=n_samples,
        p=[0.85, 0.15],
    )

    base = rng.normal(loc=10.0, scale=1.0, size=(n_samples, n_features))

    # Batches strongly confounded with condition (technical variance dominates)
    batches = []
    for c in condition_labels:
        if c == "Condition_0":
            batches.append(rng.choice(["Batch_0", "Batch_1"], p=[0.7, 0.3]))
        else:
            batches.append("Batch_2")
    batches = np.array(batches)

    # Inject a LARGE, systematic batch shift + scale change per batch
    for b_idx, b in enumerate(sorted(set(batches))):
        mask = batches == b
        shift = (b_idx + 1) * rng.normal(4.0, 0.5, size=n_features)
        scale = 1.0 + b_idx * 0.8
        base[mask, :] = base[mask, :] * scale + shift

    centers = np.array([f"Center_{c}" for c in (np.arange(n_samples) % 2)])
    rng.shuffle(centers)

    df = pd.DataFrame(base, columns=feature_names)
    df.insert(0, "SampleID", [f"S{str(i).zfill(4)}" for i in range(1, n_samples + 1)])
    df["Batch"] = batches
    df["Condition"] = condition_labels
    df["Center"] = centers

    # Inject extreme outlier samples (~8% of cohort)
    n_outliers = max(2, int(0.08 * n_samples))
    outlier_rows = rng.choice(n_samples, n_outliers, replace=False)
    for r in outlier_rows:
        blown_features = rng.choice(feature_names, size=n_features // 3, replace=False)
        df.loc[r, blown_features] = df.loc[r, blown_features] * rng.choice([-1, 1]) * rng.uniform(8, 15)

    # High missingness (~15%), including a couple of near-empty rows
    n_missing = int(0.15 * n_samples * n_features)
    rows = rng.integers(0, n_samples, n_missing)
    cols = rng.choice(feature_names, n_missing)
    for r, c in zip(rows, cols):
        df.loc[r, c] = np.nan

    # A couple of duplicate sample IDs to trip the completeness engine
    if n_samples > 5:
        df.loc[1, "SampleID"] = df.loc[0, "SampleID"]

    # Zero-inflate a chunk of features for a subset of samples
    zero_features = rng.choice(feature_names, size=max(1, n_features // 8), replace=False)
    zero_rows = rng.choice(n_samples, size=n_samples // 4, replace=False)
    for r in zero_rows:
        df.loc[r, zero_features] = 0.0

    return df


SAMPLE_DATASETS = {
    "Benchmarked Clean Cohort": generate_clean_cohort,
    "Compromised / Batch-Corrupted Cohort": generate_corrupted_cohort,
}


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Look up and generate a sample dataset by display name."""
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"Unknown sample dataset: {name}")
    return SAMPLE_DATASETS[name]()
