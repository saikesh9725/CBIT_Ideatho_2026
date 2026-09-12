"""
app.py
------
Bio Trust OS: AI-Powered Biological Data Trust & Readiness Platform.

Streamlit entry point. Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import SAMPLE_DATASETS, load_sample_dataset
import qc_engines as qce


# ==========================================================================
# Page configuration & global style
# ==========================================================================

st.set_page_config(
    page_title="Bio Trust OS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp {
        background-color: #0B0F19;
        color: #E5E7EB;
    }
    section[data-testid="stSidebar"] {
        background-color: #10141F;
        border-right: 1px solid #1F2937;
    }
    div[data-testid="stMetric"] {
        background-color: #131826;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #9CA3AF; }
    .gate-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        text-align: center;
    }
    .gate-green  { background-color: rgba(16,185,129,0.15); color: #10B981; border: 1px solid #10B981; }
    .gate-amber  { background-color: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid #F59E0B; }
    .gate-red    { background-color: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid #EF4444; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #E5E7EB;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
        border-left: 4px solid #6366F1;
        padding-left: 10px;
    }
    .remediation-item {
        background-color: #131826;
        border: 1px solid #1F2937;
        border-left: 3px solid #6366F1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .small-note { color: #9CA3AF; font-size: 0.85rem; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"


# ==========================================================================
# Helpers
# ==========================================================================

def render_gate_badge(gate_status: str, gate_color: str) -> str:
    css_class = {"green": "gate-green", "amber": "gate-amber", "red": "gate-red"}.get(gate_color, "gate-amber")
    return f'<span class="gate-badge {css_class}">{gate_status}</span>'


@st.cache_data(show_spinner=False)
def _load_sample(name: str) -> pd.DataFrame:
    return load_sample_dataset(name)


def run_full_pipeline(df: pd.DataFrame):
    """Run all four QC engines + composite scoring. Fully defensive."""
    columns = qce.detect_columns(df)

    completeness = qce.run_completeness_engine(df, columns)
    anomaly = qce.run_anomaly_engine(df, columns)
    batch = qce.run_batch_engine(df, columns)
    balance = qce.run_balance_engine(df, columns)

    trust_result = qce.compute_trust_score(
        completeness_score=completeness["score"],
        batch_score=batch["score"],
        outlier_score=anomaly["score"],
        balance_score=balance["score"],
    )

    remediation = qce.get_remediation_narrative(completeness, anomaly, batch, balance, trust_result)

    return {
        "columns": columns,
        "completeness": completeness,
        "anomaly": anomaly,
        "batch": batch,
        "balance": balance,
        "trust_result": trust_result,
        "remediation": remediation,
    }


def build_report_markdown(df: pd.DataFrame, results: dict, dataset_name: str) -> str:
    tr = results["trust_result"]
    lines = [
        f"# Bio Trust OS -- Executive QC Summary Report",
        f"",
        f"**Dataset:** {dataset_name}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Samples:** {len(df)}  **Features analyzed:** {len(results['columns'].get('features', []))}",
        f"",
        f"## Biological Trust Score: {tr.trust_score} / 100",
        f"### Decision Gate: {tr.gate_status}",
        f"",
        f"## Quality Pillar Scores",
        f"| Pillar | Score |",
        f"|---|---|",
    ]
    for k, v in tr.sub_scores.items():
        lines.append(f"| {k} | {v:.1f} |")

    lines += ["", "## Diagnostic Notes"]
    for engine_name, engine_result in [
        ("Completeness", results["completeness"]),
        ("Outlier / Anomaly", results["anomaly"]),
        ("Batch Effect", results["batch"]),
        ("Cohort Balance", results["balance"]),
    ]:
        lines.append(f"**{engine_name}:**")
        for note in engine_result.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## AI Remediation Recommendations")
    lines.append(results["remediation"]["summary"])
    lines.append("")
    for rec in results["remediation"]["recommendations"]:
        lines.append(f"- {rec}")

    lines.append("")
    lines.append(f"_Report generated by Bio Trust OS ({results['remediation']['agent_mode']} agent)._")

    return "\n".join(lines)


def build_report_json(df: pd.DataFrame, results: dict, dataset_name: str) -> str:
    tr = results["trust_result"]
    payload = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(results["columns"].get("features", [])),
        "trust_score": tr.trust_score,
        "gate_status": tr.gate_status,
        "sub_scores": tr.sub_scores,
        "completeness": {k: v for k, v in results["completeness"].items() if k != "notes" or True},
        "anomaly": {k: v for k, v in results["anomaly"].items() if k != "anomaly_scores"},
        "batch": {k: v for k, v in results["batch"].items() if k != "pca_df"},
        "balance": results["balance"],
        "remediation": results["remediation"],
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    return json.dumps(payload, indent=2, default=_default)


# ==========================================================================
# Sidebar: data input
# ==========================================================================

st.sidebar.markdown("## 🧬 Bio Trust OS")
st.sidebar.markdown(
    '<span class="small-note">AI-Powered Biological Data Trust & Readiness Platform</span>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

input_mode = st.sidebar.radio(
    "Data Source",
    ["Load Sample Benchmark Dataset", "Upload CSV / TSV"],
)

dataset_name = None
uploaded_df = None
error_msg = None

if input_mode == "Load Sample Benchmark Dataset":
    dataset_name = st.sidebar.selectbox("Benchmark Cohort", list(SAMPLE_DATASETS.keys()))
    if st.sidebar.button("🔄 Load / Regenerate Dataset", use_container_width=True):
        _load_sample.clear()
    try:
        uploaded_df = _load_sample(dataset_name)
    except Exception as e:  # pragma: no cover - defensive
        error_msg = f"Failed to generate sample dataset: {e}"
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or TSV file", type=["csv", "tsv", "txt"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
        try:
            sep = "\t" if uploaded_file.name.lower().endswith((".tsv", ".txt")) else ","
            uploaded_df = pd.read_csv(uploaded_file, sep=sep)
            if uploaded_df.shape[1] == 1 and sep == ",":
                # Retry as TSV in case delimiter was mis-detected
                uploaded_file.seek(0)
                uploaded_df = pd.read_csv(uploaded_file, sep="\t")
        except Exception as e:
            error_msg = f"Failed to parse uploaded file: {e}"

st.sidebar.divider()
with st.sidebar.expander("⚙️ Anomaly Detection Settings"):
    contamination = st.slider(
        "Expected outlier fraction (Isolation Forest contamination)",
        min_value=0.01, max_value=0.4, value=0.1, step=0.01,
    )

st.sidebar.divider()
st.sidebar.markdown(
    '<span class="small-note">Expected columns: SampleID, Batch/Plate, Condition, Center '
    '(optional metadata) + numeric feature columns.</span>',
    unsafe_allow_html=True,
)


# ==========================================================================
# Main header
# ==========================================================================

st.markdown("# 🧬 Bio Trust OS")
st.markdown(
    "##### AI-Powered Biological Data Trust & Readiness Platform "
    "&nbsp;|&nbsp; Anomaly Detection · Batch-Effect QC · Trust Scoring · Go/No-Go Gate"
)
st.divider()

if error_msg:
    st.error(error_msg)
    st.stop()

if uploaded_df is None:
    st.info("⬅️ Select a benchmark cohort or upload a CSV/TSV file from the sidebar to begin analysis.")
    st.stop()

if uploaded_df.empty:
    st.error("The loaded dataset is empty. Please provide a non-empty CSV/TSV file.")
    st.stop()

df = uploaded_df.copy()

with st.spinner("Running Bio Trust OS analysis engines..."):
    try:
        results = run_full_pipeline(df)
    except Exception as e:
        st.error(f"Analysis pipeline failed: {e}")
        st.stop()

tr = results["trust_result"]
columns = results["columns"]


# ==========================================================================
# Top section: Trust score, gate, key stats
# ==========================================================================

top_left, top_right = st.columns([1.3, 2])

with top_left:
    st.markdown('<div class="section-header">Biological Trust Score</div>', unsafe_allow_html=True)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tr.trust_score,
        number={"suffix": " / 100", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9CA3AF"},
            "bar": {"color": "#6366F1"},
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.25)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.25)"},
                {"range": [75, 100], "color": "rgba(16,185,129,0.25)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": tr.trust_score,
            },
        },
    ))
    gauge.update_layout(
        template=PLOTLY_TEMPLATE, height=260,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(render_gate_badge(tr.gate_status, tr.gate_color), unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="section-header">Key Stats</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples", f"{len(df):,}")
    m2.metric("Features Analyzed", f"{len(columns.get('features', [])):,}")
    m3.metric("Anomalous Samples", f"{results['anomaly'].get('flagged_count', 0):,}")
    m4.metric("Missing Cells", f"{results['completeness'].get('missing_pct') or 0.0:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Completeness", f"{results['completeness']['score']:.0f}")
    s2.metric("Batch Consistency", f"{results['batch']['score']:.0f}")
    s3.metric("Outlier Integrity", f"{results['anomaly']['score']:.0f}")
    s4.metric("Balance", f"{results['balance']['score']:.0f}")

st.divider()


# ==========================================================================
# Visualizations
# ==========================================================================

viz_col1, viz_col2 = st.columns(2)

# --- PCA scatter -----------------------------------------------------------
with viz_col1:
    st.markdown('<div class="section-header">PCA -- Batch / Condition Structure</div>', unsafe_allow_html=True)
    pca_df = results["batch"].get("pca_df")
    if pca_df is not None and {"PC1", "PC2"}.issubset(pca_df.columns):
        color_options = [c for c in ["Batch", "Condition"] if c in pca_df.columns]
        if color_options:
            color_by = st.radio("Color by", color_options, horizontal=True, key="pca_color")
        else:
            color_by = None

        view_mode = st.radio("View", ["2D", "3D"], horizontal=True, key="pca_view")

        if view_mode == "3D" and "PC3" in pca_df.columns:
            fig = px.scatter_3d(
                pca_df, x="PC1", y="PC2", z="PC3",
                color=color_by if color_by else None,
                template=PLOTLY_TEMPLATE,
                opacity=0.85,
            )
            fig.update_traces(marker=dict(size=5))
        else:
            fig = px.scatter(
                pca_df, x="PC1", y="PC2",
                color=color_by if color_by else None,
                template=PLOTLY_TEMPLATE,
                opacity=0.85,
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#0B0F19")))

        fig.update_layout(
            height=440, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PCA could not be computed (insufficient numeric features or samples).")

# --- Anomaly distribution ---------------------------------------------------
with viz_col2:
    st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
    anomaly_scores = results["anomaly"].get("anomaly_scores")
    if anomaly_scores is not None:
        anomaly_plot_df = pd.DataFrame({
            "Anomaly Score": anomaly_scores,
            "Sample": columns.get("sample_id") and df[columns["sample_id"]].values or df.index,
        })
        fig2 = px.histogram(
            anomaly_plot_df, x="Anomaly Score", nbins=30,
            template=PLOTLY_TEMPLATE, opacity=0.9,
            color_discrete_sequence=["#6366F1"],
        )
        fig2.add_vline(x=0, line_dash="dash", line_color="#EF4444",
                        annotation_text="Isolation Forest threshold", annotation_position="top")
        fig2.update_layout(
            height=440, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Isolation Forest decision score (lower = more anomalous)",
            yaxis_title="Sample count",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Anomaly scores unavailable (insufficient data for Isolation Forest).")

st.divider()

# --- Radar chart -------------------------------------------------------------
radar_col, remediation_col = st.columns([1, 1.4])

with radar_col:
    st.markdown('<div class="section-header">Quality Pillars -- Radar View</div>', unsafe_allow_html=True)
    categories = list(tr.sub_scores.keys())
    values = list(tr.sub_scores.values())
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#6366F1",
        fillcolor="rgba(99,102,241,0.35)",
        name="Score",
    ))
    fig3.update_layout(
        template=PLOTLY_TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        margin=dict(l=30, r=30, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- Remediation panel --------------------------------------------------------
with remediation_col:
    st.markdown('<div class="section-header">🩺 Actionable AI Remediation Panel</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="remediation-item"><b>Summary:</b> {results["remediation"]["summary"]}</div>',
        unsafe_allow_html=True,
    )
    for rec in results["remediation"]["recommendations"]:
        st.markdown(f'<div class="remediation-item">• {rec}</div>', unsafe_allow_html=True)

    with st.expander("View raw diagnostic notes by engine"):
        for engine_label, engine_result in [
            ("Completeness Engine", results["completeness"]),
            ("Anomaly & Outlier Engine", results["anomaly"]),
            ("Batch Effect Engine", results["batch"]),
            ("Cohort Balance Engine", results["balance"]),
        ]:
            st.markdown(f"**{engine_label}**")
            for note in engine_result.get("notes", []):
                st.markdown(f"- {note}")

st.divider()


# ==========================================================================
# Data preview & export
# ==========================================================================

with st.expander("📄 Preview Ingested Data"):
    st.dataframe(df.head(50), use_container_width=True)

st.markdown('<div class="section-header">📤 Export Executive QC Summary Report</div>', unsafe_allow_html=True)
exp1, exp2, exp3 = st.columns([1, 1, 2])

report_md = build_report_markdown(df, results, dataset_name or "Uploaded Dataset")
report_json = build_report_json(df, results, dataset_name or "Uploaded Dataset")

with exp1:
    st.download_button(
        "⬇️ Download Markdown Report",
        data=report_md,
        file_name="biotrust_qc_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
with exp2:
    st.download_button(
        "⬇️ Download JSON Report",
        data=report_json,
        file_name="biotrust_qc_report.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown(
    '<br><span class="small-note">Bio Trust OS -- deterministic QC engines '
    '(Isolation Forest, Z-score, PCA, ANOVA, silhouette) + offline rule-based remediation agent. '
    'No external API calls required.</span>',
    unsafe_allow_html=True,
)
