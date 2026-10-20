"""
ECG Signal Quality Analyzer & Noise Testing Tool
Biomedical Engineering Portfolio & Research Demonstration Application.

Designed for beginners, students, and engineers:
- Clean white and light-grey aesthetic with a single muted teal accent
- Obvious 5-step workflow:
  Step 1: Load ECG
  Step 2: View ECG
  Step 3: Filter Signal
  Step 4: Analyze Quality
  Step 5: Download Report
- Plain-English labels with scientific terms in brackets
- Clear explanations for every control, metric, and graph axis
- Publication-quality PDF report, CSV table, and JSON data exports
"""

import io
import os
import sys
import json
import pathlib
from typing import Dict, Any, Optional, Tuple, List

# Ensure 'src' is in sys.path so app runs reliably on Windows regardless of launcher
_src_dir = pathlib.Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core biomedical signal-processing modules
from ecg_analyzer.constants import (
    CLINICAL_DISCLAIMER,
    HEURISTIC_SCORE_DISCLAIMER,
    DEFAULT_FS_MITBIH,
    DEFAULT_FS_SYNTHETIC,
    DEFAULT_BANDPASS_LOWCUT,
    DEFAULT_BANDPASS_HIGHCUT,
    DEFAULT_FILTER_ORDER,
    DEFAULT_NOTCH_FREQ,
    DEFAULT_NOTCH_Q,
    DEFAULT_AAMI_TOLERANCE_SEC,
    DEFAULT_WINDOW_LENGTH_SEC,
)
from ecg_analyzer.data.validator import assess_signal_properties, SignalValidationError
from ecg_analyzer.data.loader import load_ecg_from_csv, load_physionet_record, ECGDataRecord
from ecg_analyzer.data.synthetic import (
    generate_clean_synthetic_ecg,
    add_baseline_wander,
    add_powerline_noise,
    add_muscle_noise,
    add_flatline_segment,
    add_spike_artifacts,
    add_clipping_saturation,
)
from ecg_analyzer.filters.bandpass import apply_bandpass_filter, FilterConfigurationError
from ecg_analyzer.filters.notch import apply_notch_filter
from ecg_analyzer.filters.frequency_analysis import (
    compute_power_spectral_density,
    compute_filter_frequency_response,
    estimate_snr_db,
)
from ecg_analyzer.detection.qrs_detector import detect_qrs_pan_tompkins
from ecg_analyzer.sqi.segmentation import analyze_signal_quality
from ecg_analyzer.sqi.status import (
    SignalQualityStatus,
    STATUS_METADATA,
    QUALITY_THRESHOLDS,
    assess_overall_signal_quality,
)
from ecg_analyzer.metrics.evaluation import evaluate_beat_detection
from ecg_analyzer.reporting.exporter import (
    create_segment_quality_dataframe,
    generate_analysis_report_json,
    generate_markdown_summary,
)
from ecg_analyzer.reporting.pdf_report import generate_pdf_report

APP_VERSION = "v2.1"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Signal Quality Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN, READABLE SCIENTIFIC RESEARCH-TOOL STYLING
# High contrast, dark readable text, clear navigation headings, one muted teal accent.
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#0d766e"        # Muted teal
ACCENT_LIGHT = "#f0fdfa"  # Light teal background tint
ACCENT_HOVER = "#0f766e"  # Slightly darker hover teal

CUSTOM_CSS = """
<style>
/* ── Reset & Core Typography ── */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 0.90rem;
    line-height: 1.5;
}

/* ── Clear, Prominent Header Strip ── */
.app-header {
    padding: 1rem 0 0.6rem 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1rem;
}
.app-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.25;
}
.app-subtitle {
    font-size: 0.90rem;
    color: #334155;
    margin-top: 4px;
    font-weight: 500;
    line-height: 1.4;
}
.app-version-badge {
    display: inline-block;
    background: #e2e8f0;
    color: #0f172a;
    font-family: 'Consolas', 'SF Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Getting Started Workflow Guide ── */
.getting-started-box {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0d766e;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
.getting-started-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.getting-started-steps {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 6px;
}
.getting-started-step {
    flex: 1;
    min-width: 140px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 7px 10px;
}
.step-num {
    font-size: 0.70rem;
    font-weight: 700;
    color: #0d766e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.step-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    margin: 2px 0;
}
.step-desc {
    font-size: 0.74rem;
    color: #475569;
    line-height: 1.35;
}

/* ── Clear, High-Contrast Navigation Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #f1f5f9;
    border-radius: 6px;
    padding: 5px;
    margin-bottom: 1.25rem;
    border: 1px solid #cbd5e1;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    color: #1e293b !important; /* Bold, dark text — NEVER faint! */
    border-radius: 4px;
    padding: 0 16px;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #e2e8f0;
    color: #0f172a !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #0d766e !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

/* ── Prominent Page Headings & Explanations ── */
.page-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
}
.page-explanation {
    font-size: 0.88rem;
    color: #334155;
    margin-bottom: 1.1rem;
    line-height: 1.45;
    background: #f1f5f9;
    padding: 8px 12px;
    border-radius: 4px;
    border-left: 3px solid #0d766e;
}
.section-heading {
    font-size: 0.98rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}
.section-caption {
    font-size: 0.80rem;
    color: #475569;
    margin-bottom: 0.6rem;
    line-height: 1.4;
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 10px 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}
div[data-testid="stMetric"] label {
    color: #1e293b !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.15rem !important;
    font-weight: 800;
    font-family: 'Consolas', 'SF Mono', monospace;
}

/* ── High-Contrast Buttons ── */
.stButton > button {
    background: #0d766e;
    color: #ffffff;
    border: 1px solid #0f766e;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.84rem;
    padding: 7px 16px;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: #0f766e;
    color: #ffffff;
    border-color: #115e59;
}
.stDownloadButton > button {
    background: #ffffff;
    border: 1.5px solid #0d766e;
    color: #0d766e;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.84rem;
    padding: 7px 16px;
}
.stDownloadButton > button:hover {
    background: #f0fdfa;
    border-color: #0f766e;
    color: #0f766e;
}

/* ── Clear Sidebar Headings and Labels ── */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #cbd5e1;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
}
.stSelectbox label, .stSlider label, .stCheckbox label, .stNumberInput label {
    font-size: 0.84rem !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* ── Provenance Banner ── */
.provenance-strip {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 1rem;
    align-items: center;
}
.prov-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 700;
}
.prov-badge-clinical {
    background: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
}
.prov-badge-synthetic {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.prov-badge-upload {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #1e293b;
}

/* ── Signal Quality Status Card ── */
.status-card {
    border-radius: 4px;
    padding: 14px 18px;
    margin: 0.6rem 0 1.1rem 0;
    border-left: 4.5px solid;
    background: #ffffff;
    border-top: 1px solid #cbd5e1;
    border-right: 1px solid #cbd5e1;
    border-bottom: 1px solid #cbd5e1;
}
.status-card-good   { border-left-color: #166534; background: #f0fdf4; }
.status-card-review { border-left-color: #b45309; background: #fffbeb; }
.status-card-poor   { border-left-color: #b91c1c; background: #fef2f2; }
.status-card-unable { border-left-color: #475569; background: #f8fafc; }

.status-badge {
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
}
.status-card-good   .status-badge { color: #166534; }
.status-card-review .status-badge { color: #b45309; }
.status-card-poor   .status-badge { color: #b91c1c; }
.status-card-unable .status-badge { color: #475569; }

.status-headline {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 5px;
}
.status-summary {
    font-size: 0.86rem;
    color: #1e293b;
    line-height: 1.45;
    margin-bottom: 10px;
}
.status-grid {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-top: 8px;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 8px;
}
.status-col {
    flex: 1;
    min-width: 220px;
}
.status-col-header {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
    margin-bottom: 4px;
}
.status-col ul {
    margin: 0;
    padding-left: 16px;
    font-size: 0.82rem;
    color: #1e293b;
    line-height: 1.45;
}

/* ── Educational Notice Strips ── */
.notice-strip {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3.5px solid #b45309;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 1rem;
    font-size: 0.80rem;
    color: #78350f;
    line-height: 1.45;
}

/* ── Control Explanations ── */
.control-help {
    font-size: 0.78rem;
    color: #475569;
    margin-top: -8px;
    margin-bottom: 12px;
    line-height: 1.35;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT HELPER — Safe keyword merging with ZERO duplicate arguments
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=11, color="#1e293b"),
    margin=dict(l=54, r=20, t=38, b=42),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        font=dict(size=10, color="#1e293b"),
    ),
)


def get_plot_layout(**overrides) -> dict:
    """
    Safely build a Plotly layout dictionary by merging defaults with overrides.
    Prevents duplicate keyword argument errors (e.g. margin=...).
    """
    layout = dict(_PLOT_LAYOUT_BASE)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in layout and isinstance(layout[key], dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout


def _axis_spec(title: str, **kwargs) -> dict:
    return {
        "showgrid": True,
        "gridcolor": "#f1f5f9",
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": "#cbd5e1",
        "zerolinewidth": 1,
        "title": dict(text=title, font=dict(size=10.5, color="#1e293b", family="inherit")),
        "tickfont": dict(size=9.5, color="#334155"),
        **kwargs,
    }


def plot_ecg_interactive(
    time: np.ndarray,
    voltage: np.ndarray,
    title: str = "ECG Waveform",
    color: str = ACCENT,
    r_peaks_sec: Optional[List[float]] = None,
    r_peaks_color: str = "#b91c1c",
    r_peaks_label: str = "Detected Heartbeats [R-Peaks]",
    reference_r_peaks_sec: Optional[List[float]] = None,
    show_ecg_grid: bool = True,
    height: int = 420,
) -> go.Figure:
    """Render a clean, publication-grade interactive ECG waveform with readable labels."""
    fig = go.Figure()

    if time is None or len(time) == 0 or voltage is None or len(voltage) == 0:
        fig.add_annotation(
            text="No waveform data available to display.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="#475569")
        )
        fig.update_layout(**get_plot_layout(height=height))
        return fig

    # Main voltage line
    fig.add_trace(go.Scatter(
        x=time, y=voltage,
        mode="lines",
        name="ECG Signal",
        line=dict(color=color, width=1.6),
        hovertemplate="Time: %{x:.3f} s<br>Voltage: %{y:.3f} mV<extra></extra>",
    ))

    # Reference beats (if physician annotations are present)
    if reference_r_peaks_sec is not None and len(reference_r_peaks_sec) > 0:
        ref_y = np.interp(reference_r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=reference_r_peaks_sec, y=ref_y,
            mode="markers",
            name="Reference Beats [Physician Ground Truth]",
            marker=dict(symbol="circle-open", size=9, color="#166534", line=dict(width=2)),
            hovertemplate="Reference Beat: %{x:.3f} s<extra></extra>",
        ))

    # Detected peaks
    if r_peaks_sec is not None and len(r_peaks_sec) > 0:
        det_y = np.interp(r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=r_peaks_sec, y=det_y,
            mode="markers",
            name=r_peaks_label,
            marker=dict(symbol="x", size=8.5, color=r_peaks_color, line=dict(width=1.8)),
            hovertemplate="Detected R-Peak: %{x:.3f} s<extra></extra>",
        ))

    grid_color = "#f1f5f9" if show_ecg_grid else "#fafafa"
    fig.update_layout(**get_plot_layout(
        title=dict(text=title, font=dict(size=11.5, color="#0f172a", family="inherit"), x=0),
        xaxis={
            **_axis_spec("Time (seconds) — shows when each heartbeat occurs"),
            "gridcolor": grid_color,
            "dtick": 0.2 if show_ecg_grid else None,
        },
        yaxis={
            **_axis_spec("Voltage (millivolts / mV) — measures the electrical potential of the heart"),
            "gridcolor": grid_color,
            "dtick": 0.5 if show_ecg_grid else None,
        },
        height=height,
        hovermode="x unified",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE & QUALITY STATUS COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_provenance_banner(record: "ECGDataRecord") -> None:
    """Render a clear, readable provenance banner explaining the active recording."""
    src = record.source_type
    if src == "physionet_wfdb":
        badge_cls = "prov-badge-clinical"
        badge_text = "PhysioNet Clinical Record (Real Patient Data with Physician Annotations)"
    elif "synthetic" in src:
        badge_cls = "prov-badge-synthetic"
        badge_text = "Simulated Signal (Synthetically Generated — Not Patient Data)"
    elif src == "csv":
        badge_cls = "prov-badge-upload"
        badge_text = "User Uploaded CSV File"
    else:
        badge_cls = "prov-badge-upload"
        badge_text = "Pre-Packaged Benchmark Dataset"

    html = f"""
<div class="provenance-strip">
  <span style="font-weight:700; color:#0f172a; font-size:0.86rem;">Active Recording:</span>
  <span style="font-family:'Consolas', monospace; font-weight:700; color:#0d766e; font-size:0.88rem;">{record.record_name}</span>
  <span style="color:#cbd5e1;">|</span>
  <span style="font-size:0.80rem; color:#334155;">Lead / Channel: <strong>{record.lead_name}</strong></span>
  <span style="color:#cbd5e1;">|</span>
  <span style="font-size:0.80rem; color:#334155;">Recording Speed: <strong>{record.fs:.0f} Hz</strong></span>
  <span style="color:#cbd5e1;">|</span>
  <span style="font-size:0.80rem; color:#334155;">Duration: <strong>{record.duration_sec:.2f} s</strong> ({len(record.signal):,} samples)</span>
  <span class="prov-badge {badge_cls}">{badge_text}</span>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_quality_status_card(assessment: Dict[str, Any]) -> None:
    """Render the 4-state Signal Quality status card with clear, plain-English advice."""
    st_val = assessment.get("status", "UNABLE TO ASSESS")
    st_label = assessment.get("label", "Unable to assess signal quality")
    summary = assessment.get("summary", "")
    issues = assessment.get("detected_issues", [])
    recs = assessment.get("recommendations", [])

    cls_map = {
        "GOOD": "status-card-good",
        "NEEDS REVIEW": "status-card-review",
        "POOR": "status-card-poor",
        "UNABLE TO ASSESS": "status-card-unable",
    }
    card_cls = cls_map.get(st_val, "status-card-unable")

    issues_html = "".join(f"<li>{i}</li>" for i in issues) if issues else "<li>No physical recording artifacts detected.</li>"
    recs_html = "".join(f"<li>{r}</li>" for r in recs) if recs else "<li>Signal is clean. Ready for automated beat detection and analysis.</li>"

    html = f"""
<div class="status-card {card_cls}">
  <div class="status-badge">Overall Rating: {st_val}</div>
  <div class="status-headline">{st_label}</div>
  <div class="status-summary"><strong>Why this status:</strong> {summary}</div>
  <div class="status-grid">
    <div class="status-col">
      <div class="status-col-header">Detected Physical Findings</div>
      <ul>{issues_html}</ul>
    </div>
    <div class="status-col">
      <div class="status-col-header">Recommended Next Steps</div>
      <ul>{recs_html}</ul>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── HEADER STRIP ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-title">
            ECG Signal Quality Analyzer
            <span class="app-version-badge">{APP_VERSION}</span>
          </div>
          <div class="app-subtitle">
            A beginner-friendly tool to inspect, filter, evaluate, and report electrocardiogram signals.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── GETTING STARTED GUIDE ────────────────────────────────────────────────
    st.markdown(
        """
<div class="getting-started-box">
  <div class="getting-started-title">Getting Started — Follow These 5 Simple Steps</div>
  <div class="getting-started-steps">
    <div class="getting-started-step">
      <div class="step-num">Step 1</div>
      <div class="step-name">Load ECG</div>
      <div class="step-desc">Pick a clean sample, upload a CSV file, or generate a simulated ECG.</div>
    </div>
    <div class="getting-started-step">
      <div class="step-num">Step 2</div>
      <div class="step-name">View ECG</div>
      <div class="step-desc">Inspect the waveform graph and understand how heartbeats are displayed.</div>
    </div>
    <div class="getting-started-step">
      <div class="step-num">Step 3</div>
      <div class="step-name">Filter Signal</div>
      <div class="step-desc">Reduce baseline drift and high-frequency noise using digital filters.</div>
    </div>
    <div class="getting-started-step">
      <div class="step-num">Step 4</div>
      <div class="step-name">Analyze Quality</div>
      <div class="step-desc">Check the quality rating (Good / Review / Poor) and review detected heartbeats.</div>
    </div>
    <div class="getting-started-step">
      <div class="step-num">Step 5</div>
      <div class="step-name">Download Report</div>
      <div class="step-desc">Download a complete PDF report, CSV metrics table, or JSON data.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── SIDEBAR CONTROLS ─────────────────────────────────────────────────────
    st.sidebar.markdown("### Digital Filter Settings")
    st.sidebar.markdown(
        '<div style="font-size:0.78rem; color:#475569; margin-bottom:8px;">'
        'Adjust these controls to clean up common recording artifacts.'
        '</div>',
        unsafe_allow_html=True,
    )

    enable_filter = st.sidebar.checkbox(
        "Apply Digital Filtering",
        value=True,
        help="Enables zero-phase bandpass and notch filtering.",
    )

    st.sidebar.markdown("**Baseline Drift Filter [High-Pass]**")
    filter_lowcut = st.sidebar.slider(
        "High-pass cutoff frequency (Hz)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Reduces slow baseline drift caused by breathing or movement. Recommended: 0.5 Hz.",
    )
    st.sidebar.markdown(
        '<div class="control-help">High-pass filter: reduces slow baseline drift caused by patient breathing.</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("**Muscle Noise Filter [Low-Pass]**")
    filter_highcut = st.sidebar.slider(
        "Low-pass cutoff frequency (Hz)",
        min_value=15.0, max_value=100.0, value=40.0, step=5.0,
        help="Reduces high-frequency noise such as muscle tremor (EMG). Recommended: 40 Hz.",
    )
    st.sidebar.markdown(
        '<div class="control-help">Low-pass filter: reduces high-frequency noise such as muscle tremor.</div>',
        unsafe_allow_html=True,
    )

    filter_order = st.sidebar.selectbox(
        "Filter Sharpness [Butterworth Order]",
        [1, 2, 3, 4, 5],
        index=2,
        help="Higher values create a steeper cutoff roll-off. Recommended: 3.",
    )

    st.sidebar.markdown("### Electrical Hum Filter [Notch]")
    enable_notch = st.sidebar.checkbox(
        "Enable Mains Notch Filter",
        value=True,
        help="Reduces electrical interference from AC power wiring.",
    )
    notch_freq = st.sidebar.selectbox(
        "Mains Electrical Frequency (Hz)",
        [50.0, 60.0],
        index=0,
        help="50 Hz in Europe / Asia / UK · 60 Hz in North America.",
    )
    st.sidebar.markdown(
        '<div class="control-help">Notch filter: reduces electrical interference at the selected frequency (50/60 Hz).</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:0.78rem; color:#475569; line-height:1.4;">'
        '<strong>Educational Software:</strong> Not a medical diagnostic device. Never use to guide clinical care.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── SESSION STATE INITIALIZATION ─────────────────────────────────────────
    if "current_record" not in st.session_state:
        sample_path = pathlib.Path.cwd() / "data" / "samples" / "sample_clean_ecg.csv"
        if sample_path.exists():
            st.session_state.current_record = load_ecg_from_csv(
                str(sample_path), fs=360.0, record_name="Clean Benchmark ECG (Lead II)"
            )
        else:
            t, ecg, r = generate_clean_synthetic_ecg(duration_sec=10.0, fs=360.0)
            st.session_state.current_record = ECGDataRecord(
                signal=ecg, fs=360.0, record_name="Default Simulated ECG",
                reference_r_peaks_sec=r, source_type="synthetic_sandbox",
            )

    current_record = st.session_state.current_record
    raw_signal = current_record.signal
    fs = current_record.fs
    filtered_signal = raw_signal.copy()
    b_filt, a_filt = None, None
    filter_warning = None

    # Apply digital filtering safely
    if enable_filter:
        try:
            nyq = 0.5 * fs
            if filter_highcut >= nyq:
                raise FilterConfigurationError(
                    f"Low-pass cutoff ({filter_highcut} Hz) must be strictly below Nyquist ({nyq:.1f} Hz)."
                )
            if filter_lowcut >= filter_highcut:
                raise FilterConfigurationError("High-pass cutoff must be strictly lower than low-pass cutoff.")
            filtered_signal, (b_filt, a_filt) = apply_bandpass_filter(
                filtered_signal, fs, lowcut=filter_lowcut, highcut=filter_highcut, order=filter_order
            )
            if enable_notch:
                if notch_freq < nyq:
                    filtered_signal, _ = apply_notch_filter(filtered_signal, fs, notch_freq=notch_freq)
                else:
                    filter_warning = f"Notch frequency ({notch_freq} Hz) exceeds Nyquist limit ({nyq:.1f} Hz)."
        except (FilterConfigurationError, Exception) as fce:
            filter_warning = str(fce)
            filtered_signal = raw_signal.copy()

    analysis_sig = filtered_signal if enable_filter else raw_signal
    snr_est = estimate_snr_db(raw_signal, filtered_signal) if enable_filter else None

    # SQI windowed analysis
    try:
        sqi_data = analyze_signal_quality(
            analysis_sig, fs=fs, window_length_sec=DEFAULT_WINDOW_LENGTH_SEC,
        )
    except Exception:
        sqi_data = None

    # 4-state overall quality classification
    status_assessment = assess_overall_signal_quality(
        raw_signal=raw_signal,
        fs=fs,
        sqi_results=sqi_data,
        snr_db=snr_est,
    )

    # ── THE 5 OBVIOUS WORKFLOW TABS ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Step 1: Load ECG",
        "Step 2: View ECG",
        "Step 3: Filter Signal",
        "Step 4: Analyze Quality",
        "Step 5: Download Report",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD ECG
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="page-title">Step 1: Load ECG [Data Selection]</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-explanation">'
            '<strong>What this page does:</strong> Choose which ECG recording you want to analyze. '
            'You can load ready-made sample recordings, upload your own CSV file, download from PhysioNet, or create a simulated signal.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_provenance_banner(current_record)

        st.markdown('<div class="section-heading">Quick Start — Load a Ready-Made Sample in 1 Click</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">'
            'Click any button below to instantly load a benchmark dataset and test the application.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("Load Clean ECG (Sample)", use_container_width=True):
                p = pathlib.Path.cwd() / "data" / "samples" / "sample_clean_ecg.csv"
                if p.exists():
                    st.session_state.current_record = load_ecg_from_csv(
                        str(p), fs=360.0, record_name="Clean Benchmark ECG (Lead II)"
                    )
                    st.success("Loaded clean ECG sample.")
                    st.rerun()
        with col_q2:
            if st.button("Load Noisy ECG (Sample)", use_container_width=True):
                p = pathlib.Path.cwd() / "data" / "samples" / "sample_noisy_ecg.csv"
                if p.exists():
                    st.session_state.current_record = load_ecg_from_csv(
                        str(p), fs=360.0, record_name="Noisy Benchmark ECG"
                    )
                    st.success("Loaded noisy ECG sample.")
                    st.rerun()
        with col_q3:
            if st.button("Load Artefact ECG (Sample)", use_container_width=True):
                p = pathlib.Path.cwd() / "data" / "samples" / "sample_artifacts_ecg.csv"
                if p.exists():
                    st.session_state.current_record = load_ecg_from_csv(
                        str(p), fs=360.0, record_name="Artefact Benchmark ECG"
                    )
                    st.success("Loaded artefact ECG sample.")
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="section-heading">Other Ingestion Options</div>', unsafe_allow_html=True)

        load_csv_tab, load_pn_tab, load_synth_tab = st.tabs([
            "Option A: Upload Your Own CSV File",
            "Option B: Download from PhysioNet MIT-BIH",
            "Option C: Simulated Signal Sandbox",
        ])

        # Option A: Upload CSV
        with load_csv_tab:
            st.markdown(
                '<div class="section-caption">'
                'Upload an ECG recording saved as a CSV or TXT file. The file should contain numbers representing voltage values.'
                '</div>',
                unsafe_allow_html=True,
            )
            up_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"], key="up_file_uploader")
            col_ufs, col_udel = st.columns(2)
            with col_ufs:
                up_fs = st.number_input(
                    "Recording Speed [Sampling Rate in Hz]",
                    min_value=20.0, max_value=5000.0, value=360.0, step=10.0,
                    help="How many data points were recorded per second. If unknown, 250 Hz or 360 Hz are typical.",
                )
            with col_udel:
                up_delim = st.selectbox(
                    "Column Separator [Delimiter]", ["Auto-detect", ",", ";", "\\t", "Space"], index=0
                )
            delim_val = (
                None if up_delim == "Auto-detect"
                else ("\t" if up_delim == "\\t" else (" " if up_delim == "Space" else up_delim))
            )
            if up_file is not None:
                if st.button("Parse and Load Uploaded File", key="btn_parse_up"):
                    try:
                        new_rec = load_ecg_from_csv(
                            up_file, fs=up_fs, delimiter=delim_val, record_name=up_file.name
                        )
                        st.session_state.current_record = new_rec
                        st.success(f"Successfully loaded '{up_file.name}' ({len(new_rec.signal):,} samples).")
                        st.rerun()
                    except SignalValidationError as sve:
                        st.error(f"Validation error: {sve}")
                    except Exception as err:
                        st.error(f"Could not read file: {err}")

        # Option B: PhysioNet MIT-BIH
        with load_pn_tab:
            st.markdown(
                '<div class="section-caption">'
                'Download real clinical patient recordings directly from the PhysioNet MIT-BIH Arrhythmia Database. '
                'These recordings include physician-verified beat annotations. Requires an active internet connection.'
                '</div>',
                unsafe_allow_html=True,
            )
            pn_col1, pn_col2 = st.columns(2)
            with pn_col1:
                pn_id = st.selectbox("MIT-BIH Record ID", ["100", "101", "103", "105", "117", "200", "203", "219"], index=0)
            with pn_col2:
                pn_dur = st.slider("Duration to Download (seconds)", 5.0, 60.0, 30.0, 5.0)

            if st.button(f"Download Record {pn_id} from PhysioNet", key="btn_fetch_physionet"):
                with st.spinner(f"Connecting to PhysioNet to download Record {pn_id}…"):
                    try:
                        pn_record = load_physionet_record(
                            record_id=pn_id,
                            database="mitdb",
                            max_duration_sec=pn_dur,
                            allow_synthetic_fallback=False,
                        )
                        st.session_state.current_record = pn_record
                        st.success(f"Successfully downloaded Record {pn_id} with {len(pn_record.reference_r_peaks_sec)} verified beats.")
                        st.rerun()
                    except Exception as err:
                        st.error(f"PhysioNet connection notice: {err}. Please check internet access or use a sample dataset.")

        # Option C: Synthetic Sandbox
        with load_synth_tab:
            st.markdown(
                '<div class="notice-strip">'
                '<strong>Simulated Data:</strong> This sandbox mathematically creates artificial ECG waves with controllable noise. '
                'It is <strong>not</strong> real patient data and should never be described as such.'
                '</div>',
                unsafe_allow_html=True,
            )
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                synth_hr = st.slider("Heart Rate (Beats Per Minute / BPM)", 40.0, 160.0, 72.0, 2.0)
            with sc2:
                synth_dur = st.slider("Duration (seconds)", 5.0, 60.0, 15.0, 5.0)
            with sc3:
                synth_fs = st.selectbox("Sampling Rate (Hz)", [250.0, 360.0, 500.0, 1000.0], index=1)

            st.markdown("**Add Injected Noise Types:**")
            ai1, ai2, ai3 = st.columns(3)
            with ai1:
                inj_drift = st.checkbox("Baseline Drift [Breathing / Movement]", value=True)
                drift_amp = st.slider("Drift Height (mV)", 0.05, 1.0, 0.35, 0.05) if inj_drift else 0.0
                inj_spikes = st.checkbox("Abrupt Voltage Spikes [Sudden Motion]", value=False)
            with ai2:
                inj_mains = st.checkbox("Electrical Hum [50 Hz or 60 Hz Mains]", value=True)
                mains_f = st.selectbox("Hum Frequency (Hz)", [50.0, 60.0], index=0) if inj_mains else 50.0
                mains_a = st.slider("Hum Strength (mV)", 0.02, 0.5, 0.15, 0.01) if inj_mains else 0.0
                inj_clip = st.checkbox("Sensor Clipping [Amplifier Rail Saturation]", value=False)
            with ai3:
                inj_emg = st.checkbox("Muscle Noise [EMG Tremor]", value=False)
                emg_a = st.slider("Muscle Noise Strength (mV)", 0.02, 0.3, 0.08, 0.01) if inj_emg else 0.0
                inj_flat = st.checkbox("Sensor Disconnection [Flatline Dropout]", value=False)
                flat_d = st.slider("Flatline Duration (s)", 0.5, 5.0, 2.0, 0.5) if inj_flat else 0.0

            if st.button("Generate Simulated Signal", key="btn_create_synthetic"):
                t_syn, ecg_syn, r_syn = generate_clean_synthetic_ecg(
                    duration_sec=synth_dur, fs=synth_fs, heart_rate_bpm=synth_hr, random_seed=42,
                )
                if inj_drift: ecg_syn = add_baseline_wander(ecg_syn, synth_fs, 0.25, drift_amp)
                if inj_mains: ecg_syn = add_powerline_noise(ecg_syn, synth_fs, mains_f, mains_a)
                if inj_emg:   ecg_syn = add_muscle_noise(ecg_syn, synth_fs, emg_a)
                if inj_flat:  ecg_syn = add_flatline_segment(ecg_syn, synth_fs, synth_dur * 0.4, flat_d)
                if inj_spikes: ecg_syn = add_spike_artifacts(ecg_syn, synth_fs, [synth_dur * 0.25, synth_dur * 0.75], 4.0)
                if inj_clip:  ecg_syn = add_clipping_saturation(ecg_syn, -0.7, 0.7)

                st.session_state.current_record = ECGDataRecord(
                    signal=ecg_syn, fs=synth_fs,
                    lead_name="Simulated Lead II",
                    record_name=f"Simulated ECG ({synth_hr:.0f} BPM, {synth_dur:.0f} s)",
                    reference_r_peaks_sec=r_syn,
                    annotation_labels=["N"] * len(r_syn),
                    source_type="synthetic_sandbox",
                )
                st.success("Generated simulated ECG recording.")
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: VIEW ECG
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="page-title">Step 2: View ECG [Waveform Inspection]</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-explanation">'
            '<strong>What this page does:</strong> Inspect the electrical waveform of the heart. '
            'The horizontal axis measures time in seconds, and the vertical axis measures electrical voltage in millivolts (mV).'
            '</div>',
            unsafe_allow_html=True,
        )

        render_provenance_banner(current_record)

        # How to read the graph callout
        st.markdown(
            """
<div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:4px; padding:10px 14px; margin-bottom:12px; font-size:0.84rem; color:#1e293b; line-height:1.45;">
  <strong>How to Read This ECG Graph:</strong>
  <ul style="margin:4px 0 0 16px; padding:0;">
    <li><strong>Horizontal Axis (Time in seconds):</strong> Moves from left to right. Shows when each heartbeat happens.</li>
    <li><strong>Vertical Axis (Voltage in millivolts / mV):</strong> Shows electrical amplitude. The tall upward spikes are <strong>R-peaks</strong>, representing ventricular contractions.</li>
    <li><strong>Zoom and Pan:</strong> Click and drag on the graph to zoom into any heartbeat. Double-click to reset. Scroll to zoom in and out.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

        # Inspection Window Controls
        total_dur = current_record.duration_sec
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            show_grid = st.checkbox("Show Standard ECG Grid [0.04 s / 0.20 s]", value=True)
        with col_w2:
            time_window = st.slider(
                "Select Time Range to Inspect (seconds)",
                min_value=0.0,
                max_value=total_dur,
                value=(0.0, min(10.0, total_dur)),
                step=0.5,
            )

        start_t, end_t = time_window
        idx0 = max(0, int(round(start_t * fs)))
        idx1 = min(len(raw_signal), int(round(end_t * fs)))

        # Large readable plot
        fig_view = plot_ecg_interactive(
            current_record.time_vector[idx0:idx1],
            raw_signal[idx0:idx1],
            title=f"ECG Voltage Trace: {start_t:.1f} s to {end_t:.1f} s — {current_record.record_name}",
            show_ecg_grid=show_grid,
            height=440,
        )
        st.plotly_chart(fig_view, use_container_width=True)

        # Useful signal statistics with units
        st.markdown('<div class="section-heading">Key Signal Measurements for This Window</div>', unsafe_allow_html=True)
        sub_sig = raw_signal[idx0:idx1]
        valid_sub = sub_sig[np.isfinite(sub_sig)]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Voltage Range [Peak-to-Peak]", f"{np.ptp(valid_sub):.3f} mV" if len(valid_sub) > 0 else "N/A")
        m2.metric("Average Voltage [Mean]", f"{np.mean(valid_sub):.3f} mV" if len(valid_sub) > 0 else "N/A")
        m3.metric("Voltage Variation [Std Dev]", f"{np.std(valid_sub):.3f} mV" if len(valid_sub) > 0 else "N/A")
        m4.metric("Lowest Voltage [Min]", f"{np.min(valid_sub):.3f} mV" if len(valid_sub) > 0 else "N/A")
        m5.metric("Highest Voltage [Max]", f"{np.max(valid_sub):.3f} mV" if len(valid_sub) > 0 else "N/A")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: FILTER SIGNAL
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="page-title">Step 3: Filter Signal [Digital Preprocessing]</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-explanation">'
            '<strong>What this page does:</strong> Clean up common recording noise using digital filters. '
            'Compare the original raw ECG with the filtered ECG side-by-side to see how filtering affects the waveform.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_provenance_banner(current_record)

        if filter_warning:
            st.error(f"Filter Setting Warning: {filter_warning}")

        if not enable_filter:
            st.info(
                "Digital filtering is currently turned OFF in the left sidebar. "
                "Turn ON 'Apply Digital Filtering' in the sidebar to clean the signal."
            )
        else:
            # Active filter summary
            st.markdown(
                f"""
<div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:4px; padding:10px 14px; margin-bottom:12px; font-size:0.84rem; color:#1e293b;">
  <strong>Active Filter Settings:</strong>
  <ul style="margin:4px 0 0 16px; padding:0;">
    <li><strong>Baseline Drift Filter [High-Pass]:</strong> {filter_lowcut:.2f} Hz — reduces slow baseline drift caused by breathing.</li>
    <li><strong>Muscle Noise Filter [Low-Pass]:</strong> {filter_highcut:.1f} Hz — reduces high-frequency muscle tremor.</li>
    <li><strong>Filter Order:</strong> {filter_order} (applied forward and backward with zero phase delay).</li>
    <li><strong>Electrical Hum Filter [Mains Notch]:</strong> {"Enabled at " + str(notch_freq) + " Hz" if enable_notch else "Disabled"} — reduces powerline hum.</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

            # Important scientific caveat
            st.markdown(
                """
<div class="notice-strip">
  <strong>Important Note on Filtering:</strong> Digital filters reduce specific types of noise, but they do <strong>not</strong>
  remove all interference or guarantee improved signal quality. Aggressive filtering can also remove real heart signals or
  distort important diagnostic features (such as ST-segment elevation or depression). Always compare the raw and filtered signals.
</div>
""",
                unsafe_allow_html=True,
            )

            # Comparison Mode Selector
            comp_mode = st.radio(
                "Waveform Comparison View",
                ["Overlaid Comparison (Raw vs Filtered)", "Separate Panels (Raw / Filtered / Removed Noise)"],
                horizontal=True,
            )

            dur_c = min(8.0, current_record.duration_sec)
            n_c = int(round(dur_c * fs))
            t_c = current_record.time_vector[:n_c]
            raw_c = raw_signal[:n_c]
            flt_c = filtered_signal[:n_c]
            res_c = raw_c - flt_c

            if comp_mode == "Overlaid Comparison (Raw vs Filtered)":
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=t_c, y=raw_c, mode="lines", name="Original Raw ECG",
                    line=dict(color="#94a3b8", width=1.4),
                ))
                fig_comp.add_trace(go.Scatter(
                    x=t_c, y=flt_c, mode="lines", name="Filtered ECG (Zero-Phase)",
                    line=dict(color=ACCENT, width=1.8),
                ))
                fig_comp.update_layout(**get_plot_layout(
                    title=dict(text=f"Before/After Filter Comparison — First {dur_c:.1f} s", font=dict(size=11.5, color="#0f172a"), x=0),
                    xaxis=_axis_spec("Time (seconds)"),
                    yaxis=_axis_spec("Voltage (mV)"),
                    height=380,
                ))
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                fig_sub = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["1. Original Raw Input", "2. Filtered Output (Zero-Phase)", "3. Removed Noise (Raw − Filtered)"],
                    vertical_spacing=0.09,
                )
                fig_sub.add_trace(go.Scatter(x=t_c, y=raw_c, line=dict(color="#64748b", width=1.3)), row=1, col=1)
                fig_sub.add_trace(go.Scatter(x=t_c, y=flt_c, line=dict(color=ACCENT, width=1.6)), row=2, col=1)
                fig_sub.add_trace(go.Scatter(x=t_c, y=res_c, line=dict(color="#b91c1c", width=1.2)), row=3, col=1)
                fig_sub.update_layout(**get_plot_layout(height=500, showlegend=False))
                fig_sub.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig_sub.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig_sub, use_container_width=True)

            # Signal Clarity (SNR) & Frequency Spectrum
            col_snr, col_psd = st.columns(2)
            with col_snr:
                st.markdown('<div class="section-heading">Signal Clarity [Estimated SNR]</div>', unsafe_allow_html=True)
                snr_display = f"{snr_est:.1f} dB" if snr_est is not None else "Not Evaluated"
                st.metric("Signal-to-Noise Ratio (SNR)", snr_display)
                st.markdown(
                    '<div class="section-caption">'
                    'Signal-to-Noise Ratio (SNR) compares the power of the heart signal to the removed noise. '
                    'Higher values (above 15 dB) indicate cleaner signals.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col_psd:
                st.markdown('<div class="section-heading">Frequency Spectrum [Power Spectral Density]</div>', unsafe_allow_html=True)
                freqs_r, psd_r = compute_power_spectral_density(raw_signal, fs)
                freqs_f, psd_f = compute_power_spectral_density(filtered_signal, fs)
                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(x=freqs_r, y=psd_r, mode="lines", name="Raw Spectrum", line=dict(color="#94a3b8", width=1.3)))
                fig_psd.add_trace(go.Scatter(x=freqs_f, y=psd_f, mode="lines", name="Filtered Spectrum", line=dict(color=ACCENT, width=1.8)))
                fig_psd.update_layout(**get_plot_layout(
                    xaxis={**_axis_spec("Frequency (Hz)"), "range": [0, min(80, fs / 2)]},
                    yaxis={**_axis_spec("Power Density (mV²/Hz)"), "type": "log"},
                    height=280,
                    title=dict(text="Frequency Content Comparison (Welch Method)", font=dict(size=10.5, color="#0f172a"), x=0),
                ))
                st.plotly_chart(fig_psd, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: ANALYZE QUALITY
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="page-title">Step 4: Analyze Quality [Signal Quality & Heartbeats]</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-explanation">'
            '<strong>What this page does:</strong> Check whether the ECG recording is clean enough for reliable automated analysis. '
            'Review the overall rating, detected noise issues, calculated quality indices, and detected heartbeats.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_provenance_banner(current_record)

        # 4-State Quality Verdict Card
        st.markdown('<div class="section-heading">Overall Signal Quality Assessment</div>', unsafe_allow_html=True)
        render_quality_status_card(status_assessment)

        # Clear Non-Diagnostic Disclaimer
        st.markdown(
            """
<div style="background:#f1f5f9; border-left:3px solid #475569; padding:8px 12px; border-radius:4px; font-size:0.80rem; color:#1e293b; margin-bottom:14px; line-height:1.45;">
  <strong>Important Distinction — Signal Quality vs Medical Diagnosis:</strong>
  Signal quality only measures how clean the electrical recording is from an engineering perspective.
  A <strong>"Good"</strong> rating means the electrical signal is clear and free of noise; it does <strong>NOT</strong> mean the patient's
  heart rhythm is medically normal or healthy. For example, a recording from a patient with a dangerous cardiac arrhythmia
  can still receive a "Good" signal quality rating if the electrodes were properly attached. This application does not diagnose medical conditions.
</div>
""",
            unsafe_allow_html=True,
        )

        # Key Quality Metrics Grid
        if sqi_data:
            ov_score = sqi_data.get("overall_heuristic_sqi", 0.0)
            breakdown = sqi_data.get("classification_breakdown", {})
            supp = status_assessment.get("supporting_metrics", {})

            st.markdown('<div class="section-heading">Calculated Signal Quality Metrics (With Units)</div>', unsafe_allow_html=True)
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("Overall Quality Score", f"{ov_score:.1f} / 100")
            q2.metric("Baseline Drift [bas_sqi]", f"{supp.get('bas_sqi', 0.0) * 100:.1f} %")
            q3.metric("Muscle Noise [hf_sqi]", f"{supp.get('hf_sqi', 0.0) * 100:.1f} %")
            q4.metric("Electrical Hum [pl_sqi]", f"{supp.get('pl_sqi', 0.0) * 100:.1f} %")
            q5.metric("Peak Sharpness [Kurtosis]", f"{supp.get('kurtosis_sqi', 0.0):.2f}")

            # Metric Explanations Table
            st.markdown(
                """
<div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:4px; padding:10px 14px; margin-top:8px; margin-bottom:14px; font-size:0.80rem; color:#1e293b; line-height:1.45;">
  <strong>What Each Metric Means:</strong>
  <ul style="margin:4px 0 0 16px; padding:0;">
    <li><strong>Overall Quality Score (0–100):</strong> Combines all noise measurements into a single score. Scores above 80 are Good; scores below 55 are Poor.</li>
    <li><strong>Baseline Drift [bas_sqi]:</strong> Percentage of total energy in the 0–0.5 Hz frequency band (from breathing and chest motion). Expected: &lt; 25%.</li>
    <li><strong>Muscle Noise [hf_sqi]:</strong> Percentage of total energy above 40 Hz (from muscle contraction and tremor). Expected: &lt; 20%.</li>
    <li><strong>Electrical Hum [pl_sqi]:</strong> Percentage of energy around 50 Hz or 60 Hz mains power. Expected: &lt; 15%.</li>
    <li><strong>Peak Sharpness [Kurtosis kSQI]:</strong> Measures how sharply heartbeats stand out against the baseline. Clean ECG &gt; 3.5; flatlines or Gaussian noise &lt; 3.0.</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

            # Window-by-window timeline
            st.markdown('<div class="section-heading">Timeline of 3-Second Recording Segments</div>', unsafe_allow_html=True)
            tl_colors = {
                "Optimal": "#166534",
                "Acceptable": "#b45309",
                "Noisy": "#92400e",
                "Unreliable / Leads-Off": "#b91c1c",
            }
            fig_tl = go.Figure()
            for seg in sqi_data["segments"]:
                fig_tl.add_trace(go.Bar(
                    x=[seg["duration_sec"]],
                    y=["Quality"],
                    base=seg["start_time_sec"],
                    orientation="h",
                    name=seg["classification"],
                    marker=dict(color=tl_colors.get(seg["classification"], "#64748b")),
                    hovertemplate=(
                        f"Segment: {seg['start_time_sec']:.1f}–{seg['end_time_sec']:.1f} s<br>"
                        f"Rating: {seg['classification']}<br>"
                        f"Score: {seg['heuristic_score']:.1f}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_tl.update_layout(**get_plot_layout(
                barmode="stack",
                xaxis={**_axis_spec("Time across recording (seconds)"), "range": [0, current_record.duration_sec]},
                yaxis=dict(showticklabels=False),
                height=75,
                margin=dict(l=10, r=10, t=10, b=28),
            ))
            st.plotly_chart(fig_tl, use_container_width=True)

        # Heartbeat Detection Section
        st.markdown('<div class="section-heading">Heartbeat Detection [Pan-Tompkins Algorithm]</div>', unsafe_allow_html=True)
        with st.spinner("Detecting heartbeats…"):
            det_samples, det_sec, _ = detect_qrs_pan_tompkins(analysis_sig, fs=fs)

        hr_est = (len(det_sec) / current_record.duration_sec * 60.0 if current_record.duration_sec > 0 else 0.0)

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Detected Heartbeats", f"{len(det_sec)} beats")
        bc2.metric("Estimated Heart Rate", f"{hr_est:.1f} BPM")
        bc3.metric("Detection Algorithm", "Pan-Tompkins QRS Detector")

        disp_beats_dur = min(10.0, current_record.duration_sec)
        n_beats_disp = int(round(disp_beats_dur * fs))
        fig_beats = plot_ecg_interactive(
            current_record.time_vector[:n_beats_disp],
            analysis_sig[:n_beats_disp],
            title=f"Detected Heartbeats (Red X) — First {disp_beats_dur:.1f} s",
            r_peaks_sec=[t for t in det_sec if t <= disp_beats_dur],
            reference_r_peaks_sec=(
                [t for t in current_record.reference_r_peaks_sec if t <= disp_beats_dur]
                if current_record.has_reference_annotations else None
            ),
            height=360,
        )
        st.plotly_chart(fig_beats, use_container_width=True)

        # Beat Evaluation (ANSI/AAMI EC57)
        st.markdown('<div class="section-heading">Detector Accuracy Benchmark [ANSI/AAMI EC57]</div>', unsafe_allow_html=True)
        beat_metrics = None
        if current_record.has_reference_annotations:
            tol_ms = st.slider(
                "Matching Tolerance Window (milliseconds / ms)",
                min_value=50.0, max_value=300.0, value=150.0, step=10.0,
                help="The ANSI/AAMI EC57 standard defines a 150 ms tolerance window for matching detected beats to reference labels.",
            )
            beat_metrics = evaluate_beat_detection(
                current_record.reference_r_peaks_sec,
                det_sec,
                tolerance_sec=tol_ms / 1000.0,
                max_duration_sec=current_record.duration_sec,
                annotation_source=f"{current_record.source_type} ({current_record.record_name})",
            )

            ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
            ec1.metric("True Detections [TP]", beat_metrics["true_positives"])
            ec2.metric("Missed Beats [FN]", beat_metrics["false_negatives"])
            ec3.metric("False Alarms [FP]", beat_metrics["false_positives"])

            se_str = f"{beat_metrics['sensitivity_percentage']}%" if beat_metrics["sensitivity_percentage"] is not None else "N/A"
            ppv_str = f"{beat_metrics['ppv_percentage']}%" if beat_metrics["ppv_percentage"] is not None else "N/A"
            f1_str = f"{beat_metrics['f1_percentage']}%" if beat_metrics["f1_percentage"] is not None else "N/A"

            ec4.metric("Detection Rate [Sensitivity]", se_str)
            ec5.metric("Precision [PPV]", ppv_str)
            ec6.metric("Overall Accuracy [F1]", f1_str)

            st.markdown(
                """
<div style="font-size:0.80rem; color:#475569; margin-top:4px;">
  <strong>Sensitivity:</strong> Proportion of real heartbeats detected. Higher is better (few missed beats).<br/>
  <strong>Precision (PPV):</strong> Proportion of detector peaks that were genuine heartbeats. Higher is better (few false alarms).
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "**Benchmark Accuracy: Not Available for this Recording**\n\n"
                "Calculating sensitivity and precision strictly requires physician-verified reference beat annotations "
                "(such as PhysioNet MIT-BIH recordings). Because this recording has no reference annotations, "
                "accuracy metrics are honestly marked as Not Available rather than estimated or fabricated."
            )

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: DOWNLOAD REPORT
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="page-title">Step 5: Download Report [Export Analysis]</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-explanation">'
            '<strong>What this page does:</strong> Download your complete analysis as a PDF document, '
            'a CSV spreadsheet table, or a structured JSON data file for your records or portfolio.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_provenance_banner(current_record)

        filter_meta = {
            "lowcut": filter_lowcut if enable_filter else None,
            "highcut": filter_highcut if enable_filter else None,
            "order": filter_order if enable_filter else None,
            "notch_enabled": enable_notch if enable_filter else False,
            "notch_freq": notch_freq if (enable_filter and enable_notch) else None,
            "notch_q": DEFAULT_NOTCH_Q,
        }
        record_meta_dict = {
            "record_name": current_record.record_name,
            "lead": current_record.lead_name,
            "fs": fs,
            "duration_sec": current_record.duration_sec,
            "source_type": current_record.source_type,
            "sample_count": len(raw_signal),
        }

        # Safe beat metrics
        bm_ref = None
        if current_record.has_reference_annotations:
            try:
                bm_ref = beat_metrics
            except NameError:
                bm_ref = None

        # Safe detected peaks
        try:
            peaks_for_pdf = det_sec
        except NameError:
            _, peaks_for_pdf, _ = detect_qrs_pan_tompkins(analysis_sig, fs=fs)

        # Generate PDF in memory
        with st.spinner("Generating PDF report…"):
            try:
                pdf_bytes = generate_pdf_report(
                    record_meta=record_meta_dict,
                    status_assessment=status_assessment,
                    filter_settings=filter_meta,
                    sqi_results=sqi_data,
                    raw_signal=raw_signal,
                    filtered_signal=filtered_signal if enable_filter else None,
                    time_vector=current_record.time_vector,
                    detected_r_peaks_sec=peaks_for_pdf,
                    beat_metrics=bm_ref,
                    app_version=APP_VERSION,
                )
            except Exception as pdf_err:
                pdf_bytes = None
                st.error(f"PDF generation error: {pdf_err}")

        # JSON generation
        json_str = generate_analysis_report_json(
            record_info=record_meta_dict,
            filter_params=filter_meta,
            sqi_results=sqi_data,
            beat_metrics=bm_ref,
            status_assessment=status_assessment,
        )

        # CSV generation
        df_exp = create_segment_quality_dataframe(sqi_data)
        csv_buf = io.StringIO()
        df_exp.to_csv(csv_buf, index=False)
        csv_str = csv_buf.getvalue()

        # Markdown summary
        md_summary = generate_markdown_summary(
            record_name=current_record.record_name,
            lead_name=current_record.lead_name,
            fs=fs,
            duration_sec=current_record.duration_sec,
            sqi_results=sqi_data,
            beat_metrics=bm_ref,
            filter_settings=filter_meta,
            status_assessment=status_assessment,
        )

        safe_name = current_record.record_name.replace(" ", "_").replace("/", "-").replace(":", "-")

        st.markdown('<div class="section-heading">Download Your Analysis</div>', unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            if pdf_bytes:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"ecg_report_{safe_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Multi-page vector PDF with waveform charts, metrics, filter settings, and limitations.",
                )
            else:
                st.button("PDF Unavailable", disabled=True, use_container_width=True)

        with col_d2:
            st.download_button(
                label="Download CSV Table",
                data=csv_str,
                file_name=f"ecg_segments_{safe_name}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Table of all 3-second recording segments with noise scores.",
            )

        with col_d3:
            st.download_button(
                label="Download JSON Data",
                data=json_str,
                file_name=f"ecg_analysis_{safe_name}.json",
                mime="application/json",
                use_container_width=True,
                help="Machine-readable data file containing all results and parameters.",
            )

        st.markdown("---")
        st.markdown('<div class="section-heading">Report Text Summary Preview</div>', unsafe_allow_html=True)
        st.markdown(md_summary)


if __name__ == "__main__":
    main()
