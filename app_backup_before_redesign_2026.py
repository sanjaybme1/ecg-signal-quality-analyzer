"""
ECG Signal Quality Analyzer — Biomedical Engineering Portfolio Application.
Minimal scientific research-tool design. All signal-processing logic unchanged.
"""

import io
import os
import sys
import json
import pathlib
from typing import Dict, Any, Optional, Tuple, List

# Ensure 'src' is in sys.path so app runs out-of-the-box on Windows regardless of how streamlit is launched
_src_dir = pathlib.Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core library modules
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

APP_VERSION = "v2.0"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Signal Quality Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS & CSS — Minimal scientific research-tool aesthetic
#   Background:  warm white / very light grey   #f7f7f5
#   Surface:     white                          #ffffff
#   Text:        charcoal                       #1e2329
#   Secondary:   medium grey                    #6b7280
#   Border:      light grey                     #e5e7eb
#   Accent:      muted teal                     #2d7d72
#   Good:        restrained green               #2e7d32
#   Review:      amber                          #92400e
#   Poor:        muted red                      #991b1b
#   Unable:      neutral grey                   #4b5563
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#2d7d72"
ACCENT_LIGHT = "#e0f2f0"

CUSTOM_CSS = """
<style>
/* ── Reset Streamlit defaults ── */
.stApp {
    background-color: #f7f7f5;
    color: #1e2329;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e2329 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}
section[data-testid="stSidebar"] hr {
    border-color: #e5e7eb;
    margin: 0.8rem 0;
}

/* ── Header strip ── */
.app-header {
    padding: 1.25rem 0 0.75rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.25rem;
}
.app-title {
    font-size: 1.30rem;
    font-weight: 700;
    color: #1e2329;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.25;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 2px;
    line-height: 1.4;
}
.app-version {
    font-size: 0.72rem;
    color: #9ca3af;
    font-family: 'Consolas', monospace;
}

/* ── Educational notice banner ── */
.notice-banner {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #d97706;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 1rem;
    font-size: 0.80rem;
    color: #78350f;
    line-height: 1.45;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px 14px;
}
div[data-testid="stMetric"] label {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1e2329 !important;
    font-size: 1.15rem !important;
    font-weight: 700;
    font-family: 'Consolas', 'Courier New', monospace;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: transparent;
    border-bottom: 1px solid #e5e7eb;
    padding: 0;
    margin-bottom: 1.25rem;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    color: #6b7280;
    border-radius: 0;
    padding: 0 14px;
    font-weight: 500;
    font-size: 0.82rem;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #2d7d72 !important;
    border-bottom: 2px solid #2d7d72 !important;
    font-weight: 600 !important;
}

/* ── Section headings ── */
.section-label {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}
h3.panel-heading {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #1e2329 !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Status cards ── */
.sqstatus {
    border-radius: 6px;
    padding: 14px 18px;
    margin: 0.75rem 0 1.25rem 0;
    border-left: 4px solid;
}
.sqstatus-good   { background:#f0fdf4; border-color:#2e7d32; }
.sqstatus-review { background:#fffbeb; border-color:#b45309; }
.sqstatus-poor   { background:#fef2f2; border-color:#991b1b; }
.sqstatus-unable { background:#f9fafb; border-color:#6b7280; }

.sqstatus-badge {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.sqstatus-good   .sqstatus-badge { color:#2e7d32; }
.sqstatus-review .sqstatus-badge { color:#b45309; }
.sqstatus-poor   .sqstatus-badge { color:#991b1b; }
.sqstatus-unable .sqstatus-badge { color:#4b5563; }

.sqstatus-label {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e2329;
    margin-bottom: 6px;
}
.sqstatus-summary {
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.5;
    margin-bottom: 10px;
}
.sqstatus-cols {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.sqstatus-col { flex: 1; min-width: 220px; }
.sqstatus-col-heading {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 4px;
}
.sqstatus-col ul {
    margin: 0; padding-left: 16px;
    font-size: 0.80rem;
    color: #374151;
    line-height: 1.5;
}

/* ── Workflow step tiles ── */
.wf-grid {
    display: flex;
    gap: 8px;
    flex-wrap: nowrap;
    margin-bottom: 1.25rem;
}
.wf-tile {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 10px 12px;
    min-width: 0;
}
.wf-num {
    font-size: 0.65rem;
    font-weight: 700;
    color: #2d7d72;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.wf-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1e2329;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.wf-desc {
    font-size: 0.72rem;
    color: #6b7280;
    line-height: 1.35;
}

/* ── Info / help text ── */
.help-text {
    font-size: 0.78rem;
    color: #6b7280;
    line-height: 1.5;
    margin-top: 0.25rem;
}

/* ── Recording provenance pill ── */
.rec-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 0.78rem;
    color: #374151;
    margin-bottom: 1rem;
}
.rec-pill-label { color: #6b7280; font-weight: 600; }
.rec-pill-value { font-family: 'Consolas', monospace; color: #1e2329; }

/* ── Divider ── */
hr.section-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* ── Stray Streamlit elements ── */
.stAlert { border-radius: 5px; font-size: 0.82rem; }
.stButton > button {
    background: #2d7d72;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.83rem;
    padding: 7px 16px;
}
.stButton > button:hover { background: #245f57; }
.stDownloadButton > button {
    background: #ffffff;
    border: 1px solid #2d7d72;
    color: #2d7d72;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.83rem;
}
.stDownloadButton > button:hover { background: #e0f2f0; }

/* Prevent sidebar labels from being tiny */
.stSelectbox label, .stSlider label, .stCheckbox label, .stNumberInput label {
    font-size: 0.80rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Plotly layout base — light, clean
_PLOT_LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="'Inter','Segoe UI',sans-serif", size=11, color="#374151"),
    margin=dict(l=48, r=24, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10),
    ),
)

_GRID_STYLE = dict(
    showgrid=True,
    gridcolor="#f0f0ee",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="#d1d5db",
    zerolinewidth=1,
)


def _axis(title: str, **extra) -> dict:
    return {**_GRID_STYLE, "title": dict(text=title, font=dict(size=10, color="#6b7280")), **extra}


def plot_ecg_interactive(
    time: np.ndarray,
    voltage: np.ndarray,
    title: str = "ECG Waveform",
    color: str = ACCENT,
    r_peaks_sec: Optional[List[float]] = None,
    r_peaks_color: str = "#991b1b",
    r_peaks_label: str = "Detected R-peaks",
    reference_r_peaks_sec: Optional[List[float]] = None,
    show_ecg_grid: bool = True,
    height: int = 400,
) -> go.Figure:
    """Render a clean, light-theme interactive Plotly ECG waveform."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time, y=voltage,
        mode="lines",
        name="ECG",
        line=dict(color=color, width=1.6),
        hovertemplate="t = %{x:.3f} s<br>%{y:.3f} mV<extra></extra>",
    ))

    if reference_r_peaks_sec is not None and len(reference_r_peaks_sec) > 0:
        ref_y = np.interp(reference_r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=reference_r_peaks_sec, y=ref_y,
            mode="markers",
            name="Reference beats",
            marker=dict(symbol="circle-open", size=9, color="#2e7d32", line=dict(width=2)),
            hovertemplate="Ref: %{x:.3f} s<extra></extra>",
        ))

    if r_peaks_sec is not None and len(r_peaks_sec) > 0:
        det_y = np.interp(r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=r_peaks_sec, y=det_y,
            mode="markers",
            name=r_peaks_label,
            marker=dict(symbol="x", size=8, color=r_peaks_color, line=dict(width=1.8)),
            hovertemplate="Det: %{x:.3f} s<extra></extra>",
        ))

    grid_color = "#f3f4f6" if show_ecg_grid else "#fafafa"
    fig.update_layout(
        **_PLOT_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=12, color="#1e2329"), x=0),
        xaxis={**_axis("Time (s)"), "gridcolor": grid_color, "dtick": 0.2 if show_ecg_grid else None},
        yaxis={**_axis("Voltage (mV)"), "gridcolor": grid_color, "dtick": 0.5 if show_ecg_grid else None},
        height=height,
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STATUS CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_status_card(assessment: Dict[str, Any]) -> None:
    """Render the 4-state Signal Quality status card — minimal, text-driven."""
    st_val  = assessment.get("status", "UNABLE TO ASSESS")
    st_label = assessment.get("label", "Unable to assess signal quality")
    summary  = assessment.get("summary", "")
    issues   = assessment.get("detected_issues", [])
    recs     = assessment.get("recommendations", [])

    css_map = {
        "GOOD":           "sqstatus-good",
        "NEEDS REVIEW":   "sqstatus-review",
        "POOR":           "sqstatus-poor",
        "UNABLE TO ASSESS": "sqstatus-unable",
    }
    card_cls = css_map.get(st_val, "sqstatus-unable")

    issues_html = "".join(f"<li>{i}</li>" for i in issues) if issues else "<li>No significant artifacts detected.</li>"
    recs_html   = "".join(f"<li>{r}</li>" for r in recs)   if recs   else "<li>Proceed with analysis.</li>"

    html = f"""
<div class="sqstatus {card_cls}">
  <div class="sqstatus-badge">{st_val}</div>
  <div class="sqstatus-label">{st_label.capitalize()}</div>
  <div class="sqstatus-summary">{summary}</div>
  <div class="sqstatus-cols">
    <div class="sqstatus-col">
      <div class="sqstatus-col-heading">Detected findings</div>
      <ul>{issues_html}</ul>
    </div>
    <div class="sqstatus-col">
      <div class="sqstatus-col-heading">Recommended next steps</div>
      <ul>{recs_html}</ul>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING PROVENANCE PILL
# ─────────────────────────────────────────────────────────────────────────────

def render_record_pill(record: "ECGDataRecord") -> None:
    """Show a compact inline recording info strip."""
    source_labels = {
        "csv": "Uploaded CSV",
        "physionet_wfdb": "PhysioNet / MIT-BIH",
        "synthetic_sandbox": "Synthetic (simulated — not patient data)",
        "synthetic_mitbih_fallback": "Synthetic fallback (not patient data)",
    }
    src_label = source_labels.get(record.source_type, record.source_type)

    html = f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:0.75rem;">
  <span class="rec-pill"><span class="rec-pill-label">Record</span><span class="rec-pill-value">{record.record_name}</span></span>
  <span class="rec-pill"><span class="rec-pill-label">Lead</span><span class="rec-pill-value">{record.lead_name}</span></span>
  <span class="rec-pill"><span class="rec-pill-label">fs</span><span class="rec-pill-value">{record.fs:.0f} Hz</span></span>
  <span class="rec-pill"><span class="rec-pill-label">Duration</span><span class="rec-pill-value">{record.duration_sec:.2f} s</span></span>
  <span class="rec-pill"><span class="rec-pill-label">Source</span><span class="rec-pill-value">{src_label}</span></span>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── APPLICATION HEADER ──────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-title">ECG Signal Quality Analyzer</div>
          <div class="app-subtitle">
            Biomedical signal processing · Quality classification · ANSI/AAMI EC57 beat evaluation
          </div>
          <div class="app-version">{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Educational notice (single, compact)
    st.markdown(
        f'<div class="notice-banner">'
        f'<strong>Research &amp; Educational Use Only.</strong> {CLINICAL_DISCLAIMER}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("### Digital Filter")
    enable_filter = st.sidebar.checkbox(
        "Apply digital preprocessing",
        value=True,
        help="Zero-phase Butterworth bandpass + optional mains notch.",
    )
    filter_lowcut = st.sidebar.slider(
        "High-pass cutoff (Hz)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Suppresses slow baseline drift caused by breathing. Typical: 0.5 Hz.",
    )
    filter_highcut = st.sidebar.slider(
        "Low-pass cutoff (Hz)",
        min_value=15.0, max_value=100.0, value=40.0, step=5.0,
        help="Rejects high-frequency muscle (EMG) noise. Typical: 40 Hz.",
    )
    filter_order = st.sidebar.selectbox(
        "Butterworth order",
        [1, 2, 3, 4, 5],
        index=2,
        help="Higher order = steeper roll-off. filtfilt doubles effective order.",
    )

    st.sidebar.markdown("### Notch Filter")
    enable_notch = st.sidebar.checkbox(
        "Enable mains notch filter",
        value=True,
        help="Attenuates 50 Hz or 60 Hz power-line hum.",
    )
    notch_freq = st.sidebar.selectbox(
        "Mains frequency (Hz)",
        [50.0, 60.0],
        index=0,
        help="50 Hz: Europe / Asia · 60 Hz: North America.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="help-text">Use the <strong>Load ECG</strong> tab to select a recording.</div>',
        unsafe_allow_html=True,
    )

    # ── SESSION STATE / DEFAULT LOAD ────────────────────────────────────────
    if "current_record" not in st.session_state:
        sample_path = pathlib.Path.cwd() / "data" / "samples" / "sample_clean_ecg.csv"
        if sample_path.exists():
            st.session_state.current_record = load_ecg_from_csv(
                str(sample_path), fs=360.0, record_name="Sample Clean ECG (Lead II)"
            )
        else:
            t, ecg, r = generate_clean_synthetic_ecg(duration_sec=10.0, fs=360.0)
            st.session_state.current_record = ECGDataRecord(
                signal=ecg, fs=360.0, record_name="Default Synthetic ECG",
                reference_r_peaks_sec=r, source_type="synthetic_sandbox",
            )

    current_record = st.session_state.current_record
    raw_signal = current_record.signal
    fs = current_record.fs
    filtered_signal = raw_signal.copy()
    b_filt, a_filt = None, None
    filter_warning = None

    if enable_filter:
        try:
            nyq = 0.5 * fs
            if filter_highcut >= nyq:
                raise FilterConfigurationError(
                    f"Low-pass cutoff ({filter_highcut} Hz) must be below Nyquist ({nyq:.1f} Hz)."
                )
            if filter_lowcut >= filter_highcut:
                raise FilterConfigurationError("High-pass cutoff must be below low-pass cutoff.")
            filtered_signal, (b_filt, a_filt) = apply_bandpass_filter(
                filtered_signal, fs, lowcut=filter_lowcut, highcut=filter_highcut, order=filter_order
            )
            if enable_notch and notch_freq < nyq:
                filtered_signal, _ = apply_notch_filter(filtered_signal, fs, notch_freq=notch_freq)
        except (FilterConfigurationError, Exception) as fce:
            filter_warning = str(fce)
            filtered_signal = raw_signal.copy()

    analysis_sig = filtered_signal if enable_filter else raw_signal
    snr_est = estimate_snr_db(raw_signal, filtered_signal) if enable_filter else None

    try:
        sqi_data = analyze_signal_quality(
            analysis_sig, fs=fs, window_length_sec=DEFAULT_WINDOW_LENGTH_SEC,
        )
    except Exception:
        sqi_data = None

    status_assessment = assess_overall_signal_quality(
        raw_signal=raw_signal,
        fs=fs,
        sqi_results=sqi_data,
        snr_db=snr_est,
    )

    # ── TABS ────────────────────────────────────────────────────────────────
    tab_overview, tab_load, tab_viz, tab_filter, tab_sqi, tab_eval, tab_report = st.tabs([
        "Overview",
        "Load ECG",
        "Waveform Explorer",
        "Digital Filtering",
        "Signal Quality",
        "Beat Evaluation",
        "Reports & Export",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with tab_overview:

        # Workflow guide
        st.markdown(
            """
<div class="wf-grid">
  <div class="wf-tile"><div class="wf-num">Step 1</div><div class="wf-title">Overview</div><div class="wf-desc">Recording info and quality status at a glance.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 2</div><div class="wf-title">Load ECG</div><div class="wf-desc">Upload CSV, fetch PhysioNet, or use synthetic data.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 3</div><div class="wf-title">Waveform</div><div class="wf-desc">Inspect the ECG trace in any time window.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 4</div><div class="wf-title">Filtering</div><div class="wf-desc">Apply bandpass / notch and compare waveforms.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 5</div><div class="wf-title">Quality</div><div class="wf-desc">SQI metrics and 4-state quality classification.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 6</div><div class="wf-title">Beats</div><div class="wf-desc">QRS detection and AAMI EC57 evaluation.</div></div>
  <div class="wf-tile"><div class="wf-num">Step 7</div><div class="wf-title">Export</div><div class="wf-desc">Download PDF report, CSV table, or JSON.</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Signal quality status
        st.markdown('<div class="section-label">Overall signal quality</div>', unsafe_allow_html=True)
        render_status_card(status_assessment)

        # Key metrics
        props = assess_signal_properties(raw_signal, fs)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sampling rate", f"{fs:.0f} Hz")
        c2.metric("Duration", f"{props['duration_sec']:.2f} s")
        c3.metric("Samples", f"{props['n_samples']:,}")
        c4.metric("Voltage range",
                  f"{props['min']:.2f} – {props['max']:.2f} mV")
        c5.metric("Est. SNR", f"{snr_est:.1f} dB" if snr_est is not None else "—")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col_meta, col_checks = st.columns(2)
        with col_meta:
            st.markdown('<div class="section-label">Recording information</div>', unsafe_allow_html=True)
            render_record_pill(current_record)
            if current_record.has_reference_annotations:
                st.info(
                    f"Reference annotations available — "
                    f"{len(current_record.reference_r_peaks_sec)} expert-labelled beats."
                )
            else:
                st.warning("No reference beat annotations. Beat evaluation metrics will not be available.")

        with col_checks:
            st.markdown('<div class="section-label">Signal integrity checks</div>', unsafe_allow_html=True)
            if props["is_constant"]:
                st.error("Signal is constant throughout — zero electrical variance detected.")
            elif props["is_all_zero"]:
                st.error("Signal is entirely zero — check lead connections.")
            else:
                st.success("Physiological amplitude range intact.")

            if np.isnan(raw_signal).any():
                st.warning(
                    f"Signal contains {int(np.sum(np.isnan(raw_signal)))} missing (NaN) samples."
                )
            else:
                st.success("No missing (NaN) samples found.")

        # Quick preview
        st.markdown('<div class="section-label">Quick waveform preview — first 6 seconds</div>', unsafe_allow_html=True)
        preview_dur = min(6.0, current_record.duration_sec)
        n_prev = int(round(preview_dur * fs))
        fig_prev = plot_ecg_interactive(
            current_record.time_vector[:n_prev],
            raw_signal[:n_prev],
            title=f"Raw input — first {preview_dur:.1f} s",
            height=280,
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        st.markdown(
            '<div class="help-text">Signal quality analysis evaluates physical signal integrity, '
            'electrode contact, and noise. It does not assess heart rhythm or clinical pathology.</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — LOAD ECG
    # ════════════════════════════════════════════════════════════════════════
    with tab_load:
        st.markdown("Select a data source to load a recording into the pipeline.")

        src_bundled, src_upload, src_physionet, src_synthetic = st.tabs([
            "Bundled samples",
            "Upload CSV",
            "PhysioNet MIT-BIH",
            "Synthetic sandbox",
        ])

        # ── Bundled samples ──────────────────────────────────────────────
        with src_bundled:
            st.markdown('<div class="section-label">Pre-packaged benchmarking datasets</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="help-text">These validated recordings let you immediately test '
                'clean waveforms, multi-artefact noise, and step artefacts.</div>',
                unsafe_allow_html=True,
            )
            sample_choices = {
                "Clean ECG — Lead II, 360 Hz": "sample_clean_ecg.csv",
                "Noisy ECG — baseline drift, EMG, mains hum, flatline dropout": "sample_noisy_ecg.csv",
                "Artefact ECG — high-amplitude spikes and ADC saturation": "sample_artifacts_ecg.csv",
            }
            chosen_label = st.selectbox("Dataset", list(sample_choices.keys()))
            if st.button("Load selected sample", key="btn_load_sample"):
                fname = sample_choices[chosen_label]
                fpath = pathlib.Path.cwd() / "data" / "samples" / fname
                if fpath.exists():
                    st.session_state.current_record = load_ecg_from_csv(
                        str(fpath), fs=360.0, record_name=chosen_label
                    )
                    st.success(f"Loaded '{fname}'.")
                    st.rerun()
                else:
                    st.error(f"File not found: {fpath}")

        # ── Upload CSV ───────────────────────────────────────────────────
        with src_upload:
            st.markdown('<div class="section-label">Upload a custom ECG CSV file</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="help-text">Supports 1-D or 2-D CSV files. '
                'Delimiter and column names are inferred automatically or can be specified below.</div>',
                unsafe_allow_html=True,
            )
            up_file = st.file_uploader(
                "Select CSV or TXT file", type=["csv", "txt"], key="file_up_uploader"
            )
            col_fs, col_delim = st.columns(2)
            with col_fs:
                up_fs = st.number_input(
                    "Sampling frequency (Hz)",
                    min_value=20.0, max_value=5000.0, value=360.0, step=10.0,
                    help="Required when the file has no time column.",
                )
            with col_delim:
                up_delim = st.selectbox(
                    "Delimiter", ["Auto-detect", ",", ";", "\\t", "Space"], index=0
                )
            delim_char = (
                None if up_delim == "Auto-detect"
                else ("\t" if up_delim == "\\t" else (" " if up_delim == "Space" else up_delim))
            )
            if up_file is not None:
                if st.button("Parse and load uploaded CSV", key="btn_parse_csv"):
                    try:
                        new_rec = load_ecg_from_csv(
                            up_file, fs=up_fs, delimiter=delim_char, record_name=up_file.name
                        )
                        st.session_state.current_record = new_rec
                        st.success(
                            f"Loaded '{up_file.name}' — "
                            f"{len(new_rec.signal):,} samples at {new_rec.fs:.1f} Hz."
                        )
                        st.rerun()
                    except SignalValidationError as sve:
                        st.error(f"Validation error: {sve}")
                    except Exception as err:
                        st.error(f"Failed to parse file: {err}")

        # ── PhysioNet MIT-BIH ────────────────────────────────────────────
        with src_physionet:
            st.markdown('<div class="section-label">PhysioNet MIT-BIH Arrhythmia Database</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="help-text">Downloads annotated clinical recordings directly '
                'from PhysioNet. Requires internet access. '
                'If unavailable, a clear error is shown — synthetic data is <em>never</em> '
                'silently substituted for patient data.</div>',
                unsafe_allow_html=True,
            )
            col_pn_rec, col_pn_dur = st.columns(2)
            with col_pn_rec:
                pn_rec = st.selectbox(
                    "MIT-BIH record ID",
                    ["100", "101", "103", "105", "117", "200", "203", "219"],
                    index=0,
                )
            with col_pn_dur:
                pn_dur = st.slider(
                    "Duration to fetch (s)", min_value=5.0, max_value=60.0, value=30.0, step=5.0
                )
            if st.button(f"Download record {pn_rec} from PhysioNet", key="btn_load_pn"):
                with st.spinner(f"Connecting to PhysioNet — record {pn_rec}…"):
                    try:
                        pn_record = load_physionet_record(
                            record_id=pn_rec,
                            database="mitdb",
                            max_duration_sec=pn_dur,
                            allow_synthetic_fallback=False,
                        )
                        st.session_state.current_record = pn_record
                        st.success(
                            f"Downloaded record {pn_rec} — "
                            f"{len(pn_record.reference_r_peaks_sec)} reference annotations."
                        )
                        st.rerun()
                    except SignalValidationError as sve:
                        st.error(f"PhysioNet notice: {sve}")
                    except Exception as err:
                        st.error(
                            f"Network error: could not reach PhysioNet ({err}). "
                            "Check your internet connection or use a bundled sample."
                        )

        # ── Synthetic sandbox ────────────────────────────────────────────
        with src_synthetic:
            st.markdown('<div class="section-label">Synthetic ECG and noise simulation</div>', unsafe_allow_html=True)
            st.info(
                "This generates a mathematically synthesised ECG and injects controlled noise. "
                "It is **not** a real patient recording and must never be described as one."
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                s_hr  = st.slider("Heart rate (BPM)", 40.0, 160.0, 72.0, 2.0)
            with col_s2:
                s_dur = st.slider("Duration (s)", 5.0, 60.0, 15.0, 5.0)
            with col_s3:
                s_fs  = st.selectbox("Sampling frequency (Hz)", [250.0, 360.0, 500.0, 1000.0], index=1)

            st.markdown('<div class="section-label">Artefact injection</div>', unsafe_allow_html=True)
            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                inj_wander = st.checkbox("Baseline wander (respiration)", value=True)
                wander_amp = st.slider("Wander amplitude (mV)", 0.05, 1.0, 0.35, 0.05) if inj_wander else 0.0
                inj_spikes = st.checkbox("Motion spikes", value=False)
            with ci2:
                inj_mains = st.checkbox("Powerline hum (50/60 Hz)", value=True)
                mains_f   = st.selectbox("Mains frequency", [50.0, 60.0], index=0) if inj_mains else 50.0
                mains_amp = st.slider("Hum amplitude (mV)", 0.02, 0.5, 0.15, 0.01) if inj_mains else 0.0
                inj_clip  = st.checkbox("ADC saturation / clipping", value=False)
            with ci3:
                inj_emg   = st.checkbox("Muscle noise (EMG)", value=False)
                emg_amp   = st.slider("EMG amplitude (mV)", 0.02, 0.3, 0.08, 0.01) if inj_emg else 0.0
                inj_flat  = st.checkbox("Flatline dropout (leads off)", value=False)
                flat_dur  = st.slider("Flatline duration (s)", 0.5, 5.0, 2.0, 0.5) if inj_flat else 0.0

            if st.button("Generate synthetic signal", key="btn_gen_synth"):
                t_syn, ecg_syn, r_syn = generate_clean_synthetic_ecg(
                    duration_sec=s_dur, fs=s_fs, heart_rate_bpm=s_hr, random_seed=42,
                )
                if inj_wander: ecg_syn = add_baseline_wander(ecg_syn, s_fs, 0.25, wander_amp)
                if inj_mains:  ecg_syn = add_powerline_noise(ecg_syn, s_fs, mains_f, mains_amp)
                if inj_emg:    ecg_syn = add_muscle_noise(ecg_syn, s_fs, emg_amp)
                if inj_flat:   ecg_syn = add_flatline_segment(ecg_syn, s_fs, s_dur * 0.4, flat_dur)
                if inj_spikes: ecg_syn = add_spike_artifacts(ecg_syn, s_fs, [s_dur*0.25, s_dur*0.75], 4.0)
                if inj_clip:   ecg_syn = add_clipping_saturation(ecg_syn, -0.7, 0.7)
                st.session_state.current_record = ECGDataRecord(
                    signal=ecg_syn, fs=s_fs,
                    lead_name="Synthetic Lead II",
                    record_name=f"Synthetic — {s_hr:.0f} BPM, {s_dur:.0f} s",
                    reference_r_peaks_sec=r_syn,
                    annotation_labels=["N"] * len(r_syn),
                    source_type="synthetic_sandbox",
                )
                st.success(f"Generated synthetic signal ({s_dur:.0f} s, {s_hr:.0f} BPM).")
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — WAVEFORM EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    with tab_viz:
        render_record_pill(current_record)

        col_grid, col_win = st.columns([1, 3])
        with col_grid:
            show_grid = st.checkbox("ECG grid (0.04 s / 0.2 s intervals)", value=True)
        with col_win:
            total_dur = current_record.duration_sec
            time_range = st.slider(
                "Inspection window (seconds)",
                0.0, total_dur,
                (0.0, min(10.0, total_dur)),
                step=0.5,
            )

        start_s, end_s = time_range
        i0 = max(0, int(round(start_s * fs)))
        i1 = min(len(raw_signal), int(round(end_s * fs)))

        fig_viz = plot_ecg_interactive(
            current_record.time_vector[i0:i1],
            raw_signal[i0:i1],
            title=f"ECG — {start_s:.1f} s to {end_s:.1f} s — {current_record.lead_name}",
            show_ecg_grid=show_grid,
            height=440,
        )
        st.plotly_chart(fig_viz, use_container_width=True)

        st.markdown(
            '<div class="help-text">Zoom with mouse scroll or drag. Double-click to reset. '
            'On a standard ECG grid, one small box = 0.04 s (40 ms), one large box = 0.20 s (200 ms).</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — DIGITAL FILTERING
    # ════════════════════════════════════════════════════════════════════════
    with tab_filter:
        if filter_warning:
            st.error(f"Filter configuration problem: {filter_warning}")

        if not enable_filter:
            st.info(
                "Digital preprocessing is disabled in the sidebar. "
                "Enable it to see filtered waveforms and spectral comparisons."
            )
        else:
            render_record_pill(current_record)

            # Applied settings summary
            filt_parts = [f"{filter_lowcut} – {filter_highcut} Hz bandpass (order {filter_order})"]
            if enable_notch:
                filt_parts.append(f"{notch_freq:.0f} Hz notch")
            st.markdown(
                f'<div class="help-text"><strong>Applied:</strong> zero-phase Butterworth — '
                f'{", ".join(filt_parts)}.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="help-text" style="color:#b45309;">Filtering can remove useful '
                'information as well as noise. A high-pass cutoff above 0.67 Hz may distort '
                'ST segments and T-wave amplitude.</div>',
                unsafe_allow_html=True,
            )

            view_mode = st.radio(
                "Waveform view",
                ["Overlaid (raw vs filtered)", "Separate panels (raw / filtered / residual)"],
                horizontal=True,
            )

            dur_comp = min(8.0, current_record.duration_sec)
            n_comp = int(round(dur_comp * fs))
            t_comp   = current_record.time_vector[:n_comp]
            raw_comp = raw_signal[:n_comp]
            flt_comp = filtered_signal[:n_comp]
            res_comp = raw_comp - flt_comp

            if view_mode == "Overlaid (raw vs filtered)":
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=t_comp, y=raw_comp, mode="lines", name="Raw input",
                    line=dict(color="#9ca3af", width=1.4),
                ))
                fig_comp.add_trace(go.Scatter(
                    x=t_comp, y=flt_comp, mode="lines", name="Filtered (zero-phase)",
                    line=dict(color=ACCENT, width=2.0),
                ))
                fig_comp.update_layout(
                    **_PLOT_LAYOUT_BASE,
                    title=dict(
                        text=f"Before/after comparison — first {dur_comp:.1f} s",
                        font=dict(size=12, color="#1e2329"), x=0,
                    ),
                    xaxis=_axis("Time (s)"),
                    yaxis=_axis("Voltage (mV)"),
                    height=380,
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                fig_sub = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["Raw input", "Zero-phase filtered", "Residual (raw − filtered)"],
                    vertical_spacing=0.1,
                )
                fig_sub.add_trace(go.Scatter(x=t_comp, y=raw_comp, line=dict(color="#6b7280", width=1.4)), row=1, col=1)
                fig_sub.add_trace(go.Scatter(x=t_comp, y=flt_comp, line=dict(color=ACCENT, width=1.6)), row=2, col=1)
                fig_sub.add_trace(go.Scatter(x=t_comp, y=res_comp, line=dict(color="#991b1b", width=1.2)), row=3, col=1)
                fig_sub.update_layout(
                    **_PLOT_LAYOUT_BASE,
                    height=520, showlegend=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                )
                fig_sub.update_xaxes(showgrid=True, gridcolor="#f0f0ee")
                fig_sub.update_yaxes(showgrid=True, gridcolor="#f0f0ee")
                st.plotly_chart(fig_sub, use_container_width=True)

            # PSD and frequency response side by side
            col_psd, col_resp = st.columns(2)
            with col_psd:
                st.markdown('<div class="section-label">Power spectral density</div>', unsafe_allow_html=True)
                freqs_r, psd_r = compute_power_spectral_density(raw_signal, fs)
                freqs_f, psd_f = compute_power_spectral_density(filtered_signal, fs)
                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(
                    x=freqs_r, y=psd_r, mode="lines", name="Raw",
                    line=dict(color="#9ca3af", width=1.4),
                ))
                fig_psd.add_trace(go.Scatter(
                    x=freqs_f, y=psd_f, mode="lines", name="Filtered",
                    line=dict(color=ACCENT, width=2),
                ))
                fig_psd.update_layout(
                    **_PLOT_LAYOUT_BASE,
                    xaxis={**_axis("Frequency (Hz)"), "range": [0, min(80, fs / 2)]},
                    yaxis={**_axis("Power density (mV²/Hz)"), "type": "log"},
                    height=320,
                    title=dict(text="PSD — raw vs filtered", font=dict(size=11, color="#1e2329"), x=0),
                )
                st.plotly_chart(fig_psd, use_container_width=True)

            with col_resp:
                st.markdown('<div class="section-label">Filter frequency response |H(f)|</div>', unsafe_allow_html=True)
                if b_filt is not None and a_filt is not None:
                    w, mag_db, _ = compute_filter_frequency_response(b_filt, a_filt, fs)
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=w, y=mag_db, mode="lines", name="Magnitude (dB)",
                        line=dict(color="#2e7d32", width=2),
                    ))
                    fig_h.update_layout(
                        **_PLOT_LAYOUT_BASE,
                        xaxis={**_axis("Frequency (Hz)"), "range": [0, min(80, fs / 2)]},
                        yaxis={**_axis("Magnitude (dB)"), "range": [-60, 5]},
                        height=320,
                        title=dict(text="Filter response", font=dict(size=11, color="#1e2329"), x=0),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

            # Filter explanations
            st.markdown('<div class="section-label">What each filter does</div>', unsafe_allow_html=True)
            fe1, fe2 = st.columns(2)
            with fe1:
                st.markdown(
                    """
**High-pass filter** removes slow baseline drift caused by chest movement during breathing.
Setting the cutoff too high (above 0.67 Hz) can distort the ST segment.

**Low-pass filter** blocks high-frequency muscle tremor (EMG) and radio-frequency noise
while preserving the steep edges of the QRS complex.
""")
            with fe2:
                st.markdown(
                    """
**Mains notch filter** attenuates the 50 Hz or 60 Hz sinusoidal hum radiated by AC power lines.

**Zero-phase (filtfilt)** applies the filter in both the forward and reverse directions,
cancelling any time delay so R-peak timing is accurate.
""")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — SIGNAL QUALITY
    # ════════════════════════════════════════════════════════════════════════
    with tab_sqi:
        render_record_pill(current_record)

        st.markdown('<div class="section-label">4-state quality classification</div>', unsafe_allow_html=True)
        render_status_card(status_assessment)

        if sqi_data:
            ov_score = sqi_data.get("overall_heuristic_sqi", 0.0)
            breakdown = sqi_data.get("classification_breakdown", {})

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Heuristic SQI score", f"{ov_score:.1f} / 100")
            sm2.metric(
                "Optimal windows",
                f"{breakdown.get('Optimal',{}).get('count',0)} "
                f"({breakdown.get('Optimal',{}).get('percentage',0)}%)",
            )
            sm3.metric(
                "Acceptable windows",
                f"{breakdown.get('Acceptable',{}).get('count',0)} "
                f"({breakdown.get('Acceptable',{}).get('percentage',0)}%)",
            )
            sm4.metric(
                "Noisy windows",
                f"{breakdown.get('Noisy',{}).get('count',0)} "
                f"({breakdown.get('Noisy',{}).get('percentage',0)}%)",
            )
            sm5.metric(
                "Unreliable windows",
                f"{breakdown.get('Unreliable / Leads-Off',{}).get('count',0)} "
                f"({breakdown.get('Unreliable / Leads-Off',{}).get('percentage',0)}%)",
            )

            # Quality timeline strip
            st.markdown('<div class="section-label">Quality timeline</div>', unsafe_allow_html=True)
            tl_colors = {
                "Optimal":             "#2e7d32",
                "Acceptable":          "#b45309",
                "Noisy":               "#92400e",
                "Unreliable / Leads-Off": "#991b1b",
            }
            fig_tl = go.Figure()
            for seg in sqi_data["segments"]:
                fig_tl.add_trace(go.Bar(
                    x=[seg["duration_sec"]],
                    y=["Quality"],
                    base=seg["start_time_sec"],
                    orientation="h",
                    name=seg["classification"],
                    marker=dict(color=tl_colors.get(seg["classification"], "#6b7280")),
                    hovertemplate=(
                        f"Window {seg['segment_index']+1}: "
                        f"{seg['start_time_sec']:.1f}–{seg['end_time_sec']:.1f} s<br>"
                        f"Status: {seg['classification']}<br>"
                        f"Score: {seg['heuristic_score']:.1f}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_tl.update_layout(
                **_PLOT_LAYOUT_BASE,
                barmode="stack",
                xaxis={**_axis("Time (s)"), "range": [0, current_record.duration_sec]},
                yaxis=dict(showticklabels=False),
                height=90,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            # Segment metrics table
            st.markdown('<div class="section-label">Segment-by-segment metrics</div>', unsafe_allow_html=True)
            df_segs = create_segment_quality_dataframe(sqi_data)
            st.dataframe(df_segs, use_container_width=True, height=220)

            # Global anomaly counts
            anom = sqi_data.get("global_anomalies", {})
            ga1, ga2, ga3, ga4 = st.columns(4)
            ga1.metric("Flatline episodes", anom.get("flatline_segments_count", 0))
            ga2.metric("Missing (NaN) samples", anom.get("missing_samples_count", 0))
            ga3.metric("Saturation events", anom.get("saturation_count", 0))
            ga4.metric("Abrupt voltage spikes", anom.get("spike_count", 0))

        else:
            st.warning("Signal quality analysis could not run on this recording.")

        # SQI metric explanations
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">What these metrics mean</div>', unsafe_allow_html=True)
        sq1, sq2 = st.columns(2)
        with sq1:
            st.markdown("""
**bas\_sqi — baseline wander ratio.**
Fraction of total spectral power in 0–0.5 Hz. Measures slow drift from patient breathing.
Below 25% is expected for a clean recording. Above 45% indicates heavy drift.

**hf\_sqi — high-frequency noise ratio.**
Power above 40 Hz relative to total. Reflects muscle tremor (EMG) contamination.
Below 20% is acceptable; above 40% is severe.

**pl\_sqi — powerline hum ratio.**
Concentrated power near 50 Hz or 60 Hz. Signals poor electrode–skin impedance or
inadequate grounding.
""")
        with sq2:
            st.markdown("""
**kurtosis\_sqi (kSQI) — statistical sharpness.**
The 4th standardised moment. Clean ECG is highly leptokurtic (above 3.5) because the
tall, narrow QRS spike dominates the distribution. Gaussian noise scores around 3.0;
flatlines score near zero.

**Flatline detection.**
Sequences where consecutive point-to-point variation is below 10⁻⁴ mV for 0.5 s or more.
Indicates electrode detachment or disconnected leads.

**Amplifier saturation.**
Samples exceeding ±5.0 mV flag ADC clipping or hardware rail saturation.

> The heuristic SQI score and thresholds are engineering indicators, not
> clinically validated safety metrics.
""")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — BEAT EVALUATION
    # ════════════════════════════════════════════════════════════════════════
    with tab_eval:
        render_record_pill(current_record)

        with st.spinner("Running Pan-Tompkins QRS detection…"):
            det_samples, det_sec, _ = detect_qrs_pan_tompkins(analysis_sig, fs=fs)

        hr_est = (len(det_sec) / current_record.duration_sec * 60.0
                  if current_record.duration_sec > 0 else 0.0)

        bk1, bk2, bk3 = st.columns(3)
        bk1.metric("Detected QRS complexes", f"{len(det_sec)} beats")
        bk2.metric("Estimated heart rate", f"{hr_est:.1f} BPM")
        bk3.metric("Detector", "Pan-Tompkins (zero-phase)")

        # Waveform with detected peaks
        disp_dur = min(10.0, current_record.duration_sec)
        n_disp = int(round(disp_dur * fs))
        fig_beats = plot_ecg_interactive(
            current_record.time_vector[:n_disp],
            analysis_sig[:n_disp],
            title=f"Detected R-peaks — first {disp_dur:.1f} s",
            r_peaks_sec=[t for t in det_sec if t <= disp_dur],
            reference_r_peaks_sec=(
                [t for t in current_record.reference_r_peaks_sec if t <= disp_dur]
                if current_record.has_reference_annotations else None
            ),
            height=360,
        )
        st.plotly_chart(fig_beats, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">ANSI / AAMI EC57 quantitative evaluation</div>', unsafe_allow_html=True)

        beat_metrics = None
        if current_record.has_reference_annotations:
            tol_ms = st.slider(
                "Matching tolerance (ms)",
                min_value=50.0, max_value=300.0, value=150.0, step=10.0,
                help="ANSI/AAMI EC57 standard defines 150 ms tolerance.",
            )
            beat_metrics = evaluate_beat_detection(
                current_record.reference_r_peaks_sec,
                det_sec,
                tolerance_sec=tol_ms / 1000.0,
                max_duration_sec=current_record.duration_sec,
                annotation_source=f"{current_record.source_type} ({current_record.record_name})",
            )

            bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
            bc1.metric("True positives (TP)", beat_metrics["true_positives"])
            bc2.metric("False negatives (FN)", beat_metrics["false_negatives"])
            bc3.metric("False positives (FP)", beat_metrics["false_positives"])

            se_str  = (f"{beat_metrics['sensitivity_percentage']}%"
                       if beat_metrics["sensitivity_percentage"] is not None else "—")
            ppv_str = (f"{beat_metrics['ppv_percentage']}%"
                       if beat_metrics["ppv_percentage"] is not None else "—")
            f1_str  = (f"{beat_metrics['f1_percentage']}%"
                       if beat_metrics["f1_percentage"] is not None else "—")

            bc4.metric("Sensitivity (Se)", se_str)
            bc5.metric("PPV", ppv_str)
            bc6.metric("F1 score", f1_str)

            st.markdown(
                '<div class="help-text">'
                '<strong>Sensitivity</strong> = proportion of real beats the detector found. '
                '<strong>PPV</strong> = proportion of detections that were real beats. '
                '<strong>F1</strong> = harmonic mean of sensitivity and PPV.'
                '</div>',
                unsafe_allow_html=True,
            )

            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                st.markdown('<div class="section-label">Evaluation details</div>', unsafe_allow_html=True)
                st.write(f"- **Matching standard:** `{beat_metrics['metadata']['standard']}`")
                st.write(f"- **Tolerance:** `{beat_metrics['tolerance_ms']:.1f} ms`")
                st.write(f"- **Reference beat count:** `{beat_metrics['reference_beats_count']}`")
                st.write(f"- **Detected beat count:** `{beat_metrics['detected_beats_count']}`")
                if beat_metrics["mean_jitter_ms"] is not None:
                    st.write(
                        f"- **Mean timing jitter:** `{beat_metrics['mean_jitter_ms']:.2f} ms` "
                        f"(max: `{beat_metrics['max_jitter_ms']:.2f} ms`)"
                    )
            with col_ev2:
                st.markdown('<div class="section-label">Data quality exclusions</div>', unsafe_allow_html=True)
                ref_e = beat_metrics["exclusions"]["reference"]
                det_e = beat_metrics["exclusions"]["detected"]
                st.write(
                    f"- **Reference:** {ref_e['non_finite']} non-finite, "
                    f"{ref_e['out_of_bounds']} out-of-bounds, "
                    f"{ref_e['duplicates']} duplicates excluded"
                )
                st.write(
                    f"- **Detected:** {det_e['non_finite']} non-finite, "
                    f"{det_e['out_of_bounds']} out-of-bounds, "
                    f"{det_e['duplicates']} duplicates excluded"
                )

        else:
            st.info(
                "**Beat evaluation not available** — this recording has no reference annotations.\n\n"
                "Sensitivity (Se) and positive predictive value (PPV) require physician-verified "
                "ground-truth beat labels such as those in the PhysioNet MIT-BIH database. "
                "These metrics are shown as not available rather than fabricated or estimated."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — REPORTS & EXPORT
    # ════════════════════════════════════════════════════════════════════════
    with tab_report:
        render_record_pill(current_record)
        st.markdown(
            "Download the full analysis as a PDF report, a CSV table of segment metrics, "
            "or a structured JSON file."
        )

        filter_meta = {
            "lowcut":         filter_lowcut  if enable_filter else None,
            "highcut":        filter_highcut if enable_filter else None,
            "order":          filter_order   if enable_filter else None,
            "notch_enabled":  enable_notch   if enable_filter else False,
            "notch_freq":     notch_freq     if (enable_filter and enable_notch) else None,
            "notch_q":        DEFAULT_NOTCH_Q,
        }
        record_meta_dict = {
            "record_name":  current_record.record_name,
            "lead":         current_record.lead_name,
            "fs":           fs,
            "duration_sec": current_record.duration_sec,
            "source_type":  current_record.source_type,
            "sample_count": len(raw_signal),
        }

        # Ensure beat_metrics is defined (it may not be if user hasn't visited tab 6)
        try:
            bm_ref = beat_metrics if current_record.has_reference_annotations else None
        except NameError:
            bm_ref = None

        # PDF
        with st.spinner("Compiling PDF report…"):
            try:
                # Re-run detection to get det_sec if tab 6 wasn't visited
                try:
                    _ = det_sec
                except NameError:
                    _, det_sec, _ = detect_qrs_pan_tompkins(analysis_sig, fs=fs)

                pdf_bytes = generate_pdf_report(
                    record_meta=record_meta_dict,
                    status_assessment=status_assessment,
                    filter_settings=filter_meta,
                    sqi_results=sqi_data,
                    raw_signal=raw_signal,
                    filtered_signal=filtered_signal if enable_filter else None,
                    time_vector=current_record.time_vector,
                    detected_r_peaks_sec=det_sec,
                    beat_metrics=bm_ref,
                    app_version=APP_VERSION,
                )
            except Exception as pdf_err:
                pdf_bytes = None
                st.error(f"PDF generation error: {pdf_err}")

        # JSON
        json_str = generate_analysis_report_json(
            record_info=record_meta_dict,
            filter_params=filter_meta,
            sqi_results=sqi_data,
            beat_metrics=bm_ref,
            status_assessment=status_assessment,
        )

        # CSV
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

        safe_name = current_record.record_name.replace(" ", "_").replace("/", "-")

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if pdf_bytes:
                st.download_button(
                    label="Download PDF report",
                    data=pdf_bytes,
                    file_name=f"ecg_quality_report_{safe_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help=(
                        "Multi-page PDF including waveform plots, quality metrics, "
                        "filter settings, beat evaluation, and an educational disclaimer."
                    ),
                )
            else:
                st.button("PDF unavailable", disabled=True, use_container_width=True)

        with dc2:
            st.download_button(
                label="Download CSV table",
                data=csv_str,
                file_name=f"ecg_segments_{safe_name}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Window-by-window SQI scores and artefact flags in CSV format.",
            )

        with dc3:
            st.download_button(
                label="Download JSON data",
                data=json_str,
                file_name=f"ecg_analysis_{safe_name}.json",
                mime="application/json",
                use_container_width=True,
                help="Machine-readable JSON with all results, metadata, and disclaimers.",
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Markdown summary preview</div>', unsafe_allow_html=True)
        st.markdown(md_summary)


if __name__ == "__main__":
    main()
