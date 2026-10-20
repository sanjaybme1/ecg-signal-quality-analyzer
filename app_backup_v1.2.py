"""
ECG Signal Quality Analyzer & Noise Testing Tool
Interactive Biomedical Engineering Web Application.
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

# Application Version
APP_VERSION = "v1.2"

# Page configuration
st.set_page_config(
    page_title="ECG Signal Quality Analyzer & Noise Testing Tool",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Biomedical Telemetry Dark Theme: Navy & Cyan)
CUSTOM_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #070e1c;
        color: #e0e6ed;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #091326;
        border-right: 1px solid rgba(0, 229, 255, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00e5ff !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric card styles */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(13, 25, 48, 0.85) 0%, rgba(18, 35, 68, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 229, 255, 0.6);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: #8da4c4 !important;
        font-size: 0.80rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-family: 'Consolas', 'Courier New', monospace;
        font-weight: 700;
        font-size: 1.35rem !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(9, 19, 38, 0.8);
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid rgba(0, 229, 255, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        color: #8fa5c2;
        border-radius: 6px;
        padding: 0 16px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 229, 255, 0.15) !important;
        color: #00e5ff !important;
        border-bottom: 2px solid #00e5ff !important;
    }

    /* Disclaimer banner */
    .disclaimer-banner {
        background: rgba(255, 152, 0, 0.10);
        border-left: 4px solid #ff9800;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-size: 0.84rem;
        color: #ffe0b2;
        line-height: 1.4;
    }
    
    /* Workflow guide step card */
    .step-card {
        background: rgba(13, 24, 45, 0.65);
        border: 1px solid rgba(0, 229, 255, 0.15);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        height: 100%;
    }
    .step-num {
        color: #00e5ff;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .step-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }
    .step-desc {
        color: #9cb3ce;
        font-size: 0.80rem;
        line-height: 1.35;
    }

    /* 4-State Status Card Styles */
    .status-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0 18px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .status-card-good {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.12) 0%, rgba(0, 160, 65, 0.18) 100%);
        border: 1.5px solid #00c853;
    }
    .status-card-review {
        background: linear-gradient(135deg, rgba(255, 214, 0, 0.12) 0%, rgba(220, 180, 0, 0.18) 100%);
        border: 1.5px solid #ffd600;
    }
    .status-card-poor {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.12) 0%, rgba(210, 50, 50, 0.18) 100%);
        border: 1.5px solid #ff5252;
    }
    .status-card-unable {
        background: linear-gradient(135deg, rgba(144, 164, 174, 0.12) 0%, rgba(120, 140, 150, 0.18) 100%);
        border: 1.5px solid #90a4ae;
    }
    .status-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .status-badge-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .status-summary {
        font-size: 0.90rem;
        color: #e0e6ed;
        margin-bottom: 10px;
        line-height: 1.45;
    }
    .status-subheading {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .status-list {
        margin: 0;
        padding-left: 18px;
        font-size: 0.84rem;
        color: #cfd8dc;
        line-height: 1.4;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_status_card(assessment: Dict[str, Any]):
    """Render the prominent 4-state Signal Quality Status Card."""
    st_val = assessment.get("status", "UNABLE TO ASSESS")
    st_label = assessment.get("label", "Unable to assess signal quality")
    st_icon = assessment.get("icon", "⚪")
    st_color = assessment.get("color", "#90a4ae")
    summary = assessment.get("summary", "")
    issues = assessment.get("detected_issues", [])
    recs = assessment.get("recommendations", [])

    css_class_map = {
        "GOOD": "status-card-good",
        "NEEDS REVIEW": "status-card-review",
        "POOR": "status-card-poor",
        "UNABLE TO ASSESS": "status-card-unable",
    }
    card_class = css_class_map.get(st_val, "status-card-unable")

    issues_html = "".join([f"<li>{i}</li>" for i in issues]) if issues else "<li>None (No significant artifacts detected)</li>"
    recs_html = "".join([f"<li>{r}</li>" for r in recs]) if recs else "<li>None (Signal nominal)</li>"

    html = f"""
    <div class="status-card {card_class}">
        <div class="status-header">
            <span style="font-size: 1.5rem;">{st_icon}</span>
            <span class="status-badge-title" style="color: {st_color};">
                {st_val} — {st_label.upper()}
            </span>
        </div>
        <div class="status-summary">
            {summary}
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px;">
                <div class="status-subheading" style="color: #ffb74d;">⚠️ Detected Quality Findings:</div>
                <ul class="status-list">
                    {issues_html}
                </ul>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div class="status-subheading" style="color: #81c784;">💡 Recommended Next Actions:</div>
                <ul class="status-list">
                    {recs_html}
                </ul>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_ecg_interactive(
    time: np.ndarray,
    voltage: np.ndarray,
    title: str = "ECG Waveform",
    color: str = "#00e5ff",
    r_peaks_sec: Optional[List[float]] = None,
    r_peaks_color: str = "#ff5252",
    r_peaks_label: str = "Detected R-peaks",
    reference_r_peaks_sec: Optional[List[float]] = None,
    show_ecg_grid: bool = True,
    height: int = 420,
) -> go.Figure:
    """Helper to render an interactive Plotly ECG waveform with optional grid."""
    fig = go.Figure()

    # Base waveform
    fig.add_trace(go.Scatter(
        x=time,
        y=voltage,
        mode="lines",
        name="ECG Voltage",
        line=dict(color=color, width=1.8),
        hovertemplate="Time: %{x:.3f} s<br>Voltage: %{y:.3f} mV<extra></extra>",
    ))

    # Reference beats (if any)
    if reference_r_peaks_sec is not None and len(reference_r_peaks_sec) > 0:
        ref_y = np.interp(reference_r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=reference_r_peaks_sec,
            y=ref_y,
            mode="markers",
            name="Reference Beats (PhysioNet)",
            marker=dict(symbol="circle-open", size=9, color="#00e676", line=dict(width=2)),
            hovertemplate="Ref Beat: %{x:.3f} s<extra></extra>",
        ))

    # Detected peaks (if any)
    if r_peaks_sec is not None and len(r_peaks_sec) > 0:
        det_y = np.interp(r_peaks_sec, time, voltage)
        fig.add_trace(go.Scatter(
            x=r_peaks_sec,
            y=det_y,
            mode="markers",
            name=r_peaks_label,
            marker=dict(symbol="x", size=8, color=r_peaks_color, line=dict(width=2)),
            hovertemplate="Detected R: %{x:.3f} s<extra></extra>",
        ))

    # Standard ECG Grid styling
    xaxis_config = dict(
        title="Time (seconds)",
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.08)" if show_ecg_grid else "rgba(255,255,255,0.03)",
        dtick=0.2 if show_ecg_grid else None,
        zeroline=True,
        zerolinecolor="rgba(0, 229, 255, 0.2)",
    )
    yaxis_config = dict(
        title="Voltage (mV)",
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.08)" if show_ecg_grid else "rgba(255,255,255,0.03)",
        dtick=0.5 if show_ecg_grid else None,
        zeroline=True,
        zerolinecolor="rgba(0, 229, 255, 0.2)",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e0")),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        template="plotly_dark",
        plot_bgcolor="rgba(11, 19, 36, 0.7)",
        paper_bgcolor="rgba(11, 19, 36, 0.2)",
        margin=dict(l=50, r=30, t=45, b=40),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main():
    # ---------------- HEADER & LOGO ----------------
    col_head_logo, col_head_title = st.columns([1, 10])
    with col_head_logo:
        st.markdown("<h1 style='text-align: center; margin: 0; padding: 0;'>🫀</h1>", unsafe_allow_html=True)
    with col_head_title:
        st.title("ECG Signal Quality Analyzer & Noise Testing Tool")
        st.caption(f"Biomedical Engineering Signal Processing, Quality Classification & ANSI/AAMI EC57 Evaluation Engine ({APP_VERSION})")

    # Regulatory Disclaimer Banner
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Research & Educational Notice:</strong> {CLINICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SIDEBAR: QUICK CONFIG & FILTERING ----------------
    st.sidebar.markdown("### ⚙️ Quick Settings")
    st.sidebar.markdown("Use the **📁 2. Load ECG** tab in the main pane to ingest recordings or generate synthetic waveforms.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Digital Filter Controls")
    enable_filter = st.sidebar.checkbox("Enable Digital Preprocessing", value=True, help="Applies bidirectional zero-phase bandpass and notch filtering.")
    filter_lowcut = st.sidebar.slider("Highpass Cutoff (Hz)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, help="Suppresses respiratory baseline wander (<0.5 Hz).")
    filter_highcut = st.sidebar.slider("Lowpass Cutoff (Hz)", min_value=15.0, max_value=100.0, value=40.0, step=5.0, help="Suppresses high-frequency EMG muscle noise (>40 Hz).")
    filter_order = st.sidebar.selectbox("Butterworth Order", [1, 2, 3, 4, 5], index=2, help="Filter steepness. 3rd-order Butterworth achieves 6th-order effective attenuation via filtfilt.")
    enable_notch = st.sidebar.checkbox("Enable Mains Notch Filter", value=True, help="Attenuates 50 Hz or 60 Hz sinusoidal AC mains hum.")
    notch_freq = st.sidebar.selectbox("Notch Frequency (Hz)", [50.0, 60.0], index=0, help="50 Hz in Europe/Asia; 60 Hz in North America.")

    # ---------------- SESSION STATE FOR LOADED DATA ----------------
    if "current_record" not in st.session_state:
        # Load default clean bundled sample
        sample_path = pathlib.Path.cwd() / "data" / "samples" / "sample_clean_ecg.csv"
        if sample_path.exists():
            st.session_state.current_record = load_ecg_from_csv(str(sample_path), fs=360.0, record_name="Sample Clean ECG (Lead II)")
        else:
            t, ecg, r = generate_clean_synthetic_ecg(duration_sec=10.0, fs=360.0)
            st.session_state.current_record = ECGDataRecord(signal=ecg, fs=360.0, record_name="Default Synthetic ECG", reference_r_peaks_sec=r)

    # Filter signal
    current_record = st.session_state.current_record
    raw_signal = current_record.signal
    fs = current_record.fs
    filtered_signal = raw_signal.copy()
    b_filt, a_filt = None, None
    filter_warning = None

    if enable_filter:
        try:
            # Validate cutoff vs Nyquist
            nyq = 0.5 * fs
            if filter_highcut >= nyq:
                raise FilterConfigurationError(f"Lowpass cutoff ({filter_highcut} Hz) must be strictly less than Nyquist ({nyq:.1f} Hz).")
            if filter_lowcut >= filter_highcut:
                raise FilterConfigurationError("Highpass cutoff must be strictly less than lowpass cutoff.")

            filtered_signal, (b_filt, a_filt) = apply_bandpass_filter(
                filtered_signal, fs, lowcut=filter_lowcut, highcut=filter_highcut, order=filter_order
            )
            if enable_notch and notch_freq < nyq:
                filtered_signal, _ = apply_notch_filter(filtered_signal, fs, notch_freq=notch_freq)
        except (FilterConfigurationError, Exception) as fce:
            filter_warning = str(fce)
            filtered_signal = raw_signal.copy()

    # Pre-calculate SQI and 4-state Quality Assessment
    analysis_sig = filtered_signal if enable_filter else raw_signal
    snr_est = estimate_snr_db(raw_signal, filtered_signal) if enable_filter else None

    try:
        sqi_data = analyze_signal_quality(
            analysis_sig,
            fs=fs,
            window_length_sec=DEFAULT_WINDOW_LENGTH_SEC,
        )
    except Exception:
        sqi_data = None

    status_assessment = assess_overall_signal_quality(
        raw_signal=raw_signal,
        fs=fs,
        sqi_results=sqi_data,
        snr_db=snr_est,
    )

    # ---------------- 7 MAIN TABS ----------------
    tab_overview, tab_load, tab_viz, tab_filter, tab_sqi, tab_eval, tab_report = st.tabs([
        "📊 1. Overview",
        "📁 2. Load ECG",
        "📈 3. Waveform Explorer",
        "🎛️ 4. Digital Filtering",
        "🛡️ 5. Signal Quality Analysis",
        "💓 6. Beat Evaluation",
        "📑 7. Reports & Export",
    ])

    # ==================== TAB 1: OVERVIEW ====================
    with tab_overview:
        st.subheader("System Overview & Signal Provenance")

        # Workflow Guide Cards
        st.markdown("#### 🚀 End-to-End Analysis Workflow")
        col_w1, col_w2, col_w3, col_w4, col_w5, col_w6, col_w7 = st.columns(7)
        with col_w1:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 1</div><div class="step-title">Overview</div><div class="step-desc">Inspect metadata, provenance & 4-state status.</div></div>""", unsafe_allow_html=True)
        with col_w2:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 2</div><div class="step-title">Load ECG</div><div class="step-desc">Upload CSV, fetch PhysioNet or simulate noise.</div></div>""", unsafe_allow_html=True)
        with col_w3:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 3</div><div class="step-title">Explorer</div><div class="step-desc">Zoom into P-Q-R-S-T on 0.04s standard grid.</div></div>""", unsafe_allow_html=True)
        with col_w4:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 4</div><div class="step-title">Filtering</div><div class="step-desc">Zero-phase Butterworth bandpass & notch.</div></div>""", unsafe_allow_html=True)
        with col_w5:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 5</div><div class="step-title">SQI Analysis</div><div class="step-desc">Evaluate spectral, statistical & artifact indices.</div></div>""", unsafe_allow_html=True)
        with col_w6:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 6</div><div class="step-title">Beat Eval</div><div class="step-desc">Pan-Tompkins QRS & ANSI/AAMI EC57 metrics.</div></div>""", unsafe_allow_html=True)
        with col_w7:
            st.markdown("""<div class="step-card"><div class="step-num">STEP 7</div><div class="step-title">Export</div><div class="step-desc">Download PDF Report, CSV table & JSON data.</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Prominent 4-State Quality Status Card
        st.markdown("#### 🛡️ Overall Signal Quality Assessment")
        render_status_card(status_assessment)

        # Summary Metrics
        props = assess_signal_properties(raw_signal, fs)
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Sampling Rate", f"{fs:.1f} Hz")
        kpi2.metric("Total Duration", f"{props['duration_sec']:.2f} s")
        kpi3.metric("Sample Points", f"{props['n_samples']:,}")
        kpi4.metric("Voltage Span", f"{props['min']:.2f} to {props['max']:.2f} mV")
        kpi5.metric("Estimated SNR", f"{snr_est:.1f} dB" if snr_est is not None else "N/A")

        # Provenance Details
        col_meta_left, col_meta_right = st.columns([1, 1])
        with col_meta_left:
            st.markdown("#### 📋 Recording Profile")
            st.write(f"- **Record Identifier:** `{current_record.record_name}`")
            st.write(f"- **Lead / Channel:** `{current_record.lead_name}`")
            st.write(f"- **Voltage Units:** `{current_record.units}`")
            st.write(f"- **Ingestion Source:** `{current_record.source_type}`")
            if current_record.has_reference_annotations:
                st.write(f"- **Reference Annotations:** `{len(current_record.reference_r_peaks_sec)} expert beats available`")
            else:
                st.write("- **Reference Annotations:** *None (Unannotated recording)*")

        with col_meta_right:
            st.markdown("#### 🔍 Hardware & Transmission Checks")
            if props["is_constant"]:
                st.error("❌ Constant Flatline: Zero electrical variance detected across recording.")
            elif props["is_all_zero"]:
                st.error("❌ All-Zero Signal: Complete absence of electrical potential (leads off).")
            else:
                st.success("✅ Physiological dynamic range intact.")

            if np.isnan(raw_signal).any():
                st.warning(f"⚠️ Signal contains {np.sum(np.isnan(raw_signal))} missing/NaN samples.")
            else:
                st.success("✅ Zero missing (NaN) samples detected.")

        # Quick Waveform Preview
        st.markdown("#### ⚡ Quick Waveform Preview (First 6 Seconds)")
        preview_dur = min(6.0, current_record.duration_sec)
        preview_samples = int(round(preview_dur * fs))
        fig_preview = plot_ecg_interactive(
            current_record.time_vector[:preview_samples],
            raw_signal[:preview_samples],
            title=f"Raw Input ECG — First {preview_dur:.1f} Seconds",
            color="#00e5ff",
            height=300,
        )
        st.plotly_chart(fig_preview, use_container_width=True)

        st.caption("ℹ️ **Scientific Notice:** Signal quality analysis evaluates physical signal integrity, electrode contact, and noise. It does NOT diagnose heart rhythm or clinical pathology.")

    # ==================== TAB 2: LOAD ECG ====================
    with tab_load:
        st.subheader("Data Acquisition & Signal Ingestion")
        st.write("Select a data source to ingest an ECG recording into the analysis pipeline:")

        source_tab1, source_tab2, source_tab3, source_tab4 = st.tabs([
            "📦 Bundled Samples",
            "📤 Upload Custom CSV",
            "🌐 PhysioNet MIT-BIH",
            "🧪 Synthetic & Noise Sandbox",
        ])

        # Subtab 1: Bundled Samples
        with source_tab1:
            st.markdown("##### Pre-Packaged Benchmarking Datasets")
            st.write("These validated recordings allow immediate testing of clean waveforms, severe multi-artifact noise, and sudden step artifacts.")
            sample_opt = st.selectbox(
                "Choose Bundled Sample Dataset",
                [
                    "sample_clean_ecg.csv (Nominal Lead II, 360 Hz)",
                    "sample_noisy_ecg.csv (Baseline Drift, Muscle EMG, Line Hum, Flatline Dropout)",
                    "sample_artifacts_ecg.csv (High-Amplitude Voltage Spikes & Saturation)",
                ]
            )
            file_map = {
                "sample_clean_ecg.csv (Nominal Lead II, 360 Hz)": "sample_clean_ecg.csv",
                "sample_noisy_ecg.csv (Baseline Drift, Muscle EMG, Line Hum, Flatline Dropout)": "sample_noisy_ecg.csv",
                "sample_artifacts_ecg.csv (High-Amplitude Voltage Spikes & Saturation)": "sample_artifacts_ecg.csv",
            }
            if st.button("📥 Load Selected Bundled Sample", key="btn_load_sample"):
                filename = file_map[sample_opt]
                filepath = pathlib.Path.cwd() / "data" / "samples" / filename
                if filepath.exists():
                    st.session_state.current_record = load_ecg_from_csv(str(filepath), fs=360.0, record_name=sample_opt)
                    st.success(f"Successfully loaded '{filename}'!")
                    st.rerun()
                else:
                    st.error(f"Sample file not found at {filepath}.")

        # Subtab 2: Upload CSV
        with source_tab2:
            st.markdown("##### Upload Custom ECG CSV File")
            st.write("Supports arbitrary 1D or 2D CSV files. Delimiter and columns are inferred automatically, or can be specified:")
            up_file = st.file_uploader("Select CSV or TXT file", type=["csv", "txt"], key="file_up_uploader")
            col_csv_fs, col_csv_delim = st.columns(2)
            with col_csv_fs:
                up_fs = st.number_input("Sampling Frequency (Hz)", min_value=20.0, max_value=5000.0, value=360.0, step=10.0, help="Specify fs if CSV does not contain a time column.")
            with col_csv_delim:
                up_delim = st.selectbox("Delimiter", ["Auto-detect", ",", ";", "\\t", "Space"], index=0)

            delim_char = None if up_delim == "Auto-detect" else ("\t" if up_delim == "\\t" else (" " if up_delim == "Space" else up_delim))

            if up_file is not None:
                if st.button("📥 Parse & Ingest Uploaded CSV", key="btn_parse_csv"):
                    try:
                        new_rec = load_ecg_from_csv(
                            up_file,
                            fs=up_fs,
                            delimiter=delim_char,
                            record_name=up_file.name
                        )
                        st.session_state.current_record = new_rec
                        st.success(f"Successfully parsed '{up_file.name}' ({len(new_rec.signal):,} samples at {new_rec.fs:.1f} Hz).")
                        st.rerun()
                    except SignalValidationError as sve:
                        st.error(f"Signal Validation Error: {sve}")
                    except Exception as err:
                        st.error(f"Failed to parse CSV file: {err}")

        # Subtab 3: PhysioNet MIT-BIH Database
        with source_tab3:
            st.markdown("##### PhysioNet MIT-BIH Arrhythmia Database")
            st.write("Directly download annotated clinical recordings from the world-standard MIT-BIH database.")
            col_pn_rec, col_pn_dur = st.columns(2)
            with col_pn_rec:
                pn_rec = st.selectbox("MIT-BIH Record ID", ["100", "101", "103", "105", "117", "200", "203", "219"], index=0)
            with col_pn_dur:
                pn_dur = st.slider("Duration to fetch (seconds)", min_value=5.0, max_value=60.0, value=30.0, step=5.0)

            st.caption("🔒 **Scientific Integrity Policy:** If a network connection is unavailable, this application will **never** silently replace clinical patient data with synthetic data. An explicit failure message will be shown.")

            if st.button(f"🌐 Download Record {pn_rec} from PhysioNet", key="btn_load_pn"):
                with st.spinner(f"Connecting to PhysioNet and fetching Record {pn_rec}..."):
                    try:
                        pn_record = load_physionet_record(
                            record_id=pn_rec,
                            database="mitdb",
                            max_duration_sec=pn_dur,
                            allow_synthetic_fallback=False,
                        )
                        st.session_state.current_record = pn_record
                        st.success(f"Successfully downloaded Record {pn_rec} with {len(pn_record.reference_r_peaks_sec)} reference annotations!")
                        st.rerun()
                    except SignalValidationError as sve:
                        st.error(f"⚠️ PhysioNet Download Notice: {sve}")
                    except Exception as err:
                        st.error(f"⚠️ Network Error: Could not reach PhysioNet server ({err}). Please check your internet connection or use the Bundled Samples / Synthetic Sandbox.")

        # Subtab 4: Synthetic Sandbox
        with source_tab4:
            st.markdown("##### Synthetic ECG & Noise Simulation Sandbox")
            st.write("Generate reproducible, mathematically synthesized Gaussian P-Q-R-S-T complexes and inject controlled physiological or environmental noise:")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                s_hr = st.slider("Heart Rate (BPM)", 40.0, 160.0, 72.0, 2.0)
            with col_s2:
                s_dur = st.slider("Signal Duration (sec)", 5.0, 60.0, 15.0, 5.0)
            with col_s3:
                s_fs = st.selectbox("Sampling Frequency (Hz)", [250.0, 360.0, 500.0, 1000.0], index=1)

            st.markdown("###### Inject Artifacts & Noise")
            c_inj1, c_inj2, c_inj3 = st.columns(3)
            with c_inj1:
                inj_wander = st.checkbox("Baseline Wander (Respiration)", value=True)
                wander_amp = st.slider("Wander Amplitude (mV)", 0.05, 1.0, 0.35, 0.05) if inj_wander else 0.0

                inj_spikes = st.checkbox("Electrode Motion Spikes", value=False)

            with c_inj2:
                inj_mains = st.checkbox("Powerline Hum (50/60 Hz)", value=True)
                mains_f = st.selectbox("Mains Frequency", [50.0, 60.0], index=0) if inj_mains else 50.0
                mains_amp = st.slider("Mains Amplitude (mV)", 0.02, 0.5, 0.15, 0.01) if inj_mains else 0.0

                inj_clip = st.checkbox("ADC Rail Saturation / Clipping", value=False)

            with c_inj3:
                inj_emg = st.checkbox("Muscle Tremor (EMG Noise)", value=False)
                emg_amp = st.slider("EMG Amplitude (mV)", 0.02, 0.3, 0.08, 0.01) if inj_emg else 0.0

                inj_flat = st.checkbox("Flatline Dropout (Leads-Off)", value=False)
                flat_dur = st.slider("Flatline Duration (sec)", 0.5, 5.0, 2.0, 0.5) if inj_flat else 0.0

            if st.button("⚡ Generate Synthetic Signal & Load", key="btn_gen_synth"):
                t_syn, ecg_syn, r_syn = generate_clean_synthetic_ecg(
                    duration_sec=s_dur,
                    fs=s_fs,
                    heart_rate_bpm=s_hr,
                    random_seed=42,
                )
                if inj_wander:
                    ecg_syn = add_baseline_wander(ecg_syn, s_fs, frequency_hz=0.25, amplitude_mv=wander_amp)
                if inj_mains:
                    ecg_syn = add_powerline_noise(ecg_syn, s_fs, frequency_hz=mains_f, amplitude_mv=mains_amp)
                if inj_emg:
                    ecg_syn = add_muscle_noise(ecg_syn, s_fs, amplitude_mv=emg_amp)
                if inj_flat:
                    ecg_syn = add_flatline_segment(ecg_syn, s_fs, start_sec=s_dur * 0.4, duration_sec=flat_dur)
                if inj_spikes:
                    ecg_syn = add_spike_artifacts(ecg_syn, s_fs, timestamps_sec=[s_dur * 0.25, s_dur * 0.75], spike_amplitude_mv=4.0)
                if inj_clip:
                    ecg_syn = add_clipping_saturation(ecg_syn, clip_min_mv=-0.7, clip_max_mv=0.7)

                st.session_state.current_record = ECGDataRecord(
                    signal=ecg_syn,
                    fs=s_fs,
                    lead_name="Synthetic Lead II",
                    record_name=f"Synthetic Simulation ({s_hr:.0f} BPM)",
                    reference_r_peaks_sec=r_syn,
                    annotation_labels=["N"] * len(r_syn),
                    source_type="synthetic_sandbox",
                )
                st.success(f"Generated synthetic signal ({s_dur:.1f} s, {s_hr:.0f} BPM)!")
                st.rerun()

    # ==================== TAB 3: WAVEFORM EXPLORER ====================
    with tab_viz:
        st.subheader("Interactive ECG Waveform Explorer")
        st.write(f"Inspecting **{current_record.record_name}** | Lead: **{current_record.lead_name}** | $f_s$: **{fs:.1f} Hz**")

        col_grid, col_win = st.columns([1, 2])
        with col_grid:
            show_grid = st.checkbox("Display Standard Clinical ECG Grid (0.04s / 0.2s)", value=True)
        with col_win:
            total_dur = current_record.duration_sec
            time_range = st.slider(
                "Inspection Window (seconds)",
                min_value=0.0,
                max_value=total_dur,
                value=(0.0, min(10.0, total_dur)),
                step=0.5,
            )

        start_s, end_s = time_range
        idx_start = max(0, int(round(start_s * fs)))
        idx_end = min(len(raw_signal), int(round(end_s * fs)))

        sub_t = current_record.time_vector[idx_start:idx_end]
        sub_raw = raw_signal[idx_start:idx_end]

        fig_viz = plot_ecg_interactive(
            sub_t,
            sub_raw,
            title=f"ECG Trace [{start_s:.1f}s to {end_s:.1f}s] — {current_record.lead_name}",
            color="#00e5ff",
            show_ecg_grid=show_grid,
            height=460,
        )
        st.plotly_chart(fig_viz, use_container_width=True)

        st.info("💡 **Clinical Engineering Tip:** Use your mouse to zoom into individual P-waves, QRS complexes, and T-waves. Double-click to reset zoom. On a standard ECG grid, 1 small horizontal box represents $0.04\\text{ s}$ ($40\\text{ ms}$) and 1 large box represents $0.20\\text{ s}$ ($200\\text{ ms}$).")

    # ==================== TAB 4: DIGITAL FILTERING ====================
    with tab_filter:
        st.subheader("Zero-Phase Digital Filtering & Spectral Analysis")

        if filter_warning:
            st.error(f"⚠️ Filter Configuration Issue: {filter_warning}")

        if not enable_filter:
            st.warning("Digital preprocessing is currently disabled in the sidebar. Enable it to view filtered waveforms, residuals, and frequency responses.")
        else:
            view_mode = st.radio("Waveform Display Mode", ["Overlaid Comparison (Raw vs Filtered)", "Separate Residual Subplots (Raw, Filtered, Residual)"], horizontal=True)

            dur_comp = min(8.0, current_record.duration_sec)
            comp_samples = int(round(dur_comp * fs))
            t_comp = current_record.time_vector[:comp_samples]
            raw_comp = raw_signal[:comp_samples]
            filt_comp = filtered_signal[:comp_samples]
            res_comp = raw_comp - filt_comp

            if view_mode == "Overlaid Comparison (Raw vs Filtered)":
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=t_comp, y=raw_comp, mode="lines", name="Raw Input ECG", line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5)))
                fig_comp.add_trace(go.Scatter(x=t_comp, y=filt_comp, mode="lines", name="Filtered ECG (Zero-Phase)", line=dict(color="#00e5ff", width=2.0)))
                fig_comp.update_layout(
                    title=f"Before-and-After Waveform Comparison (0 – {dur_comp:.1f} s)",
                    xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Voltage (mV)", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                fig_sub = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["1. Raw Input ECG", "2. Zero-Phase Filtered ECG", "3. Noise Residual (Raw - Filtered)"])
                fig_sub.add_trace(go.Scatter(x=t_comp, y=raw_comp, line=dict(color="#ff9800", width=1.5)), row=1, col=1)
                fig_sub.add_trace(go.Scatter(x=t_comp, y=filt_comp, line=dict(color="#00e5ff", width=1.5)), row=2, col=1)
                fig_sub.add_trace(go.Scatter(x=t_comp, y=res_comp, line=dict(color="#e91e63", width=1.2)), row=3, col=1)
                fig_sub.update_layout(template="plotly_dark", height=520, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_sub, use_container_width=True)

            # Frequency Domain Analysis
            col_psd, col_resp = st.columns(2)
            with col_psd:
                st.markdown("#### Power Spectral Density (PSD)")
                freqs_raw, psd_raw = compute_power_spectral_density(raw_signal, fs)
                freqs_filt, psd_filt = compute_power_spectral_density(filtered_signal, fs)

                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_raw, mode="lines", name="Raw PSD", line=dict(color="#ff9800", width=1.5)))
                fig_psd.add_trace(go.Scatter(x=freqs_filt, y=psd_filt, mode="lines", name="Filtered PSD", line=dict(color="#00e5ff", width=2)))
                fig_psd.update_layout(
                    xaxis=dict(title="Frequency (Hz)", range=[0, min(80, fs / 2)]),
                    yaxis=dict(title="Power Density (mV²/Hz)", type="log"),
                    template="plotly_dark",
                    height=340,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig_psd, use_container_width=True)

            with col_resp:
                st.markdown("#### Filter Frequency Response |H(f)|")
                if b_filt is not None and a_filt is not None:
                    w, mag_db, phase = compute_filter_frequency_response(b_filt, a_filt, fs)
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=w, y=mag_db, mode="lines", name="Magnitude (dB)", line=dict(color="#00e676", width=2)))
                    fig_h.update_layout(
                        xaxis=dict(title="Frequency (Hz)", range=[0, min(80, fs / 2)]),
                        yaxis=dict(title="Magnitude (dB)", range=[-60, 5]),
                        template="plotly_dark",
                        height=340,
                        margin=dict(l=40, r=20, t=30, b=40),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

            # Educational Filter Explanations
            col_edu1, col_edu2 = st.columns(2)
            with col_edu1:
                st.markdown("""
                ##### 📘 Digital Filters in Simple English:
                - **High-pass Filter ($0.5\\text{ Hz}$):** Rejects slow baseline drift caused by chest expansion during patient breathing, while preserving slow repolarization T-waves.
                - **Low-pass Filter ($40.0\\text{ Hz}$):** Blocks high-frequency muscle tremor (EMG) and radio-frequency noise while preserving steep QRS edges.
                - **Butterworth Design:** Provides a maximally flat magnitude response in the passband with no ripple.
                - **Mains Notch Filter ($50/60\\text{ Hz}$):** Specifically attenuates electrical hum radiated by AC powerlines.
                """)
            with col_edu2:
                st.markdown("""
                ##### ⚠️ Waveform Distortion Warning:
                > [!WARNING]
                > **Filtering alters morphological waveforms:**
                > 1. Setting highpass cutoff above $0.67\\text{ Hz}$ can artificially distort **ST segment elevation or depression**, mimicking or masking myocardial ischemia.
                > 2. Aggressive lowpass filtering below $30\\text{ Hz}$ attenuates sharp **QRS amplitudes** and pacemaker spikes.
                > 3. Zero-phase bidirectional filtering (`filtfilt`) is critical to prevent artificial phase shift delays.
                """)

    # ==================== TAB 5: SIGNAL QUALITY ANALYSIS ====================
    with tab_sqi:
        st.subheader("Signal Quality Indices (SQI) & Windowed Evaluation")

        # Status Card
        render_status_card(status_assessment)

        # Overview Metrics
        if sqi_data:
            ov_score = sqi_data.get("overall_heuristic_sqi", 0.0)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Overall Heuristic Score", f"{ov_score:.1f} / 100")
            m2.metric("Optimal Windows", f"{sqi_data['classification_breakdown']['Optimal']['count']} ({sqi_data['classification_breakdown']['Optimal']['percentage']}%)")
            m3.metric("Acceptable Windows", f"{sqi_data['classification_breakdown']['Acceptable']['count']} ({sqi_data['classification_breakdown']['Acceptable']['percentage']}%)")
            m4.metric("Noisy Windows", f"{sqi_data['classification_breakdown']['Noisy']['count']} ({sqi_data['classification_breakdown']['Noisy']['percentage']}%)")
            m5.metric("Unreliable Windows", f"{sqi_data['classification_breakdown']['Unreliable / Leads-Off']['count']} ({sqi_data['classification_breakdown']['Unreliable / Leads-Off']['percentage']}%)")

            # Color-Coded Timeline Strip
            st.markdown("#### Quality Timeline Strip")
            timeline_colors = {
                "Optimal": "#00c853",
                "Acceptable": "#ffd600",
                "Noisy": "#ff9100",
                "Unreliable / Leads-Off": "#d50000",
            }
            fig_timeline = go.Figure()
            for s in sqi_data["segments"]:
                fig_timeline.add_trace(go.Bar(
                    x=[s["duration_sec"]],
                    y=["Quality"],
                    base=s["start_time_sec"],
                    orientation="h",
                    name=s["classification"],
                    marker=dict(color=timeline_colors.get(s["classification"], "#888")),
                    hovertemplate=f"Window {s['segment_index']+1}: {s['start_time_sec']:.1f}s - {s['end_time_sec']:.1f}s<br>Status: {s['classification']}<br>Score: {s['heuristic_score']}<br>Issues: {'; '.join(s['flagged_issues'])}<extra></extra>",
                    showlegend=False,
                ))
            fig_timeline.update_layout(
                barmode="stack",
                xaxis=dict(title="Time (seconds)", range=[0, current_record.duration_sec]),
                yaxis=dict(showticklabels=False),
                template="plotly_dark",
                height=110,
                margin=dict(l=20, r=20, t=10, b=30),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Segment Table
            st.markdown("#### Segment-by-Segment Metrics Table")
            df_segments = create_segment_quality_dataframe(sqi_data)
            st.dataframe(df_segments, use_container_width=True, height=240)

            # Detected Global Anomalies
            anom = sqi_data.get("global_anomalies", {})
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            col_g1.metric("Flatline Episodes", anom.get("flatline_segments_count", 0))
            col_g2.metric("Missing (NaN) Samples", anom.get("missing_samples_count", 0))
            col_g3.metric("Amplifier Saturations", anom.get("saturation_count", 0))
            col_g4.metric("Abrupt Voltage Spikes", anom.get("spike_count", 0))

        # Educational Explanations of SQI Metrics
        st.markdown("---")
        st.markdown("#### 📖 SQI Metric Guide & Engineering Rationale")
        col_sqi_exp1, col_sqi_exp2 = st.columns(2)
        with col_sqi_exp1:
            st.markdown("""
            - **`bas_sqi` (Baseline Wander Ratio):** Ratio of power in $0.0 - 0.5\\text{ Hz}$ relative to total spectral power. Measures respiration and patient movement. Threshold $< 0.25$ for clean ECG.
            - **`hf_sqi` (High-Frequency Noise Ratio):** Ratio of spectral power above $40.0\\text{ Hz}$ (up to Nyquist). Quantifies motor-unit electromyographic (EMG) muscle tremor. Threshold $< 0.20$ for clean trace.
            - **`pl_sqi` (Powerline Interference Ratio):** Concentrated power near $50\\text{ Hz}$ or $60\\text{ Hz}$. Indicates poor electrode-skin impedance or ungrounded surroundings.
            """)
        with col_sqi_exp2:
            st.markdown("""
            - **`kurtosis_sqi` (kSQI):** 4th standardized statistical moment. Because ventricular depolarization creates tall, narrow, infrequent QRS spikes, clean ECG is highly leptokurtic ($> 3.5$). Flatline or Gaussian noise produces kurtosis near or below $3.0$.
            - **Flatline Detection:** Identifies sequences where consecutive point-to-point voltage variation is $< 10^{-4}\\text{ mV}$ for $\\ge 0.5\\text{ s}$, flagging electrode detachment or disconnected leads.
            - **Amplifier Saturation:** Samples exceeding $\\pm 5.0\\text{ mV}$ flag ADC clipping or hardware rail saturation.
            """)

    # ==================== TAB 6: BEAT EVALUATION ====================
    with tab_eval:
        st.subheader("QRS Detection (Pan-Tompkins) & ANSI/AAMI EC57 Evaluation")

        with st.spinner("Running Pan-Tompkins QRS peak detection..."):
            det_samples, det_sec, intermediates = detect_qrs_pan_tompkins(
                analysis_sig,
                fs=fs,
            )

        heart_rate_est = (len(det_sec) / current_record.duration_sec) * 60.0 if current_record.duration_sec > 0 else 0.0

        b_kpi1, b_kpi2, b_kpi3 = st.columns(3)
        b_kpi1.metric("Detected QRS Complexes", f"{len(det_sec)} beats")
        b_kpi2.metric("Estimated Heart Rate", f"{heart_rate_est:.1f} BPM")
        b_kpi3.metric("Detector Algorithm", "Pan-Tompkins (Zero-Phase)")

        # Waveform with Detected R-peaks
        disp_dur = min(10.0, current_record.duration_sec)
        disp_samples = int(round(disp_dur * fs))
        fig_beats = plot_ecg_interactive(
            current_record.time_vector[:disp_samples],
            analysis_sig[:disp_samples],
            title=f"Detected R-Peaks (First {disp_dur:.1f} s)",
            color="#00e5ff",
            r_peaks_sec=[t for t in det_sec if t <= disp_dur],
            reference_r_peaks_sec=[t for t in current_record.reference_r_peaks_sec if t <= disp_dur] if current_record.has_reference_annotations else None,
            height=380,
        )
        st.plotly_chart(fig_beats, use_container_width=True)

        st.markdown("---")
        st.markdown("### ANSI/AAMI EC57 Quantitative Evaluation")

        beat_metrics = None
        if current_record.has_reference_annotations:
            tol_slider = st.slider("Matching Tolerance Window (ms)", min_value=50.0, max_value=300.0, value=150.0, step=10.0, help="ANSI/AAMI EC57 standard defines 150 ms tolerance window.")
            beat_metrics = evaluate_beat_detection(
                current_record.reference_r_peaks_sec,
                det_sec,
                tolerance_sec=tol_slider / 1000.0,
                max_duration_sec=current_record.duration_sec,
                annotation_source=f"{current_record.source_type} ({current_record.record_name})",
            )

            c_tp, c_fn, c_fp, c_se, c_ppv, c_f1 = st.columns(6)
            c_tp.metric("True Positives (TP)", f"{beat_metrics['true_positives']}")
            c_fn.metric("False Negatives (FN)", f"{beat_metrics['false_negatives']}")
            c_fp.metric("False Positives (FP)", f"{beat_metrics['false_positives']}")

            se_str = f"{beat_metrics['sensitivity_percentage']}%" if beat_metrics['sensitivity_percentage'] is not None else "N/A"
            ppv_str = f"{beat_metrics['ppv_percentage']}%" if beat_metrics['ppv_percentage'] is not None else "N/A"
            f1_str = f"{beat_metrics['f1_percentage']}%" if beat_metrics['f1_percentage'] is not None else "N/A"

            c_se.metric("Sensitivity (Se)", se_str)
            c_ppv.metric("PPV", ppv_str)
            c_f1.metric("F1 Score", f1_str)

            col_eval_left, col_eval_right = st.columns(2)
            with col_eval_left:
                st.markdown("#### Evaluation Statistics")
                st.write(f"- **Matching Standard:** `{beat_metrics['metadata']['standard']}`")
                st.write(f"- **Tolerance Window:** `{beat_metrics['tolerance_ms']:.1f} ms`")
                st.write(f"- **Reference Beat Count:** `{beat_metrics['reference_beats_count']}`")
                st.write(f"- **Detected Beat Count:** `{beat_metrics['detected_beats_count']}`")
                if beat_metrics["mean_jitter_ms"] is not None:
                    st.write(f"- **Mean Timing Jitter:** `{beat_metrics['mean_jitter_ms']:.2f} ms` (Max: `{beat_metrics['max_jitter_ms']:.2f} ms`)")

            with col_eval_right:
                st.markdown("#### Exclusions & Sanitation")
                ref_excl = beat_metrics["exclusions"]["reference"]
                det_excl = beat_metrics["exclusions"]["detected"]
                st.write(f"- **Reference Exclusions:** Non-finite: {ref_excl['non_finite']}, Out-of-bounds: {ref_excl['out_of_bounds']}, Duplicates: {ref_excl['duplicates']}")
                st.write(f"- **Detection Exclusions:** Non-finite: {det_excl['non_finite']}, Out-of-bounds: {det_excl['out_of_bounds']}, Duplicates: {det_excl['duplicates']}")

        else:
            st.info(
                """
                **ℹ️ Beat-Detection Evaluation Status: Not Available (Unannotated Recording)**
                
                Quantitative sensitivity and positive predictive value (PPV) metrics strictly require **ground-truth reference beat annotations** (such as physician annotations from the PhysioNet MIT-BIH Arrhythmia Database).
                
                Because this loaded signal contains no reference annotations, diagnostic performance metrics are intentionally reported as **Not Available** rather than misleadingly reporting zero or fabricated figures.
                """
            )

    # ==================== TAB 7: REPORTS & EXPORT ====================
    with tab_report:
        st.subheader("Clinical-Grade Documentation & Report Exports")
        st.write("Generate and download comprehensive analysis documentation in PDF, CSV, JSON, and Markdown formats:")

        filter_meta = {
            "lowcut": filter_lowcut if enable_filter else None,
            "highcut": filter_highcut if enable_filter else None,
            "order": filter_order if enable_filter else None,
            "notch_enabled": enable_notch if enable_filter else False,
            "notch_freq": notch_freq if enable_filter and enable_notch else None,
            "notch_q": DEFAULT_NOTCH_Q,
        }

        record_meta = {
            "record_name": current_record.record_name,
            "lead": current_record.lead_name,
            "fs": fs,
            "duration_sec": current_record.duration_sec,
            "source_type": current_record.source_type,
            "sample_count": len(raw_signal),
        }

        # 1. PDF Report Generation
        with st.spinner("Compiling publication-quality PDF report..."):
            try:
                pdf_report_bytes = generate_pdf_report(
                    record_meta=record_meta,
                    status_assessment=status_assessment,
                    filter_settings=filter_meta,
                    sqi_results=sqi_data,
                    raw_signal=raw_signal,
                    filtered_signal=filtered_signal if enable_filter else None,
                    time_vector=current_record.time_vector,
                    detected_r_peaks_sec=det_sec,
                    beat_metrics=beat_metrics if current_record.has_reference_annotations else None,
                    app_version=APP_VERSION,
                )
            except Exception as pdf_err:
                pdf_report_bytes = None
                st.error(f"Error compiling PDF: {pdf_err}")

        # 2. JSON Report Generation
        json_report_str = generate_analysis_report_json(
            record_info=record_meta,
            filter_params=filter_meta,
            sqi_results=sqi_data,
            beat_metrics=beat_metrics if current_record.has_reference_annotations else None,
            status_assessment=status_assessment,
        )

        # 3. CSV Segment Export
        df_export = create_segment_quality_dataframe(sqi_data)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()

        # 4. Markdown Summary
        md_report_str = generate_markdown_summary(
            record_name=current_record.record_name,
            lead_name=current_record.lead_name,
            fs=fs,
            duration_sec=current_record.duration_sec,
            sqi_results=sqi_data,
            beat_metrics=beat_metrics if current_record.has_reference_annotations else None,
            filter_settings=filter_meta,
            status_assessment=status_assessment,
        )

        # Download Buttons in 3 columns
        col_d_pdf, col_d_csv, col_d_json = st.columns(3)
        with col_d_pdf:
            if pdf_report_bytes:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_report_bytes,
                    file_name=f"ecg_quality_report_{current_record.record_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Multi-page publication-quality PDF report including charts, executive summary, SQI table, and disclaimers.",
                )
            else:
                st.button("📄 PDF Generation Unavailable", disabled=True, use_container_width=True)

        with col_d_csv:
            st.download_button(
                label="📊 Download CSV Table",
                data=csv_str,
                file_name=f"ecg_quality_segments_{current_record.record_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Window-by-window numerical SQI scores and artifact flags in CSV format.",
            )

        with col_d_json:
            st.download_button(
                label="💾 Download JSON Data",
                data=json_report_str,
                file_name=f"ecg_analysis_{current_record.record_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
                help="Machine-readable structured JSON report with provenance metadata and evaluation results.",
            )

        st.markdown("---")
        st.markdown("### Markdown Summary Preview")
        st.markdown(md_report_str)


if __name__ == "__main__":
    main()
