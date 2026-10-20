# ECG Signal Quality Analyzer & Noise Testing Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests passing](https://img.shields.io/badge/tests-86%20passed-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal, research-grade biomedical engineering application designed for students, researchers, and developers. It provides physiological ECG signal validation, digital filtering, 4-state heuristic quality scoring, Pan-Tompkins heartbeat detection, and ANSI/AAMI EC57 accuracy benchmarking.

---

## Visual Interface & Design Direction

The user interface follows a **clean, minimal scientific research-tool aesthetic**:
- **Palette**: Clean white and light-grey background (`#f8fafc`), white panels (`#ffffff`), dark readable text (`#0f172a`), subtle grey borders (`#cbd5e1`), and a single muted teal primary accent (`#0d766e`).
- **High-Contrast Navigation Headings**: Prominent, dark navigation tabs (`#1e293b`) that are bold, tactile, and easy to read on any screen.
- **Signal Quality Ratings**: Restrained green for *Good* (`#166534`), amber for *Needs Review* (`#b45309`), deep red for *Poor* (`#b91c1c`), and neutral slate for *Unable to Assess* (`#475569`).
- **Beginner-Friendly Labels**: Plain-English labels with scientific terms in brackets `[like this]`.
- **Transparent Provenance**: Distinguishes real clinical patient data (PhysioNet MIT-BIH) from uploaded files, pre-packaged samples, and mathematical synthetic simulations.

---

## The 5-Step Workflow

```
[Step 1: Load ECG] ──► [Step 2: View ECG] ──► [Step 3: Filter Signal]
                                                        │
[Step 5: Download Report] ◄── [Step 4: Analyze Quality] ◄┘
```

1. **Step 1: Load ECG**: Choose a pre-loaded sample dataset, upload a custom CSV/TXT file, download from the PhysioNet MIT-BIH clinical database, or create a simulated signal with controlled noise.
2. **Step 2: View ECG**: Inspect the calibrated voltage waveform on a large interactive graph. Both axes are explained: Time in seconds on the horizontal axis and Voltage in millivolts (mV) on the vertical axis.
3. **Step 3: Filter Signal**: Apply zero-phase Butterworth bandpass and mains notch filters to clean the signal. Inspect raw vs. filtered waveforms and Welch Power Spectral Density (PSD).
4. **Step 4: Analyze Quality**: Check the 4-state quality verdict (Good / Needs Review / Poor / Unable to Assess), review detected physical issues, view detected heartbeats (Pan-Tompkins), and benchmark against ANSI/AAMI EC57 ground truth.
5. **Step 5: Download Report**: Download a multi-page vector PDF report, CSV spreadsheet table, or machine-readable JSON data file.

---

## Regulatory & Scientific Boundaries

- **Heuristic Indicators**: Signal Quality Index (SQI) scores are heuristic engineering metrics derived from spectral, statistical, and amplitude boundaries. They are not clinically validated diagnostic metrics.
- **Signal Quality vs. Medical Diagnosis**: A "Good" signal quality rating indicates clean electrical contact, not a healthy heart. A recording from a patient with a dangerous arrhythmia can still receive a "Good" quality rating if sensors are attached properly. This application does not diagnose clinical conditions.
- **Ground-Truth Rigour**: Sensitivity and precision require verified physician annotations; unannotated recordings report accuracy as *Not Available* rather than fabricating numbers.

---

## Quickstart & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip package manager

### 1. Clone or Download Repository
```powershell
cd c:\Users\User\OneDrive\Documents\ecg-signal-quality-analyzer
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```powershell
$env:PYTHONPATH="src"
python -m streamlit run app.py
```
*Or on Windows, double-click `Run_ECG_App.bat`.*
Open **`http://localhost:8501`** in any web browser.

---

## Running Automated Tests

Run the full pytest suite (86 unit and integration tests):

```powershell
python -m pytest
```

Run with verbose per-test breakdown:
```powershell
python -m pytest tests/ -v
```
**Test Result**: `86 passed in 5.12s` (all 86 unit and integration tests passing).

---

## Generating and Downloading Reports

1. In the application, navigate to **`Step 5: Download Report`**.
2. Click **`Download PDF Report`** to generate a multi-page vector PDF containing:
   - Header with UTC generation timestamp and regulatory notice.
   - Executive summary and color-coded 4-state quality status verdict.
   - Provenance table (record ID, lead, sampling frequency, duration, sample count, source).
   - High-resolution dual-panel waveform plots (Original ECG and Zero-Phase Filtered ECG with detected R-peaks).
   - Digital filtering settings and phase response notes.
   - Quantitative SQI metrics table (`bas_sqi`, `hf_sqi`, `pl_sqi`, `kSQI`, flatline duration, saturation events) with physiological thresholds.
   - Actionable recommendations and detected physical findings.
   - ANSI/AAMI EC57 beat detection evaluation (or explicit unannotated statement).
   - Scientific limitations and educational non-diagnostic disclaimers.
3. Download windowed metrics as **CSV** or complete structured data as **JSON**.

---

## Project Structure

```
ecg-signal-quality-analyzer/
├── app.py                          # Streamlit application with 5-step beginner workflow
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration (pythonpath = src)
├── CODE_WALKTHROUGH.md             # Beginner-friendly architectural & code guide
├── README.md                       # Project overview & documentation
├── Run_ECG_App.bat                 # Windows one-click batch launcher
├── data/
│   ├── samples/                    # Pre-packaged benchmark CSV datasets
│   └── physionet_cache/            # Local cache for downloaded WFDB records
├── scripts/
│   └── generate_sample_data.py     # Reproducible sample generation script
├── src/
│   └── ecg_analyzer/
│       ├── constants.py            # Central physiological thresholds & disclaimers
│       ├── data/
│       │   ├── loader.py           # CSV parsing and PhysioNet WFDB record loader
│       │   ├── synthetic.py        # Synthetic ECG generation & artifact injections
│       │   └── validator.py        # Array, sampling rate, and property validators
│       ├── filters/
│       │   ├── bandpass.py         # Zero-phase Butterworth bandpass filter
│       │   ├── notch.py            # IIR mains notch filter (50/60 Hz)
│       │   └── frequency_analysis.py # Welch PSD, filter response, SNR estimation
│       ├── detection/
│       │   └── qrs_detector.py     # Pan-Tompkins QRS peak detection
│       ├── sqi/
│       │   ├── indices.py          # bas_sqi, hf_sqi, pl_sqi, kurtosis, flatlines, clipping
│       │   ├── segmentation.py     # Sliding window segmentation & heuristic scoring
│       │   └── status.py           # 4-state standardized quality assessment engine
│       ├── metrics/
│       │   └── evaluation.py       # ANSI/AAMI EC57 beat evaluation & jitter analysis
│       └── reporting/
│           ├── exporter.py         # CSV DataFrame, JSON, and Markdown summaries
│           └── pdf_report.py       # ReportLab multi-page vector PDF generator
└── tests/                          # 86 comprehensive unit & integration tests
```
