# ECG Signal Quality Analyzer — Beginner's Code Walkthrough

> **Welcome to Biomedical Signal Processing!**
> This guide explains how this ECG analysis tool works, designed especially for beginners, biomedical engineering students, and developers. Every concept is explained in simple language with clear scientific definitions in brackets `[like this]`.

---

## Table of Contents
1. [What is an ECG Signal?](#1-what-is-an-ecg-signal)
2. [Why Signal Quality Matters](#2-why-signal-quality-matters)
3. [The 5-Step Workflow Overview](#3-the-5-step-workflow-overview)
4. [Application Architecture Diagram](#4-application-architecture-diagram)
5. [Tour of the Python Modules & Important Functions](#5-tour-of-the-python-modules--important-functions)
   - `app.py` — The User Interface
   - `src/ecg_analyzer/constants.py` — Physiological Settings & Disclaimers
   - `src/ecg_analyzer/data/loader.py` — Loading Files & Datasets
   - `src/ecg_analyzer/data/validator.py` — Checking Input Signals
   - `src/ecg_analyzer/data/synthetic.py` — Generating Simulated Signals & Noise
   - `src/ecg_analyzer/filters/bandpass.py` — Removing Drift and High-Frequency Noise
   - `src/ecg_analyzer/filters/notch.py` — Removing Electrical Hum
   - `src/ecg_analyzer/filters/frequency_analysis.py` — Frequency Spectrum & SNR
   - `src/ecg_analyzer/sqi/indices.py` — Calculating Quality Numbers [SQI]
   - `src/ecg_analyzer/sqi/segmentation.py` — 3-Second Window Analysis
   - `src/ecg_analyzer/sqi/status.py` — 4-State Quality Verdict Engine
   - `src/ecg_analyzer/detection/qrs_detector.py` — Detecting Heartbeats
   - `src/ecg_analyzer/metrics/evaluation.py` — Measuring Accuracy Against Standards
   - `src/ecg_analyzer/reporting/exporter.py` — CSV and JSON Exporters
   - `src/ecg_analyzer/reporting/pdf_report.py` — PDF Report Generator
6. [Digital Filters Explained in Plain English](#6-digital-filters-explained-in-plain-english)
7. [Signal Quality Metrics and SNR Explained](#7-signal-quality-metrics-and-snr-explained)
8. [Heartbeat Detection & Accuracy Benchmarking](#8-heartbeat-detection--accuracy-benchmarking)
9. [How the User Interface Connects to the Code](#9-how-the-user-interface-connects-to-the-code)
10. [How Reports Are Generated (PDF, CSV, JSON)](#10-how-reports-are-generated-pdf-csv-json)
11. [How to Install, Run, Test, and Troubleshoot](#11-how-to-install-run-test-and-troubleshoot)
12. [Biomedical Engineering Glossary](#12-biomedical-engineering-glossary)

---

## 1. What is an ECG Signal?

An **electrocardiogram (ECG)** is a recording of the electrical activity of the heart over time.

Each time the heart beats, an electrical wave spreads through the heart muscle, making it contract and pump blood. We measure this voltage in **millivolts (mV)** using sensors (electrodes) placed on the skin.

A single normal heartbeat looks like this:

```
        R (Ventricular Contraction — Main Pump)
       / \
      /   \
     /     \
 P  /       \        T (Recovery Wave)
_/\_         \      _/\_
    \       / \____/
     \     /
      Q   S
```

- **P Wave**: The top chambers of the heart (atria) contract to fill the main chambers.
- **QRS Complex**: The main pumping chambers (ventricles) contract with a strong electrical burst. The tall sharp spike is the **R-peak**.
- **T Wave**: The heart relaxes and recharges electrically before the next beat.
- **Baseline**: The flat electrical line between heartbeats when the heart is resting.

---

## 2. Why Signal Quality Matters

Automated computer algorithms — such as ICU heart monitors, automated external defibrillators (AEDs), and smartwatches — rely on clean ECG signals.

When an ECG has noise, computers can make serious mistakes:
- **Baseline Drift [Low-Frequency Noise]**: Patient breathing or chest movement makes the signal wander up and down. This can look like heart attack patterns (ST-elevation) to a computer.
- **Muscle Tremor [EMG High-Frequency Noise]**: Patient shivering or tense muscles produces rapid electrical spikes that look like extra heartbeats.
- **Electrical Hum [Powerline Interference]**: Power cables radiate 50 Hz or 60 Hz hum, blurring the fine details of the heartbeat.
- **Loose Electrodes [Flatline Dropouts]**: If a sensor detaches, the signal goes flat or spikes violently.

**Signal Quality Analysis** checks whether the signal is clear enough to trust *before* calculating heart rate or running diagnostic algorithms.

---

## 3. The 5-Step Workflow Overview

The application organizes your analysis into 5 obvious, sequential steps:

1. **Step 1: Load ECG**: Select a pre-loaded sample dataset, upload your own CSV file, download from the clinical PhysioNet repository, or create a simulated signal with added noise.
2. **Step 2: View ECG**: Inspect the raw waveform on a large, interactive graph with clearly explained time and voltage axes.
3. **Step 3: Filter Signal**: Apply high-pass, low-pass, and notch filters to clean the signal. Compare raw and filtered signals side-by-side.
4. **Step 4: Analyze Quality**: Check the 4-state quality rating (Good / Needs Review / Poor / Unable to Assess), review detected issues, and see detected heartbeats.
5. **Step 5: Download Report**: Save a complete vector PDF report, CSV spreadsheet table, or JSON data file.

---

## 4. Application Architecture Diagram

```
                 ┌────────────────────────────────────────┐
                 │  Step 1: Load ECG                      │
                 │  • Sample Datasets (Clean/Noisy/Spike) │
                 │  • Custom CSV / TXT Upload             │
                 │  • PhysioNet MIT-BIH Clinical Records  │
                 │  • Simulated Signal Sandbox            │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────┐
                 │  Step 2: View ECG                      │
                 │  • Interactive Plotly Waveform Graph   │
                 │  • Time (s) and Voltage (mV) axes      │
                 │  • Signal Statistics (Peak-to-Peak)    │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────┐
                 │  Step 3: Filter Signal                 │
                 │  • Baseline Drift Filter (High-Pass)   │
                 │  • Muscle Noise Filter (Low-Pass)      │
                 │  • Electrical Hum Filter (Mains Notch) │
                 │  • Zero-Phase Bidirectional (filtfilt) │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────┐
                 │  Step 4: Analyze Quality               │
                 │  • 4-State Rating (Good / Review/ Poor)│
                 │  • Noise Metrics (bas_sqi, hf_sqi, pl) │
                 │  • Pan-Tompkins Heartbeat Detection    │
                 │  • ANSI/AAMI EC57 Accuracy Benchmark   │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────┐
                 │  Step 5: Download Report               │
                 │  • Multi-Page Vector PDF (ReportLab)   │
                 │  • Segment Spreadsheet Table (CSV)     │
                 │  • Complete Structured Data (JSON)     │
                 └────────────────────────────────────────┘
```

---

## 5. Tour of the Python Modules & Important Functions

### `app.py` — The User Interface
- **What it does**: The central Streamlit application script. It displays the controls, draws the interactive graphs, and manages the 5 workflow tabs.
- **Important Functions**:
  - `get_plot_layout(**overrides)`: Merges default layout settings with custom options while preventing duplicate keyword arguments (such as duplicate `margin`).
  - `plot_ecg_interactive(...)`: Draws large Plotly graphs with labeled axes (`Time (s)`, `Voltage (mV)`), zoom/pan tools, and detected heartbeat markers.
  - `render_provenance_banner(record)`: Displays the record name, speed, duration, and whether it is real patient data or simulated.
  - `render_quality_status_card(assessment)`: Shows the 4-state rating card with plain-English reasons and next steps.
  - `main()`: The entry point that executes when you run `streamlit run app.py`.

### `src/ecg_analyzer/constants.py` — Physiological Settings
- **What it does**: Holds standard default values:
  - `DEFAULT_FS_MITBIH = 360.0`: Standard 360 samples per second for MIT-BIH recordings.
  - `DEFAULT_BANDPASS_LOWCUT = 0.5`: High-pass cutoff to remove breathing drift.
  - `DEFAULT_BANDPASS_HIGHCUT = 40.0`: Low-pass cutoff to remove muscle tremor.
  - `CLINICAL_DISCLAIMER`: Required notice stating this is educational software, not a medical device.

### `src/ecg_analyzer/data/loader.py` — Loading Files & Datasets
- **`ECGDataRecord`**: A data container holding the voltage array, sampling speed, channel name, and doctor annotations.
- **`load_ecg_from_csv(file, fs, ...)`**: Reads numbers from a CSV or TXT file and returns an `ECGDataRecord`.
- **`load_physionet_record(record_id, ...)`**: Uses the `wfdb` library to download real patient records and cardiologist beat labels from PhysioNet.

### `src/ecg_analyzer/data/validator.py` — Checking Input Signals
- **`validate_signal_array(signal)`**: Verifies that the input data contains valid numbers and is not empty.
- **`validate_sampling_frequency(fs)`**: Ensures the recording speed is positive and realistic (between 20 Hz and 10,000 Hz).
- **`assess_signal_properties(signal, fs)`**: Calculates min, max, duration, and checks if the signal is all zeros (disconnected sensor) or flat.

### `src/ecg_analyzer/data/synthetic.py` — Simulated Signals & Noise
- **`generate_clean_synthetic_ecg(duration_sec, fs, heart_rate_bpm)`**: Uses mathematical Gaussian equations to create artificial P-QRS-T heartbeats.
- Noise injection functions:
  - `add_baseline_wander()`: Adds slow sinusoidal drift.
  - `add_powerline_noise()`: Adds 50 Hz or 60 Hz hum.
  - `add_muscle_noise()`: Adds random high-frequency muscle tremor.
  - `add_flatline_segment()`: Inserts a flat line to simulate a loose sensor.

### `src/ecg_analyzer/filters/bandpass.py` — Removing Drift and Noise
- **`apply_bandpass_filter(signal, fs, lowcut, highcut, order)`**:
  - Calculates Butterworth filter equations (`scipy.signal.butter`).
  - Applies zero-phase filtering (`scipy.signal.filtfilt`) so that peak timestamps do not shift in time.

### `src/ecg_analyzer/filters/notch.py` — Removing Electrical Hum
- **`apply_notch_filter(signal, fs, notch_freq, quality_factor)`**:
  - Uses `scipy.signal.iirnotch` to surgically cut out 50 Hz or 60 Hz electrical hum without affecting nearby heart frequencies.

### `src/ecg_analyzer/filters/frequency_analysis.py` — Frequency Content & SNR
- **`compute_power_spectral_density(signal, fs)`**: Uses Welch's method to calculate how much power exists at each frequency.
- **`estimate_snr_db(raw, filtered)`**: Calculates Signal-to-Noise Ratio (SNR) in decibels ($dB$) by comparing the filtered signal to the removed noise.

### `src/ecg_analyzer/sqi/indices.py` — Calculating Quality Numbers
- **`compute_spectral_sqi(signal, fs)`**:
  - `bas_sqi`: Percentage of energy below 0.5 Hz (baseline drift).
  - `hf_sqi`: Percentage of energy above 40 Hz (muscle noise).
  - `pl_sqi`: Percentage of energy near 50/60 Hz (mains hum).
- **`compute_statistical_sqi(signal)`**:
  - `kurtosis_sqi`: Measures how tall and sharp the R-peaks are compared to the baseline.

### `src/ecg_analyzer/sqi/segmentation.py` — 3-Second Window Analysis
- **`analyze_signal_quality(signal, fs, window_length_sec=3.0)`**: Splits the recording into 3-second blocks and computes quality scores for each block.

### `src/ecg_analyzer/sqi/status.py` — 4-State Quality Verdict
- **`assess_overall_signal_quality(raw_signal, fs, sqi_results, snr_db)`**:
  - Evaluates the signal using rule-based decision logic.
  - Returns `GOOD`, `NEEDS REVIEW`, `POOR`, or `UNABLE TO ASSESS`.
  - Missing metrics are **never** treated as passing.

### `src/ecg_analyzer/detection/qrs_detector.py` — Detecting Heartbeats
- **`detect_qrs_pan_tompkins(signal, fs)`**:
  - Implements the classic Pan-Tompkins algorithm:
    1. Filters 5–15 Hz to isolate R-peaks.
    2. Differentiates to measure steep slopes.
    3. Squares the values to emphasize large peaks.
    4. Smooths with moving-window integration.
    5. Applies adaptive thresholds to locate heartbeat timestamps.

### `src/ecg_analyzer/metrics/evaluation.py` — Accuracy Benchmarking
- **`evaluate_beat_detection(reference_peaks, detected_peaks, tolerance_sec=0.150)`**:
  - Follows the ANSI/AAMI EC57 international standard.
  - Matches detected beats against doctor-verified reference beats within a $\pm 150\text{ ms}$ window.
  - Computes True Positives (TP), Missed Beats (FN), False Alarms (FP), Sensitivity (Recall), and Precision (PPV).

### `src/ecg_analyzer/reporting/pdf_report.py` — PDF Generator
- **`generate_pdf_report(...)`**:
  - Uses ReportLab and Matplotlib to assemble a vector PDF report.
  - Uses a two-pass `NumberedCanvas` to draw running headers and "Page X of Y" footers.

---

## 6. Digital Filters Explained in Plain English

### High-Pass Filter: Reduces Slow Baseline Drift
- **What it does**: Lets high frequencies pass through and blocks slow frequencies below the cutoff.
- **Why we use it**: Patient breathing causes slow up-and-down wandering (<0.5 Hz). High-pass filtering flattens the baseline.
- **Important limit**: Setting the cutoff too high (>0.67 Hz) distorts the ST segment and T-wave.

### Low-Pass Filter: Reduces High-Frequency Noise
- **What it does**: Lets slow frequencies pass through and blocks rapid noise above the cutoff.
- **Why we use it**: Skeletal muscle shivering (EMG) creates rapid jitter (>40 Hz). Low-pass filtering smooths the signal.
- **Important limit**: Setting the cutoff too low (<30 Hz) blunts the tips of sharp R-peaks.

### Notch Filter: Reduces Electrical Hum
- **What it does**: Cuts out a single specific frequency (50 Hz or 60 Hz) like a sharp notch.
- **Why we use it**: Building electrical wiring radiates 50 Hz (Europe/Asia/UK) or 60 Hz (North America) electromagnetic hum into sensor leads.

### Zero-Phase Bidirectional Filtering (`filtfilt`)
- Standard digital filters shift the signal slightly forward in time (phase delay).
- `scipy.signal.filtfilt` filters the signal forward, flips it around, and filters it backward.
- This cancels out all time delay, ensuring R-peaks remain at their exact timestamps.

---

## 7. Signal Quality Metrics and SNR Explained

| Metric | Scientific Term | Plain-English Meaning | Healthy Target |
|---|---|---|---|
| **Overall Score** | `overall_heuristic_sqi` | Summary quality score from 0 to 100 | $> 80$ is Good, $< 55$ is Poor |
| **Baseline Drift** | `bas_sqi` | Fraction of energy from breathing drift (<0.5 Hz) | $< 25\%$ |
| **Muscle Noise** | `hf_sqi` | Fraction of energy from muscle tremor (>40 Hz) | $< 20\%$ |
| **Electrical Hum** | `pl_sqi` | Fraction of energy from 50/60 Hz mains wiring | $< 15\%$ |
| **Peak Sharpness** | `kurtosis_sqi` (kSQI) | How sharply heartbeats spike above baseline | $> 3.5$ for clean ECG |
| **Flatline Duration**| `flatline_duration_sec`| Consecutive seconds with zero electrical variance | $0.0$ s (no dropouts) |
| **Voltage Clipping** | `saturation_count` | Number of samples hitting amplifier limits ($\pm 5$ mV) | $0$ events |
| **Signal Clarity** | `snr_db` | Signal-to-Noise Ratio in decibels ($dB$) | $> 15$ dB |

---

## 8. Heartbeat Detection & Accuracy Benchmarking

### What is ANSI/AAMI EC57?
ANSI/AAMI EC57 is the international biomedical engineering standard for evaluating cardiac rhythm devices.

- A detection is counted as a **True Positive (TP)** if it occurs within $\pm 150\text{ ms}$ of a doctor-labeled reference beat.
- A reference beat with no detection is a **False Negative (FN)** [Missed Beat].
- A detection with no reference beat is a **False Positive (FP)** [False Alarm].

$$\text{Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}} \times 100\%$$
$$\text{Precision (PPV)} = \frac{\text{TP}}{\text{TP} + \text{FP}} \times 100\%$$

*Note: If a recording does not have doctor labels (like an uploaded CSV), accuracy metrics are honestly shown as "Not Available" rather than guessed.*

---

## 9. How the User Interface Connects to the Code

Streamlit runs `app.py` from top to bottom whenever you click a button or change a slider:
1. `st.session_state.current_record`: Holds the active ECG data in memory between clicks.
2. Sidebar controls read your filter settings (lowcut, highcut, notch).
3. `apply_bandpass_filter()` and `apply_notch_filter()` filter the voltage array.
4. `analyze_signal_quality()` and `assess_overall_signal_quality()` calculate the quality rating.
5. The selected tab displays the interactive graphs, metrics, and download buttons.

---

## 10. How Reports Are Generated (PDF, CSV, JSON)

- **PDF Report**: `generate_pdf_report()` creates a vector document using ReportLab. Waveform plots are drawn using Matplotlib Agg into an in-memory byte buffer and embedded as high-resolution images.
- **CSV Table**: `create_segment_quality_dataframe()` converts 3-second window scores into a table and exports it as a `.csv` file.
- **JSON Data**: `generate_analysis_report_json()` packages all numbers, settings, and disclaimers into structured text.

---

## 11. How to Install, Run, Test, and Troubleshoot

### 1. Install Dependencies
Open Windows PowerShell in the project directory:
```powershell
pip install -r requirements.txt
```

### 2. Run the Application
```powershell
$env:PYTHONPATH="src"
python -m streamlit run app.py
```
*Or double-click `Run_ECG_App.bat`.*
Open your web browser to `http://localhost:8501`.

### 3. Run the Automated Tests
```powershell
python -m pytest
```
*All 86 tests should pass.*

### Troubleshooting Tips
- **Plotly layout error**: Solved using `get_plot_layout()`, which safely merges settings without duplicate `margin` parameters.
- **Navigation labels too faint**: Navigation tabs now use dark `#1e293b` text on clear background buttons.
- **ModuleNotFoundError**: Always ensure `$env:PYTHONPATH="src"` is set before running.

---

## 12. Biomedical Engineering Glossary

- **AAMI EC57**: Standard for testing cardiac rhythm algorithms.
- **Bandpass Filter**: Filter that passes frequencies within a desired band and blocks higher and lower frequencies.
- **Baseline Wander**: Slow drift (<0.5 Hz) caused by patient respiration.
- **BPM**: Beats Per Minute (heart rate).
- **ECG / EKG**: Electrocardiogram, a record of the heart's electrical voltages.
- **EMG**: Electromyogram, electrical noise from skeletal muscles.
- **filtfilt**: Zero-phase bidirectional filtering.
- **F1 Score**: Harmonic mean of sensitivity and precision.
- **Leptokurtic**: A statistical distribution with sharp peaks (Kurtosis > 3).
- **Notch Filter**: A narrow filter that removes one specific frequency (50/60 Hz mains hum).
- **Nyquist Limit**: Half of the sampling frequency ($f_s / 2$). Digital filters cannot operate at or above this frequency.
- **Pan-Tompkins**: Standard algorithm for detecting R-peaks in ECG signals.
- **PPV**: Positive Predictive Value (Precision) = True Positives / Total Detections.
- **QRS Complex**: The large electrical spike representing ventricular contraction.
- **Sensitivity (Se)**: Detection Recall = True Positives / Total Real Beats.
- **SNR**: Signal-to-Noise Ratio, measured in decibels ($dB$).
- **SQI**: Signal Quality Index, a numerical rating of signal clarity.
- **WFDB**: WaveForm DataBase, PhysioNet's format for clinical records.
