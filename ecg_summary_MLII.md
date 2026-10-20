# ECG Signal Quality & Noise Analysis Report
**Generated:** 2026-10-11 13:25:41 UTC

---
## 1. Recording Metadata
- **Record Name:** `PhysioNet mitdb/100`
- **Channel / Lead:** `MLII`
- **Sampling Frequency:** `360.0 Hz`
- **Recording Duration:** `30.00 s` (10800 samples)

## 2. Digital Preprocessing Settings
- **Bandpass Filter:** 0.5 Hz to 40.0 Hz (Order 3, Zero-Phase Butterworth)
- **Notch Filter:** True (50.0 Hz, Q=30.0)
- **Phase Distortion:** Zero (Bidirectional `filtfilt` forward-backward filtering)

## 3. Signal Quality Summary (SQI)
- **Overall Heuristic SQI Score:** `95.0 / 100`
- **Window Duration:** `3.0 s`
- **Total Windows Evaluated:** `10`

### Classification Breakdown:
  - **Optimal:** 10 segments (100.0%)
  - **Acceptable:** 0 segments (0.0%)
  - **Noisy:** 0 segments (0.0%)
  - **Unreliable / Leads-Off:** 0 segments (0.0%)

### Detected Global Anomalies:
  - Flatline intervals: `0`
  - Missing (NaN/Inf) samples: `0`
  - Amplifier saturation instances: `0`
  - Abrupt spikes: `0`

## 4. Quantitative Beat Detection Evaluation
- **Standard:** `ANSI/AAMI EC57 (1-to-1 greedy matching)`
- **Matching Tolerance:** `150.0 ms`
- **Reference Beats:** `38`
- **Detected Beats:** `37`
- **True Positives (TP):** `37`
- **False Negatives (FN):** `1`
- **False Positives (FP):** `0`
- **Sensitivity (Se):** `97.37%` (37/38)
- **Positive Predictive Value (PPV):** `100.0%` (37/37)
- **F1 Score:** `98.67%`

## 5. Regulatory & Clinical Limitations
> [!WARNING]
> RESEARCH & EDUCATIONAL USE ONLY. This software is designed for biomedical engineering education, signal analysis experimentation, software testing, and portfolio demonstration. It is NOT a medical device, is NOT approved by the FDA/CE, and is NOT clinically validated for diagnostic, clinical, or therapeutic use. Never use these findings to guide patient care.

> [!NOTE]
> The Signal Quality Index (SQI) and classification thresholds presented here are heuristic engineering indicators derived from spectral, statistical, and amplitude boundaries. They do NOT represent clinically validated diagnostic safety metrics.