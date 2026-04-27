# EEG-Based-Cognitive-State-Classification-and-Cognitive-Load-Estimation-Using-Spectral-Analysis
EEG-Based Cognitive State Classification and Cognitive Load Estimation Using Spectral Analysis
PhysioNet EEGMMIDB | GradientBoosting | Hugging Face Deployment
https://ranjith445-eeg-cognitive-analyzer.hf.space
1. Project Overview
This project implements an end-to-end EEG signal analysis pipeline that classifies brain cognitive states (Relaxed vs Focused) and estimates cognitive load (Low vs High) from raw EEG signals using spectral band power features. The system is deployed as an interactive web application on Hugging Face Spaces with biological insight visualizations, scalp topographic maps in the PDF report, and downloadable PDF reports.
2. Key Metrics
Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)
Training Subjects: 29 subjects (S01 to S29)
Test Subjects: 5 completely unseen subjects (S30 to S34)
Total Epochs: 5,346 epochs
Features: 45 (Theta/Alpha/Beta band powers + ratios per channel)
Model: GradientBoosting Classifier
CV Accuracy: 76.7% (5-fold cross-validation)
Deployment: Docker on Hugging Face Spaces
3. Pipeline
Step 1 — Data Collection: PhysioNet EEGMMIDB dataset downloaded via MNE-Python auto-download. 64-channel EEG recorded at 160 Hz. Rest runs (1-2) labeled as Relaxed/Low Load. Motor imagery runs (3-4) labeled as Focused/High Load.
Step 2 — Preprocessing: Bandpass filter 1.0–40.0 Hz using Hamming window FIR filter. Notch filter at 50 Hz. Standard 10-05 montage applied.
Step 3 — Feature Extraction: Sliding window epochs of 2 seconds. Welch method PSD computed per channel. Features from 9 key channels: Fz, C3, Cz, C4, Pz, F3, F4, O1, O2. Features: Theta/Alpha/Beta power, Alpha/Beta ratio, Theta/Alpha ratio.
Step 4 — Model Training: Three classifiers compared — RandomForest, SVM-RBF, GradientBoosting. GradientBoosting selected with 76.7% CV accuracy. 5-fold stratified cross-validation.
Step 5 — Deployment: Streamlit + Docker on Hugging Face. Pre-extracted NPZ files for 5 unseen subjects. PDF report with charts, topomaps, and reference values.
4. Biological Insight Engine
Theta (4-8 Hz): Memory and Fatigue
Alpha (8-13 Hz): Relaxation and idle rhythm
Beta (13-30 Hz): Active thinking and motor control
Alpha/Beta > 1.0 → Relaxed | Alpha/Beta < 1.0 → Focused
Theta/Alpha > 0.6 → Cognitive load | Theta/Alpha < 0.6 → Alert
5. Application Features
5 unseen test subjects (S30-S34), EEG waveform visualization, Brain State + Cognitive Load prediction, Biological insight cards, Ratio markers, Brain region analysis, Reference values table, PDF report download.
6. Technology Stack
Signal Processing: MNE-Python, SciPy | ML: Scikit-learn | Viz: Matplotlib | App: Streamlit | PDF: ReportLab | Deploy: Docker, Hugging Face
7. Evaluation Strategy
Training: S01–S29 | Validation: 5-fold CV | Test: S30–S34 (unseen)
76.7% CV accuracy with correct predictions on unseen subjects confirms cross-subject generalization.
8. Links
Live App: https://ranjith445-eeg-cognitive-analyzer.hf.space
HF Space: https://huggingface.co/spaces/Ranjith445/eeg-cognitive-analyzer
Dataset: https://physionet.org/content/eegmmidb/1.0.0
