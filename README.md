

---

# Brain State and Cognitive Load Detection using EEG

**Live App:** [https://ranjith445-eeg-cognitive-analyzer.hf.space](https://ranjith445-eeg-cognitive-analyzer.hf.space)

---

## What is this project about?

The human brain produces electrical signals.
These signals can be recorded using a test called **EEG (Electroencephalogram)**.

By studying these signals, we can understand:

* Whether a person is relaxed or focused
* How much mental effort they are using

This project builds an **AI system that analyzes brain signals and predicts mental state and cognitive load**.

---

## What does this system do?

This system:

* Takes EEG brain signals as input
* Predicts:

  * Brain state: **Relaxed or Focused**
  * Cognitive load: **Low or High**
* Shows brain activity visually
* Explains the results using simple biological rules
* Generates a downloadable report

---

## How does it work (simple explanation)

The system works step by step:

### 1. Collects brain signals

* EEG signals are recorded from the scalp
* Multiple sensors (channels) capture brain activity

---

### 2. Cleans the signals

* Removes noise and unwanted signals
* Keeps only useful brain activity

---

### 3. Extracts important patterns

The system looks at different frequency bands:

* **Theta** → related to memory and fatigue
* **Alpha** → related to relaxation
* **Beta** → related to active thinking

---

### 4. Uses these patterns to understand the brain

The system calculates ratios like:

* Alpha/Beta ratio
* Theta/Alpha ratio

These help determine:

* Relaxed vs Focused
* Low vs High mental load

---

### 5. Makes prediction using AI

A machine learning model analyzes these features and predicts:

* Brain state
* Cognitive load

---

## What data was used?

* Dataset: PhysioNet EEG Motor Movement/Imagery dataset
* EEG signals from multiple subjects

For this project:

* 29 subjects used for training
* 5 new subjects used for testing

---

## What results does it give?

* Model accuracy: **76.7%**

The model works well across different people, showing it can generalize.

---

## What makes this project special?

### Uses real brain signals

* Works on real EEG data

### Combines science and AI

* Uses biological rules + machine learning

### Easy to understand output

* Shows simple explanations
* Displays brain activity visually

### Generates reports

* Provides downloadable PDF with analysis

---

## Features of the application

* Select EEG data from test subjects
* View brain signal waveform
* Get brain state prediction
* Get cognitive load estimation
* See biological explanations
* Download PDF report

---

## Important Note

* Built for **learning and demonstration**
* Not a medical or clinical tool

---

## Limitations

* Moderate accuracy (76.7%)
* Works on limited dataset
* Not tested in real-world environments

---

## Future Improvements

* Use more subjects and data
* Improve model accuracy
* Add real-time EEG analysis

---

## Disclaimer

This project is for educational purposes only.
It should not be used for medical or psychological decisions.

---

## One-Line Summary

An AI system that reads brain signals and predicts whether a person is relaxed or focused, along with their mental workload.

---

## Author

Built by M. Ranjith Kumar as a biomedical AI portfolio project.

