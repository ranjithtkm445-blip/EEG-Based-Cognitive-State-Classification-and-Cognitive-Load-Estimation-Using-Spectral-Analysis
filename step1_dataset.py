# step1_dataset.py
# Purpose: Download PhysioNet EEG Motor Movement/Imagery dataset using MNE
# and inspect signal structure, channels, and events.

import os
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Subjects to download (start with 3 for quick test)
SUBJECTS = [1, 2, 3]

# Runs:
# Run 1  → Eyes open (Relaxed baseline)
# Run 2  → Eyes closed (Relaxed baseline)
# Run 3  → Motor imagery task (Focused/Active)
# Run 4  → Motor execution task (Focused/Active)
RUNS = [1, 2, 3, 4]

# Output folder for plots
OUTPUT_DIR = "D:/cognitive/outputs/step1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1A: Download Data
# ─────────────────────────────────────────

print("=" * 60)
print("STEP 1: EEG Dataset Download & Inspection")
print("=" * 60)

print(f"\n[INFO] Downloading data for subjects: {SUBJECTS}")
print(f"[INFO] Runs: {RUNS}")
print("[INFO] Data will be saved to: ~/mne_data/MNE-eegbci-data/\n")

all_raws = []

for subject in SUBJECTS:
    print(f"\n--- Subject {subject} ---")

    # Auto-download EDF files for this subject and runs
    # MNE 1.12 compatible: positional arguments
    fnames = eegbci.load_data(subject, RUNS, verbose=False)

    # Load each run
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]

    # Concatenate all runs for this subject
    raw = concatenate_raws(raws)
    all_raws.append(raw)

    print(f"  Channels     : {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration     : {raw.times[-1]:.1f} seconds")
    print(f"  Total samples: {raw.n_times}")

# ─────────────────────────────────────────
# STEP 1B: Inspect First Subject
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Inspecting Subject 1 in detail")
print("=" * 60)

raw1 = all_raws[0]

# Standardize channel names (e.g., EEG 001 -> Fp1)
mne.datasets.eegbci.standardize(raw1)

# Set montage (electrode positions)
montage = mne.channels.make_standard_montage('standard_1005')
raw1.set_montage(montage, on_missing='ignore', verbose=False)

# Print first 10 channel names
print(f"\nFirst 10 channels: {raw1.ch_names[:10]}")

# ─────────────────────────────────────────
# STEP 1C: Extract and Print Events
# ─────────────────────────────────────────

print("\n--- Events ---")
events, event_id = mne.events_from_annotations(raw1, verbose=False)
print(f"Event types found: {event_id}")
print(f"Total events     : {len(events)}")

# ─────────────────────────────────────────
# STEP 1D: Plot Raw EEG (first 10 seconds)
# ─────────────────────────────────────────

print("\n--- Saving raw EEG plot (first 10 seconds) ---")

# Pick a few channels for visualization
picks = mne.pick_channels(raw1.ch_names, include=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'])

fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Raw EEG Signal - Subject 1 (First 10 seconds)", fontsize=14)

channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']
data, times = raw1[picks, :int(raw1.info['sfreq'] * 10)]

for i, ax in enumerate(axes):
    ax.plot(times, data[i] * 1e6, linewidth=0.8, color='steelblue')  # Convert to microvolts
    ax.set_ylabel(f"{channel_names[i]}\n(µV)", fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "raw_eeg_subject1.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[SAVED] {plot_path}")

# ─────────────────────────────────────────
# STEP 1E: Summary
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"  Subjects downloaded : {len(SUBJECTS)}")
print(f"  Channels per subject: {len(raw1.ch_names)}")
print(f"  Sampling rate       : {raw1.info['sfreq']} Hz")
print(f"  Event types         : {list(event_id.keys())}")
print(f"  Plot saved to       : {plot_path}")
print("\nReady for Step 2: Preprocessing")