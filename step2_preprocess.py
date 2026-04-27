# step2_preprocess.py
# Purpose: Preprocess raw EEG data - bandpass filter, epoch extraction,
# artifact rejection, and save clean epochs for feature extraction.

import os
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

SUBJECTS         = [1, 2, 3]
RUNS             = [1, 2, 3, 4]

# Bandpass filter range (Hz)
LOW_FREQ         = 0.5
HIGH_FREQ        = 40.0

# Epoch window around each event (seconds)
TMIN             = -0.5   # 0.5s before event
TMAX             = 2.0    # 2.0s after event

# Artifact rejection threshold - raised to be less strict
REJECT_THRESHOLD = 500e-6  # 500 µV

OUTPUT_DIR       = "D:/cognitive/outputs/step2"
EPOCHS_DIR       = "D:/cognitive/data/epochs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EPOCHS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# STEP 2A: Load + Filter + Epoch
# ─────────────────────────────────────────

print("=" * 60)
print("STEP 2: EEG Preprocessing")
print("=" * 60)

all_epochs = []

for subject in SUBJECTS:
    print(f"\n--- Processing Subject {subject} ---")

    # Load raw files
    fnames = eegbci.load_data(subject, RUNS, verbose=False)
    raws   = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw    = concatenate_raws(raws)

    # Standardize channel names
    mne.datasets.eegbci.standardize(raw)

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore', verbose=False)

    # Bandpass Filter
    print(f"  [1] Bandpass filtering: {LOW_FREQ}-{HIGH_FREQ} Hz")
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_window='hamming', verbose=False)

    # Notch Filter (remove 50 Hz powerline noise)
    print(f"  [2] Notch filter: 50 Hz")
    raw.notch_filter(50.0, verbose=False)

    # Extract Events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"  [3] Events found: {event_id}")

    # Get actual key names sorted by value (T0, T1, T2)
    keys = sorted(event_id.keys(), key=lambda k: event_id[k])

    if len(keys) < 3:
        print(f"  [SKIP] Not enough event types, skipping subject {subject}")
        continue

    event_id_clean = {
        'rest':       event_id[keys[0]],  # T0
        'left_hand':  event_id[keys[1]],  # T1
        'right_hand': event_id[keys[2]],  # T2
    }

    # Create Epochs
    print(f"  [4] Creating epochs: {TMIN}s to {TMAX}s")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_clean,
        tmin=TMIN,
        tmax=TMAX,
        baseline=(None, 0),
        reject=dict(eeg=REJECT_THRESHOLD),
        preload=True,
        verbose=False
    )

    epochs.drop_bad(verbose=False)
    print(f"  [5] Clean epochs : {len(epochs)}")

    if len(epochs) == 0:
        print(f"  [SKIP] No valid epochs for subject {subject}")
        continue

    print(f"  [6] Epoch shape  : {epochs.get_data().shape}")
    all_epochs.append(epochs)

    save_path = os.path.join(EPOCHS_DIR, f"subject{subject:02d}-epo.fif")
    epochs.save(save_path, overwrite=True, verbose=False)
    print(f"  [SAVED] {save_path}")

# ─────────────────────────────────────────
# STEP 2B: Combine all subjects
# ─────────────────────────────────────────

print("\n--- Combining all subjects ---")

if len(all_epochs) == 0:
    print("[ERROR] No valid epochs found across any subject!")
    print("        Try increasing REJECT_THRESHOLD further.")
    exit(1)

epochs_all = mne.concatenate_epochs(all_epochs, verbose=False)
print(f"  Total epochs (all subjects): {len(epochs_all)}")
print(f"  Combined shape             : {epochs_all.get_data().shape}")
print(f"  Labels                     : {epochs_all.event_id}")

combined_path = os.path.join(EPOCHS_DIR, "all_subjects-epo.fif")
epochs_all.save(combined_path, overwrite=True, verbose=False)
print(f"  [SAVED] {combined_path}")

# ─────────────────────────────────────────
# STEP 2C: Plot epoch comparison
# ─────────────────────────────────────────

print("\n--- Saving epoch comparison plot ---")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("EEG Epoch Comparison: Rest vs Motor Imagery (Channel C3)", fontsize=13)

ch_idx = epochs_all.ch_names.index('C3')
times  = epochs_all.times

rest_data  = epochs_all['rest'].get_data()[:, ch_idx, :] * 1e6

motor_list = []
if 'left_hand' in epochs_all.event_id:
    motor_list.append(epochs_all['left_hand'])
if 'right_hand' in epochs_all.event_id:
    motor_list.append(epochs_all['right_hand'])
motor_data = mne.concatenate_epochs(motor_list).get_data()[:, ch_idx, :] * 1e6

def plot_mean_std(ax, times, data, color, label):
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    ax.plot(times, mean, color=color, linewidth=2, label=label)
    ax.fill_between(times, mean - std, mean + std, alpha=0.2, color=color)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Event onset')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Amplitude (uV)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_mean_std(axes[0], times, rest_data,  'steelblue', f'Rest (n={len(rest_data)})')
plot_mean_std(axes[1], times, motor_data, 'tomato',    f'Motor Imagery (n={len(motor_data)})')
axes[1].set_xlabel("Time (seconds)")
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "epoch_comparison.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[SAVED] {plot_path}")

# ─────────────────────────────────────────
# STEP 2D: Summary
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
print(f"  Subjects processed  : {len(all_epochs)}/{len(SUBJECTS)}")
print(f"  Total clean epochs  : {len(epochs_all)}")
print(f"  Epoch shape         : {epochs_all.get_data().shape}")
print(f"  Sampling rate       : {epochs_all.info['sfreq']} Hz")
print(f"  Epoch window        : {TMIN}s to {TMAX}s")
print(f"  Epochs saved to     : {EPOCHS_DIR}")
print(f"  Plot saved to       : {plot_path}")
print("\nReady for Step 3: Feature Extraction")