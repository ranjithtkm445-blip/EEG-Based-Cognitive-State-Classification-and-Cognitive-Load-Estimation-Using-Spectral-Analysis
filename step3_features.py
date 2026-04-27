# step3_features.py
# Purpose: Extract EEG features using correct run-based labeling strategy.
# Uses dedicated rest runs vs motor imagery runs for clean label separation.

import os
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

SUBJECTS   = list(range(1, 30))  # 29 subjects
LOW_FREQ   = 1.0
HIGH_FREQ  = 40.0
EPOCH_LEN  = 2.0   # seconds per epoch (non-overlapping windows)

# Correct run mapping from PhysioNet documentation:
# Run 1,2   = baseline (eyes open / eyes closed) = REST
# Run 3,4   = motor imagery left/right fist      = FOCUSED
REST_RUNS   = [1, 2]
MOTOR_RUNS  = [3, 4]

OUTPUT_DIR  = "D:/cognitive/outputs/step3"
DATA_DIR    = "D:/cognitive/data/features"
EPOCHS_DIR  = "D:/cognitive/data/epochs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EPOCHS_DIR, exist_ok=True)

KEY_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'F3', 'F4', 'O1', 'O2']

BANDS = {
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def band_power(signal, sfreq, fmin, fmax):
    freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq))
    return np.mean(psd[np.logical_and(freqs >= fmin, freqs <= fmax)])

def extract_features(segment, sfreq, available_channels):
    """Extract band powers + ratios from a single segment (n_channels x n_times)."""
    feats = []
    for ch_idx in range(len(available_channels)):
        sig = segment[ch_idx]
        bp  = {b: band_power(sig, sfreq, lo, hi) for b, (lo, hi) in BANDS.items()}
        feats += [bp['theta'], bp['alpha'], bp['beta']]
        feats.append(bp['alpha'] / (bp['beta']  + 1e-10))  # alpha/beta ratio
        feats.append(bp['theta'] / (bp['alpha'] + 1e-10))  # theta/alpha ratio
    return np.array(feats)

def load_and_filter(subject, runs):
    """Load EDF files for a subject+runs, filter, standardize."""
    fnames = eegbci.load_data(subject, runs, verbose=False)
    raws   = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw    = concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_window='hamming', verbose=False)
    return raw

def sliding_epochs(raw, available, epoch_len, label, sfreq):
    """Cut continuous signal into fixed-length non-overlapping epochs."""
    data      = raw.copy().pick(available).get_data()
    n_samples = int(epoch_len * sfreq)
    n_epochs  = data.shape[1] // n_samples
    X_sub, y_sub = [], []
    for i in range(n_epochs):
        seg = data[:, i*n_samples:(i+1)*n_samples]
        X_sub.append(extract_features(seg, sfreq, available))
        y_sub.append(label)
    return np.array(X_sub), np.array(y_sub)

# ─────────────────────────────────────────
# STEP 3A: Build Dataset
# ─────────────────────────────────────────

print("=" * 60)
print("STEP 3: Feature Extraction (Fixed Label Strategy)")
print("=" * 60)

X_all, y_state_all, y_load_all = [], [], []
valid_subjects = 0

for subject in SUBJECTS:
    try:
        # Load rest runs
        raw_rest  = load_and_filter(subject, REST_RUNS)
        # Load motor imagery runs
        raw_motor = load_and_filter(subject, MOTOR_RUNS)

        sfreq     = raw_rest.info['sfreq']
        available = [ch for ch in KEY_CHANNELS if ch in raw_rest.ch_names]

        # Slice rest into 2s epochs → label 0 (Relaxed, Low Load)
        X_rest, y_rest = sliding_epochs(raw_rest,  available, EPOCH_LEN, 0, sfreq)

        # Slice motor into 2s epochs → label 1 (Focused, High Load)
        X_motor, y_motor = sliding_epochs(raw_motor, available, EPOCH_LEN, 1, sfreq)

        # Cognitive load: rest=0 (Low), motor=2 (High)
        y_load_rest  = np.zeros(len(y_rest),  dtype=int)
        y_load_motor = np.full(len(y_motor),  2, dtype=int)

        X_all.append(np.vstack([X_rest, X_motor]))
        y_state_all.append(np.concatenate([y_rest, y_motor]))
        y_load_all.append(np.concatenate([y_load_rest, y_load_motor]))

        valid_subjects += 1
        print(f"  Subject {subject:02d}: rest={len(X_rest)}, motor={len(X_motor)} epochs")

    except Exception as e:
        print(f"  Subject {subject:02d}: [SKIP] {e}")
        continue

# ─────────────────────────────────────────
# STEP 3B: Stack + Save
# ─────────────────────────────────────────

X       = np.vstack(X_all)
y_state = np.concatenate(y_state_all)
y_load  = np.concatenate(y_load_all)

print(f"\n  Total subjects : {valid_subjects}")
print(f"  Total epochs   : {len(X)}")
print(f"  Feature shape  : {X.shape}")
print(f"  Brain State    — 0:Relaxed={np.sum(y_state==0)}, 1:Focused={np.sum(y_state==1)}")
print(f"  Cogn. Load     — 0:Low={np.sum(y_load==0)}, 2:High={np.sum(y_load==2)}")

# Build feature names
feature_names = []
for ch in available:
    for b in BANDS:
        feature_names.append(f"{b}_{ch}_abs")
    feature_names.append(f"ab_ratio_{ch}")
    feature_names.append(f"ta_ratio_{ch}")

np.save(os.path.join(DATA_DIR, "X.npy"),             X)
np.save(os.path.join(DATA_DIR, "y_state.npy"),       y_state)
np.save(os.path.join(DATA_DIR, "y_load.npy"),        y_load)
np.save(os.path.join(DATA_DIR, "feature_names.npy"), np.array(feature_names))

print(f"\n  [SAVED] X.npy            shape={X.shape}")
print(f"  [SAVED] y_state.npy      shape={y_state.shape}")
print(f"  [SAVED] y_load.npy       shape={y_load.shape}")

# ─────────────────────────────────────────
# STEP 3C: Band Power Plot
# ─────────────────────────────────────────

print("\n--- Saving band power comparison plot ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Band Power: Relaxed vs Focused (Channel C3)", fontsize=13)

c3_idx   = available.index('C3') if 'C3' in available else 0
band_idx = {'theta': c3_idx*5+0, 'alpha': c3_idx*5+1, 'beta': c3_idx*5+2}

for b_idx, (band, feat_i) in enumerate(band_idx.items()):
    ax     = axes[b_idx]
    rest_v = X[y_state == 0, feat_i]
    mot_v  = X[y_state == 1, feat_i]
    ax.bar(['Relaxed', 'Focused'],
           [rest_v.mean(), mot_v.mean()],
           yerr=[rest_v.std(), mot_v.std()],
           color=['steelblue', 'tomato'], alpha=0.8,
           capsize=5, edgecolor='black')
    ax.set_title(f"{band.capitalize()} Power")
    ax.set_ylabel("Power (V²/Hz)")
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "band_power_comparison.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  [SAVED] {plot_path}")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"  Subjects  : {valid_subjects}")
print(f"  Epochs    : {len(X)}")
print(f"  Features  : {X.shape[1]}")
print("\nReady for Step 4: Model Training")