# step6_extract_samples.py
# Purpose: Pre-extract features + raw signal for subjects 30-34
# so Hugging Face app never needs to download EDF files

import os
import json
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from scipy.signal import welch

SAMPLES_DIR  = "D:/cognitive/samples"
KEY_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'F3', 'F4', 'O1', 'O2']
BANDS        = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
LOW_FREQ     = 1.0
HIGH_FREQ    = 40.0
EPOCH_LEN    = 2.0
SUBJECTS     = [30, 31, 32, 33, 34]

def band_power(signal, sfreq, fmin, fmax):
    freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq))
    return float(np.mean(psd[np.logical_and(freqs >= fmin, freqs <= fmax)]))

print("=" * 60)
print("STEP 6: Pre-extracting features for deployment subjects")
print("=" * 60)

for subject in SUBJECTS:
    print(f"\n--- Subject {subject} ---")

    fnames = eegbci.load_data(subject, [3], verbose=False)
    raw    = read_raw_edf(str(fnames[0]), preload=True, verbose=False)

    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_window='hamming', verbose=False)

    available = [ch for ch in KEY_CHANNELS if ch in raw.ch_names]
    sfreq     = raw.info['sfreq']
    n_samples = int(EPOCH_LEN * sfreq)

    # Extract raw signal (first 10 seconds, 5 channels for waveform plot)
    plot_chans = available[:5]
    data_plot, times = raw.copy().pick(plot_chans).get_data(return_times=True)
    n_plot    = int(10 * sfreq)
    data_plot = data_plot[:, :n_plot] * 1e6  # to microvolts
    times     = times[:n_plot]

    # Extract features
    data_full = raw.copy().pick(available).get_data()
    n_epochs  = data_full.shape[1] // n_samples

    all_feats, all_bands = [], []
    for i in range(n_epochs):
        seg   = data_full[:, i*n_samples:(i+1)*n_samples]
        feats = []
        bvals = {}
        for ch_idx, ch in enumerate(available):
            sig = seg[ch_idx]
            bp  = {b: band_power(sig, sfreq, lo, hi) for b, (lo, hi) in BANDS.items()}
            bvals[ch] = bp
            feats += [bp['theta'], bp['alpha'], bp['beta']]
            feats.append(bp['alpha'] / (bp['beta']  + 1e-10))
            feats.append(bp['theta'] / (bp['alpha'] + 1e-10))
        all_feats.append(feats)
        all_bands.append(bvals)

    X = np.array(all_feats)

    # Save everything as npz
    save_path = os.path.join(SAMPLES_DIR, f"subject_{subject}_data.npz")
    np.savez(save_path,
             X=X,
             data_plot=data_plot,
             times=times,
             available=np.array(available),
             sfreq=np.array([sfreq]),
             all_bands=np.array([json.dumps(all_bands)])
             )

    print(f"  Epochs    : {len(X)}")
    print(f"  Features  : {X.shape}")
    print(f"  Channels  : {available}")
    print(f"  [SAVED]   : {save_path}")

print("\n" + "=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)
print("Upload samples/subject_3X_data.npz files to Hugging Face")