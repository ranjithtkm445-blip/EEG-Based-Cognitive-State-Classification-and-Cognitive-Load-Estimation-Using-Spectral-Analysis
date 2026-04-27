# step5_prepare_samples.py
# Purpose: Pre-generate EEG plots for 5 UNTRAINED subjects (30-34)
# These subjects were never seen during model training (trained on 1-29)

import os
import json
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

SAMPLES_DIR = "D:/cognitive/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Subjects 30-34 — completely unseen during training
SUBJECTS    = [30, 31, 32, 33, 34]
RUN         = [3]   # Motor imagery run
KEY_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz']
PLOT_SECS   = 10
LOW_FREQ    = 1.0
HIGH_FREQ   = 40.0

print("=" * 60)
print("STEP 5A: Preparing 5 Untrained Subject EEG Samples")
print("=" * 60)
print("Subjects: 30, 31, 32, 33, 34 (never seen during training)\n")

sample_meta = []

for subject in SUBJECTS:
    print(f"--- Subject {subject} ---")
    try:
        fnames = eegbci.load_data(subject, RUN, verbose=False)
        raw    = read_raw_edf(fnames[0], preload=True, verbose=False)

        # Standardize + filter
        mne.datasets.eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
        raw.filter(LOW_FREQ, HIGH_FREQ, fir_window='hamming', verbose=False)

        sfreq     = raw.info['sfreq']
        available = [ch for ch in KEY_CHANNELS if ch in raw.ch_names]
        n_samples = int(PLOT_SECS * sfreq)

        data, times = raw.copy().pick(available).get_data(return_times=True)
        data  = data[:, :n_samples] * 1e6
        times = times[:n_samples]

        # ── Plot ──
        band_colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
        fig, axes   = plt.subplots(len(available), 1,
                                   figsize=(14, 8), sharex=True)
        fig.patch.set_facecolor('#fafafa')
        fig.suptitle(
            f"Subject {subject} — Motor Imagery EEG  |  Unseen Test Subject",
            fontsize=13, fontweight='bold', color='#1a1a2e'
        )

        for i, (ax, ch) in enumerate(zip(axes, available)):
            ax.plot(times, data[i], linewidth=0.75,
                    color=band_colors[i % len(band_colors)], alpha=0.9)
            ax.set_ylabel(f"{ch}\n(µV)", fontsize=9, color='#333')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f4f6fb' if i % 2 == 0 else '#ffffff')

            # Annotate min/max
            ax.annotate(f"max={data[i].max():.0f}µV",
                        xy=(times[-1]*0.92, data[i].max()),
                        fontsize=7, color='gray')

        axes[-1].set_xlabel("Time (seconds)", fontsize=10)

        # Add channel legend
        for i, (ch, c) in enumerate(zip(available, band_colors)):
            fig.text(0.91 + (i*0.0), 0.88 - i*0.07, f"● {ch}",
                     color=c, fontsize=8, transform=fig.transFigure)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        img_path = os.path.join(SAMPLES_DIR, f"subject_{subject}_eeg.png")
        plt.savefig(img_path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Channels : {available}")
        print(f"  Duration : {raw.times[-1]:.1f}s  |  Sfreq: {sfreq} Hz")
        print(f"  [SAVED]  : {img_path}")

        sample_meta.append({
            'subject':  subject,
            'label':    f"Subject {subject} (Unseen)",
            'img_path': img_path,
            'edf_path': str(fnames[0]),
            'channels': available,
            'sfreq':    float(sfreq),
            'duration': float(raw.times[-1]),
            'run':      'Motor Imagery (Run 3)',
            'trained':  False
        })

    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

# Save metadata
meta_path = os.path.join(SAMPLES_DIR, "sample_meta.json")
with open(meta_path, 'w') as f:
    json.dump(sample_meta, f, indent=2)
print(f"\n[SAVED] {meta_path}")

print("\n" + "=" * 60)
print("STEP 5A COMPLETE")
print("=" * 60)
print(f"  Samples generated : {len(sample_meta)}/5")
print(f"  Subjects          : {[m['subject'] for m in sample_meta]}")
print(f"  Saved to          : {SAMPLES_DIR}")
print("\nReady — run: streamlit run app.py")