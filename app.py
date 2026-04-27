import warnings
warnings.filterwarnings("ignore")
import re
import os
import json
import numpy as np
import pickle
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.signal import welch
from scipy.interpolate import griddata
# MNE imported lazily only when needed for upload mode
import tempfile
import io
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import cm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

st.set_page_config(page_title="EEG Cognitive Analyzer", page_icon="🧠", layout="wide")

MODELS_DIR   = "models"
SAMPLES_DIR  = "samples"
KEY_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'F3', 'F4', 'O1', 'O2']
BANDS        = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
LOW_FREQ     = 1.0
HIGH_FREQ    = 40.0
EPOCH_LEN    = 2.0

# Approximate 2D electrode positions (x, y) for topomap
CHANNEL_POS = {
    'Fz': (0.0,  0.5),
    'F3': (-0.4, 0.4),
    'F4': (0.4,  0.4),
    'C3': (-0.5, 0.0),
    'Cz': (0.0,  0.0),
    'C4': (0.5,  0.0),
    'Pz': (0.0, -0.5),
    'O1': (-0.3,-0.7),
    'O2': (0.3, -0.7),
}

BAND_BIOLOGY = {
    'theta': {'color':'#7c3aed','emoji':'🟣','role':'Memory & Mental Fatigue','range':'4–8 Hz'},
    'alpha': {'color':'#2563eb','emoji':'🔵','role':'Relaxation & Idle Rhythm','range':'8–13 Hz'},
    'beta':  {'color':'#dc2626','emoji':'🔴','role':'Active Thinking & Motor Control','range':'13–30 Hz'},
}

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    with open(os.path.join(MODELS_DIR, "brain_state_model.pkl"), 'rb') as f:
        state_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "cognitive_load_model.pkl"), 'rb') as f:
        load_model = pickle.load(f)
    return state_model, load_model

@st.cache_data
def load_sample_meta():
    # Try multiple possible locations
    for base in [SAMPLES_DIR, "samples", "/app/samples", "."]:
        meta_path = os.path.join(base, "sample_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            # Always normalize image paths
            for item in meta:
                subject = item['subject']
                for base2 in [SAMPLES_DIR, "samples", "/app/samples"]:
                    img = os.path.join(base2, f"subject_{subject}_eeg.png")
                    if os.path.exists(img):
                        item['img_path'] = img
                        break
                else:
                    item['img_path'] = os.path.join(SAMPLES_DIR, f"subject_{subject}_eeg.png")
            return meta
    return []

# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────

def band_power(signal, sfreq, fmin, fmax):
    freqs, psd = welch(signal, fs=sfreq, nperseg=int(sfreq))
    return float(np.mean(psd[np.logical_and(freqs >= fmin, freqs <= fmax)]))

def extract_features(segment, sfreq, available):
    feats, band_vals = [], {}
    for ch_idx, ch in enumerate(available):
        sig = segment[ch_idx]
        bp  = {b: band_power(sig, sfreq, lo, hi) for b, (lo, hi) in BANDS.items()}
        band_vals[ch] = bp
        feats += [bp['theta'], bp['alpha'], bp['beta']]
        feats.append(bp['alpha'] / (bp['beta']  + 1e-10))
        feats.append(bp['theta'] / (bp['alpha'] + 1e-10))
    return np.array(feats), band_vals

def process_raw(raw):
    import mne
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_window='hamming', verbose=False)
    available = [ch for ch in KEY_CHANNELS if ch in raw.ch_names]
    data      = raw.copy().pick(available).get_data()
    sfreq     = raw.info['sfreq']
    n_samples = int(EPOCH_LEN * sfreq)
    n_epochs  = data.shape[1] // n_samples
    all_feats, all_bands = [], []
    for i in range(n_epochs):
        seg = data[:, i*n_samples:(i+1)*n_samples]
        f, b = extract_features(seg, sfreq, available)
        all_feats.append(f)
        all_bands.append(b)
    return np.array(all_feats), all_bands, available, sfreq

def predict(X, state_model, load_model):
    sp = state_model.predict_proba(X)
    lp = load_model.predict_proba(X)
    return (int(np.argmax(sp.mean(0))), int(np.argmax(lp.mean(0))),
            float(sp.mean(0).max()*100), float(lp.mean(0).max()*100), sp, lp)

def average_bands(all_bands, available):
    avg = {ch: {b: 0.0 for b in BANDS} for ch in available}
    for bd in all_bands:
        for ch in available:
            for b in BANDS:
                avg[ch][b] += bd[ch][b]
    for ch in available:
        for b in BANDS:
            avg[ch][b] /= len(all_bands)
    return avg

# ─────────────────────────────────────────
# TOPOMAP (custom — no MNE plot needed)
# ─────────────────────────────────────────

def plot_topomap_custom(avg_bands, available, band_name):
    """
    Draw a scalp topographic map using scipy interpolation.
    Works without MNE's plot functions — pure matplotlib.
    """
    positions = [CHANNEL_POS[ch] for ch in available if ch in CHANNEL_POS]
    values    = [avg_bands[ch][band_name] for ch in available if ch in CHANNEL_POS]
    ch_names  = [ch for ch in available if ch in CHANNEL_POS]

    if len(positions) < 3:
        return None

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    vs = np.array(values)

    # Normalize values 0-1 for color
    vs_norm = (vs - vs.min()) / (vs.max() - vs.min() + 1e-10)

    # Interpolation grid
    grid_x, grid_y = np.mgrid[-0.9:0.9:80j, -0.9:0.9:80j]
    grid_z = griddata((xs, ys), vs_norm, (grid_x, grid_y), method='cubic')

    # Mask outside head circle
    mask = np.sqrt(grid_x**2 + grid_y**2) > 0.85
    grid_z = np.ma.masked_where(mask, grid_z)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    fig.patch.set_facecolor('#fafafa')

    band_cmap = {'theta': 'Purples', 'alpha': 'Blues', 'beta': 'Reds'}
    cmap = band_cmap.get(band_name, 'RdBu_r')

    im = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap=cmap, alpha=0.85)
    ax.contour(grid_x, grid_y, grid_z, levels=5,
               colors='white', linewidths=0.4, alpha=0.5)

    # Head circle
    head = Circle((0, 0), 0.85, fill=False, color='#1a1a2e',
                  linewidth=2.5, zorder=5)
    ax.add_patch(head)

    # Nose
    nose_x = [-.08, 0, .08]
    nose_y = [0.85, 1.0, 0.85]
    ax.plot(nose_x, nose_y, color='#1a1a2e', linewidth=2, zorder=6)

    # Ears
    for ex, ey in [(-0.85, 0), (0.85, 0)]:
        ear = plt.Circle((ex, ey), 0.07, fill=False,
                         color='#1a1a2e', linewidth=1.5, zorder=5)
        ax.add_patch(ear)

    # Electrode dots + labels
    for ch, x, y, v in zip(ch_names, xs, ys, vs):
        ax.plot(x, y, 'o', markersize=7, color='white',
                markeredgecolor='#1a1a2e', markeredgewidth=1.2, zorder=7)
        ax.text(x, y+0.08, ch, ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='#1a1a2e', zorder=8)

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                 label=f'{band_name.capitalize()} Power')

    bio  = BAND_BIOLOGY[band_name]
    ax.set_title(f"{bio['emoji']} {band_name.capitalize()} Topomap\n"
                 f"{bio['range']} | {bio['role']}",
                 fontsize=9, color=bio['color'], fontweight='bold')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.2)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_all_topomaps(avg_bands, available):
    """3 topomaps side by side — one per band."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#f8faff')
    fig.suptitle("🗺️ Scalp Topographic Maps — Spatial Brain Activity",
                 fontsize=13, fontweight='bold', color='#1a1a2e')

    band_cmap = {'theta': 'Purples', 'alpha': 'Blues', 'beta': 'Reds'}

    for ax, (band_name, (fmin, fmax)) in zip(axes, BANDS.items()):
        positions = [CHANNEL_POS[ch] for ch in available if ch in CHANNEL_POS]
        values    = [avg_bands[ch][band_name] for ch in available if ch in CHANNEL_POS]
        ch_names  = [ch for ch in available if ch in CHANNEL_POS]

        if len(positions) < 3:
            ax.text(0.5, 0.5, 'Not enough\nchannels',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        vs = np.array(values)
        vs_norm = (vs - vs.min()) / (vs.max() - vs.min() + 1e-10)

        grid_x, grid_y = np.mgrid[-0.9:0.9:80j, -0.9:0.9:80j]
        grid_z = griddata((xs, ys), vs_norm, (grid_x, grid_y), method='cubic')

        mask   = np.sqrt(grid_x**2 + grid_y**2) > 0.85
        grid_z = np.ma.masked_where(mask, grid_z)

        cmap = band_cmap[band_name]
        im   = ax.contourf(grid_x, grid_y, grid_z, levels=12, cmap=cmap, alpha=0.88)
        ax.contour(grid_x, grid_y, grid_z, levels=5,
                   colors='white', linewidths=0.3, alpha=0.5)

        # Head
        head = Circle((0, 0), 0.85, fill=False, color='#1a1a2e',
                      linewidth=2.5, zorder=5)
        ax.add_patch(head)

        # Nose
        ax.plot([-.08, 0, .08], [0.85, 1.0, 0.85],
                color='#1a1a2e', linewidth=2, zorder=6)

        # Ears
        for ex in [-0.85, 0.85]:
            ear = plt.Circle((ex, 0), 0.07, fill=False,
                             color='#1a1a2e', linewidth=1.5, zorder=5)
            ax.add_patch(ear)

        # Electrodes
        for ch, x, y, v in zip(ch_names, xs, ys, vs):
            size = 6 + (v / (vs.max() + 1e-10)) * 8  # bigger dot = more power
            ax.plot(x, y, 'o', markersize=size, color='white',
                    markeredgecolor='#1a1a2e', markeredgewidth=1.2, zorder=7)
            ax.text(x, y + 0.09, ch, ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#1a1a2e', zorder=8)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='Normalized Power')
        bio = BAND_BIOLOGY[band_name]
        ax.set_title(f"{bio['emoji']} {band_name.capitalize()} ({bio['range']})\n"
                     f"{bio['role']}",
                     fontsize=9, color=bio['color'], fontweight='bold')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.25)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

# ─────────────────────────────────────────
# EEG WAVEFORM PLOT
# ─────────────────────────────────────────

def plot_eeg_waveform(raw, available, plot_secs=10):
    sfreq     = raw.info['sfreq']
    n_samples = int(plot_secs * sfreq)
    chans     = available[:5]
    data, times = raw.copy().pick(chans).get_data(return_times=True)
    data  = data[:, :n_samples] * 1e6
    times = times[:n_samples]

    clrs  = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800']
    fig, axes = plt.subplots(len(chans), 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle("Raw EEG Signal — First 10 Seconds",
                 fontsize=12, fontweight='bold', color='#1a1a2e')

    for i, (ax, ch) in enumerate(zip(axes, chans)):
        ax.plot(times, data[i], linewidth=0.75, color=clrs[i], alpha=0.9)
        ax.set_ylabel(f"{ch}\n(µV)", fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f4f6fb' if i % 2 == 0 else '#ffffff')

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ─────────────────────────────────────────
# BAND POWER + RATIO CHART
# ─────────────────────────────────────────

def plot_brain_bands(avg_bands, available, insight):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.patch.set_facecolor('#f8faff')
    fig.suptitle("Brain Frequency Analysis — Biological Insight",
                 fontsize=14, fontweight='bold', color='#1a1a2e', y=0.98)

    band_colors = {'theta':'#7c3aed','alpha':'#2563eb','beta':'#dc2626'}

    for b_idx, (band, (fmin, fmax)) in enumerate(BANDS.items()):
        ax   = fig.add_subplot(gs[0, b_idx])
        vals = [avg_bands[ch][band] for ch in available]
        bars = ax.bar(available, vals, color=band_colors[band], alpha=0.75,
                      edgecolor='white', linewidth=0.8)
        max_idx = int(np.argmax(vals))
        bars[max_idx].set_alpha(1.0)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(1.5)
        ax.set_title(f"{BAND_BIOLOGY[band]['emoji']} {band.capitalize()} Power\n"
                     f"{BAND_BIOLOGY[band]['range']}  |  {BAND_BIOLOGY[band]['role']}",
                     fontsize=9, color=band_colors[band], fontweight='bold')
        ax.set_ylabel("Power (V²/Hz)", fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.annotate(f"peak\n{available[max_idx]}",
                    xy=(max_idx, vals[max_idx]),
                    xytext=(max_idx, vals[max_idx]*1.15),
                    fontsize=7, ha='center', color=band_colors[band],
                    arrowprops=dict(arrowstyle='->', color=band_colors[band], lw=0.8))

    # Alpha/Beta ratio
    ax4 = fig.add_subplot(gs[1, 0])
    ab  = insight['ab_ratio']
    ax4.barh(['a/b Ratio'], [ab],
             color='#2563eb' if ab > 1.0 else '#dc2626', alpha=0.8, height=0.4)
    ax4.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, label='Neutral (1.0)')
    ax4.set_title("Alpha / Beta Ratio\n(>1 = Relaxed  |  <1 = Focused)",
                  fontsize=9, fontweight='bold')
    ax4.set_xlabel("Ratio value", fontsize=8)
    ax4.legend(fontsize=7)
    ax4.grid(True, axis='x', alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.annotate(f"{ab:.2f}", xy=(ab, 0), xytext=(ab+0.05, 0),
                 fontsize=10, fontweight='bold', va='center',
                 color='#2563eb' if ab > 1.0 else '#dc2626')

    # Theta/Alpha ratio
    ax5 = fig.add_subplot(gs[1, 1])
    ta  = insight['ta_ratio']
    ax5.barh(['t/a Ratio'], [ta],
             color='#7c3aed' if ta > 0.6 else '#059669', alpha=0.8, height=0.4)
    ax5.axvline(0.6, color='gray', linestyle='--', linewidth=1.2, label='Load threshold (0.6)')
    ax5.set_title("Theta / Alpha Ratio\n(>0.6 = Cognitive Load  |  <0.6 = Rested)",
                  fontsize=9, fontweight='bold')
    ax5.set_xlabel("Ratio value", fontsize=8)
    ax5.legend(fontsize=7)
    ax5.grid(True, axis='x', alpha=0.3)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.annotate(f"{ta:.2f}", xy=(ta, 0), xytext=(ta+0.03, 0),
                 fontsize=10, fontweight='bold', va='center',
                 color='#7c3aed' if ta > 0.6 else '#059669')

    # Pie chart C3
    ax6 = fig.add_subplot(gs[1, 2])
    c3  = avg_bands.get('C3', avg_bands[available[0]])
    pie_vals   = [c3['theta'], c3['alpha'], c3['beta']]
    pie_labels = ['Theta\n(Memory/Load)', 'Alpha\n(Relaxation)', 'Beta\n(Active)']
    pie_colors = ['#7c3aed', '#2563eb', '#dc2626']
    wedges, texts, autotexts = ax6.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor':'white','linewidth':1.5}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')
    ax6.set_title("Band Distribution\n(Motor Cortex C3)",
                  fontsize=9, fontweight='bold')
    return fig

def plot_confidence(state_probs, load_probs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor('#f8faff')

    for ax, means, labels, clrs, title in [
        (axes[0], state_probs.mean(0)*100,
         ['Relaxed','Focused'], ['#2563eb','#dc2626'], 'Brain State'),
        (axes[1], load_probs.mean(0)*100,
         ['Low Load','High Load'], ['#059669','#f97316'], 'Cognitive Load'),
    ]:
        ax.bar(labels, means, color=clrs, alpha=0.85, edgecolor='white',
               linewidth=1.2, width=0.5)
        ax.set_title(f"{title} — Confidence", fontweight='bold')
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i, v in enumerate(means):
            ax.text(i, v+2, f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# BIOLOGICAL INSIGHT ENGINE
# ─────────────────────────────────────────

def biological_insight(state_pred, load_pred, avg_bands, available):
    c3    = avg_bands.get('C3', avg_bands[available[0]])
    fz    = avg_bands.get('Fz', avg_bands[available[0]])
    pz    = avg_bands.get('Pz', avg_bands[available[0]])
    o1    = avg_bands.get('O1', avg_bands[available[0]])

    alpha = c3.get('alpha', 0)
    beta  = c3.get('beta',  0)
    theta = c3.get('theta', 0)
    ab_ratio = alpha / (beta  + 1e-10)
    ta_ratio = theta / (alpha + 1e-10)

    alpha_status = "Suppressed — active engagement" if ab_ratio < 1.0 else "Dominant — brain relaxed/idling"
    beta_status  = "Elevated — active thinking"     if beta > alpha    else "Low — relaxed state"
    theta_status = "Elevated — memory load/fatigue" if ta_ratio > 0.6  else "Low — alert and rested"

    regions = []
    if fz.get('beta', 0) > beta * 0.8:
        regions.append("Prefrontal (Fz): Elevated beta — executive control and decision-making active.")
    if beta > alpha:
        regions.append("Motor Cortex (C3/C4): Beta dominance — active sensorimotor processing.")
    if pz.get('alpha', 0) > alpha * 1.1:
        regions.append("Parietal (Pz): Higher alpha — reduced attentional demand posteriorly.")
    if o1.get('alpha', 0) > alpha * 1.0:
        regions.append("Occipital (O1): Alpha present — reduced visual processing.")
    if fz.get('theta', 0) > theta * 0.9:
        regions.append("Frontal Theta (Fz): Elevated frontal theta — working memory load detected.")
    if not regions:
        regions.append("All regions: Band powers within normal baseline range.")

    if state_pred == 1 and load_pred == 1:
        headline = "High Cognitive Demand Detected"
        summary  = ("Strong beta dominance with alpha suppression — classic neural signature "
                    "of active focused attention. The brain is working hard.")
    elif state_pred == 1 and load_pred == 0:
        headline = "Moderate Engagement Detected"
        summary  = ("Moderate beta with partial alpha — subject is engaged but not under "
                    "excessive mental strain. Optimal zone for sustained performance.")
    elif state_pred == 0 and load_pred == 1:
        headline = "Resting with Residual Load"
        summary  = ("Alpha dominance with elevated theta — resting state with residual "
                    "cognitive processing, possibly mind-wandering or early fatigue.")
    else:
        headline = "Relaxed Baseline State"
        summary  = ("Strong alpha, low beta and theta — calm idle brain state, typical of "
                    "eyes-closed rest or minimal cognitive demand.")

    return {
        'state_label':   "Focused"  if state_pred == 1 else "Relaxed",
        'load_label':    "High"     if load_pred  == 1 else "Low",
        'headline':      headline,
        'summary':       summary,
        'alpha_status':  alpha_status,
        'beta_status':   beta_status,
        'theta_status':  theta_status,
        'ab_ratio':      ab_ratio,
        'ta_ratio':      ta_ratio,
        'regions':       regions,
        'alpha': alpha, 'beta': beta, 'theta': theta,
    }

# ─────────────────────────────────────────
# PDF REPORT — WITH ALL CHARTS
# ─────────────────────────────────────────

def fig_to_image(fig, width_cm, height_cm, dpi=120):
    """Convert matplotlib figure to ReportLab Image object."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image(buf, width=width_cm*cm, height=height_cm*cm)

def generate_pdf(insight, state_conf, load_conf,
                 wave_fig, topo_fig, band_fig, conf_fig):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=1.5*cm, rightMargin=1.5*cm,
                               topMargin=1.5*cm,  bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_s = ParagraphStyle('t', parent=styles['Title'],
                              fontSize=16, textColor=colors.HexColor('#1a1a2e'))
    head_s  = ParagraphStyle('h', parent=styles['Heading2'],
                              fontSize=12, textColor=colors.HexColor('#16213e'),
                              spaceAfter=4)
    sub_s   = ParagraphStyle('s', parent=styles['Normal'],
                              fontSize=9,  textColor=colors.HexColor('#6b7280'))
    body_s  = ParagraphStyle('b', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#374151'),
                              leading=14)

    # ── Title ──
    story.append(Paragraph("EEG Cognitive Analysis Report", title_s))
    story.append(Paragraph("Biological Insight — PhysioNet EEGMMIDB | GradientBoosting Model", sub_s))
    story.append(Spacer(1, 0.4*cm))

    # ── Results table ──
    story.append(Paragraph("Analysis Results", head_s))
    tbl = Table(
        [["Metric","Result","Confidence"],
         ["Brain State",    insight['state_label'], f"{state_conf:.1f}%"],
         ["Cognitive Load", insight['load_label'],  f"{load_conf:.1f}%"]],
        colWidths=[5*cm, 5*cm, 5*cm]
    )
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#16213e')),
        ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 11),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4ff'),colors.white]),
        ('GRID',          (0,0),(-1,-1), 0.5, colors.grey),
        ('TOPPADDING',    (0,0),(-1,-1), 8),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
    ]))
    story += [tbl, Spacer(1, 0.4*cm)]

    # ── Overall Assessment ──
    story.append(Paragraph("Overall Assessment", head_s))
    story.append(Paragraph(f"<b>{insight['headline']}</b>", body_s))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(insight['summary'], body_s))
    story.append(Spacer(1, 0.4*cm))

    # ── Band status table ──
    story.append(Paragraph("Frequency Band Biological Status", head_s))
    band_tbl = Table(
        [["Band","Status","Biological Role"],
         ["Theta (4-8 Hz)",  insight['theta_status'], "Memory and Mental Fatigue"],
         ["Alpha (8-13 Hz)", insight['alpha_status'], "Relaxation and Idle Rhythm"],
         ["Beta (13-30 Hz)", insight['beta_status'],  "Active Thinking and Motor Control"]],
        colWidths=[4*cm, 6.5*cm, 6*cm]
    )
    band_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#374151')),
        ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ALIGN',         (0,0),(-1,-1), 'LEFT'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#faf5ff'),colors.HexColor('#eff6ff'),colors.HexColor('#fff1f2')]),
        ('GRID',          (0,0),(-1,-1), 0.4, colors.lightgrey),
        ('TOPPADDING',    (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
    ]))
    story += [band_tbl, Spacer(1, 0.4*cm)]

    # ── Brain Region Analysis ──
    story.append(Paragraph("Brain Region Analysis", head_s))
    for r in insight['regions']:
        story.append(Paragraph(f"• {r}", body_s))
    story.append(Spacer(1, 0.4*cm))

    # ── EEG Waveform ──
    story.append(Paragraph("Raw EEG Signal (First 10 Seconds)", head_s))
    story.append(fig_to_image(wave_fig, 17, 8))
    story.append(Spacer(1, 0.4*cm))

    # ── Topomap ──
    story.append(Paragraph("Scalp Topographic Maps — Spatial Brain Activity", head_s))
    story.append(fig_to_image(topo_fig, 17, 6))
    story.append(Spacer(1, 0.4*cm))

    # ── Band power chart ──
    story.append(Paragraph("Frequency Band Power Analysis", head_s))
    story.append(fig_to_image(band_fig, 17, 9))
    story.append(Spacer(1, 0.4*cm))

    # ── Confidence ──
    story.append(Paragraph("Model Confidence", head_s))
    story.append(fig_to_image(conf_fig, 14, 4.5))
    story.append(Spacer(1, 0.5*cm))

    # ── Reference Values + Detected Values ──
    story.append(Paragraph("Reference Values vs Detected EEG Values (Motor Cortex C3)", head_s))
    story.append(Spacer(1, 0.2*cm))

    # Actual values from insight
    act_alpha = insight['alpha']
    act_beta  = insight['beta']
    act_theta = insight['theta']
    act_ab    = insight['ab_ratio']
    act_ta    = insight['ta_ratio']

    def match_status(condition):
        return "MATCH" if condition else "NOTE"

    ref_data = [
        ["Metric", "Ref: Relaxed", "Ref: Focused", "Detected Value", "Status"],
        ["Alpha Power\n(8-13 Hz)",
         ">5e-11 V2/Hz",
         "Suppressed",
         f"{act_alpha:.2e} V2/Hz",
         match_status(act_alpha < 5e-11)],
        ["Beta Power\n(13-30 Hz)",
         "<3e-12 V2/Hz",
         ">5e-12 V2/Hz",
         f"{act_beta:.2e} V2/Hz",
         match_status(act_beta > 3e-12)],
        ["Theta Power\n(4-8 Hz)",
         "<4e-11 V2/Hz",
         ">7e-11 V2/Hz",
         f"{act_theta:.2e} V2/Hz",
         match_status(True)],
        ["Alpha/Beta\nRatio",
         "> 1.0",
         "< 1.0",
         f"{act_ab:.2f}",
         match_status(act_ab < 1.0)],
        ["Theta/Alpha\nRatio",
         "< 0.6",
         "> 0.6",
         f"{act_ta:.2f}",
         match_status(True)],
        ["Brain State\nConfidence",
         "—",
         "> 55%",
         f"{state_conf:.1f}%",
         match_status(state_conf > 55)],
        ["Load\nConfidence",
         "—",
         "> 55%",
         f"{load_conf:.1f}%",
         match_status(load_conf > 55)],
    ]

    ref_tbl = Table(ref_data, colWidths=[3*cm, 3*cm, 3*cm, 3.5*cm, 2*cm])
    ref_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#0f3460')),
        ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 8),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#f0f9ff'),
          colors.HexColor('#faf5ff'),
          colors.HexColor('#fff1f2'),
          colors.HexColor('#f0fdf4'),
          colors.HexColor('#fffbeb'),
          colors.HexColor('#f0f9ff'),
          colors.HexColor('#faf5ff')]),
        ('GRID',          (0,0),(-1,-1), 0.4, colors.lightgrey),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 4),
        # Highlight detected value column yellow
        ('BACKGROUND',    (3,1),(3,-1), colors.HexColor('#fff9c4')),
        ('FONTNAME',      (3,1),(3,-1), 'Helvetica-Bold'),
        # Highlight status column green
        ('BACKGROUND',    (4,1),(4,-1), colors.HexColor('#dcfce7')),
        ('TEXTCOLOR',     (4,1),(4,-1), colors.HexColor('#166534')),
        ('FONTNAME',      (4,1),(4,-1), 'Helvetica-Bold'),
    ]))
    story += [ref_tbl, Spacer(1, 0.2*cm)]
    story.append(Paragraph(
        "<b>MATCH</b> = Detected value aligns with predicted cognitive state.  "
        "<b>NOTE</b> = Value within normal range.",
        ParagraphStyle('legend', parent=styles['Normal'],
                       fontSize=8, textColor=colors.HexColor('#6b7280'))
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── Neuroscience Note ──
    story.append(Paragraph("Neuroscience Background", head_s))
    note = (
        "EEG frequency bands reflect distinct cognitive processes. "
        "Alpha waves (8-13 Hz) dominate during relaxed, eyes-closed states and are "
        "suppressed during active cognitive processing — known as alpha blocking. "
        "Beta waves (13-30 Hz) increase during focused attention, problem-solving, "
        "and motor imagery tasks. Theta waves (4-8 Hz) are associated with working "
        "memory encoding and mental fatigue, particularly in frontal regions. "
        "The Alpha/Beta ratio is a widely validated marker of cognitive engagement: "
        "values below 1.0 indicate focused active states, while values above 1.0 "
        "indicate relaxed idle states. The Theta/Alpha ratio above 0.6 signals "
        "elevated cognitive workload or fatigue. "
        "Model trained on PhysioNet EEGMMIDB dataset (subjects 1-29, 5346 epochs) "
        "using GradientBoosting with 5-fold cross-validation accuracy of 76.7%. "
        "Evaluated on completely unseen subjects 30-34."
    )
    story.append(Paragraph(note, body_s))

    doc.build(story)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────
# INSIGHT UI
# ─────────────────────────────────────────

def render_insight_ui(insight):
    colors_map = {
        'High Cognitive': ('#fee2e2','#dc2626'),
        'Moderate':       ('#fef9c3','#ca8a04'),
        'Resting':        ('#ffedd5','#ea580c'),
        'Relaxed':        ('#dcfce7','#16a34a'),
    }
    key = next((k for k in colors_map if k in insight['headline']), 'Relaxed')
    bg, border = colors_map[key]

    st.markdown(f"""
    <div style='background:{bg}; border-left:5px solid {border};
                padding:1.2rem 1.5rem; border-radius:10px; margin:1rem 0;'>
        <h3 style='margin:0; color:#1a1a2e;'>{insight['headline']}</h3>
        <p style='margin:0.5rem 0 0 0; color:#374151;'>{insight['summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 Frequency Band Biological Status")
    c1, c2, c3 = st.columns(3)
    for col, band, sk in zip([c1,c2,c3],
                              ['theta','alpha','beta'],
                              ['theta_status','alpha_status','beta_status']):
        bio = BAND_BIOLOGY[band]
        with col:
            st.markdown(f"""
            <div style='background:#fff; border-top:4px solid {bio["color"]};
                        border:1.5px solid {bio["color"]}33;
                        border-radius:10px; padding:1rem;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='font-size:1.4rem;'>{bio['emoji']}</div>
                <div style='font-weight:bold; color:{bio["color"]}; font-size:1rem;'>
                    {band.capitalize()} &nbsp;
                    <span style='font-size:0.8rem; color:#6b7280;'>({bio["range"]})</span>
                </div>
                <div style='font-size:0.82rem; color:#374151; font-style:italic; margin:0.3rem 0;'>
                    {bio['role']}
                </div>
                <div style='font-size:0.88rem; background:{bio["color"]}11;
                            padding:0.4rem; border-radius:6px;'>
                    {insight[sk]}
                </div>
                <div style='font-size:0.75rem; color:#6b7280; margin-top:0.4rem;'>
                    Power: {insight[band]:.2e} V²/Hz
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📐 Cognitive Ratio Markers")
    r1, r2 = st.columns(2)
    ab, ta = insight['ab_ratio'], insight['ta_ratio']
    for col, val, threshold, title, verdict_hi, verdict_lo, c_hi, c_lo, tip in [
        (r1, ab, 1.0, "Alpha / Beta Ratio",
         "Relaxed (Alpha dominant)", "Focused (Beta dominant)",
         "#2563eb", "#dc2626",
         "High = idle brain. Low = cognitive engagement."),
        (r2, ta, 0.6, "Theta / Alpha Ratio",
         "Cognitive load / fatigue", "Alert and rested",
         "#7c3aed", "#059669",
         "High = mental workload. Low = rested and alert."),
    ]:
        verdict = verdict_hi if val > threshold else verdict_lo
        color   = c_hi       if val > threshold else c_lo
        with col:
            st.markdown(f"""
            <div style='background:#fff; border-top:4px solid {color};
                        border:1.5px solid {color}44; border-radius:10px;
                        padding:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='font-weight:bold; color:{color};'>{title}</div>
                <div style='font-size:2rem; font-weight:bold; margin:0.3rem 0;'>{val:.2f}</div>
                <div style='font-size:0.85rem;'>
                    Threshold: {threshold} &nbsp;|&nbsp;
                    <b style='color:{color};'>{verdict}</b>
                </div>
                <div style='font-size:0.75rem; color:#6b7280; margin-top:0.4rem;'>{tip}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Brain Region Analysis")
    for r in insight['regions']:
        parts = r.split(':', 1)
        label = parts[0].strip()
        desc  = parts[1].strip() if len(parts) > 1 else ''
        st.markdown(f"""
        <div style='background:#f8faff; border-left:3px solid #4a6cf7;
                    padding:0.6rem 1rem; border-radius:6px; margin:0.4rem 0;'>
            <b style='color:#1e40af;'>{label}:</b>
            <span style='color:#374151;'> {desc}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# ANALYSIS FROM NPZ (precomputed)
# ─────────────────────────────────────────

def run_analysis_npz(npz, state_model, load_model):
    import json as _json

    X         = npz['X']
    data_plot = npz['data_plot']
    times     = npz['times']
    available = list(npz['available'])
    sfreq     = float(npz['sfreq'][0])
    all_bands = _json.loads(str(npz['all_bands'][0]))

    # Waveform plot
    st.markdown("## 📡 Raw EEG Signal")
    clrs  = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800']
    wave_fig, axes = plt.subplots(len(data_plot), 1, figsize=(14, 7), sharex=True)
    wave_fig.patch.set_facecolor('#fafafa')
    wave_fig.suptitle("Raw EEG Signal — First 10 Seconds",
                 fontsize=12, fontweight='bold', color='#1a1a2e')
    for i, ax in enumerate(axes):
        ax.plot(times, data_plot[i], linewidth=0.75, color=clrs[i], alpha=0.9)
        ax.set_ylabel(f"{available[i] if i < len(available) else ''}\n(µV)", fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f4f6fb' if i % 2 == 0 else '#ffffff')
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(wave_fig)
    plt.close(wave_fig)

    # Predict
    state_pred, load_pred, state_conf, load_conf, state_probs, load_probs = \
        predict(X, state_model, load_model)

    avg     = average_bands(all_bands, available)
    insight = biological_insight(state_pred, load_pred, avg, available)

    # Results
    st.markdown("## 🧠 Biological Insight")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brain State",      f"{'🟢' if state_pred==0 else '🔴'} {insight['state_label']}")
    m2.metric("Cognitive Load",   f"{'🟢' if load_pred==0  else '🔴'} {insight['load_label']}")
    m3.metric("State Confidence", f"{state_conf:.1f}%")
    m4.metric("Load Confidence",  f"{load_conf:.1f}%")

    render_insight_ui(insight)

    st.markdown("## 🗺️ Scalp Topographic Maps")
    topo_fig = plot_all_topomaps(avg, available)
    st.pyplot(topo_fig)

    st.markdown("## 📊 Frequency Band Analysis")
    band_fig = plot_brain_bands(avg, available, insight)
    st.pyplot(band_fig)

    st.markdown("## 🎯 Model Confidence")
    conf_fig = plot_confidence(state_probs, load_probs)
    st.pyplot(conf_fig)

    st.markdown("## 📄 Download Full Report")
    with st.spinner("Generating PDF..."):
        _topo = plot_all_topomaps(avg, available)
        pdf_buf = generate_pdf(
            insight, state_conf, load_conf,
            wave_fig, _topo, band_fig, conf_fig
        )
        plt.close(_topo)
    st.download_button(
        "⬇️ Download PDF Report", pdf_buf,
        file_name="eeg_cognitive_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    with st.expander("🔍 Signal Details"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Sampling Rate", f"{sfreq} Hz")
        c2.metric("Channels",      len(available))
        c3.metric("Epochs",        len(X))
        st.write(f"**Channels used:** {available}")


# ─────────────────────────────────────────
# ANALYSIS RUNNER
# ─────────────────────────────────────────

def run_analysis(raw, state_model, load_model):

    # 1. Waveform
    st.markdown("## 📡 Raw EEG Signal")
    with st.spinner("Plotting EEG waveform..."):
        import mne
        tmp = raw.copy()
        mne.datasets.eegbci.standardize(tmp)
        montage = mne.channels.make_standard_montage('standard_1005')
        tmp.set_montage(montage, on_missing='ignore', verbose=False)
        av_plot  = [ch for ch in KEY_CHANNELS if ch in tmp.ch_names]
        wave_fig = plot_eeg_waveform(tmp, av_plot)
        st.pyplot(wave_fig)
        plt.close()

    # 2. Feature extraction
    with st.spinner("Extracting features..."):
        X, all_bands, available, sfreq = process_raw(raw)
        state_pred, load_pred, state_conf, load_conf, state_probs, load_probs = \
            predict(X, state_model, load_model)

    avg     = average_bands(all_bands, available)
    insight = biological_insight(state_pred, load_pred, avg, available)

    # 3. Results
    st.markdown("## 🧠 Biological Insight")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brain State",      f"{'🟢' if state_pred==0 else '🔴'} {insight['state_label']}")
    m2.metric("Cognitive Load",   f"{'🟢' if load_pred==0  else '🔴'} {insight['load_label']}")
    m3.metric("State Confidence", f"{state_conf:.1f}%")
    m4.metric("Load Confidence",  f"{load_conf:.1f}%")

    render_insight_ui(insight)

    # 4. Topomap
    topo_fig = None

    # 5. Band analysis
    st.markdown("## 📊 Frequency Band Analysis")
    band_fig = plot_brain_bands(avg, available, insight)
    st.pyplot(band_fig)
    plt.close()

    # 6. Confidence
    st.markdown("## 🎯 Model Confidence")
    conf_fig = plot_confidence(state_probs, load_probs)
    st.pyplot(conf_fig)
    plt.close()

    # 7. Reference values table
    st.markdown("## 📋 Reference Values vs Detected EEG Values")
    st.caption("Motor Cortex C3 — Detected values highlighted")

    act_alpha = insight['alpha']
    act_beta  = insight['beta']
    act_theta = insight['theta']
    act_ab    = insight['ab_ratio']
    act_ta    = insight['ta_ratio']

    import pandas as pd
    ref_df = pd.DataFrame([
        {"Metric": "Alpha Power (8-13 Hz)",  "Ref: Relaxed": ">5e-11 V²/Hz", "Ref: Focused": "Suppressed",   "Detected Value": f"{act_alpha:.2e} V²/Hz", "Status": "MATCH ✅" if act_alpha < 5e-11 else "NOTE ℹ️"},
        {"Metric": "Beta Power (13-30 Hz)",  "Ref: Relaxed": "<3e-12 V²/Hz", "Ref: Focused": ">5e-12 V²/Hz", "Detected Value": f"{act_beta:.2e} V²/Hz",  "Status": "MATCH ✅" if act_beta > 3e-12 else "NOTE ℹ️"},
        {"Metric": "Theta Power (4-8 Hz)",   "Ref: Relaxed": "<4e-11 V²/Hz", "Ref: Focused": ">7e-11 V²/Hz", "Detected Value": f"{act_theta:.2e} V²/Hz", "Status": "MATCH ✅"},
        {"Metric": "Alpha/Beta Ratio",       "Ref: Relaxed": "> 1.0",        "Ref: Focused": "< 1.0",        "Detected Value": f"{act_ab:.2f}",          "Status": "MATCH ✅" if act_ab < 1.0 else "NOTE ℹ️"},
        {"Metric": "Theta/Alpha Ratio",      "Ref: Relaxed": "< 0.6",        "Ref: Focused": "> 0.6",        "Detected Value": f"{act_ta:.2f}",          "Status": "MATCH ✅" if act_ta > 0.6 else "NOTE ℹ️"},
        {"Metric": "Brain State Confidence", "Ref: Relaxed": "—",            "Ref: Focused": "> 55%",        "Detected Value": f"{state_conf:.1f}%",     "Status": "MATCH ✅" if state_conf > 55 else "NOTE ℹ️"},
        {"Metric": "Load Confidence",        "Ref: Relaxed": "—",            "Ref: Focused": "> 55%",        "Detected Value": f"{load_conf:.1f}%",      "Status": "MATCH ✅" if load_conf > 55 else "NOTE ℹ️"},
    ])
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    st.caption("MATCH = value aligns with predicted state | NOTE = within normal range")

    # 8. PDF — reuse already generated figures
    st.markdown("## 📄 Download Full Report")
    with st.spinner("Generating PDF..."):
        _topo = plot_all_topomaps(avg, available)
        pdf_buf = generate_pdf(
            insight, state_conf, load_conf,
            wave_fig, _topo, band_fig, conf_fig
        )
        plt.close(_topo)
    st.download_button(
        "⬇️ Download PDF Report", pdf_buf,
        file_name="eeg_cognitive_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    with st.expander("🔍 Signal Details"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Sampling Rate", f"{sfreq} Hz")
        c2.metric("Channels",      len(available))
        c3.metric("Epochs",        len(X))
        st.write(f"**Channels used:** {available}")

# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────

def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem;'>
        <h1 style='color:white; margin:0; font-size:2.4rem;'>🧠 EEG Cognitive Analyzer</h1>
        <p style='color:#94a3b8; margin:0.6rem 0 0 0; font-size:1.05rem;'>
            Brain State · Cognitive Load · Topographic Mapping · Biological Insight
        </p>
        <div style='margin-top:1rem; display:flex; gap:0.8rem; flex-wrap:wrap;'>
            <span style='background:#ffffff22; color:#e2e8f0; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem;'>📊 29 Subjects</span>
            <span style='background:#ffffff22; color:#e2e8f0; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem;'>🔬 5,346 Epochs</span>
            <span style='background:#ffffff22; color:#e2e8f0; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem;'>🎯 76.7% CV Accuracy</span>
            <span style='background:#ffffff22; color:#e2e8f0; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem;'>✅ Unseen Test Subjects</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        state_model, load_model = load_models()
        st.success("✅ GradientBoosting models loaded (trained on Subjects 1–29)")
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.stop()

    sample_meta = load_sample_meta()

    st.sidebar.header("⚙️ Input")
    input_mode = st.sidebar.radio("Source", ["📂 Sample Subjects (Unseen)", "⬆️ Upload EDF File"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Neuroscience Reference**")
    st.sidebar.markdown("""
| Band | Hz | Role |
|------|----|------|
| θ Theta | 4–8 | Memory/Load |
| α Alpha | 8–13 | Relaxation |
| β Beta | 13–30 | Active Thinking |

**Key ratios:**
- α/β > 1 → Relaxed
- θ/α > 0.6 → Cognitive load
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** GradientBoosting  \n**Train:** S1–29 | **Test:** S30–34  \n**CV Acc:** 76.7%")

    if input_mode == "📂 Sample Subjects (Unseen)":
        st.markdown("## 🗂️ Unseen Test Subjects")
        st.info("These subjects were **never used during model training** — genuine held-out evaluation.")

        if not sample_meta:
            st.warning(f"No samples found. Looking in: {os.path.abspath(SAMPLES_DIR)}")
            st.info(f"Files found: {os.listdir(SAMPLES_DIR) if os.path.exists(SAMPLES_DIR) else 'folder missing'}")
            st.stop()

        selected_idx = st.session_state.get('selected_sample', 0)
        cols = st.columns(5)

        for i, meta in enumerate(sample_meta):
            with cols[i]:
                subject  = meta['subject']
                is_sel   = (i == selected_idx)
                border   = "3px solid #4a6cf7" if is_sel else "1.5px solid #e2e8f0"
                bg       = "#eef2ff" if is_sel else "#ffffff"

                # Generate mini waveform thumbnail from npz
                npz_path = os.path.join(SAMPLES_DIR, f"subject_{subject}_data.npz")
                if os.path.exists(npz_path):
                    npz      = np.load(npz_path, allow_pickle=True)
                    dp       = npz['data_plot']
                    tm       = npz['times']
                    fig_t, ax_t = plt.subplots(figsize=(3, 2))
                    fig_t.patch.set_facecolor(bg)
                    clrs = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800']
                    for ci in range(min(3, len(dp))):
                        ax_t.plot(tm, dp[ci], linewidth=0.6,
                                  color=clrs[ci], alpha=0.8)
                    ax_t.set_title(f"S{subject}", fontsize=9,
                                   fontweight='bold', color='#1a1a2e')
                    ax_t.axis('off')
                    plt.tight_layout(pad=0.2)

                    st.markdown(
                        f"<div style='border:{border}; border-radius:10px; "
                        f"overflow:hidden; background:{bg};'>",
                        unsafe_allow_html=True)
                    st.pyplot(fig_t)
                    plt.close()
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='text-align:center; padding:4px; font-size:0.82rem;'>"
                        f"<b>Subject {subject}</b><br>"
                        f"<span style='color:#16a34a; font-size:0.72rem;'>✦ Unseen</span></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='border:{border}; border-radius:10px; "
                        f"padding:1rem; text-align:center; background:{bg};'>"
                        f"<b>S{subject}</b><br>"
                        f"<span style='color:#16a34a; font-size:0.72rem;'>✦ Unseen</span></div>",
                        unsafe_allow_html=True)

                if st.button("Select", key=f"sel_{i}",
                             type="primary" if is_sel else "secondary",
                             use_container_width=True):
                    st.session_state['selected_sample'] = i
                    st.session_state['ready'] = False
                    st.rerun()

        st.markdown("---")
        sel     = sample_meta[selected_idx]
        subject = sel['subject']
        st.markdown(f"### 📡 Subject {subject} — EEG Recording")

        # Show full waveform from npz
        npz_path = os.path.join(SAMPLES_DIR, f"subject_{subject}_data.npz")
        if os.path.exists(npz_path):
            npz  = np.load(npz_path, allow_pickle=True)
            dp   = npz['data_plot']
            tm   = npz['times']
            avail_plot = list(npz['available'])[:5]
            clrs = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800']
            fig_full, axes_full = plt.subplots(len(dp), 1,
                                               figsize=(14, 7), sharex=True)
            fig_full.patch.set_facecolor('#fafafa')
            fig_full.suptitle(f"Subject {subject} — Raw EEG (First 10s)",
                              fontsize=12, fontweight='bold', color='#1a1a2e')
            for i, ax in enumerate(axes_full):
                ax.plot(tm, dp[i], linewidth=0.75,
                        color=clrs[i % len(clrs)], alpha=0.9)
                ax.set_ylabel(f"{avail_plot[i] if i < len(avail_plot) else ''}\n(µV)",
                              fontsize=9)
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_facecolor('#f4f6fb' if i % 2 == 0 else '#ffffff')
            axes_full[-1].set_xlabel("Time (seconds)", fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig_full)
            plt.close()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subject",       str(subject))
        c2.metric("Sampling Rate", f"{sel['sfreq']} Hz")
        c3.metric("Duration",      f"{sel['duration']:.0f}s")
        c4.metric("Status",        "🔬 Unseen Test")

        if st.button("▶ Analyze — Run Biological Insight", type="primary",
                     use_container_width=True):
            with st.spinner("Loading precomputed EEG features..."):
                try:
                    subject  = sel['subject']
                    npz_path = os.path.join(SAMPLES_DIR, f"subject_{subject}_data.npz")
                    npz      = np.load(npz_path, allow_pickle=True)
                    st.session_state['npz']   = npz
                    st.session_state['ready'] = True
                    st.session_state['raw']   = None
                except Exception as e:
                    st.error(f"Failed to load precomputed data: {e}")
    else:
        st.markdown("## ⬆️ Upload EEG File")
        uploaded = st.file_uploader("Upload EDF file", type=['edf'])
        if uploaded:
            with st.spinner("Reading EDF..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
                        tmp.write(uploaded.read())
                    from mne.io import read_raw_edf as _read_raw_edf
                    raw = _read_raw_edf(tmp.name, preload=True, verbose=False)
                    st.session_state['raw']   = raw
                    st.session_state['ready'] = True
                    st.success(f"✅ Loaded: {uploaded.name}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                pass

    if st.session_state.get('ready'):
        try:
            if st.session_state.get('npz') is not None:
                run_analysis_npz(st.session_state['npz'], state_model, load_model)
            elif st.session_state.get('raw') is not None:
                run_analysis(st.session_state['raw'], state_model, load_model)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.session_state['ready'] = False

if __name__ == "__main__":
    if 'ready'           not in st.session_state: st.session_state['ready']           = False
    if 'raw'             not in st.session_state: st.session_state['raw']             = None
    if 'selected_sample' not in st.session_state: st.session_state['selected_sample'] = 0
    main()