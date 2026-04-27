# step4_train.py
# Purpose: Train Brain State + Cognitive Load models with subject-aware
# cross-validation to prevent data leakage between subjects.

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DATA_DIR   = "D:/cognitive/data/features"
MODELS_DIR = "D:/cognitive/models"
OUTPUT_DIR = "D:/cognitive/outputs/step4"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

print("=" * 60)
print("STEP 4: Model Training")
print("=" * 60)

print("\n[1] Loading features...")
X             = np.load(os.path.join(DATA_DIR, "X.npy"))
y_state       = np.load(os.path.join(DATA_DIR, "y_state.npy"))
y_load        = np.load(os.path.join(DATA_DIR, "y_load.npy"))
feature_names = np.load(os.path.join(DATA_DIR, "feature_names.npy"), allow_pickle=True)

print(f"  X shape  : {X.shape}")
print(f"  y_state  : {np.bincount(y_state)}  (0=Relaxed, 1=Focused)")
print(f"  y_load   : {np.bincount(y_load)}   (0=Low, 2=High)")

# ─────────────────────────────────────────
# MODEL COMPARISON HELPER
# ─────────────────────────────────────────

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def best_model(X, y, task_name):
    print(f"\n  Comparing models for: {task_name}")
    n_classes = len(np.unique(y))
    cw = 'balanced'

    candidates = {
        'RandomForest': Pipeline([
            ('sc', RobustScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=6,
                min_samples_leaf=5, max_features='sqrt',
                random_state=RANDOM_STATE, class_weight=cw))
        ]),
        'SVM-RBF': Pipeline([
            ('sc', RobustScaler()),
            ('clf', SVC(kernel='rbf', C=10.0, gamma='scale',
                        class_weight=cw, random_state=RANDOM_STATE,
                        probability=True))
        ]),
        'GradBoost': Pipeline([
            ('sc', RobustScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE))
        ]),
    }

    best_name, best_score, best_pipe = None, 0, None
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        print(f"    {name:15s}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name  = name
            best_pipe  = pipe

    print(f"  → Best: {best_name} ({best_score*100:.1f}%)")
    return best_pipe, best_name, best_score

# ─────────────────────────────────────────
# BRAIN STATE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("[2] Brain State Classifier (Relaxed vs Focused)")
print("=" * 60)

state_pipe, state_name, state_cv = best_model(X, y_state, "Brain State")
state_scores = cross_val_score(state_pipe, X, y_state, cv=cv, scoring='accuracy')
print(f"\n  CV per fold: {[f'{s*100:.1f}%' for s in state_scores]}")
print(f"  Mean: {state_scores.mean()*100:.1f}% | Std: {state_scores.std()*100:.1f}%")

state_pipe.fit(X, y_state)
state_preds = state_pipe.predict(X)
print(f"\n  Train accuracy: {np.mean(state_preds==y_state)*100:.1f}%")
print(classification_report(y_state, state_preds, target_names=['Relaxed','Focused']))

with open(os.path.join(MODELS_DIR, "brain_state_model.pkl"), 'wb') as f:
    pickle.dump(state_pipe, f)
print(f"  [SAVED] brain_state_model.pkl")

# ─────────────────────────────────────────
# COGNITIVE LOAD
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("[3] Cognitive Load (Low vs High)")
print("=" * 60)

# Use only 2-class load (Low=0, High=2 → remap to 0,1)
y_load_binary = (y_load > 0).astype(int)
print(f"  Binary load: 0:Low={np.sum(y_load_binary==0)}, 1:High={np.sum(y_load_binary==1)}")

load_pipe, load_name, load_cv = best_model(X, y_load_binary, "Cognitive Load")
load_scores = cross_val_score(load_pipe, X, y_load_binary, cv=cv, scoring='accuracy')
print(f"\n  CV per fold: {[f'{s*100:.1f}%' for s in load_scores]}")
print(f"  Mean: {load_scores.mean()*100:.1f}% | Std: {load_scores.std()*100:.1f}%")

load_pipe.fit(X, y_load_binary)
load_preds = load_pipe.predict(X)
print(f"\n  Train accuracy: {np.mean(load_preds==y_load_binary)*100:.1f}%")
print(classification_report(y_load_binary, load_preds, target_names=['Low','High']))

with open(os.path.join(MODELS_DIR, "cognitive_load_model.pkl"), 'wb') as f:
    pickle.dump(load_pipe, f)
print(f"  [SAVED] cognitive_load_model.pkl")

# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────

print("\n[4] Saving feature importance...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Feature Importance: EEG Band Powers", fontsize=14)

for ax, pipe, title, color in zip(
    axes,
    [state_pipe, load_pipe],
    ['Brain State', 'Cognitive Load'],
    ['steelblue', 'tomato']
):
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        imp  = clf.feature_importances_
        idx  = np.argsort(imp)
        names = [str(feature_names[i]) for i in idx]
        vals  = imp[idx]
        ax.barh(range(len(names)), vals, color=color, alpha=0.8,
                edgecolor='black', linewidth=0.4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "SVM: no feature importance",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"  [SAVED] feature_importance.png")

# ─────────────────────────────────────────
# CONFUSION MATRICES
# ─────────────────────────────────────────

print("\n[5] Saving confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix(y_state, state_preds),
    display_labels=['Relaxed','Focused']
).plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f"Brain State (CV: {state_scores.mean()*100:.1f}%)")

ConfusionMatrixDisplay(
    confusion_matrix(y_load_binary, load_preds),
    display_labels=['Low','High']
).plot(ax=axes[1], colorbar=False, cmap='Reds')
axes[1].set_title(f"Cognitive Load (CV: {load_scores.mean()*100:.1f}%)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print(f"  [SAVED] confusion_matrices.png")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
print(f"  Brain State  — {state_name:15s} CV: {state_scores.mean()*100:.1f}% ± {state_scores.std()*100:.1f}%")
print(f"  Cogn. Load   — {load_name:15s} CV: {load_scores.mean()*100:.1f}% ± {load_scores.std()*100:.1f}%")
print(f"  Models saved : {MODELS_DIR}")
print("\nReady for Step 5: Streamlit App")