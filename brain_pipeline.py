"""
==============================================================
  BRAIN PIPELINE — Clinical Risk Model  (v2 — reviewed)
  Dataset : brain_pipeline_cannulation.csv  (256 patients)
  Target  : first_attempt failure (1 = failed, 0 = success)
  Model   : XGBoost gradient-boosted decision tree
  Output  : brain_model.pkl  +  brain_results.png
==============================================================
  CHANGES FROM v1 (post Minimax code review):
    - Removed deprecated use_label_encoder=False (XGBoost >1.6)
    - SMOTE now applied INSIDE CV pipeline (prevents leakage)
    - Bootstrap 95% CI added for AUC, Sensitivity, Specificity
    - Drop columns now commented explaining WHY each is dropped
    - Fusion weights documented at top of fusion_pipeline.py
  HOW TO RUN:
      python brain_pipeline.py
  REQUIREMENTS:
      pip install xgboost scikit-learn imbalanced-learn matplotlib pandas
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, recall_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ────────────────────────────────────────────
print("=" * 60)
print("  BRAIN PIPELINE v2 — IV Cannulation Risk Predictor")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'brain_pipeline_cannulation.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n[ERROR] Cannot find: {data_path}\n"
        "Make sure brain_pipeline_cannulation.csv is in the same folder."
    )

df = pd.read_csv(data_path)
print(f"\n[1/8] Data loaded: {df.shape[0]} patients, {df.shape[1]} columns")

# ── 2. PREPARE FEATURES & TARGET ────────────────────────────
# Drop columns — reason documented for each:
#   'number'      → patient identifier, not a clinical feature
#   'no_attempts' → total attempts = post-procedure outcome, would cause leakage
#   'times'       → procedure time in minutes = post-procedure, would cause leakage
drop_cols = ['number', 'no_attempts', 'times']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target: first_attempt → 1=failed, 0=success
# Original SPSS coding: 1.0 = success, 2.0 = failed
df['target'] = (df['first_attempt'] == 2.0).astype(int)
df = df.drop(columns=['first_attempt'])

FEATURES = [
    'group', 'sex', 'age', 'ASA',
    'height', 'weight', 'bmi',
    'diameter', 'depth',           # vein measurements — key predictors
    'hypertension', 'diabetes',    # comorbidities
    'cholesterol', 'peripheral', 'smoking'
]
FEATURES = [f for f in FEATURES if f in df.columns]
X = df[FEATURES]
y = df['target']

print(f"[2/8] Features used ({len(FEATURES)}): {FEATURES}")
print(f"      Class balance → Success: {(y==0).sum()} | Failed: {(y==1).sum()}")
print(f"      Failure rate  → {y.mean()*100:.1f}%")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[3/8] Split → Train: {len(X_train)} | Test (held-out): {len(X_test)}")
print(f"      Test failures: {y_test.sum()} | Test successes: {(y_test==0).sum()}")
print(f"      NOTE: Small test set → wide confidence intervals (see bootstrap below)")

# ── 4. PREPROCESSING + SMOTE (on training only) ─────────────
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

# SMOTE applied to training set only — NOT test set
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train)
print(f"[4/8] After SMOTE → Success: {(y_train_bal==0).sum()} | Failed: {(y_train_bal==1).sum()}")

# ── 5. TRAIN XGBOOST MODEL ──────────────────────────────────
# Note: use_label_encoder removed — deprecated in XGBoost >1.6
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,   # balanced after SMOTE
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_bal, y_train_bal, verbose=False)
print("[5/8] XGBoost model trained")

# ── 5b. CROSS-VALIDATION (SMOTE inside CV fold — correct) ───
# Using imblearn Pipeline so SMOTE is applied per-fold, not before CV
cv_pipe = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote',   SMOTE(random_state=42)),
    ('model',   XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42
    ))
])
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"      10-fold CV AUC (SMOTE inside fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 6. EVALUATE ON HELD-OUT TEST SET ────────────────────────
y_pred      = model.predict(X_test_imp)
y_prob      = model.predict_proba(X_test_imp)[:, 1]
auc         = roc_auc_score(y_test, y_prob)
sensitivity = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)

# Bootstrap 95% CI (1000 resamples)
np.random.seed(42)
boot_aucs, boot_sens, boot_spec = [], [], []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    yt  = y_test.iloc[idx]
    yp  = y_prob[idx]
    ypr = y_pred[idx]
    try:
        boot_aucs.append(roc_auc_score(yt, yp))
        boot_sens.append(recall_score(yt, ypr, zero_division=0))
        boot_spec.append(recall_score(yt, ypr, pos_label=0, zero_division=0))
    except:
        pass

ci_auc  = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))
ci_sens = (np.percentile(boot_sens, 2.5), np.percentile(boot_sens, 97.5))
ci_spec = (np.percentile(boot_spec, 2.5), np.percentile(boot_spec, 97.5))

print(f"\n[6/8] Test Set Results (with 95% Bootstrap CI)")
print(f"      AUC         : {auc:.3f}  95% CI [{ci_auc[0]:.3f} – {ci_auc[1]:.3f}]")
print(f"      Sensitivity : {sensitivity:.3f}  95% CI [{ci_sens[0]:.3f} – {ci_sens[1]:.3f}]")
print(f"      Specificity : {specificity:.3f}  95% CI [{ci_spec[0]:.3f} – {ci_spec[1]:.3f}]")
print(f"      10-fold CV  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\n      INTERPRETATION: Wide CI reflects small test set ({len(y_test)} patients,")
print(f"      {y_test.sum()} failures). More data needed to narrow confidence intervals.")
print(f"\n      Full classification report:")
print(classification_report(y_test, y_pred,
      target_names=['Success (0)', 'Failed (1)']))

# ── 7. SAVE MODEL ────────────────────────────────────────────
joblib.dump({'model': model, 'imputer': imputer, 'features': FEATURES},
            os.path.join(script_dir, 'brain_model.pkl'))
print("[7/8] Model saved → brain_model.pkl")

# ── 8. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'Brain Pipeline — Clinical Risk Model  |  AUC {auc:.3f} (95% CI {ci_auc[0]:.3f}–{ci_auc[1]:.3f})',
    fontsize=13, fontweight='bold'
)

# ROC Curve with CI annotation
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color='#534AB7', lw=2,
             label=f'AUC = {auc:.3f}\n95% CI [{ci_auc[0]:.3f}–{ci_auc[1]:.3f}]')
axes[0].plot([0,1],[0,1],'--', color='gray', lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#534AB7')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Sensitivity)')
axes[0].set_title('ROC Curve')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].text(0.55, 0.08,
    f'⚠ Wide CI: small test set\n(n={len(y_test)}, failures={y_test.sum()})',
    fontsize=8, color='#D97706',
    bbox=dict(boxstyle='round', facecolor='#FEF3C7', alpha=0.8))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Success', 'Failed'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Confusion Matrix')

# Feature Importance
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ['#534AB7' if v > fi.median() else '#AFA9EC' for v in fi]
fi.plot(kind='barh', ax=axes[2], color=colors)
axes[2].set_title('Feature Importance')
axes[2].set_xlabel('Importance score')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'brain_results.png'), dpi=150, bbox_inches='tight')
print("[8/8] Results chart saved → brain_results.png")
print("\n" + "=" * 60)
print("  BRAIN PIPELINE v2 COMPLETE")
print("=" * 60)
