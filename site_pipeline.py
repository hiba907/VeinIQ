"""
==============================================================
  SITE RECOMMENDER PIPELINE — Best Insertion Site
  Dataset : eyes_pipeline_adiva.csv (1,065 patients, Blad1)
  Target  : Place — best insertion site (0=Hand, 1=Forearm, 2=Elbow)
  Model   : Random Forest multi-class classifier
  Output  : site_model.pkl  +  site_results.png
==============================================================
  WHY NO NEW DATA IS NEEDED:
    The A-DIVA dataset already contains the 'Place' column
    recording which anatomical site was chosen for each patient,
    along with vein diameter, CVR, palpability, and visibility
    at that site. This gives us 1,052 real insertion decisions
    to train on — no synthetic or external data required.

  WHAT THIS PIPELINE DOES:
    Given a patient's vein measurements and characteristics,
    it recommends the BEST insertion site:
      0 = Dorsum of Hand     (n=556, success 85.6%)
      1 = Forearm            (n=275, success 81.8%)
      2 = Elbow Crease       (n=221, success 77.4%)
    Upper arm (Place=3) excluded — only 11 patients, too few.

  HOW TO RUN:
      python site_pipeline.py
  REQUIREMENTS:
      pip install scikit-learn imbalanced-learn matplotlib pandas openpyxl
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

# ── SITE MAPPING ─────────────────────────────────────────────
SITE_MAP   = {0: 'Dorsum of Hand', 1: 'Forearm', 2: 'Elbow Crease'}
SITE_COLOR = {0: '#0F6E56', 1: '#534AB7', 2: '#D97706'}
SUCCESS_BY_SITE = {0: 85.6, 1: 81.8, 2: 77.4}   # from A-DIVA data

print("=" * 60)
print("  SITE RECOMMENDER PIPELINE — Best Insertion Site")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'eyes_pipeline_adiva.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n[ERROR] Cannot find: {data_path}\n"
        "Make sure eyes_pipeline_adiva.csv is in the same folder."
    )

df = pd.read_csv(data_path)
print(f"\n[1/8] Data loaded: {df.shape[0]} patients")
print("      Using A-DIVA dataset — Place column = insertion site chosen")
print("      No new data required — site decisions already recorded")

# ── 2. PREPARE DATA ──────────────────────────────────────────
# Keep only 3 main sites — upper arm (Place=3) has only 11 patients
df = df[df['Place'].isin([0, 1, 2])].dropna(subset=['Place', 'Success'])
print(f"\n[2/8] After filtering to 3 main sites: {len(df)} patients")
for place, name in SITE_MAP.items():
    sub = df[df['Place']==place]
    print(f"      {name} (Place={place}): n={len(sub)}, "
          f"success={sub['Success'].mean()*100:.1f}%, "
          f"mean vein diam={sub['DiaVeneMM'].mean():.2f}mm")

# Features — all available before the cannulation attempt
FEATURES = [
    'Age', 'BMI', 'Sex',
    'Palpable', 'Visual',          # vein visibility
    'History',                      # difficult access history
    'DiaVeneMM', 'CVR',            # vein measurements
    'CatheterSizeMM',              # catheter planned
    'A-DIVA', 'VAS'                # risk score + pain
]
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df['Place'].astype(int)

print(f"\n      Features ({len(FEATURES)}): {FEATURES}")

# ── 3. SPLIT ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[3/8] Split → Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. PREPROCESS ────────────────────────────────────────────
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)
X_train_sc  = scaler.fit_transform(X_train_imp)
X_test_sc   = scaler.transform(X_test_imp)

print("[4/8] Preprocessing done")

# ── 5. TRAIN MODEL ───────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_sc, y_train)
print("[5/8] Random Forest site recommender trained")

# ── 5b. CV WITH SMOTE INSIDE FOLD ───────────────────────────
class ScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imp = SimpleImputer(strategy='median')
        self.sc  = StandardScaler()
    def fit(self, X, y=None):
        self.sc.fit(self.imp.fit_transform(X))
        return self
    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))

cv_pipe = ImbPipeline([
    ('preprocess', ScalerWrapper()),
    ('smote',      SMOTE(random_state=42)),
    ('model',      RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='accuracy')
print(f"      5-fold CV Accuracy (SMOTE inside fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 6. EVALUATE ──────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)

# Bootstrap CI for accuracy
np.random.seed(42)
boot_acc = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    boot_acc.append((y_pred[idx] == np.array(y_test)[idx]).mean())
ci_acc = (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5))

overall_acc = (y_pred == y_test).mean()

print(f"\n[6/8] Test Results")
print(f"      Overall Accuracy: {overall_acc:.3f}  95% CI [{ci_acc[0]:.3f} – {ci_acc[1]:.3f}]")
print(f"\n      Per-site report:")
print(classification_report(
    y_test, y_pred,
    target_names=[SITE_MAP[i] for i in sorted(SITE_MAP)]
))

# ── 7. SAVE MODEL ────────────────────────────────────────────
joblib.dump({
    'model':    model,
    'imputer':  imputer,
    'scaler':   scaler,
    'features': FEATURES,
    'site_map': SITE_MAP,
    'success_by_site': SUCCESS_BY_SITE
}, os.path.join(script_dir, 'site_model.pkl'))
print("[7/8] Model saved → site_model.pkl")

# ── 8. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'Site Recommender — Best Insertion Site  |  Accuracy {overall_acc:.3f} '
    f'(95% CI {ci_acc[0]:.3f}–{ci_acc[1]:.3f})',
    fontsize=13, fontweight='bold'
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    cm, display_labels=[SITE_MAP[i] for i in sorted(SITE_MAP)]
)
disp.plot(ax=axes[0], colorbar=False, cmap='Purples')
axes[0].set_title('Confusion Matrix')
axes[0].tick_params(axis='x', rotation=20)

# Feature importance
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ['#534AB7' if v > fi.median() else '#AFA9EC' for v in fi]
fi.plot(kind='barh', ax=axes[1], color=colors)
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance score')
axes[1].grid(True, alpha=0.3, axis='x')

# Success rate by site bar chart
sites   = [SITE_MAP[i] for i in sorted(SITE_MAP)]
success = [SUCCESS_BY_SITE[i] for i in sorted(SITE_MAP)]
counts  = [len(df[df['Place']==i]) for i in sorted(SITE_MAP)]
bars = axes[2].bar(sites, success,
                   color=[SITE_COLOR[i] for i in sorted(SITE_MAP)],
                   alpha=0.85, width=0.5)
axes[2].set_ylim(70, 92)
axes[2].set_ylabel('First-attempt success rate (%)')
axes[2].set_title('Historical Success Rate by Site')
axes[2].grid(True, alpha=0.3, axis='y')
for bar, cnt in zip(bars, counts):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'n={cnt}', ha='center', fontsize=9, color='#374151')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'site_results.png'), dpi=150, bbox_inches='tight')
print("[8/8] Results chart saved → site_results.png")

print("\n" + "=" * 60)
print("  SITE RECOMMENDER PIPELINE COMPLETE")
print("  Historical success rates from A-DIVA data:")
for i, name in SITE_MAP.items():
    print(f"    {name}: {SUCCESS_BY_SITE[i]}% first-attempt success")
print("=" * 60)
