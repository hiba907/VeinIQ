"""
==============================================================
  LATE FUSION PIPELINE v3 — PIVC Suitability + Site
  Combines:
    Brain  → Clinical risk score
    Eyes   → Vein suitability score
    Site   → Best insertion site recommendation
  Output  : fusion_results.png
==============================================================
  FUSION WEIGHT JUSTIFICATION:
    Brain AUC = 0.624  → Weight = 0.20
    Eyes  AUC = 0.936  → Weight = 0.80
    Weights reflect actual predictive strength of each pipeline.

  VALIDATION NOTE:
    Brain and Eyes datasets contain different patients.
    Fusion is illustrative until a dataset with ALL modalities
    is collected for the same patients.

  HOW TO RUN (run all 3 model pipelines first):
      python brain_pipeline.py
      python eyes_pipeline.py
      python site_pipeline.py
      python fusion_pipeline.py
  OR simply run:
      python run_all.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

# ── CONFIGURABLE FUSION WEIGHTS ──────────────────────────────
BRAIN_WEIGHT = 0.20
EYES_WEIGHT  = 0.80
assert abs(BRAIN_WEIGHT + EYES_WEIGHT - 1.0) < 1e-9, "Weights must sum to 1.0"

print("=" * 60)
print("  LATE FUSION PIPELINE v3 — PIVC Suitability + Site")
print(f"  Weights: Brain={BRAIN_WEIGHT} | Eyes={EYES_WEIGHT}")
print("=" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))

for fname in ['brain_model.pkl', 'eyes_model.pkl', 'site_model.pkl']:
    if not os.path.exists(os.path.join(script_dir, fname)):
        raise FileNotFoundError(
            f"[ERROR] {fname} not found. Run the corresponding pipeline first."
        )

brain_b = joblib.load(os.path.join(script_dir, 'brain_model.pkl'))
eyes_b  = joblib.load(os.path.join(script_dir, 'eyes_model.pkl'))
site_b  = joblib.load(os.path.join(script_dir, 'site_model.pkl'))

SITE_MAP         = site_b['site_map']
SUCCESS_BY_SITE  = site_b['success_by_site']

print(f"\n[1/4] All 3 models loaded")

# ── FULL FUSION FUNCTION ─────────────────────────────────────
def assess_patient(brain_inputs, eyes_inputs,
                   brain_weight=BRAIN_WEIGHT,
                   eyes_weight=EYES_WEIGHT):
    """
    Full VeinIQ assessment combining all 3 pipelines.

    Parameters
    ----------
    brain_inputs : dict  — clinical/demographic patient data
    eyes_inputs  : dict  — vein assessment data (also used for site)

    Returns
    -------
    dict with suitability index, recommendation, and best site
    """
    # ── Brain: risk score ──
    b_df       = pd.DataFrame([brain_inputs])
    b_imp      = brain_b['imputer'].transform(b_df[brain_b['features']])
    risk_score = brain_b['model'].predict_proba(b_imp)[0][1]

    # ── Eyes: vein suitability ──
    e_df       = pd.DataFrame([eyes_inputs])
    e_imp      = eyes_b['imputer'].transform(e_df[eyes_b['features']])
    e_sc       = eyes_b['scaler'].transform(e_imp)
    vein_score = eyes_b['model'].predict_proba(e_sc)[0][1]

    # ── Site recommender ──
    s_df       = pd.DataFrame([eyes_inputs])
    s_feats    = site_b['features']
    s_imp      = site_b['imputer'].transform(s_df[s_feats])
    s_sc       = site_b['scaler'].transform(s_imp)
    site_probs = site_b['model'].predict_proba(s_sc)[0]
    site_classes = site_b['model'].classes_
    best_site_idx = int(site_classes[np.argmax(site_probs)])
    best_site_name = SITE_MAP[best_site_idx]
    best_site_prob = round(float(np.max(site_probs)), 3)
    best_site_success = SUCCESS_BY_SITE[best_site_idx]

    # Site confidence text
    site_ranking = sorted(
        [(SITE_MAP[site_classes[i]], round(float(site_probs[i]),3))
         for i in range(len(site_classes))],
        key=lambda x: -x[1]
    )

    # ── Fusion ──
    safety_score      = 1.0 - risk_score
    suitability_index = (brain_weight * safety_score) + (eyes_weight * vein_score)

    # ── Recommendation ──
    if suitability_index >= 0.70:
        recommendation = "PROCEED — Standard cannulation recommended"
        gauge_guidance = "18–20G catheter suitable"
        urgency        = "GREEN"
    elif suitability_index >= 0.45:
        recommendation = "CAUTION — Consider ultrasound guidance"
        gauge_guidance = "20–22G catheter recommended"
        urgency        = "AMBER"
    else:
        recommendation = "HIGH RISK — Seek senior/specialist assistance"
        gauge_guidance = "22G or central line; avoid blind attempts"
        urgency        = "RED"

    return {
        'risk_score'        : round(float(risk_score), 3),
        'vein_score'        : round(float(vein_score), 3),
        'safety_score'      : round(float(safety_score), 3),
        'suitability_index' : round(float(suitability_index), 3),
        'urgency'           : urgency,
        'recommendation'    : recommendation,
        'gauge_guidance'    : gauge_guidance,
        'best_site'         : best_site_name,
        'best_site_prob'    : best_site_prob,
        'best_site_success' : best_site_success,
        'site_ranking'      : site_ranking
    }

print("[2/4] Full fusion function (Brain + Eyes + Site) ready")

# ── DEMO PATIENTS ────────────────────────────────────────────
print(f"\n[3/4] Demo patients\n")

patients = [
    {
        "label": "Patient A — Low risk (healthy adult, good veins)",
        "brain": {'group':1,'sex':1,'age':45,'ASA':1,'height':175,'weight':72,'bmi':23.5,
                  'diameter':3.0,'depth':2.0,'hypertension':0,'diabetes':0,
                  'cholesterol':0,'peripheral':0,'smoking':0},
        "eyes":  {'Age':45,'Length':175,'Weight':72,'BMI':23.5,'Sex':1,
                  'Palpable':1,'Visual':1,'History':0,'SidDom':0,'Place':1,
                  'VAS':1,'CatheterSizeMM':1.0,'DiaVeneMM':3.0,'CVR':0.33,'A-DIVA':0}
    },
    {
        "label": "Patient B — Moderate risk (elderly, hypertensive, thin veins)",
        "brain": {'group':1,'sex':0,'age':74,'ASA':2,'height':158,'weight':58,'bmi':23.2,
                  'diameter':2.0,'depth':3.5,'hypertension':1,'diabetes':0,
                  'cholesterol':1,'peripheral':0,'smoking':0},
        "eyes":  {'Age':74,'Length':158,'Weight':58,'BMI':23.2,'Sex':0,
                  'Palpable':1,'Visual':0,'History':0,'SidDom':0,'Place':2,
                  'VAS':3,'CatheterSizeMM':1.1,'DiaVeneMM':2.0,'CVR':0.55,'A-DIVA':2}
    },
    {
        "label": "Patient C — High risk (elderly, diabetic, peripheral VD)",
        "brain": {'group':2,'sex':0,'age':85,'ASA':3,'height':152,'weight':50,'bmi':21.6,
                  'diameter':1.4,'depth':5.2,'hypertension':1,'diabetes':1,
                  'cholesterol':1,'peripheral':1,'smoking':0},
        "eyes":  {'Age':85,'Length':152,'Weight':50,'BMI':21.6,'Sex':0,
                  'Palpable':0,'Visual':0,'History':1,'SidDom':0,'Place':3,
                  'VAS':6,'CatheterSizeMM':1.2,'DiaVeneMM':1.4,'CVR':0.86,'A-DIVA':4}
    }
]

results = []
for p in patients:
    r = assess_patient(p['brain'], p['eyes'])
    r['label'] = p['label']
    results.append(r)

    print(f"  {p['label']}")
    print(f"  ─── SUITABILITY ──────────────────────")
    print(f"  Brain risk score      : {r['risk_score']}")
    print(f"  Eyes vein score       : {r['vein_score']}")
    print(f"  PIVC Suitability Index: {r['suitability_index']}  → {r['urgency']}")
    print(f"  Recommendation        : {r['recommendation']}")
    print(f"  Gauge guidance        : {r['gauge_guidance']}")
    print(f"  ─── SITE RECOMMENDATION ──────────────")
    print(f"  Best insertion site   : {r['best_site']} (confidence {r['best_site_prob']})")
    print(f"  Historical success    : {r['best_site_success']}% at this site (A-DIVA data)")
    print(f"  Site ranking          : {r['site_ranking']}")
    print()

# ── VISUALISE ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))
fig.suptitle(
    'VeinIQ — Full Assessment: Suitability + Best Insertion Site',
    fontsize=14, fontweight='bold'
)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])

labels      = [f"Patient {chr(65+i)}" for i in range(len(results))]
risk_s      = [r['risk_score']   for r in results]
vein_s      = [r['vein_score']   for r in results]
suit_s      = [r['suitability_index'] for r in results]
urgency     = [r['urgency'] for r in results]
bar_cols    = {'GREEN':'#2E9E6B', 'AMBER':'#E5A800', 'RED':'#CC3333'}
site_colors = {'Dorsum of Hand':'#0F6E56', 'Forearm':'#534AB7', 'Elbow Crease':'#D97706'}

# Subplot 1 — Score breakdown
x = np.arange(len(labels))
w = 0.25
ax1.bar(x-w, risk_s, w, label='Brain risk',  color='#534AB7', alpha=0.85)
ax1.bar(x,   vein_s, w, label='Eyes vein',   color='#0F6E56', alpha=0.85)
ax1.bar(x+w, suit_s, w, label='Fusion index',
        color=[bar_cols[u] for u in urgency], alpha=0.95)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylim(0, 1.1)
ax1.axhline(0.70, color='green',  linestyle='--', lw=1)
ax1.axhline(0.45, color='orange', linestyle='--', lw=1)
ax1.set_title('Score Breakdown')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3, axis='y')

# Subplot 2 — Suitability gauge
bars = ax2.barh(labels, suit_s,
                color=[bar_cols[u] for u in urgency], height=0.5)
ax2.axvline(0.70, color='green',  linestyle='--', lw=1.5)
ax2.axvline(0.45, color='orange', linestyle='--', lw=1.5)
ax2.set_xlim(0, 1)
ax2.set_xlabel('Suitability Index')
ax2.set_title('Suitability Recommendation')
ax2.grid(True, alpha=0.3, axis='x')
for bar, r in zip(bars, results):
    ax2.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
             r['urgency'], va='center', fontweight='bold',
             color=bar_cols[r['urgency']], fontsize=9)

# Subplot 3 — Best site per patient
best_sites  = [r['best_site'] for r in results]
site_probs  = [r['best_site_prob'] for r in results]
site_succ   = [r['best_site_success'] for r in results]
site_bar_c  = [site_colors.get(s, '#888') for s in best_sites]
bars3 = ax3.bar(labels, site_probs,
                color=site_bar_c, alpha=0.85, width=0.5)
ax3.set_ylim(0, 1.0)
ax3.set_ylabel('Model confidence')
ax3.set_title('Recommended Insertion Site')
ax3.grid(True, alpha=0.3, axis='y')
for bar, site, succ in zip(bars3, best_sites, site_succ):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{site}\n({succ}% hist.)',
             ha='center', fontsize=8, color='#374151')

# Bottom — Full summary table
ax4.axis('off')
table_data = [
    ['Patient', 'Urgency', 'Suitability\nIndex',
     'Brain\nRisk', 'Eyes\nVein',
     'Best Site', 'Site\nConfidence', 'Historical\nSuccess', 'Gauge\nGuidance']
]
for r in results:
    table_data.append([
        r['label'].split('—')[0].strip(),
        r['urgency'],
        str(r['suitability_index']),
        str(r['risk_score']),
        str(r['vein_score']),
        r['best_site'],
        str(r['best_site_prob']),
        f"{r['best_site_success']}%",
        r['gauge_guidance']
    ])

tbl = ax4.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1B3A6B')
        cell.set_text_props(color='white', fontweight='bold')
    elif results[row-1]['urgency'] == 'GREEN':
        cell.set_facecolor('#D1FAE5')
    elif results[row-1]['urgency'] == 'AMBER':
        cell.set_facecolor('#FEF3C7')
    else:
        cell.set_facecolor('#FEE2E2')
    cell.set_edgecolor('#CCCCCC')

ax4.set_title('Full Assessment Summary', fontweight='bold', pad=10)

plt.savefig(os.path.join(script_dir, 'fusion_results.png'), dpi=150, bbox_inches='tight')
print(f"[4/4] Full fusion chart saved → fusion_results.png")

print("\n" + "=" * 60)
print("  LATE FUSION PIPELINE v3 COMPLETE")
print("  3 pipelines fused: Brain + Eyes + Site Recommender")
print("=" * 60)
