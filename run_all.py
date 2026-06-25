"""
==============================================================
  MASTER RUN SCRIPT v3
  Runs all 4 pipelines in correct order:
    1. brain_pipeline.py   → brain_model.pkl
    2. eyes_pipeline.py    → eyes_model.pkl
    3. site_pipeline.py    → site_model.pkl
    4. fusion_pipeline.py  → fusion_results.png

  Just run:  python run_all.py
==============================================================
"""
import subprocess, sys, os

scripts = [
    'brain_pipeline.py',
    'eyes_pipeline.py',
    'site_pipeline.py',
    'fusion_pipeline.py'
]
script_dir = os.path.dirname(os.path.abspath(__file__))

for script in scripts:
    path = os.path.join(script_dir, script)
    print(f"\n{'='*60}")
    print(f"  RUNNING: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, path], check=True)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(1)

print("\n" + "="*60)
print("  ALL 4 PIPELINES COMPLETED SUCCESSFULLY")
print()
print("  Output files in your folder:")
print("    brain_results.png    — Brain model performance")
print("    eyes_results.png     — Eyes model performance")
print("    site_results.png     — Site recommender performance")
print("    fusion_results.png   — Full assessment (all 3 fused)")
print("    brain_model.pkl      — Saved Brain model")
print("    eyes_model.pkl       — Saved Eyes model")
print("    site_model.pkl       — Saved Site model")
print("="*60)
