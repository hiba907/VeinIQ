VeinIQ — Peripheral IV Cannulation Decision Support v7
=======================================================

WHAT'S IN THIS ZIP
------------------
app/main.py              Streamlit clinical app (models embedded — NO xgboost needed)
run_all.py               Master runner — retrain all models
brain_pipeline.py        XGBoost clinical risk model (n=256, age 66-75)
eyes_pipeline.py         RF+LR vein suitability model (n=1065, age 26-52)
site_pipeline.py         Multiclass best-site recommender
fusion_pipeline.py       VeinIQ late fusion engine (0.20*Brain + 0.80*Eyes)
vascular_pipeline.py     CUBITAL NIR pipeline v2 (CUBITAL ONLY, calibrated x0.65)
models/                  3 trained .pkl files (brain, eyes, site)
data/                    Training CSVs + CUBITAL datasheet + vascular assessment
results/                 Performance charts (ROC, confusion, feature importance)
docs/VeinIQ_Limitations.docx   7 known limitations documented
docs/VeinIQ_NextSteps.docx     5 next steps to go to clinical deployment

HOW TO RUN THE APP LOCALLY
---------------------------
pip install streamlit pandas numpy scikit-learn matplotlib scipy
streamlit run app/main.py

PACKAGES (APP ONLY — NO xgboost/joblib/openpyxl/pillow needed)
---------------------------------------------------------------
streamlit  pandas  numpy  scikit-learn  matplotlib  scipy  pickle(built-in)

HOW TO RETRAIN MODELS
---------------------
pip install scikit-learn xgboost pandas numpy matplotlib scipy
python run_all.py

POPULATION COVERAGE
-------------------
Brain model : n=256,   age 66-75 (surgical elderly)
Eyes model  : n=1065,  age 26-52 (ED/ward adults)
CUBITAL NIR : n=7993,  age 6-16  (paediatric — vision module only)
Age gaps    : 17-25 and 53-65 (documented — see Limitations.docx)

VEINIQ SCORE
------------
Score = 0.80 x Eyes suitability + 0.20 x (1 - Brain failure risk)
>=0.70  → GREEN  PROCEED      (standard 18-20G)
>=0.45  → AMBER  CAUTION      (use US guidance)
 <0.45  → RED    HIGH RISK    (consider central line)
