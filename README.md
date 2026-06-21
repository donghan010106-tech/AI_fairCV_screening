# FairCV — A Comparative Study of Fusion Strategies for Fair AI-Based Resume Evaluation

Capstone project (DAP391m). A reproducible study comparing **Early, Late, and Weighted
Hybrid Fusion** for fair resume screening on the **FairCVdb** benchmark (24,000 synthetic
profiles, Peña et al. 2020/2023), plus an interactive recruiter-assist application with
SHAP explanations and a human-in-the-loop workflow.

**Position studied:** Data Engineer · **Sensitive attributes:** gender, ethnicity

---

## Key findings

- **Structured features alone are enough.** Adding SBERT text embeddings via Early, Late,
  or Weighted Hybrid Fusion does **not** improve over the structured-only baseline.
  Best F1: Structured-Only = **0.9658**, Early = 0.9632, Late = 0.9606, Weighted Hybrid = 0.6993.
- **All models are reasonably fair.** Demographic-parity gaps are small across strategies
  (min DP gap gender ≈ 0.0019). SHAP shows the model uses features almost identically
  across gender and ethnicity groups.
- **Weighted Hybrid Fusion degrades sharply** — PCA-aligning 384-dim text down to 8 dims
  collapses both accuracy and fairness.

---

## Repository structure

```
faircv-capstone/
├── notebooks/
│   ├── FairCV_EDA.ipynb          
│   └── FairCV_Models.ipynb       
├── app/
│   ├── streamlit_app.py          # Recruiter-assist screening app
│   ├── extract_cv.py             # CV PDF -> 8 features (Gemini + rubric)
│   ├── screening.py              # Predict + rank Top 10 / Top 3
│   ├── model_structured.pkl      # Trained Logistic Regression model
│   ├── scaler_structured.pkl     # Feature scaler
│   └── requirements.txt
├── results/
│   ├── fusion_results.csv        # RQ1 + RQ2: performance & fairness per strategy
│   ├── ablation_sbert.csv        # RQ3: SBERT ablation
│   ├── bias_mitigation_results.csv  # RQ4: mitigation experiments
│   └── figures/                  
├── paper/
│   └── 
└── README.md
```

---

## Dataset

**FairCVdb** (Peña et al., 2020/2023): 24,000 synthetic candidate profiles.
- 8 structured competency features, a biography text, demographic attributes.
- Three label types: `blind_label` (used for training), `gender_biased`, `ethnicity_biased`
  (used only as EDA evidence of bias).
- Split: 19,200 train / 4,800 test.

> The dataset is not redistributed here. Place `FairCVdb.csv` in a `data/` folder to
> re-run the notebooks.

---

## Methodology

1. **Feature extraction** — SBERT (`all-MiniLM-L6-v2`, 384-dim) for text; 8 competency
   features for structured data; scaling fit on train only.
2. **Fusion strategies** — Early (concatenate), Late (weighted probability average),
   Weighted Hybrid (PCA-aligned feature combination).
3. **Classifiers** — Logistic Regression, Random Forest, MLP.
4. **Fairness metrics** — Demographic Parity Gap, Equal Opportunity Gap, Disparate Impact
   (four-fifths rule).
5. **Explainability** — global and per-group SHAP (gender, ethnicity).

---

## Reproduce the results

```bash
# 1. Run EDA
notebooks/FairCV_EDA.ipynb

# 2. Run models + fusion + fairness + Group SHAP
notebooks/FairCV_Models.ipynb
# -> exports fusion_results.csv, ablation_sbert.csv, SHAP figures
```

Fixed random seed (42) is used throughout for reproducibility.

---

## Run the application

The app reads a CV PDF, extracts 8 competency features (via Gemini + a normalization
rubric), scores candidates with the structured-only Logistic Regression model, ranks them,
explains the scores with SHAP, and lets a **recruiter make the final decision**.

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Gemini API key** (free at https://aistudio.google.com/apikey):
- Local: enter it in the sidebar, or
- Streamlit Cloud: add `GEMINI_API_KEY` under *Manage app → Settings → Secrets*.

**App pipeline:** Job description → upload CVs → extract features → predict → rank
Top 10 / Top 3 → LLM summary → SHAP explanation → chatbot → recruiter selects final
candidate (human-in-the-loop).

---

## Limitations

- FairCVdb is **synthetic**; results may not transfer directly to real hiring data.
- In the app, the 8 features are extracted from real CVs via an LLM and a **team-designed
  rubric**, not validated against the training distribution — predictions are a
  proof-of-concept.
- `suitability` is redefined as CV-to-role relevance (differs from the original
  sector-fixed definition in Peña et al.) and tends to dominate the score.
- The app uses the Gemini free tier (limited daily requests); large-scale use needs a paid API.

---

## Team

- Tran Loi Kieu Nhi
- Dong Bao Han
- Pham Hoang Tu Ngan

**Mentor:** Le Vo Minh Thu (HCMUS, VNU-HCM)

---

## References


---

## Acknowledgements

Built with scikit-learn, Sentence-Transformers, SHAP, Streamlit, and the Google Gemini API.
This project uses the ProxyAudit fairness framework (Le Vo Minh Thu et al.) as a theoretical
reference for interventional SHAP and fairness interpretation.
