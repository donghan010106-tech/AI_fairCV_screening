# FairCV — A Comparative Study of Fusion Strategies for Fair AI-Based Resume Evaluation

Capstone project (DAP391m). A reproducible study comparing **Early, Late, and Weighted
Hybrid Fusion** for fair resume screening on the **FairCVdb** benchmark (24,000 synthetic
profiles, Peña et al. 2020/2023), plus an interactive recruiter-assist application with
SHAP explanations and a human-in-the-loop workflow.

**Position studied:** Data Engineer · **Sensitive attributes:** gender, ethnicity

---

## Pipeline

![FairCV pipeline](docs/architecture_diagram.png)

From data sources → preprocessing (structured + text streams) → models (LR/RF/MLP) and
fusion architectures → fairness evaluation core → XAI → reports & application output.

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
│   ├── FairCV_EDA.ipynb          # Exploratory data analysis
│   └── FairCV_Models.ipynb       # Models, fusion, fairness metrics, Group SHAP
├── app/
│   ├── streamlit_app.py          # Recruiter-assist screening app
│   ├── extract_cv.py             # CV PDF -> 8 features (Gemini + rubric)
│   ├── screening.py              # Predict + rank Top 10 / Top 3
│   ├── model_structured.pkl      # Trained Logistic Regression model
│   ├── scaler_structured.pkl     # Feature scaler
│   └── requirements.txt
├── results/
│   ├── fusion_results.csv        # RQ1 + RQ2: performance & fairness
│   ├── ablation_sbert.csv        # RQ3: SBERT ablation
│   ├── bias_mitigation_results.csv  # RQ4: mitigation experiments
│   └── figures/                  # SHAP plots (gender, ethnicity)
├── docs/
│   └── architecture_diagram.png  # End-to-end pipeline
├── paper/                        # (to be added)
└── README.md
```

---

## Dataset

**FairCVdb** (Peña et al., 2020/2023): 24,000 synthetic candidate profiles.
- 8 structured competency features, a biography text, demographic attributes.
- Three label types: `blind_label` (used for training), `gender_biased`, `ethnicity_biased`
  (used only as EDA evidence of bias).
- Split: 19,200 train / 4,800 test. Binarized at the median of `blind_label`.

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

## Normalization rubric (CV → 8 competency features)

In the application, real CVs are read by an LLM and mapped to the 8 competency features,
each scaled to **[0, 1]**. The LLM only extracts **raw facts**; the scores are computed in
Python from the rubric below (so the numbers are deterministic, not guessed by the LLM).

| # | Feature | What is read | Mapping to [0, 1] |
|---|---------|--------------|-------------------|
| 1 | **suitability** | Candidate's industry/role relevance to **Data Engineer** | very_high=1.0, high=0.85, medium=0.45, low=0.25, very_low=0.10 (refined by data-related keywords: SQL/Python/ETL/cloud) |
| 2 | **educ_attainment** | Highest completed degree | PhD=1.0, Master=0.85, Bachelor=0.65, Associate=0.45, Highschool=0.25, coursework-only=0.45, none=0.0 |
| 3 | **prev_experience** | Years of experience (earliest → most recent job) | `min(years / 10, 1.0)` — capped at 10 years = 1.0 |
| 4 | **recommendation** | Whether the CV has a references section | has references = 1.0, none = 0.0 |
| 5 | **availability** | Stated availability | immediate=1.0, ≤1 month=0.7, ≤3 months=0.4, >3 months=0.2, not stated=0.5 |
| 6 | **lang_prof_1** | English proficiency (inferred from writing if absent) | native=1.0, fluent=0.8, intermediate=0.5, basic=0.25, none=0.0 |
| 7 | **lang_prof_2** | Strongest other language stated | same scale as English; none=0.0 |
| 8 | **lang_prof_3** | Second other language stated | same scale as English; none=0.0 |

**Industry-to-Data-Engineer suitability reference** (used as a baseline for feature 1):
IT=1.0, Engineering=0.85, Finance/Banking/Business-Dev/Digital-Media=0.45,
Accountant/Consultant=0.40, Sales/BPO/Aviation/Automobile=0.30,
Designer/PR/HR/Construction=0.25, Arts/Apparel/Agriculture/Healthcare=0.20,
Advocate/Teacher=0.15, Chef/Fitness=0.10.

> This rubric is **team-designed** and not validated against the FairCVdb training
> distribution, so app predictions on real CVs are a proof-of-concept (see Limitations).

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

The app reads a CV PDF, extracts the 8 competency features (via the rubric above), scores
candidates with the structured-only Logistic Regression model, ranks them, explains the
scores with SHAP, and lets a **recruiter make the final decision**.

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**App pipeline:** upload CVs → extract features → predict → rank Top 10 / Top 3 →
LLM summary → SHAP explanation → chatbot → recruiter selects final candidate
(human-in-the-loop). The position is fixed to **Data Engineer**.

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

1. Peña et al. *Bias in multimodal AI: Testbed for fair automatic recruitment.* CVPRW 2020.
2. Peña et al. *Human-centric multimodal ML: AI-based recruitment.* SN Computer Science, 2023.
3. Swati, Roy, Ntoutsi. *Exploring Fusion Techniques in Multimodal AI-Based Recruitment: FairCVdb.* EWAF 2024.
4. Hardt, Price, Srebro. *Equality of Opportunity in Supervised Learning.* NeurIPS 2016.
5. Lundberg & Lee. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.

---

## Acknowledgements

Built with scikit-learn, Sentence-Transformers, SHAP, Streamlit, and the Google Gemini API.
This project uses the ProxyAudit fairness framework (Le Vo Minh Thu et al.) as a theoretical
reference for interventional SHAP and fairness interpretation.
