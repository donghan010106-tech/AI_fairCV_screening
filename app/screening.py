"""
screening.py - Screening pipeline: many CVs -> predict -> rank -> Top 10 / Top 3

Uses extract_cv.py. Flow:
    CV PDFs -> extract 8 features (Gemini) -> LR model predicts score
            -> rank by score -> Top 10 -> Top 3 shortlist
            -> recruiter makes final choice (human-in-the-loop)

Model: Structured-Only Logistic Regression (best in notebook benchmark,
       F1=0.9658, AUC=0.9966).

Requires:
    - model_structured.pkl, scaler_structured.pkl  (saved from notebook)
    - extract_cv.py  (same folder)
    - pip install joblib scikit-learn
"""

import os
import joblib
import numpy as np
import pandas as pd

from extract_cv import process_cv, COMPETENCY, init_gemini


def load_model(model_path="model_structured.pkl",
               scaler_path="scaler_structured.pkl"):
    """Load the saved LR model + scaler. Looks for the .pkl files in the same
    folder as this script, so it works regardless of the working directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    # If only a filename is given, resolve it next to this script
    if not os.path.isabs(model_path):
        model_path = os.path.join(here, model_path)
    if not os.path.isabs(scaler_path):
        scaler_path = os.path.join(here, scaler_path)
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_one(features, model, scaler):
    """8 features [0,1] -> recommendation score [0,1] + label."""
    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    label = "Shortlisted" if prob >= 0.5 else "Not Shortlisted"
    return prob, label


def screen_cvs(pdf_paths, model, scaler, gemini_model, top_n=10, shortlist_n=3):
    """Run the full pipeline on a list of CV PDFs. Returns a ranked DataFrame."""
    rows = []
    for i, pdf in enumerate(pdf_paths, 1):
        name = os.path.basename(pdf)
        print(f"  [{i}/{len(pdf_paths)}] Processing: {name} ...")
        try:
            out = process_cv(pdf, gemini_model)
            prob, label = predict_one(out["features"], model, scaler)
            row = {
                "CV": name,
                "Role": out["detected_role"],
                "Industry": out["detected_industry"],
                "Score": round(prob, 4),
                "Decision": label,
            }
            for feat, val in zip(COMPETENCY, out["features"]):
                row[feat] = val
            rows.append(row)
        except Exception as e:
            print(f"     WARNING: failed on {name}: {e}")
            continue

    if not rows:
        raise RuntimeError("No CV could be processed.")

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    df["Top10"] = df["Rank"] <= top_n
    df["Top3"]  = df["Rank"] <= shortlist_n
    return df


if __name__ == "__main__":
    import glob, sys

    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not API_KEY:
        print("WARNING: No GEMINI_API_KEY.")
        sys.exit(1)

    cv_dir = sys.argv[1] if len(sys.argv) > 1 else "cvs"
    pdfs = sorted(glob.glob(os.path.join(cv_dir, "*.pdf")))
    if not pdfs:
        print(f"No PDF found in '{cv_dir}/'")
        sys.exit(1)

    print(f"Found {len(pdfs)} CVs. Starting screening...\n")
    model, scaler = load_model()
    gemini = init_gemini(API_KEY)

    result = screen_cvs(pdfs, model, scaler, gemini)

    print("\n" + "=" * 70)
    print("  RANKING RESULTS")
    print("=" * 70)
    cols = ["Rank", "CV", "Role", "Score", "Decision"]
    print(result[cols].to_string(index=False))

    print("\n  TOP 3 SHORTLIST (recommended to recruiter):")
    for _, r in result[result["Top3"]].iterrows():
        print(f"     #{r['Rank']}  {r['CV']}  ({r['Role']}) - score {r['Score']}")

    print("\n  NOTE: this is a RECOMMENDATION. The recruiter makes the final decision.")
    result.to_csv("screening_results.csv", index=False)
    print("\n  Saved screening_results.csv")
