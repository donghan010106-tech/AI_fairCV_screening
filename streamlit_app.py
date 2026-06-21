"""
streamlit_app.py - FairCV Resume Screening App

Full pipeline:
  1. Recruiter enters Job title + Job Description
  2. Upload CV PDFs
  3. Extract 8 features (Gemini)
  4. LR model predicts score
  5. Rank -> Top N -> Top 3
  6. LLM summarizes Top 10 + explains Top 3
  7. SHAP-style bar chart per candidate
  8. Chatbot explains score & fairness
  9. Recruiter selects the final candidate (human-in-the-loop)

Run:
    streamlit run streamlit_app.py

Same folder: extract_cv.py, screening.py, model_structured.pkl, scaler_structured.pkl
Install:
    pip install streamlit google-generativeai pypdf joblib scikit-learn matplotlib pandas
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from extract_cv import process_cv, init_gemini, COMPETENCY
from screening import load_model, predict_one

st.set_page_config(page_title="FairCV Screening", layout="wide", page_icon="📋")

GEMINI_MODEL = "gemini-2.5-flash-lite"

FEATURE_LABELS = {
    "suitability":     "Job Suitability",
    "educ_attainment": "Education",
    "prev_experience": "Experience",
    "recommendation":  "References",
    "availability":    "Availability",
    "lang_prof_1":     "English",
    "lang_prof_2":     "Language 2",
    "lang_prof_3":     "Language 3",
}


@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_gemini(api_key):
    return init_gemini(api_key, model_name=GEMINI_MODEL)


# ---------------------------------------------------------------- SHAP
def feature_contributions(features, model, scaler):
    X = np.array(features, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)[0]
    coef = model.coef_[0]
    contribs = coef * Xs
    pairs = [(FEATURE_LABELS.get(f, f), float(c))
             for f, c in zip(COMPETENCY, contribs)]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs


def plot_contributions(pairs, cv_name):
    labels = [p[0] for p in pairs][::-1]
    vals   = [p[1] for p in pairs][::-1]
    colors = ["#2E9E5B" if v >= 0 else "#C2384A" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Contribution to score  (green = raises, red = lowers)")
    ax.set_title(f"Why this score? — {cv_name}", fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------- LLM summary
def summarize_top(gemini_model, df, job_title, top_k=10, shortlist_k=3):
    top = df.head(top_k)
    lines = []
    for _, r in top.iterrows():
        lines.append(f"#{r['Rank']} {r['CV']} | role: {r['Role']} | "
                     f"score: {r['Score']:.3f} | {r['Decision']}")
    table = "\n".join(lines)
    prompt = f"""You are an HR assistant for a {job_title} hiring process.
Below is the AI-ranked Top {top_k} candidates (score 0-1, higher = better fit).

{table}

Write a concise summary in two parts:
1. A 2-3 sentence overview of the candidate pool quality.
2. Recommend the TOP {shortlist_k} candidates to shortlist. For each, give ONE short
   sentence on why (based on role relevance and score).

Be honest and concise. Remember the recruiter makes the final decision; this is advisory."""
    return gemini_model.generate_content(prompt).text


# ------------------------------------------------------------- Chatbot
def build_chat_context(row):
    feat_lines = "\n".join(
        f"  - {FEATURE_LABELS.get(f, f)}: {row[f]:.2f}" for f in COMPETENCY)
    return (f"Candidate: {row['CV']}\nDetected role: {row['Role']}\n"
            f"Detected industry: {row['Industry']}\n"
            f"Final score: {row['Score']:.4f}  ({row['Decision']})\n"
            f"8 competency features (0=low, 1=high):\n{feat_lines}")


def ask_chatbot(gemini_model, context, question, job_title):
    prompt = f"""You are an HR assistant explaining an AI resume-screening result for a
{job_title} position. Be concise, clear, and honest.

System facts:
- The model scores candidates from 8 structured competency features (each 0-1).
- 'Job Suitability' measures how relevant the candidate's background is to the role.
- The model was trained on the FairCVdb fairness benchmark as a 'blind' model:
  it never sees gender or ethnicity, so it does not use protected attributes directly.
- The recruiter always makes the final hiring decision (human-in-the-loop).

CANDIDATE RESULT:
{context}

RECRUITER QUESTION: {question}

Answer in 3-6 sentences. For fairness questions, explain the model does not use
gender/ethnicity, the score reflects only the competency features, and name which
features most affected this candidate."""
    return gemini_model.generate_content(prompt).text


# ====================================================================== UI
st.title("📋 FairCV — AI Resume Screening")
st.caption("Recruiter-assist tool. The model is blind to gender/ethnicity. "
           "Final decision is always the recruiter's.")

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API key", type="password",
                                help="Free at aistudio.google.com/apikey")
threshold = st.sidebar.slider("Shortlist threshold", 0.0, 1.0, 0.5, 0.05)
top_n = st.sidebar.number_input("Top N to show", 3, 50, 10)

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start.")
    st.stop()

try:
    model, scaler = get_model()
    gemini = get_gemini(api_key)
except Exception as e:
    st.error(f"Could not load model or Gemini: {e}")
    st.stop()

# --- Step 1: Job description ---
st.subheader("1️⃣ Job posting")
colj1, colj2 = st.columns([1, 2])
with colj1:
    job_title = st.text_input("Job title", value="Data Engineer")
with colj2:
    jd_text = st.text_area("Job description (optional, improves relevance)",
                           height=80,
                           placeholder="Paste the JD here: required skills, tools, experience...")

# --- Step 2: Upload ---
st.subheader("2️⃣ Upload CVs")
uploaded = st.file_uploader("CV PDFs", type="pdf", accept_multiple_files=True)

if uploaded and st.button("▶ Run screening", type="primary"):
    rows = []
    progress = st.progress(0.0, text="Processing CVs...")
    for i, uf in enumerate(uploaded, 1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue()); tmp_path = tmp.name
        try:
            out = process_cv(tmp_path, gemini, job_title=job_title, jd_text=jd_text)
            prob, _ = predict_one(out["features"], model, scaler)
            decision = "Shortlisted" if prob >= threshold else "Not Shortlisted"
            row = {"CV": uf.name, "Role": out["detected_role"],
                   "Industry": out["detected_industry"],
                   "Score": round(prob, 4), "Decision": decision}
            for f, v in zip(COMPETENCY, out["features"]):
                row[f] = v
            rows.append(row)
        except Exception as e:
            st.warning(f"Failed on {uf.name}: {e}")
        finally:
            os.unlink(tmp_path)
        progress.progress(i / len(uploaded), text=f"Processed {i}/{len(uploaded)}")
    progress.empty()

    if not rows:
        st.error("No CV could be processed."); st.stop()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    st.session_state["result"] = df
    st.session_state["job_title"] = job_title
    st.session_state.pop("summary", None)      # reset old summary
    st.session_state.pop("final_pick", None)

# --- Results ---
if "result" in st.session_state:
    df = st.session_state["result"]
    job_title = st.session_state.get("job_title", "Data Engineer")

    st.subheader("🏆 Ranking")
    show_cols = ["Rank", "CV", "Role", "Industry", "Score", "Decision"]
    st.dataframe(df[show_cols].head(int(top_n)), use_container_width=True, hide_index=True)

    # --- LLM summary of Top 10 + Top 3 ---
    st.subheader("🤖 AI summary & shortlist recommendation")
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            try:
                st.session_state["summary"] = summarize_top(
                    gemini, df, job_title, top_k=min(10, len(df)))
            except Exception as e:
                st.error(f"Summary error: {e}")
    if "summary" in st.session_state:
        st.markdown(st.session_state["summary"])
    st.caption("⚠️ Advisory only. The recruiter makes the final decision.")

    st.divider()

    # --- Explanation ---
    st.subheader("🔍 Explain a candidate")
    pick = st.selectbox("Choose a CV", df["CV"].tolist())
    row = df[df["CV"] == pick].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        feats = [row[f] for f in COMPETENCY]
        st.pyplot(plot_contributions(feature_contributions(feats, model, scaler), pick))
    with c2:
        st.markdown("**Feature values**")
        st.dataframe(pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in COMPETENCY],
            "Value":   [row[f] for f in COMPETENCY]}),
            use_container_width=True, hide_index=True)

    # --- Chatbot ---
    st.subheader("💬 Ask about this result")
    st.caption(f"Discussing: **{pick}**")
    q = st.text_input("Your question",
                      placeholder="e.g. Why is this score low? Is it fair?")
    if q:
        with st.spinner("Thinking..."):
            try:
                st.markdown(ask_chatbot(gemini, build_chat_context(row), q, job_title))
            except Exception as e:
                st.error(f"Chatbot error: {e}")

    st.divider()

    # --- Step: recruiter final decision (human-in-the-loop) ---
    st.subheader("✅ Final decision (recruiter)")
    st.caption("The AI only recommends. You choose who to hire.")
    final = st.selectbox("Select the candidate you choose to advance",
                         ["— none yet —"] + df["CV"].tolist())
    if final != "— none yet —":
        if st.button("Confirm selection", type="primary"):
            st.session_state["final_pick"] = final
    if st.session_state.get("final_pick"):
        fr = df[df["CV"] == st.session_state["final_pick"]].iloc[0]
        st.success(f"✔ Recruiter selected: **{fr['CV']}** "
                   f"({fr['Role']}) — score {fr['Score']:.3f}. "
                   "Decision recorded by a human, not the AI.")
