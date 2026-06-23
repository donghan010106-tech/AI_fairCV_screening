"""
streamlit_app.py - FairCV Resume Screening App (multi-model, fairness trade-off)

Pipeline:
  1. Recruiter enters the target position (flexible)
  2. Upload CV PDFs
  3. Gemini extracts 7 features + writes a FairCVdb-style bio
  4. SBERT computes suitability = similarity(role, target position)
  5. Main model (Structured LR) predicts score -> rank -> Top N -> Top 3
  6. Model comparison: Structured LR / Structured RF / Early Fusion RF (SBERT)
     highlighting the accuracy vs fairness trade-off
  7. SHAP bar chart per candidate
  8. Chatbot explains score & fairness
  9. Recruiter selects the final candidate (human-in-the-loop)

Same folder: extract_cv.py, fusion_predict.py, and the 4 .pkl files
Install:
    pip install streamlit google-generativeai pypdf joblib scikit-learn
                matplotlib pandas sentence-transformers numpy
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from extract_cv import process_cv, init_gemini, COMPETENCY
from fusion_predict import predict, role_suitability, get_scaler, get_model, MODEL_INFO

st.set_page_config(page_title="FairCV Screening", layout="wide", page_icon="📋")

GEMINI_MODEL = "gemini-2.5-flash"

FEATURE_LABELS = {
    "suitability": "Job Suitability", "educ_attainment": "Education",
    "prev_experience": "Experience", "recommendation": "References",
    "availability": "Availability", "lang_prof_1": "English",
    "lang_prof_2": "Language 2", "lang_prof_3": "Language 3",
}


@st.cache_resource
def get_gemini(api_key):
    return init_gemini(api_key, model_name=GEMINI_MODEL)


# ---- SHAP-style explanation (Structured LR = linear, exact) ----------
def feature_contributions(features):
    model = get_model("struct_lr")
    scaler = get_scaler()
    Xs = scaler.transform(np.array(features, dtype=float).reshape(1, -1))[0]
    contribs = model.coef_[0] * Xs
    pairs = [(FEATURE_LABELS.get(f, f), float(c))
             for f, c in zip(COMPETENCY, contribs)]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs


def plot_contributions(pairs, cv_name):
    labels = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    colors = ["#2E9E5B" if v >= 0 else "#C2384A" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Contribution to score (green = raises, red = lowers)")
    ax.set_title(f"What drives this score? — {cv_name}", fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ---- Chatbot ---------------------------------------------------------
def ask_chatbot(gemini_model, row, question, target):
    feat_lines = "\n".join(
        f"  - {FEATURE_LABELS.get(f, f)}: {row[f]:.2f}" for f in COMPETENCY)
    prompt = f"""You are an HR assistant explaining an AI resume-screening result for the
position: {target}. Be concise, clear, and honest.

System facts:
- The score comes from 8 competency features (each 0-1).
- 'Job Suitability' is the semantic similarity between the candidate's role and the
  target position (computed with Sentence-BERT).
- The model is trained on FairCVdb as a 'blind' model: it never sees gender or
  ethnicity, so it does not use protected attributes directly.
- The recruiter always makes the final hiring decision (human-in-the-loop).

CANDIDATE: {row['CV']} | role: {row['Role']} | score: {row['Score']:.3f} ({row['Decision']})
Features:
{feat_lines}

QUESTION: {question}

Answer in 3-6 sentences. For fairness questions, explain the model does not use
gender/ethnicity and name which features most affected this candidate."""
    return gemini_model.generate_content(prompt).text


# ====================================================================== UI
st.title("📋 FairCV — AI Resume Screening")
st.caption("Blind to gender/ethnicity · explainable · the recruiter decides. "
           "Compares structured vs. fusion models and their fairness trade-off.")

st.sidebar.header("Settings")
api_key = ""
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""
if not api_key:
    api_key = st.sidebar.text_input("Gemini API key", type="password")
threshold = st.sidebar.slider("Shortlist threshold", 0.0, 1.0, 0.5, 0.05)
top_n = st.sidebar.number_input("Top N to show", 2, 50, 10)

if not api_key:
    st.info("👈 Enter your Gemini API key to start.")
    st.stop()

try:
    gemini = get_gemini(api_key)
except Exception as e:
    st.error(f"Gemini init failed: {e}")
    st.stop()

# --- Step 1: target position (flexible) ---
st.subheader("1️⃣ Position you are hiring for")
target = st.text_input("Target position", value="Data Engineer")

# --- Step 2: upload ---
st.subheader("2️⃣ Upload CVs")
uploaded = st.file_uploader("CV PDFs", type="pdf", accept_multiple_files=True)

if uploaded and st.button("▶ Run screening", type="primary"):
    rows = []
    progress = st.progress(0.0, text="Processing CVs...")
    for i, uf in enumerate(uploaded, 1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue()); tmp_path = tmp.name
        try:
            out = process_cv(tmp_path, gemini, job_title=target)
            # suitability via SBERT (overrides Gemini's guess)
            suit = role_suitability(out["detected_role"], target)
            feats = out["features"].copy()
            feats[0] = round(suit, 4)              # index 0 = suitability

            # main score from Structured LR
            prob, _ = predict(feats, "struct_lr")
            decision = "Shortlisted" if prob >= threshold else "Not Shortlisted"

            # model scores: LR + Early Fusion RF only
            score_lr = prob
            # score_rf, _ = predict(feats, "struct_rf")  # ← REMOVED
            score_ef, _ = predict(feats, "early_rf", bio_text=out.get("bio_summary", ""))

            row = {"CV": uf.name, "Role": out["detected_role"],
                   "Industry": out["detected_industry"],
                   "Score": round(prob, 4), "Decision": decision,
                   "bio": out.get("bio_summary", ""),
                   "score_lr": round(score_lr, 4),
                   # "score_rf": round(score_rf, 4),  # ← REMOVED
                   "score_ef": round(score_ef, 4)}
            for f, v in zip(COMPETENCY, feats):
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
    st.session_state["target"] = target
    st.session_state.pop("final_pick", None)

# ---- Results ----
if "result" in st.session_state:
    df = st.session_state["result"]
    target = st.session_state.get("target", "Data Engineer")

    # --- Simple view: ranking (for everyone) ---
    st.subheader("🏆 Ranking")
    st.caption(f"Scored for: **{target}**. Higher = better fit.")
    st.dataframe(df[["Rank", "CV", "Role", "Score", "Decision"]].head(int(top_n)),
                 use_container_width=True, hide_index=True)

    st.subheader("⭐ Top 3 Shortlist (recommended)")
    for _, r in df.head(3).iterrows():
        st.markdown(f"**#{r['Rank']} · {r['CV']}** — {r['Role']} · "
                    f"score **{r['Score']:.3f}** · {r['Decision']}")
    st.caption("⚠️ Advisory. The recruiter makes the final decision.")

    st.divider()

    # --- Model comparison: accuracy vs fairness trade-off (for experts) ---
    with st.expander("🔬 Model comparison — accuracy vs. fairness trade-off"):
        st.markdown("Two models scored every candidate. They trade off differently: "
                    "the structured model is most **accurate**, while the SBERT fusion "
                    "model is the **fairest** on gender.")
        comp = pd.DataFrame([
            {"Model": MODEL_INFO["struct_lr"]["name"], "Accuracy": MODEL_INFO["struct_lr"]["acc"],
             "F1": MODEL_INFO["struct_lr"]["f1"], "DP Gap (Gender)": MODEL_INFO["struct_lr"]["dp_gender"]},
            # {"Model": MODEL_INFO["struct_rf"]["name"], "Accuracy": MODEL_INFO["struct_rf"]["acc"],
            #  "F1": MODEL_INFO["struct_rf"]["f1"], "DP Gap (Gender)": MODEL_INFO["struct_rf"]["dp_gender"]},
            {"Model": MODEL_INFO["early_rf"]["name"], "Accuracy": MODEL_INFO["early_rf"]["acc"],
             "F1": MODEL_INFO["early_rf"]["f1"], "DP Gap (Gender)": MODEL_INFO["early_rf"]["dp_gender"]},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.markdown("- **Most accurate:** Structured-Only LR (F1 0.9658)\n"
                    "- **Fairest on gender:** Early Fusion RF (DP gap 0.0019) — but lower accuracy\n"
                    "- This is the **fairness–accuracy trade-off**: adding SBERT text fusion "
                    "improves gender fairness but reduces accuracy.")

        # per-candidate: 2 model scores
        st.markdown("**Per-candidate scores from each model:**")
        show = df[["Rank", "CV", "score_lr", "score_ef"]].copy()
        show.columns = ["Rank", "CV", "Structured LR", "Early Fusion RF"]
        st.dataframe(show.head(int(top_n)), use_container_width=True, hide_index=True)

    st.divider()

    # --- Explain a candidate ---
    st.subheader("🔍 Explain a candidate")
    pick = st.selectbox("Choose a CV", df["CV"].tolist())
    row = df[df["CV"] == pick].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(plot_contributions(feature_contributions([row[f] for f in COMPETENCY]), pick))
    with c2:
        st.markdown("**Feature values**")
        st.dataframe(pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in COMPETENCY],
            "Value": [row[f] for f in COMPETENCY]}),
            use_container_width=True, hide_index=True)
        if row.get("bio"):
            st.caption(f"📝 Generated bio (FairCVdb style): {row['bio'][:200]}")

    # --- Chatbot ---
    st.subheader("💬 Ask about this result")
    st.caption(f"Discussing: **{pick}**")
    q = st.text_input("Your question",
                      placeholder="e.g. Why this score? Is it fair? What if two CVs tie?")
    if q:
        with st.spinner("Thinking..."):
            try:
                st.markdown(ask_chatbot(gemini, row, q, target))
            except Exception as e:
                st.error(f"Chatbot error: {e}")

    st.divider()

    # --- Recruiter final decision ---
    st.subheader("✅ Final decision (recruiter)")
    st.caption("The AI only recommends. You choose who to hire.")
    final = st.selectbox("Select the candidate you advance",
                         ["— none yet —"] + df["CV"].tolist())
    if final != "— none yet —" and st.button("Confirm selection", type="primary"):
        st.session_state["final_pick"] = final
    if st.session_state.get("final_pick"):
        fr = df[df["CV"] == st.session_state["final_pick"]].iloc[0]
        st.success(f"✔ Recruiter selected: **{fr['CV']}** ({fr['Role']}) — "
                   f"score {fr['Score']:.3f}. Decision made by a human, not the AI.")
