import streamlit as st
import pandas as pd
import io
from components.model_backend import (
    FEATURE_NAMES, OCCUPATION_SUITABILITY, predict, fairness_check
)

REQUIRED_COLS = ['occupation', 'gender', 'ethnicity',
                 'educ_attainment', 'prev_experience', 'recommendation',
                 'availability', 'lang_prof_1', 'lang_prof_2', 'lang_prof_3']

def render_recruiter(model_choice: str):
    st.markdown("""
    <div class="app-hero">
        <div class="app-title">Recruiter Batch Screening</div>
        <p class="app-sub">Upload a CSV of candidates. The system scores all,
        then runs a fairness audit across gender and ethnicity groups.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Template download ─────────────────────────────────────────
    st.markdown('<div class="section-hdr">Step 1 — Download CSV Template</div>',
                unsafe_allow_html=True)

    sample = pd.DataFrame([
        ['Teacher',   'Male',   'G1', 0.8, 0.7, 0.75, 0.9, 0.8, 0.7, 0.75],
        ['Nurse',     'Female', 'G2', 0.7, 0.6, 0.65, 0.8, 0.7, 0.8, 0.70],
        ['Attorney',  'Male',   'G3', 0.6, 0.8, 0.70, 0.7, 0.6, 0.7, 0.65],
        ['Journalist','Female', 'G1', 0.5, 0.5, 0.55, 0.6, 0.5, 0.6, 0.55],
    ], columns=REQUIRED_COLS)

    csv_bytes = sample.to_csv(index=False).encode()
    st.download_button(
        label="Download template CSV",
        data=csv_bytes,
        file_name="faircv_candidate_template.csv",
        mime="text/csv",
    )

    with st.expander("Required columns"):
        cols_info = [
            ("occupation",      "One of 10 occupations (Teacher, Nurse, Surgeon, etc.)"),
            ("gender",          "Male or Female"),
            ("ethnicity",       "G1, G2, or G3"),
            ("educ_attainment", "0.0 – 1.0"),
            ("prev_experience", "0.0 – 1.0"),
            ("recommendation",  "0.0 – 1.0"),
            ("availability",    "0.0 – 1.0"),
            ("lang_prof_1/2/3", "0.0 – 1.0 each"),
        ]
        for col, desc in cols_info:
            st.markdown(f"`{col}` — {desc}")

    # ── Upload ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Step 2 — Upload Candidates CSV</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type="csv",
                                 label_visibility="collapsed")

    if uploaded is None:
        st.markdown("""
        <div class="amber-box">
            No file uploaded yet. Download the template above,
            fill in your candidates, then upload here.
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    # ── Run predictions ───────────────────────────────────────────
    scores, labels, verdicts = [], [], []
    for _, row in df.iterrows():
        suitability = OCCUPATION_SUITABILITY.get(row['occupation'], 0.5)
        feats = {
            'suitability':    suitability,
            'educ_attainment': float(row['educ_attainment']),
            'prev_experience': float(row['prev_experience']),
            'recommendation':  float(row['recommendation']),
            'availability':    float(row['availability']),
            'lang_prof_1':     float(row['lang_prof_1']),
            'lang_prof_2':     float(row['lang_prof_2']),
            'lang_prof_3':     float(row['lang_prof_3']),
        }
        s, l, _ = predict(feats, model_choice)
        scores.append(round(s, 4))
        labels.append(l)
        verdicts.append("Recommended" if l else "Not Recommended")

    df['score']    = scores
    df['verdict']  = verdicts

    # ── Results table ─────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Step 3 — Screening Results</div>',
                unsafe_allow_html=True)

    rec_count = sum(labels)
    rej_count = len(labels) - rec_count
    c1, c2, c3 = st.columns(3)
    for col, val, label, color in [
        (c1, len(df), "Total Candidates", "#6b7280"),
        (c2, rec_count, "Recommended", "#4cda8f"),
        (c3, rej_count, "Not Recommended", "#ff5c7c"),
    ]:
        with col:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def style_verdict(val):
        if val == "Recommended":
            return "color:#4cda8f;font-weight:600;"
        return "color:#ff5c7c;font-weight:600;"

    display_cols = ['occupation', 'gender', 'ethnicity', 'score', 'verdict']
    styled = df[display_cols].style.applymap(
        style_verdict, subset=['verdict']
    ).format({'score': '{:.4f}'})
    st.dataframe(styled, use_container_width=True)

    # ── Fairness Audit ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Step 4 — Fairness Audit</div>',
                unsafe_allow_html=True)

    tab_gender, tab_eth = st.tabs(["Gender Fairness", "Ethnicity Fairness"])

    with tab_gender:
        gender_groups = {}
        for g in df['gender'].unique():
            gender_groups[g] = df[df['gender'] == g]['score'].tolist()
        result_g = fairness_check(gender_groups)
        _render_fairness_result(result_g, "Gender")

    with tab_eth:
        eth_groups = {}
        for g in df['ethnicity'].unique():
            eth_groups[g] = df[df['ethnicity'] == g]['score'].tolist()
        result_e = fairness_check(eth_groups)
        _render_fairness_result(result_e, "Ethnicity")

    # ── Export ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Step 5 — Export Results</div>',
                unsafe_allow_html=True)
    st.download_button(
        label="Download results CSV",
        data=df.to_csv(index=False).encode(),
        file_name="faircv_screening_results.csv",
        mime="text/csv",
    )


def _render_fairness_result(result: dict, group_type: str):
    if not result:
        st.warning("Need at least 2 groups to compute fairness metrics.")
        return

    dp_gap = result['dp_gap']
    di     = result['di']
    eeoc   = result['eeoc_pass']

    c1, c2, c3 = st.columns(3)
    for col, val, label, good_thresh, lower_better in [
        (c1, f"{dp_gap:.4f}", "DP Gap",           0.05,  True),
        (c2, f"{di:.4f}",     "Disparate Impact",  0.80,  False),
        (c3, "PASS" if eeoc else "FAIL", "EEOC 4/5 Rule", None, None),
    ]:
        with col:
            if label == "EEOC 4/5 Rule":
                badge = "fairness-badge-pass" if eeoc else "fairness-badge-fail"
                col.markdown(f"""
                <div class="metric-card">
                    <div style="margin:0.4rem 0;">
                        <span class="{badge}">{val}</span>
                    </div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                color = "#4cda8f" if (lower_better and float(val) < good_thresh) or \
                                      (not lower_better and float(val) >= good_thresh) \
                               else "#ff5c7c"
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:{color};">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Positive rate per group
    st.markdown(f"**Positive recommendation rate by {group_type}:**")
    for group, rate in result['pos_rates'].items():
        pct = int(rate * 100)
        bar_color = "#00c9a7" if pct >= 45 else "#ff5c7c"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;width:80px;">{group}</div>
            <div style="flex:1;height:20px;background:#1e2333;border-radius:4px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;"></div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                        color:{bar_color};width:40px;">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    if not eeoc:
        st.markdown(f"""
        <div class="warn-box">
            <strong>EEOC Violation detected.</strong>
            {result['min_group']} has a positive rate below 80% of
            {result['max_group']} (DI = {di:.3f}).
            Consider reviewing the candidate pool or applying bias mitigation.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            No EEOC violation. All groups pass the 4/5 Disparate Impact threshold
            (DI = {di:.3f} >= 0.80).
        </div>
        """, unsafe_allow_html=True)
