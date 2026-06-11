import streamlit as st
from components.model_backend import (
    FEATURE_NAMES, OCCUPATION_SUITABILITY, predict
)

def render_shap_bar(feat_name, shap_val, max_abs=0.15):
    pct = min(abs(shap_val) / max_abs * 100, 100)
    color_cls = "shap-fill-pos" if shap_val >= 0 else "shap-fill-neg"
    sign  = f"+{shap_val:.4f}" if shap_val >= 0 else f"{shap_val:.4f}"
    color = "#00c9a7" if shap_val >= 0 else "#ff5c7c"
    st.markdown(f"""
    <div class="shap-row">
        <div class="shap-feat">{feat_name}</div>
        <div class="shap-bar-wrap">
            <div class="{color_cls}" style="width:{pct}%;"></div>
        </div>
        <div class="shap-val" style="color:{color};">{sign}</div>
    </div>
    """, unsafe_allow_html=True)


def render_screener(model_choice: str):
    st.markdown("""
    <div class="app-hero">
        <div class="app-title">CV Screening</div>
        <p class="app-sub">Enter candidate competency scores below.
        The model predicts hiring recommendation and explains each factor's contribution.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Candidate Information</div>', unsafe_allow_html=True)

    col_occ, col_gender, col_eth = st.columns(3)
    with col_occ:
        occupation = st.selectbox("Occupation", list(OCCUPATION_SUITABILITY.keys()))
    with col_gender:
        gender = st.selectbox("Gender (for fairness audit only)", ["Male", "Female"])
    with col_eth:
        ethnicity = st.selectbox("Ethnicity (for fairness audit only)", ["G1", "G2", "G3"])

    suitability = OCCUPATION_SUITABILITY[occupation]
    st.markdown(f"""
    <div class="insight-box">
        Suitability score auto-assigned from occupation:
        <strong style="color:#00c9a7;font-family:'IBM Plex Mono',monospace;">
        {occupation} → {suitability:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Competency Scores (0.0 — 1.0)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        educ   = st.slider("Education Attainment",   0.0, 1.0, 0.7, 0.05)
        exp    = st.slider("Previous Experience",     0.0, 1.0, 0.6, 0.05)
        rec    = st.slider("Recommendation Quality",  0.0, 1.0, 0.6, 0.05)
        avail  = st.slider("Availability",            0.0, 1.0, 0.8, 0.05)
    with c2:
        lang1  = st.slider("Language Proficiency 1",  0.0, 1.0, 0.7, 0.05)
        lang2  = st.slider("Language Proficiency 2",  0.0, 1.0, 0.6, 0.05)
        lang3  = st.slider("Language Proficiency 3",  0.0, 1.0, 0.65, 0.05)

    features = {
        'suitability':    suitability,
        'educ_attainment': educ,
        'prev_experience': exp,
        'recommendation':  rec,
        'availability':    avail,
        'lang_prof_1':     lang1,
        'lang_prof_2':     lang2,
        'lang_prof_3':     lang3,
    }

    # ── Run prediction ────────────────────────────────────────────
    run = st.button("Run Screening", use_container_width=False)

    if run or st.session_state.get('last_result'):
        if run:
            score, label, shap_vals = predict(features, model_choice)
            st.session_state['last_result'] = {
                'score': score, 'label': label,
                'shap': shap_vals, 'features': features,
                'gender': gender, 'ethnicity': ethnicity,
                'occupation': occupation,
            }

        res = st.session_state['last_result']
        score   = res['score']
        label   = res['label']
        shap_vals = res['shap']

        st.markdown('<div class="section-hdr">Screening Result</div>', unsafe_allow_html=True)

        # Score + Verdict
        c_score, c_verdict, c_info = st.columns([1, 2, 2])
        with c_score:
            pct = int(score * 100)
            ring_color = "#00c9a7" if label else "#ff5c7c"
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:12px;padding:1.4rem 1rem;text-align:center;">
                <div style="width:110px;height:110px;border-radius:50%;
                             border:4px solid {ring_color};
                             background:{'rgba(0,201,167,0.1)' if label else 'rgba(255,92,124,0.1)'};
                             display:flex;flex-direction:column;
                             align-items:center;justify-content:center;margin:0 auto;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                                font-weight:600;color:{ring_color};line-height:1;">{pct}</div>
                    <div style="font-size:0.6rem;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-top:2px;">score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_verdict:
            if label:
                st.markdown("""
                <div class="verdict-rec" style="margin-top:0.3rem;">
                    <div class="verdict-text" style="color:#4cda8f;">RECOMMENDED</div>
                    <div style="font-size:0.8rem;color:#6b7280;margin-top:0.3rem;">
                        Score exceeds hiring threshold (0.50)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-rej" style="margin-top:0.3rem;">
                    <div class="verdict-text" style="color:#ff5c7c;">NOT RECOMMENDED</div>
                    <div style="font-size:0.8rem;color:#6b7280;margin-top:0.3rem;">
                        Score below hiring threshold (0.50)</div>
                </div>
                """, unsafe_allow_html=True)

        with c_info:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:10px;padding:1.1rem 1.3rem;font-size:0.85rem;line-height:1.9;">
                <span style="color:#6b7280;">Occupation</span><br>
                <strong style="font-family:'IBM Plex Mono',monospace;">{res['occupation']}</strong><br>
                <span style="color:#6b7280;">Model used</span><br>
                <strong style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;">{model_choice}</strong>
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP Explanation ──────────────────────────────────────
        st.markdown('<div class="section-hdr">Why this decision? — Feature Contributions</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            Each bar shows how much a feature pushed the score toward
            <strong style="color:#00c9a7;">Recommended (+)</strong> or
            <strong style="color:#ff5c7c;">Not Recommended (−)</strong>.
        </div>
        """, unsafe_allow_html=True)

        sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        max_abs = max(abs(v) for _, v in sorted_shap) or 0.01

        for feat, val in sorted_shap:
            render_shap_bar(feat, val, max_abs)

        # ── Top driver summary ────────────────────────────────────
        top_pos = [(f,v) for f,v in sorted_shap if v > 0]
        top_neg = [(f,v) for f,v in sorted_shap if v < 0]

        st.markdown('<div class="section-hdr">Improvement Suggestions</div>', unsafe_allow_html=True)
        c_pos, c_neg = st.columns(2)
        with c_pos:
            if top_pos:
                items = "".join(f"<li><strong>{f}</strong> — contributing positively</li>"
                                for f, _ in top_pos[:3])
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Strengths</strong><br>
                    <ul style="margin:0.5rem 0 0 1rem;padding:0;font-size:0.85rem;line-height:1.8;">
                    {items}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        with c_neg:
            if top_neg:
                items = "".join(f"<li>Improve <strong>{f}</strong> to increase score</li>"
                                for f, _ in top_neg[:3])
                st.markdown(f"""
                <div class="warn-box">
                    <strong>Areas to improve</strong><br>
                    <ul style="margin:0.5rem 0 0 1rem;padding:0;font-size:0.85rem;line-height:1.8;">
                    {items}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # ── Fairness single-candidate note ────────────────────────
        st.markdown('<div class="section-hdr">Individual Fairness Note</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="violet-box">
            This result is based purely on competency scores.
            Demographic attributes (<strong>{res['gender']}</strong>,
            <strong>{res['ethnicity']}</strong>) were <strong>not used</strong>
            as model features — only recorded for batch fairness auditing
            in the Recruiter tab.
        </div>
        """, unsafe_allow_html=True)
