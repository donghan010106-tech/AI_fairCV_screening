import streamlit as st

def render_about():
    st.markdown("""
    <div class="app-hero">
        <div class="app-title">About FairCV</div>
        <p class="app-sub">A fair AI-based resume screening system built on
        research-grade fairness evaluation — for both candidates and recruiters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">What this app does</div>', unsafe_allow_html=True)
    for title, desc in [
        ("CV Screening",       "Predicts hiring recommendation from 8 competency scores using trained FairCV models (LR, RF, MLP). Explains each feature's contribution via SHAP-style values."),
        ("Recruiter Batch",    "Upload multiple candidates as CSV. Runs batch predictions and generates a fairness audit report — DP Gap, Disparate Impact, EEOC 4/5 compliance check."),
        ("AI Assistant",       "Conversational AI (Claude API) that explains screening results, SHAP values, fairness metrics, and model behavior in plain language."),
        ("Fairness First",     "Demographic attributes (gender, ethnicity) are recorded only for fairness auditing — never used as model input features. All predictions are competency-based."),
    ]:
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:10px;padding:1.1rem 1.4rem;margin-bottom:0.8rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                        color:#00c9a7;margin-bottom:0.4rem;">{title}</div>
            <div style="font-size:0.87rem;color:#b0b8c8;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Model & Dataset</div>', unsafe_allow_html=True)
    rows = [
        ("Dataset",    "FairCVdb — Peña et al. (2023)"),
        ("Profiles",   "24,000 synthetic resume profiles"),
        ("Features",   "8 competency scores (Setting A)"),
        ("Target",     "blind_label (fair, no demographic penalty)"),
        ("Classifiers","LR · RF · MLP"),
        ("Fusion",     "Baseline · Early · Late · Weighted Hybrid"),
        ("Fairness",   "DP Gap · EOO Gap · Disparate Impact"),
    ]
    st.markdown("""<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
    <tbody>""" + "".join(f"""
    <tr>
        <td style="padding:0.5rem 0.8rem;border-bottom:1px solid #252d3d;
                   color:#6b7280;font-family:'IBM Plex Mono',monospace;
                   font-size:0.78rem;width:160px;">{k}</td>
        <td style="padding:0.5rem 0.8rem;border-bottom:1px solid #252d3d;
                   color:#e8edf5;">{v}</td>
    </tr>""" for k, v in rows) + "</tbody></table>", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Research Papers</div>', unsafe_allow_html=True)
    for num, title, ref in [
        (1, "FairCVdb Dataset & Benchmark",
           "Peña et al. SN Computer Science 4:434 (2023)"),
        (2, "Fusion Techniques in AI Recruitment",
           "Swati, Roy, Ntoutsi. EWAF'24 (2024)"),
        (3, "FAIRE: LLM Bias Benchmark",
           "Wen et al. arXiv:2504.01420 (2025)"),
    ]:
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:8px;padding:0.9rem 1.2rem;margin-bottom:0.6rem;
                    display:flex;gap:1rem;align-items:flex-start;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                        color:#00c9a7;background:rgba(0,201,167,0.1);
                        border-radius:4px;padding:2px 8px;flex-shrink:0;">[{num}]</div>
            <div>
                <div style="font-size:0.87rem;font-weight:500;margin-bottom:2px;">{title}</div>
                <div style="font-size:0.78rem;color:#6b7280;">{ref}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="amber-box" style="margin-top:1.2rem;">
        <strong>Disclaimer:</strong> This app is a research prototype built for the
        DAP391m capstone project. Predictions are based on synthetic training data
        and should not be used for actual hiring decisions without further validation.
    </div>
    """, unsafe_allow_html=True)
