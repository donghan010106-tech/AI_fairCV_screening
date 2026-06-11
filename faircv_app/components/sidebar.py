import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;color:#00c9a7;">FairCV</div>
            <div style="font-size:0.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-top:2px;">CV Screening App</div>
        </div>
        <hr style="border-color:#252d3d;margin:0.8rem 0 1.2rem;">
        """, unsafe_allow_html=True)

        mode = st.radio("Mode", ["Candidate", "Recruiter"],
                        label_visibility="collapsed",
                        horizontal=True)

        st.markdown("""<div style="font-size:0.7rem;color:#6b7280;text-transform:uppercase;
                       letter-spacing:1px;margin:1.2rem 0 0.6rem;">Model</div>""",
                    unsafe_allow_html=True)
        model_choice = st.selectbox("Model", ["Late Fusion — LR (Best Balance)",
                                               "Baseline — LR",
                                               "Baseline — RF",
                                               "Baseline — MLP"],
                                    label_visibility="collapsed")

        st.markdown("""
        <hr style="border-color:#252d3d;margin:1.2rem 0 1rem;">
        <div style="font-size:0.68rem;color:#6b7280;line-height:1.6;">
            Powered by FairCV research model<br>
            Dataset: FairCVdb (Peña et al., 2023)<br>
            Fairness: DP · EOO · Disparate Impact
        </div>
        """, unsafe_allow_html=True)

    return mode, model_choice
