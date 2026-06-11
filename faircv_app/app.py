"""
FairCV Screening App — CV Evaluation & Bias Audit Tool
For both candidates and recruiters.
"""
import streamlit as st

st.set_page_config(
    page_title="FairCV Screening App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0f1117;
    --bg-card:   #171b26;
    --bg-card2:  #1e2333;
    --teal:      #00c9a7;
    --teal-dim:  rgba(0,201,167,0.12);
    --teal-border: rgba(0,201,167,0.3);
    --amber:     #ffb547;
    --amber-dim: rgba(255,181,71,0.10);
    --rose:      #ff5c7c;
    --rose-dim:  rgba(255,92,124,0.10);
    --violet:    #9b6dff;
    --violet-dim:rgba(155,109,255,0.10);
    --sky:       #4fc3f7;
    --green:     #4cda8f;
    --text:      #e8edf5;
    --muted:     #6b7280;
    --border:    #252d3d;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0b0e16 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* Input */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Tabs */
[data-baseweb="tab"] { color: var(--muted) !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.82rem !important; }
[aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 8px; padding: 3px; gap: 2px; }

/* Expander */
div[data-testid="stExpander"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Button */
.stButton > button {
    background: var(--teal-dim) !important;
    border: 1px solid var(--teal-border) !important;
    color: var(--teal) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { background: rgba(0,201,167,0.22) !important; }

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Custom classes ── */
.app-hero {
    background: linear-gradient(135deg, #0f1117 0%, #141b2d 50%, #0d1a1f 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-hero::after {
    content:'';
    position:absolute; top:0; right:0; bottom:0; left:0;
    background: radial-gradient(ellipse at 80% 40%, rgba(0,201,167,0.06) 0%, transparent 55%);
    pointer-events:none;
}
.app-title { font-family:'IBM Plex Mono',monospace; font-size:1.9rem; font-weight:600; color:var(--teal); letter-spacing:-0.5px; margin:0 0 0.4rem 0; }
.app-sub { font-size:0.9rem; color:var(--muted); max-width:520px; line-height:1.6; margin:0; }

.score-ring {
    width:140px; height:140px; border-radius:50%;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    border: 4px solid var(--teal);
    background: var(--teal-dim);
    margin: 0 auto;
}
.score-number { font-family:'IBM Plex Mono',monospace; font-size:2.4rem; font-weight:600; color:var(--teal); line-height:1; }
.score-label  { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

.verdict-rec  { background:rgba(76,218,143,0.12); border:1px solid rgba(76,218,143,0.35); border-radius:8px; padding:0.8rem 1.2rem; text-align:center; }
.verdict-rej  { background:var(--rose-dim); border:1px solid rgba(255,92,124,0.35); border-radius:8px; padding:0.8rem 1.2rem; text-align:center; }
.verdict-text { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600; }

.metric-card { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:1.1rem 1.3rem; text-align:center; }
.metric-val  { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600; color:var(--teal); line-height:1; margin-bottom:3px; }
.metric-label{ font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.8px; }

.insight-box  { background:var(--bg-card2); border-left:3px solid var(--teal);  border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin:0.6rem 0; font-size:0.87rem; line-height:1.6; }
.warn-box     { background:var(--rose-dim);  border-left:3px solid var(--rose);  border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin:0.6rem 0; font-size:0.87rem; line-height:1.6; }
.amber-box    { background:var(--amber-dim); border-left:3px solid var(--amber); border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin:0.6rem 0; font-size:0.87rem; line-height:1.6; }
.violet-box   { background:var(--violet-dim);border-left:3px solid var(--violet);border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin:0.6rem 0; font-size:0.87rem; line-height:1.6; }

.section-hdr  { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:var(--teal); border-bottom:1px solid var(--border); padding-bottom:0.4rem; margin:1.6rem 0 1rem 0; }

.shap-row { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.shap-feat { font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:var(--text); width:130px; flex-shrink:0; }
.shap-bar-wrap { flex:1; height:18px; background:var(--bg-card2); border-radius:4px; overflow:hidden; position:relative; }
.shap-fill-pos { height:100%; background:var(--teal);  border-radius:4px; }
.shap-fill-neg { height:100%; background:var(--rose);  border-radius:4px; }
.shap-val  { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; width:56px; flex-shrink:0; text-align:right; }

.fairness-badge-pass { display:inline-block; background:rgba(76,218,143,0.15); border:1px solid rgba(76,218,143,0.4); border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600; color:#4cda8f; font-family:'IBM Plex Mono',monospace; }
.fairness-badge-fail { display:inline-block; background:var(--rose-dim); border:1px solid rgba(255,92,124,0.4); border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600; color:var(--rose); font-family:'IBM Plex Mono',monospace; }

[data-testid="stMarkdownContainer"] p { color:var(--text); line-height:1.65; }
</style>
""", unsafe_allow_html=True)

from components.sidebar   import render_sidebar
from components.screener  import render_screener
from components.recruiter import render_recruiter
from components.chatbot   import render_chatbot
from components.about     import render_about

def main():
    mode, model_choice = render_sidebar()
    tab_screen, tab_recruiter, tab_chat, tab_about = st.tabs([
        "CV Screening", "Recruiter Batch", "AI Assistant", "About"
    ])
    with tab_screen:
        render_screener(model_choice)
    with tab_recruiter:
        render_recruiter(model_choice)
    with tab_chat:
        render_chatbot()
    with tab_about:
        render_about()

if __name__ == "__main__":
    main()
