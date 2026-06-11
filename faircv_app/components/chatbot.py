import streamlit as st
import json

def _build_system_prompt(context: dict | None) -> str:
    base = """You are FairCV Assistant — an AI expert in fair hiring and resume screening.
You help both candidates understand their screening results and recruiters interpret fairness metrics.

Your expertise:
- FairCV model: trained on FairCVdb (24,000 synthetic profiles, Peña et al. 2023)
- Fairness metrics: Demographic Parity Gap, Equal Opportunity Gap, Disparate Impact (EEOC 4/5 rule)
- Features: suitability, education, experience, recommendation, availability, language proficiency (3)
- Fusion strategies: Baseline, Early Fusion, Late Fusion, Weighted Hybrid Fusion (SBERT + structured)
- XAI: SHAP values explain each feature's contribution to the prediction

Guidelines:
- Be clear and concise. Use plain language.
- When explaining SHAP values, say "this feature pushed the score up/down by X"
- When asked about fairness, explain the metric before interpreting the number
- Never make definitive hiring recommendations — always note the model is a decision-support tool
- If asked something outside your scope, say so honestly
"""
    if context:
        base += f"\n\nCurrent screening context:\n{json.dumps(context, indent=2)}"
    return base


def _call_claude(messages: list, system: str) -> str:
    """Call Anthropic API via fetch (runs in Streamlit server context)."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except ImportError:
        return _fallback_response(messages[-1]['content'])
    except Exception as e:
        return f"API error: {str(e)}. Please check your Anthropic API key."


def _fallback_response(user_msg: str) -> str:
    """Rule-based fallback when API is not available."""
    msg = user_msg.lower()
    if any(w in msg for w in ['shap', 'why', 'reason', 'explain']):
        return ("SHAP (SHapley Additive exPlanations) measures each feature's contribution "
                "to the prediction. A positive SHAP value means the feature pushed the score "
                "toward 'Recommended'. The bar chart in the Screening tab shows this visually. "
                "Suitability typically has the largest contribution in FairCV models.")
    if any(w in msg for w in ['fair', 'bias', 'dp gap', 'disparate', 'eeoc']):
        return ("Fairness is measured by three metrics: "
                "(1) DP Gap — are positive rates equal across groups? "
                "(2) EOO Gap — among qualified candidates, are they equally recommended? "
                "(3) Disparate Impact — if DI < 0.80, EEOC considers it a potential violation. "
                "In FairCV, training with the blind label keeps all three metrics near-zero.")
    if any(w in msg for w in ['improve', 'suggestion', 'increase', 'score']):
        return ("To improve your screening score, focus on the features with negative SHAP values "
                "shown in your result. Language proficiency and recommendation quality often have "
                "strong positive weights. Suitability is determined by occupation and cannot be changed. "
                "Availability and education attainment are also meaningful contributors.")
    if any(w in msg for w in ['fusion', 'early', 'late', 'hybrid']):
        return ("FairCV compares three fusion strategies: "
                "Early Fusion concatenates SBERT text embeddings (384-dim) with structured features (8-dim). "
                "Late Fusion runs two separate models and blends their predictions (β=0.5). "
                "Weighted Hybrid blends feature representations before classification. "
                "Late Fusion with LR achieves the best accuracy-fairness balance in research results.")
    return ("I'm FairCV Assistant. I can help you understand screening results, SHAP explanations, "
            "fairness metrics (DP Gap, EOO Gap, Disparate Impact), and fusion strategies. "
            "What would you like to know?")


def render_chatbot():
    st.markdown("""
    <div class="app-hero">
        <div class="app-title">AI Assistant</div>
        <p class="app-sub">Ask anything about your screening result, fairness metrics,
        SHAP explanations, or how the FairCV model works.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Context from last screening ───────────────────────────────
    context = None
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        context = {
            'score':      round(res['score'], 4),
            'verdict':    'Recommended' if res['label'] else 'Not Recommended',
            'occupation': res.get('occupation', ''),
            'gender':     res.get('gender', ''),
            'ethnicity':  res.get('ethnicity', ''),
            'shap_values': {k: round(v, 4) for k, v in res['shap'].items()},
            'top_positive_feature': max(res['shap'], key=res['shap'].get),
            'top_negative_feature': min(res['shap'], key=res['shap'].get),
        }
        st.markdown(f"""
        <div class="insight-box">
            Context loaded from last screening:
            <strong style="color:#00c9a7;">{context['verdict']}</strong>
            (score: {context['score']}) — {context['occupation']}.
            Ask me to explain this result.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="amber-box">
            No screening result loaded yet. Run a screening in the CV Screening tab first,
            or ask general questions about FairCV and fairness metrics.
        </div>
        """, unsafe_allow_html=True)

    # ── Suggested questions ───────────────────────────────────────
    st.markdown("**Suggested questions:**")
    suggestions = [
        "Why was this candidate recommended?",
        "What does the SHAP value mean?",
        "How does Disparate Impact work?",
        "What is the difference between DP Gap and EOO Gap?",
        "How can the candidate improve their score?",
        "Which fusion strategy is most fair?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q, key=f"sq_{i}"):
            st.session_state.setdefault('chat_history', [])
            st.session_state['chat_history'].append({'role':'user','content':q})
            system = _build_system_prompt(context)
            reply  = _call_claude(st.session_state['chat_history'], system)
            st.session_state['chat_history'].append({'role':'assistant','content':reply})
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Conversation</div>', unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # ── Input ─────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your result, fairness metrics, or SHAP values..."):
        st.session_state['chat_history'].append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        system = _build_system_prompt(context)
        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                reply = _call_claude(st.session_state['chat_history'], system)
            st.markdown(reply)
        st.session_state['chat_history'].append({'role':'assistant','content':reply})

    if st.session_state['chat_history']:
        if st.button("Clear conversation"):
            st.session_state['chat_history'] = []
            st.rerun()
