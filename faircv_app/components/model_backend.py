"""
FairCV Model Backend
Simulates trained models using learned coefficients from the research.
In production: load actual .pkl files trained in FairCV_Models.ipynb
"""
import numpy as np

# ── Feature names ──────────────────────────────────────────────────
FEATURE_NAMES = [
    'suitability', 'educ_attainment', 'prev_experience',
    'recommendation', 'availability',
    'lang_prof_1', 'lang_prof_2', 'lang_prof_3'
]

OCCUPATION_SUITABILITY = {
    'Teacher':      1.00, 'Professor':    1.00,
    'Nurse':        0.75, 'Surgeon':      0.75, 'Physician':    0.75,
    'Attorney':     0.50, 'Accountant':   0.50,
    'Journalist':   0.25, 'Photographer': 0.25, 'Filmmaker':    0.25,
}

# ── LR Coefficients from research (Setting A, blind label) ────────
LR_COEF = np.array([10.446, 5.554, 5.918, 9.878, 4.043, 6.402, 6.267, 6.307])
LR_INTERCEPT = -24.5   # approximate to center predictions

# ── RF weights (approximate from Gini importance) ─────────────────
RF_WEIGHTS = np.array([0.2796, 0.0965, 0.1033, 0.1168, 0.0576, 0.1164, 0.1147, 0.1151])

def predict(features: dict, model: str = "Late Fusion — LR (Best Balance)"):
    """
    Predict recommendation score and label.
    features: dict with FEATURE_NAMES keys, values in [0,1]
    Returns: score (float 0-1), label (bool), shap_values (dict)
    """
    x = np.array([features[f] for f in FEATURE_NAMES])
    x_scaled = (x - 0.5) / 0.3   # approximate StandardScaler

    if "RF" in model:
        # RF: weighted sum approximation
        raw = np.dot(RF_WEIGHTS, x)
        score = float(np.clip(raw * 1.2, 0, 1))
        # SHAP-like values: contribution = weight * (value - mean)
        shap_vals = RF_WEIGHTS * (x - 0.5)
    else:
        # LR: sigmoid of linear combination
        logit = float(np.dot(LR_COEF, x_scaled) + LR_INTERCEPT)
        score = float(1 / (1 + np.exp(-logit)))
        # SHAP-like: LR = coefficient * scaled_value (exact for LR)
        shap_vals = (LR_COEF * x_scaled) / np.sum(np.abs(LR_COEF * x_scaled)) * (score - 0.5)

    label = score >= 0.5
    shap_dict = {f: float(v) for f, v in zip(FEATURE_NAMES, shap_vals)}
    return score, label, shap_dict


def fairness_check(scores_by_group: dict):
    """
    Compute DP Gap and DI from a dict of {group: [scores]}
    """
    pos_rates = {}
    for group, scores in scores_by_group.items():
        if len(scores) > 0:
            pos_rates[group] = sum(1 for s in scores if s >= 0.5) / len(scores)

    if len(pos_rates) < 2:
        return {}

    rates = list(pos_rates.values())
    groups = list(pos_rates.keys())
    max_r = max(rates)
    min_r = min(rates)
    max_g = groups[rates.index(max_r)]
    min_g = groups[rates.index(min_r)]

    dp_gap = abs(max_r - min_r)
    di     = min_r / max_r if max_r > 0 else 1.0

    return {
        'pos_rates':  pos_rates,
        'dp_gap':     dp_gap,
        'di':         di,
        'max_group':  max_g,
        'min_group':  min_g,
        'eeoc_pass':  di >= 0.80,
    }
