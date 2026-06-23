"""
fusion_predict.py - FIXED VERSION
Multi-model prediction with SBERT support.

FIX: RF không được scaled (train trên unscaled data)
     LR, MLP, Early Fusion được scaled

Models:
  - Structured-Only LR  (8 features, SCALED)        -> model_structured.pkl
  - Structured-Only RF  (8 features, UNSCALED)      -> base_structured_rf.pkl
  - Early Fusion RF     (392 = 384 SBERT + 8, SCALED) -> early_fusion_rf.pkl

Requires (same folder):
  model_structured.pkl, scaler_structured.pkl,
  base_structured_rf.pkl, early_fusion_rf.pkl
Install:
  pip install sentence-transformers joblib scikit-learn numpy
"""

import os
import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _path(name):
    return name if os.path.isabs(name) else os.path.join(_HERE, name)


# ---- lazy-loaded singletons (load once) ------------------------------
_sbert = None
_models = {}
_scaler = None


def get_sbert():
    """Load SBERT once. ~80MB download on first run."""
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert


def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(_path("scaler_structured.pkl"))
    return _scaler


def get_model(key):
    """key in {'struct_lr', 'struct_rf', 'early_rf'}."""
    if key in _models:
        return _models[key]
    files = {
        "struct_lr": "model_structured.pkl",
        "struct_rf": "base_structured_rf.pkl",
        "early_rf":  "early_fusion_rf.pkl",
    }
    _models[key] = joblib.load(_path(files[key]))
    return _models[key]


# ---- prediction ------------------------------------------------------
def predict(features, model_key, bio_text=""):
    """Predict recommendation score for one candidate.

    features  : list of 8 structured features [0,1] (order = COMPETENCY)
    model_key : 'struct_lr' | 'struct_rf' | 'early_rf'
    bio_text  : required only for 'early_rf' (the LLM-written bio)

    Returns: (prob, label)
    """
    x_struct = np.array(features, dtype=float).reshape(1, -1)   # (1,8)
    
    # ===== KEY FIX: Handle scaling per model =====
    if model_key == "struct_rf":
        # RF was trained on UNSCALED data
        x = x_struct  # NO SCALING
        model = get_model("struct_rf")
    elif model_key == "early_rf":
        # Early Fusion RF needs scaled structured + SBERT encoding
        scaler = get_scaler()
        x_struct_scaled = scaler.transform(x_struct)
        emb = get_sbert().encode([bio_text or ""], normalize_embeddings=True)  # (1,384)
        x = np.concatenate([emb, x_struct_scaled], axis=1)     # (1,392)
        model = get_model("early_rf")
    else:
        # LR was trained on SCALED data
        scaler = get_scaler()
        x_struct_scaled = scaler.transform(x_struct)
        x = x_struct_scaled  # (1,8)
        model = get_model(model_key)

    prob = float(model.predict_proba(x)[0, 1])
    label = "Shortlisted" if prob >= 0.5 else "Not Shortlisted"
    return prob, label


# ---- suitability via SBERT role-to-position similarity ---------------
def role_suitability(detected_role, target_position):
    """Compute suitability [0,1] = semantic similarity between the candidate's
    role (from CV) and the position being hired for.

    Strategy:
    1. Check exact match / keyword overlap first
    2. Use SBERT cosine similarity
    3. Rescale smartly to [0,1]
    """
    if not detected_role or not target_position:
        return 0.5
    
    role_lower = detected_role.lower().strip()
    target_lower = target_position.lower().strip()
    
    # Exact match
    if role_lower == target_lower:
        return 1.0
    
    # Keyword overlap
    role_words = set(role_lower.split())
    target_words = set(target_lower.split())
    overlap = role_words & target_words
    if len(overlap) >= 2 or (len(overlap) == 1 and len(target_words) <= 3):
        return 0.85
    
    # SBERT semantic similarity
    model = get_sbert()
    emb = model.encode([detected_role, target_position], normalize_embeddings=True)
    cos = float(np.dot(emb[0], emb[1]))
    
    # Smarter rescaling
    if cos < 0.3:
        scaled = cos * (0.2 / 0.3)
    else:
        scaled = 0.2 + (cos - 0.3) * (0.8 / 0.65)
    
    return float(max(0.0, min(1.0, scaled)))


# Model display names + metrics (from notebook benchmark)
MODEL_INFO = {
    "struct_lr": {"name": "Structured-Only LR",
                  "acc": 0.9665, "f1": 0.9658,
                  "dp_gender": 0.0046, "dp_eth": 0.0230},
    "struct_rf": {"name": "Structured-Only RF",
                  "acc": 0.9367, "f1": 0.9359,
                  "dp_gender": 0.0115, "dp_eth": 0.0218},
    "early_rf":  {"name": "Early Fusion RF (SBERT)",
                  "acc": 0.8087, "f1": 0.8141,
                  "dp_gender": 0.0019, "dp_eth": 0.0472},
}
