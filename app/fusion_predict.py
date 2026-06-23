"""
fusion_predict.py - Multi-model prediction with SBERT support.

Models:
  - Structured-Only LR  (8 features)        -> model_structured.pkl
  - Structured-Only RF  (8 features)        -> base_structured_rf.pkl
  - Early Fusion RF     (392 = 384 SBERT + 8) -> early_fusion_rf.pkl

For Early Fusion, the app generates a short bio (FairCVdb style), encodes it with
SBERT (all-MiniLM-L6-v2, 384-dim), concatenates with the scaled 8 structured
features, and feeds the 392-dim vector to the RF model.

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
    scaler = get_scaler()
    x_struct = np.array(features, dtype=float).reshape(1, -1)   # (1,8)
    x_struct_scaled = scaler.transform(x_struct)               # scale like training

    if model_key == "early_rf":
        # Encode bio with SBERT -> 384-dim, concat with scaled structured -> 392
        emb = get_sbert().encode([bio_text or ""], normalize_embeddings=True)  # (1,384)
        x = np.concatenate([emb, x_struct_scaled], axis=1)     # (1,392)
        model = get_model("early_rf")
    else:
        x = x_struct_scaled                                    # (1,8)
        model = get_model(model_key)

    prob = float(model.predict_proba(x)[0, 1])
    label = "Shortlisted" if prob >= 0.5 else "Not Shortlisted"
    return prob, label



# ---- suitability via SBERT role-to-position similarity ---------------
def role_suitability(detected_role, target_position):
    """Compute suitability [0,1] = semantic similarity between the candidate's
    role (from CV) and the position being hired for.

    Strategy:
    1. Check exact match / keyword overlap first (fallback if cosine is unreliable)
    2. Use SBERT cosine similarity (normalized embeddings: cosine in [0,1])
    3. Rescale smartly: cosine [0.3, 0.95] -> [0.0, 1.0]
       - <0.3: very low match (0.0-0.2)
       - 0.3-0.95: linear stretch (0.2-1.0)
       - >0.95: high match (0.95-1.0)
    """
    if not detected_role or not target_position:
        return 0.5  # neutral default if missing
    
    # Normalize strings for keyword matching
    role_lower = detected_role.lower().strip()
    target_lower = target_position.lower().strip()
    
    # Exact match -> best score
    if role_lower == target_lower:
        return 1.0
    
    # Keyword overlap: if role contains key words from target (or vice versa)
    role_words = set(role_lower.split())
    target_words = set(target_lower.split())
    overlap = role_words & target_words  # intersection
    if len(overlap) >= 2 or (len(overlap) == 1 and len(target_words) <= 3):
        # Strong keyword match -> boost score
        return 0.85
    
    # SBERT semantic similarity as tiebreaker
    model = get_sbert()
    emb = model.encode([detected_role, target_position], normalize_embeddings=True)
    cos = float(np.dot(emb[0], emb[1]))  # cosine in [0, 1] for normalized vectors
    
    # Smarter rescaling: [0.3, 0.95] -> [0.0, 1.0]
    if cos < 0.3:
        scaled = cos * (0.2 / 0.3)         # 0-0.3 -> 0-0.2 (very poor match)
    else:
        scaled = 0.2 + (cos - 0.3) * (0.8 / 0.65)  # 0.3-0.95 -> 0.2-1.0
    
    return float(max(0.0, min(1.0, scaled)))


# Model display names + metrics (from notebook benchmark, for the comparison table)
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
