"""Proxy-channel neutralisation operators.

Given the leaking feature set localised by PLS, we offer three interventions,
in increasing subtlety:

  * 'drop'      : remove the leaking features entirely (targeted unawareness).
  * 'suppress'  : zero-out / mean-impute the leaking features at inference
                  (keeps dimensionality, useful for fixed-architecture models).
  * 'orthogonalize' : residualise the leaking features against the predicted
                  protected direction Ahat=g(X_clean), removing the component
                  aligned with A while preserving A-orthogonal signal.

Compared with the baselines' BLANKET fairness-through-unawareness (drop A) or a
GLOBAL in-processing penalty (FairJob's lambda), these interventions are
TARGETED: they touch only the localised proxy channel.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def neutralize(X, proxy_cols, mode="orthogonalize", A=None,
               clean_cols=None, fitted=None):
    """Return a transformed copy of X with the proxy channel neutralised.

    `fitted` lets the test set reuse transforms fit on train (no leakage).
    Returns (X_new, fitted_state).
    """
    X = X.copy()
    cols = list(X.columns)
    proxy_cols = [c for c in proxy_cols if c in cols]

    if mode == "drop":
        return X.drop(columns=proxy_cols), {"mode": "drop", "dropped": proxy_cols}

    if mode == "suppress":
        if fitted is None:
            fill = {c: float(X[c].mean()) for c in proxy_cols}
            fitted = {"mode": "suppress", "fill": fill}
        for c in proxy_cols:
            X[c] = fitted["fill"][c]
        return X, fitted

    if mode == "orthogonalize":
        # Remove the protected direction that lives INSIDE the proxy block,
        # while preserving the proxy-orthogonal (legitimate) signal. The
        # transform is learned with A on the training set but is deployable at
        # inference WITHOUT A (it only needs the proxy features).
        if len(proxy_cols) == 0:
            return X, {"mode": "orthogonalize", "betas": {}, "w": None,
                       "proxy_cols": []}
        if fitted is None:
            assert A is not None, "orthogonalize needs A on the training call"
            P = X[proxy_cols].values.astype(float)
            Pc = P - P.mean(axis=0, keepdims=True)
            g = LogisticRegression(max_iter=500, C=1.0)
            g.fit(Pc, A)
            w = g.coef_.ravel()
            nw = np.linalg.norm(w) + 1e-12
            w = w / nw
            ahat = Pc @ w                       # protected direction within proxies
            va = float(ahat @ ahat) + 1e-12
            betas = {c: float((Pc[:, i] @ ahat) / va) for i, c in enumerate(proxy_cols)}
            fitted = {"mode": "orthogonalize", "w": w, "betas": betas,
                      "proxy_cols": proxy_cols, "mean": P.mean(axis=0)}
        w = fitted["w"]; betas = fitted["betas"]; mean = fitted["mean"]
        P = X[proxy_cols].values.astype(float)
        Pc = P - mean
        ahat = Pc @ w
        for i, c in enumerate(proxy_cols):
            X[c] = X[c].values - betas[c] * ahat
        return X, fitted

    raise ValueError(f"unknown mode {mode}")
