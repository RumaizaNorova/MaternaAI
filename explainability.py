"""
IBM AI Explainability 360 (AIX360) — LIME local explanations.
Replaces fake feature_importance × deviation with real per-patient LIME.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from aix360.algorithms.lime import LimeTabularExplainer
from data import FEATURE_COLS, FEATURE_LABELS

_explainer = None

def get_explainer(X_train):
    global _explainer
    if _explainer is None:
        _explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=FEATURE_COLS,
            class_names=["Low Risk", "Medium Risk", "High Risk"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
    return _explainer

def lime_explain(model, X_train, patient_row, predicted_class: int) -> list[dict]:
    """
    Returns list of {feature, condition, contribution, direction} for the predicted class.
    Positive contribution → pushes toward predicted class.
    """
    explainer = get_explainer(X_train)
    exp = explainer.explain_instance(
        patient_row.values.flatten(),
        model.predict_proba,
        num_features=6,
        top_labels=3,
    )
    results = []
    for condition, contrib in exp.as_list(label=predicted_class):
        # Parse feature name from condition string (e.g. "SystolicBP > 120.00")
        feat = None
        for f in FEATURE_COLS:
            if f in condition:
                feat = f
                break
        results.append({
            "feature": feat or condition,
            "label": FEATURE_LABELS.get(feat, feat) if feat else condition,
            "condition": condition,
            "contribution": round(float(contrib), 4),
            "direction": "increases" if contrib > 0 else "decreases",
        })
    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results
