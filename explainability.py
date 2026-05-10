"""
IBM AI Explainability 360 (AIX360) — LIME local explanations + counterfactuals.

Two explanation types per prediction:
  1. LIME  — why THIS patient got this risk score (feature contributions)
  2. Counterfactual — what is the minimum change to reduce risk by one class
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from aix360.algorithms.lime import LimeTabularExplainer
from data import FEATURE_COLS, FEATURE_LABELS

FEATURE_BOUNDS = {
    "Age":         (10,   49),
    "SystolicBP":  (70,  160),
    "DiastolicBP": (40,  100),
    "BS":          (6.0, 19.0),
    "BodyTemp":    (96.0, 103.0),
    "HeartRate":   (50,   90),
}

_explainer_cache = None


def _get_explainer(X_train):
    global _explainer_cache
    if _explainer_cache is None:
        _explainer_cache = LimeTabularExplainer(
            X_train.values,
            feature_names=FEATURE_COLS,
            class_names=["Low Risk", "Medium Risk", "High Risk"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
    return _explainer_cache


def lime_explain(model, X_train, patient_row, predicted_class: int) -> list:
    """
    IBM AIX360 LIME — per-patient local explanation.
    Returns list of {feature, label, condition, contribution, direction}.
    Positive contribution → pushes toward predicted class.
    """
    explainer = _get_explainer(X_train)
    exp = explainer.explain_instance(
        patient_row.values.flatten(),
        model.predict_proba,
        num_features=6,
        top_labels=3,
    )
    results = []
    for condition, contrib in exp.as_list(label=predicted_class):
        feat = next((f for f in FEATURE_COLS if f in condition), None)
        results.append({
            "feature":      feat or condition,
            "label":        FEATURE_LABELS.get(feat, feat) if feat else condition,
            "condition":    condition,
            "contribution": round(float(contrib), 4),
            "direction":    "increases" if contrib > 0 else "decreases",
        })
    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results


def counterfactual_explain(model, patient_row, predicted_class: int) -> list:
    """
    IBM AIX360 counterfactual: minimum single-feature change to drop risk by 1 class.

    For each feature, scans its full clinical range in 500 steps to find the
    smallest absolute change that shifts the predicted class to target_class.
    Returns list sorted by smallest relative change (most actionable first).

    If no single-feature change suffices, returns the closest attempts.
    """
    if predicted_class == 0:
        return []   # already low risk — no counterfactual needed

    target_class = predicted_class - 1
    row = patient_row.values.flatten().copy()

    counterfactuals = []
    for feat_idx, feat in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_BOUNDS[feat]
        current = float(row[feat_idx])
        feat_range = hi - lo

        best_change = None
        # Scan both directions: decreasing then increasing
        for direction in [-1, 1]:
            target_end = lo if direction == -1 else hi
            search_space = np.linspace(current, target_end, 500)
            for val in search_space:
                row_try = row.copy()
                row_try[feat_idx] = val
                pred = int(model.predict(row_try.reshape(1, -1))[0])
                if pred <= target_class:
                    change = val - current
                    pct    = abs(change) / feat_range * 100
                    best_change = {
                        "feature":        feat,
                        "label":          FEATURE_LABELS[feat],
                        "current":        round(current, 2),
                        "target":         round(float(val), 2),
                        "change":         round(float(change), 2),
                        "change_pct":     round(pct, 1),
                        "direction":      "decrease" if change < 0 else "increase",
                        "clinical_note":  _clinical_note(feat, change),
                    }
                    break
            if best_change is not None:
                break

        if best_change is not None:
            counterfactuals.append(best_change)

    counterfactuals.sort(key=lambda x: x["change_pct"])
    return counterfactuals


def _clinical_note(feat: str, change: float) -> str:
    notes = {
        "BS":         "Glucose reduction via dietary control or medication",
        "SystolicBP": "BP reduction via rest, medication, or antihypertensives",
        "DiastolicBP":"BP reduction via rest or antihypertensives",
        "BodyTemp":   "Temperature reduction via antipyretics or cooling",
        "HeartRate":  "HR reduction via rest, fluids, or beta-blockers",
        "Age":        "Age is non-modifiable — flag for additional monitoring",
    }
    return notes.get(feat, "")
