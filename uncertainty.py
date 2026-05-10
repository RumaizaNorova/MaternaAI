"""
Uncertainty Quantification for maternal health predictions.
Implements calibration + bootstrap confidence intervals.
Inspired by IBM Uncertainty Quantification 360 (UQ360) methodology.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from data import FEATURE_COLS

def calibrate_model(base_model, X_train, y_train):
    """Isotonic calibration on a held-out calibration set."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=7)
    base_model.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_cal, y_cal)
    return calibrated

def bootstrap_confidence(model, X_single: np.ndarray, n_bootstrap: int = 200) -> dict:
    """
    Bootstrap uncertainty: perturb input slightly, observe prediction variance.
    Returns confidence interval and stability score.
    """
    row = X_single.values.flatten() if hasattr(X_single, "values") else X_single.flatten()
    preds = []
    noise_scale = 0.02  # 2% gaussian noise on each feature

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        noisy = row * (1 + rng.normal(0, noise_scale, size=row.shape))
        noisy = noisy.reshape(1, -1)
        prob = model.predict_proba(noisy)[0]
        preds.append(np.argmax(prob))

    preds = np.array(preds)
    # Stability = fraction of bootstrap samples agreeing with majority prediction
    majority = np.bincount(preds).argmax()
    stability = (preds == majority).mean()

    # Confidence intervals on probabilities
    prob_samples = []
    for _ in range(n_bootstrap):
        noisy = row * (1 + rng.normal(0, noise_scale, size=row.shape))
        prob_samples.append(model.predict_proba(noisy.reshape(1,-1))[0])

    prob_samples = np.array(prob_samples)
    ci_low  = np.percentile(prob_samples, 5,  axis=0)
    ci_high = np.percentile(prob_samples, 95, axis=0)

    if stability >= 0.90:
        interval_label = "high confidence"
    elif stability >= 0.75:
        interval_label = "moderate confidence"
    else:
        interval_label = "low confidence — verify clinically"

    return {
        "stability": round(float(stability), 3),
        "confidence": round(float(stability), 3),
        "interval_label": interval_label,
        "ci_low":  ci_low.tolist(),
        "ci_high": ci_high.tolist(),
        "majority_class": int(majority),
    }

def calibration_metrics(model, X_test, y_test) -> dict:
    """Compute calibration curve data for reliability diagram."""
    from sklearn.preprocessing import label_binarize
    classes = [0, 1, 2]
    results = {}
    y_bin = label_binarize(y_test, classes=classes)
    probs = model.predict_proba(X_test)
    for i, cls in enumerate(["Low Risk", "Medium Risk", "High Risk"]):
        try:
            frac_pos, mean_pred = calibration_curve(y_bin[:, i], probs[:, i], n_bins=8)
            results[cls] = {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}
        except Exception:
            results[cls] = {"frac_pos": [], "mean_pred": []}
    return results
