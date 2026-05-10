"""
IBM Adversarial Robustness Toolbox (ART) — security audit for maternal health AI.

Clinical threat model: A compromised data entry system slightly alters patient vitals
to downgrade high-risk mothers to low-risk, causing missed referrals and preventable deaths.
We test this attack and measure how robust MaternaAI is against it.
"""
import numpy as np
import pandas as pd
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump, ZooAttack
from art.attacks.evasion import BoundaryAttack
from data import FEATURE_COLS

# Clinical bounds: attacker is limited to physiologically plausible perturbations
FEATURE_BOUNDS = {
    "Age":         (10,  49),
    "SystolicBP":  (70,  160),
    "DiastolicBP": (40,  100),
    "BS":          (6.0, 19.0),
    "BodyTemp":    (96.0,103.0),
    "HeartRate":   (50,  90),
}

def _clip_to_bounds(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for i, feat in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_BOUNDS[feat]
        X[:, i] = np.clip(X[:, i], lo, hi)
    return X

def run_adversarial_audit(model, X_test: pd.DataFrame, y_test: pd.Series,
                          epsilon_pct: float = 0.05) -> dict:
    """
    IBM ART robustness audit.
    epsilon_pct: max perturbation as % of feature range (default 5% — clinically small).

    Returns attack success rates and robustness scores per risk class.
    """
    X_arr = X_test.values.astype(np.float32)
    y_arr = y_test.values

    # Scale epsilon per feature
    clip_min = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_COLS], dtype=np.float32)
    clip_max = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_COLS], dtype=np.float32)

    art_clf = SklearnClassifier(model=model, clip_values=(clip_min, clip_max))

    # Focus on high-risk patients (most critical to protect)
    high_risk_mask = y_arr == 2
    if high_risk_mask.sum() < 5:
        return {"error": "insufficient high-risk samples"}

    X_hr = X_arr[high_risk_mask][:20]   # test on up to 20 high-risk patients
    y_hr = y_arr[high_risk_mask][:20]

    results = {}

    # ── Gaussian noise baseline ────────────────────────────────────────────
    rng = np.random.RandomState(42)
    noise = rng.normal(0, epsilon_pct, X_hr.shape) * (clip_max - clip_min)
    X_noisy = _clip_to_bounds(X_hr + noise.astype(np.float32))
    y_noisy_pred = model.predict(X_noisy)
    noise_attack_rate = float((y_noisy_pred != y_hr).mean())
    results["gaussian_noise"] = {
        "attack_success_rate": round(noise_attack_rate, 3),
        "robustness": round(1 - noise_attack_rate, 3),
        "description": f"Random noise ±{epsilon_pct*100:.0f}% of feature range",
    }

    # ── Targeted iterative feature attack (black-box, no gradients) ────────
    try:
        # Greedy feature-wise attack: push most sensitive features toward safe range
        n_steps, step_frac = 15, 0.03
        X_adv = X_hr.copy()
        for step in range(n_steps):
            for i, feat in enumerate(FEATURE_COLS):
                lo, hi = float(clip_min[i]), float(clip_max[i])
                delta = (hi - lo) * step_frac
                # Push toward the low-risk centroid direction
                X_try = X_adv.copy()
                X_try[:, i] = np.clip(X_adv[:, i] - delta, lo, hi)
                p_before = model.predict_proba(X_adv)[:, 2].mean()
                p_after  = model.predict_proba(X_try)[:, 2].mean()
                if p_after < p_before:
                    X_adv = X_try
        X_adv = _clip_to_bounds(X_adv)

        perturbation = np.abs(X_adv - X_hr) / (clip_max - clip_min + 1e-6)
        mean_perturb = float(perturbation.mean())
        y_adv_pred   = model.predict(X_adv)
        it_success   = float((y_adv_pred != y_hr).mean())

        results["iterative_black_box"] = {
            "attack_success_rate": round(it_success, 3),
            "robustness": round(1 - it_success, 3),
            "mean_perturbation_pct": round(mean_perturb * 100, 2),
            "description": "Iterative black-box feature attack (high→low risk)",
        }
    except Exception as e:
        results["iterative_black_box"] = {"error": str(e)[:80]}

    # ── Feature sensitivity (which vital signs are most exploitable) ───────
    sensitivities = {}
    for i, feat in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_BOUNDS[feat]
        delta = (hi - lo) * epsilon_pct
        X_up = X_hr.copy(); X_up[:, i] = np.clip(X_up[:, i] + delta, lo, hi)
        X_dn = X_hr.copy(); X_dn[:, i] = np.clip(X_dn[:, i] - delta, lo, hi)
        p_orig = model.predict_proba(X_hr)[:, 2].mean()
        p_up   = model.predict_proba(X_up)[:, 2].mean()
        p_dn   = model.predict_proba(X_dn)[:, 2].mean()
        sensitivities[feat] = round(float(max(abs(p_up - p_orig), abs(p_dn - p_orig))), 4)

    results["feature_sensitivity"] = dict(
        sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Overall robustness score ───────────────────────────────────────────
    noise_rob = results["gaussian_noise"]["robustness"]
    it_rob    = results.get("iterative_black_box", {}).get("robustness", noise_rob)
    results["overall_robustness_score"] = round((noise_rob + it_rob) / 2, 3)

    return results


def shap_explanation(model, X_background: pd.DataFrame, X_patient: pd.DataFrame) -> dict:
    """SHAP explanation for individual patient — works with any sklearn model."""
    import shap
    try:
        # Use KernelExplainer as universal fallback (works with any model)
        bg = shap.sample(X_background, 50)
        predict_fn = lambda x: model.predict_proba(x)
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_vals = explainer.shap_values(X_patient.values, nsamples=80)
        base_vals = explainer.expected_value
        return {
            "shap_values": [sv.tolist() for sv in shap_vals],
            "base_values": base_vals if hasattr(base_vals, '__iter__') else [float(base_vals)]*3,
            "feature_names": FEATURE_COLS,
            "patient_values": X_patient.values[0].tolist(),
        }
    except Exception as e:
        return {"error": str(e)}
