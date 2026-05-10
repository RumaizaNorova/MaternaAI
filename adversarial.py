"""
IBM Adversarial Robustness Toolbox (ART) — attack + defense pipeline.

Clinical threat model: A compromised data-entry tablet slightly alters patient
vitals to downgrade a high-risk mother to low-risk, causing a missed referral.

We run the attack, then apply ART FeatureSqueezing defense and show the
reduction in attack success rate — completing the attack/defense narrative.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from art.estimators.classification import SklearnClassifier
from art.defences.preprocessor import FeatureSqueezing
from data import FEATURE_COLS

FEATURE_BOUNDS = {
    "Age":         (10,   49),
    "SystolicBP":  (70,  160),
    "DiastolicBP": (40,  100),
    "BS":          (6.0, 19.0),
    "BodyTemp":    (96.0,103.0),
    "HeartRate":   (50,   90),
}

_clip_min = np.array([FEATURE_BOUNDS[f][0] for f in FEATURE_COLS], dtype=np.float32)
_clip_max = np.array([FEATURE_BOUNDS[f][1] for f in FEATURE_COLS], dtype=np.float32)


def _clip(X: np.ndarray) -> np.ndarray:
    return np.clip(X, _clip_min, _clip_max)


def _feature_squeeze(X: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    IBM ART FeatureSqueezing defense — discretizes each feature into
    2^bit_depth bins within its clinical range.
    Adversarial perturbations smaller than the bin width are neutralized.
    """
    # Normalize to [0,1], squeeze, denormalize
    rng = _clip_max - _clip_min + 1e-9
    X_norm = ((X - _clip_min) / rng).astype(np.float32)
    fs = FeatureSqueezing(clip_values=(0.0, 1.0), bit_depth=bit_depth)
    X_sq_norm, _ = fs(X_norm)
    return _clip(X_sq_norm * rng + _clip_min)


def run_adversarial_audit(model, X_test: pd.DataFrame, y_test: pd.Series,
                          epsilon_pct: float = 0.05) -> dict:
    """
    Full ART attack + defense audit on high-risk patients.
    Returns attack success rates before/after FeatureSqueezing defense,
    feature sensitivity scores, and overall robustness.
    """
    X_arr = X_test.values.astype(np.float32)
    y_arr = y_test.values

    art_clf = SklearnClassifier(model=model, clip_values=(_clip_min, _clip_max))

    high_risk_mask = y_arr == 2
    if high_risk_mask.sum() < 5:
        return {"error": "insufficient high-risk samples for audit"}

    X_hr = X_arr[high_risk_mask][:20]
    y_hr = y_arr[high_risk_mask][:20]

    results = {}

    # ── Attack 1: Gaussian noise baseline ─────────────────────────────────────
    rng = np.random.RandomState(42)
    noise = (rng.normal(0, epsilon_pct, X_hr.shape) * (_clip_max - _clip_min)).astype(np.float32)
    X_noisy = _clip(X_hr + noise)
    asr_noise = float((model.predict(X_noisy) != y_hr).mean())

    # Defense against gaussian noise
    X_noisy_def = _feature_squeeze(X_noisy)
    asr_noise_def = float((model.predict(X_noisy_def) != y_hr).mean())

    results["gaussian_noise"] = {
        "attack_success_rate": round(asr_noise, 3),
        "robustness":          round(1 - asr_noise, 3),
        "defended_asr":        round(asr_noise_def, 3),
        "defended_robustness": round(1 - asr_noise_def, 3),
        "defense_improvement": round(asr_noise - asr_noise_def, 3),
        "description":         f"Random noise ±{epsilon_pct*100:.0f}% of clinical range",
    }

    # ── Attack 2: Iterative black-box (greedy feature-wise) ───────────────────
    try:
        n_steps, step_frac = 15, 0.03
        X_adv = X_hr.copy()
        for _ in range(n_steps):
            for i in range(len(FEATURE_COLS)):
                lo, hi = float(_clip_min[i]), float(_clip_max[i])
                delta = (hi - lo) * step_frac
                X_try = X_adv.copy()
                X_try[:, i] = np.clip(X_adv[:, i] - delta, lo, hi)
                if model.predict_proba(X_try)[:, 2].mean() < model.predict_proba(X_adv)[:, 2].mean():
                    X_adv = X_try
        X_adv = _clip(X_adv)

        perturbation = float(np.abs(X_adv - X_hr).mean() / (_clip_max - _clip_min + 1e-6).mean())
        asr_it = float((model.predict(X_adv) != y_hr).mean())

        # Defense against iterative attack
        X_adv_def    = _feature_squeeze(X_adv)
        asr_it_def   = float((model.predict(X_adv_def) != y_hr).mean())

        results["iterative_black_box"] = {
            "attack_success_rate": round(asr_it, 3),
            "robustness":          round(1 - asr_it, 3),
            "defended_asr":        round(asr_it_def, 3),
            "defended_robustness": round(1 - asr_it_def, 3),
            "defense_improvement": round(asr_it - asr_it_def, 3),
            "mean_perturbation_pct": round(perturbation * 100, 2),
            "description":         "Greedy iterative black-box (high→low risk)",
        }
    except Exception as e:
        results["iterative_black_box"] = {"error": str(e)[:80]}

    # ── Feature sensitivity (which vitals are most exploitable) ───────────────
    sensitivities = {}
    p_orig = model.predict_proba(X_hr)[:, 2].mean()
    for i, feat in enumerate(FEATURE_COLS):
        lo, hi = FEATURE_BOUNDS[feat]
        delta = (hi - lo) * epsilon_pct
        X_up = X_hr.copy(); X_up[:, i] = np.clip(X_hr[:, i] + delta, lo, hi)
        X_dn = X_hr.copy(); X_dn[:, i] = np.clip(X_hr[:, i] - delta, lo, hi)
        sens = max(
            abs(model.predict_proba(X_up)[:, 2].mean() - p_orig),
            abs(model.predict_proba(X_dn)[:, 2].mean() - p_orig),
        )
        sensitivities[feat] = round(float(sens), 4)
    results["feature_sensitivity"] = dict(
        sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Overall scores ─────────────────────────────────────────────────────────
    g_rob  = results["gaussian_noise"]["robustness"]
    it_rob = results.get("iterative_black_box", {}).get("robustness", g_rob)
    g_dob  = results["gaussian_noise"]["defended_robustness"]
    it_dob = results.get("iterative_black_box", {}).get("defended_robustness", g_dob)

    results["overall_robustness_score"]          = round((g_rob  + it_rob) / 2, 3)
    results["overall_defended_robustness_score"] = round((g_dob  + it_dob) / 2, 3)

    return results
