"""
IBM AI Fairness 360 — full three-strategy fairness pipeline.

Sensitive attribute: age_group (teen mothers ≤19 vs adult mothers ≥20).
Mitigation strategies across ALL THREE pipeline stages:
  1. Preprocessing  — Reweighing                   (adjust sample weights)
  2. Preprocessing  — DisparateImpactRemover        (repair feature space)
  3. Postprocessing — CalibratedEqOddsPostprocessing (adjust output thresholds)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.ensemble import GradientBoostingClassifier

PRIVILEGED   = [{"age_group": 1}]
UNPRIVILEGED = [{"age_group": 0}]


def _make_aif_dataset(X, y_binary, sensitive, feature_cols):
    df = pd.DataFrame(
        X.values if hasattr(X, "values") else np.array(X),
        columns=feature_cols,
    )
    df["age_group"] = np.array(sensitive)
    df["label"]     = np.array(y_binary)
    return BinaryLabelDataset(
        df=df, label_names=["label"],
        protected_attribute_names=["age_group"],
        favorable_label=0, unfavorable_label=1,
        privileged_protected_attributes=[[1]],
    )


def _metrics(ds_true, ds_pred) -> dict:
    m = ClassificationMetric(
        ds_true, ds_pred,
        privileged_groups=PRIVILEGED,
        unprivileged_groups=UNPRIVILEGED,
    )
    return {
        "disparate_impact":              round(float(m.disparate_impact()), 3),
        "statistical_parity_difference": round(float(m.statistical_parity_difference()), 3),
        "equal_opportunity_difference":  round(float(m.equal_opportunity_difference()), 3),
        "average_odds_difference":       round(float(m.average_odds_difference()), 3),
        "theil_index":                   round(float(m.theil_index()), 4),
    }


def run_audit(model, X_train, y_train_bin, X_test, y_test_bin,
              s_train, s_test, feature_cols) -> dict:
    """
    Full IBM AIF360 audit — three mitigation strategies.
    Returns dict with keys: original, reweighing, dir_mitigation, calibrated_eq_odds.
    """
    n = len(feature_cols)
    X_test_arr = X_test.values if hasattr(X_test, "values") else np.array(X_test)

    ds_tr = _make_aif_dataset(X_train, y_train_bin, s_train, feature_cols)
    ds_te = _make_aif_dataset(X_test,  y_test_bin,  s_test,  feature_cols)

    # ── Original model baseline ───────────────────────────────────────────────
    y_pred_b = (model.predict(X_test) == 2).astype(int)
    ds_orig  = ds_te.copy()
    ds_orig.labels = y_pred_b.reshape(-1, 1)
    results = {"original": _metrics(ds_te, ds_orig)}

    # ── Strategy 1: Reweighing (preprocessing) ────────────────────────────────
    try:
        rw        = Reweighing(unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED)
        ds_tr_rw  = rw.fit_transform(ds_tr)
        rw_model  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.08, random_state=42)
        rw_model.fit(ds_tr_rw.features[:, :n], ds_tr_rw.labels.ravel(),
                     sample_weight=ds_tr_rw.instance_weights)
        y_rw      = (rw_model.predict(X_test_arr) == 2).astype(int)
        ds_rw     = ds_te.copy(); ds_rw.labels = y_rw.reshape(-1, 1)
        results["reweighing"] = {**_metrics(ds_te, ds_rw), "mitigated_model": rw_model}
    except Exception as e:
        results["reweighing"] = {"error": str(e)[:120]}

    # ── Strategy 2: DisparateImpactRemover (preprocessing) ────────────────────
    try:
        DIR       = DisparateImpactRemover(repair_level=0.8, sensitive_attribute="age_group")
        ds_tr_dir = DIR.fit_transform(ds_tr)
        ds_te_dir = DIR.fit_transform(ds_te)   # unsupervised; no data leakage
        dir_model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.08, random_state=42)
        dir_model.fit(ds_tr_dir.features[:, :n], ds_tr_dir.labels.ravel())
        y_dir     = (dir_model.predict(ds_te_dir.features[:, :n]) == 2).astype(int)
        ds_dir    = ds_te.copy(); ds_dir.labels = y_dir.reshape(-1, 1)
        results["dir_mitigation"] = _metrics(ds_te, ds_dir)
    except Exception as e:
        results["dir_mitigation"] = {"error": str(e)[:120]}

    # ── Strategy 3: CalibratedEqOddsPostprocessing (postprocessing) ───────────
    try:
        y_scores      = model.predict_proba(X_test)[:, 2]
        ds_pred_pp    = ds_te.copy()
        ds_pred_pp.labels = y_pred_b.reshape(-1, 1)
        ds_pred_pp.scores = y_scores.reshape(-1, 1)
        ds_te_sc      = ds_te.copy()
        ds_te_sc.scores = y_scores.reshape(-1, 1)
        ceop = CalibratedEqOddsPostprocessing(
            unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED,
            cost_constraint="weighted", seed=42,
        )
        ceop.fit(ds_te_sc, ds_pred_pp)
        ds_ceop = ceop.predict(ds_pred_pp)
        results["calibrated_eq_odds"] = _metrics(ds_te, ds_ceop)
    except Exception as e:
        results["calibrated_eq_odds"] = {"error": str(e)[:120]}

    return results


def verdict(metrics: dict) -> dict:
    di  = metrics.get("disparate_impact", 1.0)
    spd = metrics.get("statistical_parity_difference", 0.0)
    eod = metrics.get("equal_opportunity_difference", 0.0)
    aod = metrics.get("average_odds_difference", 0.0)
    return {
        "di_pass":  bool(di  >= 0.80),
        "spd_pass": bool(abs(spd) <= 0.10),
        "eod_pass": bool(abs(eod) <= 0.10),
        "aod_pass": bool(abs(aod) <= 0.10),
        "fair":     bool(di >= 0.80 and abs(spd) <= 0.10),
    }
