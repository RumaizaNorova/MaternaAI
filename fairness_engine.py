"""
IBM AI Fairness 360 — maternal health equity audit.
Sensitive attribute: age_group (teen mothers ≤19 vs adult mothers ≥20).
Teenage mothers face the highest mortality risk AND the most systemic neglect.
"""
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.ensemble import GradientBoostingClassifier

PRIVILEGED   = [{"age_group": 1}]   # adult mothers (20+)
UNPRIVILEGED = [{"age_group": 0}]   # teen mothers (≤19)

def _make_aif_dataset(X, y_binary, sensitive, feature_cols):
    df = pd.DataFrame(
        X.values if hasattr(X, "values") else np.array(X),
        columns=feature_cols
    )
    df["age_group"] = np.array(sensitive)
    df["label"]     = np.array(y_binary)
    return BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["age_group"],
        favorable_label=0,       # 0 = NOT high risk (favorable outcome)
        unfavorable_label=1,
        privileged_protected_attributes=[[1]],
    )

def run_audit(model, X_train, y_train_bin, X_test, y_test_bin,
              s_train, s_test, feature_cols) -> dict:
    results = {}

    # ── Original model ──────────────────────────────────────────────────────
    ds_test  = _make_aif_dataset(X_test, y_test_bin, s_test, feature_cols)
    y_pred   = model.predict(X_test)
    y_pred_b = (y_pred == 2).astype(int)   # high-risk vs not

    df_pred  = pd.DataFrame(
        X_test.values if hasattr(X_test, "values") else np.array(X_test),
        columns=feature_cols
    )
    df_pred["age_group"] = np.array(s_test)
    df_pred["label"]     = y_pred_b
    ds_pred = BinaryLabelDataset(
        df=df_pred, label_names=["label"],
        protected_attribute_names=["age_group"],
        favorable_label=0, unfavorable_label=1,
        privileged_protected_attributes=[[1]],
    )

    m = ClassificationMetric(ds_test, ds_pred,
                             privileged_groups=PRIVILEGED,
                             unprivileged_groups=UNPRIVILEGED)
    results["original"] = {
        "disparate_impact":               round(m.disparate_impact(), 3),
        "statistical_parity_difference":  round(m.statistical_parity_difference(), 3),
        "equal_opportunity_difference":   round(m.equal_opportunity_difference(), 3),
        "average_odds_difference":        round(m.average_odds_difference(), 3),
        "theil_index":                    round(m.theil_index(), 4),
    }

    # ── IBM AIF360 Reweighing mitigation ────────────────────────────────────
    try:
        ds_train = _make_aif_dataset(X_train, y_train_bin, s_train, feature_cols)
        rw       = Reweighing(unprivileged_groups=UNPRIVILEGED,
                              privileged_groups=PRIVILEGED)
        ds_train_rw = rw.fit_transform(ds_train)

        n_feat = len(feature_cols)
        mit_model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.08, random_state=42)
        mit_model.fit(
            ds_train_rw.features[:, :n_feat],
            ds_train_rw.labels.ravel(),
            sample_weight=ds_train_rw.instance_weights,
        )

        X_test_arr  = X_test.values if hasattr(X_test, "values") else np.array(X_test)
        y_pred_mit  = mit_model.predict(X_test_arr)
        y_pred_mit_b = (y_pred_mit == 2).astype(int)

        df_mit = pd.DataFrame(X_test_arr, columns=feature_cols)
        df_mit["age_group"] = np.array(s_test)
        df_mit["label"]     = y_pred_mit_b
        ds_pred_mit = BinaryLabelDataset(
            df=df_mit, label_names=["label"],
            protected_attribute_names=["age_group"],
            favorable_label=0, unfavorable_label=1,
            privileged_protected_attributes=[[1]],
        )
        m2 = ClassificationMetric(ds_test, ds_pred_mit,
                                  privileged_groups=PRIVILEGED,
                                  unprivileged_groups=UNPRIVILEGED)
        results["mitigated"] = {
            "disparate_impact":               round(m2.disparate_impact(), 3),
            "statistical_parity_difference":  round(m2.statistical_parity_difference(), 3),
            "equal_opportunity_difference":   round(m2.equal_opportunity_difference(), 3),
            "average_odds_difference":        round(m2.average_odds_difference(), 3),
            "theil_index":                    round(m2.theil_index(), 4),
            "mitigated_model":                mit_model,
        }
    except Exception as e:
        results["mitigated"] = {"error": str(e)}

    return results

def verdict(metrics: dict) -> dict:
    di  = metrics.get("disparate_impact", 1.0)
    spd = metrics.get("statistical_parity_difference", 0.0)
    eod = metrics.get("equal_opportunity_difference", 0.0)
    aod = metrics.get("average_odds_difference", 0.0)
    return {
        "di_pass":   di  >= 0.80,
        "spd_pass":  abs(spd) <= 0.10,
        "eod_pass":  abs(eod) <= 0.10,
        "aod_pass":  abs(aod) <= 0.10,
        "fair":      di >= 0.80 and abs(spd) <= 0.10,
    }
