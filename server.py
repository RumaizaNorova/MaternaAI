"""
MaternaAI — Flask API server.
Four IBM open-source libraries: AIF360, ART, AIX360, diffprivlib.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import subprocess
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd

from data import load_dataset, get_binary_df, FEATURE_COLS, FEATURE_LABELS, RISK_LABEL
from model import train_models, predict_risk, get_feature_importance, DP_EPSILON
from fairness_engine import run_audit, verdict
from uncertainty import calibrate_model, bootstrap_confidence, calibration_metrics
from adversarial import run_adversarial_audit
from explainability import lime_explain, counterfactual_explain
from granite import explain_risk, governance_policy

app = Flask(__name__)

# Expose a build id in the UI to make deploy/debug obvious
def _build_id() -> str:
    env = os.environ.get("BUILD_ID")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"

BUILD_ID = _build_id()

# ── Startup pipeline ───────────────────────────────────────────────────────────
print("MaternaAI — initialising pipeline…")
df     = load_dataset()
df_bin = get_binary_df(df)
results, best_name, X_train, X_test, y_train, y_test, dp_result = train_models(df)
model  = results[best_name]["model"]

print("  Calibrating…")
cal_model = calibrate_model(model, X_train, y_train)
uq_cal    = calibration_metrics(cal_model, X_test, y_test)

print("  IBM AIF360 — three-strategy fairness audit…")
ytrb = df_bin.loc[X_train.index, "label_binary"].values
yteb = df_bin.loc[X_test.index,  "label_binary"].values
str_ = df_bin.loc[X_train.index, "age_group"].values
ste_ = df_bin.loc[X_test.index,  "age_group"].values
fairness_data    = run_audit(model, X_train, ytrb, X_test, yteb, str_, ste_, FEATURE_COLS)
fairness_verdict = verdict(fairness_data.get("original", {}))

print("  IBM ART — adversarial attack + FeatureSqueezing defense…")
adv_data = run_adversarial_audit(model, X_test, y_test)

print("  IBM diffprivlib — BudgetAccountant initialised…")
from diffprivlib.accountant import BudgetAccountant
_privacy_accountant = BudgetAccountant(epsilon=10.0, delta=0)
_query_count = 0

print(f"  Ready. Best: {best_name} AUC={results[best_name]['auc']}")


# ── Serialisation helper ───────────────────────────────────────────────────────
_NON_SERIAL = {"mitigated_model"}

def _safe(obj):
    if isinstance(obj, (np.bool_,)):   return bool(obj)
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items() if k not in _NON_SERIAL}
    if isinstance(obj, list):          return [_safe(i) for i in obj]
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return None   # drop sklearn models and other non-serialisable objects
    return obj


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", build_id=BUILD_ID)


@app.route("/api/predict", methods=["POST"])
def predict():
    global _query_count
    try:
        data = request.get_json()
        vitals = {
            "Age":         float(data["age"]),
            "SystolicBP":  float(data["systolic"]),
            "DiastolicBP": float(data["diastolic"]),
            "BS":          float(data["glucose"]),
            "BodyTemp":    float(data["temp"]),
            "HeartRate":   float(data["hr"]),
        }

        probs, pred = predict_risk(cal_model, vitals)
        X_row = pd.DataFrame([vitals])[FEATURE_COLS]

        # IBM UQ360 methodology — bootstrap 90% CI
        uq = bootstrap_confidence(cal_model, X_row)

        # IBM AIX360 — LIME local explanation
        lime = lime_explain(model, X_train, X_row, pred)

        # IBM AIX360 — counterfactual minimum change
        cf = counterfactual_explain(model, X_row, pred)

        # Clinical AI brief
        fi    = get_feature_importance(model)
        top_f = list(fi.items())[:3]
        brief = explain_risk(vitals, RISK_LABEL[pred], list(probs), uq, top_f)

        # IBM diffprivlib — track privacy budget
        _query_count += 1
        epsilon_per_query = 0.05
        total_spent = round(min(_query_count * epsilon_per_query, 10.0), 3)
        budget_remaining = round(max(10.0 - total_spent, 0.0), 3)

        return jsonify(_safe({
            "pred":            pred,
            "label":           RISK_LABEL[pred],
            "probs":           list(probs),
            "ci_low":          uq["ci_low"],
            "ci_high":         uq["ci_high"],
            "confidence":      uq["confidence"],
            "interval_label":  uq["interval_label"],
            "lime":            lime,
            "counterfactual":  cf,
            "brief":           brief,
            "is_teen":         vitals["Age"] <= 19,
            "privacy_budget":  {
                "total":     10.0,
                "spent":     total_spent,
                "remaining": budget_remaining,
                "query_num": _query_count,
                "epsilon_per_query": epsilon_per_query,
            },
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fairness")
def fairness():
    orig = fairness_data.get("original", {})
    rw   = fairness_data.get("reweighing", {})
    dir_ = fairness_data.get("dir_mitigation", {})
    ceop = fairness_data.get("calibrated_eq_odds", {})
    vd   = fairness_verdict
    policy = governance_policy(orig, rw)
    return jsonify(_safe({
        "original":          orig,
        "reweighing":        rw,
        "dir_mitigation":    dir_,
        "calibrated_eq_odds":ceop,
        "verdict":           vd,
        "policy":            policy,
    }))


@app.route("/api/security")
def security():
    return jsonify(_safe(adv_data))


@app.route("/api/privacy-budget")
def privacy_budget():
    epsilon_per_query = 0.05
    total_spent = round(min(_query_count * epsilon_per_query, 10.0), 3)
    dp = {k: v for k, v in (dp_result or {}).items() if k != "model"}
    return jsonify(_safe({
        "accountant": {
            "total_epsilon": 10.0,
            "spent":         total_spent,
            "remaining":     round(max(10.0 - total_spent, 0.0), 3),
            "query_count":   _query_count,
            "epsilon_per_query": epsilon_per_query,
        },
        "dp_model": dp,
        "dp_epsilon": DP_EPSILON,
    }))


@app.route("/api/model-info")
def model_info():
    fi  = get_feature_importance(model)
    dp  = {k: v for k, v in (dp_result or {}).items() if k != "model"}
    rep = results[best_name]["report"]
    return jsonify(_safe({
        "best_name":        best_name,
        "models":           {n: {"auc": r["auc"], "f1": r["f1"]} for n, r in results.items()},
        "report":           rep,
        "feature_importance": fi,
        "feature_labels":   FEATURE_LABELS,
        "dp":               dp,
        "dp_epsilon":       DP_EPSILON,
        "calibration":      uq_cal,
        "dataset_size":     len(df),
        "high_risk_pct":    round(float((df["label"] == 2).mean() * 100), 1),
        "teen_pct":         round(float((df["Age"] <= 19).mean() * 100), 1),
    }))


@app.route("/api/population")
def population():
    records = [
        {
            "age":        int(row["Age"]),
            "sbp":        float(row["SystolicBP"]),
            "glucose":    float(row["BS"]),
            "label":      int(row["label"]),
            "risk_label": RISK_LABEL[int(row["label"])],
        }
        for _, row in df.iterrows()
    ]
    return jsonify(records)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
