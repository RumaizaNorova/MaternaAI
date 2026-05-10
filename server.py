"""
MaternaAI — Flask API server.
Replaces Streamlit. ML pipeline stays identical.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd

from data import load_dataset, get_binary_df, FEATURE_COLS, FEATURE_LABELS, RISK_LABEL, RISK_COLOR
from model import train_models, predict_risk, get_feature_importance, DP_EPSILON
from fairness_engine import run_audit, verdict
from uncertainty import calibrate_model, bootstrap_confidence, calibration_metrics
from adversarial import run_adversarial_audit
from explainability import lime_explain
from granite import explain_risk, governance_policy, is_live

app = Flask(__name__)

# ── One-time startup pipeline ──────────────────────────────────────────────────
print("MaternaAI — initialising pipeline…")
df     = load_dataset()
df_bin = get_binary_df(df)
results, best_name, X_train, X_test, y_train, y_test, dp_result = train_models(df)
model  = results[best_name]["model"]

print("  Calibrating…")
cal_model  = calibrate_model(model, X_train, y_train)
uq_cal     = calibration_metrics(cal_model, X_test, y_test)

print("  IBM AIF360 fairness audit…")
ytrb = df_bin.loc[X_train.index, "label_binary"].values
yteb = df_bin.loc[X_test.index,  "label_binary"].values
str_ = df_bin.loc[X_train.index, "age_group"].values
ste_ = df_bin.loc[X_test.index,  "age_group"].values
fairness_data = run_audit(model, X_train, ytrb, X_test, yteb, str_, ste_, FEATURE_COLS)
fairness_verdict = verdict(fairness_data.get("original", {}))

print("  IBM ART adversarial audit…")
adv_data = run_adversarial_audit(model, X_test, y_test)

print("  IBM diffprivlib DP model ready.")
print(f"  Ready. Best model: {best_name} AUC={results[best_name]['auc']}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _np_safe(obj):
    """JSON-serialise numpy scalars and strip non-serialisable objects."""
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_np_safe(i) for i in obj]
    # Drop anything that isn't a primitive (e.g. sklearn model objects)
    if not isinstance(obj, (str, int, float, bool, type(None))): return str(type(obj).__name__)
    return obj


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
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

        # IBM UQ360 methodology — bootstrap CI
        uq = bootstrap_confidence(cal_model, X_row)

        # IBM AIX360 — LIME local explanation
        lime = lime_explain(model, X_train, X_row, pred)

        # AI clinical brief (Qwen)
        fi = get_feature_importance(model)
        top_f = list(fi.items())[:3]
        brief = explain_risk(vitals, RISK_LABEL[pred], list(probs), uq, top_f)

        return jsonify(_np_safe({
            "pred": pred,
            "label": RISK_LABEL[pred],
            "probs": list(probs),
            "ci_low":  uq["ci_low"],
            "ci_high": uq["ci_high"],
            "confidence": uq["confidence"],
            "interval_label": uq["interval_label"],
            "lime": lime,
            "brief": brief,
            "is_teen": vitals["Age"] <= 19,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fairness")
def fairness():
    # Strip non-serializable objects (model) from mitigated dict
    _SKIP = {"mitigated_model"}
    orig = {k: v for k, v in fairness_data.get("original", {}).items() if k not in _SKIP}
    mit  = {k: v for k, v in fairness_data.get("mitigated", {}).items() if k not in _SKIP}
    vd   = fairness_verdict
    policy = governance_policy(orig, mit)
    return jsonify(_np_safe({
        "original": orig,
        "mitigated": mit,
        "verdict": vd,
        "policy": policy,
    }))


@app.route("/api/security")
def security():
    return jsonify(_np_safe(adv_data))


@app.route("/api/model-info")
def model_info():
    rep = results[best_name]["report"]
    fi  = get_feature_importance(model)
    dp  = {k: v for k, v in (dp_result or {}).items() if k != "model"}
    return jsonify(_np_safe({
        "best_name": best_name,
        "models": {
            name: {"auc": r["auc"], "f1": r["f1"]}
            for name, r in results.items()
        },
        "report": rep,
        "feature_importance": fi,
        "feature_labels": FEATURE_LABELS,
        "dp": dp,
        "dp_epsilon": DP_EPSILON,
        "calibration": uq_cal,
        "dataset_size": len(df),
        "high_risk_pct": round(float((df["label"] == 2).mean() * 100), 1),
        "teen_pct": round(float((df["Age"] <= 19).mean() * 100), 1),
    }))


@app.route("/api/population")
def population():
    records = []
    for _, row in df.iterrows():
        records.append({
            "age": int(row["Age"]),
            "sbp": float(row["SystolicBP"]),
            "glucose": float(row["BS"]),
            "label": int(row["label"]),
            "risk_label": RISK_LABEL[int(row["label"])],
        })
    return jsonify(records)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
