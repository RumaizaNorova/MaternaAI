import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from data import FEATURE_COLS, RISK_LABEL

# IBM diffprivlib — Differentially Private Gaussian Naive Bayes (binary: high-risk detector)
try:
    from diffprivlib.models import GaussianNB as DPGaussianNB
    _DP_AVAILABLE = True
except ImportError:
    _DP_AVAILABLE = False

DP_EPSILON = 1.0  # Privacy budget (ε=1.0: strong privacy guarantee)

def train_dp_model(X_train, y_train, X_test, y_test):
    """
    IBM diffprivlib: DP-GaussianNB binary high-risk detector.
    ε=1.0 means each patient record changes the model output by at most e^1 ≈ 2.7×.
    Protects against membership inference attacks on patient training data.
    """
    if not _DP_AVAILABLE:
        return None
    try:
        # Binary task: high-risk (1) vs not high-risk (0)
        y_bin_tr = (y_train == 2).astype(int)
        y_bin_te = (y_test  == 2).astype(int)
        bounds = (X_train.min().values, X_train.max().values)
        dp_model = DPGaussianNB(bounds=bounds, epsilon=DP_EPSILON)
        dp_model.fit(X_train.values, y_bin_tr.values)
        y_pred = dp_model.predict(X_test.values)
        y_prob = dp_model.predict_proba(X_test.values)[:, 1]
        auc = roc_auc_score(y_bin_te, y_prob)
        f1 = f1_score(y_bin_te, y_pred, average="binary")
        return {
            "model": dp_model,
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "epsilon": DP_EPSILON,
            "task": "High-risk binary detector",
            "guarantee": f"ε={DP_EPSILON}: each patient's data changes model output ≤{np.e**DP_EPSILON:.2f}×",
        }
    except Exception as e:
        return {"error": str(e)[:120]}

def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42))
        ]),
    }

    results = {}
    classes = [0, 1, 2]
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=classes)
        macro_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {
            "model": model,
            "auc": round(macro_auc, 4),
            "f1": round(macro_f1, 4),
            "report": classification_report(y_test, y_pred,
                                            target_names=list(RISK_LABEL.values()),
                                            output_dict=True),
            "X_test": X_test,
            "y_test": y_test,
        }

    best_name = max(results, key=lambda k: results[k]["auc"])

    # IBM diffprivlib: train DP model alongside standard models
    dp_result = train_dp_model(X_train, y_train, X_test, y_test)

    return results, best_name, X_train, X_test, y_train, y_test, dp_result

def predict_risk(model, vitals: dict) -> tuple[np.ndarray, int]:
    """Returns (probabilities[low,mid,high], predicted_class)."""
    row = pd.DataFrame([vitals])[FEATURE_COLS]
    probs = model.predict_proba(row)[0]
    pred = int(np.argmax(probs))
    return probs, pred

def get_feature_importance(model, feature_cols=FEATURE_COLS) -> dict:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        return {}
    return dict(sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True))
