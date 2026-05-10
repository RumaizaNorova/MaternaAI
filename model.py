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
    return results, best_name, X_train, X_test, y_train, y_test

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
