"""Core ML logic (no UI) - mirrors ML_Mini_Project.ipynb."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

FEATURES = ["Attendance (%)", "Midterm_Score", "Assignments_Avg", "Quizzes_Avg",
            "Participation_Score", "Projects_Score", "Study_Hours_per_Week"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["Academic_Risk"] = df["Grade"].apply(lambda g: "At Risk" if g in ("D", "F") else "Not At Risk")
    df["Academic_Risk_Num"] = (df["Academic_Risk"] == "At Risk").astype(int)
    return df


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10,
                                                min_samples_leaf=3, class_weight="balanced", random_state=42),
    }


def train_all(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df["Academic_Risk_Num"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, rows, curves, matrices = build_models(), [], {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred, prob = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        curves[name], matrices[name] = (fpr, tpr), confusion_matrix(y_te, pred)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred, zero_division=0),
                     "F1 Score": f1_score(y_te, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_te, prob),
                     "CV F1 (5-fold)": cross_val_score(m, X, y, cv=cv, scoring="f1").mean()})
    results = pd.DataFrame(rows)
    best = results.loc[results["F1 Score"].idxmax(), "Model"]
    return {"models": models, "results": results, "best": best, "roc": curves,
            "cm": matrices, "means": X_tr.mean(), "n_train": len(X_tr), "n_test": len(X_te)}


def predict(model, values: dict) -> float:
    """Probability (0-1) that the student is At Risk."""
    return float(model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1])


def explain(model, values: dict, means: pd.Series) -> pd.Series:
    """Model-agnostic explanation: change in risk probability when a feature is
    replaced by the class-average value. Positive = pushes the student toward At Risk."""
    base = predict(model, values)
    out = {}
    for f in FEATURES:
        alt = dict(values); alt[f] = float(means[f])
        out[f] = base - predict(model, alt)
    return pd.Series(out).sort_values()
