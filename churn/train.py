"""Reproducible model selection; validation chooses threshold, test reports results."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churn.schema import Customer, FEATURES, NUMERIC, CATEGORIES, frame

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def pipeline(estimator):
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), list(NUMERIC)),
        ("category", OneHotEncoder(handle_unknown="error", sparse_output=False), list(CATEGORIES)),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])

def metrics(y, probability, threshold):
    prediction = probability >= threshold
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction)),
            "f1": float(f1_score(y, prediction)),
            "f2": float(fbeta_score(y, prediction, beta=2)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}

def capacity(y, probability, share):
    count = max(1, int(np.ceil(len(y) * share)))
    selected = np.argsort(-np.asarray(probability), kind="stable")[:count]
    captured = int(np.asarray(y)[selected].sum())
    precision = captured / count
    return {"share": share, "contact_count": count, "churners_captured": captured,
            "precision": precision, "recall": captured / int(np.asarray(y).sum()),
            "lift": precision / float(np.mean(y))}

def main():
    raw_path = ROOT / "data/raw/telco.csv"
    if not raw_path.exists():
        raise SystemExit("Run python scripts/download_data.py first")
    source = json.loads((ROOT / "data/source.json").read_text())
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("Source data checksum mismatch")
    data = pd.read_csv(raw_path, keep_default_na=False)
    if len(data) != 7043 or data.customerID.duplicated().any() or set(data.Churn) != {"Yes", "No"}:
        raise ValueError("Unexpected dataset rows, duplicate IDs or labels")
    records = [Customer.model_validate(row).model_dump() for row in data[FEATURES].to_dict("records")]
    X, y = frame(records), (data.Churn == "Yes").astype(int)
    # Split before fitting *any* learned transform. No final refit on test or validation.
    train_ids, remainder = train_test_split(np.arange(len(data)), test_size=.4, stratify=y, random_state=SEED)
    validation_ids, test_ids = train_test_split(remainder, test_size=.5, stratify=y.iloc[remainder], random_state=SEED)
    candidates = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression (C=0.1)": LogisticRegression(C=.1, max_iter=2000, random_state=SEED),
        "Logistic regression (C=1)": LogisticRegression(C=1, max_iter=2000, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
            learning_rate=.05, l2_regularization=5, random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    comparison = []
    for name, estimator in candidates.items():
        scores = cross_val_score(pipeline(estimator), X.iloc[train_ids], y.iloc[train_ids],
                                 cv=cv, scoring="average_precision", n_jobs=1)
        comparison.append({"name": name, "cv_ap_mean": float(scores.mean()),
                           "cv_ap_std": float(scores.std()), "fold_ap": scores.tolist()})
        print(name, round(scores.mean(), 4), flush=True)
    winner = max(comparison, key=lambda item: item["cv_ap_mean"])["name"]
    model = pipeline(candidates[winner]).fit(X.iloc[train_ids], y.iloc[train_ids])
    val_y = y.iloc[validation_ids]
    val_probability = model.predict_proba(X.iloc[validation_ids])[:, 1]
    thresholds = np.arange(.05, .951, .01)
    f2_scores = [fbeta_score(val_y, val_probability >= t, beta=2) for t in thresholds]
    # If tied, choose the higher threshold (fewer contacts).
    threshold = float(thresholds[np.flatnonzero(np.isclose(f2_scores, max(f2_scores)))[-1]])
    test_y = y.iloc[test_ids]
    test_probability = model.predict_proba(X.iloc[test_ids])[:, 1]
    fraction, mean = calibration_curve(test_y, test_probability, n_bins=8, strategy="quantile")
    importance = permutation_importance(model, X.iloc[validation_ids], val_y, n_repeats=5,
        scoring="average_precision", random_state=SEED, n_jobs=1)
    # Percentile intervals describe sampling uncertainty on this fixed test split.
    rng = np.random.default_rng(SEED)
    bootstrap = {"roc_auc": [], "average_precision": []}
    for _ in range(300):
        sample = rng.integers(0, len(test_y), len(test_y))
        labels, probabilities = test_y.to_numpy()[sample], test_probability[sample]
        if len(np.unique(labels)) == 2:
            bootstrap["roc_auc"].append(roc_auc_score(labels, probabilities))
            bootstrap["average_precision"].append(average_precision_score(labels, probabilities))
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    model_path = artifact_dir / "pipeline.joblib"
    joblib.dump(model, model_path)
    report = {
        "model_version": "2.0.0", "schema_version": "1", "model_name": winner,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "dataset_sha256": digest, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "rows": len(data), "churn_rate": float(y.mean()), "seed": SEED, "features": FEATURES,
        "split": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "comparison": comparison, "threshold": round(threshold, 2),
        "threshold_objective": "Maximum validation F2 on 0.05..0.95 (0.01 steps); beta=2 favors recall",
        "validation": metrics(val_y, val_probability, threshold),
        "test": metrics(test_y, test_probability, threshold),
        "test_at_0_5": metrics(test_y, test_probability, .5),
        "test_positive_rate": float(test_y.mean()),
        "test_ci95": {key: np.quantile(values, [.025, .975]).tolist() for key, values in bootstrap.items()},
        "capacity": [capacity(test_y, test_probability, share) for share in [.05, .1, .2, .3, .5]],
        "calibration": [{"predicted": float(p), "observed": float(o)} for p, o in zip(mean, fraction)],
        "importance": sorted([{"feature": name, "ap_drop": float(value), "std": float(std)}
            for name, value, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
            key=lambda item: item["ap_drop"], reverse=True),
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (artifact_dir / "splits.json").write_text(json.dumps({name: data.customerID.iloc[ids].tolist()
        for name, ids in [("train", train_ids), ("validation", validation_ids), ("test", test_ids)]}, indent=2))
    (ROOT / "data/examples").mkdir(parents=True, exist_ok=True)
    # Explicitly a benchmark excerpt; never described as live incoming customers.
    data.iloc[test_ids[:20]][["customerID"] + FEATURES].to_csv(ROOT / "data/examples/customers.csv", index=False)
    print(json.dumps({"selected": winner, "threshold": threshold, "test": report["test"]}, indent=2))

if __name__ == "__main__":
    main()
