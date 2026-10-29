"""Load only the locally trained artifact and apply its complete pipeline."""
from pathlib import Path
import hashlib
import json
import joblib
from churn.schema import frame

ROOT = Path(__file__).resolve().parents[1]

class Scorer:
    def __init__(self, directory=ROOT / "artifacts"):
        self.report = json.loads((directory / "report.json").read_text())
        path = directory / "pipeline.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.report["model_sha256"]:
            raise ValueError("Model checksum mismatch; retrain the local artifact")
        # joblib is executable serialization: only load artifacts you trained and trust.
        self.model = joblib.load(path)

    def score(self, records, identifiers, threshold=None, budget=None):
        cutoff = self.report["threshold"] if threshold is None else threshold
        probabilities = self.model.predict_proba(frame(records))[:, 1]
        rows = [{"customer_id": identifier, "churn_probability": float(probability),
                 "above_threshold": bool(probability >= cutoff)}
                for identifier, probability in zip(identifiers, probabilities)]
        rows.sort(key=lambda row: -row["churn_probability"])
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
            row["selected_for_contact"] = rank <= budget if budget is not None else row["above_threshold"]
        return {"model_version": self.report["model_version"], "threshold": cutoff,
                "selection_rule": "top_n" if budget is not None else "threshold",
                "customer_count": len(rows), "contact_count": sum(r["selected_for_contact"] for r in rows),
                "mean_probability": float(probabilities.mean()), "results": rows}
