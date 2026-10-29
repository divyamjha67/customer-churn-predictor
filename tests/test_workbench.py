import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from churn.api import app, MAX_BYTES
from churn.schema import Customer, FEATURES, frame
from churn.service import Scorer, ROOT

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as connection:
        yield connection

@pytest.fixture
def customer():
    row = pd.read_csv(ROOT / "data/examples/customers.csv", keep_default_na=False).iloc[0]
    return Customer.model_validate(row[FEATURES].to_dict()).model_dump()

def csv_bytes(records, ids=None):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["customerID"] + FEATURES)
    writer.writeheader()
    for index, row in enumerate(records):
        writer.writerow({"customerID": ids[index] if ids else f"Customer {index}", **row})
    return output.getvalue().encode()

def post_csv(client, content, query=""):
    return client.post("/api/batch" + query, content=content, headers={"Content-Type": "text/csv"})

def test_app_and_readiness(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/api/schema").json()["features"] == FEATURES

def test_score_matches_pipeline_and_batch(client, customer):
    individual = client.post("/api/predict", json=customer).json()["results"][0]
    expected = Scorer().model.predict_proba(frame([customer]))[0, 1]
    batch = post_csv(client, csv_bytes([customer])).json()["results"][0]
    assert individual["churn_probability"] == pytest.approx(expected)
    assert batch["churn_probability"] == pytest.approx(expected)

def test_threshold_changes_decision_not_probability(client, customer):
    low = client.post("/api/predict?threshold=0.01", json=customer).json()["results"][0]
    high = client.post("/api/predict?threshold=0.99", json=customer).json()["results"][0]
    assert low["churn_probability"] == high["churn_probability"]
    assert low["above_threshold"] and not high["above_threshold"]

@pytest.mark.parametrize("field,value", [("tenure", -1), ("tenure", 2.5), ("tenure", True),
    ("MonthlyCharges", "NaN"), ("MonthlyCharges", "inf"), ("TotalCharges", -10),
    ("Contract", "Other"), ("InternetService", "invalid"), ("Churn", "Yes")])
def test_invalid_json_rejected(client, customer, field, value):
    customer[field] = value
    assert client.post("/api/predict", json=customer).status_code == 422

def test_missing_required_field(client, customer):
    customer.pop("Contract")
    assert client.post("/api/predict", json=customer).status_code == 422

def test_inconsistent_services_rejected(client, customer):
    customer.update(InternetService="No", OnlineSecurity="Yes")
    assert client.post("/api/predict", json=customer).status_code == 422
    response = post_csv(client, csv_bytes([customer]))
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["row"] == 2

def test_missing_total_charges_supported(client, customer):
    for value in [None, ""]:
        customer["TotalCharges"] = value
        assert client.post("/api/predict", json=customer).status_code == 200
        assert post_csv(client, csv_bytes([customer])).status_code == 200

def test_atomic_batch_validation(client, customer):
    bad = dict(customer, tenure=-1)
    response = post_csv(client, csv_bytes([customer, bad]))
    assert response.status_code == 422
    assert response.json()["detail"]["invalid_row_count"] == 1
    assert "results" not in response.json()

def test_budget_and_stable_ties(client, customer):
    response = post_csv(client, csv_bytes([customer]*3, ["First", "Second", "Third"]), "?budget=2")
    assert response.status_code == 200
    body = response.json()
    assert body["contact_count"] == 2
    assert [row["customer_id"] for row in body["results"]] == ["First", "Second", "Third"]
    assert body["results"][2]["selected_for_contact"] is False
    assert post_csv(client, csv_bytes([customer]), "?budget=2").status_code == 422

def test_batch_sorted_and_budget_exact(client):
    content = (ROOT / "data/examples/customers.csv").read_bytes()
    body = post_csv(client, content, "?budget=5").json()
    scores = [row["churn_probability"] for row in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["contact_count"] == 5
    assert len(body["results"]) == 20

@pytest.mark.parametrize("threshold", [0, 1, -0.1, "NaN"])
def test_threshold_bounds(client, customer, threshold):
    assert client.post(f"/api/predict?threshold={threshold}", json=customer).status_code == 422

def test_csv_contract_and_duplicates(client, customer):
    assert post_csv(client, b"wrong,columns\na,b").status_code == 422
    assert post_csv(client, b"tenure,tenure\n1,2").status_code == 422
    assert post_csv(client, csv_bytes([customer, customer], ["same", "same"])).status_code == 422
    assert post_csv(client, csv_bytes([])).status_code == 422
    assert post_csv(client, b"\xff\xfe").status_code == 422
    assert client.post("/api/batch", content=b"a").status_code == 415

def test_upload_limits(client, customer):
    assert post_csv(client, b"x"*(MAX_BYTES+1)).status_code == 413
    assert post_csv(client, csv_bytes([customer]*1001)).status_code == 413

def test_malformed_csv_header_returns_validation_error(client):
    response = post_csv(client, b'"unterminated header\n1,2,3')
    assert response.status_code == 422
    assert "header quoting" in response.json()["detail"]

def test_ids_are_data(client, customer):
    identifier = '<script>alert(1)</script>'
    body = post_csv(client, csv_bytes([customer], [identifier])).json()
    assert body["results"][0]["customer_id"] == identifier

def test_model_checksum_enforced(tmp_path):
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "pipeline.joblib").write_bytes(b"corrupt artifact")
    with pytest.raises(ValueError, match="checksum"):
        Scorer(tmp_path)

def test_split_integrity_and_fitted_statistics():
    splits = json.loads((ROOT / "artifacts/splits.json").read_text())
    train, validation, test = (set(splits[name]) for name in ["train", "validation", "test"])
    assert not (train & validation or train & test or validation & test)
    assert len(train | validation | test) == 7043
    raw = pd.read_csv(ROOT / "data/raw/telco.csv")
    values = raw.loc[raw.customerID.isin(train), ["tenure", "MonthlyCharges", "TotalCharges"]].apply(pd.to_numeric, errors="coerce")
    imputer = Scorer().model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, values.median().to_numpy())

def test_report_consistency(client):
    report = client.get("/api/report").json()
    assert report["model_name"] == max(report["comparison"], key=lambda row: row["cv_ap_mean"])["name"]
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["split"]["test"]
    assert report["dataset_sha256"] == hashlib.sha256((ROOT / "data/raw/telco.csv").read_bytes()).hexdigest()
    assert 0 < report["test"]["average_precision"] < 1

def test_unavailable_model_returns_actionable_error(client):
    saved = app.state.scorer
    try:
        app.state.scorer = None
        assert client.get("/api/health").status_code == 503
        assert "churn.train" in client.get("/api/report").json()["detail"]
    finally:
        app.state.scorer = saved
