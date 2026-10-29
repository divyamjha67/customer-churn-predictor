"""Local retention workbench. Prediction payloads are processed in memory."""
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from churn.schema import Customer, FEATURES, NUMERIC, CATEGORIES
from churn.service import Scorer, ROOT

MAX_BYTES = 1_000_000
MAX_ROWS = 1000

@asynccontextmanager
async def lifespan(application):
    try:
        application.state.scorer = Scorer()
    except (FileNotFoundError, ValueError):
        application.state.scorer = None
    yield

app = FastAPI(title="Customer Churn Workbench", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

def scorer(request):
    engine = getattr(request.app.state, "scorer", None)
    if engine is None:
        raise HTTPException(503, "Model unavailable. Run python -m churn.train and restart the server.")
    return engine

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web/index.html")

@app.get("/api/health")
def health(request: Request):
    engine = scorer(request)
    return {"status": "ready", "model_version": engine.report["model_version"]}

@app.get("/api/schema")
def schema():
    return {"numeric": NUMERIC, "categories": CATEGORIES, "features": FEATURES,
            "max_rows": MAX_ROWS, "max_bytes": MAX_BYTES}

@app.get("/api/report")
def report(request: Request):
    return scorer(request).report

@app.get("/api/example.csv", response_class=Response)
def example():
    return FileResponse(ROOT / "data/examples/customers.csv", media_type="text/csv",
                        filename="benchmark-customers.csv")

@app.post("/api/predict")
def predict(customer: Customer, request: Request, threshold: float | None = Query(None, ge=.01, le=.99)):
    return scorer(request).score([customer.model_dump()], ["Individual customer"], threshold)

def parse_csv(content):
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "Use a UTF-8 CSV file")
    reader = csv.DictReader(io.StringIO(decoded), strict=True)
    try:
        columns = reader.fieldnames or []
    except csv.Error:
        raise HTTPException(422, "Malformed CSV header quoting")
    if len(columns) != len(set(columns)):
        raise HTTPException(422, "Duplicate CSV column names")
    missing = set(FEATURES) - set(columns)
    unknown = set(columns) - set(FEATURES) - {"customerID"}
    if missing or unknown:
        raise HTTPException(422, {"missing_columns": sorted(missing), "unexpected_columns": sorted(unknown)})
    records, identifiers, errors = [], [], []
    try:
        for index, row in enumerate(reader, 2):
            if len(records) + len(errors) >= MAX_ROWS:
                raise HTTPException(413, f"Maximum {MAX_ROWS} customers per upload")
            if None in row or any(value is None for value in row.values()):
                raise HTTPException(422, f"CSV row {index}: wrong number of columns")
            identifier = row.pop("customerID", "").strip() or f"Row {index}"
            if len(identifier) > 80:
                raise HTTPException(422, f"CSV row {index}: customerID must be at most 80 characters")
            try:
                records.append(Customer.model_validate(row).model_dump())
                identifiers.append(identifier)
            except ValidationError as exc:
                errors.append({"row": index, "errors": [{"field": str(e["loc"][0]) if e["loc"] else "services", "message": e["msg"]}
                    for e in exc.errors()]})
    except csv.Error:
        raise HTTPException(422, "Malformed CSV quoting")
    if errors:
        raise HTTPException(422, {"message": "No rows scored. Correct the invalid rows and upload again.",
                                  "invalid_row_count": len(errors), "errors": errors[:10]})
    if not records:
        raise HTTPException(422, "CSV contains no customer rows")
    if len(identifiers) != len(set(identifiers)):
        raise HTTPException(422, "customerID values must be unique")
    return records, identifiers

@app.post("/api/batch")
async def batch(request: Request, threshold: float | None = Query(None, ge=.01, le=.99),
                budget: int | None = Query(None, ge=1, le=MAX_ROWS)):
    engine = scorer(request)
    if request.headers.get("content-type", "").split(";")[0] != "text/csv":
        raise HTTPException(415, "Send the CSV as a raw text/csv request body")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "CSV must be at most 1,000,000 bytes")
    records, identifiers = parse_csv(bytes(content))
    if budget is not None and budget > len(records):
        raise HTTPException(422, "Contact budget cannot exceed the number of uploaded customers")
    return await run_in_threadpool(engine.score, records, identifiers, threshold, budget)
