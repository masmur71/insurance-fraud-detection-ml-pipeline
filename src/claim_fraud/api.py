"""Serve a trusted frozen bundle: uvicorn claim_fraud.api:app."""
import hashlib
import logging
import math
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from .monitoring import Monitor
from .schemas import ClaimRequest, HealthResponse, PredictionResponse


def load_bundle(path):
    # Pickle/joblib is executable: use only local, trusted model artifacts.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(path)
    metadata = bundle["metadata"]
    if metadata["sklearn"] != sklearn.__version__:
        raise ValueError("Model and serving scikit-learn versions must match")
    threshold = bundle["threshold"]
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("Invalid model threshold")
    if metadata["dataset"] != "synthetic data" or metadata["seed"] != 20261028:
        raise ValueError("Unexpected model dataset metadata")
    features = bundle["pipeline"].named_steps["features"].input_columns_
    if set(features) != set(ClaimRequest.model_fields) or set(metadata["input_columns"]) != set(features):
        raise ValueError("Model features do not match the API request contract")
    bundle["model_version"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return bundle


def reason_codes(claim):
    """Input rule indicators, not model feature attribution or fraud evidence."""
    reasons = []
    if claim.report_date < claim.incident_date:
        reasons.append("INVALID_REPORT_DELAY")
    if (claim.claim_amount is not None and claim.sum_insured is not None
            and claim.sum_insured > 0 and claim.claim_amount > claim.sum_insured):
        reasons.append("CLAIM_EXCEEDS_SUM_INSURED")
    if claim.police_report_filed == 0:
        reasons.append("NO_POLICE_REPORT")
    return reasons


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.environ.get("MODEL_PATH", "models/phase-3/model.joblib"))
        app.state.bundle = load_bundle(path)
        reference = app.state.bundle["metadata"].get("selected", {}).get(
            "validation", {}).get("review_fraction", 0.05)
        app.state.monitor = Monitor(reference)
        logger = logging.getLogger("claim_fraud.predictions")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        # Apply once per process rather than mutating thread limits per request.
        with threadpool_limits(limits=1):
            yield

    app = FastAPI(title="Synthetic claim review API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_metrics(request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = request.url.path
            route = path if path in {"/predict", "/health", "/metrics", "/monitoring"} else "other"
            app.state.monitor.observe_request(route, status, time.perf_counter() - start)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint():
        return PlainTextResponse(app.state.monitor.prometheus(),
                                 media_type="text/plain; version=0.0.4")

    @app.get("/monitoring")
    def monitoring():
        return app.state.monitor.snapshot()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Raw invalid values can contain NaN/Infinity and cannot be JSON-serialized.
        detail = [{key: error[key] for key in ("loc", "msg", "type")}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(model_version=app.state.bundle["model_version"])

    @app.post("/predict", response_model=PredictionResponse)
    def predict(claim: ClaimRequest):
        bundle = app.state.bundle
        frame = pd.DataFrame([claim.model_dump()])
        score = float(bundle["pipeline"].predict_proba(frame)[0, 1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Model returned an invalid score")
        app.state.monitor.observe_prediction(score, score >= bundle["threshold"],
                                             bundle["model_version"])
        return PredictionResponse(score=score, threshold=bundle["threshold"],
                                  review_required=score >= bundle["threshold"],
                                  reason_codes=reason_codes(claim),
                                  model_version=bundle["model_version"])

    return app


app = create_app()
