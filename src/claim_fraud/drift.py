"""Local distribution checks on fitted, pre-imputation model features."""
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
from typing import get_args

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from .data import DATES, assemble
from .schemas import ClaimRequest

VERSION = 1
THRESHOLDS = {"missing_rate": 0.05, "standardized_mean_shift": 0.5,
              "total_variation": 0.1}


def _integer_annotation(annotation):
    return annotation is int or any(_integer_annotation(item) for item in get_args(annotation))


INTEGER_FIELDS = {name for name, field in ClaimRequest.model_fields.items()
                  if _integer_annotation(field.annotation)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def training_inputs(data_dir):
    """Push training cutoff into Parquet reading before assembling histories."""
    frame = assemble(data_dir, before=pd.Timestamp("2024-01-01"))
    return frame.loc[frame.split.eq("train") & frame.eligible,
                     list(ClaimRequest.model_fields)].reset_index(drop=True)


def validate_claims(frame):
    if frame.empty:
        raise ValueError("Input must contain at least one claim")
    if not frame.columns.is_unique or set(frame.columns) != set(ClaimRequest.model_fields):
        raise ValueError("Input columns must exactly match ClaimRequest fields")
    records = []
    for index, record in enumerate(frame.to_dict(orient="records")):
        for key, value in record.items():
            if pd.api.types.is_scalar(value) and pd.isna(value):
                record[key] = None
            elif (key in INTEGER_FIELDS and isinstance(value, float)
                  and np.isfinite(value) and value.is_integer()):
                # Nullable integer Parquet columns can arrive as float64 in pandas.
                record[key] = int(value)
            elif key in DATES and isinstance(value, (pd.Timestamp, datetime)):
                if value.tzinfo is not None or value.time() != datetime.min.time():
                    raise ValueError(f"Invalid claim at row {index}, field {key}: calendar_date")
                record[key] = value.date()
        try:
            records.append(ClaimRequest.model_validate(record).model_dump())
        except ValidationError as exc:
            locations = [{"row": index, "field": list(item["loc"]), "type": item["type"]}
                         for item in exc.errors(include_input=False, include_context=False,
                                                include_url=False)]
            raise ValueError("Invalid claim: " + json.dumps(locations)) from None
    return pd.DataFrame(records)


def feature_statistics(frame):
    statistics = {}
    for name in frame:
        values = frame[name].dropna()
        entry = {"kind": "numeric" if pd.api.types.is_numeric_dtype(frame[name]) else "categorical",
                 "missing_rate": float(frame[name].isna().mean()), "available": len(values)}
        if entry["kind"] == "numeric":
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite transformed feature: {name}")
            entry.update(mean=float(values.mean()) if len(values) else None,
                         std=float(values.std(ddof=0)) if len(values) else None,
                         minimum=float(values.min()) if len(values) else None,
                         maximum=float(values.max()) if len(values) else None)
        else:
            entry["probabilities"] = {str(key): float(value) for key, value in
                                      values.value_counts(normalize=True).sort_index().items()}
        statistics[name] = entry
    return statistics


def load_model(path):
    bundle = joblib.load(path)
    transformer = bundle["pipeline"].named_steps["features"]
    if not hasattr(transformer, "input_columns_"):
        raise ValueError("Model feature transformer must already be fitted")
    return bundle, transformer


def build_reference(model_path, frame, source):
    _, transformer = load_model(model_path)
    valid = validate_claims(frame)
    return {"profile_version": VERSION, "model_sha256": sha256(model_path),
            "source": source, "sample_size": len(valid),
            "training_input_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(valid, index=False).values.tobytes()).hexdigest(),
            "features": feature_statistics(transformer.transform(valid))}


def compare_features(reference, current):
    results = {}
    for name, ref in reference.items():
        now = current[name]
        missing = now["missing_rate"] - ref["missing_rate"]
        flags = []
        if abs(missing) > THRESHOLDS["missing_rate"]:
            flags.append("missing_rate")
        result = {"reference": ref, "current": now, "missing_rate_difference": missing,
                  "standardized_mean_shift": None, "total_variation": None,
                  "constant_value_change": None, "unseen_categories": [], "status": "ok"}
        if not ref["available"]:
            result["status"] = "insufficient_reference_data"
        elif not now["available"]:
            result["status"] = "insufficient_current_data"
        elif ref["kind"] == "numeric":
            if ref["minimum"] == ref["maximum"]:
                changed = now["minimum"] != ref["minimum"] or now["maximum"] != ref["minimum"]
                result["constant_value_change"] = changed
                if changed:
                    flags.append("constant_value_change")
            else:
                shift = (now["mean"] - ref["mean"]) / ref["std"]
                if not np.isfinite(shift):
                    raise ValueError(f"Non-finite mean shift: {name}")
                result["standardized_mean_shift"] = shift
                if abs(shift) > THRESHOLDS["standardized_mean_shift"]:
                    flags.append("standardized_mean_shift")
        else:
            previous, present = ref["probabilities"], now["probabilities"]
            variation = 0.5 * sum(abs(previous.get(key, 0) - present.get(key, 0))
                                  for key in previous.keys() | present.keys())
            result["total_variation"] = variation
            result["unseen_categories"] = sorted(present.keys() - previous.keys())
            if variation > THRESHOLDS["total_variation"]:
                flags.append("total_variation")
        result["flags"] = flags
        results[name] = result
    return results


def analyze(model_path, reference, frame, input_identity):
    if reference.get("profile_version") != VERSION:
        raise ValueError("Unsupported reference profile version")
    if reference.get("model_sha256") != sha256(model_path):
        raise ValueError("Reference model hash mismatch")
    bundle, transformer = load_model(model_path)
    valid = validate_claims(frame)
    current = feature_statistics(transformer.transform(valid))
    if set(reference["features"]) != set(current):
        raise ValueError("Reference features do not match model")
    results = compare_features(reference["features"], current)
    flagged = [name for name, value in results.items() if value["flags"]]
    insufficient = len(valid) < 100
    return {"report_version": VERSION, "model_sha256": reference["model_sha256"],
            "model_threshold": bundle["threshold"], "input": input_identity,
            "reference_source": reference["source"], "reference_sample_size": reference["sample_size"],
            "reference_training_input_sha256": reference["training_input_sha256"],
            "sample_size": len(valid), "thresholds": THRESHOLDS,
            "status": "insufficient_data" if insufficient else "complete",
            "drift_detected": None if insufficient else bool(flagged),
            "flagged_features": flagged, "features": results,
            "interpretation": "Distribution flags are heuristic; they do not establish accuracy loss. "
                              "No retraining or threshold change is performed."}


def write_report(output_dir, report):
    # Serialize before creating outputs, rejecting undefined JSON numbers.
    encoded = json.dumps(report, indent=2, allow_nan=False)
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Report output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    rows = "".join("<tr><td>" + escape(name) + "</td><td>" +
                   escape(", ".join(value["flags"]) or "none") + "</td><td>" +
                   escape(value["status"]) + "</td></tr>"
                   for name, value in report["features"].items())
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Local data drift report</title><style>'
            'body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;'
            'color:#163047;background:#f6f8fa}table{border-collapse:collapse;width:100%;'
            'background:white}td,th{padding:10px;border:1px solid #ccd5dd;text-align:left}'
            'pre{background:white;padding:20px;white-space:pre-wrap;overflow-wrap:anywhere}'
            '</style><h1>Local data drift report</h1><p>Status: <strong>' +
            escape(report["status"]) + '</strong> | Input rows: ' + str(report["sample_size"]) +
            ' | Reference rows: ' + str(report["reference_sample_size"]) +
            '</p><p>Overall drift indication: <strong>' +
            ("insufficient data" if report["drift_detected"] is None else
             "flagged" if report["drift_detected"] else "no defined indicators flagged") +
            '</strong></p><p>' +
            escape(report["interpretation"]) + '</p><table><thead><tr><th>Feature</th>'
            '<th>Indicators</th><th>Availability</th></tr></thead><tbody>' + rows +
            '</tbody></table><h2>Identities, thresholds and feature statistics</h2><pre>' +
            escape(encoded) + '</pre></html>')
    (directory / "drift-report.json").write_text(encoded + "\n", encoding="utf-8")
    (directory / "drift-report.html").write_text(html, encoding="utf-8")
