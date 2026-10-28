"""Raw-table assembly and strictly past-only, label-free history."""
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "vehicle_age_years", "vehicle_value", "sum_insured", "annual_premium", "claim_amount",
    "incident_hour", "police_report_filed", "witness_count", "is_authorized_workshop",
    "prior_claims_12m", "prior_claims_total", "workshop_claim_count_90d",
    "report_delay_days", "invalid_report_delay", "days_since_policy_start",
    "claim_to_sum_insured_ratio", "claim_to_premium_ratio", "customer_tenure_months",
]
CATEGORICAL = ["incident_type", "coverage_type", "vehicle_type", "sales_channel",
               "payment_frequency"]
DEMOGRAPHIC = ["occupation_group", "province", "incident_province"]
RAW_NUMERIC = NUMERIC[:12]
DATES = ["incident_date", "report_date", "policy_start_date", "customer_since"]


def past_count(frame, entity, days=None):
    """Count availability dates in [t-days, t), excluding all same-day peers."""
    result = pd.Series(0, index=frame.index, dtype="int64")
    for indices in frame.groupby(entity, sort=False).groups.values():
        dates = frame.loc[indices, "available_date"].to_numpy(dtype="datetime64[D]")
        ordered = np.sort(dates)
        right = np.searchsorted(ordered, dates, side="left")
        left = 0 if days is None else np.searchsorted(
            ordered, dates - np.timedelta64(days, "D"), side="left")
        result.loc[indices] = right - left
    return result


def assemble(data_dir):
    """Read raw tables only; analysis/truth tables are intentionally never opened."""
    data_dir = Path(data_dir)
    claims = pd.read_parquet(data_dir / "claims.parquet")
    if len(claims.drop_duplicates()) != claims.claim_id.nunique():
        raise ValueError("Conflicting claim duplicates")
    frame = claims.drop_duplicates("claim_id", keep="first").copy()
    # Discard generator histories and outcome fields before joining.
    columns = ["claim_id", "policy_id", "customer_id", "workshop_id", "is_fraud",
               "incident_date", "report_date", "incident_province", "incident_type",
               "incident_hour", "claim_amount", "police_report_filed", "witness_count",
               "is_authorized_workshop"]
    frame = frame[columns]
    policies = pd.read_parquet(data_dir / "policies.parquet")
    owners = policies.set_index("policy_id").customer_id
    if not frame.policy_id.map(owners).eq(frame.customer_id).all():
        raise ValueError("Missing policy or inconsistent policy owner")
    frame = frame.merge(policies.drop(columns="customer_id"), on="policy_id",
                        how="left", validate="many_to_one")
    customers = pd.read_parquet(data_dir / "customers.parquet")
    if not frame.customer_id.isin(customers.customer_id).all():
        raise ValueError("Missing customer")
    frame = frame.merge(customers[["customer_id", "customer_since", "age",
                                   "occupation_group", "province"]],
                        on="customer_id", how="left", validate="many_to_one")
    for column in DATES:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[DATES].isna().any().any():
        raise ValueError("Required dates cannot be missing")
    frame["available_date"] = frame[["incident_date", "report_date"]].max(axis=1)
    frame["prior_claims_total"] = past_count(frame, "customer_id")
    frame["prior_claims_12m"] = past_count(frame, "customer_id", 365)
    frame["workshop_claim_count_90d"] = past_count(frame, "workshop_id", 90)
    frame["split"] = np.select(
        [frame.incident_date < "2024-01-01", frame.incident_date < "2024-07-01"],
        ["train", "validation"], default="test")
    frame["eligible"] = True
    for split, cutoff in (("train", "2024-01-01"), ("validation", "2024-07-01")):
        frame.loc[frame.split.eq(split), "eligible"] = frame.loc[
            frame.split.eq(split), "available_date"] < cutoff
    if not frame.is_fraud.isin([0, 1]).all():
        raise ValueError("Target must be binary")
    return frame
