"""Request contract at claim report time; histories are supplied by the caller."""
from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Amount = Annotated[float, Field(strict=True, ge=0, le=1e15, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0, le=2_147_483_647)]
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle_age_years: Annotated[float, Field(strict=True, ge=0, le=150, allow_inf_nan=False)] | None
    vehicle_value: Amount | None
    sum_insured: Amount | None
    annual_premium: Amount | None
    claim_amount: Amount | None
    incident_hour: Annotated[int, Field(strict=True, ge=0, le=23)] | None
    police_report_filed: Annotated[int, Field(strict=True, ge=0, le=1)] | None
    witness_count: Count | None
    is_authorized_workshop: Annotated[int, Field(strict=True, ge=0, le=1)] | None
    prior_claims_12m: Count
    prior_claims_total: Count
    workshop_claim_count_90d: Count
    incident_type: Category | None
    coverage_type: Category | None
    vehicle_type: Category | None
    sales_channel: Category | None
    payment_frequency: Category | None
    age: Annotated[int, Field(strict=True, ge=0, le=120)] | None
    occupation_group: Category | None
    province: Category | None
    incident_province: Category | None
    incident_date: date
    report_date: date
    policy_start_date: date
    customer_since: date

    @field_validator("incident_date", "report_date", "policy_start_date", "customer_since", mode="before")
    @classmethod
    def calendar_date_only(cls, value):
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("Expected a calendar date in YYYY-MM-DD format") from exc
            if parsed.isoformat() != value:
                raise ValueError("Expected a calendar date in YYYY-MM-DD format")
        elif type(value) is date:
            parsed = value
        else:
            raise ValueError("Expected a calendar date in YYYY-MM-DD format")
        if not date(1900, 1, 1) <= parsed <= date(2100, 12, 31):
            raise ValueError("Date must be between 1900 and 2100")
        return parsed

    @model_validator(mode="after")
    def consistent_history(self):
        if self.prior_claims_12m > self.prior_claims_total:
            raise ValueError("prior_claims_12m cannot exceed prior_claims_total")
        return self


class PredictionResponse(BaseModel):
    score: float
    threshold: float
    review_required: bool
    reason_codes: list[str]
    model_version: str
    score_kind: Literal["uncalibrated_model_score"] = "uncalibrated_model_score"
    dataset: Literal["synthetic data"] = "synthetic data"
    seed: int = 20261028


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str


class BatchClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: Annotated[list[ClaimRequest], Field(min_length=1, max_length=100,
                                               description="Between 1 and 100 claims, in order")]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
