"""Serializable shared preprocessing for training and future API serving."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import CATEGORICAL, DATES, DEMOGRAPHIC, NUMERIC, RAW_NUMERIC


class ClaimFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, demographics=False):
        self.demographics = demographics

    def fit(self, X, y=None):
        self.amount_cap_ = pd.to_numeric(X["claim_amount"]).quantile(0.995)
        self.input_columns_ = RAW_NUMERIC + CATEGORICAL + DATES
        if self.demographics:
            self.input_columns_ = self.input_columns_ + ["age"] + DEMOGRAPHIC
        return self

    def transform(self, X):
        numeric = RAW_NUMERIC + (["age"] if self.demographics else [])
        categorical = CATEGORICAL + (DEMOGRAPHIC if self.demographics else [])
        frame = X[numeric + categorical + DATES].copy()
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        for column in DATES:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        frame["claim_amount"] = frame.claim_amount.clip(upper=self.amount_cap_)
        delay = (frame.report_date - frame.incident_date).dt.days
        frame["invalid_report_delay"] = delay.lt(0).astype(int)
        frame["report_delay_days"] = delay.mask(delay < 0)
        frame["days_since_policy_start"] = (frame.incident_date - frame.policy_start_date).dt.days
        frame["customer_tenure_months"] = (
            frame.policy_start_date - frame.customer_since).dt.days / 30.4375
        for denominator, name in (("sum_insured", "claim_to_sum_insured_ratio"),
                                   ("annual_premium", "claim_to_premium_ratio")):
            frame[name] = frame.claim_amount / frame[denominator].where(frame[denominator] > 0)
        for column in categorical:
            frame[column] = frame[column].map(
                lambda value: str(value).strip().lower() if pd.notna(value) else np.nan)
        output = NUMERIC + (["age"] if self.demographics else []) + categorical
        frame = frame[output]
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        return frame
