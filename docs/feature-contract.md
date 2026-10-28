# Feature and evaluation contract

## Prediction point and target

Score a claim when reported, for manual review only. Target: observed `is_fraud`.
Never train on latent truth, investigation outcomes, settlement data, or labels as inputs.
Every result must say **synthetic data**, seed `20261028`.

## Feature sets

Primary numeric inputs: vehicle_age_years, vehicle_value, sum_insured, annual_premium,
claim_amount, incident_hour, police_report_filed, witness_count, is_authorized_workshop.
Primary categorical inputs: incident_type, coverage_type, vehicle_type, sales_channel,
payment_frequency. Derived inputs: report_delay_days, days_since_policy_start,
claim_to_sum_insured_ratio, claim_to_premium_ratio, customer_tenure_months.
Historical inputs: prior_claims_12m, prior_claims_total, workshop_claim_count_90d.
Demographic experiment adds age, occupation_group, province, incident_province.

IDs are join/audit keys only. Exclude workshop name/city, raw dates after derivation,
customer_tenure_months_snapshot, n_policies_total (snapshot information), split,
claim_status (including normalized variants), payout_amount, siu_investigation_flag,
days_to_settlement, true_fraud, detected, fraud_probability, and ring_era.
Use an explicit allowlist rather than dropping only known leakage columns.

## Temporal semantics

Preserve incident-date splits: train through 2023-12-31, validation through 2024-06-30,
test from 2024-07-01. Train eligibility also requires report_date before 2024-01-01;
validation eligibility requires report_date before 2024-07-01. Report excluded counts.
Negative report delays become missing with an indicator; use incident_date as the
conservative availability date for these invalid records.

Rebuild historical counts after deduplication using availability date
max(incident_date, report_date), strictly earlier than the current availability date.
Window bounds are inclusive at t-365 days / t-90 days and exclusive at t.
Same-day records do not count each other. History may cross splits but cannot use
future records or labels. Never reuse generator counts: they use incident order,
include same-day predecessors, and can count claims not yet reported.
Customer tenure is measured at policy start, not the final snapshot.

No label-observation timestamps exist. Assume labels for eligible training records
are available at cutoff for this synthetic benchmark; explicitly document that real
deployment requires label-maturity dates. No realistic delayed-label claim is made.

## Preprocessing and selection

Normalize categories with strip/lower; fit all learned transforms only on train.
Use missing indicators, train-fitted imputation, and train-fitted upper amount clipping
at the 99.5th percentile. Invalid denominators yield missing ratios, never infinity.
API and training share these transforms; unseen categories must be supported.
Deduplicate identical claim IDs keeping the first; conflicting duplicates fail audit.

Choose threshold on validation for highest recall with at most 5% flagged; allow a
threshold above the maximum score when ties prevent a feasible nonzero selection.
Among equal-recall thresholds choose higher precision, then higher threshold.
Report 2% and 10% sensitivity scenarios. This is a validation constraint, not a
guarantee of production or test review volume. Freeze before inspecting test metrics.
Compare models by validation recall at capacity, then average precision, then prefer
no demographics and logistic regression. No minimum model performance is promised.

## Serving boundary

Caller supplies the documented historical counts computed with these semantics,
alongside claim/policy attributes and dates for derived features. No local history
database or automatic snapshot updates in the MVP. Validate nonnegative integer
counts and prior_claims_12m <= prior_claims_total. Scores are uncalibrated model
scores unless calibration is explicitly validated; reason codes are rule indicators.
