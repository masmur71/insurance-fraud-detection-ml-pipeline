# Synthetic Vehicle Insurance Fraud Dataset

**Language:** English | [Bahasa Indonesia](README.id.md)

This folder contains the **generator notebook and documentation** for a synthetic Indonesian vehicle insurance fraud dataset. The generated dataset is not published with the repository: run the generator to create it locally with seed `20261028`.

> **Important:** all generated data is **synthetic**. A model can only recover the rules planted by the generator, not real-world fraud behaviour. Every metric you quote must be labelled **"synthetic data"**.

## Contents

- [Repository layout](#repository-layout)
- [Generated artifacts](#generated-artifacts)
- [Configuration](#configuration)
- [Regenerating the dataset](#regenerating-the-dataset)
- [Table schemas](#table-schemas)
- [Planted fraud mechanism](#planted-fraud-mechanism)
- [Leakage trap](#leakage-trap)
- [Loading the data](#loading-the-data)
- [EDA notebook](#eda-notebook)
- [Limitations and ethics](#limitations-and-ethics)
- [Pre-modeling checklist](#pre-modeling-checklist)
- [Reproducibility](#reproducibility)
- [Documentation maintenance](#documentation-maintenance)

## Repository layout

Only the notebooks and the two README files are intended to be published:

```text
data/
├── README.md                 English documentation (default)
├── README.id.md              Indonesian documentation
└── notebooks/
    ├── vehicle_insurance_fraud_synth.ipynb   generator + inline EDA
    └── eda_vehicle_insurance_fraud.ipynb     read-only EDA
```

The Parquet files and generated JSON files are **not part of the published repository**. They are written to `data/` when the generator runs and should stay local.

## Generated artifacts

Running the generator produces the following files next to this README:

| File | Purpose |
|---|---|
| `customers.parquet` | one row per customer |
| `policies.parquet` | one row per policy |
| `claims.parquet` | claim-level table; duplicate claim IDs are intentionally injected |
| `workshops.parquet` | workshop reference table |
| `claims_truth.parquet` | latent labels, detection flag, fraud probability, and ring era; analysis only |
| `claims_features.parquet` | de-duplicated analysis table with policy/customer attributes and derived features |
| `generator_config.json` | locked generator parameters and generation metadata |
| `manifest.json` | seed, library versions, row counts, file sizes, and feature metadata |

## Configuration

| Setting | Value |
|---|---|
| Seed | `20261028` |
| Configured claim period | 2021-01-01 to 2025-12-31 |
| Split policy | train 2021-2023, validation 2024 H1, test 2024 H2-2025 |
| Label noise | 15% of fraudulent claims are labelled 0 |
| Fraud mechanism | logistic model with noise, MNAR missingness, dirty records, and 2025 drift |

All values above come from the locked generator configuration. Observed row counts and fraud rates are recorded in `manifest.json` after generation.

## Regenerating the dataset

1. Open [`notebooks/vehicle_insurance_fraud_synth.ipynb`](notebooks/vehicle_insurance_fraud_synth.ipynb).
2. Run **Run all**. The notebook writes to the same `data/` folder (on Colab: `/content/data`).
3. It creates the raw tables, truth file, `claims_features.parquet`, and `generator_config.json`, then refreshes `manifest.json`.

Run the generator before opening the EDA notebook. The EDA needs the generated Parquet files.

The generator and an inline EDA still live in the same notebook. For day-to-day exploration, use [`eda_vehicle_insurance_fraud.ipynb`](notebooks/eda_vehicle_insurance_fraud.ipynb), which is **read-only**: it writes or modifies nothing.

> Re-running the generator **overwrites** the generated files. Do it only when you intend to regenerate the dataset.

## Table schemas

### `customers.parquet` — one row per customer

| Column | Description |
|---|---|
| `customer_id` | primary key, format `C000001` |
| `age` | age 18-75 |
| `occupation_group` | employee, self_employed, entrepreneur, student, professional, retired, civil_servant, other |
| `province` | synthetic province labels |
| `customer_since` | date the customer joined |
| `customer_tenure_months` | tenure at snapshot |
| `n_policies_total` | 1 or 2 policies |

### `policies.parquet` — one row per policy

| Column | Description |
|---|---|
| `policy_id` | primary key, format `P0000001` |
| `customer_id` | foreign key to `customers` |
| `policy_start_date`, `policy_end_date` | one-year policy term |
| `coverage_type` | `Comprehensive` or `TLO` |
| `vehicle_type` | motorcycle, private_car, commercial_vehicle, truck |
| `vehicle_age_years` | vehicle age |
| `vehicle_value` | vehicle value (IDR) |
| `sum_insured` | sum insured (IDR) |
| `annual_premium` | annual premium (IDR) |
| `sales_channel` | agent, online, broker |
| `payment_frequency` | annual, semiannual, quarterly, monthly |

### `claims.parquet` — one row per claim

| Column | Description |
|---|---|
| `claim_id` | claim ID; duplicates are intentionally injected, so it is not unique |
| `policy_id`, `customer_id` | foreign keys |
| `incident_date`, `report_date` | incident and report dates |
| `incident_type` | collision, theft, flood, vandalism, fire |
| `incident_hour` | hour 0-23; contains missing values (MNAR) |
| `incident_province` | province of the incident |
| `claim_amount` | claim amount (IDR); contains extreme outliers |
| `police_report_filed` | 0/1; contains missing values (MNAR) |
| `witness_count` | number of witnesses; contains missing values (MNAR) |
| `workshop_id` | repair workshop |
| `is_authorized_workshop` | 0/1 |
| `prior_claims_12m` | claims in the prior 12 months |
| `prior_claims_total` | total prior claims |
| `is_fraud` | **observed label** used for modeling |
| `claim_status` | approved, rejected, under_review — **leakage** |
| `payout_amount` | payout amount — **leakage** |
| `siu_investigation_flag` | SIU investigation flag — **leakage** |
| `days_to_settlement` | time to settlement — **leakage** |

### `workshops.parquet` — one row per workshop

| Column | Description |
|---|---|
| `workshop_id` | primary key, format `W0001` |
| `workshop_name`, `province`, `city` | location attributes |
| `is_authorized_workshop` | 0/1 |
| `opened_date` | opening date; a small set of workshops opens in 2025 for the drift story |

### `claims_truth.parquet` — analysis only; never use as model features

| Column | Description |
|---|---|
| `claim_id` | primary key |
| `true_fraud` | latent label before label noise |
| `detected` | whether fraud was detected |
| `fraud_probability` | fraud probability from the generator's logit model |
| `ring_era` | `none`, `old`, or `new` (workshop ring) |

### `claims_features.parquet` — analysis table

Combines `claims` (after de-duplication), policy attributes, and customer attributes. It includes raw columns, normalized category versions (`*_clean`), derived features, `split`, and the **leakage columns**.

Key derived features:

| Column | Formula / meaning |
|---|---|
| `report_delay_days` | `report_date - incident_date` |
| `days_since_policy_start` | `incident_date - policy_start_date` |
| `claim_to_sum_insured_ratio` | `claim_amount / sum_insured` |
| `claim_to_premium_ratio` | `claim_amount / annual_premium` |
| `workshop_claim_count_90d` | claims at the same workshop in the prior 90 days |
| `customer_tenure_months` | customer tenure when the policy started |
| `split` | train / validation / test based on incident date |

## Planted fraud mechanism

Fraud is generated by a logit model with noise, then passed through 15% label noise. The values below are odds multipliers: the generator adds their natural logarithms to the logit. They are not logit coefficients.

| Trigger | Coefficient |
|---|---:|
| Claim within the first 60 days of the policy | 3.0 |
| Report delay > 14 days | 2.0 |
| Theft without a police report | 4.0 |
| Claim / sum insured ratio > 0.8 | 2.5 |
| Prior claims in 12 months >= 3 | 2.0 |
| Incident between 00:00 and 04:00 | 1.5 |
| Non-authorized workshop | 1.5 |
| Overvaluation | 2.0 |
| Online channel + new customer | 1.8 |
| Early claim x large claim interaction | 1.5 |

Additional effects:

- **Workshop rings:** a legacy ring and a new ring introduced in 2025. The legacy ring is strong before 2025 and weaker in 2025; the new ring is active only in 2025.
- **2025 drift:** the online-channel effect is multiplied by 1.55 and ring composition changes.
- **MNAR missingness:** `witness_count`, `police_report_filed`, and `incident_hour` are more likely to be missing for fraudulent claims.
- **Dirty data:** category capitalization variants (3%), negative report delays (0.3%), claim-amount outliers (0.5%), and duplicates (0.2%).

## Leakage trap

These four columns are only known after a claim has been processed:

| Column | Why it leaks |
|---|---|
| `claim_status` | Decision outcome; detected fraud is almost always rejected/under_review |
| `payout_amount` | Zero for rejected fraud claims; payout is a consequence of the decision |
| `siu_investigation_flag` | Output of investigation triage, not an input when the claim is reported |
| `days_to_settlement` | Only known after the process is complete |

If included, the model will look almost perfect but will be useless for new claims. `claims_features.parquet` intentionally keeps these columns so the demonstration is reproducible; **drop them before modeling**.

## Loading the data

Run the generator first, then load the generated files:

```python
from pathlib import Path
import pandas as pd

DATA = Path("data")

claims = pd.read_parquet(DATA / "claims.parquet")
features = pd.read_parquet(DATA / "claims_features.parquet")

LEAKAGE = ["claim_status", "payout_amount", "siu_investigation_flag", "days_to_settlement"]
TARGET = "is_fraud"

model_frame = (
    features
    .drop_duplicates("claim_id")
    .drop(columns=LEAKAGE)
)

train = model_frame[model_frame["split"] == "train"]
valid = model_frame[model_frame["split"] == "validation"]
test = model_frame[model_frame["split"] == "test"]

X_train = train.drop(columns=[TARGET, "claim_id", "split"])
y_train = train[TARGET]
```

`claims_truth.parquet` is for label-noise diagnostics and error analysis only. Never join it into model features.

## EDA notebook

[`notebooks/eda_vehicle_insurance_fraud.ipynb`](notebooks/eda_vehicle_insurance_fraud.ipynb) contains 15 sections:

1. Setup and data-folder resolution
2. File inventory and schema
3. Relational integrity
4. Target and label noise
5. MNAR missingness
6. Deliberate dirty data
7. Univariate signal (lift + Wilson CI)
8. Categories and workshops
9. Leakage trap
10. 2025 time drift
11. Fairness check (`age`, `province`)
12. Split sanity check
13. Non-leakage feature correlation
14. Summary and checklist
15. Limitations

The notebook **ships without saved outputs**. After generating the dataset, run **Run all** to produce the tables and plots locally. It writes no files.

## Limitations and ethics

- Synthetic data; not real fraud behaviour.
- Claim frequency and exposure are not modelled (~1.5 claims per policy is only an assumption).
- Premiums are not actuarial rates; IDR amounts only look plausible.
- There is no claim narrative, so text analysis is out of scope.
- `age` and `province` do not generate fraud. If used, explicitly discuss the profiling risk.
- Fraud detection is assumed to be a constant 85%; real investigations are selective and biased.

## Pre-modeling checklist

1. Drop the four leakage columns and all truth columns.
2. De-duplicate `claim_id`.
3. Normalize categories: `strip()` + `lower()`.
4. Handle negative report delays and claim-amount outliers explicitly.
5. Add missing-value indicators for MNAR columns.
6. Split by time; never split at random.
7. Choose the threshold from a cost matrix, not F1 alone.
8. Report PR-AUC, ROC-AUC, calibration, and a confusion matrix per split.
9. Decide whether to use `age` and `province`; document the rationale.
10. Label every metric **"synthetic data"** and include seed `20261028`.

## Reproducibility

- `generator_config.json` stores all parameters, the calibrated intercept, row counts, and fraud rates.
- `manifest.json` stores the seed, Python/NumPy/pandas versions, the size of every generated Parquet file, and `claims_features.parquet` metadata (`feature_rows`, `feature_columns`, `features_exclude_truth`).
- The generator writes an initial manifest after the raw tables, then refreshes it after `claims_features.parquet` is created.
- With the same seed, outputs are reproducible; only the `generated_at_utc` timestamp changes.

## Documentation maintenance

`README.md` and [`README.id.md`](README.id.md) are maintained as a pair. When the schema, repository layout, or generator changes, update both files together.
