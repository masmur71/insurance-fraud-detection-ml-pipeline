# Local data drift reports

Compare the inputs of a new claim batch with the data used for training. A flag
means the input mix has changed enough to investigate. It does not establish that
fraud detection has become less accurate, and it never retrains the model or
changes the frozen decision cutoff `0.08607093703233727`.

## Try the demonstration

Run from the repository root with the environment activated and an existing
frozen model at `models/frozen-model/model.joblib`:

```powershell
python scripts/demo_drift.py
Start-Process reports/drift-demo/controlled-shift/drift-report.html
```

Use a new `--output-dir` for each run. The demonstration uses eligible training
claims and controlled changes with seed `20261028`. It checks that identical
inputs have zero differences, then increases vehicle values, introduces a new
incident category, and removes some vehicle ages. The local `verification.json`
records the assertions. No consumed holdout rows are loaded or scored.

## Analyze your own local batch

```powershell
python scripts/build_drift_reference.py --model models/frozen-model/model.joblib --output reports/drift-reference.json
python scripts/report_drift.py --model models/frozen-model/model.joblib --reference reports/drift-reference.json --input reports/new-claims.parquet --output-dir reports/new-claims-drift
Start-Process reports/new-claims-drift/drift-report.html
```

Supply `new-claims.parquet` yourself; it must contain exactly the fields in the
[API feature contract](feature-contract.md), without labels or identifiers. Each
row follows the same validation rules as a prediction request. Parquet nulls are
treated as missing, midnight date timestamps as calendar dates, and integral
floating representations of integer fields as integers. Fractional counts,
boolean counts, numeric strings, timestamps with a time or timezone, invalid
claims, and empty inputs are rejected. Errors identify the zero-based row and
field without echoing claim values. No report is written for invalid input.

The reference builder filters incidents before January 1, 2024 at the Parquet
reader, then keeps training claims whose report is also available before that
date. It applies the model's fitted feature transformer without fitting again.
This retains its existing amount cap, derived ratios, dates, and category
normalization. Analysis happens before imputation, scaling, and one-hot encoding.

The reference contains aggregate feature statistics, a format version, model
SHA-256, training input digest and sample size. It contains no raw claim rows.
The report records the input file digest, model identity, reference identity,
applied thresholds, statistics and flag summaries. Reference and model hashes
must match. Existing reference files and nonempty report folders are protected
against accidental overwrites. Generated reports and fixtures remain Git-ignored.

## Read the indicators

| Indicator | Calculation | Flag when |
| --- | --- | --- |
| Missingness | Batch missing rate minus training missing rate | Absolute difference > 0.05 |
| Numeric shift | Difference in nonmissing means divided by training population standard deviation | Absolute shift > 0.5 |
| Category shift | Half the sum of absolute frequency differences over the union of nonmissing categories | Total variation > 0.1 |
| Constant value change | Any available numeric value differs from the constant training value | True |

Categories unseen during training are included in total variation and listed
explicitly. Missing values are excluded from category frequencies because
missingness has its own indicator. A feature without available training values
has `insufficient_reference_data`; one without available batch values has
`insufficient_current_data`. Undefined metrics are JSON `null`, never NaN or
Infinity. Missingness can still be flagged in either case.

Below 100 input rows the report status is `insufficient_data`. Statistics and
feature indicators remain visible, but `drift_detected` is `null`. At 100 or more
rows, `drift_detected` reports whether any defined indicator was flagged. A false
result does not prove stability for features with unavailable comparison data.

These initial limits are heuristics, not significance tests. The HTML is a
standalone local file with no external scripts or assets. Aggregate category
names remain visible; handle reports according to your local data policy.
There is no drift API endpoint, API payload collection, external notification,
cloud deployment, or automated model replacement.
