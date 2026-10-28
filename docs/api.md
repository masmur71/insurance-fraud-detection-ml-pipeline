# API and container verification

Original single-claim API and container verification: 2026-10-28.
See [batch prediction](batch-prediction.md) for the batch endpoint.

## Verification criteria

FastAPI `/predict` and `/health`, Pydantic input validation, behavioral HTTP tests,
and a Docker image using Python 3.11. Verification criteria:
all tests pass on Python 3.11, the container is healthy, and HTTP scores match the
frozen pipeline. The consumed holdout was not evaluated again.

## Implementation

The lifespan loads one trusted joblib bundle. Startup checks the fitted input contract,
synthetic-data metadata, finite threshold, and exact scikit-learn version. The artifact
SHA-256 identifies every response. `/predict` uses the same ClaimFeatures and fitted
preprocessing pipeline as training. No API-specific clipping, imputation, or retraining occurs.
Health becomes available only after successful model loading.

All feature names are required, with null allowed for measurements/categories. Dates
and nonnegative integer histories are mandatory. Validation rejects fractional counts,
booleans used as counts, inconsistent history counts, out-of-range inputs, non-finite
numbers, invalid dates, empty categories, and extra outcome/truth fields. Validation
errors return locations/messages/types without echoing potentially non-JSON values.

Reason codes describe input rules only, including invalid reporting delay. They can be
empty or present regardless of the review decision; they are not feature attribution.
The service includes no history database. Caller-supplied histories must respect the
availability and same-day exclusion semantics in the feature contract.

Docker builds from an allowlisted context and embeds the selected artifact. It runs
as UID/GID 10001, limits numeric threads, disables access logs, and checks `/health`.
Python 3.11, pinned estimator/numeric libraries, and pinned API validation libraries are
used. This does not promise bitwise image reproducibility: the base image and some
transitive dependencies remain unpinned.

## Verification

- Local Windows Python 3.13: 16 tests and 18 invalid-input subtests pass.
- Linux container Python 3.11.18: the same complete suite passes.
- The built container reaches Docker health status `healthy`.
- Real HTTP smoke checks compare three inputs against the saved full-data model:
  normal data, missing values/unseen categories, and invalid delay/zero denominator.
  Scores agree with relative and absolute tolerance 1e-12; thresholds and decisions agree.
- Health version matches the artifact hash; extra truth fields receive HTTP 422.
- Ruff checks pass for the new API, schema, API tests, and HTTP smoke script.

Versions exercised: scikit-learn 1.6.1, NumPy 2.2.0, pandas 2.2.3, FastAPI 0.162.0,
Starlette 1.10.0, Pydantic 2.14.1, uvicorn 0.54.0 in the container.
The older logistic baseline emits SciPy optimizer deprecation warnings in Python 3.11
tests, without failures. Windows sandbox networking prevents TestClient's event loop
from opening its internal socket; running those tests outside the sandbox succeeds.

## Limitations

All model results concern **synthetic data**, seed `20261028`. The model is uncalibrated,
uses demographic features, and assumes label availability without maturity timestamps.
The test review rate is 7.43%, not a guaranteed 5% operational queue.
Python 3.11 inference and behavioral tests are verified; full-data training was run
on Python 3.13 and was not repeated after holdout evaluation.

See the [operations guide](operations.md) for CI/CD, local deployment, and monitoring.
The local service has no authentication or rate limiting.
