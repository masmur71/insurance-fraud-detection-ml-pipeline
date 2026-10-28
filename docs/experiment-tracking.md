# Experiment tracking verification

Local tracking and artifact persistence verified on 2026-10-29.

- Frozen recipe and threshold `0.08607093703233727` preserved: 78,352 eligible
  training rows and 14,600 validation rows; no consumed holdout rows loaded or scored.
- Current verification run: `2491f9dbff8d4741a3a91f9f752e469c`, experiment
  `frozen-model-reproduction`, loopback port 5059.
- Verified 17 parameters, 11 metrics, provenance tags, model and build-report
  artifacts, UI HTTP 200, and identical run metadata after server restart.
- Downloaded model/report SHA-256 hashes matched their local sources. Serving the
  downloaded model passed health, individual/batch parity, and invalid-input checks.
- Local evidence: `models/tracking-integration/tracking-verification.json`.
  Persistent stores: `.mlflow/tracking.db` and `.mlflow/artifacts`, excluded from Git.
- A separate build without tracking succeeded at `models/frozen-model`.
- MLflow remains optional and absent from the serving image. Tracking failures
  preserve completed local artifacts and return a failure status.

Restart `./scripts/start_mlflow.ps1 -Port 5059` to inspect the retained run. Earlier
runs remain in the same store; changing the default experiment name does not
rename or modify their historical records. See [operations](operations.md#local-mlflow)
for reproduction commands and [local drift reports](data-drift.md) for aggregate
input comparisons independent of tracking.
