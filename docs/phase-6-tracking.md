# Phase 6 verification

Verified locally on 2026-10-29; phases 7-8 remain unimplemented.

- Frozen recipe and threshold `0.08607093703233727` preserved; training uses
  78,352 eligible rows and validation uses 14,600. No holdout scoring or tuning.
- Successful MLflow run: `effb0e82e6c248b2b7681e504e93aae7`, experiment
  `frozen-phase-3-reproduction`. UI served on loopback port 5058.
- Integration verified 17 parameters, 11 metrics, source/dirty tags, both
  artifacts, UI HTTP 200, metadata equality after server restart, and downloaded
  model/report SHA-256 equality. The downloaded model passed the existing HTTP
  smoke test (health, three prediction parity cases, invalid-input rejection).
- Local evidence: `models/phase-6-verified-2/tracking-verification.json`; persistent
  stores: `.mlflow/tracking.db` and `.mlflow/artifacts` (all excluded from Git).
- Real build without tracking succeeded at `models/phase-6-untracked`.
- CI-equivalent Python 3.11 container checks: 22 tests and 18 subtests passed,
  fatal-error lint passed, and MLflow was absent. Existing Dockerfile built
  successfully and its HTTP smoke test passed. GitHub-hosted CI was not dispatched;
  no commit, push, or registry publication was performed.
- Tests verify opt-in behavior, server-URI validation, reader-level holdout
  exclusion, and preservation of completed artifacts on tracking failure.

Restart with `./scripts/start_mlflow.ps1 -Port 5058` to inspect the retained run.
See [operations](operations.md#local-mlflow-phase-6) for reproduction commands.
