# Deployment and monitoring

## Batch prediction

```powershell
$claim = Get-Content examples/claim.json -Raw | ConvertFrom-Json
$body = @{ claims = @($claim, $claim) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/batch -ContentType 'application/json' -Body $body
python scripts/benchmark_batch.py --url http://127.0.0.1:8000 --size 100 --repeats 10
```

The synchronous endpoint accepts 1-100 claims and returns predictions in order.
Any invalid item rejects the whole request with HTTP 422 and an indexed error;
there are no partial predictions. OpenAPI documents the limits. Batch requests
increment the HTTP counter once and successful prediction counters per claim.
The benchmark writes `reports/batch-benchmark.json`, including model hash, raw
round times, median latency, and aggregate throughput. It uses synthetic example
inputs and performs no holdout evaluation. Run both modes against the same server
and artifact. See [batch behavior and measured results](phase-7-batch.md).

## Release path

The GitHub Actions workflow tests Python 3.11, generates synthetic raw data, audits
it, builds the fixed phase 3 recipe, builds the Docker image, and checks HTTP/pipeline
parity. Pull requests run checks without publication. Successful `main` runs publish
the tested image to `ghcr.io/masmur71/insurance-fraud-detection-ml-pipeline` with both
the full commit SHA and `latest`. No cloud hosting or paid service is required.

GitHub's workflow token needs `packages: write`; no personal access token is stored
in this repository. This follows the [official GHCR workflow documentation](https://docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions).
GitHub may create a package as private even for a public repository. If anonymous
pulls fail, the package owner can make the package public in GitHub package settings,
or consumers can authenticate to GHCR with an appropriately scoped account token.
Do not paste tokens into files or commit history.

## Local deployment and rollback

```powershell
docker compose up --build -d
python scripts/wait_api.py
python scripts/smoke_api.py
docker compose logs --tail 30 api
docker compose down
```

To deploy a registry image instead of rebuilding locally:

```powershell
$env:API_IMAGE = 'ghcr.io/masmur71/insurance-fraud-detection-ml-pipeline:latest'
docker compose pull
docker compose up -d --no-build
python scripts/wait_api.py
```

Use a verified full commit SHA instead of `latest` for a stable deployment. Roll back
by setting `API_IMAGE` to a previous tested SHA, pulling, and restarting Compose.
The image embeds its model. Keep the image digest, source commit, and `/health`
model version together when recording a deployment. `smoke_api.py` additionally
requires a local copy of the exact model inside the image; it deliberately fails
when the local artifact hash differs. A rebuilt CI model can have a different
artifact hash from the original Windows benchmark, even with equivalent predictions.

The Compose service binds only to loopback, uses an unprivileged account, a read-only
filesystem, no Linux capabilities, and a temporary `/tmp`. It restarts unless stopped.
It is a local demo, without public ingress, TLS termination, or authentication.

## Observability

- `/metrics`: Prometheus text counters for requests by bounded route/status,
  cumulative handler duration, successful predictions, and review flags.
- `/monitoring`: process-lifetime prediction count, mean score, ten equal-width score
  bins, review fraction, and its difference from model validation review fraction.
- Prediction logs: model hash, score, and review flag. No raw claim payload, customer
  identifier, age, or province is logged. Use `docker compose logs` to inspect them.

`review_volume_alert` becomes true after at least 100 predictions when review exceeds
7.5%. This is an illustrative operating rule, not a statistically validated drift
test. It does not deliver external notifications. Investigate changes in input mix,
score distribution, data availability, and capacity; do not automatically retune.
The current request to `/metrics` is counted after its response snapshot is taken.
Durations cover handler execution through response creation, not network transfer.

Counters are thread-safe and bounded in memory, per process, and reset on restart.
Use one worker for this demo. There is no persistent metric store, alert delivery,
or distributed aggregation. Score and review-volume changes are proxies, not proof
of feature drift or worsening fraud-detection accuracy. Labels are needed for that.

## Failure handling

Missing/incompatible model: startup fails; health never becomes ready. Rebuild with
the verified serving requirements and a trusted model. Invalid requests: HTTP 422
with field locations/messages. Repeated 5xx: inspect container logs and roll back to
a known image. Never load an untrusted joblib file; loading it can execute code.

## Local MLflow (phase 6)

Use a separate environment for tracking when serving dependency isolation is required.
From the repository root, activate the environment and run:

```powershell
python -m pip install -e '.[tracking]'
./scripts/start_mlflow.ps1
```

Open http://127.0.0.1:5000. Metadata persists in `.mlflow/tracking.db` and
proxied artifacts in `.mlflow/artifacts`, both ignored by Git. Stop with Ctrl+C
and run the same command from the repository root to restart. Back up both together.
This follows the [MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

In another terminal with the same environment activated:

```powershell
python scripts/build_model.py --output-dir models/phase-6-demo --tracking-uri http://127.0.0.1:5000
```

The `frozen-phase-3-reproduction` experiment records configuration, threshold,
validation metrics including confusion counts, eligible row counts, raw input file
hashes, runtime versions, Git commit and dirty status, artifact hashes, and copies
of the model and report. These are reproductions, not fresh model-selection or
holdout evidence. The Parquet reader excludes incidents on or after July 1, 2024;
input file hashes identify the original files without evaluating holdout rows.
The frozen recipe and threshold remain unchanged. Existing output directories must
be empty. An explicitly requested tracking failure exits nonzero after preserving
the completed local artifacts. Omitting `--tracking-uri` does not import MLflow.

For an automated Windows integration check (use a fresh output path):

```powershell
python scripts/verify_tracking.py --output-dir models/phase-6-verification
```

This starts a loopback server on 5056, checks run completeness and the UI response,
restarts with the same SQLite/artifact stores, downloads both files and compares
SHA-256 hashes, then serves the downloaded model on 8056 and runs the HTTP smoke
test. It stops its child processes and writes `tracking-verification.json` locally.
The script requires these ports to be free. To inspect the retained run, restart
`./scripts/start_mlflow.ps1 -Port 5056`.

Serving still uses an explicit file via `MODEL_PATH`; downloading does not promote
or replace the existing serving model. No Model Registry is used. The Docker image
and CI install serving dependencies without the tracking extra.
