# Deployment and monitoring

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
