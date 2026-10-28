# Insurance Claim Review API

[![CI and container delivery](https://github.com/masmur71/insurance-fraud-detection-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/masmur71/insurance-fraud-detection-ml-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Data](https://img.shields.io/badge/Data-100%25%20synthetic-orange)

This project uses machine learning to help a reviewer decide which vehicle insurance claims to check first. It takes claim details, returns a score, and flags claims that need human review. It supports one claim at a time or a batch of up to 100 claims.

The project demonstrates the full workflow: prepare data, train a model, make predictions through an API, and run the service locally with Docker.

## About the data

**The dataset is entirely synthetic: it is generated for learning and experimentation, rather than collected from real customers or insurance claims.** For more details about the dataset, how it is generated, and its limitations, see [Synthetic Vehicle Insurance Fraud](https://github.com/masmur71/synthetic-vehicle-insurance-fraud).

Results on this dataset do not show how well the model would detect real fraud. A review flag is a prompt for a person to investigate, never an automatic reason to reject a claim. The score is not a confirmed probability of fraud.

## What you can try

- Submit a claim and receive a score and review recommendation.
- Submit 1-100 claims together and receive results in the same order.
- View basic service health and prediction summaries.
- Track model builds locally with the optional MLflow tools.
- Run automated checks and package the API as a Docker image.

## Run locally

You need **Python 3.11, Git, and Docker with Linux containers**. Use PowerShell and run these commands from the project folder.

```powershell
git clone https://github.com/masmur71/insurance-fraud-detection-ml-pipeline.git
cd insurance-fraud-detection-ml-pipeline
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-serving.txt -e ".[dev,data]"
```

Generate the synthetic data, check it, and build the model:

```powershell
.venv/Scripts/python.exe scripts/generate_data.py
.venv/Scripts/python.exe scripts/audit_data.py
.venv/Scripts/python.exe scripts/build_model.py
```

Start the API and try the demo:

```powershell
docker compose up --build -d
.venv/Scripts/python.exe scripts/wait_api.py
.venv/Scripts/python.exe scripts/demo.py
```

Open [the interactive API page](http://localhost:8000/docs) to explore requests and responses. Stop the service with `docker compose down`.

Generated data and models stay on your machine. The generator will not overwrite existing data, and the model build needs an empty output folder. If you already have the raw data, you can skip generation. Building uses the existing model recipe and decision cutoff; it does not repeat model selection or evaluate the reserved test data.

### Use the published image

To run the image published by a successful GitHub build, without training locally:

```powershell
$env:API_IMAGE = 'ghcr.io/masmur71/insurance-fraud-detection-ml-pipeline:latest'
docker compose pull
docker compose up -d --no-build
```

Then open [the API page](http://localhost:8000/docs). If the image requires sign-in, see the [operations guide](docs/operations.md). The service runs locally; this repository does not provide a public API server.

## Make a prediction

A complete example claim is included in [examples/claim.json](examples/claim.json).

```powershell
$payload = Get-Content examples/claim.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $payload
```

The main response fields are:

| Field | What it tells you |
| --- | --- |
| `score` | The model's score, between 0 and 1 |
| `threshold` | The fixed cutoff used to flag a claim |
| `review_required` | Whether the claim should be checked by a person |
| `reason_codes` | Simple input checks, such as a missing police report; these are not explanations of the model's score |
| `model_version` | Identifies the exact model file used |

See [an example response](examples/prediction.json). A rebuilt model file can have a different version identifier.

### Score several claims

```powershell
$claim = Get-Content examples/claim.json -Raw | ConvertFrom-Json
$body = @{ claims = @($claim, $claim) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/batch -ContentType 'application/json' -Body $body
```

A batch accepts **1-100 claims** and returns a `predictions` list in the same order. If any claim is invalid, the whole request is rejected and the error identifies the item to fix.

Use the example file as your starting point: all its input fields are required, although some values may be `null`. Dates use `YYYY-MM-DD`. For detailed input rules, see the [feature contract](docs/feature-contract.md).

### Available endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Score one claim |
| `POST /predict/batch` | Score 1-100 claims |
| `GET /health` | Check that the service and model are ready |
| `GET /monitoring` | View prediction and review-volume summaries |
| `GET /metrics` | Read counters for monitoring tools |
| `GET /docs` | Explore the API interactively |

## What the results mean

On the **synthetic validation data**, the selected model flagged about **5% of claims** for review and found about **30% of the claims labelled as fraud**. About **17% of its flagged claims** had a fraud label.

On the separate test data, the same fixed cutoff flagged **7.43% of claims**, above the intended 5% review workload. This illustrates a practical issue: the number of claims needing review can change even when the model stays the same.

These are historical experiment results, not new measurements or evidence of real-world performance. The reserved test set has already been used and must not be reused for tuning. For the full results and limitations, read the [model card](docs/model-card.md) and [model analysis](docs/model-selection.md).

## Checks and model tracking

Run the automated tests:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

With the API running, check that it agrees with the local model file:

```powershell
.venv/Scripts/python.exe scripts/smoke_api.py
```

The smoke check covers individual and batch predictions. It needs the same model file as the running service. GitHub Actions also runs tests, builds the model and Docker image, and checks the API before publishing an image.

Optional MLflow tracking records how a model was built and keeps a copy of its model file and report. See the [setup instructions](docs/operations.md#local-mlflow). MLflow is not needed to serve predictions.

## Further reading

- [Operations guide](docs/operations.md): Docker, deployment, monitoring, and MLflow setup.
- [Batch API and benchmark](docs/batch-prediction.md): batch behavior and measured performance.
- [Model card](docs/model-card.md): intended use, results, and limitations.
- [Feature contract](docs/feature-contract.md): expected inputs and history rules.

The main code is in `src/claim_fraud/`, runnable commands are in `scripts/`, and automated checks are in `tests/`.

This is a learning project. Real deployment would require real-world validation, security, fairness assessment, and a reliable source of claim histories. Monitoring here tracks scores and review volume; it cannot tell whether new predictions are correct without confirmed outcomes.
