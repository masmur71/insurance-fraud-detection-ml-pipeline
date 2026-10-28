# Synchronous batch prediction

`POST /predict/batch` accepts `{"claims": [...]}` with 1-100 claims and returns
`{"predictions": [...]}` in input order. Each claim uses the existing single-request
contract; each prediction uses the same score, threshold, review flag, reason codes,
model version, and synthetic-data metadata as `/predict`.

All claims are validated before scoring. An invalid item rejects the entire request
with HTTP 422, including its zero-based index in the error location, for example
`["body", "claims", 1, "age"]`. Empty or oversized lists are also rejected.
Invalid batches do not increment prediction counters. A successful batch counts
one HTTP request and one prediction/review observation per claim. Both endpoints
share inference code; a batch constructs one DataFrame and calls `predict_proba` once.

## Local benchmark

Measured on 2026-10-29 using the same local Docker container and frozen artifact
for both modes. The fixture uses the example claim with deterministic amount
variations; it reads no training, validation, or holdout dataset.
Artifact SHA-256: `df51b29e695a231799e4fe29443c47016c5360135f397287e2da41e41f136306`.

| Mode | Median time per 100 claims | Aggregate claims/second |
| --- | ---: | ---: |
| Sequential individual HTTP requests | 1.8592 s | 52.47 |
| One batch HTTP request | 0.0240 s | 3816.84 |

The client ran Windows with Python 3.13; the Docker server ran Python 3.11 with
one serving worker and one model thread. Each mode had one warmup and ten measured
rounds, alternating order. Times include HTTP serialization and round trips over
loopback. This compares complete processing of 100 claims, not individual claim
latency. Results describe this machine and workload; throughput is not a production
capacity estimate. Client requests are sequential and use urllib's default
connection behavior. See `reports/batch-benchmark.json` for raw timings and the
artifact SHA-256; that local report is excluded from Git.

## Verification

Tests cover sizes 0, 1, 100, and 101, mixed inputs with unseen categories and
permitted missing values, indexed validation errors, order, score parity within
`1e-12`, all response metadata, model-call count, telemetry, and invalid model
scores without partial prediction counting. The existing Docker HTTP smoke now
also compares batch and individual results. The recipe, threshold, and historical
holdout results remain unchanged.
