"""Measure sequential HTTP requests versus batch on one server/artifact; no holdout."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--size", type=int, default=100, choices=range(1, 101))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/batch-benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    def health():
        with urlopen(args.url + "/health", timeout=30) as response:
            return json.load(response)["model_version"]
    version = health()
    original = json.loads((ROOT / "examples/claim.json").read_text())
    claims = [{**original, "claim_amount": float(1000 + i * 100)} for i in range(args.size)]
    def post(route, payload):
        request = Request(args.url + route, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    def run(mode):
        start = time.perf_counter()
        results = ([post("/predict", claim) for claim in claims] if mode == "sequential"
                   else post("/predict/batch", {"claims": claims})["predictions"])
        elapsed = time.perf_counter() - start
        assert len(results) == args.size
        assert all(item["model_version"] == version for item in results)
        return elapsed, results
    _, singles = run("sequential")
    _, batch = run("batch")
    for left, right in zip(singles, batch):
        assert abs(left["score"] - right["score"]) <= 1e-12
        assert {k: v for k, v in left.items() if k != "score"} == {
            k: v for k, v in right.items() if k != "score"}
    timings = {"sequential": [], "batch": []}
    for repeat in range(args.repeats):
        # Alternate order to reduce warmup/order bias; both modes get one warmup.
        modes = ("sequential", "batch") if repeat % 2 == 0 else ("batch", "sequential")
        for mode in modes:
            elapsed, _ = run(mode)
            timings[mode].append(elapsed)
    assert health() == version
    report = {"model_sha256": version, "server_url": args.url,
              "client_platform": platform.platform(), "client_python": platform.python_version(),
              "claims_per_round": args.size, "repeats": args.repeats,
              "method": "Loopback HTTP, sequential clients, one warmup per mode, alternating order",
              "fixture": "Example claim with deterministic amount variations; no dataset reads",
              "results": {mode: {"round_seconds": values,
                  "median_round_seconds": statistics.median(values),
                  "claims_per_second": args.size * len(values) / sum(values)}
                  for mode, values in timings.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
