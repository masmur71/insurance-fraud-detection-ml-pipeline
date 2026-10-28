"""Small HTTP demo for a running service; requires only the Python standard library."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    payload = json.loads((Path(__file__).resolve().parents[1] / "examples/claim.json").read_text())
    for path, body in (("/health", None), ("/predict", payload),
                       ("/predict", {**payload, "prior_claims_total": -1}), ("/monitoring", None)):
        request = Request(args.url + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10) as response:
                print(path, response.status, json.dumps(json.load(response), indent=2))
        except HTTPError as exc:
            if exc.code != 422:
                raise
            print(path, exc.code, json.dumps(json.load(exc), indent=2))


if __name__ == "__main__":
    main()
