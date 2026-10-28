"""Compare a local Parquet claim batch with an aggregate training reference."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_fraud.drift import analyze, sha256, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        report = analyze(args.model, reference, pd.read_parquet(args.input),
                         {"file": args.input.name, "sha256": sha256(args.input)})
        write_report(args.output_dir, report)
    except ValueError as exc:
        parser.exit(2, str(exc) + "\n")
    print(f"Report written: {args.output_dir}; status: {report['status']}")


if __name__ == "__main__":
    main()
