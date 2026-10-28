"""Build an aggregate drift reference from eligible training claims only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_fraud.drift import build_reference, training_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/drift-reference.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Reference output already exists")
    reference = build_reference(args.model, training_inputs(args.data_dir), {
        "kind": "eligible_training", "incident_date_before": "2024-01-01",
        "available_date_before": "2024-01-01", "seed": 20261028})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, reference)
    print(f"Reference written: {args.output}; training rows: {reference['sample_size']}")


if __name__ == "__main__":
    main()
