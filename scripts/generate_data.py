"""Run the locked notebook's generation/export cells, without EDA or diagnostics."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.glob("*.parquet")):
        raise FileExistsError("Output contains data; use an empty directory to preserve existing data")
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/notebooks/vehicle_insurance_fraud_synth.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if "def build_features" not in "".join(cells[20]["source"]):
        raise ValueError("Notebook layout changed; review the generation boundary")
    namespace = {"__name__": "__generator__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(cells[:21]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Execute only the repository's trusted generator, never a user-supplied notebook.
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)  # noqa: S102
            if index == 3:
                namespace.update(DATA_DIR=output, TRUTH_DIR=output, display=lambda *a, **k: None)
    print(f"Generated synthetic data, seed 20261028, in {output}")
    print(f"Generator SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
