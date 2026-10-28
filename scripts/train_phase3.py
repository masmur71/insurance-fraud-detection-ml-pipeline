"""Repository-local entry point for phase 3 tuning and frozen evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_fraud.experiment import main

if __name__ == "__main__":
    main()
