"""Repository-local entry point; package installation is optional."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_fraud.train import main

if __name__ == "__main__":
    main()
