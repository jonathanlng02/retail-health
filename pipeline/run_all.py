"""Run all pipeline steps with checkpoint-based skip logic."""
import sys
import time
from pathlib import Path

# Add pipeline dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINTS, DATA_DIR, INTERMEDIATE_DIR, OUTPUT_DIR


def ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [DATA_DIR, INTERMEDIATE_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    print("=" * 60)
    print("Retail Health Score Pipeline")
    print("=" * 60)

    ensure_dirs()
    start = time.time()

    # Import and run each step
    import importlib
    steps = [
        ("01_download", "Download DFW POI data"),
        ("02_enrich", "Enrich with categories & H3"),
        ("03_aggregate", "Aggregate to H3 hexagons"),
        ("04_score", "Compute retail health scores"),
        ("05_cluster", "Cluster retail nodes"),
        ("06_export", "Export JSON for web app"),
    ]

    for module_name, description in steps:
        print(f"\n{'-' * 60}")
        print(f"  {description}")
        print(f"{'-' * 60}")
        step_start = time.time()

        module = importlib.import_module(module_name)
        module.run()

        elapsed = time.time() - step_start
        print(f"  ({elapsed:.1f}s)")

    total = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total:.1f}s")
    print(f"Output: {CHECKPOINTS['export']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pipeline()
