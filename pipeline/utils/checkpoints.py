"""Checkpoint helpers for idempotent pipeline steps."""
from pathlib import Path
import pandas as pd


def checkpoint_exists(path: Path) -> bool:
    """Check if a checkpoint file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def load_checkpoint(path: Path) -> pd.DataFrame:
    """Load a parquet checkpoint."""
    return pd.read_parquet(path)


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a parquet checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved checkpoint: {path} ({len(df):,} rows)")
