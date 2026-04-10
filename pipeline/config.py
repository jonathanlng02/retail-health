"""Central configuration for the Retail Health pipeline."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DFW CBSA bounding box (covers all 13 MSA counties generously)
DFW_BBOX = {
    "min_lat": 32.0,
    "max_lat": 33.55,
    "min_lng": -98.0,
    "max_lng": -96.0,
}

# H3 resolution 8: ~0.74 km² per hex, ~460m edge length
H3_RESOLUTION = 8

# Scoring weights
SCORE_WEIGHTS = {
    "poi_density": 0.30,
    "category_diversity": 0.25,
    "anchor_presence": 0.25,
    "category_mix": 0.20,
}

# Ideal category distribution for mix score (JSD target)
IDEAL_CATEGORY_MIX = {
    "Food & Dining": 0.30,
    "Shopping": 0.25,
    "Services": 0.20,
    "Entertainment & Fitness": 0.10,
    "Convenience": 0.10,
    "Grocery": 0.05,
}

# Cluster type classification thresholds
CLUSTER_RULES = {
    "Entertainment Corridor": {
        "food_dining_pct_min": 0.45,
        "entertainment_pct_min": 0.15,
    },
    "Experiential Retail": {
        "shopping_pct_min": 0.35,
        "diversity_above_median": True,
    },
    "Family-Oriented": {
        "grocery_pct_min": 0.08,
        "requires_family_anchors": True,
        "food_pct_min": 0.20,
    },
    "Convenience/Daily Needs": {
        "convenience_pct_min": 0.25,
        "alt_low_poi_services": {"max_poi": 15, "services_pct_min": 0.30},
    },
    "Mixed-Use/Lifestyle": {
        "no_dominant_category_max": 0.40,
        "entropy_above_75th": True,
    },
}

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OUTPUT_DIR = DATA_DIR / "output"

# Checkpoint file paths
CHECKPOINTS = {
    "download": INTERMEDIATE_DIR / "dfw_places.parquet",
    "enrich": INTERMEDIATE_DIR / "dfw_enriched.parquet",
    "aggregate": INTERMEDIATE_DIR / "h3_aggregated.parquet",
    "score": INTERMEDIATE_DIR / "h3_scored.parquet",
    "cluster": INTERMEDIATE_DIR / "h3_clustered.parquet",
    "export": OUTPUT_DIR / "hexagons.json",
}

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
