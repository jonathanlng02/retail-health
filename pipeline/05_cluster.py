"""Step 5: Cluster hexagons into retail nodes and classify types using brand archetypes."""
import numpy as np
import pandas as pd
import h3
from collections import Counter
from sklearn.preprocessing import StandardScaler
from config import (
    CHECKPOINTS, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
)
from utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint

# Cluster types (ordered by classification priority)
CLUSTER_TYPES = [
    "Entertainment Corridor",
    "DTC & Experiential",
    "Family-Oriented",
    "Value & Bulk",
    "Fitness & Wellness Hub",
    "Home & Lifestyle",
    "Convenience/Daily Needs",
    "Mixed-Use/Lifestyle",
]


def classify_cluster(members, df_global):
    """Classify a spatial cluster based on brand archetype and category signals."""
    total_pois = members["poi_count"].sum()
    if total_pois == 0:
        return "Mixed-Use/Lifestyle"

    # Aggregate category proportions
    food_pct = members["food_count"].sum() / total_pois
    shopping_pct = members["shopping_count"].sum() / total_pois
    entertainment_pct = members["entertainment_count"].sum() / total_pois
    services_pct = members["services_count"].sum() / total_pois
    convenience_pct = members["convenience_count"].sum() / total_pois
    grocery_pct = members["grocery_count"].sum() / total_pois

    # Aggregate brand archetype totals
    ba_dtc = members["ba_dtc_experiential"].sum()
    ba_fc = members["ba_fast_casual_trendy"].sum()
    ba_value = members["ba_value_bulk"].sum()
    ba_family = members["ba_family_essentials"].sum()
    ba_nightlife = members["ba_nightlife_social"].sum()
    ba_fitness = members["ba_fitness_wellness"].sum()
    ba_home = members["ba_home_lifestyle"].sum()

    total_branded = ba_dtc + ba_fc + ba_value + ba_family + ba_nightlife + ba_fitness + ba_home
    has_family_anchor = members["family_anchor_count"].sum() > 0

    avg_entropy = members["shannon_entropy"].mean()
    entropy_75th = df_global["shannon_entropy"].quantile(0.75)
    entropy_median = df_global["shannon_entropy"].median()
    max_cat_pct = max(food_pct, shopping_pct, entertainment_pct,
                      services_pct, convenience_pct, grocery_pct)

    # --- Classification rules using brand archetypes ---

    # Entertainment Corridor: heavy nightlife/social + food/dining
    if (ba_nightlife >= 3 and food_pct > 0.35) or (food_pct > 0.40 and entertainment_pct > 0.12):
        return "Entertainment Corridor"

    # DTC & Experiential: presence of DTC/experiential brands + fast casual trendy
    if ba_dtc >= 2 or (ba_dtc >= 1 and ba_fc >= 2 and shopping_pct > 0.15):
        return "DTC & Experiential"

    # Family-Oriented: family essentials brands + grocery presence
    if (ba_family >= 3 or (ba_family >= 1 and grocery_pct > 0.05 and has_family_anchor)):
        return "Family-Oriented"

    # Value & Bulk: value/bulk brands dominant
    if ba_value >= 3 or (ba_value >= 2 and convenience_pct > 0.10):
        return "Value & Bulk"

    # Fitness & Wellness Hub: fitness brands concentrated
    if ba_fitness >= 3 or (ba_fitness >= 2 and entertainment_pct > 0.15):
        return "Fitness & Wellness Hub"

    # Home & Lifestyle: home/lifestyle brands
    if ba_home >= 2 or (ba_home >= 1 and shopping_pct > 0.30):
        return "Home & Lifestyle"

    # Convenience/Daily Needs: high convenience ratio or low-density services
    avg_poi = members["poi_count"].mean()
    if convenience_pct > 0.20 or (avg_poi < 15 and services_pct > 0.25):
        return "Convenience/Daily Needs"

    # Mixed-Use/Lifestyle: diverse mix, no single dominant category
    if max_cat_pct < 0.45 and avg_entropy > entropy_75th:
        return "Mixed-Use/Lifestyle"

    # Fallback: classify by dominant signal
    if food_pct > 0.35:
        return "Entertainment Corridor"
    if shopping_pct > 0.25 and avg_entropy > entropy_median:
        return "DTC & Experiential"
    if ba_family >= 1 or grocery_pct > 0.05:
        return "Family-Oriented"

    return "Mixed-Use/Lifestyle"


def run():
    output_path = CHECKPOINTS["cluster"]

    if checkpoint_exists(output_path):
        print("Step 5 (cluster): checkpoint exists, skipping.")
        return

    print("Step 5: Clustering retail nodes...")
    df = load_checkpoint(CHECKPOINTS["score"])
    print(f"  Loaded {len(df):,} scored hexagons")

    # Only cluster hexes with meaningful retail presence (at least 3 POIs)
    mask = df["poi_count"] >= 3
    cluster_df = df[mask].copy()
    print(f"  Clustering {len(cluster_df):,} hexes with 3+ POIs")

    # Phase A: Spatial clustering with HDBSCAN
    coords = cluster_df[["lat", "lng"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
        )
        cluster_labels = clusterer.fit_predict(coords_scaled)
    except ImportError:
        print("  WARNING: hdbscan not installed, falling back to DBSCAN")
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.3, min_samples=HDBSCAN_MIN_SAMPLES)
        cluster_labels = clusterer.fit_predict(coords_scaled)

    cluster_df["cluster_id"] = cluster_labels
    n_clusters = len(set(cluster_labels) - {-1})
    n_noise = (cluster_labels == -1).sum()
    print(f"  Found {n_clusters} spatial clusters, {n_noise} noise hexes")

    # Phase B: Classify cluster types using brand archetypes
    cluster_types = {}
    for cid in sorted(set(cluster_labels) - {-1}):
        members = cluster_df[cluster_df["cluster_id"] == cid]
        cluster_types[cid] = classify_cluster(members, df)

    # Map cluster types back to hexes
    cluster_df["cluster_type"] = cluster_df["cluster_id"].map(
        lambda x: cluster_types.get(x, None) if x != -1 else None
    )

    # Merge back with full df
    df["cluster_id"] = None
    df["cluster_type"] = None
    df.loc[mask, "cluster_id"] = cluster_df["cluster_id"].values
    df.loc[mask, "cluster_type"] = cluster_df["cluster_type"].values

    # Set noise (-1) to null
    df.loc[df["cluster_id"] == -1, "cluster_id"] = None
    df.loc[df["cluster_id"] == -1, "cluster_type"] = None

    # Phase C: Neighbor-vote to assign unclustered hexes near cluster boundaries
    # If an unclustered hex (with 2+ POIs) has neighbors that are mostly one cluster type,
    # adopt that cluster type. This creates smoother neighborhood boundaries.
    print("  Smoothing cluster boundaries via neighbor voting...")
    hex_to_ctype = dict(zip(df["h3_index"], df["cluster_type"]))
    assigned = 0
    for idx, row in df.iterrows():
        if row["cluster_type"] is not None or row["poi_count"] < 2:
            continue
        neighbors = h3.grid_disk(row["h3_index"], 1)
        neighbor_types = [hex_to_ctype.get(n) for n in neighbors
                          if n != row["h3_index"] and hex_to_ctype.get(n) is not None]
        if len(neighbor_types) >= 2:
            # Majority vote
            most_common = Counter(neighbor_types).most_common(1)[0]
            if most_common[1] >= 2:
                df.at[idx, "cluster_type"] = most_common[0]
                hex_to_ctype[row["h3_index"]] = most_common[0]
                assigned += 1
    print(f"  Assigned {assigned} border hexes via neighbor voting")

    print(f"  Cluster type distribution:")
    print(df["cluster_type"].value_counts().to_string(header=False))

    save_checkpoint(df, output_path)


if __name__ == "__main__":
    run()
