"""Step 6: Export hexagon data as JSON for the web app."""
import json
import pandas as pd
from config import CHECKPOINTS, OUTPUT_DIR
from utils.checkpoints import checkpoint_exists, load_checkpoint
from utils.categories import is_anchor


def _to_list(val):
    """Convert a value that might be a numpy array, list, or None to a plain list."""
    if val is None:
        return []
    if hasattr(val, 'tolist'):
        return val.tolist()
    if isinstance(val, list):
        return val
    return []


CLUSTER_TYPE_SHORT = {
    "Entertainment Corridor": "entertainment",
    "DTC & Experiential": "dtc_experiential",
    "Family-Oriented": "family",
    "Value & Bulk": "value_bulk",
    "Fitness & Wellness Hub": "fitness_wellness",
    "Home & Lifestyle": "home_lifestyle",
    "Convenience/Daily Needs": "convenience",
    "Mixed-Use/Lifestyle": "mixed_use",
}


def run():
    output_path = CHECKPOINTS["export"]

    if checkpoint_exists(output_path):
        print("Step 6 (export): checkpoint exists, skipping.")
        return

    print("Step 6: Exporting data for web app...")
    df = load_checkpoint(CHECKPOINTS["cluster"])
    print(f"  Loaded {len(df):,} hexagons")

    # Also load enriched POI data for detailed POI lists
    poi_df = load_checkpoint(CHECKPOINTS["enrich"])
    print(f"  Loaded {len(poi_df):,} POIs for detail")

    # Build per-hex POI lists with grouping of duplicates
    # e.g. "4x Bank of America ATM" instead of listing 4 separate entries
    poi_by_hex = {}
    for hex_id, group in poi_df.groupby("h3_index"):
        # Group by (name, primary_category) to consolidate duplicates
        from collections import Counter, defaultdict
        name_cat_counts = Counter()
        name_cat_meta = {}
        for _, poi in group.iterrows():
            name = poi.get("name")
            if name is None or pd.isna(name):
                continue
            cat = str(poi.get("primary_category", "")) if pd.notna(poi.get("primary_category")) else ""
            key = (str(name), cat)
            name_cat_counts[key] += 1
            if key not in name_cat_meta:
                name_cat_meta[key] = {
                    "sc": str(poi.get("super_category", "")),
                    "an": bool(poi.get("is_anchor", False)),
                    "q": bool(poi.get("is_quality", False)),
                    "inf": bool(poi.get("is_infrastructure", False)),
                    "ba": str(poi.get("brand_archetype", "")) if poi.get("brand_archetype") else "",
                }

        pois = []
        for (name, cat), count in name_cat_counts.most_common():
            meta = name_cat_meta[(name, cat)]
            pois.append({
                "n": name,
                "c": cat,
                "sc": meta["sc"],
                "an": meta["an"],
                "q": meta["q"],
                "inf": meta["inf"],
                "ba": meta["ba"],
                "cnt": count,
            })
        # Sort: anchors first, then quality signals, then by count desc
        pois.sort(key=lambda p: (not p["an"], not p["q"], -p["cnt"], p["n"]))
        poi_by_hex[hex_id] = pois  # include all POIs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hexagons = []
    for _, row in df.iterrows():
        hex_id = row["h3_index"]
        hex_data = {
            "h": hex_id,
            "s": int(row["retail_health_score"]),
            "t": row.get("score_tier", ""),
            "d": [
                int(row["food_count"]),
                int(row["shopping_count"]),
                int(row["entertainment_count"]),
                int(row["services_count"]),
                int(row["convenience_count"]),
                int(row["grocery_count"]),
            ],
            "pc": int(row["poi_count"]),
            "e": float(row["shannon_entropy"]),
            "a": int(row["anchor_count"]),
            "ct": CLUSTER_TYPE_SHORT.get(row.get("cluster_type"), None),
            "ci": int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else None,
            "q": int(row.get("quality_count", 0)),
            "inf": int(row.get("infrastructure_count", 0)),
            "sb": {
                "quality": round(float(row.get("quality_score", 0)), 3),
                "grocery": round(float(row.get("grocery_proximity", 0)), 3),
                "mix": round(float(row.get("mix_score", 0)), 3),
                "density": round(float(row.get("density_score", 0)), 3),
                "infra_penalty": round(float(row.get("infra_penalty", 0)), 3),
            },
            "pois": poi_by_hex.get(hex_id, []),
        }
        hexagons.append(hex_data)

    # Build a search index: count retailer name occurrences, keep those appearing 2+ times
    from collections import Counter
    name_counts = Counter()
    all_categories = set()
    for pois in poi_by_hex.values():
        for p in pois:
            if p["n"]:
                name_counts[p["n"]] += 1
            if p["c"]:
                all_categories.add(p["c"])
    # Only include retailers that appear in multiple hexes (chain-like)
    all_names = {name for name, count in name_counts.items() if count >= 2}

    output = {
        "metadata": {
            "generated": str(pd.Timestamp.now()),
            "hex_count": len(hexagons),
            "h3_resolution": 8,
            "score_range": [0, 100],
            "bbox": {
                "min_lat": float(df["lat"].min()),
                "max_lat": float(df["lat"].max()),
                "min_lng": float(df["lng"].min()),
                "max_lng": float(df["lng"].max()),
            },
        },
        "hexagons": hexagons,
        "search_index": {
            "retailers": sorted(all_names)[:2000],
            "categories": sorted(all_categories),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported {len(hexagons):,} hexagons to {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Search index: {len(all_names):,} retailers, {len(all_categories):,} categories")


if __name__ == "__main__":
    run()
