"""Step 3: Aggregate POI data to H3 hexagons."""
import numpy as np
import pandas as pd
from scipy.stats import entropy
from config import CHECKPOINTS
from utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint

SUPER_CATEGORIES = [
    "Food & Dining", "Shopping", "Entertainment & Fitness",
    "Services", "Convenience", "Grocery",
]

BRAND_ARCHETYPES = [
    "experiential", "fast_casual_trendy", "value_bulk",
    "family_essentials", "nightlife_social", "fitness_wellness",
    "home_lifestyle",
]


def run():
    output_path = CHECKPOINTS["aggregate"]

    if checkpoint_exists(output_path):
        print("Step 3 (aggregate): checkpoint exists, skipping.")
        return

    print("Step 3: Aggregating to H3 hexagons...")
    df = load_checkpoint(CHECKPOINTS["enrich"])
    print(f"  Loaded {len(df):,} enriched POIs across {df['h3_index'].nunique():,} hexes")

    # Group by H3 hex
    hex_groups = df.groupby("h3_index")

    records = []
    for hex_id, group in hex_groups:
        poi_count = len(group)
        unique_categories = group["primary_category"].nunique()

        # Category counts
        cat_counts = group["super_category"].value_counts()
        food_count = int(cat_counts.get("Food & Dining", 0))
        shopping_count = int(cat_counts.get("Shopping", 0))
        entertainment_count = int(cat_counts.get("Entertainment & Fitness", 0))
        services_count = int(cat_counts.get("Services", 0))
        convenience_count = int(cat_counts.get("Convenience", 0))
        grocery_count = int(cat_counts.get("Grocery", 0))
        other_count = int(cat_counts.get("Other", 0))

        # Category proportions (excluding Other)
        retail_total = food_count + shopping_count + entertainment_count + services_count + convenience_count + grocery_count
        if retail_total > 0:
            food_pct = food_count / retail_total
            shopping_pct = shopping_count / retail_total
            entertainment_pct = entertainment_count / retail_total
            services_pct = services_count / retail_total
            convenience_pct = convenience_count / retail_total
            grocery_pct = grocery_count / retail_total
        else:
            food_pct = shopping_pct = entertainment_pct = 0.0
            services_pct = convenience_pct = grocery_pct = 0.0

        # Shannon entropy over super-category distribution
        proportions = [food_pct, shopping_pct, entertainment_pct,
                       services_pct, convenience_pct, grocery_pct]
        proportions = [p for p in proportions if p > 0]
        shannon = float(entropy(proportions, base=2)) if proportions else 0.0

        # Anchor / quality / infrastructure counts
        anchor_count = int(group["is_anchor"].sum())
        quality_count = int(group["is_quality"].sum())
        infrastructure_count = int(group["is_infrastructure"].sum())
        family_anchor_count = int(group["is_family_anchor"].sum())

        # Brand archetype counts
        archetype_counts = group[group["brand_archetype"] != ""]["brand_archetype"].value_counts()
        archetype_data = {f"ba_{a}": int(archetype_counts.get(a, 0)) for a in BRAND_ARCHETYPES}

        # Top POI names (for display)
        top_pois = (
            group[group["name"].notna()]
            .nlargest(min(20, len(group)), "confidence", keep="all")
            if "confidence" in group.columns and group["confidence"].notna().any()
            else group[group["name"].notna()].head(20)
        )
        poi_names = top_pois["name"].tolist()[:20]
        poi_categories = top_pois["primary_category"].tolist()[:20]
        poi_supers = top_pois["super_category"].tolist()[:20]

        # Get hex center for clustering later
        lat, lng = group["latitude"].mean(), group["longitude"].mean()

        record = {
            "h3_index": hex_id,
            "lat": lat,
            "lng": lng,
            "poi_count": poi_count,
            "unique_categories": unique_categories,
            "shannon_entropy": round(shannon, 3),
            "anchor_count": anchor_count,
            "quality_count": quality_count,
            "infrastructure_count": infrastructure_count,
            "family_anchor_count": family_anchor_count,
            "food_count": food_count,
            "shopping_count": shopping_count,
            "entertainment_count": entertainment_count,
            "services_count": services_count,
            "convenience_count": convenience_count,
            "grocery_count": grocery_count,
            "other_count": other_count,
            "food_pct": round(food_pct, 4),
            "shopping_pct": round(shopping_pct, 4),
            "entertainment_pct": round(entertainment_pct, 4),
            "services_pct": round(services_pct, 4),
            "convenience_pct": round(convenience_pct, 4),
            "grocery_pct": round(grocery_pct, 4),
            "poi_names": poi_names,
            "poi_categories": poi_categories,
            "poi_supers": poi_supers,
        }
        record.update(archetype_data)
        records.append(record)

    result = pd.DataFrame(records)
    print(f"  Aggregated to {len(result):,} hexagons")
    print(f"  POI count range: {result['poi_count'].min()} - {result['poi_count'].max()}")
    print(f"  Mean POIs per hex: {result['poi_count'].mean():.1f}")

    # Show archetype presence
    for a in BRAND_ARCHETYPES:
        col = f"ba_{a}"
        nonzero = (result[col] > 0).sum()
        total = result[col].sum()
        print(f"  {a}: {total:,} POIs across {nonzero:,} hexes")

    save_checkpoint(result, output_path)


if __name__ == "__main__":
    run()
