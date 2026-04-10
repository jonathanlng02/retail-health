"""Step 4: Compute retail health score with grocery proximity and infrastructure penalties."""
import numpy as np
import pandas as pd
import h3
from scipy.spatial.distance import jensenshannon
from config import SCORE_WEIGHTS, IDEAL_CATEGORY_MIX, CHECKPOINTS
from utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")


def spatial_smooth(df, column, k=1, self_weight=0.6):
    """Smooth a column using H3 k-ring neighbors."""
    hex_to_val = dict(zip(df["h3_index"], df[column]))
    smoothed = []
    for _, row in df.iterrows():
        own_val = row[column]
        neighbors = h3.grid_disk(row["h3_index"], k)
        neighbor_vals = [hex_to_val[n] for n in neighbors if n in hex_to_val and n != row["h3_index"]]
        if neighbor_vals:
            neighbor_mean = np.mean(neighbor_vals)
            smoothed.append(self_weight * own_val + (1 - self_weight) * neighbor_mean)
        else:
            smoothed.append(own_val)
    return smoothed


def run():
    output_path = CHECKPOINTS["score"]

    if checkpoint_exists(output_path):
        print("Step 4 (score): checkpoint exists, skipping.")
        return

    print("Step 4: Computing retail health scores...")
    df = load_checkpoint(CHECKPOINTS["aggregate"])
    print(f"  Loaded {len(df):,} hexagons")

    # ── Component 1: POI Density (log-transformed percentile) ──
    df["log_poi_count"] = np.log1p(df["poi_count"])
    df["density_score"] = percentile_rank(df["log_poi_count"])

    # ── Component 2: Category Diversity (Shannon entropy percentile) ──
    df["diversity_score"] = percentile_rank(df["shannon_entropy"])

    # ── Component 3: Destination Anchor presence (NOT gas stations) ──
    # Capped at 4 anchors. These are real foot-traffic drivers.
    df["anchor_score"] = (df["anchor_count"].clip(upper=4) / 4.0)

    # ── Component 4: Category Mix (JSD from ideal) ──
    ideal = np.array([
        IDEAL_CATEGORY_MIX["Food & Dining"],
        IDEAL_CATEGORY_MIX["Shopping"],
        IDEAL_CATEGORY_MIX["Entertainment & Fitness"],
        IDEAL_CATEGORY_MIX["Services"],
        IDEAL_CATEGORY_MIX["Convenience"],
        IDEAL_CATEGORY_MIX["Grocery"],
    ])

    def compute_mix_score(row):
        dist = np.array([
            row["food_pct"], row["shopping_pct"], row["entertainment_pct"],
            row["services_pct"], row["convenience_pct"], row["grocery_pct"],
        ])
        if dist.sum() == 0:
            return 0.0
        dist = dist / dist.sum()
        jsd = jensenshannon(dist, ideal)
        return max(0.0, 1.0 - jsd)

    df["mix_score"] = df.apply(compute_mix_score, axis=1)

    # ── Component 5: Grocery Proximity (NEW) ──
    # Having a grocery store in your hex or immediate neighbors is a strong positive
    print("  Computing grocery proximity...")
    hex_to_grocery = dict(zip(df["h3_index"], df["grocery_count"]))
    grocery_prox = []
    for _, row in df.iterrows():
        own_grocery = row["grocery_count"]
        if own_grocery > 0:
            grocery_prox.append(1.0)  # Has grocery in this hex
        else:
            # Check k=1 and k=2 rings
            ring1 = h3.grid_disk(row["h3_index"], 1)
            ring1_grocery = sum(hex_to_grocery.get(n, 0) for n in ring1 if n != row["h3_index"])
            if ring1_grocery > 0:
                grocery_prox.append(0.7)  # Grocery within 1 hex (~460m)
            else:
                ring2 = h3.grid_disk(row["h3_index"], 2)
                ring2_grocery = sum(hex_to_grocery.get(n, 0) for n in ring2
                                    if n not in ring1 and n != row["h3_index"])
                if ring2_grocery > 0:
                    grocery_prox.append(0.35)  # Grocery within 2 hexes (~920m)
                else:
                    grocery_prox.append(0.0)  # No grocery nearby
    df["grocery_proximity"] = grocery_prox

    # ── Component 6: Quality Signal (NEW) ──
    # DTC/experiential brands signal a premium retail environment
    df["quality_score"] = (df["quality_count"].clip(upper=5) / 5.0)

    # ── Penalty: Infrastructure Saturation (NEW) ──
    # Too many gas stations/ATMs/parking relative to other POIs = bad signal
    # A few is fine, but when infrastructure is >25% of POIs, it drags score down
    df["infra_ratio"] = df["infrastructure_count"] / df["poi_count"].clip(lower=1)
    df["infra_penalty"] = np.where(
        df["infra_ratio"] > 0.25,
        (df["infra_ratio"] - 0.25).clip(upper=0.5) * 0.6,  # Max penalty ~0.3
        0.0
    )

    # ── Composite Score ──
    # Weights: density 0.20, diversity 0.15, anchors 0.20, mix 0.15, grocery 0.15, quality 0.15
    df["raw_score"] = (
        0.20 * df["density_score"]
        + 0.15 * df["diversity_score"]
        + 0.20 * df["anchor_score"]
        + 0.15 * df["mix_score"]
        + 0.15 * df["grocery_proximity"]
        + 0.15 * df["quality_score"]
        - df["infra_penalty"]
    ).clip(lower=0)

    # Spatial smoothing
    print("  Applying spatial smoothing (k=1 ring)...")
    df["smoothed_score"] = spatial_smooth(df, "raw_score", k=1, self_weight=0.6)

    # Scale to 0-100
    df["retail_health_score"] = (df["smoothed_score"] * 100).round(0).astype(int)
    df["retail_health_score"] = df["retail_health_score"].clip(0, 100)

    def score_tier(s):
        if s >= 75: return "Excellent"
        elif s >= 55: return "Good"
        elif s >= 35: return "Moderate"
        elif s >= 18: return "Below Average"
        else: return "Weak"

    df["score_tier"] = df["retail_health_score"].apply(score_tier)

    print(f"  Score distribution:")
    print(df["score_tier"].value_counts().to_string(header=False))
    print(f"  Score range: {df['retail_health_score'].min()} - {df['retail_health_score'].max()}")
    print(f"  Mean score: {df['retail_health_score'].mean():.1f}")
    print(f"  Hexes with grocery proximity: {(df['grocery_proximity'] > 0).sum():,}")
    print(f"  Hexes with infrastructure penalty: {(df['infra_penalty'] > 0).sum():,}")

    save_checkpoint(df, output_path)


if __name__ == "__main__":
    run()
