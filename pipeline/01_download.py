"""Step 1: Download DFW POI data from Overture Maps using the overturemaps package."""
import sys
import pyarrow as pa
import pandas as pd
import overturemaps
from config import DFW_BBOX, CHECKPOINTS, INTERMEDIATE_DIR
from utils.checkpoints import checkpoint_exists


def run():
    output_path = CHECKPOINTS["download"]

    if checkpoint_exists(output_path):
        print("Step 1 (download): checkpoint exists, skipping.")
        return

    print("Step 1: Downloading DFW POI data from Overture Maps...")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    bbox = DFW_BBOX
    print(f"  Bounding box: lat [{bbox['min_lat']}, {bbox['max_lat']}], lng [{bbox['min_lng']}, {bbox['max_lng']}]")

    # Use overturemaps package - it handles S3/Azure connection internally
    # bbox format: (min_lng, min_lat, max_lng, max_lat)  i.e. (west, south, east, north)
    bbox_tuple = (bbox["min_lng"], bbox["min_lat"], bbox["max_lng"], bbox["max_lat"])

    print("  Querying Overture Maps (this may take a few minutes)...")
    reader = overturemaps.record_batch_reader("place", bbox=bbox_tuple)

    if reader is None:
        print("  ERROR: No data returned from Overture Maps.")
        sys.exit(1)

    # Read all batches into a table then convert to pandas
    table = reader.read_all()
    df = table.to_pandas()

    print(f"  Downloaded {len(df):,} POIs")
    print(f"  Columns: {list(df.columns)}")

    if len(df) == 0:
        print("  ERROR: No POIs found in bounding box. Aborting.")
        sys.exit(1)

    # Save checkpoint
    df.to_parquet(str(output_path), index=False)
    print(f"  Saved to {output_path} ({len(df):,} rows)")


if __name__ == "__main__":
    run()
