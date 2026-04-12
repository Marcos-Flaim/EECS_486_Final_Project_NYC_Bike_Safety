import requests
import pandas as pd
import time
from pathlib import Path

def load_crash_data(
    limit=50000,
    max_rows=2500000,
    use_cache=True,
    cache_path="data/crash_data.parquet"
):
    """
    Load NYC crash data from API or cached parquet file.

    Returns:
        pandas.DataFrame
    """

    cache_file = Path(cache_path)

    # Use cached file if it exists
    if use_cache and cache_file.exists():
        print("Loading crash data from cache...")
        return pd.read_parquet(cache_file)

    print("Fetching crash data from API...")

    offset = 0
    chunks = []

    while offset < max_rows:
        url = (
            "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
            f"?$limit={limit}&$offset={offset}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()

        if not data:
            print("No more data.")
            break

        df = pd.DataFrame(data)

        # 🔹 Normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # 🔹 Ensure key columns exist
        required_cols = [
            "collision_id",
            "crash_date",
            "latitude",
            "longitude",
            "number_of_persons_killed",
            "number_of_persons_injured"
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # 🔹 Convert numeric columns
        for col in df.columns:
            if "number_of" in col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 🔹 Convert lat/lon
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # 🔹 Drop invalid coordinates
        df = df.dropna(subset=["latitude", "longitude"])
        df = df[
            (df["latitude"] != 0) &
            (df["longitude"] != 0)
        ]

        # NYC bounding box
        df = df[
            (df["latitude"].between(40.4, 41.0)) &
            (df["longitude"].between(-74.3, -73.6))
        ]

        chunks.append(df)

        print(f"Fetched {len(df)} rows (offset={offset})")

        offset += limit
        time.sleep(1)  # rate limit safety

    full_df = pd.concat(chunks, ignore_index=True)

    # 🔹 Convert crash_date once at the end
    if "crash_date" in full_df.columns:
        full_df["crash_date"] = pd.to_datetime(full_df["crash_date"], errors="coerce")

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(cache_file)

    print(f"Saved cache to {cache_path}")

    return full_df