"""One-shot script to generate master_dataset.csv from simulation (no HTTP)."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from config.settings import Paths, PROVINCES
from src.data.ingestion import DataIngestion

Paths.ensure_dirs()

ing = DataIngestion.__new__(DataIngestion)
ing.timeout = 1

print("Simulating PAHO dengue data...")
paho_frames = [ing._simulate_paho_data(y) for y in range(2020, 2026)]
dengue_df = pd.concat(paho_frames, ignore_index=True)
print(f"  dengue shape: {dengue_df.shape}")

print("Simulating climate data for 32 provinces...")
climate_df = pd.concat(
    [ing._simulate_climate_data(p, 104) for p in PROVINCES],
    ignore_index=True,
)
print(f"  climate shape: {climate_df.shape}")

print("Getting demographic reference data...")
demographic_df = ing._get_reference_demographic_data()
print(f"  demographic shape: {demographic_df.shape}")

print("Simulating ENSO data...")
enso_df = ing._simulate_enso_data()
print(f"  enso shape: {enso_df.shape}")

print("Merging all datasets...")
master_df = ing._merge_datasets(dengue_df, climate_df, demographic_df, enso_df)
print(f"  master shape: {master_df.shape}")

out = Paths.DATA_RAW / "master_dataset.csv"
master_df.to_csv(out, index=False)
print(f"Saved to {out}")
print("Columns:", list(master_df.columns))
