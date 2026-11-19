"""Export the exact CRIC split used in the paper from the preprocessing output.

Usage (after running notebooks/01_cric_preprocessing.ipynb):
    python src/export_split.py /kaggle/working/cric_prep splits/cric_split.csv
"""
import json
import sys

import pandas as pd

prep, out = sys.argv[1], sys.argv[2]
meta = pd.read_csv(f"{prep}/metadata.csv")
cols = [c for c in ["cell_id", "image_filename", "nucleus_x", "nucleus_y", "bethesda_system", "grade", "group", "split"]
        if c in meta.columns]
meta[cols].to_csv(out, index=False)
rep = json.load(open(f"{prep}/report.json"))
print(f"{len(meta)} cells -> {out}")
print("split hash (report.json):", rep.get("split_hash"), "| expected: 2761d09ffa4dd206")
print(meta.groupby("split").agg(cells=("split", "size"), images=("image_filename", "nunique")))
