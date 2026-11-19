# CRIC split

The split is produced deterministically by `notebooks/01_cric_preprocessing.ipynb`:

1. Annotation QC: nuclei ≤ 6 px apart in the same image are one cell annotated twice; same label → keep one (15 removed), conflicting labels → drop both (2 cells). 11,534 → 11,517 cells.
2. Near-duplicate images: 64-bit difference hash, Hamming distance ≤ 4, merged with union–find (2 pairs found). The duplicate group is the unit of splitting.
3. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, stratified by the six Bethesda classes and grouped by duplicate group. Fold 0 = test, fold 4 = validation, folds 1–3 = training.

| Split | Images | Cells | Normal | Low-grade | High-grade |
| --- | --- | --- | --- | --- | --- |
| Train | 239 | 7,030 | 4,078 | 1,223 | 1,729 |
| Validation | 81 | 2,284 | 1,422 | 429 | 433 |
| Test | 80 | 2,203 | 1,273 | 310 | 620 |

**Split hash:** `2761d09ffa4dd206` (SHA-256 of the ordered `cell_id:split` pairs, first 16 hex digits). The training notebook refuses to start if any of the 24 compliance checks fails.

To save the cell-level assignment as a CSV after running notebook 01:

```bash
python src/export_split.py /kaggle/working/cric_prep splits/cric_split.csv
```
