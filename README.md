# CytoGrade

**Leakage-free, calibrated ordinal grading of Pap smear cells for cervical cancer screening, with external validation and conformal referral.**

CytoGrade grades single cervical cells as **Normal**, **Low-grade** or **High-grade**. It reads each cell at two fields of view through one shared EfficientNet-B3, fuses them with a learned gate, predicts the grade with a rank-consistent ordinal (CORAL) head, and refers uncertain cells to a cytopathologist through split conformal prediction. It is trained on a leakage-audited, image-grouped split of CRIC Cervix and tested without adaptation on Herlev.

This repository contains the code and the split definition used in the accompanying manuscript.

> **Scope.** CytoGrade is a cell-level research model. It does not diagnose patients and has not been validated at slide or patient level. It is not a medical device.

---

## Repository structure

```
CytoGrade/
├── notebooks/
│   ├── 01_cric_preprocessing.ipynb      # curation, near-duplicate grouping, image-grouped split, 24-check leakage audit
│   └── 02_cytograde_train_evaluate.ipynb  # CytoGrade + 5 baselines, leakage study, Herlev, seeds, deep ensembles
├── src/
│   ├── cric_preprocessing.py            # same code as notebook 01 (one "# %%" block per cell)
│   ├── cytograde_pipeline.py            # same code as notebook 02
│   └── export_split.py                  # writes splits/cric_split.csv from the preprocessing output
├── splits/                              # split definition (see splits/README.md)
├── data/README.md                       # where to obtain CRIC and Herlev
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Data

The datasets are **not redistributed** here. See [`data/README.md`](data/README.md) for sources and licences.

| Dataset | Role | Cells |
| --- | --- | --- |
| CRIC Cervix (conventional Pap smears, Bethesda labels) | training, validation, test | 11,534 annotated → 11,517 after curation |
| Herlev (single-cell images, WHO labels) | external test only; no label used for any choice | 917 |

---

## Reproducing the results

The experiments were run on Kaggle (one NVIDIA T4, 16 GB).

1. **Preprocessing (CPU, ~5 min).** Create a Kaggle notebook from `notebooks/01_cric_preprocessing.ipynb`, add the CRIC dataset as input and run all cells. All 24 checks must pass, and the split hash must be **`2761d09ffa4dd206`**.
2. **Training and evaluation (GPU T4, Internet ON).** Create a notebook from `notebooks/02_cytograde_train_evaluate.ipynb` and add three inputs: the output of step 1, the Herlev dataset, and (when resuming) this notebook's previous version. Use *Save Version → Save & Run All*.
   - Total training time exceeds one Kaggle session. Finished runs are saved in `cg2_out/runs/`, restored automatically from the previous version and skipped.
3. **Outputs.** Tables are written to `cg2_out/tables/` (CSV and LaTeX) and figures to `cg2_out/figures/` (PNG and PDF, 300 dpi).

The paths in the first cell of each notebook point to the Kaggle copies of the datasets. Change them if you run elsewhere.

**Determinism.** The split is fully deterministic (hash-checked). Training uses fixed seeds (42, 1, 2), but GPU non-determinism means retrained models can differ slightly from the reported numbers.

### Key settings

| Component | Setting |
| --- | --- |
| Input | 192-px context crop and its central 96-px detail crop, both resized to 256 px |
| Backbone | EfficientNet-B3 (ImageNet), shared across views, gated fusion |
| Head | CORAL ordinal head (2 thresholds) |
| Loss | class-weighted ordinal BCE + 0.5 × Jensen–Shannon two-view consistency + 0.5 × cosine anchoring to frozen Phikon features (training only) |
| Augmentation | HED stain jitter (σ = 0.05), Fourier amplitude mixing (p = 0.5, r = 0.1), scale 0.8–1.25, dihedral + affine, brightness/contrast, blur |
| Optimisation | AdamW 2×10⁻⁴, weight decay 0.05, batch 32, 2-epoch warm-up + cosine, EMA 0.999, early stopping on validation macro-F1 |
| Post-hoc | temperature scaling and split conformal prediction (α = 0.1) fitted on CRIC validation; 8-pose TTA |
| Statistics | image/source-ID cluster bootstrap (1,000 resamples), exact McNemar with Holm correction, seed-paired tests |

In the conformal step, the top-ranked grade is always included in the prediction set, so sets are never empty. A cell is referred when its set holds more than one grade.

---

## Model weights

Trained weights are not included in this repository. They can be regenerated with notebook 02.

---

## Citation

If you use this code or the split, please cite the manuscript (citation details will be added on publication) and the datasets listed in [`data/README.md`](data/README.md). See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT License (see [`LICENSE`](LICENSE)). The CRIC and Herlev datasets remain under their own terms.
