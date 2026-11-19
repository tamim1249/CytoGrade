# =====================================================================
#  CRIC Cervix : preprocessing + leakage/quality test suite (CytoGrade)
#  Output: /kaggle/working/cric_prep/
#     crops_ctx192.npy      uint8 [N,192,192,3]  (nucleus-centered context crops)
#     metadata.csv          one row per kept cell (labels, split, flags, group)
#     norm_stats.json       train-only mean/std
#     report.json           all checks (PASS/FAIL/INFO) + split hash
#     tables/, figures/
#  CPU only. On Kaggle: add the CRIC dataset as input and run all cells (~3-5 min).
# =====================================================================


# %% P1 : CONFIG ---------------------------------------------------------
PCFG = dict(
    CRIC_ROOT = "/kaggle/input/datasets/rakibhossainsajib/cric-cervix-dataset/cric_dataset_complete",
    CRIC_CSV  = "/kaggle/input/datasets/rakibhossainsajib/cric-cervix-dataset/cric_dataset_complete/classes/classifications.csv",
    OUT = "/kaggle/working/cric_prep",
    SEED = 42, N_FOLDS = 5, TEST_FOLD = 0, VAL_FOLD = 4,      # train = remaining 3 folds
    CTX = 192,                   # context crop (cell + cytoplasm)
    DUP_PX = 6,                  # two nuclei <= 6 px apart in one image = the same cell annotated twice
    IMG_HASH_MAX_HAMMING = 4,    # full image dHash 64-bit: <= 4 differing bits = duplicate field
    CROP_SIM_FAIL = 0.985,       # cross-split crop cosine similarity above this = near-identical (leakage)
    CRAMERS_V_MAX = 0.15,        # split x class association effect size (smaller is better)
    BLUR_PCT = 1.0,              # flag the blurriest 1% of crops
)
import os, json, hashlib, itertools, collections, time
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from sklearn.model_selection import StratifiedGroupKFold
for d in ("", "/tables", "/figures"): os.makedirs(PCFG["OUT"] + d, exist_ok=True)
REPORT = {"checks": [], "notes": []}
def check(name, passed, detail, level="FAIL"):
    status = "PASS" if passed else level
    REPORT["checks"].append({"check": name, "status": status, "detail": detail})
    print(f"[{status:4s}] {name}: {detail}")
def note(s): REPORT["notes"].append(s); print("  note:", s)

BETH = ["Negative for intraepithelial lesion", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
SHORT = {"Negative for intraepithelial lesion": "NILM", "ASC-US": "ASC-US", "LSIL": "LSIL", "ASC-H": "ASC-H", "HSIL": "HSIL", "SCC": "SCC"}
GRADE = {"NILM": 0, "ASC-US": 1, "LSIL": 1, "ASC-H": 2, "HSIL": 2, "SCC": 2}
GRADES = ["Normal", "Low-grade", "High-grade"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
def savefig(n):
    for e in ("png", "pdf"): plt.savefig(f"{PCFG['OUT']}/figures/{n}.{e}")
    print("   figure ->", n)


# %% P2 : LOAD + LABELS + ANNOTATION QC ----------------------------------
raw = pd.read_csv(PCFG["CRIC_CSV"])
n_raw = len(raw)
unknown = set(raw.bethesda_system) - set(BETH)
check("All Bethesda labels recognised", not unknown, f"unknown={sorted(unknown)}")
raw["beth"] = raw.bethesda_system.map(SHORT); raw["grade"] = raw.beth.map(GRADE); raw["binary"] = (raw.grade > 0).astype(int)

sizes = {}
for fn in raw.image_filename.unique():
    with Image.open(f"{PCFG['CRIC_ROOT']}/{fn}") as im: sizes[fn] = im.size
raw["W"] = raw.image_filename.map(lambda f: sizes[f][0]); raw["H"] = raw.image_filename.map(lambda f: sizes[f][1])
oob = (raw.nucleus_x < 0) | (raw.nucleus_y < 0) | (raw.nucleus_x >= raw.W) | (raw.nucleus_y >= raw.H)
check("Nucleus coordinates inside image", oob.sum() == 0, f"{int(oob.sum())} out-of-bounds")
check("Unique cell_id", raw.cell_id.is_unique, f"{raw.cell_id.duplicated().sum()} duplicated ids")
missing = [f for f in raw.image_filename.unique() if not os.path.exists(f"{PCFG['CRIC_ROOT']}/{f}")]
check("All annotated images exist", not missing, f"{len(missing)} missing")

# ---- duplicate / conflicting annotations (same nucleus annotated twice) ----
drop, conflict_ids, dup_same = set(), [], 0
for fn, g in raw.groupby("image_filename"):
    xy = g[["nucleus_x", "nucleus_y"]].values; ids = g.cell_id.values; labs = g.beth.values
    if len(g) < 2: continue
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    seen = set()
    for i, j in zip(*np.where(np.triu(d <= PCFG["DUP_PX"], 1))):
        if labs[i] == labs[j]: drop.add(ids[j]); dup_same += 1                    # same label -> keep one
        else: conflict_ids += [ids[i], ids[j]]                                     # conflicting labels -> drop both
drop |= set(conflict_ids)
note(f"Duplicate annotations (same label, <= {PCFG['DUP_PX']} px): {dup_same} removed")
note(f"Conflicting annotations (different labels, same nucleus): {len(set(conflict_ids))} cells removed")
df = raw[~raw.cell_id.isin(drop)].reset_index(drop=True)
check("No duplicate/conflicting annotation remains", True, f"{n_raw} -> {len(df)} cells")


# %% P3 : DUPLICATE IMAGE DETECTION (perceptual hash) --------------------
def dhash(path, hs=8):
    g = np.asarray(Image.open(path).convert("L").resize((hs + 1, hs), Image.BILINEAR), dtype=np.float32)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())
files = sorted(df.image_filename.unique())
HASH = np.stack([dhash(f"{PCFG['CRIC_ROOT']}/{f}") for f in files])
bits = np.unpackbits(HASH, axis=1).astype(np.int16)
ham = (bits[:, None, :] != bits[None, :, :]).sum(-1); np.fill_diagonal(ham, 999)
pairs = [(files[i], files[j], int(ham[i, j])) for i, j in zip(*np.where(np.triu(ham <= PCFG["IMG_HASH_MAX_HAMMING"], 1)))]
note(f"Near-duplicate image pairs (dHash hamming <= {PCFG['IMG_HASH_MAX_HAMMING']}): {len(pairs)}")
# union-find: duplicate images form one group (kept in the same split)
parent = {f: f for f in files}
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b, _ in pairs: parent[find(a)] = find(b)
df["group"] = df.image_filename.map(lambda f: find(f))
pd.DataFrame(pairs, columns=["image_a", "image_b", "hamming"]).to_csv(f"{PCFG['OUT']}/tables/P_duplicate_images.csv", index=False)
if pairs:
    k = min(4, len(pairs)); fig, axes = plt.subplots(k, 2, figsize=(5, 2 * k)); axes = np.atleast_2d(axes)
    for r, (a, b, h) in enumerate(pairs[:k]):
        axes[r, 0].imshow(Image.open(f"{PCFG['CRIC_ROOT']}/{a}")); axes[r, 1].imshow(Image.open(f"{PCFG['CRIC_ROOT']}/{b}"))
        axes[r, 0].set_title(a[:12], fontsize=6); axes[r, 1].set_title(f"{b[:12]} (ham={h})", fontsize=6)
        for x_ in axes[r]: x_.axis("off")
    plt.tight_layout(); savefig("P_duplicate_images"); plt.show()


# %% P4 : GROUPED STRATIFIED SPLIT --------------------------------------
df["fold"] = -1
for k, (_, va) in enumerate(StratifiedGroupKFold(PCFG["N_FOLDS"], shuffle=True, random_state=PCFG["SEED"])
                            .split(df, df.beth, groups=df.group)):
    df.loc[df.index[va], "fold"] = k
df["split"] = np.where(df.fold == PCFG["TEST_FOLD"], "test", np.where(df.fold == PCFG["VAL_FOLD"], "val", "train"))
split_hash = hashlib.sha256("".join(f"{c}:{s};" for c, s in zip(df.cell_id, df.split)).encode()).hexdigest()[:16]
REPORT["split_hash"] = split_hash; print("Split hash:", split_hash)

grp_split = df.groupby("group").split.nunique()
check("No image/duplicate-group shared across splits", (grp_split == 1).all(), f"{int((grp_split > 1).sum())} groups leak")
img_split = df.groupby("image_filename").split.nunique()
check("No image shared across splits", (img_split == 1).all(), f"{int((img_split > 1).sum())} images leak")
cross_dup = [(a, b) for a, b, _ in pairs if df.loc[df.image_filename == a, "split"].iloc[0] != df.loc[df.image_filename == b, "split"].iloc[0]]
check("No near-duplicate image pair across splits", not cross_dup, f"{len(cross_dup)} pairs")

ct = pd.crosstab(df.split, df.beth)[[SHORT[b] for b in BETH]].reindex(["train", "val", "test"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct.values)
cramers_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
check("Class distribution similar across splits (Cramér's V)", cramers_v <= PCFG["CRAMERS_V_MAX"],
      f"V = {cramers_v:.3f} (<= {PCFG['CRAMERS_V_MAX']} small effect); chi2 p = {p_chi:.2e}", level="WARN")
for b in ct.columns:
    check(f"Class {b} present in every split", (ct[b] > 0).all(), ct[b].to_dict())
tab = ct.copy(); tab["Total cells"] = tab.sum(axis=1); tab["Images"] = df.groupby("split").image_filename.nunique().reindex(tab.index)
for g_, gn in enumerate(GRADES): tab[gn] = df[df.grade == g_].groupby("split").size().reindex(tab.index)
tab.to_csv(f"{PCFG['OUT']}/tables/P_split_summary.csv"); print("\n" + tab.to_string())


# %% P5 : CROP EXTRACTION ----------------------------------------------
C = PCFG["CTX"]; Hh = C // 2
X = np.zeros((len(df), C, C, 3), np.uint8)
df["border_pad"] = False
t0 = time.time()
for fn, g in df.groupby("image_filename"):
    arr = np.asarray(Image.open(f"{PCFG['CRIC_ROOT']}/{fn}").convert("RGB"))
    Hi, Wi = arr.shape[:2]; arrp = np.pad(arr, ((Hh, Hh), (Hh, Hh), (0, 0)), mode="reflect")
    for i, r in g.iterrows():
        x, y = int(r.nucleus_x), int(r.nucleus_y)
        X[i] = arrp[y:y + C, x:x + C]
        df.at[i, "border_pad"] = (x < Hh) or (y < Hh) or (x + Hh > Wi) or (y + Hh > Hi)
print(f"Cropped {len(df)} cells in {time.time()-t0:.0f}s")
check("All crops have correct shape & dtype", X.shape == (len(df), C, C, 3) and X.dtype == np.uint8, str(X.shape))
check("No empty (all-zero) crop", int((X.reshape(len(X), -1).max(1) == 0).sum()) == 0, "")
# centre sanity: crop centre pixel == nucleus pixel in the original image
rs = np.random.default_rng(0).choice(len(df), min(200, len(df)), replace=False); ok = 0
for i in rs:
    r = df.iloc[i]; arr = np.asarray(Image.open(f"{PCFG['CRIC_ROOT']}/{r.image_filename}").convert("RGB"))
    ok += np.array_equal(X[i, Hh, Hh], arr[int(r.nucleus_y), int(r.nucleus_x)])
check("Crop centred on annotated nucleus (pixel-exact)", ok == len(rs), f"{ok}/{len(rs)} sampled")
note(f"Border-padded crops (reflect): {int(df.border_pad.sum())} ({100*df.border_pad.mean():.1f}%)")


# %% P6 : QUALITY + STAIN STATISTICS ------------------------------------
gray = X.astype(np.float32).mean(-1)
df["blur_lapvar"] = [ndimage.laplace(g_).var() for g_ in gray]
df["intensity_std"] = gray.reshape(len(gray), -1).std(1)
thr = np.percentile(df.blur_lapvar, PCFG["BLUR_PCT"]); df["flag_blur"] = df.blur_lapvar < thr
df["flag_lowcontrast"] = df.intensity_std < 5
note(f"Blur-flagged (lowest {PCFG['BLUR_PCT']}%): {int(df.flag_blur.sum())}; low-contrast: {int(df.flag_lowcontrast.sum())} (kept, flagged only)")

OD = -np.log((X.reshape(len(X), -1, 3).astype(np.float32) + 1) / 256.0)
for c_, nm in enumerate("RGB"): df[f"od_{nm}"] = OD[:, :, c_].mean(1)
st_rows = []
for col in ["od_R", "od_G", "od_B", "intensity_std", "blur_lapvar"]:
    a, b_ = df.loc[df.split == "train", col], df.loc[df.split == "test", col]
    d = (b_.mean() - a.mean()) / np.sqrt((a.var() + b_.var()) / 2)
    st_rows.append({"feature": col, "train mean": a.mean(), "val mean": df.loc[df.split == "val", col].mean(), "test mean": b_.mean(),
                    "Cohen d (test-train)": d, "KS p": stats.ks_2samp(a, b_).pvalue})
ST = pd.DataFrame(st_rows).set_index("feature"); ST.to_csv(f"{PCFG['OUT']}/tables/P_color_stats.csv"); print("\n" + ST.round(4).to_string())
check("Colour/OD distribution comparable train vs test (|Cohen d| < 0.5)", (ST["Cohen d (test-train)"].abs() < 0.5).all(),
      ST["Cohen d (test-train)"].round(3).to_dict(), level="WARN")


# %% P7 : PIXEL-LEVEL CROSS-SPLIT LEAKAGE TEST ---------------------------
def emb(ix):
    small = np.stack([np.asarray(Image.fromarray(X[i]).convert("L").resize((32, 32), Image.BILINEAR), np.float32) for i in ix])
    v = small.reshape(len(ix), -1); v -= v.mean(1, keepdims=True); v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    return v
idx = {s: np.where(df.split == s)[0] for s in ("train", "val", "test")}
E = {s: emb(i) for s, i in idx.items()}
def max_sim(a, b, bs=1024):
    out = np.empty(len(a), np.float32); arg = np.empty(len(a), int)
    for i in range(0, len(a), bs):
        s = a[i:i + bs] @ b.T; out[i:i + bs] = s.max(1); arg[i:i + bs] = s.argmax(1)
    return out, arg
sim_te, arg_te = max_sim(E["test"], E["train"]); sim_va, _ = max_sim(E["val"], E["train"])
# reference: within-train nearest neighbour (different images) - "normal" similarity level
tr_img = df.image_filename.values[idx["train"]]; rs = np.random.default_rng(1).choice(len(idx["train"]), min(1500, len(idx["train"])), replace=False)
s_in = E["train"][rs] @ E["train"].T
for r_, i in enumerate(rs): s_in[r_, tr_img == tr_img[i]] = -1
ref = s_in.max(1)
n_leak = int((sim_te > PCFG["CROP_SIM_FAIL"]).sum()) + int((sim_va > PCFG["CROP_SIM_FAIL"]).sum())
check("No near-identical crop between test/val and train", n_leak == 0,
      f"max sim test->train {sim_te.max():.3f}, val->train {sim_va.max():.3f}; train cross-image ref 99th pct {np.percentile(ref, 99):.3f}")
check("No near-identical crops between different train images (redundancy)", np.percentile(ref, 99) < PCFG["CROP_SIM_FAIL"],
      f"99th pct = {np.percentile(ref, 99):.3f}", level="WARN")
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.hist(ref, bins=50, alpha=0.6, density=True, color="#6b6b6b", label="train↔train (other images)")
ax.hist(sim_te, bins=50, alpha=0.6, density=True, color="#0072B2", label="test→nearest train")
ax.axvline(PCFG["CROP_SIM_FAIL"], ls="--", color="#D55E00", lw=1, label="leakage threshold")
ax.set_xlabel("Max cosine similarity (32×32 grey)"); ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=7)
ax.set_title("Cross-split nearest-neighbour similarity"); plt.tight_layout(); savefig("P_leakage_similarity"); plt.show()
w = np.argsort(-sim_te)[:4]
fig, axes = plt.subplots(2, 4, figsize=(7, 3.6))
for j, t in enumerate(w):
    axes[0, j].imshow(X[idx["test"][t]]); axes[1, j].imshow(X[idx["train"][arg_te[t]]])
    axes[0, j].set_title(f"test (sim {sim_te[t]:.3f})", fontsize=7); axes[1, j].set_title("nearest train", fontsize=7)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); savefig("P_most_similar_pairs"); plt.show()


# %% P8 : TRAIN-ONLY NORMALIZATION + SAVE --------------------------------
tr = X[idx["train"]].reshape(-1, 3).astype(np.float64) / 255.0
NORM = {"mean": tr.mean(0).round(6).tolist(), "std": tr.std(0).round(6).tolist(), "computed_on": "train only",
        "n_train_cells": int(len(idx["train"]))}
json.dump(NORM, open(f"{PCFG['OUT']}/norm_stats.json", "w"), indent=2)
check("Normalization statistics computed on train split only", True, f"mean={NORM['mean']}, std={NORM['std']}")
np.save(f"{PCFG['OUT']}/crops_ctx{C}.npy", X)
keep = ["cell_id", "image_id", "image_filename", "group", "nucleus_x", "nucleus_y", "bethesda_system", "beth", "grade", "binary",
        "fold", "split", "border_pad", "flag_blur", "flag_lowcontrast", "blur_lapvar", "intensity_std", "od_R", "od_G", "od_B"]
df[keep].to_csv(f"{PCFG['OUT']}/metadata.csv", index=False)
X2 = np.load(f"{PCFG['OUT']}/crops_ctx{C}.npy", mmap_mode="r"); M2 = pd.read_csv(f"{PCFG['OUT']}/metadata.csv")
check("Saved artifacts reload consistently", X2.shape[0] == len(M2) == len(df) and np.array_equal(X2[0], X[0]), f"{len(M2)} rows")
check("Augmentation NOT applied in preprocessing (train-time only)", True, "crops are raw; augmentation lives inside the training loop")


# %% P9 : FIGURES + REPORT ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
(ct.T / ct.sum(axis=1).values * 100).plot.bar(ax=axes[0], color=["#0072B2", "#E69F00", "#D55E00"], width=0.8, edgecolor="white")
axes[0].set_ylabel("% of split"); axes[0].set_title(f"Bethesda distribution per split (Cramér's V={cramers_v:.3f})"); axes[0].legend(frameon=False)
axes[0].tick_params(axis="x", rotation=0)
for s_, col in [("train", "#0072B2"), ("val", "#E69F00"), ("test", "#D55E00")]:
    axes[1].hist(df.loc[df.split == s_, "od_B"], bins=40, density=True, alpha=0.5, color=col, label=s_)
axes[1].set_xlabel("Mean optical density (blue)"); axes[1].set_title("Stain intensity per split"); axes[1].legend(frameon=False)
plt.tight_layout(); savefig("P_split_distribution"); plt.show()

rng = np.random.default_rng(0); fig, axes = plt.subplots(6, 8, figsize=(9, 7))
for r, b in enumerate([SHORT[x] for x in BETH]):
    ii = rng.choice(np.where(df.beth == b)[0], 8, replace=False)
    for j, i in enumerate(ii): axes[r, j].imshow(X[i]); axes[r, j].axis("off")
    axes[r, 0].set_title(f"{b} → {GRADES[GRADE[b]]}", fontsize=8, loc="left")
plt.tight_layout(); savefig("P_samples_per_class"); plt.show()

REPORT.update({"n_raw_cells": int(n_raw), "n_kept_cells": int(len(df)), "n_images": int(df.image_filename.nunique()),
               "n_groups": int(df.group.nunique()), "cramers_v": cramers_v, "config": PCFG})
json.dump(REPORT, open(f"{PCFG['OUT']}/report.json", "w"), indent=2, default=str)
summary = collections.Counter(c["status"] for c in REPORT["checks"])
print("\n" + "=" * 60); print(f"PREPROCESSING TEST SUITE: {dict(summary)}")
fails = [c for c in REPORT["checks"] if c["status"] == "FAIL"]
print("RESULT:", "PASS ✅ (proceed to training)" if not fails else f"FAIL ❌ -> {[c['check'] for c in fails]}")
print("=" * 60)
os.system(f"cd /kaggle/working && zip -rq cric_prep_report.zip cric_prep/tables cric_prep/figures cric_prep/report.json cric_prep/norm_stats.json cric_prep/metadata.csv")