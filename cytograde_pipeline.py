# =====================================================================
#  CytoGrade — training and evaluation pipeline (CRIC + Herlev)
#  Reproduces Tables 1-10 and Figures 2-10 of the manuscript.
#  Requires the output of 01_cric_preprocessing (cric_prep/) and the Herlev dataset.
#  Kaggle: GPU T4, Internet ON. One "# %% CELL" per notebook cell.
#  Long runs: commit, then add this notebook's previous version as input; finished runs are restored and skipped.
# =====================================================================

# %% CELL 1 : CONFIG -----------------------------------------------------
CFG = dict(
    PREP_DIR = None,          # None = auto-detect "cric_prep" (metadata.csv + crops_ctx192.npy) under /kaggle/input
    NORM = "imagenet",        # "imagenet" (recommended for pretrained backbones) | "train" (norm_stats.json)
    OUT = "/kaggle/working/cg2_out", SEED = 42, CLASSES = ["Normal", "Low-grade", "High-grade"],

    # ---- dual field-of-view ----
    CTX_CROP = 192,            # context: whole cell + cytoplasm (N:C ratio)
    DET_CROP = 96,             # detail: nucleus chromatin (center of context)
    IMG_SIZE = 256,            # each view is resized to 256x256
    FOV = "dual",              # "dual" | "ctx" | "det"  (ablation)

    # ---- model / training ----
    BACKBONE = "efficientnet_b3", LR = 2e-4, WD = 0.05,
    EPOCHS = 50, PATIENCE = 8, WARMUP = 2, BATCH = 32, EMA = 0.999,
    GRAD_CKPT = True,          # gradient checkpointing: avoids OOM on a T4 (~20% slower)
    HEAD = "ordinal", UNDER_W = 1.0,
    AMPMIX = True, AMPMIX_P = 0.5, AMPMIX_R = 0.1,
    SCALE_AUG = True, SCALE_P = 0.5, SCALE_RANGE = (0.8, 1.25),
    TWO_VIEW = True, LAMBDA_CONS = 0.5,
    LAMBDA_FM = 0.5, FM_NAME = "owkin/phikon",

    # ---- evaluation ----
    TTA = True, CONFORMAL_ALPHA = 0.10, ECE_BINS = 15, N_BOOT = 1000,

    # ---- run control ----
    RUN_ABLATIONS = False,     # legacy switch, kept False (no ablation in the manuscript)
)
import os, shutil
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # must be set before CUDA initialises
for d in ("tables", "figures", "runs"): os.makedirs(f"{CFG['OUT']}/{d}", exist_ok=True)
_restored = []
for dp, dn, fn in os.walk("/kaggle/input"):                    # previous notebook version or an uploaded runs/ folder
    if os.path.basename(dp) == "runs" and any(os.path.exists(f"{dp}/{d}/preds.npz") for d in dn):
        shutil.copytree(dp, f"{CFG['OUT']}/runs", dirs_exist_ok=True); _restored += [d for d in dn if os.path.exists(f"{dp}/{d}/preds.npz")]
print("Restored runs:", sorted(set(_restored)) if _restored else "NONE (all runs will be trained)")
print("Output:", CFG["OUT"])


# %% CELL 2 : IMPORTS + METRICS + POST-HOC + STATS (numpy) --------------
import json, math, time, copy, gc, random, collections
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, precision_score)
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = CFG["CLASSES"]; K = len(CLASSES)
TAB, FIG, RUNS = (f"{CFG['OUT']}/{d}" for d in ("tables", "figures", "runs"))
CCOL = {"Normal": "#0072B2", "Low-grade": "#E69F00", "High-grade": "#D55E00"}
MCOL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "savefig.dpi": 300, "savefig.bbox": "tight"})

def savefig(name):
    for ext in ("png", "pdf"): plt.savefig(f"{FIG}/{name}.{ext}")
    print(f"   figure -> {name}")

def savetab(df, name, title):
    df.to_csv(f"{TAB}/{name}.csv")
    try: df.to_latex(f"{TAB}/{name}.tex", float_format="%.2f")
    except Exception: pass
    print(f"\n{title}\n" + df.to_string())

def _sig(z): return 1 / (1 + np.exp(-z))
def probs_from_logits(logits, head, T=1.0):
    """logits: [N, C] or [n_tta, N, C] (TTA -> probability average)."""
    if logits.ndim == 3: return np.mean([probs_from_logits(l, head, T) for l in logits], 0)
    z = logits / T
    if head == "ordinal":
        s = _sig(z); s1 = s[:, 0]; s2 = np.minimum(s[:, 1], s1); p = np.stack([1 - s1, s1 - s2, s2], 1)
    else:
        z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return np.clip(p, 1e-7, 1)

def ece_score(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1); e = 0.0; b = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def full_metrics(p, y):
    pred = p.argmax(1); cm = confusion_matrix(y, pred, labels=list(range(K)))
    m = dict(macro_f1=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
             bal_acc=balanced_accuracy_score(y, pred), acc=accuracy_score(y, pred),
             precision=precision_score(y, pred, average="macro", zero_division=0), mcc=matthews_corrcoef(y, pred),
             qwk=cohen_kappa_score(y, pred, weights="quadratic"), ord_mae=np.abs(pred - y).mean())
    try: m["auc"] = roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError: m["auc"] = np.nan
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp; tn = cm.sum() - tp - fn - fp
        m[f"sens_{c}"] = tp / max(tp + fn, 1); m[f"spec_{c}"] = tn / max(tn + fp, 1)
        m[f"f1_{c}"] = 2 * tp / max(2 * tp + fp + fn, 1)
    yb, pb = (y > 0).astype(int), 1 - p[:, 0]; prb = (pb >= 0.5).astype(int)
    m["bin_acc"] = (prb == yb).mean()
    m["bin_sens"] = ((prb == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)
    m["bin_spec"] = ((prb == 0) & (yb == 0)).sum() / max((yb == 0).sum(), 1)
    m["bin_f1"] = f1_score(yb, prb)
    try: m["bin_auc"] = roc_auc_score(yb, pb)
    except ValueError: m["bin_auc"] = np.nan
    m["under_grade"] = (pred < y).mean(); m["over_grade"] = (pred > y).mean()
    m["high_to_low"] = cm[2, 1] / max(cm[2].sum(), 1)
    m["ece"] = ece_score(p, y, CFG["ECE_BINS"]); m["brier"] = ((p - np.eye(K)[y]) ** 2).sum(1).mean()
    m["nll"] = -np.log(p[np.arange(len(y)), y]).mean()
    return {k: float(v) for k, v in m.items()}

def fit_temperature(logits, y, head):
    Ts = np.exp(np.linspace(np.log(0.1), np.log(5.0), 100))
    return float(Ts[int(np.argmin([-np.log(probs_from_logits(logits, head, T)[np.arange(len(y)), y]).mean() for T in Ts]))])

def conformal_threshold(p_cal, y_cal, alpha):
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]; n = len(s)
    return float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
def conformal_sets(p, qhat):
    S = p >= 1 - qhat; S[np.arange(len(p)), p.argmax(1)] = True; return S
def conformal_metrics(S, y):
    size = S.sum(1)
    return dict(coverage=float(S[np.arange(len(y)), y].mean()), avg_size=float(size.mean()),
                referral=float((size > 1).mean()),
                singleton_acc=float(S[np.arange(len(y)), y][size == 1].mean()) if (size == 1).any() else np.nan)

def grouped_boot_ci(fn, p, y, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(np.arange(len(y))).groupby(groups).indices; keys = np.array(list(g)); v = []
    for _ in range(n_boot):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        try: v.append(fn(p[idx], y[idx]))
        except ValueError: pass
    return np.percentile(v, [2.5, 97.5])
def grouped_boot_delta(fn, pa, pb, y, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(np.arange(len(y))).groupby(groups).indices; keys = np.array(list(g)); d = []
    for _ in range(n_boot):
        idx = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        try: d.append(fn(pa[idx], y[idx]) - fn(pb[idx], y[idx]))
        except ValueError: pass
    return np.percentile(d, [2.5, 97.5])
def mcnemar_p(pa, pb, y):
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y; b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj
MF = {"Macro-F1": lambda p, y: f1_score(y, p.argmax(1), average="macro"),
      "Bal.Acc": lambda p, y: balanced_accuracy_score(y, p.argmax(1)),
      "AUC": lambda p, y: roc_auc_score(y, p, multi_class="ovr"),
      "QWK": lambda p, y: cohen_kappa_score(y, p.argmax(1), weights="quadratic"),
      "AUC (Normal vs Abn.)": lambda p, y: roc_auc_score((y > 0).astype(int), 1 - p[:, 0])}
print("Utilities ready.")


# %% CELL 3 : LOAD PREPROCESSED DATA (cric_prep) ------------------------
import torch, torch.nn as nn, torch.nn.functional as F, timm
from timm.data import resolve_data_config
from tqdm.auto import tqdm
device = torch.device("cuda")
def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); os.environ["PYTHONHASHSEED"] = str(s)
seed_everything(CFG["SEED"])

PREP = CFG["PREP_DIR"]
if PREP is None:
    for dp, dn, fn in os.walk("/kaggle/input"):
        if "metadata.csv" in fn and any(f.startswith("crops_ctx") for f in fn): PREP = dp; break
    if PREP is None and os.path.exists("/kaggle/working/cric_prep/metadata.csv"): PREP = "/kaggle/working/cric_prep"
assert PREP, "cric_prep not found: add the output of 01_cric_preprocessing as an input"
print("Preprocessed data:", PREP)
REP = json.load(open(f"{PREP}/report.json"))
fails = [c["check"] for c in REP["checks"] if c["status"] == "FAIL"]
assert not fails, f"Preprocessing checks failed: {fails}"
print(f"Preprocessing report: {len(REP['checks'])} checks, 0 FAIL | split hash {REP['split_hash']}")

meta = pd.read_csv(f"{PREP}/metadata.csv")
Xc = np.load(f"{PREP}/crops_ctx{CFG['CTX_CROP']}.npy")
assert len(Xc) == len(meta)
LABEL_MAP = {"Negative for intraepithelial lesion": 0, "ASC-US": 1, "LSIL": 1, "ASC-H": 2, "HSIL": 2, "SCC": 2}
BETH_ORDER = ["Negative for intraepithelial lesion", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
BETH_SHORT = {"Negative for intraepithelial lesion": "NILM"}
assert (meta.bethesda_system.map(LABEL_MAP) == meta.grade).all()
meta["label"] = meta["grade"]
SPLIT = {s_: np.where(meta.split == s_)[0] for s_ in ("train", "val", "test")}
for a_ in SPLIT:
    for b_ in SPLIT:
        if a_ < b_: assert not set(meta.group.values[SPLIT[a_]]) & set(meta.group.values[SPLIT[b_]]), "LEAKAGE"
print("Leakage re-check passed:", {s_: f"{len(i)} cells / {meta.image_filename.values[i].size and len(set(meta.image_filename.values[i]))} images" for s_, i in SPLIT.items()})

X = torch.from_numpy(Xc).permute(0, 3, 1, 2).contiguous().to(device)
Y = meta.label.values; G = meta.group.values          # bootstrap: image/duplicate-group level
NORM_STATS = json.load(open(f"{PREP}/norm_stats.json"))
print("GPU tensor:", tuple(X.shape), f"{X.numel()/1e9:.2f} GB")

rows = []
for name, ix in [("Train", SPLIT["train"]), ("Validation", SPLIT["val"]), ("Test", SPLIT["test"])]:
    rows.append({"Split": name, "Images": len(set(meta.image_filename.values[ix])), "Cells": len(ix),
                 **{BETH_SHORT.get(b, b): int((meta.bethesda_system.values[ix] == b).sum()) for b in BETH_ORDER},
                 **{c: int((Y[ix] == k).sum()) for k, c in enumerate(CLASSES)}})
savetab(pd.DataFrame(rows).set_index("Split"), "T01_dataset_split", "TABLE 1. CRIC split after cleaning (image/duplicate-group level, no leakage)")

d0 = (CFG["CTX_CROP"] - CFG["DET_CROP"]) // 2; rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
for j, b in enumerate(BETH_ORDER):
    i = rng.choice(np.where(meta.bethesda_system.values == b)[0])
    axes[0, j].imshow(Xc[i]); axes[0, j].add_patch(plt.Rectangle((d0, d0), CFG["DET_CROP"], CFG["DET_CROP"], fill=False, ec="#D55E00", lw=1.2))
    axes[1, j].imshow(Xc[i, d0:d0 + CFG["DET_CROP"], d0:d0 + CFG["DET_CROP"]])
    axes[0, j].set_title(f"{BETH_SHORT.get(b, b)} → {CLASSES[LABEL_MAP[b]]}", fontsize=7)
for a__ in axes.ravel(): a__.axis("off")
axes[0, 0].text(-0.08, 0.5, "Context\n(192 px)", transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=8)
axes[1, 0].text(-0.08, 0.5, "Detail\n(96 px)", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=8)
plt.tight_layout(); savefig("F01_dual_fov_samples"); plt.show()


# %% CELL 4 : GPU AUGMENTATION (geometric on context, then derive both views)
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], device=device)
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED); _LOG_ADJ = math.log(1e-6)
_BLUR = (torch.tensor([1., 2., 1.], device=device)[:, None] * torch.tensor([1., 2., 1.], device=device)[None]) / 16
_BLUR = _BLUR.expand(3, 1, 3, 3).contiguous()

def hed_jitter(x, sigma=0.05):
    rgb = x.permute(0, 2, 3, 1).clamp_min(1e-6); hed = ((torch.log(rgb) / _LOG_ADJ) @ _HED_FROM_RGB).clamp_min(0)
    a = torch.empty(x.shape[0], 1, 1, 3, device=x.device).uniform_(1 - sigma, 1 + sigma)
    b = torch.empty(x.shape[0], 1, 1, 3, device=x.device).uniform_(-sigma, sigma)
    return torch.exp(((hed * a + b) * _LOG_ADJ) @ _RGB_FROM_HED).clamp(0, 1).permute(0, 3, 1, 2)

def ampmix(x, p, r):
    B, _, Hh, Ww = x.shape; m = torch.rand(B, device=x.device) < p
    if not m.any(): return x
    Fx = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)); amp, pha = Fx.abs(), Fx.angle()
    perm = torch.randperm(B, device=x.device); h, w = max(1, int(Hh * r)), max(1, int(Ww * r))
    lam = torch.rand(B, 1, 1, 1, device=x.device) * m.view(B, 1, 1, 1)
    sl = (slice(None), slice(None), slice(Hh // 2 - h, Hh // 2 + h), slice(Ww // 2 - w, Ww // 2 + w))
    a2 = amp.clone(); a2[sl] = (1 - lam) * amp[sl] + lam * amp[perm][sl]
    return torch.fft.ifft2(torch.fft.ifftshift(a2 * torch.exp(1j * pha), dim=(-2, -1))).real.clamp(0, 1)

def _affine(x, m, th):
    g = F.affine_grid(th, x[m].shape, align_corners=False)
    x[m] = F.grid_sample(x[m], g, mode="bilinear", padding_mode="reflection", align_corners=False)

def dihedral(x, t):                     # TTA: t in 0..7
    x = torch.rot90(x, t % 4, (2, 3)); return x.flip(3) if t >= 4 else x

def make_views(xu8, size, mean, std, train, cfg, tta=0):
    x = xu8.float() / 255.; B = x.shape[0]; dev = x.device
    if tta: x = dihedral(x, tta)
    if train:
        x = hed_jitter(x)
        if cfg.get("AMPMIX"): x = ampmix(x, cfg["AMPMIX_P"], cfg["AMPMIX_R"])
        x = x.clone()
        m = torch.rand(B, device=dev) < 0.5; x[m] = x[m].flip(3)
        m = torch.rand(B, device=dev) < 0.5; x[m] = x[m].flip(2)
        m = torch.rand(B, device=dev) < 0.5; k = torch.randint(1, 4, (B,), device=dev)
        for kk in (1, 2, 3):
            mk = m & (k == kk)
            if mk.any(): x[mk] = torch.rot90(x[mk], kk, (2, 3))
        m = torch.rand(B, device=dev) < (cfg["SCALE_P"] if cfg.get("SCALE_AUG") else 0.0)
        if m.any():
            n = int(m.sum()); s = torch.empty(n, device=dev).uniform_(*cfg["SCALE_RANGE"])
            th = torch.zeros(n, 2, 3, device=dev); th[:, 0, 0] = 1 / s; th[:, 1, 1] = 1 / s; _affine(x, m, th)
        m = torch.rand(B, device=dev) < 0.5
        if m.any():                                                    # small translation keeps the nucleus centred
            n = int(m.sum()); ang = torch.empty(n, device=dev).uniform_(-15, 15) * math.pi / 180
            th = torch.zeros(n, 2, 3, device=dev)
            th[:, 0, 0] = torch.cos(ang); th[:, 0, 1] = -torch.sin(ang); th[:, 0, 2] = torch.empty(n, device=dev).uniform_(-.04, .04)
            th[:, 1, 0] = torch.sin(ang); th[:, 1, 1] = torch.cos(ang);  th[:, 1, 2] = torch.empty(n, device=dev).uniform_(-.04, .04)
            _affine(x, m, th)
        m = torch.rand(B, device=dev) < 0.3
        if m.any():
            n = int(m.sum())
            x[m] = (x[m] * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
                    + torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)).clamp(0, 1)
        m = torch.rand(B, device=dev) < 0.1
        if m.any(): x[m] = F.conv2d(F.pad(x[m], (1, 1, 1, 1), mode="reflect"), _BLUR, groups=3)
    c0 = (x.shape[-1] - CFG["DET_CROP"]) // 2
    ctx = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    det = F.interpolate(x[:, :, c0:c0 + CFG["DET_CROP"], c0:c0 + CFG["DET_CROP"]], size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (ctx - mean) / std, (det - mean) / std


# %% CELL 5 : MODEL ------------------------------------------------------
class OrdinalHead(nn.Module):
    def __init__(self, d, K):
        super().__init__(); self.fc = nn.Linear(d, 1, bias=False); self.bias = nn.Parameter(torch.linspace(1.0, -1.0, K - 1))
    def forward(self, z): return self.fc(z) + self.bias

class DualFOVNet(nn.Module):
    """Shared backbone on context (cell + cytoplasm) and detail (nucleus) views, gated fusion, ordinal head."""
    def __init__(self, cfg, fm_dim=0, pretrained=True):
        super().__init__()
        self.fov, self.head_kind, self.img_size = cfg["FOV"], cfg["HEAD"], cfg["IMG_SIZE"]
        self.backbone = timm.create_model(cfg["BACKBONE"], pretrained=pretrained, num_classes=0)
        if cfg.get("GRAD_CKPT"):
            try: self.backbone.set_grad_checkpointing(True)
            except Exception as e: print("   grad checkpointing unavailable:", e)
        d = self.backbone.num_features
        if self.fov == "dual":
            self.view_emb = nn.Parameter(torch.zeros(2, d))            # context/detail identity
            self.gate = nn.Sequential(nn.Linear(2 * d, d // 4), nn.GELU(), nn.Linear(d // 4, d), nn.Sigmoid())
        self.neck = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3))
        self.head = OrdinalHead(d, K) if cfg["HEAD"] == "ordinal" else nn.Linear(d, K)
        self.fm_proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, fm_dim)) if fm_dim else None
    def encode(self, x):
        fmap = self.backbone.forward_features(x); return self.backbone.forward_head(fmap, pre_logits=True), fmap
    def forward(self, ctx, det, return_all=False):
        if self.fov == "dual":
            z2, fmap2 = self.encode(torch.cat([ctx, det]))
            zc, zd = z2.chunk(2); zc = zc + self.view_emb[0]; zd = zd + self.view_emb[1]
            g = self.gate(torch.cat([zc, zd], 1)); z = g * zc + (1 - g) * zd; fmaps = fmap2.chunk(2)
        else:
            z, fm = self.encode(ctx if self.fov == "ctx" else det); g = None; fmaps = (fm,)
        logits = self.head(self.neck(z)).float()
        return (logits, z, g, fmaps) if return_all else (logits, z)

def torch_probs(logits, head):
    if head == "ordinal":
        s = torch.sigmoid(logits); s1 = s[:, 0]; s2 = torch.minimum(s[:, 1], s1); return torch.stack([1 - s1, s1 - s2, s2], 1)
    return logits.softmax(1)

def model_norm(model):
    if CFG["NORM"] == "train":
        return (torch.tensor(NORM_STATS["mean"], device=device).view(1, 3, 1, 1), torch.tensor(NORM_STATS["std"], device=device).view(1, 3, 1, 1))
    dc = resolve_data_config({}, model=model.backbone)
    return (torch.tensor(dc["mean"], device=device).view(1, 3, 1, 1), torch.tensor(dc["std"], device=device).view(1, 3, 1, 1))

FM_FEATS = {}
@torch.no_grad()
def fm_features():
    if "cric" in FM_FEATS: return FM_FEATS["cric"]
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(CFG["FM_NAME"]).to(device).eval()
    m_ = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1); s_ = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    gc.collect(); torch.cuda.empty_cache()
    print(f"   GPU memory before Phikon: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated "
          f"(expected ~1.3 GB = crop tensor; restart the session if it is much higher)")
    out = []
    for i in range(0, len(X), 64):
        ctx, _ = make_views(X[i:i + 64], 224, m_, s_, False, {})
        with torch.autocast("cuda", dtype=torch.float16): out.append(bb(pixel_values=ctx).last_hidden_state[:, 0].float())
    FM_FEATS["cric"] = F.normalize(torch.cat(out), dim=1); del bb; torch.cuda.empty_cache(); return FM_FEATS["cric"]


# %% CELL 6 : TRAINER ----------------------------------------------------
def js_div(p1, p2, eps=1e-8):
    m = 0.5 * (p1 + p2)
    return 0.5 * ((p1 * (torch.log(p1 + eps) - torch.log(m + eps))).sum(1) + (p2 * (torch.log(p2 + eps) - torch.log(m + eps))).sum(1)).mean()

def sup_loss(logits, y, cw, head, under_w):
    if head == "ordinal":
        t = torch.stack([(y > 0), (y > 1)], 1).float(); w = torch.where(t > 0, under_w, 1.0) * cw[y][:, None]
        return F.binary_cross_entropy_with_logits(logits, t, weight=w, reduction="sum") / w.sum()
    return F.cross_entropy(logits, y, weight=cw)

@torch.no_grad()
def predict(model, idx, tta=False, bs=128, feats=False, Xsrc=None):
    """idx=None -> pura Xsrc. Xsrc=None -> CRIC tensor X."""
    Xsrc = X if Xsrc is None else Xsrc; idx = np.arange(len(Xsrc)) if idx is None else idx
    model.eval(); mean, std = model_norm(model); size = getattr(model, "img_size", CFG["IMG_SIZE"]); outs, Z, Gt = [], [], []
    for t in (range(8) if tta else [0]):
        L = []
        for i in range(0, len(idx), bs):
            ctx, det = make_views(Xsrc[torch.as_tensor(idx[i:i + bs], device=device)], size, mean, std, False, {}, tta=t)
            with torch.autocast("cuda", dtype=torch.float16):
                lg, z, g, _ = model(ctx, det, return_all=True)
            L.append(torch.nan_to_num(lg.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50).cpu())
            if feats and t == 0:
                Z.append(z.float().cpu()); Gt.append(g.float().mean(1).cpu() if g is not None else torch.zeros(len(z)))
        outs.append(torch.cat(L).numpy())
    lg = np.stack(outs) if tta else outs[0]
    return (lg, torch.cat(Z).numpy(), torch.cat(Gt).numpy()) if feats else lg

def train_model(name, cfg):
    rdir = f"{RUNS}/{name}"; os.makedirs(rdir, exist_ok=True)
    if os.path.exists(f"{rdir}/preds.npz") and os.path.exists(f"{rdir}/best.pt"):
        print(f"[{name}] already done -> load"); return dict(np.load(f"{rdir}/preds.npz", allow_pickle=True))
    seed_everything(cfg.get("SEED", CFG["SEED"])); t_start = time.time()
    T_FM = fm_features() if cfg["LAMBDA_FM"] > 0 else None
    model = DualFOVNet(cfg, fm_dim=T_FM.shape[1] if T_FM is not None else 0).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    mean, std = model_norm(model); size = cfg["IMG_SIZE"]
    Yt = torch.as_tensor(Y, device=device); tr = torch.as_tensor(SPLIT["train"], device=device)
    cw = torch.tensor(len(SPLIT["train"]) / (K * np.bincount(Y[SPLIT["train"]], minlength=K)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    spe = len(SPLIT["train"]) // cfg["BATCH"]; total = cfg["EPOCHS"] * spe; warm = cfg["WARMUP"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda"); best, bad, best_state, log, step = -1, 0, None, [], 0; y_va = Y[SPLIT["val"]]
    for ep in range(cfg["EPOCHS"]):
        t0 = time.time(); model.train(); agg = collections.defaultdict(float)
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(spe):
            bi = perm[i * cfg["BATCH"]:(i + 1) * cfg["BATCH"]]; yb = Yt[bi]
            nv = 2 if cfg["TWO_VIEW"] else 1
            views = [make_views(X[bi], size, mean, std, True, cfg) for _ in range(nv)]   # fp32 (FFT)
            with torch.autocast("cuda", dtype=torch.float16):
                outs = [model(*v) for v in views]
            loss = sum(sup_loss(lg, yb, cw, cfg["HEAD"], cfg["UNDER_W"]) for lg, _ in outs) / nv; agg["sup"] += loss.item()
            if nv == 2 and cfg["LAMBDA_CONS"] > 0:
                lc = js_div(torch_probs(outs[0][0], cfg["HEAD"]), torch_probs(outs[1][0], cfg["HEAD"]))
                loss = loss + cfg["LAMBDA_CONS"] * lc; agg["cons"] += lc.item()
            if T_FM is not None:
                lf = sum(1 - F.cosine_similarity(model.fm_proj(z.float()), T_FM[bi], dim=1).mean() for _, z in outs) / nv
                loss = loss + cfg["LAMBDA_FM"] * lf; agg["fm"] += lf.item()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            step += 1; dec = min(cfg["EMA"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dec).add_(pm.detach(), alpha=1 - dec)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
        pv = probs_from_logits(predict(ema, SPLIT["val"]), cfg["HEAD"]); m = full_metrics(pv, y_va)
        log.append(dict(epoch=ep, **{k: v / spe for k, v in agg.items()}, **{f"val_{k}": v for k, v in m.items()}, sec=time.time() - t0))
        pd.DataFrame(log).to_csv(f"{rdir}/log.csv", index=False)
        imp = m["macro_f1"] > best
        if imp: best, bad, best_state = m["macro_f1"], 0, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        else: bad += 1
        print(f"[{name}] E{ep:02d} | " + " ".join(f"{k} {v/spe:.3f}" for k, v in agg.items()) +
              f" | val F1 {m['macro_f1']:.4f} QWK {m['qwk']:.4f} AUC {m['auc']:.4f} binAUC {m['bin_auc']:.4f} | {time.time()-t0:.0f}s {'*' if imp else ''}")
        if bad >= cfg["PATIENCE"]: print(f"   early stop E{ep}"); break
    ema.load_state_dict(best_state)
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}, f"{rdir}/best.pt")
    out = {"head": np.array(cfg["HEAD"]), "best_epoch": np.array(int(np.argmax([r["val_macro_f1"] for r in log]))),
           "train_min": np.array((time.time() - t_start) / 60),
           "val_logits": predict(ema, SPLIT["val"]), "test_logits": predict(ema, SPLIT["test"])}
    if cfg.get("TTA", CFG["TTA"]):
        out["val_logits_tta"] = predict(ema, SPLIT["val"], tta=True); out["test_logits_tta"] = predict(ema, SPLIT["test"], tta=True)
    np.savez(f"{rdir}/preds.npz", **out)
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, open(f"{rdir}/config.json", "w"), indent=2)
    print(f"[{name}] done in {(time.time()-t_start)/60:.1f} min, best epoch {int(out['best_epoch'])}")
    del model, ema, opt; gc.collect(); torch.cuda.empty_cache(); return out


# %% CELL 7 : RUN --------------------------------------------------------
def variant(**kw): return {**CFG, **kw}
OURS = "CytoGrade-v2"
EXPS = {OURS: variant()}
if CFG["RUN_ABLATIONS"]:
    EXPS.update({"Context view only": variant(FOV="ctx"), "Detail view only": variant(FOV="det"),
                 "w/o ordinal head": variant(HEAD="ce"), "w/o FM anchoring": variant(LAMBDA_FM=0.0),
                 "w/o domain randomization": variant(AMPMIX=False, SCALE_AUG=False, TWO_VIEW=False, LAMBDA_CONS=0.0)})
def preflight(cfg):
    """Run one real training step to measure peak GPU memory; halve the batch if it is too high."""
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    while True:
        m_ = opt_ = outs_ = vs_ = loss_ = None
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m_ = DualFOVNet(cfg, fm_dim=768 if cfg["LAMBDA_FM"] > 0 else 0).to(device).train(); mean_, std_ = model_norm(m_)
            ema_bytes = sum(p.numel() for p in m_.parameters()) * 4 / 1e9
            opt_ = torch.optim.AdamW(m_.parameters(), lr=1e-8); sc_ = torch.amp.GradScaler("cuda")
            bi = torch.as_tensor(SPLIT["train"][:cfg["BATCH"]], device=device)
            vs_ = [make_views(X[bi], cfg["IMG_SIZE"], mean_, std_, True, cfg) for _ in range(2 if cfg["TWO_VIEW"] else 1)]
            with torch.autocast("cuda", dtype=torch.float16): outs_ = [m_(*v) for v in vs_]
            loss_ = sum(o[0].float().mean() + (m_.fm_proj(o[1].float()).mean() if m_.fm_proj is not None else 0) for o in outs_)
            sc_.scale(loss_).backward(); sc_.step(opt_); torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9 + ema_bytes + 0.5
            if peak > 0.88 * total: raise torch.OutOfMemoryError(f"peak {peak:.1f} GB too close to {total:.1f} GB")
            print(f"Preflight OK: batch {cfg['BATCH']}, est. peak {peak:.1f} / {total:.1f} GB"); return cfg
        except torch.OutOfMemoryError as e:
            print(f"Preflight: batch {cfg['BATCH']} too large ({str(e)[:60]}) -> halving")
            cfg = {**cfg, "BATCH": cfg["BATCH"] // 2}
            if cfg["BATCH"] < 8: raise RuntimeError("Batch 8 does not fit in memory: use IMG_SIZE 224")
        finally:
            del m_, opt_, outs_, vs_, loss_; gc.collect(); torch.cuda.empty_cache()

RES = {}; T_ALL = time.time()
for n, c in EXPS.items():
    key_ = n.replace("/", "_").replace(" ", "_")
    if not os.path.exists(f"{RUNS}/{key_}/preds.npz"): c = preflight(c); EXPS[n] = c
    RES[n] = train_model(key_, c); print(f"== elapsed {(time.time()-T_ALL)/60:.1f} min ==")


# %% CELL 8 : POST-HOC + MAIN TABLES -------------------------------------
y_val, y_te, g_te = Y[SPLIT["val"]], Y[SPLIT["test"]], G[SPLIT["test"]]
POST = {}
for n, r in RES.items():
    h = str(r["head"]); T = fit_temperature(r["val_logits"], y_val, h)
    d = dict(T=T, val=probs_from_logits(r["val_logits"], h, T), test_raw=probs_from_logits(r["test_logits"], h, 1.0),
             test=probs_from_logits(r["test_logits"], h, T))
    if "test_logits_tta" in r:
        Tt = fit_temperature(r["val_logits_tta"], y_val, h)
        d.update(T_tta=Tt, val_tta=probs_from_logits(r["val_logits_tta"], h, Tt), test_tta=probs_from_logits(r["test_logits_tta"], h, Tt))
    POST[n] = d
P_MAIN = POST[OURS].get("test_tta", POST[OURS]["test"]); P_VAL = POST[OURS].get("val_tta", POST[OURS]["val"])
print("Temperature:", {n: round(v["T"], 2) for n, v in POST.items()})

KEY = ["macro_f1", "weighted_f1", "bal_acc", "acc", "precision", "mcc", "auc", "qwk", "ord_mae",
       "bin_acc", "bin_sens", "bin_spec", "bin_f1", "bin_auc", "ece", "brier", "nll"]
NICE = {"macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1", "bal_acc": "Bal.Acc", "acc": "Acc", "precision": "Precision",
        "mcc": "MCC", "auc": "AUC", "qwk": "QWK", "ord_mae": "Ord.MAE↓", "bin_acc": "Acc(N vs A)", "bin_sens": "Sens(abn)",
        "bin_spec": "Spec(norm)", "bin_f1": "F1(N vs A)", "bin_auc": "AUC(N vs A)", "ece": "ECE↓", "brier": "Brier↓", "nll": "NLL↓"}
def fmt(rows):
    df = pd.DataFrame(rows).set_index("Model")
    for c in df.columns: df[c] = df[c].round(3) if c in ("ord_mae", "brier", "nll") else (100 * df[c]).round(2)
    return df.rename(columns=NICE)

rows = [{"Model": f"{OURS} (single pass)", **{k: full_metrics(POST[OURS]["test"], y_te)[k] for k in KEY}}]
if "test_tta" in POST[OURS]: rows.append({"Model": f"{OURS} (+TTA×8)", **{k: full_metrics(POST[OURS]["test_tta"], y_te)[k] for k in KEY}})
savetab(fmt(rows), "T02_main_test", "TABLE 2. CRIC held-out test (3-class grading + Normal vs Abnormal screening)")

# ---- validation-tuned decision thresholds (test never used for tuning) ----
def ord_pred(p, t1, t2):
    """grade = [P(y>Normal) > t1] + [P(y>Low) > t2]  (cumulative ordinal decision)."""
    return np.where(1 - p[:, 0] > t1, np.where(p[:, 2] > t2, 2, 1), 0)
def pred_metrics(pred, y):
    cm_ = confusion_matrix(y, pred, labels=list(range(K)))
    out = dict(macro_f1=f1_score(y, pred, average="macro"), bal_acc=balanced_accuracy_score(y, pred), acc=accuracy_score(y, pred),
               mcc=matthews_corrcoef(y, pred), qwk=cohen_kappa_score(y, pred, weights="quadratic"),
               bin_sens=((pred > 0) & (y > 0)).sum() / max((y > 0).sum(), 1), bin_spec=((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    for k, c in enumerate(CLASSES): out[f"sens_{c}"] = cm_[k, k] / max(cm_[k].sum(), 1)
    return out
grid = np.round(np.arange(0.20, 0.81, 0.02), 2)
best_t, best_f = (0.5, 0.5), -1
for t1 in grid:
    for t2 in grid:
        f = f1_score(y_val, ord_pred(P_VAL, t1, t2), average="macro")
        if f > best_f: best_f, best_t = f, (t1, t2)
THR = best_t; PRED_T = ord_pred(P_MAIN, *THR)
print(f"Validation-tuned thresholds: t1={THR[0]}, t2={THR[1]} (val macro-F1 {100*best_f:.2f})")
tt = pd.DataFrame([{"Decision rule": "argmax (default)", **pred_metrics(P_MAIN.argmax(1), y_te)},
                   {"Decision rule": f"val-tuned thresholds t1={THR[0]}, t2={THR[1]}", **pred_metrics(PRED_T, y_te)}]).set_index("Decision rule")
savetab((100 * tt).round(2), "T02b_thresholds", "TABLE 2b. Test performance: argmax vs validation-tuned ordinal thresholds")

ci = []
for mn, fn in MF.items():
    lo, hi = grouped_boot_ci(fn, P_MAIN, y_te, g_te, CFG["N_BOOT"])
    ci.append({"Metric": mn, "Value": round(100 * fn(P_MAIN, y_te), 2), "95% CI (image-level bootstrap)": f"[{100*lo:.2f}, {100*hi:.2f}]"})
savetab(pd.DataFrame(ci).set_index("Metric"), "T03_ci", "TABLE 3. 95% confidence intervals (main model)")

m = full_metrics(P_MAIN, y_te)
savetab((100 * pd.DataFrame({c: {"Sensitivity": m[f"sens_{c}"], "Specificity": m[f"spec_{c}"], "F1": m[f"f1_{c}"]}
                              for c in CLASSES}).T).round(2), "T04_per_class", "TABLE 4. Per-class performance")

# Bethesda subtype analysis: which subtype goes to which grade
pred = P_MAIN.argmax(1); beth = meta.bethesda_system.values[SPLIT["test"]]
bt = pd.DataFrame([{"Bethesda": BETH_SHORT.get(b, b), "Target grade": CLASSES[LABEL_MAP[b]], "n": int((beth == b).sum()),
                    **{f"→ {c} %": 100 * np.mean(pred[beth == b] == k) for k, c in enumerate(CLASSES)},
                    "Correct %": 100 * np.mean(pred[beth == b] == LABEL_MAP[b])} for b in BETH_ORDER]).set_index("Bethesda")
savetab(bt.round(2), "T05_bethesda_subtype", "TABLE 5. Bethesda subtype → predicted grade (test)")



# %% CELL 9 : ABLATION + RELIABILITY + COMPLEXITY TABLES -----------------
if len(RES) > 1:
    ab = [{"Model": n, **{k: full_metrics(POST[n]["test"], y_te)[k] for k in ["macro_f1", "bal_acc", "auc", "qwk", "bin_auc", "ece"]}} for n in RES]
    AB = fmt(ab); savetab(AB, "T07_ablation", "TABLE 7. Ablation (single pass, test)")
    st = []
    for n in RES:
        if n == OURS: continue
        lo, hi = grouped_boot_delta(MF["Macro-F1"], POST[OURS]["test"], POST[n]["test"], y_te, g_te, CFG["N_BOOT"])
        st.append({"Variant": n, "ΔMacro-F1 (full − variant)": round(100 * (MF["Macro-F1"](POST[OURS]["test"], y_te) - MF["Macro-F1"](POST[n]["test"], y_te)), 2),
                   "95% CI": f"[{100*lo:.2f}, {100*hi:.2f}]", "McNemar p": mcnemar_p(POST[OURS]["test"], POST[n]["test"], y_te)})
    ST = pd.DataFrame(st); ST["McNemar p (Holm)"] = holm(ST["McNemar p"]).round(4)
    savetab(ST.set_index("Variant"), "T08_ablation_stats", "TABLE 8. Ablation significance")

qhat = conformal_threshold(P_VAL, y_val, CFG["CONFORMAL_ALPHA"])
rel = []
for lab, p in [("Raw (T=1)", POST[OURS]["test_raw"]), ("Temperature-scaled", POST[OURS]["test"])] + \
              ([("Temp. + TTA", POST[OURS]["test_tta"])] if "test_tta" in POST[OURS] else []):
    mm = full_metrics(p, y_te); rel.append({"Setting": lab, "ECE %": 100 * mm["ece"], "Brier": mm["brier"], "NLL": mm["nll"]})
cmx = conformal_metrics(conformal_sets(P_MAIN, qhat), y_te)
rel.append({"Setting": f"Conformal (α={CFG['CONFORMAL_ALPHA']})", "Coverage %": 100 * cmx["coverage"], "Avg set size": cmx["avg_size"],
            "Referral %": 100 * cmx["referral"], "Accuracy on auto-decided %": 100 * cmx["singleton_acc"]})
savetab(pd.DataFrame(rel).set_index("Setting").round(3), "T09_reliability", "TABLE 9. Calibration & conformal referral (test)")

from torch.utils.flop_counter import FlopCounterMode
@torch.no_grad()
def complexity(cfg, name):
    m_ = DualFOVNet(cfg, pretrained=False).to(device).eval(); tot = sum(p.numel() for p in m_.parameters())
    s = cfg["IMG_SIZE"]; a = torch.randn(1, 3, s, s, device=device)
    with FlopCounterMode(display=False) as fc: m_(a, a)
    def lat(bs, it=40):
        x = torch.randn(bs, 3, s, s, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(8): m_(x, x)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(it): m_(x, x)
            torch.cuda.synchronize()
        return (time.time() - t0) / it
    g1, g32 = lat(1), lat(32); mc = m_.float().cpu(); xc = torch.randn(1, 3, s, s)
    for _ in range(2): mc(xc, xc)
    t0 = time.time()
    for _ in range(8): mc(xc, xc)
    cpu = (time.time() - t0) / 8; del m_, mc; torch.cuda.empty_cache()
    return {"Model": name, "Params (M)": tot / 1e6, "GFLOPs": fc.get_total_flops() / 1e9, "GPU ms/cell (bs1)": 1000 * g1,
            "GPU cells/s (bs32)": 32 / g32, "CPU ms/cell": 1000 * cpu, "Size MB (fp16)": tot * 2 / 1e6}
cx = [complexity(EXPS[OURS], OURS)]
cx.append({**cx[0], "Model": OURS + " +TTA×8", "GFLOPs": 8 * cx[0]["GFLOPs"], "GPU ms/cell (bs1)": 8 * cx[0]["GPU ms/cell (bs1)"],
           "GPU cells/s (bs32)": cx[0]["GPU cells/s (bs32)"] / 8, "CPU ms/cell": 8 * cx[0]["CPU ms/cell"]})
savetab(pd.DataFrame(cx).set_index("Model").round(2), "T10_complexity", "TABLE 10. Inference complexity (T4 fp16 / Kaggle CPU)")


# %% CELL 10 : FIGURES ---------------------------------------------------
# F02 learning curves
lg = pd.read_csv(f"{RUNS}/{OURS}/log.csv")
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
axes[0].plot(lg.epoch, lg.sup, color="#0072B2", label="supervised")
if "cons" in lg: axes[0].plot(lg.epoch, lg.cons, color="#E69F00", label="consistency")
if "fm" in lg: axes[0].plot(lg.epoch, lg.fm, color="#009E73", label="FM anchoring")
axes[0].set_title("Training losses"); axes[0].legend(frameon=False)
for c, col in [("val_macro_f1", "#0072B2"), ("val_qwk", "#D55E00")]: axes[1].plot(lg.epoch, 100 * lg[c], color=col, label=c[4:])
axes[1].axvline(lg.val_macro_f1.idxmax(), ls=":", color=MUTED); axes[1].set_title("Validation (EMA)"); axes[1].legend(frameon=False)
for c, col in [("val_auc", "#0072B2"), ("val_bin_auc", "#009E73")]: axes[2].plot(lg.epoch, 100 * lg[c], color=col, label=c[4:])
axes[2].set_title("Validation AUC"); axes[2].legend(frameon=False)
for a in axes: a.set_xlabel("Epoch")
plt.tight_layout(); savefig("F02_learning_curves"); plt.show()

# F03 confusion matrix (3-class + binary)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, (yt, pr, labs, title) in zip(axes, [(y_te, P_MAIN.argmax(1), CLASSES, "3-class grading"),
                                             ((y_te > 0).astype(int), (1 - P_MAIN[:, 0] >= 0.5).astype(int), ["Normal", "Abnormal"], "Screening")]):
    cmn = confusion_matrix(yt, pr, normalize="true"); cmc = confusion_matrix(yt, pr)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{100*cmn[i,j]:.1f}%\n({cmc[i,j]})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)), labs); ax.set_yticks(range(len(labs)), labs); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
plt.tight_layout(); savefig("F03_confusion"); plt.show()

# F04 ROC + PR
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
for k, c in enumerate(CLASSES):
    yk = (y_te == k).astype(int); fpr, tpr, _ = roc_curve(yk, P_MAIN[:, k]); pr_, rc_, _ = precision_recall_curve(yk, P_MAIN[:, k])
    axes[0].plot(fpr, tpr, color=CCOL[c], label=f"{c} ({roc_auc_score(yk, P_MAIN[:, k]):.3f})")
    axes[1].plot(rc_, pr_, color=CCOL[c], label=f"{c} (AP {average_precision_score(yk, P_MAIN[:, k]):.3f})")
yb = (y_te > 0).astype(int); fpr, tpr, thr = roc_curve(yb, 1 - P_MAIN[:, 0])
axes[2].plot(fpr, tpr, color="#0072B2", label=f"AUC {roc_auc_score(yb, 1-P_MAIN[:, 0]):.3f}")
j = np.argmax(tpr - fpr); axes[2].scatter(fpr[j], tpr[j], color="#D55E00", zorder=3, s=25, label=f"Youden: sens {tpr[j]:.2f}, spec {1-fpr[j]:.2f}")
for a in (axes[0], axes[2]): a.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[0].set_title("ROC (one-vs-rest)"); axes[1].set_title("Precision-recall"); axes[2].set_title("Normal vs Abnormal ROC")
for a in axes: a.legend(frameon=False, fontsize=7, loc="lower right" if a is not axes[1] else "lower left")
plt.tight_layout(); savefig("F04_roc_pr"); plt.show()

# F05 Bethesda subtype → predicted grade (stacked)
fig, ax = plt.subplots(figsize=(6.2, 3.0)); left = np.zeros(len(bt))
for k, c in enumerate(CLASSES):
    v = bt[f"→ {c} %"].values; ax.barh(bt.index, v, left=left, color=CCOL[c], edgecolor="white", height=0.7, label=f"Predicted {c}")
    for i, (l, w) in enumerate(zip(left, v)):
        if w > 9: ax.text(l + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=7, color="white")
    left += v
ax.invert_yaxis(); ax.set_xlabel("% of cells"); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_title("Bethesda subtype → predicted grade (test)"); ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout(); savefig("F05_bethesda_subtype"); plt.show()

# F06 reliability (raw vs temperature) + conformal coverage vs alpha
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
def rel_bars(ax, p, y, t):
    conf, pr = p.max(1), p.argmax(1); b = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        mm_ = (conf > lo) & (conf <= hi)
        if mm_.sum() >= 5: mids.append(conf[mm_].mean()); accs.append((pr[mm_] == y[mm_]).mean())
    ax.bar(mids, accs, width=0.08, color="#0072B2", alpha=0.85, edgecolor="white"); ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(f"{t} · ECE {100*ece_score(p, y):.2f}%", fontsize=9)
rel_bars(axes[0], POST[OURS]["test_raw"], y_te, "Raw"); rel_bars(axes[1], P_MAIN, y_te, "Calibrated")
al = np.linspace(0.02, 0.30, 15); cov, size = [], []
for a in al:
    cm_ = conformal_metrics(conformal_sets(P_MAIN, conformal_threshold(P_VAL, y_val, a)), y_te); cov.append(100 * cm_["coverage"]); size.append(cm_["avg_size"])
axes[2].plot(100 * (1 - al), cov, color="#0072B2", marker="o", ms=3, label="Empirical coverage")
axes[2].plot([70, 98], [70, 98], ls="--", color=MUTED, lw=1, label="Target"); axes[2].set_xlabel("Target coverage (%)"); axes[2].set_ylabel("Coverage (%)")
ax2 = axes[2].twinx(); ax2.plot(100 * (1 - al), size, color="#D55E00", marker="s", ms=3); ax2.set_ylabel("Avg set size", color="#D55E00"); ax2.grid(False)
axes[2].set_title("Conformal prediction"); axes[2].legend(frameon=False, fontsize=7, loc="upper left")
plt.tight_layout(); savefig("F06_calibration_conformal"); plt.show()

# F07 risk-coverage (selective prediction)
u = 1 - P_MAIN.max(1); o = np.argsort(u, kind="stable"); wrong = (P_MAIN.argmax(1) != y_te)[o]; n_ = np.arange(1, len(o) + 1)
cov_ = n_ / len(o); risk = np.cumsum(wrong) / n_; f1c = [f1_score(y_te[o[:k]], P_MAIN.argmax(1)[o[:k]], average="macro") for k in (cov_ * len(o)).astype(int)[::20]]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
axes[0].plot(100 * cov_, 100 * risk, color="#0072B2"); axes[0].invert_xaxis(); axes[0].set_xlabel("Coverage (%) ← more referred")
axes[0].set_ylabel("Error rate (%)"); axes[0].set_title(f"Risk-coverage (AURC {100*risk.mean():.2f})")
axes[1].plot(100 * cov_[::20], 100 * np.array(f1c), color="#D55E00"); axes[1].invert_xaxis(); axes[1].set_xlabel("Coverage (%)")
axes[1].set_ylabel("Macro-F1 (%)"); axes[1].set_title("Macro-F1 on auto-decided cells")
for c in (0.9, 0.8):
    k = int(c * len(o)); axes[1].axvline(100 * c, ls=":", color=MUTED)
    axes[1].text(100 * c, axes[1].get_ylim()[0] + 1, f"{100*f1_score(y_te[o[:k]], P_MAIN.argmax(1)[o[:k]], average='macro'):.1f}", fontsize=7, ha="right")
plt.tight_layout(); savefig("F07_selective_prediction"); plt.show()

# F08 per-class bars
fig, ax = plt.subplots(figsize=(5.2, 3.0)); xx = np.arange(K); w = 0.27
for j, (lab, key, col) in enumerate([("Sensitivity", "sens", "#0072B2"), ("Specificity", "spec", "#E69F00"), ("F1", "f1", "#009E73")]):
    v = [100 * m[f"{key}_{c}"] for c in CLASSES]; ax.bar(xx + (j - 1) * w, v, w * 0.92, color=col, edgecolor="white", label=lab)
    for x_, v_ in zip(xx + (j - 1) * w, v): ax.text(x_, v_ + 1, f"{v_:.1f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(xx, CLASSES); ax.set_ylim(0, 108); ax.set_ylabel("%"); ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=7)
ax.set_title("Per-class performance (test)"); plt.tight_layout(); savefig("F08_per_class"); plt.show()

# F09 ablation
if len(RES) > 1:
    fig, ax = plt.subplots(figsize=(6.6, 3.0)); xx = np.arange(len(AB))
    ax.bar(xx - 0.2, AB["Macro-F1"], 0.38, color="#0072B2", edgecolor="white", label="Macro-F1")
    ax.bar(xx + 0.2, AB["QWK"], 0.38, color="#D55E00", edgecolor="white", label="QWK")
    ax.set_xticks(xx, AB.index, rotation=20, ha="right"); ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.legend(frameon=False)
    ax.set_title("Ablation (test)"); plt.tight_layout(); savefig("F09_ablation"); plt.show()


# %% CELL 11 : EXPLAINABILITY (dual-FOV Grad-CAM, gate, t-SNE) ----------
sd = torch.load(f"{RUNS}/{OURS}/best.pt", map_location=device)
xm = DualFOVNet(EXPS[OURS], fm_dim=sd["fm_proj.2.weight"].shape[0] if "fm_proj.2.weight" in sd else 0, pretrained=False).to(device)
xm.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); xm.eval(); mean, std = model_norm(xm)

def gradcam(i):
    ctx, det = make_views(X[torch.as_tensor([i], device=device)], CFG["IMG_SIZE"], mean, std, False, {})
    xm.zero_grad(set_to_none=True)
    if xm.fov == "dual":
        z2, fmap = xm.encode(torch.cat([ctx, det])); fmap.retain_grad()
        zc, zd = z2.chunk(2); zc = zc + xm.view_emb[0]; zd = zd + xm.view_emb[1]
        g = xm.gate(torch.cat([zc, zd], 1)); z = g * zc + (1 - g) * zd
    else:
        z, fmap = xm.encode(ctx if xm.fov == "ctx" else det); fmap.retain_grad(); g = None
    logits = xm.head(xm.neck(z)).float()
    p = torch_probs(logits, EXPS[OURS]["HEAD"]); c = int(p.argmax(1)); torch.log(p[0, c] + 1e-8).backward()
    cams = []
    for f_, gr_ in zip(fmap.detach().chunk(fmap.shape[0]), fmap.grad.chunk(fmap.shape[0])):
        w = gr_.mean((2, 3), keepdim=True); cm_ = F.relu((f_ * w).sum(1, keepdim=True))
        cm_ = F.interpolate(cm_, size=(CFG["IMG_SIZE"],) * 2, mode="bilinear", align_corners=False)[0, 0]
        cams.append(((cm_ - cm_.min()) / (cm_.max() - cm_.min() + 1e-8)).detach().cpu().numpy())
    return cams, c, float(g.detach().mean()) if g is not None else np.nan

te = SPLIT["test"]; pr_ = P_MAIN.argmax(1); rng = np.random.default_rng(3); pick = []
for k in range(K):
    ok = np.where((pr_ == k) & (y_te == k))[0]; pick += [te[i] for i in rng.choice(ok, min(2, len(ok)), replace=False)]
bad = np.where(pr_ < y_te)[0]; pick += [te[i] for i in rng.choice(bad, min(2, len(bad)), replace=False)]
fig, axes = plt.subplots(len(pick), 4, figsize=(6.2, 1.6 * len(pick)))
for r, i in enumerate(pick):
    cams, c, gv = gradcam(i); ctx_img = np.asarray(Image.fromarray(Xc[i]).resize((CFG["IMG_SIZE"],) * 2))
    det_img = np.asarray(Image.fromarray(Xc[i, d0:d0 + CFG["DET_CROP"], d0:d0 + CFG["DET_CROP"]]).resize((CFG["IMG_SIZE"],) * 2, Image.BICUBIC))
    for j, (img, cam) in enumerate([(ctx_img, None), (ctx_img, cams[0]), (det_img, None), (det_img, cams[-1])]):
        axes[r, j].imshow(img)
        if cam is not None: axes[r, j].imshow(cam, cmap="jet", alpha=0.4)
        axes[r, j].set_xticks([]); axes[r, j].set_yticks([]); axes[r, j].grid(False)
    axes[r, 0].set_ylabel(f"{'✓' if c == Y[i] else '✗'} T:{CLASSES[Y[i]][:4]}\nP:{CLASSES[c][:4]}  g={gv:.2f}", rotation=0, ha="right", va="center", fontsize=7)
for j, t in enumerate(["Context", "Context CAM", "Detail", "Detail CAM"]): axes[0, j].set_title(t, fontsize=8)
plt.tight_layout(); savefig("F10_gradcam_dual_fov"); plt.show()

_, Z, GATE = predict(xm, te, feats=True)
if EXPS[OURS]["FOV"] == "dual":
    fig, ax = plt.subplots(figsize=(4.2, 3.0)); data = [GATE[y_te == k] for k in range(K)]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
    for b_, c in zip(bp["boxes"], CLASSES): b_.set_facecolor(CCOL[c]); b_.set_alpha(0.75)
    ax.set_xticks([1, 2, 3], CLASSES); ax.axhline(0.5, ls="--", color=MUTED, lw=1); ax.set_ylabel("Gate g (context weight)")
    ax.set_title(f"Context vs detail reliance (Kruskal p={stats.kruskal(*data).pvalue:.1e})", fontsize=9)
    plt.tight_layout(); savefig("F11_gate_per_class"); plt.show()

from sklearn.manifold import TSNE
emb = TSNE(2, perplexity=30, init="pca", random_state=CFG["SEED"]).fit_transform(Z)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for k, c in enumerate(CLASSES):
    mm_ = y_te == k; axes[0].scatter(emb[mm_, 0], emb[mm_, 1], s=6, color=CCOL[c], alpha=0.7, edgecolor="none", label=c)
sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=6, c=1 - P_MAIN.max(1), cmap="viridis", edgecolor="none")
plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02, label="Uncertainty (1 - max p)")
axes[0].legend(frameon=False, markerscale=2.5); axes[0].set_title("t-SNE: true grade"); axes[1].set_title("t-SNE: uncertainty")
for a in axes: a.set_xticks([]); a.set_yticks([]); a.grid(False)
plt.tight_layout(); savefig("F12_tsne"); plt.show()


# %% CELL 12 : PACKAGE ---------------------------------------------------
os.system(f"cd {CFG['OUT']} && zip -rq /kaggle/working/cytograde_v2_results.zip tables figures && ls -lh /kaggle/working/cytograde_v2_results.zip")
print("Download: /kaggle/working/cytograde_v2_results.zip (tables + figures + predictions + model weights)")



# =====================================================================
#  PHASE 2 : SAME-SPLIT BASELINES + STATISTICS + LEAKAGE STUDY + HERLEV
# =====================================================================

# %% CELL 13 : SAME-SPLIT BASELINES (same pipeline, softmax head) -------
BASE_COMMON = dict(FOV="ctx", HEAD="ce", UNDER_W=1.0, AMPMIX=False, SCALE_AUG=False, TWO_VIEW=False, LAMBDA_CONS=0.0,
                   LAMBDA_FM=0.0, TTA=False, BATCH=64, GRAD_CKPT=True)
BASELINES = {
    "EfficientNet-B0":       variant(**BASE_COMMON, BACKBONE="efficientnet_b0", LR=3e-4, IMG_SIZE=224),
    "ResNet-50":             variant(**BASE_COMMON, BACKBONE="resnet50", LR=1e-4, IMG_SIZE=224),
    "Swin-T":                variant(**BASE_COMMON, BACKBONE="swin_tiny_patch4_window7_224", LR=5e-5, IMG_SIZE=224),
    "ViT-B/16":              variant(**BASE_COMMON, BACKBONE="vit_base_patch16_224", LR=5e-5, IMG_SIZE=224),
    "EfficientNet-B3 (plain)": variant(**BASE_COMMON, BACKBONE="efficientnet_b3", LR=2e-4, IMG_SIZE=256),
}
CFG["TTA_MAIN"] = CFG["TTA"]
for n, c in BASELINES.items():
    key_ = "BASE_" + n.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    if not os.path.exists(f"{RUNS}/{key_}/preds.npz"): c = preflight(c); BASELINES[n] = c
    RES[n] = train_model(key_, c); print(f"== elapsed {(time.time()-T_ALL)/60:.1f} min ==")
ALL = {OURS: EXPS[OURS], **BASELINES}

def post_of(r):
    h = str(r["head"]); T = fit_temperature(r["val_logits"], y_val, h)
    return dict(T=T, head=h, val=probs_from_logits(r["val_logits"], h, T), test=probs_from_logits(r["test_logits"], h, T))
for n in BASELINES: POST[n] = post_of(RES[n])


# %% CELL 14 : CRIC COMPARISON + STATISTICS + EFFICIENCY -----------------
CMP_KEY = ["macro_f1", "bal_acc", "acc", "mcc", "auc", "qwk", "ord_mae", "bin_sens", "bin_spec", "bin_auc", "ece"]
CX = {n: complexity(c, n) for n, c in ALL.items()}
rows = []
for n in ALL:
    mm = full_metrics(POST[n]["test"], y_te)                       # single pass for every model (fair)
    rows.append({"Model": n, **{k: mm[k] for k in CMP_KEY}})
T11 = fmt(rows)
T11["Params (M)"] = [round(CX[n]["Params (M)"], 2) for n in T11.index]
T11["GFLOPs"] = [round(CX[n]["GFLOPs"], 2) for n in T11.index]
T11["CPU ms/cell"] = [round(CX[n]["CPU ms/cell"], 1) for n in T11.index]
savetab(T11, "T11_same_split_comparison", "TABLE 11. CRIC test, SAME cleaned split, same pipeline (single pass, temperature-scaled)")

st = []
for b in BASELINES:
    pa, pb = POST[OURS]["test"], POST[b]["test"]
    f_lo, f_hi = grouped_boot_delta(MF["Macro-F1"], pa, pb, y_te, g_te, CFG["N_BOOT"])
    q_lo, q_hi = grouped_boot_delta(MF["QWK"], pa, pb, y_te, g_te, CFG["N_BOOT"])
    st.append({"Baseline": b, "ΔMacro-F1": round(100 * (MF["Macro-F1"](pa, y_te) - MF["Macro-F1"](pb, y_te)), 2),
               "ΔF1 95% CI": f"[{100*f_lo:.2f}, {100*f_hi:.2f}]", "ΔQWK": round(100 * (MF["QWK"](pa, y_te) - MF["QWK"](pb, y_te)), 2),
               "ΔQWK 95% CI": f"[{100*q_lo:.2f}, {100*q_hi:.2f}]", "McNemar p": mcnemar_p(pa, pb, y_te)})
T12 = pd.DataFrame(st); T12["McNemar p (Holm)"] = holm(T12["McNemar p"]).round(5)
savetab(T12.set_index("Baseline"), "T12_stats_vs_baselines", f"TABLE 12. {OURS} − baseline on CRIC test (image-group bootstrap CI, Holm-corrected McNemar)")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for i, n in enumerate(ALL):
    ax.scatter(CX[n]["GFLOPs"], T11.loc[n, "Macro-F1"], s=70 if n == OURS else 40, color=MCOL[i % len(MCOL)],
               edgecolor="white", zorder=3, marker="*" if n == OURS else "o")
    ax.annotate(n, (CX[n]["GFLOPs"], T11.loc[n, "Macro-F1"]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
ax.set_xscale("log"); ax.set_xlabel("GFLOPs per cell (log)"); ax.set_ylabel("Macro-F1 (%)"); ax.set_title("Accuracy vs compute (CRIC test)")
plt.tight_layout(); savefig("F13_efficiency_frontier"); plt.show()


# %% CELL 15 : LEAKAGE INFLATION STUDY (random cell split vs image-group split)
from sklearn.model_selection import train_test_split
SPLIT_GROUPED = SPLIT
rng_ = np.random.default_rng(CFG["SEED"]); allidx = np.arange(len(Y))
tr_r, rest = train_test_split(allidx, train_size=len(SPLIT["train"]), stratify=meta.bethesda_system, random_state=CFG["SEED"])
va_r, te_r = train_test_split(rest, train_size=len(SPLIT["val"]), stratify=meta.bethesda_system.values[rest], random_state=CFG["SEED"])
SPLIT_RANDOM = {"train": np.sort(tr_r), "val": np.sort(va_r), "test": np.sort(te_r)}
shared = np.isin(meta.group.values[SPLIT_RANDOM["test"]], meta.group.values[SPLIT_RANDOM["train"]]).mean()
print(f"Random cell split: {100*shared:.1f}% of test cells come from an image that is also in training")
SPLIT = SPLIT_RANDOM
r_leak = train_model("LEAK_EfficientNet-B0_random_cell_split", BASELINES["EfficientNet-B0"])
SPLIT = SPLIT_GROUPED
h_ = str(r_leak["head"]); yv_r, yt_r = Y[SPLIT_RANDOM["val"]], Y[SPLIT_RANDOM["test"]]
p_leak = probs_from_logits(r_leak["test_logits"], h_, fit_temperature(r_leak["val_logits"], yv_r, h_))
m_leak, m_grp = full_metrics(p_leak, yt_r), full_metrics(POST["EfficientNet-B0"]["test"], y_te)
LK = ["macro_f1", "acc", "auc", "qwk", "bin_auc"]
T13 = pd.DataFrame([{"Protocol": "Random cell-level split (common in literature)", "Test cells sharing an image with train %": 100 * shared, **{k: 100 * m_leak[k] for k in LK}},
                    {"Protocol": "Image/duplicate-group split (ours)", "Test cells sharing an image with train %": 0.0, **{k: 100 * m_grp[k] for k in LK}}]).set_index("Protocol")
T13.loc["Inflation (random − grouped)"] = T13.iloc[0] - T13.iloc[1]
savetab(T13.round(2).rename(columns=NICE), "T13_leakage_inflation", "TABLE 13. Leakage inflation: same model (EfficientNet-B0), two split protocols")
fig, ax = plt.subplots(figsize=(5.0, 3.0)); xx = np.arange(len(LK))
ax.bar(xx - 0.2, T13.iloc[0][LK], 0.38, color="#D55E00", edgecolor="white", label="Random cell split (leaky)")
ax.bar(xx + 0.2, T13.iloc[1][LK], 0.38, color="#0072B2", edgecolor="white", label="Image-group split (ours)")
ax.set_xticks(xx, [NICE[k] for k in LK]); ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title("Same model, different split protocol"); plt.tight_layout(); savefig("F14_leakage_inflation"); plt.show()


# %% CELL 16 : HERLEV LOAD (external, never used for training/selection) --
def find_herlev():
    for dp, dn, fn in os.walk("/kaggle/input"):
        low = [d.lower() for d in dn]
        if any("carcinoma" in d for d in low) and any("normal" in d for d in low) and any("dysplastic" in d for d in low): return dp
    return None
HERLEV_ROOT = find_herlev(); assert HERLEV_ROOT, "Herlev dataset not found: add herlev-cervical-cancer-dataset as an input"
print("Herlev:", HERLEV_ROOT)
def _pad_square(img):
    h, w = img.shape[:2]; s_ = max(h, w); pt, pl = (s_ - h) // 2, (s_ - w) // 2; pads = ((pt, s_ - h - pt), (pl, s_ - w - pl), (0, 0))
    try: return np.pad(img, pads, mode="reflect")
    except ValueError: return np.pad(img, pads, mode="edge")
def herlev_grade(folder):
    f = folder.lower(); return 0 if "normal" in f else 1 if ("light" in f or "mild" in f) else 2
HX, HY, HG, HSRC = [], [], [], []
for dp, _, fns in os.walk(HERLEV_ROOT):
    for f in sorted(fns):
        if not f.lower().endswith((".bmp", ".png", ".jpg", ".tif")) or "-d." in f.lower(): continue
        im = _pad_square(np.asarray(Image.open(os.path.join(dp, f)).convert("RGB")))
        HX.append(np.asarray(Image.fromarray(im).resize((CFG["CTX_CROP"],) * 2, Image.BILINEAR)))
        HY.append(herlev_grade(os.path.basename(dp))); HG.append(os.path.splitext(f)[0].rsplit("-", 1)[0]); HSRC.append(os.path.basename(dp))
X_HER = torch.from_numpy(np.stack(HX)).permute(0, 3, 1, 2).contiguous().to(device); Y_HER, G_HER = np.array(HY), np.array(HG)
print(f"Herlev: {len(Y_HER)} cells, {len(set(G_HER))} source IDs, grades {np.bincount(Y_HER).tolist()} | folders {pd.Series(HSRC).value_counts().to_dict()}")

def load_trained(n):
    key_ = n if n == OURS else "BASE_" + n.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    sd = torch.load(f"{RUNS}/{key_}/best.pt", map_location=device)
    m_ = DualFOVNet(ALL[n], fm_dim=sd["fm_proj.2.weight"].shape[0] if "fm_proj.2.weight" in sd else 0, pretrained=False).to(device)
    m_.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); return m_.eval()
HER = {}
for n in ALL:
    m_ = load_trained(n); HER[n] = probs_from_logits(predict(m_, None, Xsrc=X_HER), POST[n]["head"] if "head" in POST[n] else str(RES[n]["head"]), POST[n]["T"])
    del m_; torch.cuda.empty_cache()


# %% CELL 17 : HERLEV TABLES + STATISTICS --------------------------------
HK = ["macro_f1", "bal_acc", "acc", "auc", "qwk", "bin_sens", "bin_spec", "bin_auc", "under_grade", "high_to_low", "ece"]
rows = []
for n in ALL:
    mh, mc = full_metrics(HER[n], Y_HER), full_metrics(POST[n]["test"], y_te)
    rows.append({"Model": n, **{k: mh[k] for k in HK}, "drop_f1": mc["macro_f1"] - mh["macro_f1"]})
T14 = fmt(rows).rename(columns={"under_grade": "Under-grade%↓", "high_to_low": "High→Low%↓", "drop_f1": "F1 drop vs CRIC↓"})
savetab(T14, "T14_herlev_external", "TABLE 14. Herlev external validation (trained on CRIC only, zero-shot, CRIC-val temperature)")

st = []
for b in BASELINES:
    pa, pb = HER[OURS], HER[b]
    f_lo, f_hi = grouped_boot_delta(MF["Macro-F1"], pa, pb, Y_HER, G_HER, CFG["N_BOOT"])
    st.append({"Baseline": b, "ΔMacro-F1": round(100 * (MF["Macro-F1"](pa, Y_HER) - MF["Macro-F1"](pb, Y_HER)), 2),
               "ΔF1 95% CI": f"[{100*f_lo:.2f}, {100*f_hi:.2f}]", "ΔQWK": round(100 * (MF["QWK"](pa, Y_HER) - MF["QWK"](pb, Y_HER)), 2),
               "McNemar p": mcnemar_p(pa, pb, Y_HER)})
T15 = pd.DataFrame(st); T15["McNemar p (Holm)"] = holm(T15["McNemar p"]).round(5)
savetab(T15.set_index("Baseline"), "T15_herlev_stats", f"TABLE 15. {OURS} − baseline on Herlev (source-ID bootstrap CI)")

rel = []
for n in ALL:
    q = conformal_threshold(POST[n]["val"], y_val, CFG["CONFORMAL_ALPHA"])
    for setn, p, y in [("CRIC test", POST[n]["test"], y_te), ("Herlev", HER[n], Y_HER)]:
        cm_ = conformal_metrics(conformal_sets(p, q), y)
        rel.append({"Model": n, "Set": setn, "Coverage %": 100 * cm_["coverage"], "Avg set size": cm_["avg_size"],
                    "Referral %": 100 * cm_["referral"], "ECE %": 100 * ece_score(p, y)})
savetab(pd.DataFrame(rel).set_index(["Model", "Set"]).round(2), "T16_conformal_cric_vs_herlev",
        "TABLE 16. Conformal referral (α=0.10, calibrated on CRIC val): in-domain vs external")


# %% CELL 18 : HERLEV FIGURES -------------------------------------------
# F15 CRIC vs Herlev macro-F1 per model
fig, ax = plt.subplots(figsize=(6.6, 3.2)); xx = np.arange(len(ALL)); names = list(ALL)
ax.bar(xx - 0.2, [T11.loc[n, "Macro-F1"] for n in names], 0.38, color="#0072B2", edgecolor="white", label="CRIC test (in-domain)")
ax.bar(xx + 0.2, [T14.loc[n, "Macro-F1"] for n in names], 0.38, color="#D55E00", edgecolor="white", label="Herlev (external)")
for i, n in enumerate(names): ax.text(i + 0.2, T14.loc[n, "Macro-F1"] + 1.5, f"{T14.loc[n, 'Macro-F1']:.1f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(xx, names, rotation=20, ha="right"); ax.set_ylim(0, 100); ax.set_ylabel("Macro-F1 (%)"); ax.legend(frameon=False, fontsize=7)
ax.set_title("In-domain vs external performance"); plt.tight_layout(); savefig("F15_cric_vs_herlev"); plt.show()

# F16 Herlev confusion: best baseline vs ours
best_b = max(BASELINES, key=lambda b: MF["Macro-F1"](HER[b], Y_HER))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, n in zip(axes, [best_b, OURS]):
    cmn = confusion_matrix(Y_HER, HER[n].argmax(1), labels=[0, 1, 2], normalize="true"); cmc = confusion_matrix(Y_HER, HER[n].argmax(1), labels=[0, 1, 2])
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100*cmn[i,j]:.1f}%\n({cmc[i,j]})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{n} · Herlev", fontsize=9)
plt.tight_layout(); savefig("F16_herlev_confusion"); plt.show()

# F17 under-grading on Herlev
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(xx - 0.2, [T14.loc[n, "High→Low%↓"] for n in names], 0.38, color="#D55E00", edgecolor="white", label="High→Low")
ax.bar(xx + 0.2, [T14.loc[n, "Under-grade%↓"] for n in names], 0.38, color="#E69F00", edgecolor="white", label="Any under-grade")
ax.set_xticks(xx, names, rotation=20, ha="right"); ax.set_ylabel("% (lower is safer)"); ax.legend(frameon=False)
ax.set_title("Herlev: clinically dangerous under-grading"); plt.tight_layout(); savefig("F17_herlev_undergrading"); plt.show()

# F18 t-SNE CRIC test vs Herlev (ours)
xm2 = load_trained(OURS)
sub = np.random.default_rng(0).choice(SPLIT["test"], min(1200, len(SPLIT["test"])), replace=False)
_, zc, _ = predict(xm2, sub, feats=True); _, zh, _ = predict(xm2, None, feats=True, Xsrc=X_HER); del xm2; torch.cuda.empty_cache()
emb = TSNE(2, perplexity=30, init="pca", random_state=CFG["SEED"]).fit_transform(np.concatenate([zc, zh]))
yy = np.concatenate([Y[sub], Y_HER]); dom = np.r_[np.zeros(len(sub)), np.ones(len(Y_HER))]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for d, (lab, col) in enumerate([("CRIC", "#0072B2"), ("Herlev", "#D55E00")]):
    mm_ = dom == d; axes[0].scatter(emb[mm_, 0], emb[mm_, 1], s=6, color=col, alpha=0.6, edgecolor="none", label=lab)
for k, c in enumerate(CLASSES):
    for d, mk in enumerate(["o", "^"]):
        mm_ = (yy == k) & (dom == d)
        axes[1].scatter(emb[mm_, 0], emb[mm_, 1], s=7, marker=mk, color=CCOL[c], alpha=0.6, edgecolor="none", label=f"{c} ({'CRIC' if d == 0 else 'Herlev'})")
axes[0].set_title("t-SNE by dataset"); axes[1].set_title("t-SNE by grade (o CRIC, ^ Herlev)")
for a in axes: a.set_xticks([]); a.set_yticks([]); a.grid(False); a.legend(frameon=False, fontsize=6, markerscale=2)
plt.tight_layout(); savefig("F18_tsne_cric_vs_herlev"); plt.show()

os.system(f"cd {CFG['OUT']} && zip -rq /kaggle/working/cytograde_v2_phase2_results.zip tables figures && ls -lh /kaggle/working/cytograde_v2_phase2_results.zip")
print("Download: /kaggle/working/cytograde_v2_phase2_results.zip")




# =====================================================================
#  PHASE 3 : SEED ROBUSTNESS (seeds 1, 2) + DEEP-ENSEMBLE COMPARISON (CRIC + Herlev)
# =====================================================================

# %% CELL 19 : SEED ROBUSTNESS — TRAIN SEEDS 1 AND 2 ----------------------------------------
NEW_SEEDS = [1, 2]                                  # original seed 42 already trained
SEED_MODELS = [OURS, "EfficientNet-B3 (plain)", "ViT-B/16"]   # optional: add "EfficientNet-B0", "Swin-T" (+~1 h)
def run_key(n, seed):
    base = n if n == OURS else "BASE_" + n.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return base if seed == CFG["SEED"] else f"SEED{seed}_{base}"
SEED_RES = {}
for seed in NEW_SEEDS:
    for n in SEED_MODELS:
        c = {**ALL[n], "SEED": seed, "TTA": False}
        k = run_key(n, seed)
        if not os.path.exists(f"{RUNS}/{k}/preds.npz"): c = preflight(c)
        SEED_RES[(n, seed)] = (train_model(k, c), c)
        print(f"== seed {seed} | {n} done | elapsed {(time.time()-T_ALL)/60:.1f} min ==")
for n in SEED_MODELS: SEED_RES[(n, CFG["SEED"])] = (RES[n], ALL[n])


# %% CELL 20 : SEED-AGGREGATED RESULTS (CRIC + Herlev) ------------------
def eval_run(n, seed):
    r, c = SEED_RES[(n, seed)]; h = str(r["head"]); T = fit_temperature(r["val_logits"], y_val, h)
    pv, pt = probs_from_logits(r["val_logits"], h, T), probs_from_logits(r["test_logits"], h, T)
    sd = torch.load(f"{RUNS}/{run_key(n, seed)}/best.pt", map_location=device)
    m_ = DualFOVNet(c, fm_dim=sd["fm_proj.2.weight"].shape[0] if "fm_proj.2.weight" in sd else 0, pretrained=False).to(device)
    m_.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); m_.eval()
    ph = probs_from_logits(predict(m_, None, Xsrc=X_HER), h, T); del m_; torch.cuda.empty_cache()
    q = conformal_threshold(pv, y_val, CFG["CONFORMAL_ALPHA"])
    return dict(test=pt, her=ph, ref_cric=conformal_metrics(conformal_sets(pt, q), y_te)["referral"],
                ref_her=conformal_metrics(conformal_sets(ph, q), Y_HER)["referral"],
                cov_her=conformal_metrics(conformal_sets(ph, q), Y_HER)["coverage"])
SEEDS_ALL = [CFG["SEED"]] + NEW_SEEDS
EV = {(n, s_): eval_run(n, s_) for n in SEED_MODELS for s_ in SEEDS_ALL}

rows = []
for n in SEED_MODELS:
    for s_ in SEEDS_ALL:
        e = EV[(n, s_)]; mc, mh = full_metrics(e["test"], y_te), full_metrics(e["her"], Y_HER)
        rows.append({"Model": n, "Seed": s_, "CRIC F1": 100 * mc["macro_f1"], "CRIC QWK": 100 * mc["qwk"], "CRIC ECE": 100 * mc["ece"],
                     "Herlev F1": 100 * mh["macro_f1"], "Herlev QWK": 100 * mh["qwk"], "Herlev AUC": 100 * mh["auc"],
                     "Herlev ECE": 100 * mh["ece"], "Herlev High→Low": 100 * mh["high_to_low"],
                     "Referral CRIC %": 100 * e["ref_cric"], "Referral Herlev %": 100 * e["ref_her"], "Coverage Herlev %": 100 * e["cov_her"]})
PS = pd.DataFrame(rows); PS.to_csv(f"{TAB}/T17_per_seed.csv", index=False)
agg = PS.drop(columns="Seed").groupby("Model", sort=False).agg(["mean", "std"])
T17 = pd.DataFrame({c: agg[c]["mean"].map("{:.2f}".format) + " ± " + agg[c]["std"].map("{:.2f}".format) for c in agg.columns.levels[0]}
                   ).reindex(SEED_MODELS)[[c for c in PS.columns if c not in ("Model", "Seed")]]
savetab(T17, "T17_seed_mean_std", f"TABLE 17. Mean ± SD over {len(SEEDS_ALL)} seeds (same split)")

# paired by seed: ours - baseline
pr = []
for b in [m for m in SEED_MODELS if m != OURS]:
    for metric, key_, better in [("Herlev F1", "macro_f1", 1), ("Herlev ECE", "ece", -1), ("CRIC F1", "macro_f1", 1)]:
        d = []
        for s_ in SEEDS_ALL:
            pa = EV[(OURS, s_)]["her" if "Herlev" in metric else "test"]; pb = EV[(b, s_)]["her" if "Herlev" in metric else "test"]
            yy = Y_HER if "Herlev" in metric else y_te
            d.append(100 * (full_metrics(pa, yy)[key_] - full_metrics(pb, yy)[key_]))
        d = np.array(d); wins = int((better * d > 0).sum())
        pr.append({"Comparison": f"{OURS} − {b}", "Metric": metric, "Δ per seed": ", ".join(f"{v:+.2f}" for v in d),
                   "Mean Δ": round(d.mean(), 2), "SD": round(d.std(ddof=1), 2), f"Ours better in": f"{wins}/{len(d)} seeds",
                   "paired t p": round(stats.ttest_rel(d, np.zeros_like(d)).pvalue, 4) if len(d) > 2 else np.nan})
savetab(pd.DataFrame(pr).set_index(["Comparison", "Metric"]), "T18_seed_paired",
        "TABLE 18. Paired by seed (Herlev ECE: negative Δ = ours better calibrated)")

# per-seed grouped bootstrap on Herlev (ours vs plain B3)
bt = []
for s_ in SEEDS_ALL:
    pa, pb = EV[(OURS, s_)]["her"], EV[("EfficientNet-B3 (plain)", s_)]["her"]
    lo, hi = grouped_boot_delta(MF["Macro-F1"], pa, pb, Y_HER, G_HER, CFG["N_BOOT"])
    bt.append({"Seed": s_, "ΔHerlev F1 (ours − plain B3)": round(100 * (MF["Macro-F1"](pa, Y_HER) - MF["Macro-F1"](pb, Y_HER)), 2),
               "95% CI": f"[{100*lo:.2f}, {100*hi:.2f}]", "McNemar p": round(mcnemar_p(pa, pb, Y_HER), 5)})
savetab(pd.DataFrame(bt).set_index("Seed"), "T19_seed_bootstrap_vs_B3", "TABLE 19. Same backbone comparison per seed (Herlev)")

# Figure: per-seed points + mean
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
for ax, col, ttl in zip(axes, ["Herlev F1", "Herlev ECE", "Referral Herlev %"], ["Herlev Macro-F1 (%) ↑", "Herlev ECE (%) ↓", "Herlev conformal referral (%)"]):
    for i, n in enumerate(SEED_MODELS):
        v = PS.loc[PS.Model == n, col].values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, color=MCOL[i], s=28, edgecolor="white", zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=2)
    ax.set_xticks(range(len(SEED_MODELS)), [n.replace(" (plain)", "\n(plain)") for n in SEED_MODELS], fontsize=7); ax.set_title(ttl, fontsize=9)
plt.tight_layout(); savefig("F19_seed_robustness"); plt.show()

# %% CELL 21 : DEEP ENSEMBLE MEMBERS (reuses seed runs) ------------
ENS_SEEDS = [CFG["SEED"], 1, 2]
ENS_MODELS = ["ViT-B/16", "EfficientNet-B3 (plain)"]
def run_key(n, seed):
    base = n if n == OURS else "BASE_" + n.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return base if seed == CFG["SEED"] else f"SEED{seed}_{base}"
def probs_cric_her(key_, cfg, r):
    h = str(r["head"]); T = fit_temperature(r["val_logits"], y_val, h)
    sd = torch.load(f"{RUNS}/{key_}/best.pt", map_location=device)
    m_ = DualFOVNet(cfg, fm_dim=sd["fm_proj.2.weight"].shape[0] if "fm_proj.2.weight" in sd else 0, pretrained=False).to(device)
    m_.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}); m_.eval()
    out = dict(val=probs_from_logits(r["val_logits"], h, T), test=probs_from_logits(r["test_logits"], h, T),
               her=probs_from_logits(predict(m_, None, Xsrc=X_HER), h, T))
    del m_; torch.cuda.empty_cache(); return out
MEMBERS = {}
for n in ENS_MODELS:
    for sd_ in ENS_SEEDS:
        c = {**ALL[n], "SEED": sd_, "TTA": False}; k = run_key(n, sd_)
        if not os.path.exists(f"{RUNS}/{k}/preds.npz"): c = preflight(c)
        r = RES[n] if sd_ == CFG["SEED"] else train_model(k, c)
        MEMBERS[(n, sd_)] = probs_cric_her(k, c, r)
        print(f"== {n} seed {sd_} ready | elapsed {(time.time()-T_ALL)/60:.1f} min ==")
MEMBERS[(OURS, CFG["SEED"])] = probs_cric_her(OURS, ALL[OURS], RES[OURS])


# %% CELL 22 : DEEP ENSEMBLE vs CytoGrade (calibration under shift) -------
def rel_row(name, pv, pt, ph, n_models, gflops):
    q = conformal_threshold(pv, y_val, CFG["CONFORMAL_ALPHA"]); mc, mh = full_metrics(pt, y_te), full_metrics(ph, Y_HER)
    ch = conformal_metrics(conformal_sets(ph, q), Y_HER)
    return {"Model": name, "Models": n_models, "GFLOPs": gflops, "CRIC F1": 100 * mc["macro_f1"], "CRIC ECE": 100 * mc["ece"],
            "Herlev F1": 100 * mh["macro_f1"], "Herlev AUC": 100 * mh["auc"], "Herlev ECE": 100 * mh["ece"], "Herlev Brier": mh["brier"],
            "Herlev NLL": mh["nll"], "Herlev referral %": 100 * ch["referral"], "Herlev coverage %": 100 * ch["coverage"]}
rows = []
o = MEMBERS[(OURS, CFG["SEED"])]
rows.append(rel_row(f"{OURS} (single)", o["val"], o["test"], o["her"], 1, CX[OURS]["GFLOPs"]))
ENS = {}
for n in ENS_MODELS:
    m0 = MEMBERS[(n, CFG["SEED"])]
    rows.append(rel_row(f"{n} (single)", m0["val"], m0["test"], m0["her"], 1, CX[n]["GFLOPs"]))
    pv, pt, ph = (np.mean([MEMBERS[(n, s_)][k] for s_ in ENS_SEEDS], 0) for k in ("val", "test", "her"))
    ENS[n] = dict(val=pv, test=pt, her=ph)
    rows.append(rel_row(f"{n} (deep ensemble ×{len(ENS_SEEDS)})", pv, pt, ph, len(ENS_SEEDS), len(ENS_SEEDS) * CX[n]["GFLOPs"]))
T20 = pd.DataFrame(rows).set_index("Model").round(3)
savetab(T20, "T20_deep_ensemble", "TABLE 20. CytoGrade (single model) vs deep ensembles: calibration under domain shift")
st = []
for n in ENS_MODELS:
    for lab, fn in [("Herlev ECE", lambda p, y: ece_score(p, y)), ("Herlev Macro-F1", MF["Macro-F1"])]:
        lo, hi = grouped_boot_delta(fn, o["her"], ENS[n]["her"], Y_HER, G_HER, CFG["N_BOOT"])
        st.append({"Comparison": f"{OURS} − {n} ensemble", "Metric": lab, "Δ": round(100 * (fn(o["her"], Y_HER) - fn(ENS[n]["her"], Y_HER)), 2),
                   "95% CI": f"[{100*lo:.2f}, {100*hi:.2f}]"})
savetab(pd.DataFrame(st).set_index(["Comparison", "Metric"]), "T20b_ensemble_stats", "TABLE 20b. Ours − ensemble on Herlev (source-ID bootstrap; ECE: negative = ours better)")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
for i, (nm, r_) in enumerate(T20.iterrows()):
    col = MCOL[i % len(MCOL)]; mk = "*" if OURS in nm else ("s" if "ensemble" in nm else "o")
    axes[0].scatter(r_["GFLOPs"], r_["Herlev ECE"], s=90 if OURS in nm else 45, marker=mk, color=col, edgecolor="white", zorder=3, label=nm)
    axes[1].scatter(r_["Herlev ECE"], r_["Herlev F1"], s=90 if OURS in nm else 45, marker=mk, color=col, edgecolor="white", zorder=3)
axes[0].set_xscale("log"); axes[0].set_xlabel("Inference GFLOPs per cell (log)"); axes[0].set_ylabel("Herlev ECE (%) ↓")
axes[0].set_title("Calibration under shift vs compute"); axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
axes[1].set_xlabel("Herlev ECE (%) ↓"); axes[1].set_ylabel("Herlev Macro-F1 (%) ↑"); axes[1].set_title("Accuracy vs calibration (Herlev)")
plt.tight_layout(); savefig("F20_ensemble_vs_ours"); plt.show()


# %% CELL 23 : PACKAGE -----------------------------------------------------
os.system(f"cd {CFG['OUT']} && zip -rq /kaggle/working/cytograde_results.zip tables figures && ls -lh /kaggle/working/cytograde_results.zip")
print("Tables and figures: /kaggle/working/cytograde_results.zip | weights + predictions: cg2_out/runs/")
