# -*- coding: utf-8 -*-
"""
Immune atlas pretraining (GRL) for CRC + NSCLC + COVID integration.

Updates in this version (v2):
  1. Integration evaluation metrics (no scib dependency):
       Biology preservation : ARI, NMI, ASW_cell
       Batch mixing         : ASW_batch, Graph Connectivity, kBET
     Full evaluation runs on EVERY plot step (sync with --plot-every).
  2. UMAP saved as PNG + PDF + SVG (vector) for downstream editing.
  3. UMAP rendering uses sc.pl.umap exactly like the original (same point size,
     alpha, draw order, palette).  We ONLY post-process the axes:
       - hide the four spines (no surrounding rectangle)
       - hide the default UMAP1 / UMAP2 axis labels
       - draw small arrow axes labeled UMAP1 / UMAP2 in the bottom-left corner
     The scatter itself is not redrawn.
"""

import os, sys, argparse, re, json, pickle, datetime, warnings, glob, csv
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from typing import Optional, Tuple, Dict
import anndata

_PCA_CACHE = None

# ================= "主要类"集合（拆分 Myeloid_MonoMacro） =================
COMMON_BIG = {
    "CD4_T", "CD8_T", "T_gdT", "B_lineage", "Plasma", "DC",
    "Monocyte", "Macrophage",
    "Neutrophil", "NK_ILC", "Mast",
}

def get_args():
    p = argparse.ArgumentParser()

    # ======== 数据路径 ========
    p.add_argument("--crc-h5ad", dest="crc_h5ad",
                   default=os.environ.get("IMMUNELENS_CRC_RNA", "data/crc/rna.h5ad"))
    p.add_argument("--crc-meta", dest="crc_meta",
                   default=os.environ.get("IMMUNELENS_CRC_META", "data/crc/CRC-ICB_metadata.tsv"))
    p.add_argument("--ns-h5ad", dest="ns_h5ad",
                   default=os.environ.get(
                       "IMMUNELENS_NSCLC_RNA",
                       "data/nsclc/GSE243013_NSCLC_immune_scRNA.h5ad"))
    p.add_argument("--ns-meta", dest="ns_meta",
                   default=os.environ.get(
                       "IMMUNELENS_NSCLC_META",
                       "data/nsclc/GSE243013_NSCLC_immune_scRNA_metadata.csv"))
    p.add_argument("--covid-h5ad", dest="covid_h5ad",
                   default=os.environ.get(
                       "IMMUNELENS_COVID_RNA", "data/covid/covid19_data.h5ad"))
    p.add_argument("--covid-meta", dest="covid_meta",
                   default=os.environ.get(
                       "IMMUNELENS_COVID_META", "data/covid/cell_annotation.csv"))

    # ======== 训练基础 ========
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--save-dir", dest="save_dir", default="./result/GRL2PDF2mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--plot-every", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-disc", type=float, default=1e-4, help="判别器学习率")
    p.add_argument("--rna-mask", type=float, default=0.3,
                   help="[DEPRECATED] full reconstruction, 该参数不生效")
    p.add_argument("--pca-dims", type=int, default=64)
    p.add_argument("--pca-l2norm", action="store_true")
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--umap-dpi", type=int, default=150)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sample-per-class", type=int, default=0)
    p.add_argument("--reuse-pca", action="store_true")
    p.add_argument("--try-rapids", action="store_true")
    p.add_argument("--grl-lam-max", type=float, default=0.3)
    p.add_argument("--grl-k", type=float, default=2.0)
    p.add_argument("--disc-width", type=int, default=256)
    p.add_argument("--disc-depth", type=int, default=2)
    p.add_argument("--disc-dropout", type=float, default=0.2)
    p.add_argument("--disc-on-z", action="store_true")
    p.add_argument("--eval-with-h", action="store_true")
    p.add_argument("--balance-common", action="store_true")
    p.add_argument("--balance-seed", type=int, default=42)

    p.add_argument("--tensor-dtype", choices=["fp16","fp32"], default="fp16")
    p.add_argument("--min-count", type=int, default=1000)

    # ================= Encoder/Decoder =================
    p.add_argument("--enc-use-attn", action="store_true")
    p.add_argument("--enc-topk", type=int, default=512)
    p.add_argument("--enc-topk-abs", action="store_true")
    p.add_argument("--enc-detach-embed", action="store_true")
    p.add_argument("--enc-width", type=int, default=512)
    p.add_argument("--enc-depth", type=int, default=3)
    p.add_argument("--enc-mult", type=int, default=4)
    p.add_argument("--enc-dropout", type=float, default=0.1)
    p.add_argument("--enc-feature-drop", action="store_true")
    p.add_argument("--enc-n-latents", type=int, default=256)
    p.add_argument("--enc-n-heads", type=int, default=8)
    p.add_argument("--enc-latent-self-attn", action="store_true")

    # ================= 布局快照 =================
    p.add_argument("--snapshot-enable", action="store_true")
    p.add_argument("--snapshot-keep", type=int, default=8)
    p.add_argument("--save-ref-on-plot", action="store_true")
    p.add_argument("--keep-tagged-ref", action="store_true")

    # ================= 评估 =================
    p.add_argument("--metrics-csv", default="metrics_log.csv")
    p.add_argument("--eval-max-cells", type=int, default=50000,
                   help="ASW/kBET 下采样到该数量; 0=全量")
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--leiden-resolution", type=float, default=1.0)

    # ================= 输出格式 =================
    p.add_argument("--fig-formats", default="png,pdf,svg",
                   help="UMAP 输出格式列表; pdf/svg 为矢量")

    return p.parse_args()

args = get_args()
os.makedirs(args.save_dir, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if "cuda" in args.device and not torch.cuda.is_available():
    print("[WARN] CUDA 不可用，回退到 CPU")
device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
SEED = args.seed
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
DTYPE_TORCH = torch.float16 if (args.tensor_dtype == "fp16" and device.type == "cuda") else torch.float32

FIG_FORMATS = [s.strip().lower() for s in args.fig_formats.split(",") if s.strip()]

print(f"[CFG] RNA reconstruction mode: FULL (no masking).")
print(f"[CFG] fig_formats={FIG_FORMATS}")


def read_table_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)
    return df

def _prune_layout_snapshots(snap_dir: str, keep: int = 8):
    snaps = sorted(glob.glob(os.path.join(snap_dir, "snapshot_*.npz")), key=os.path.getmtime, reverse=True)
    to_del = snaps[keep:]
    def _tag(p): return re.sub(r"^snapshot_|\.npz$", "", os.path.basename(p))
    for sp in to_del:
        tag = _tag(sp)
        for name in [f"snapshot_{tag}.npz", f"pca_{tag}.pkl", f"umap_{tag}.npz", f"params_{tag}.json"]:
            p = os.path.join(snap_dir, name)
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

def _prep_for_match(label: str) -> str:
    if not isinstance(label, str): return ""
    q = label.strip().lower()
    q = re.sub(r"^c\d+[_-]", "", q)
    q = q.replace("t_cd4", "cd4").replace("cd4t", "cd4")
    q = q.replace("t_cd8", "cd8").replace("cd8t", "cd8")
    q = q.replace("plasma_cell", "plasmab").replace("plasma cell", "plasmab").replace("plasmablast", "plasmab")
    q = q.replace("mast cell", "mast").replace("mφ", "macro")
    q = q.replace("mono.", "mono")
    q = re.sub(r"[\s\-]+", "_", q)
    q = re.sub(r"_+", "_", q).strip("_")
    return q

def _has_token(q: str, tok: str) -> bool:
    return re.search(rf"(?:^|_)({re.escape(tok)})(?:_|$)", q) is not None

def map_to_big(label: str) -> Optional[str]:
    q = _prep_for_match(label)
    if not q: return None
    if any(k in q for k in ("epi","epithe","coloncyte","goblet","eec","tuft","tumor","at2","ciliated","secretory","squamous")): return None
    if any(k in q for k in ("endo","lyve1","ackr1","cxcl12","col4a1","fibro","sox6","adamdec1","pi16","acta2","des","fap","pericyte","rgs5","smc","myh11","glia","plp1")): return None
    if any(k in q for k in ("mega","platelet")): return None

    if any(k in q for k in ("plasmab", "mzb1", "xbp1", "ighg", "igha", "txndc5", "plasma")): return "Plasma"

    if "naiveb" in q or _has_token(q,"bn") or "tcl1a" in q or "ighd" in q: return "B_lineage"
    if ("memory_b" in q or _has_token(q,"bm")
        or any(k in q for k in ("cd27","ms4a1","cd74","gpr183","fcrl4","mt2a","pde4d","tnf","tnfsf9","rgs13","gcb"))): return "B_lineage"
    if _has_token(q, "b") and not any(t in q for t in ("t_", "cd4", "cd8")): return "B_lineage"

    if "gdt" in q or "gamma_delta" in q or "trdv1" in q or "trdv2" in q or _has_token(q,"gd_t"): return "T_gdT"
    if "cd4" in q or _has_token(q,"t_cd4"): return "CD4_T"
    if "cd8" in q or _has_token(q,"t_cd8"): return "CD8_T"
    if ("nk" in q) or _has_token(q,"ilc") or _has_token(q,"ilc3") or ("kit" in q and "ilc" in q): return "NK_ILC"

    if any(k in q for k in ("clec9a","xcr1","batf3")): return "DC"
    if any(k in q for k in ("cd1c","fcer1a")): return "DC"
    if any(k in q for k in ("lilra4","tcf4","tcf7l2","clec4c","pdc")): return "DC"
    if "lamp3" in q: return "DC"

    if any(k in q for k in ("mono","cd14","s100a8","s100a9","lyz","fcgr3a","vcan","sirpa","tnfaip6","ifit1","ifit2","isg15","mx1","cxcl10","cxcl2","mmp9","lilrb1")):
        return "Monocyte"

    if any(k in q for k in ("macro","c1qc","apoe","folr2","marco","spp1","mrc1","cd163","trem2","msr1","ch25h","fabp4","fabp5","lgals3","lpl","ctsd","csf1r","cx3cr1")):
        return "Macrophage"

    if any(k in q for k in ("neutro","neutrophil","fcgr3b","cxcr2","cxcr4")) or _has_token(q,"neu"): return "Neutrophil"
    if "mast" in q or "tpsab1" in q: return "Mast"
    return None

def _dist_table(adata, dataset_name, label_col="CellClass"):
    vc = adata.obs[label_col].value_counts().sort_index()
    df = pd.DataFrame({"count": vc})
    df["dataset"] = dataset_name
    df.index.name = "class"
    return df.reset_index()

def print_counts_triple(adata_crc, adata_ns, adata_cv, label_col="CellClass", header=""):
    d_crc = _dist_table(adata_crc, "CRC", label_col)
    d_ns  = _dist_table(adata_ns,  "NSCLC", label_col)
    d_cv  = _dist_table(adata_cv,  "COVID", label_col)
    d_all = pd.concat([d_crc, d_ns, d_cv], ignore_index=True)
    if header: print(f"\n===== {header} =====")
    print(d_all.pivot(index="class", columns="dataset", values="count").fillna(0).astype(int))

def balance_by_class_triplet(adata_crc, adata_ns, adata_cv,
                             label_col="CellClass", seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    classes = sorted(set(adata_crc.obs[label_col]) & set(adata_ns.obs[label_col]) & set(adata_cv.obs[label_col]))
    keep_crc, keep_ns, keep_cv, log_rows = [], [], [], []
    for c in classes:
        idx_crc = np.where(adata_crc.obs[label_col].values == c)[0]
        idx_ns  = np.where(adata_ns .obs[label_col].values == c)[0]
        idx_cv  = np.where(adata_cv .obs[label_col].values == c)[0]
        n_crc, n_ns, n_cv = len(idx_crc), len(idx_ns), len(idx_cv)
        n_keep = min(n_crc, n_ns, n_cv)
        if n_keep == 0: continue
        if n_crc > n_keep: idx_crc = rng.choice(idx_crc, size=n_keep, replace=False)
        if n_ns  > n_keep: idx_ns  = rng.choice(idx_ns,  size=n_keep, replace=False)
        if n_cv  > n_keep: idx_cv  = rng.choice(idx_cv,  size=n_keep, replace=False)
        keep_crc.append(idx_crc); keep_ns.append(idx_ns); keep_cv.append(idx_cv)
        log_rows.append((c, n_crc, n_ns, n_cv, n_keep))
    def _merge(idxs):
        if len(idxs):
            x = np.concatenate(idxs); x.sort(); return x
        return np.array([], dtype=int)
    keep_crc = _merge(keep_crc); keep_ns = _merge(keep_ns); keep_cv = _merge(keep_cv)
    if verbose and len(log_rows):
        df = pd.DataFrame(log_rows, columns=["class","CRC_before","NSCLC_before","COVID_before","kept_each_side"])
        print("\n[BALANCE] 对齐到三侧最小计数（每类）：")
        print(df.sort_values("class").to_string(index=False))
    return adata_crc[keep_crc].copy(), adata_ns[keep_ns].copy(), adata_cv[keep_cv].copy()

# -------------------- 读取三套数据 & 元数据 --------------------
print("读取 CRC / NSCLC / COVID 数据 ...")
adata_crc = sc.read_h5ad(args.crc_h5ad)
adata_ns  = sc.read_h5ad(args.ns_h5ad)
adata_cv  = sc.read_h5ad(args.covid_h5ad)

def read_meta_crc(path):
    df = read_table_auto(path)
    key = None
    for c in ["cell", "Cell", "cellID", "cellId", "barcode", "cell_name", "orig.ident"]:
        if c in df.columns: key = c; break
    if key is None:
        raise KeyError(f"[CRC] Metadata 未找到细胞级 ID 列；现有列: {df.columns.tolist()[:12]}")
    df[key] = df[key].astype(str).str.strip()
    df = df.drop_duplicates(subset=[key]).set_index(key)
    if "SubCellType" not in df.columns:
        raise KeyError(f"[CRC] 缺列 SubCellType; 现有: {df.columns.tolist()[:12]}")
    return df

def read_meta_ns(path):
    df = read_table_auto(path)
    if "cellID" not in df.columns or "sub_cell_type" not in df.columns:
        raise KeyError(f"[NSCLC] 需要列 cellID / sub_cell_type; 现有: {df.columns.tolist()[:12]}")
    df["cellID"] = df["cellID"].astype(str).str.strip()
    return df.drop_duplicates(subset=["cellID"]).set_index("cellID")

def read_meta_covid(path):
    df = read_table_auto(path)
    if "cellName" not in df.columns or "celltype" not in df.columns:
        raise KeyError(f"[COVID] 需要列 cellName / celltype; 现有: {df.columns.tolist()[:12]}")
    df["cellName"] = df["cellName"].astype(str).str.strip()
    return df.drop_duplicates(subset=["cellName"]).set_index("cellName")

meta_crc  = read_meta_crc(args.crc_meta)
meta_ns   = read_meta_ns(args.ns_meta)
meta_cv   = read_meta_covid(args.covid_meta)

for ad in (adata_crc, adata_ns, adata_cv):
    ad.var_names = ad.var_names.astype(str)
    ad.var_names_make_unique()
    ad.obs_names = ad.obs_names.astype(str).str.strip()

def align_and_map_big(name, adata, meta, label_col_meta, raw_label_obs="SubCellType"):
    inter = adata.obs_names.isin(meta.index)
    hit = int(inter.sum()); tot = int(adata.n_obs)
    print(f"[ALIGN] {name}: meta 命中 {hit}/{tot} ({hit/max(1,tot):.1%})")
    if hit / max(1, tot) < 0.5:
        print(f"[WARN] {name}: 元数据命中率较低，请检查 ID 对齐列。")
    adata = adata[inter].copy()
    adata.obs[raw_label_obs] = meta.loc[adata.obs_names, label_col_meta].astype(str).values
    adata.obs["CellClass"] = adata.obs[raw_label_obs].map(map_to_big)
    before = adata.n_obs
    adata = adata[adata.obs["CellClass"].isin(COMMON_BIG)].copy()
    print(f"[FILTER] {name}: kept {adata.n_obs}/{before} (映射到 major immune classes)")
    if adata.n_obs == 0:
        raise ValueError(f"[{name}] 过滤后没有样本，请检查映射规则或原始标签。")
    return adata

adata_crc = align_and_map_big("CRC",   adata_crc, meta_crc, "SubCellType",   "SubCellType")
adata_ns  = align_and_map_big("NSCLC", adata_ns,  meta_ns,  "sub_cell_type", "SubCellType")
adata_cv  = align_and_map_big("COVID", adata_cv,  meta_cv,  "celltype",      "SubCellType")
for _ad, nm in [(adata_crc,"CRC"),(adata_ns,"NSCLC"),(adata_cv,"COVID")]:
    _ad.obs["dataset"] = nm


# -------------------- 公共类筛选 --------------------
def _counts(df): return df.obs["CellClass"].value_counts()
vc_crc = _counts(adata_crc); vc_ns = _counts(adata_ns); vc_cv = _counts(adata_cv)

min_count = max(1, int(args.min_count))
cands = sorted(set(vc_crc.index) & set(vc_ns.index) & set(vc_cv.index))
qualified = []
for c in cands:
    if (vc_crc.get(c, 0) >= min_count) and (vc_ns.get(c, 0) >= min_count) and (vc_cv.get(c, 0) >= min_count):
        qualified.append(c)

print("\n[COMMON筛选] 候选：", cands)
print(f"[COMMON筛选] 计数阈值: >= {min_count} ；最终保留: {qualified}")
if not qualified:
    raise ValueError("三数据集无满足阈值的共同类；请调低 --min-count 或检查映射。")

def keep_only_qualified(ad, name):
    before = ad.n_obs
    ad = ad[ad.obs["CellClass"].isin(qualified)].copy()
    print(f"[FILTER-QUALIFIED] {name}: kept {ad.n_obs}/{before}")
    return ad

adata_crc = keep_only_qualified(adata_crc, "CRC")
adata_ns  = keep_only_qualified(adata_ns,  "NSCLC")
adata_cv  = keep_only_qualified(adata_cv,  "COVID")

def _counts_table_to_csv(ad_crc, ad_ns, ad_cv, save_dir, tag):
    d_crc = _dist_table(ad_crc, "CRC"); d_ns = _dist_table(ad_ns, "NSCLC"); d_cv = _dist_table(ad_cv, "COVID")
    d_all = pd.concat([d_crc, d_ns, d_cv], ignore_index=True)
    path = os.path.join(save_dir, f"class_counts_{tag}.csv")
    d_all.to_csv(path, index=False)
    print(f"[SAVE] class counts: {path}")

print_counts_triple(adata_crc, adata_ns, adata_cv, "CellClass", header="[Before balance] CellClass counts (tri, qualified)")

if args.balance_common:
    print("\n执行按 CellClass 成对下采样以平衡 CRC / NSCLC / COVID ...")
    adata_crc, adata_ns, adata_cv = balance_by_class_triplet(
        adata_crc, adata_ns, adata_cv,
        label_col="CellClass",
        seed=args.balance_seed,
        verbose=True
    )
    print_counts_triple(adata_crc, adata_ns, adata_cv, "CellClass", header="[After balance] CellClass counts (tri, paired)")

# -------------------- 预处理与合并 --------------------
print("合并数据 ...")
for _ad in (adata_crc, adata_ns, adata_cv):
    sc.pp.filter_cells(_ad, min_genes=200)
    sc.pp.normalize_total(_ad, target_sum=1e4)
    sc.pp.log1p(_ad)

adata_all = anndata.concat(
    [adata_crc, adata_ns, adata_cv],
    label="dataset_batch",
    keys=["CRC", "NSCLC", "COVID"],
    join="outer",
    index_unique="-"
)

sc.pp.filter_genes(adata_all, min_cells=3)
try:
    sc.pp.highly_variable_genes(
        adata_all, n_top_genes=2000, flavor="seurat_v3",
        batch_key="dataset_batch", subset=True
    )
except Exception as e:
    print("[HVG] seurat_v3 失败，回退到 seurat：", repr(e))
    sc.pp.highly_variable_genes(
        adata_all, n_top_genes=2000, flavor="seurat",
        batch_key="dataset_batch", subset=True
    )

sc.pp.scale(adata_all, max_value=10)

# ========= Dataset =========
class RNADataset(Dataset):
    def __init__(self, X_mat, batch_idx_tensor, dtype=DTYPE_TORCH):
        self.X = X_mat
        self.bidx = batch_idx_tensor
        self.dtype = dtype
        self.N = self.X.shape[0]
    def __getitem__(self, idx):
        row = self.X[idx:idx+1]
        if sparse.issparse(row):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32, order="C")
        row = torch.as_tensor(row, dtype=self.dtype).squeeze(0)
        return row, self.bidx[idx]
    def __len__(self):
        return self.N

batch_codes = pd.Categorical(adata_all.obs["dataset_batch"]).codes
batch_idx_all = torch.tensor(batch_codes, dtype=torch.long)

loader = DataLoader(
    RNADataset(adata_all.X, batch_idx_all, dtype=DTYPE_TORCH),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=(0 if sys.platform.startswith("win") else args.num_workers),
    pin_memory=(device.type == "cuda")
)

# -------------------- 模型 --------------------
from immunelens.models.model import GeneMLPEncoder, RNADecoder

def mask_rna_batch(x: torch.Tensor, mask_prob: float, mask_token: torch.Tensor):
    B, G = x.shape
    mask = torch.rand(B, G, device=x.device) < mask_prob
    tok = mask_token.to(dtype=x.dtype, device=x.device).expand(B, G)
    x_in = torch.where(mask, tok, x)
    return x_in, x, mask

class RNAMaskedAutoEncoder(nn.Module):
    def __init__(self, gene_dim, latent_dim=256, proj_hidden=512, proj_out=256, proj_dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GeneMLPEncoder(
            input_dim=gene_dim,
            out_dim=latent_dim,
            width=args.enc_width,
            depth=args.enc_depth,
            mult=args.enc_mult,
            dropout=args.enc_dropout,
            feature_drop=0.0,
            use_attention=args.enc_use_attn,
            n_latents=args.enc_n_latents,
            n_heads=args.enc_n_heads,
            latent_self_attn=args.enc_latent_self_attn,
            select_topk=(None if (args.enc_topk is None or args.enc_topk <= 0) else args.enc_topk),
            abs_topk=args.enc_topk_abs,
            detach_gene_embed=args.enc_detach_embed
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, proj_hidden), nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden, proj_out),
        )
        try:
            self.decoder = RNADecoder(d_model=proj_out, gene_dim=gene_dim)
        except TypeError:
            self.decoder = RNADecoder(d_model=proj_out, out_dim=gene_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, gene_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def encode(self, x, return_h=False):
        z = self.norm(self.encoder(x))
        if return_h:
            h = self.proj(z)
            return z, h
        return z

    def forward(self, x, add_noise_std=0.05, return_h=False):
        z = self.norm(self.encoder(x))
        h = z + (torch.randn_like(z) * add_noise_std if self.training and add_noise_std > 0 else 0.0)
        h = self.proj(h)
        recon = self.decoder(h)
        return (recon, z, h) if return_h else (recon, z)

gene_dim = adata_all.X.shape[1]
model = RNAMaskedAutoEncoder(gene_dim=gene_dim).to(device)

# -------------------- GRL 与判别器 --------------------
from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None
def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

import torch.nn.utils as nnutils
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nnutils.spectral_norm(nn.Linear(dim, dim))
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nnutils.spectral_norm(nn.Linear(dim, dim))
    def forward(self, x):
        y = self.norm(x); y = self.fc1(y); y = self.act(y); y = self.drop(y); y = self.fc2(y)
        return x + y

class BatchDiscriminator(nn.Module):
    def __init__(self, in_dim, n_batches=3, width=256, depth=2, dropout=0.2):
        super().__init__()
        self.inp  = nnutils.spectral_norm(nn.Linear(in_dim, width))
        self.act  = nn.GELU()
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(max(1, depth))])
        self.norm = nn.LayerNorm(width)
        self.head = nnutils.spectral_norm(nn.Linear(width, n_batches))
    def forward(self, x):
        h = self.act(self.inp(x))
        for blk in self.blocks: h = blk(h)
        h = self.norm(h)
        return self.head(h)

disc_in_dim = (model.latent_dim if args.disc_on_z else 256)
disc = BatchDiscriminator(
    in_dim=disc_in_dim, n_batches=3,
    width=args.disc_width, depth=args.disc_depth,
    dropout=args.disc_dropout
).to(device)

print(f"[CFG] disc_on_z={args.disc_on_z}  eval_with_h={args.eval_with_h}")

optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=args.lr, weight_decay=0.05)
opt_disc = torch.optim.AdamW(disc.parameters(), lr=args.lr_disc, weight_decay=0.01)
from torch.amp import GradScaler, autocast
scaler = GradScaler(enabled=(device.type == "cuda"))


def dann_lambda(progress, lam_max=0.3, k=5.0):
    return lam_max * (2.0 / (1.0 + np.exp(-k * progress)) - 1.0)

def acc_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


# ===================================================================
# Fixed palettes (UNCHANGED from original)
# ===================================================================
def _apply_fixed_palettes(ad: sc.AnnData):
    import matplotlib.pyplot as plt
    for col in ["dataset", "CellClass"]:
        if col in ad.obs.columns:
            ad.obs[col] = ad.obs[col].astype("category")
    dataset_color_map = {
        "CRC":   "#1f77b4",
        "NSCLC": "#ff7f0e",
        "COVID": "#2ca02c",
    }
    if "dataset" in ad.obs.columns:
        cats = list(ad.obs["dataset"].cat.categories)
        ad.uns["dataset_colors"] = [dataset_color_map.get(c, "#7f7f7f") for c in cats]
    cellclass_color_map = {
        "CD4_T":       "#d62728",
        "CD8_T":       "#9467bd",
        "T_gdT":       "#8c564b",
        "B_lineage":   "#17becf",
        "Plasma":      "#e377c2",
        "DC":          "#7f7f7f",
        "Monocyte":    "#bcbd22",
        "Macrophage":  "#b15928",
        "Neutrophil":  "#1f9e89",
        "NK_ILC":      "#6baed6",
        "Mast":        "#ffd92f",
    }
    if "CellClass" in ad.obs.columns:
        cats = list(ad.obs["CellClass"].cat.categories)
        import matplotlib
        fallback = plt.get_cmap("tab20")
        forbidden = set(dataset_color_map.values())
        colors = []
        f_i = 0
        for cls in cats:
            if cls in cellclass_color_map:
                col = cellclass_color_map[cls]
            else:
                while True:
                    col = matplotlib_to_hex(fallback(f_i / 20.0)); f_i = (f_i + 1) % 20
                    if col not in forbidden: break
            colors.append(col)
        ad.uns["CellClass_colors"] = colors

def matplotlib_to_hex(rgba):
    import matplotlib
    return matplotlib.colors.to_hex(rgba, keep_alpha=False)


# ===================================================================
# Axes post-processing.
# Only this is new; the scatter itself comes from sc.pl.umap as before.
# ===================================================================

def _postprocess_umap_axes(ax, frac: float = 0.18, lw: float = 1.4, font_size: int = 10):
    """
    Take a matplotlib Axes that already has the scanpy UMAP scatter on it, and:
      - hide the four spines (no surrounding rectangle)
      - hide tick marks AND the default 'UMAP1' / 'UMAP2' axis labels
      - draw small arrow axes labeled UMAP1 / UMAP2 in the bottom-left corner

    The scatter is NOT redrawn; every visual property set by sc.pl.umap
    (point size, alpha, draw order, colormap, jitter) is preserved.
    """
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, right=False, top=False,
                   labelleft=False, labelbottom=False, labelright=False, labeltop=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin
    if xr <= 0 or yr <= 0:
        return

    # Anchor arrow origin slightly inside the bottom-left of current view
    x0 = xmin + 0.02 * xr
    y0 = ymin + 0.02 * yr
    dx = frac * xr
    dy = frac * yr

    arrow_kw = dict(length_includes_head=True,
                    head_width=0.04 * min(dx, dy),
                    head_length=0.06 * min(dx, dy),
                    fc="black", ec="black",
                    linewidth=lw, clip_on=False, zorder=10)
    ax.arrow(x0, y0, dx, 0, **arrow_kw)
    ax.arrow(x0, y0, 0, dy, **arrow_kw)
    ax.text(x0 + dx * 0.5, y0 - 0.04 * yr, "UMAP1",
            ha="center", va="top", fontsize=font_size, clip_on=False, zorder=10)
    ax.text(x0 - 0.03 * xr, y0 + dy * 0.5, "UMAP2",
            ha="right", va="center", rotation=90,
            fontsize=font_size, clip_on=False, zorder=10)


def _save_current_fig_multi(base_path_no_ext: str):
    """Save the current figure in every requested format.

    File-size reduction for vector formats (pdf/svg):
      * rasterize scatter point collections (and any images), so the heavy
        per-cell vector paths become a single embedded bitmap;
      * text, axes, arrows, legend, titles stay vector and remain editable
        in Illustrator/Inkscape (pdf.fonttype/svg.fonttype already set).
    PNG output is unchanged in behaviour.
    """
    import matplotlib
    # Ensure editable text in vector outputs (no-op if already set)
    matplotlib.rcParams["pdf.fonttype"]  = 42
    matplotlib.rcParams["ps.fonttype"]   = 42
    matplotlib.rcParams["svg.fonttype"]  = "none"

    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # Mark scatter collections as rasterized so vector PDF/SVG embed them as
    # a bitmap. This is what keeps file size manageable for 10^5+ cells.
    rasterized_artists = []
    for ax in fig.axes:
        for coll in ax.collections:
            try:
                coll.set_rasterized(True)
                coll.set_zorder(0)
                rasterized_artists.append(coll)
            except Exception:
                pass
        for im in ax.images:
            try:
                im.set_rasterized(True)
                rasterized_artists.append(im)
            except Exception:
                pass

    for ext in FIG_FORMATS:
        out = f"{base_path_no_ext}.{ext}"
        transparent = (ext in ("pdf", "svg"))
        # For vector targets, dpi controls only the rasterized scatter layer;
        # text and axes remain vector.
        fig.savefig(out, dpi=args.umap_dpi, bbox_inches="tight",
                    transparent=transparent)
        print(f"[PLOT] Saved: {out}")


# -------------------- 绘图主函数（与原版几乎一致, 只是结尾改 axes & 多格式保存） --------------------
def plot_umap_fast(adata_full, Z_latent, suffix, color_modes=("CellClass","dataset"),
                   quick_sample_per_class=0, n_neighbors=None, dpi=None,
                   try_rapids=False, reuse_prev_pca=False):
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from numpy.linalg import norm

    global _PCA_CACHE
    if n_neighbors is None:
        n_neighbors = min(getattr(args, "n_neighbors", 25), 30)
    if dpi is None:
        dpi = getattr(args, "umap_dpi", 150)

    N = Z_latent.shape[0]
    idx = np.arange(N)

    if quick_sample_per_class and quick_sample_per_class > 0:
        df = adata_full.obs.iloc[idx].copy()
        for col in ("CellClass", "dataset"):
            if col not in df.columns: df[col] = "NA"
        df["_row"] = idx
        keep_idx = []
        for _, g in df.groupby(["CellClass", "dataset"], sort=False):
            if len(g) > quick_sample_per_class:
                keep_idx.append(g.sample(quick_sample_per_class, random_state=SEED)["_row"].values)
            else:
                keep_idx.append(g["_row"].values)
        idx = np.concatenate(keep_idx, axis=0); idx.sort()

    X_sub = Z_latent[idx]

    # ===== PCA（可复用）=====
    pca = None
    if reuse_prev_pca and (_PCA_CACHE is not None) and (_PCA_CACHE.get("n_comp") == args.pca_dims):
        pca = _PCA_CACHE["pca"]
        try:
            X_pca = pca.transform(X_sub)
        except Exception:
            pca = None
    if pca is None:
        if X_sub.shape[0] > 50000:
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA(n_components=args.pca_dims)
            bs = 8192
            X_pca_parts = []
            for i in range(0, X_sub.shape[0], bs):
                pca.partial_fit(X_sub[i:i+bs])
            for i in range(0, X_sub.shape[0], bs):
                X_pca_parts.append(pca.transform(X_sub[i:i+bs]))
            X_pca = np.vstack(X_pca_parts)
        else:
            from sklearn.decomposition import PCA
            p = args.pca_dims
            pca = PCA(n_components=p, random_state=SEED, svd_solver="auto")
            X_pca = pca.fit_transform(X_sub)
        _PCA_CACHE = {"pca": pca, "n_comp": args.pca_dims}

    if args.pca_l2norm:
        X_pca = X_pca / (norm(X_pca, axis=1, keepdims=True) + 1e-8)

    wanted_cols = []
    for cm in color_modes:
        if cm in adata_full.obs.columns: wanted_cols.append(cm)
        else: print(f"[PLOT][skip] '{cm}' 不在 adata_full.obs.columns，跳过该配色。")
    for base_col in ["CellClass","dataset"]:
        if base_col in adata_full.obs.columns and base_col not in wanted_cols:
            wanted_cols.append(base_col)

    obs_small = adata_full.obs.iloc[idx][wanted_cols].copy()

    ad = sc.AnnData(X=np.zeros((len(idx),1), dtype=np.float32), obs=obs_small)
    ad.obsm["X_latent_pca"] = X_pca.astype(float)

    used_rapids = False
    try:
        if try_rapids:
            sc.pp.neighbors(ad, use_rep="X_latent_pca", n_neighbors=n_neighbors, metric="cosine", method="rapids")
            used_rapids = True
        else:
            raise RuntimeError("skip rapids")
    except Exception:
        sc.pp.neighbors(ad, use_rep="X_latent_pca", n_neighbors=n_neighbors, metric="cosine")
        used_rapids = False

    sc.tl.umap(ad, min_dist=0.6, spread=1.0, random_state=SEED)
    U_sub = np.asarray(ad.obsm["X_umap"], dtype=np.float32)

    _apply_fixed_palettes(ad)

    if getattr(args, "snapshot_enable", False):
        snap_dir = os.path.join(args.save_dir, "layout_snapshot")
        os.makedirs(snap_dir, exist_ok=True)
        tag = str(suffix)
        np.savez_compressed(os.path.join(snap_dir, f"snapshot_{tag}.npz"),
                            idx=idx.astype(np.int64),
                            X_pca_sub=X_pca.astype(np.float32))
        with open(os.path.join(snap_dir, f"pca_{tag}.pkl"), "wb") as f:
            pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(os.path.join(snap_dir, f"umap_{tag}.npz"), U_sub=U_sub)
        with open(os.path.join(snap_dir, f"params_{tag}.json"), "w") as f:
            json.dump({
                "use_h": bool(getattr(args, "eval_with_h", False)),
                "pca_dims": int(args.pca_dims),
                "pca_l2norm": bool(args.pca_l2norm),
                "n_neighbors": int(n_neighbors),
                "metric": "cosine",
                "min_dist": 0.6,
                "spread": 1.0,
                "seed": int(SEED),
                "suffix": tag,
                "count_points": int(len(idx))
            }, f, ensure_ascii=False, indent=2)
        try:
            _prune_layout_snapshots(snap_dir, keep=getattr(args, "snapshot_keep", 8))
        except Exception as e:
            print(f"[SNAPSHOT][warn] prune failed: {e}")
        print(f"[SNAPSHOT] saved layout snapshot: {tag}")

    # ---- Render via sc.pl.umap (same as the original), then post-process axes ----
    for cm in wanted_cols:
        try:
            base = os.path.join(args.save_dir, f"UMAP_{cm}_{suffix}")
            ax_or_list = sc.pl.umap(
                ad, color=cm, legend_loc="right margin",
                title=f"CRC+NSCLC+COVID - {cm} ({'RAPIDS' if used_rapids else 'CPU'}) [Immune major]",
                show=False
            )
            axes = ax_or_list if isinstance(ax_or_list, (list, tuple)) else [ax_or_list]
            for ax in axes:
                _postprocess_umap_axes(ax)

            _save_current_fig_multi(base)
            plt.close('all')
        except Exception as e:
            print(f"[PLOT][skip] 画 {cm} 失败：{e}")


# ===================================================================
# Evaluation metrics (no scib)
# ===================================================================

def _subsample_for_eval(n: int, seed: int) -> np.ndarray:
    if args.eval_max_cells and args.eval_max_cells > 0 and n > args.eval_max_cells:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=args.eval_max_cells, replace=False))
    return np.arange(n)


def eval_linear_separability(emb, labels, standardize=True):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        if standardize:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler(with_mean=False).fit_transform(emb)
        else:
            X = emb
        Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)
        clf = LogisticRegression(max_iter=300, solver="saga", multi_class="multinomial", n_jobs=1).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
    except Exception as e:
        print("[WARN] sklearn 评估失败：", e)
        acc = np.nan
    return acc


def metric_ari_nmi(emb: np.ndarray, cell_labels: np.ndarray, seed: int,
                   resolution: float = 1.0) -> Tuple[float, float, int]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    try:
        ad = sc.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32))
        ad.obsm["X_emb"] = emb.astype(np.float32)
        sc.pp.neighbors(ad, use_rep="X_emb",
                        n_neighbors=min(args.n_neighbors, 30), metric="cosine")
        try:
            sc.tl.leiden(ad, resolution=resolution, random_state=seed,
                         flavor="igraph", n_iterations=2, directed=False)
        except TypeError:
            sc.tl.leiden(ad, resolution=resolution, random_state=seed)
        clu = ad.obs["leiden"].astype(str).values
        ari = float(adjusted_rand_score(cell_labels, clu))
        nmi = float(normalized_mutual_info_score(cell_labels, clu))
        return ari, nmi, int(len(np.unique(clu)))
    except Exception as e:
        print(f"[METRIC][ARI/NMI] failed: {e}")
        return float("nan"), float("nan"), -1


def metric_asw_cell(emb: np.ndarray, cell_labels: np.ndarray, seed: int) -> float:
    from sklearn.metrics import silhouette_score
    try:
        idx = _subsample_for_eval(emb.shape[0], seed)
        if len(np.unique(cell_labels[idx])) < 2:
            return float("nan")
        s = silhouette_score(emb[idx], cell_labels[idx], metric="euclidean",
                             sample_size=None, random_state=seed)
        return float((s + 1.0) / 2.0)
    except Exception as e:
        print(f"[METRIC][ASW_cell] failed: {e}")
        return float("nan")


def metric_asw_batch(emb: np.ndarray, batch_labels: np.ndarray, cell_labels: np.ndarray,
                     seed: int) -> float:
    from sklearn.metrics import silhouette_samples
    try:
        idx = _subsample_for_eval(emb.shape[0], seed)
        emb_s = emb[idx]; bat_s = batch_labels[idx]; cell_s = cell_labels[idx]
        per_label = []
        for c in np.unique(cell_s):
            m = (cell_s == c)
            if m.sum() < 10:
                continue
            if len(np.unique(bat_s[m])) < 2:
                continue
            try:
                sil = silhouette_samples(emb_s[m], bat_s[m], metric="euclidean")
            except Exception:
                continue
            per_label.append(float(np.mean(1.0 - np.abs(sil))))
        return float(np.mean(per_label)) if per_label else float("nan")
    except Exception as e:
        print(f"[METRIC][ASW_batch] failed: {e}")
        return float("nan")


def metric_graph_connectivity(emb: np.ndarray, cell_labels: np.ndarray,
                              n_neighbors: int = 15) -> float:
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    try:
        labels_unique = np.unique(cell_labels)
        per_label = []
        for c in labels_unique:
            m = np.where(cell_labels == c)[0]
            if len(m) < 2:
                continue
            k = min(n_neighbors, len(m) - 1)
            sub = emb[m]
            knn = kneighbors_graph(sub, n_neighbors=k, mode="connectivity",
                                   include_self=False, metric="euclidean")
            knn = knn.maximum(knn.T)
            n_comp, comp_labels = connected_components(knn, directed=False)
            _, counts = np.unique(comp_labels, return_counts=True)
            per_label.append(float(counts.max() / len(m)))
        return float(np.mean(per_label)) if per_label else float("nan")
    except Exception as e:
        print(f"[METRIC][GraphConn] failed: {e}")
        return float("nan")


def metric_kbet(emb: np.ndarray, batch_labels: np.ndarray, seed: int,
                k: int = 50, sample_size: int = 1000) -> float:
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import chi2
    try:
        n = emb.shape[0]
        rng = np.random.default_rng(seed)

        batches, batch_codes_int = np.unique(batch_labels, return_inverse=True)
        n_batch = len(batches)
        if n_batch < 2:
            return float("nan")
        global_freq = np.bincount(batch_codes_int, minlength=n_batch) / float(n)

        idx_eval = _subsample_for_eval(n, seed)
        emb_eval = emb[idx_eval]

        k = min(k, len(idx_eval) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(emb_eval)
        _, knn_idx = nn.kneighbors(emb_eval)
        knn_idx = knn_idx[:, 1:]
        knn_codes = batch_codes_int[idx_eval[knn_idx]]

        m = min(sample_size, knn_codes.shape[0])
        sample_rows = rng.choice(knn_codes.shape[0], size=m, replace=False)
        sample = knn_codes[sample_rows]

        passed = 0
        for row in sample:
            obs = np.bincount(row, minlength=n_batch).astype(float)
            exp = global_freq * k
            exp = np.clip(exp, 1e-6, None)
            stat = float(np.sum((obs - exp) ** 2 / exp))
            pval = 1.0 - chi2.cdf(stat, df=n_batch - 1)
            if pval > 0.05:
                passed += 1
        return float(passed / m)
    except Exception as e:
        print(f"[METRIC][kBET] failed: {e}")
        return float("nan")


def run_full_evaluation(emb: np.ndarray, batch_labels: np.ndarray, cell_labels: np.ndarray) -> Dict[str, float]:
    print("\n===== Integration metrics =====")
    out: Dict[str, float] = {}

    out["logreg_acc_batch"] = eval_linear_separability(emb, batch_labels)
    out["logreg_acc_cell"] = eval_linear_separability(emb, cell_labels)
    print(f"[Eval] LogReg_batch={out['logreg_acc_batch']:.4f}  LogReg_cell={out['logreg_acc_cell']:.4f}")

    ari, nmi, n_clu = metric_ari_nmi(emb, cell_labels, args.eval_seed, resolution=args.leiden_resolution)
    out["ARI"] = ari; out["NMI"] = nmi; out["n_leiden"] = float(n_clu)
    print(f"[Eval] ARI={ari:.4f}  NMI={nmi:.4f}  n_leiden={n_clu}")

    out["ASW_cell"] = metric_asw_cell(emb, cell_labels, args.eval_seed)
    print(f"[Eval] ASW_cell={out['ASW_cell']:.4f}")

    out["ASW_batch"] = metric_asw_batch(emb, batch_labels, cell_labels, args.eval_seed)
    print(f"[Eval] ASW_batch={out['ASW_batch']:.4f}")

    out["GraphConn"] = metric_graph_connectivity(emb, cell_labels,
                                                 n_neighbors=max(15, args.n_neighbors))
    print(f"[Eval] GraphConn={out['GraphConn']:.4f}")

    out["kBET"] = metric_kbet(emb, batch_labels, args.eval_seed, k=50, sample_size=1000)
    print(f"[Eval] kBET_accept={out['kBET']:.4f}")

    bio_keys = ["ARI", "NMI", "ASW_cell"]
    batch_keys = ["ASW_batch", "GraphConn", "kBET"]
    bio_vals = [out[k] for k in bio_keys if not np.isnan(out[k])]
    batch_vals = [out[k] for k in batch_keys if not np.isnan(out[k])]
    out["bio_avg"] = float(np.mean(bio_vals)) if bio_vals else float("nan")
    out["batch_avg"] = float(np.mean(batch_vals)) if batch_vals else float("nan")
    if not (np.isnan(out["bio_avg"]) or np.isnan(out["batch_avg"])):
        out["overall"] = 0.6 * out["bio_avg"] + 0.4 * out["batch_avg"]
    else:
        out["overall"] = float("nan")
    print(f"[Eval] bio_avg={out['bio_avg']:.4f}  batch_avg={out['batch_avg']:.4f}  overall={out['overall']:.4f}")
    return out


def _metrics_row_pack(metrics: Dict[str, float]) -> Dict:
    def _r(x):
        return round(float(x), 6) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else -1.0
    return {
        "logreg_acc_batch": _r(metrics.get("logreg_acc_batch", float("nan"))),
        "logreg_acc_cell":  _r(metrics.get("logreg_acc_cell",  float("nan"))),
        "ARI":      _r(metrics.get("ARI", float("nan"))),
        "NMI":      _r(metrics.get("NMI", float("nan"))),
        "n_leiden": int(metrics.get("n_leiden", -1)) if metrics.get("n_leiden", -1) != -1 else -1,
        "ASW_cell": _r(metrics.get("ASW_cell", float("nan"))),
        "ASW_batch":_r(metrics.get("ASW_batch", float("nan"))),
        "GraphConn":_r(metrics.get("GraphConn", float("nan"))),
        "kBET":     _r(metrics.get("kBET", float("nan"))),
        "bio_avg":  _r(metrics.get("bio_avg", float("nan"))),
        "batch_avg":_r(metrics.get("batch_avg", float("nan"))),
        "overall":  _r(metrics.get("overall", float("nan"))),
    }


def compute_prototypes(emb_latent: np.ndarray, labels: pd.Series) -> Dict[str, np.ndarray]:
    protos = {}
    for cls, idx in labels.reset_index().groupby(labels.name).groups.items():
        vec = emb_latent[idx].mean(axis=0)
        protos[str(cls)] = vec
    return protos

def _compute_mean_std_from_scaled(adata):
    from scipy import sparse as _sp
    X = adata.X
    if _sp.issparse(X): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0); std  = X.std(axis=0); std[std == 0] = 1.0
    return mean, std

def _timestamp_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_all(save_dir, model, disc, args, adata_all, qualified,
             pca_obj, embedding, use_h, optimizer=None, opt_disc=None,
             X_pca_full=None, U_full=None, prototypes=None,
             tag=None):
    if tag is None:
        tag = _timestamp_tag()
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path   = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    genes_path  = os.path.join(save_dir, f"genes_hvg_order_{tag}.txt")
    cfg_path    = os.path.join(save_dir, f"config_{tag}.json")
    pca_path    = os.path.join(save_dir, f"pca_{tag}.pkl")
    stat_path   = os.path.join(save_dir, f"scale_stats_{tag}.npz")
    classes_path= os.path.join(save_dir, f"classes_{tag}.json")
    emb_path    = os.path.join(save_dir, f"embedding_{'h' if use_h else 'z'}_{tag}.npz")
    obs_path    = os.path.join(save_dir, f"obs_{'h' if use_h else 'z'}_{tag}.csv")
    proto_path  = os.path.join(save_dir, f"prototypes_big_{tag}.npz")
    ref_path    = os.path.join(save_dir, f"reference_pca_umap_{tag}.npz")

    with open(genes_path, "w") as f:
        f.write("\n".join(list(map(str, adata_all.var_names.tolist()))))
    mean, std = _compute_mean_std_from_scaled(adata_all)
    np.savez_compressed(stat_path, mean=mean, std=std)
    if pca_obj is not None:
        with open(pca_path, "wb") as f:
            pickle.dump(pca_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(classes_path, "w") as f:
        json.dump({
            "classes": sorted(list(qualified)),
            "label_space": "ImmuneMajor",
            "note": "CD4_T, CD8_T, T_gdT, B_lineage, Plasma, DC, Monocyte, Macrophage, Neutrophil, NK_ILC, Mast"
        }, f, ensure_ascii=False, indent=2)
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    state = {
        "model": model.state_dict(),
        "disc":  disc.state_dict(),
        "args":  vars(args),
        "classes": sorted(list(qualified)),
        "genes_file": os.path.basename(genes_path),
        "scale_stats": os.path.basename(stat_path),
        "pca_file": os.path.basename(pca_path),
        "ref_file": os.path.basename(ref_path),
        "obs_file": os.path.basename(obs_path),
        "embedding_file": os.path.basename(emb_path),
        "prototypes_file": os.path.basename(proto_path),
        "tag": tag,
        "recon_mode": "full",
    }
    if optimizer is not None: state["optim"] = optimizer.state_dict()
    if opt_disc  is not None: state["optim_disc"] = opt_disc.state_dict()
    torch.save(state, ckpt_path)
    print(f"[SAVE] checkpoint: {ckpt_path}")

    if embedding is not None:
        obs = adata_all.obs[["dataset_batch","dataset","CellClass"]].copy()
        obs.to_csv(obs_path)
        np.savez_compressed(emb_path, X=embedding)
        print(f"[SAVE] embedding + obs saved: {emb_path} / {obs_path}")

    if (X_pca_full is not None) and (U_full is not None):
        np.savez_compressed(ref_path, X_pca_full=X_pca_full, UMAP_2D=U_full)
        print(f"[SAVE] reference PCA & UMAP saved: {ref_path}")

    if prototypes is not None:
        cls_order = sorted(prototypes.keys())
        mat = np.stack([prototypes[c] for c in cls_order], axis=0)
        np.savez_compressed(proto_path, classes=np.array(cls_order), protos=mat)
        print(f"[SAVE] prototypes saved: {proto_path}")

    return {
        "ckpt": ckpt_path, "genes": genes_path, "stats": stat_path, "pca": pca_path,
        "classes": classes_path, "embedding": emb_path, "obs": obs_path,
        "ref": ref_path, "proto": proto_path, "cfg": cfg_path
    }

def build_reference_layout(X_lat: np.ndarray, pca_dims: int, l2norm: bool, save_dir: str,
                           use_latest_snapshot: bool = True) -> Tuple[object, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from numpy.linalg import norm
    from sklearn.neighbors import NearestNeighbors

    snap_dir = os.path.join(save_dir, "layout_snapshot")
    use_snapshot = use_latest_snapshot and os.path.isdir(snap_dir) and len(glob.glob(os.path.join(snap_dir, "snapshot_*.npz"))) > 0

    if use_snapshot:
        snaps = sorted(glob.glob(os.path.join(snap_dir, "snapshot_*.npz")), key=os.path.getmtime)
        snap_npz = snaps[-1]
        tag = os.path.basename(snap_npz)[len("snapshot_"):-len(".npz")]
        with open(os.path.join(snap_dir, f"pca_{tag}.pkl"), "rb") as f:
            pca = pickle.load(f)
        sub_npz = np.load(os.path.join(snap_dir, f"snapshot_{tag}.npz"))
        X_pca_sub = sub_npz["X_pca_sub"]
        U_sub = np.load(os.path.join(snap_dir, f"umap_{tag}.npz"))["U_sub"]

        X_pca_full = pca.transform(X_lat)
        if l2norm:
            X_pca_full = X_pca_full / (norm(X_pca_full, axis=1, keepdims=True) + 1e-8)
            X_pca_sub  = X_pca_sub  / (norm(X_pca_sub,  axis=1, keepdims=True) + 1e-8)

        k = min(max(5, args.n_neighbors), X_pca_sub.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_pca_sub)
        dists, idxs = nbrs.kneighbors(X_pca_full, return_distance=True)
        w = 1.0 / (dists + 1e-6); w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
        U_full = np.einsum("nk,nkd->nd", w, U_sub[idxs])

        print(f"[REF] Using latest snapshot '{tag}' to anchor reference layout.")
    else:
        pca = PCA(n_components=pca_dims, random_state=SEED, svd_solver="auto").fit(X_lat)
        X_pca_full = pca.transform(X_lat)
        if l2norm:
            X_pca_full = X_pca_full / (norm(X_pca_full, axis=1, keepdims=True) + 1e-8)
        ad_tmp = sc.AnnData(X=np.zeros((X_pca_full.shape[0],1), dtype=np.float32))
        ad_tmp.obsm["X_latent_pca"] = X_pca_full.astype(np.float32)
        sc.pp.neighbors(ad_tmp, use_rep="X_latent_pca", n_neighbors=min(args.n_neighbors, 30), metric="cosine")
        sc.tl.umap(ad_tmp, min_dist=0.6, spread=1.0, random_state=SEED)
        U_full = np.asarray(ad_tmp.obsm["X_umap"], dtype=np.float32)
        print("[REF] Built reference layout from full data.")
    return pca, X_pca_full, U_full

def save_metrics_row(save_dir, filename, row: dict):
    path = os.path.join(save_dir, filename)
    write_header = (not os.path.exists(path))
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path, nrows=0)
            cols = list(existing.columns)
            for k in row.keys():
                if k not in cols:
                    cols.append(k)
        except Exception:
            cols = list(row.keys())
    else:
        cols = list(row.keys())
    full_row = {k: row.get(k, "") for k in cols}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        w.writerow(full_row)
    print(f"[METRICS] appended -> {path}")

def save_checkpoint_on_plot(ep_idx: int, X_lat: np.ndarray, use_h: bool,
                            adv_acc_epoch: float, disc_acc_epoch: float,
                            gen_loss_epoch: float, disc_loss_epoch: float):
    pca_obj, X_pca_full, U_full = build_reference_layout(
        X_lat, args.pca_dims, args.pca_l2norm, args.save_dir,
        use_latest_snapshot=True
    )
    protos = compute_prototypes(X_lat, adata_all.obs["CellClass"])
    tag = f"ep{ep_idx:03d}_{_timestamp_tag()}"
    paths = save_all(
        save_dir=args.save_dir,
        model=model, disc=disc, args=args, adata_all=adata_all, qualified=qualified,
        pca_obj=pca_obj, embedding=X_lat, use_h=use_h,
        optimizer=optimizer, opt_disc=opt_disc,
        X_pca_full=X_pca_full, U_full=U_full,
        prototypes=protos, tag=tag
    )
    y_batch = pd.Categorical(adata_all.obs["dataset_batch"]).codes
    y_cell = pd.Categorical(adata_all.obs["CellClass"]).codes

    # Full evaluation every time we plot (sync with --plot-every)
    metrics = run_full_evaluation(X_lat, y_batch, y_cell)

    row = {
        "tag": tag,
        "method": "GRL",
        "epoch": ep_idx,
        "gen_loss": round(gen_loss_epoch, 6),
        "disc_loss": round(disc_loss_epoch, 6),
        "adv_acc": round(adv_acc_epoch, 6),
        "disc_acc": round(disc_acc_epoch, 6),
        **_metrics_row_pack(metrics),
        "use_h": bool(use_h),
        "pca_dims": int(args.pca_dims),
        "pca_l2norm": bool(args.pca_l2norm),
        "n_neighbors": int(args.n_neighbors),
        "embedding_file": os.path.basename(paths["embedding"]),
        "obs_file": os.path.basename(paths["obs"]),
        "pca_file": os.path.basename(paths["pca"]),
        "ref_file": os.path.basename(paths["ref"]),
        "checkpoint": os.path.basename(paths["ckpt"]),
        "prototypes_file": os.path.basename(paths["proto"])
    }
    save_metrics_row(args.save_dir, args.metrics_csv, row)
    _counts_table_to_csv(adata_crc, adata_ns, adata_cv, args.save_dir, tag)

@torch.no_grad()
def encode_all_latent(return_h=False):
    model.eval()
    outs = []
    bs = 1024
    for i in range(0, adata_all.X.shape[0], bs):
        X_block = adata_all.X[i:i+bs]
        from scipy import sparse as _sp
        if _sp.issparse(X_block):
            X_block = X_block.toarray()
        else:
            X_block = np.asarray(X_block)
        x = torch.tensor(X_block, dtype=torch.float32, device=device)
        if return_h:
            z, h = model.encode(x, return_h=True)
            outs.append(h.cpu())
        else:
            z = model.encode(x)
            outs.append(z.cpu())
        del x
    return torch.cat(outs).numpy()

def train_epoch(epoch, total_epochs):
    model.train(); disc.train()
    tot_gen, tot_disc = 0.0, 0.0
    tot_acc_adv, tot_acc_disc = 0.0, 0.0
    progress = epoch / max(1, total_epochs - 1)
    lam = dann_lambda(progress, lam_max=args.grl_lam_max, k=args.grl_k)

    for rna, bidx in loader:
        rna = rna.to(device); bidx = bidx.to(device)

        optimizer.zero_grad(set_to_none=True); opt_disc.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(device.type == "cuda")):
            recon, z, h = model(rna, add_noise_std=0.05, return_h=True)
            recon_loss = F.mse_loss(recon, rna)
            latent_reg = 1e-4 * (z.pow(2).mean())
            feat = (z if args.disc_on_z else h).float()

            for p in disc.parameters(): p.requires_grad_(False)
            logits_adv = disc(grad_reverse(feat, lambd=lam).float())
            try:
                adv_loss = F.cross_entropy(logits_adv, bidx, label_smoothing=0.05)
            except TypeError:
                adv_loss = F.cross_entropy(logits_adv, bidx)
            gen_loss = recon_loss + latent_reg + adv_loss

        scaler.scale(gen_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        tot_acc_adv += acc_from_logits(logits_adv, bidx)

        for p in disc.parameters(): p.requires_grad_(True)
        opt_disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            feat_det = (z if args.disc_on_z else h).detach().float()
        logits_disc = disc(feat_det)
        try:
            disc_loss = F.cross_entropy(logits_disc, bidx, label_smoothing=0.05)
        except TypeError:
            disc_loss = F.cross_entropy(logits_disc, bidx)
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt_disc.step()
        tot_acc_disc += acc_from_logits(logits_disc, bidx)

        tot_gen += float(gen_loss.item())
        tot_disc += float(disc_loss.item())

    nstep = len(loader)
    gen_avg = tot_gen / nstep
    disc_avg = tot_disc / nstep
    adv_acc_avg = tot_acc_adv / nstep
    disc_acc_avg = tot_acc_disc / nstep

    print(f"[Epoch {epoch+1:>2}] gen_loss={gen_avg:.4f}  disc_loss={disc_avg:.4f}  "
          f"adv_acc={adv_acc_avg:.3f}  disc_acc={disc_acc_avg:.3f}  lam={lam:.3f}")

    return gen_avg, disc_avg, adv_acc_avg, disc_acc_avg

# =================== 训练主循环 ===================
for ep in range(args.pretrain_epochs):
    gen_avg, disc_avg, adv_acc_avg, disc_acc_avg = train_epoch(ep, args.pretrain_epochs)

    do_plot = ((ep + 1) % args.plot_every == 0) or ((ep + 1) == args.pretrain_epochs)
    if do_plot:
        use_h = args.eval_with_h
        X_lat = encode_all_latent(return_h=use_h)
        which = "h(proj)" if use_h else "z"

        plot_umap_fast(
            adata_all, X_lat,
            suffix=f"epoch{ep+1:03d}_{which}",
            color_modes=("dataset","CellClass"),
            quick_sample_per_class=args.sample_per_class,
            n_neighbors=min(args.n_neighbors, 30),
            dpi=args.umap_dpi,
            try_rapids=args.try_rapids,
            reuse_prev_pca=args.reuse_pca
        )

        save_checkpoint_on_plot(
            ep_idx=ep+1, X_lat=X_lat, use_h=use_h,
            adv_acc_epoch=adv_acc_avg, disc_acc_epoch=disc_acc_avg,
            gen_loss_epoch=gen_avg, disc_loss_epoch=disc_avg
        )

# -------------------- 训练结束 final 保存 --------------------
use_h = args.eval_with_h
X_lat = encode_all_latent(return_h=use_h)
pca_obj, X_pca_full, U_full = build_reference_layout(
    X_lat, args.pca_dims, args.pca_l2norm, args.save_dir,
    use_latest_snapshot=True
)
prototypes = compute_prototypes(X_lat, adata_all.obs["CellClass"])
final_tag = f"final_{_timestamp_tag()}"

paths = save_all(
    save_dir=args.save_dir,
    model=model, disc=disc, args=args, adata_all=adata_all, qualified=qualified,
    pca_obj=pca_obj, embedding=X_lat, use_h=use_h,
    optimizer=optimizer, opt_disc=opt_disc,
    X_pca_full=X_pca_full, U_full=U_full,
    prototypes=prototypes, tag=final_tag
)

y_batch = pd.Categorical(adata_all.obs["dataset_batch"]).codes
y_cell = pd.Categorical(adata_all.obs["CellClass"]).codes
metrics_final = run_full_evaluation(X_lat, y_batch, y_cell)

save_metrics_row(args.save_dir, args.metrics_csv, {
    "tag": final_tag,
    "method": "GRL",
    "epoch": args.pretrain_epochs,
    "gen_loss": np.nan,
    "disc_loss": np.nan,
    "adv_acc": np.nan,
    "disc_acc": np.nan,
    **_metrics_row_pack(metrics_final),
    "use_h": bool(use_h),
    "pca_dims": int(args.pca_dims),
    "pca_l2norm": bool(args.pca_l2norm),
    "n_neighbors": int(args.n_neighbors),
    "embedding_file": os.path.basename(paths["embedding"]),
    "obs_file": os.path.basename(paths["obs"]),
    "pca_file": os.path.basename(paths["pca"]),
    "ref_file": os.path.basename(paths["ref"]),
    "checkpoint": os.path.basename(paths["ckpt"]),
    "prototypes_file": os.path.basename(paths["proto"])
})

print("\n[DONE] 训练完成；scatter 由 sc.pl.umap 渲染（与原版完全一致），仅 axes 改为箭头轴；")
print("       metrics 含 ARI/NMI/ASW_cell/ASW_batch/GraphConn/kBET，每次 plot 同步写入 metrics_log.csv。")