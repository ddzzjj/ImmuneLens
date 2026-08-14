#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import argparse
import warnings
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import mannwhitneyu, fisher_exact, spearmanr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, to_hex

warnings.filterwarnings("ignore")


DEFAULT_RNA_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA.h5ad"
DEFAULT_META_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA_metadata.csv"
DEFAULT_TCR_PATH = "./data/nsclc/GSE243013_T_with_TCR_annotation.csv"
DEFAULT_SAMPLE_META_PATH = "./data/nsclc/meta2.csv"
DEFAULT_OUT_DIR = "./result/nsclcclassvae2"

AASET = set(list("ACDEFGHIKLMNPQRSTVWY"))
RASTER_DPI = 300


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.compression"] = 6
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PAL = {
    "fused": "#D9742B",
    "rna": "#2A8C5E",
    "tcr": "#3F7BB6",
    "high": "#D9742B",
    "low": "#BFC3C8",
    "responder": "#D9742B",
    "nonresponder": "#7E8A97",
    "text": "#1F1F1F",
    "muted": "#6A6A6A",
    "axis": "#2B2B2B",
    "grid": "#E8EAEE",
}
HEAT_CMAP = plt.get_cmap("RdBu_r")

matplotlib.rcParams.update({
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelcolor": PAL["text"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": PAL["axis"],
    "figure.dpi": 120,
})

SIGNATURES: Dict[str, List[str]] = {
    "cytotoxicity": ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CST7", "CTSW", "KLRD1", "KLRG1"],
    "exhaustion_activation": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "CXCL13", "ENTPD1", "IFNG", "TNF", "CD69"],
    "proliferation": ["MKI67", "TOP2A", "STMN1", "TYMS", "PCNA", "HMGB2", "UBE2C", "MCM5"],
    "naive_memory": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "LTB"],
}

MARKER_GENES = [
    "CXCL13", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1", "CTLA4",
    "GZMB", "GZMH", "NKG7", "GNLY", "PRF1", "IFNG", "TNF",
    "MKI67", "TOP2A", "TCF7", "IL7R", "CCR7", "SELL", "LAYN",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_aa(seq: str) -> str:
    if pd.isna(seq):
        return ""
    s = str(seq).upper().strip()
    if s in {"", "NONE", "NAN", "NULL", "NA", "<NA>"}:
        return ""
    return "".join(ch for ch in s if ch in AASET)


def obs_to_str(s: pd.Series, fill: str = "") -> pd.Series:
    s = s.astype("object")
    s = s.where(pd.notna(s), fill)
    s = s.replace({"nan": fill, "NaN": fill, "None": fill, "<NA>": fill, "NONE": fill})
    return s.astype(str)


def find_first_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.strip().lower()
        for c in cols:
            if key in str(c).strip().lower():
                return c
    if required:
        raise KeyError(f"Cannot find any of columns {list(candidates)} in {cols}")
    return None


def _split_terms(s: str) -> List[str]:
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]


def parse_response_to_label(x, positive_terms: str = "pCR,MPR", negative_terms: str = "non-MPR") -> Optional[int]:

    if pd.isna(x):
        return None
    raw = str(x).strip()
    if raw == "":
        return None
    s = raw.lower().replace("_", "-").replace(" ", "")
    pos = [t.replace("_", "-").replace(" ", "") for t in _split_terms(positive_terms)]
    neg = [t.replace("_", "-").replace(" ", "") for t in _split_terms(negative_terms)]
    if s in pos:
        return 1
    if s in neg:
        return 0
    if s in {"pcr", "mpr", "majorpathologicalresponse", "pathologicalcompleteresponse"}:
        return 1
    if s in {"non-mpr", "nonmpr", "sd", "pd", "nonresponse", "non-responder", "nonresponder"}:
        return 0
    return None


def _read_table_auto(path: str, dtype=str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=dtype)
    else:
        df = pd.read_csv(path, sep=None, engine="python", dtype=dtype)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    ss = s.astype(str).str.strip().str.upper()
    return ss.isin(["TRUE", "T", "1", "YES", "Y", "EXPANDED", "CLONAL"])


def _format_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "p<0.0001"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3f}"


def _p_to_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _nice_name(x: str) -> str:
    x = str(x).replace("score_", "").replace("_", " ")
    return x[:1].upper() + x[1:]


def _clean_axis(ax, grid: bool = False, grid_axis: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.5, width=0.7, color=PAL["axis"], labelcolor=PAL["text"])
    if grid:
        ax.grid(True, axis=grid_axis, color=PAL["grid"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def _savefig(path: str, dpi: int = RASTER_DPI):
    if not str(path).lower().endswith(".pdf"):
        path = os.path.splitext(path)[0] + ".pdf"
    plt.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.12,
                facecolor="white", edgecolor="white", transparent=False)
    plt.close()
    try:
        print(f"  [PDF] {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
    except Exception:
        pass

def make_clone_id_df(obs: pd.DataFrame, clone_key: str = "clonotype") -> pd.Series:
    clone_key_raw = str(clone_key)
    clone_key_upper = clone_key_raw.upper()
    if clone_key_raw in obs.columns:
        s = obs[clone_key_raw].fillna("").astype(str)
        trb = obs.get("TRB_aa", pd.Series("", index=obs.index)).fillna("").astype(str)
        tra = obs.get("TRA_aa", pd.Series("", index=obs.index)).fillna("").astype(str)
        fallback = tra + "|" + trb
        return s.where(s.str.len() > 0, fallback)
    if clone_key_upper == "TRB":
        return obs["TRB_aa"].fillna("").astype(str)
    if clone_key_upper in {"TRA_TRB", "TRA+TRB", "PAIRED"}:
        tra = obs["TRA_aa"].fillna("").astype(str)
        trb = obs["TRB_aa"].fillna("").astype(str)
        return tra + "|" + trb
    if "TCR_clone.id" in obs.columns:
        return obs["TCR_clone.id"].fillna("").astype(str)
    return obs["TRB_aa"].fillna("").astype(str)


def load_nsclc_response_data(
    rna_path: str,
    meta_path: str,
    tcr_path: str,
    sample_meta_path: str,
    bag_level: str = "sample",
    require_tcr: bool = True,
    celltype_filter: str = "CD8",
    cancer_type: str = "LUSC",
    clone_key: str = "clonotype",
    response_positive_terms: str = "pCR,MPR",
    response_negative_terms: str = "non-MPR",
    min_genes: int = 200,
    remove_tcr_ig_genes: bool = True,
) -> sc.AnnData:
    bag_level = str(bag_level).lower()
    if bag_level not in {"sample", "patient"}:
        raise ValueError("--bag_level must be sample or patient")

    print("[Load] NSCLC scRNA-seq h5ad ...")
    ad = sc.read_h5ad(rna_path)
    ad.obs_names = ad.obs_names.astype(str).str.strip()
    ad.obs["cell_id_original"] = ad.obs_names.astype(str)

    print("[Load] cell annotation ...")
    meta_df = _read_table_auto(meta_path, dtype=str).fillna("")
    cell_col = find_first_col(meta_df, ["cellID", "cell_id", "cell", "cellName", "barcode"])
    meta_df["cell_id"] = meta_df[cell_col].map(normalize_id)
    meta_df = meta_df.drop_duplicates("cell_id", keep="first")

    harmonize = {
        "sub_cell_type": ["sub_cell_type", "sub_celltype", "celltype", "cell_type"],
        "major_cell_type": ["major_cell_type", "majorType", "major_celltype", "major"],
        "cancer_type": ["cancer_type", "Cancer_Type", "histology"],
        "sampleID": ["sampleID", "sample_id", "sample", "Tumor_Sample_Barcode"],
        "PatientID": ["PatientID", "patientID", "patient_id", "patient", "donor"],
    }
    for new_col, candidates in harmonize.items():
        c = find_first_col(meta_df, candidates, required=False)
        if c is not None:
            meta_df[new_col] = meta_df[c].astype(str).str.strip()
        elif new_col not in meta_df.columns:
            meta_df[new_col] = ""

    print(f"[meta] rows={len(meta_df)}, columns={list(meta_df.columns)}")
    if "cancer_type" in meta_df.columns:
        print("[meta] cancer_type distribution:")
        print(meta_df["cancer_type"].astype(str).value_counts(dropna=False).head(20))
    if "sub_cell_type" in meta_df.columns:
        print("[meta] sub_cell_type distribution:")
        print(meta_df["sub_cell_type"].astype(str).value_counts(dropna=False).head(30))

    print("[Load] TCR annotation ...")
    tcr_df = _read_table_auto(tcr_path, dtype=str).fillna("")
    tcr_cell_col = find_first_col(tcr_df, ["cellID", "cell_id", "cell", "cellBarcode", "barcode"])
    tcr_sample_col = find_first_col(tcr_df, ["sampleID", "sample_id", "sample", "Tumor_Sample_Barcode"], required=False)
    tcr_patient_col = find_first_col(tcr_df, ["PatientID", "patientID", "patient_id", "patient", "donor"], required=False)
    tra_col = find_first_col(tcr_df, ["TRA_cdr3", "TCRA_cdr3", "TRA_cdr3aa", "TCRA_cdr3aa", "TRA_aa"], required=False)
    trb_col = find_first_col(tcr_df, ["TRB_cdr3", "TCRB_cdr3", "TRB_cdr3aa", "TCRB_cdr3aa", "TRB_aa"], required=True)

    tcr_df["cell_id"] = tcr_df[tcr_cell_col].map(normalize_id)
    tcr_df["sampleID"] = tcr_df[tcr_sample_col].map(normalize_id) if tcr_sample_col is not None else ""
    tcr_df["PatientID"] = tcr_df[tcr_patient_col].map(normalize_id) if tcr_patient_col is not None else ""
    tcr_df["TRA_aa"] = tcr_df[tra_col].map(clean_aa) if tra_col is not None else ""
    tcr_df["TRB_aa"] = tcr_df[trb_col].map(clean_aa)

    exp_col = find_first_col(tcr_df, ["expansion", "is_expanded", "expanded", "clonal"], required=False)
    if exp_col is not None:
        tcr_df["is_expanded"] = parse_bool_series(tcr_df[exp_col]).astype(int)
    else:
        tcr_df["is_expanded"] = np.nan

    optional_cols = [
        "clonotype", "clonotype_number", "T_new_name", "expansion",
        "TRA_v_gene", "TRA_j_gene", "TRA_c_gene", "TRA_cdr3",
        "TRB_v_gene", "TRB_j_gene", "TRB_c_gene", "TRB_cdr3",
        "TCR_clone.id", "TCR_pclone.id", "TCR_sclone.id",
    ]
    for c in optional_cols:
        if c in tcr_df.columns:
            tcr_df[c] = tcr_df[c].astype(str).str.strip()

    tcr_df = tcr_df[tcr_df["TRB_aa"].str.len() > 0].copy()
    tcr_df = tcr_df.sort_values(["TRB_aa"], ascending=True).drop_duplicates("cell_id", keep="first")
    print(f"[TCR] valid TRB entries: {len(tcr_df)}")
    if len(tcr_df):
        print(f"[TCR] valid TRA ratio: {(tcr_df['TRA_aa'].str.len() > 0).mean():.3f}")
        if pd.Series(tcr_df["is_expanded"]).notna().any():
            print(f"[TCR] expanded ratio: {pd.to_numeric(tcr_df['is_expanded'], errors='coerce').fillna(0).mean():.3f}")

    print("[Load] sample metadata / pathological response ...")
    sample_df = _read_table_auto(sample_meta_path, dtype=str).fillna("")
    sm_sample_col = find_first_col(sample_df, ["Tumor_Sample_Barcode", "sampleID", "sample_id", "sample"])
    sm_response_col = find_first_col(sample_df, ["Pathological Response", "Response", "pathological_response", "response"])
    sm_patient_col = find_first_col(sample_df, ["PatientID", "patientID", "patient_id", "patient", "donor"], required=False)
    sample_df["sampleID"] = sample_df[sm_sample_col].map(normalize_id)
    sample_df["Response"] = sample_df[sm_response_col].astype(str).str.strip()
    sample_df["PatientID"] = sample_df[sm_patient_col].map(normalize_id) if sm_patient_col is not None else sample_df["sampleID"]
    sample_df["y"] = sample_df["Response"].map(lambda v: parse_response_to_label(v, response_positive_terms, response_negative_terms))
    sample_df = sample_df[sample_df["y"].notna()].copy()
    sample_df["y"] = sample_df["y"].astype(int)
    sample_df = sample_df.drop_duplicates("sampleID", keep="first")
    print("[Response] sample-level distribution:")
    print(sample_df["Response"].value_counts(dropna=False))
    print("[Response] binary label distribution:")
    print(sample_df["y"].value_counts().rename({1: "effective(pCR/MPR)", 0: "non-MPR"}))

    print("[Merge] annotations into AnnData.obs ...")
    obs = pd.DataFrame(index=ad.obs_names)
    obs.index.name = "cell_id"
    meta_keep_cols = [c for c in meta_df.columns if c not in {cell_col, "cell_id"}]
    meta2 = meta_df.drop_duplicates("cell_id", keep="first").set_index("cell_id")[meta_keep_cols]
    obs = obs.merge(meta2, left_index=True, right_index=True, how="left", suffixes=("", "_meta"))

    tcr_keep_cols = ["sampleID", "PatientID", "TRA_aa", "TRB_aa", "is_expanded"]
    for c in optional_cols:
        if c in tcr_df.columns and c not in tcr_keep_cols:
            tcr_keep_cols.append(c)
    tcr2 = tcr_df.set_index("cell_id")[tcr_keep_cols]
    obs = obs.merge(tcr2, left_index=True, right_index=True, how="left", suffixes=("", "_tcr"))

    if "sampleID" not in obs.columns:
        obs["sampleID"] = ""
    if "PatientID" not in obs.columns:
        obs["PatientID"] = ""
    if "sampleID_tcr" in obs.columns:
        obs["sampleID"] = obs["sampleID"].where(obs["sampleID"].astype(str).str.len() > 0, obs["sampleID_tcr"])
    if "PatientID_tcr" in obs.columns:
        obs["PatientID"] = obs["PatientID"].where(obs["PatientID"].astype(str).str.len() > 0, obs["PatientID_tcr"])

    for c in ["sampleID", "PatientID", "sub_cell_type", "major_cell_type", "cancer_type", "TRA_aa", "TRB_aa"]:
        if c not in obs.columns:
            obs[c] = ""
        obs[c] = obs_to_str(obs[c], fill="")

    obs = obs.merge(sample_df.set_index("sampleID")[["PatientID", "Response", "y"]],
                    left_on="sampleID", right_index=True, how="left", suffixes=("", "_samplemeta"))
    if "PatientID_samplemeta" in obs.columns:
        obs["PatientID"] = obs["PatientID_samplemeta"].where(
            obs["PatientID_samplemeta"].astype(str).str.len() > 0,
            obs["PatientID"].astype(str),
        )

    obs = obs.reindex(ad.obs_names)
    for c in obs.columns:
        ad.obs[c] = obs[c].values
    ad.obs["cell_id"] = ad.obs_names.astype(str)

    cancer_type = str(cancer_type).upper()
    print(f"\n[Filter] cancer_type={cancer_type}, celltype_filter={celltype_filter}, require_tcr={require_tcr}")
    if cancer_type in {"", "ALL", "NONE"}:
        keep_cancer = pd.Series(True, index=ad.obs_names)
    else:
        keep_cancer = ad.obs["cancer_type"].astype(str).str.upper().eq(cancer_type)

    filt = str(celltype_filter).strip()
    if filt.upper() in {"", "ALL", "NONE"}:
        keep_celltype = pd.Series(True, index=ad.obs_names)
    else:
        fields = []
        for c in ["sub_cell_type", "celltype", "cell_type", "major_cell_type", "majorType"]:
            if c in ad.obs.columns:
                fields.append(ad.obs[c].astype(str))
        if not fields:
            raise ValueError("No cell type column found for --celltype_filter")
        keep_celltype = fields[0].str.contains(filt, case=False, regex=True, na=False)
        for s in fields[1:]:
            keep_celltype = keep_celltype | s.str.contains(filt, case=False, regex=True, na=False)

    has_trb = ad.obs["TRB_aa"].fillna("").astype(str).str.len() > 0
    has_label = pd.notna(ad.obs["y"])
    keep = keep_cancer & keep_celltype & has_label
    if require_tcr:
        keep = keep & has_trb

    print(f"[Filter] before any filter:      {ad.n_obs}")
    print(f"[Filter] after cancer:          {int(keep_cancer.sum())}")
    print(f"[Filter] after celltype:        {int((keep_cancer & keep_celltype).sum())}")
    print(f"[Filter] after response label:  {int((keep_cancer & keep_celltype & has_label).sum())}")
    print(f"[Filter] after TRB required:    {int((keep_cancer & keep_celltype & has_label & has_trb).sum())}")

    ad = ad[keep.values].copy()
    if ad.n_obs == 0:
        raise ValueError("No cells left after filters. Check cancer_type/celltype/require_tcr/sample metadata matching.")
    if min_genes and min_genes > 0:
        sc.pp.filter_cells(ad, min_genes=int(min_genes))

    if remove_tcr_ig_genes:
        tcr_ig_pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", flags=re.IGNORECASE)
        keep_genes = [g for g in ad.var_names if not tcr_ig_pattern.match(str(g))]
        removed = ad.n_vars - len(keep_genes)
        ad = ad[:, keep_genes].copy()
        print(f"[Filter] removed {removed} TCR/IG genes; genes={ad.n_vars}")

    ad.obs["y"] = ad.obs["y"].astype(int)
    ad.obs["Response"] = obs_to_str(ad.obs["Response"], fill="")
    ad.obs["severity"] = ad.obs["Response"].astype(str)
    ad.obs["Response_binary"] = ad.obs["y"].map({1: "effective_response", 0: "non_MPR"}).astype(str)
    ad.obs["sampleID"] = obs_to_str(ad.obs["sampleID"], fill="")
    ad.obs["PatientID"] = obs_to_str(ad.obs["PatientID"], fill="")
    ad.obs["PatientID"] = ad.obs["PatientID"].where(ad.obs["PatientID"].str.len() > 0, ad.obs["sampleID"])
    ad.obs["TRA_aa"] = obs_to_str(ad.obs["TRA_aa"], fill="")
    ad.obs["TRB_aa"] = obs_to_str(ad.obs["TRB_aa"], fill="")

    ad.obs["clone_id"] = make_clone_id_df(ad.obs, clone_key=clone_key).astype(str).values
    ad.obs["clone_id"] = ad.obs["clone_id"].where(ad.obs["clone_id"].astype(str).str.len() > 0, ad.obs["TRB_aa"].astype(str))
    ad.obs["clone_size"] = (
        ad.obs.groupby(["sampleID", "clone_id"], observed=True)["clone_id"]
        .transform("size")
        .fillna(1)
        .astype(int)
    )
    if "is_expanded" not in ad.obs.columns:
        ad.obs["is_expanded"] = (ad.obs["clone_size"].astype(int) > 1).astype(int)
    else:
        ad.obs["is_expanded"] = pd.to_numeric(ad.obs["is_expanded"], errors="coerce")
        ad.obs["is_expanded"] = ad.obs["is_expanded"].where(
            ad.obs["is_expanded"].notna(),
            (ad.obs["clone_size"].astype(int) > 1).astype(int),
        ).fillna(0).astype(int)

    if bag_level == "patient":
        ad.obs["bag_id"] = ad.obs["PatientID"].astype(str)
    else:
        ad.obs["bag_id"] = ad.obs["sampleID"].astype(str)
    ad.obs["group_id"] = ad.obs["PatientID"].astype(str)

    valid_bag = ad.obs["bag_id"].astype(str).str.len() > 0
    valid_group = ad.obs["group_id"].astype(str).str.len() > 0
    ad = ad[(valid_bag & valid_group).values].copy()
    bag_label_nunique = ad.obs.groupby("bag_id", observed=True)["y"].nunique()
    good_bags = set(bag_label_nunique[bag_label_nunique == 1].index.astype(str))
    ad = ad[ad.obs["bag_id"].astype(str).isin(good_bags).values].copy()

    print("\n[Final NSCLC response data]")
    print(f"cells={ad.n_obs}, genes={ad.n_vars}")
    print(f"bags={ad.obs['bag_id'].nunique()}, patients/groups={ad.obs['group_id'].nunique()}")
    print("[Response counts by bag]")
    print(make_bag_table(ad)["y"].value_counts().rename({1: "effective(pCR/MPR)", 0: "non-MPR"}))
    if "sub_cell_type" in ad.obs.columns:
        print("[sub_cell_type after filter]")
        print(ad.obs["sub_cell_type"].astype(str).value_counts().head(30))
    print(f"[is_expanded ratio] {pd.to_numeric(ad.obs['is_expanded'], errors='coerce').fillna(0).mean():.3f}")
    return ad


def make_bag_table(ad: sc.AnnData) -> pd.DataFrame:
    rows = []
    obs = ad.obs.copy()
    if "clone_id" not in obs.columns:
        obs["clone_id"] = make_clone_id_df(obs).values
    for bid, g in obs.groupby("bag_id", observed=True):
        y = int(g["y"].astype(int).iloc[0])
        rows.append({
            "bag_id": str(bid),
            "sampleID": str(g["sampleID"].iloc[0]),
            "PatientID": str(g["PatientID"].iloc[0]),
            "group_id": str(g["group_id"].iloc[0]) if "group_id" in g.columns else str(g["PatientID"].iloc[0]),
            "y": y,
            "Response": str(g["Response"].iloc[0]) if "Response" in g.columns else "",
            "response_binary": "effective_response" if y == 1 else "non_MPR",
            "n_cells": int(len(g)),
            "n_clones": int(g["clone_id"].nunique()),
        })
    return pd.DataFrame(rows).set_index("bag_id")


def load_ablation_table(out_dir: str) -> pd.DataFrame:
    candidates = [
        os.path.join(out_dir, "ablation_table.csv"),
        os.path.join(out_dir, "patient_level_metrics_summary.csv"),
    ]
    for sub in ["fused", "rna_only", "tcr_only"]:
        candidates.append(os.path.join(out_dir, sub, "patient_level_metrics_summary.csv"))
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if {"method", "AUROC_mean", "AUPRC_mean"}.issubset(df.columns):
                print(f"[Load] performance table: {p}")
                return df

    rows = []
    for sub in ["rna_only", "tcr_only", "fused"]:
        p = os.path.join(out_dir, sub, "patient_level_metrics_summary.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "method" not in d.columns:
                d.insert(0, "method", sub)
            rows.append(d)
    if rows:
        return pd.concat(rows, axis=0, ignore_index=True)
    raise FileNotFoundError(f"Cannot find ablation metrics under {out_dir}")


def load_predictions(out_dir: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for m in ["rna_only", "tcr_only", "fused"]:
        p = os.path.join(out_dir, m, "patient_level_predictions.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "y_pred" not in d.columns and "prob" in d.columns:
                if "threshold" in d.columns:
                    d["y_pred"] = (d["prob"].astype(float) >= d["threshold"].astype(float)).astype(int)
                elif "selected_threshold" in d.columns:
                    d["y_pred"] = (d["prob"].astype(float) >= d["selected_threshold"].astype(float)).astype(int)
                else:
                    d["y_pred"] = (d["prob"].astype(float) >= 0.5).astype(int)
            out[m] = d
            print(f"[Load] {m} predictions: {p}, rows={len(d)}")
    return out


def load_clone_evidence(out_dir: str, method: str = "fused") -> pd.DataFrame:
    p = os.path.join(out_dir, method, "clone_evidence_all_folds.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Cannot find {p}. Re-run training with --export_clone_evidence.")
    ce = pd.read_csv(p)
    print(f"[Load] clone evidence: {p}, rows={len(ce)}")
    return ce


def standardize_evidence_columns(ce: pd.DataFrame, evidence_col: str = "auto") -> Tuple[pd.DataFrame, str]:
    ce = ce.copy()

    if "clone_attn" in ce.columns and "clone_attention" not in ce.columns:
        ce["clone_attention"] = ce["clone_attn"]
    if "clone_attention" in ce.columns and "clone_attn" not in ce.columns:
        ce["clone_attn"] = ce["clone_attention"]

    for c in ["clone_prob", "clone_attention", "clone_attn", "clone_size"]:
        if c in ce.columns:
            ce[c] = pd.to_numeric(ce[c], errors="coerce")

    if "weighted_noisy_or_evidence" not in ce.columns:
        if {"clone_prob", "clone_attention", "clone_size"}.issubset(ce.columns):
            ce["weighted_noisy_or_evidence"] = (
                ce["clone_prob"].fillna(0).clip(0, 1)
                * ce["clone_attention"].fillna(0).clip(lower=0)
                * np.log1p(ce["clone_size"].fillna(1).clip(lower=1))
            )
        elif {"clone_prob", "clone_attention"}.issubset(ce.columns):
            ce["weighted_noisy_or_evidence"] = ce["clone_prob"].fillna(0).clip(0, 1) * ce["clone_attention"].fillna(0).clip(lower=0)
        elif "clone_prob" in ce.columns:
            ce["weighted_noisy_or_evidence"] = ce["clone_prob"].fillna(0).clip(0, 1)

    if "scaled_weighted_noisy_or_evidence" not in ce.columns and "weighted_noisy_or_evidence" in ce.columns:
        group_cols = [c for c in ["fold", "bag_id"] if c in ce.columns]
        if not group_cols:
            group_cols = ["sampleID"] if "sampleID" in ce.columns else []
        if group_cols:
            def zscore(g):
                v = pd.to_numeric(g["weighted_noisy_or_evidence"], errors="coerce")
                sd = float(v.std(ddof=0))
                if not np.isfinite(sd) or sd < 1e-12:
                    return pd.Series(np.zeros(len(v)), index=g.index)
                return (v - float(v.mean())) / sd
            ce["scaled_weighted_noisy_or_evidence"] = ce.groupby(group_cols, group_keys=False).apply(zscore)
        else:
            v = pd.to_numeric(ce["weighted_noisy_or_evidence"], errors="coerce")
            ce["scaled_weighted_noisy_or_evidence"] = (v - v.mean()) / (v.std(ddof=0) + 1e-12)

    if evidence_col == "auto":
        for c in ["weighted_noisy_or_evidence", "clone_contribution", "clone_prob", "clone_attention"]:
            if c in ce.columns:
                evidence_col = c
                break
    if evidence_col not in ce.columns:
        raise KeyError(f"evidence_col={evidence_col} not found. Available columns: {list(ce.columns)}")
    print(f"[Evidence] using evidence_col={evidence_col}")
    return ce, evidence_col


def annotate_clone_evidence(ce: pd.DataFrame, ad: sc.AnnData, evidence_col: str, top_frac: float) -> pd.DataFrame:
    print("[Annotate] clone evidence with raw TCR expansion metadata ...")
    ce = ce.copy()
    if "bag_id" not in ce.columns and "sampleID" in ce.columns:
        ce["bag_id"] = ce["sampleID"].astype(str)
    ce["sampleID"] = ce["sampleID"].astype(str)
    ce["clone_id"] = ce["clone_id"].astype(str)

    obs = ad.obs.copy()
    clone_ann = []
    for (sample_id, clone_id), g in obs.groupby(["sampleID", "clone_id"], observed=True):
        sample_n = int((obs["sampleID"].astype(str) == str(sample_id)).sum())
        row = {
            "sampleID": str(sample_id),
            "clone_id": str(clone_id),
            "PatientID_raw": str(g["PatientID"].iloc[0]) if "PatientID" in g.columns else "",
            "Response_raw": str(g["Response"].iloc[0]) if "Response" in g.columns else "",
            "y_raw": int(g["y"].astype(int).iloc[0]) if "y" in g.columns else np.nan,
            "n_cells_raw": int(len(g)),
            "clone_cell_fraction_raw": float(len(g) / max(1, sample_n)),
            "TRB_aa_nunique": int(g["TRB_aa"].astype(str).nunique()) if "TRB_aa" in g.columns else np.nan,
            "is_expanded_raw": bool(pd.to_numeric(g.get("is_expanded", pd.Series(0, index=g.index)), errors="coerce").fillna(0).astype(int).max() > 0),
        }
        for c in ["TCR_sclone.freq", "TCR_pclone.freq", "TCR_clone.freq"]:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").max()) if c in g.columns else np.nan
        for c in ["TCR_sclonal", "TCR_pclonal", "TCR_clonal"]:
            row[c] = bool(parse_bool_series(pd.Series(g[c])).max()) if c in g.columns else False
        clone_ann.append(row)
    clone_ann = pd.DataFrame(clone_ann)

    ann = ce.merge(clone_ann, on=["sampleID", "clone_id"], how="left")
    ann[evidence_col] = pd.to_numeric(ann[evidence_col], errors="coerce")

    group_cols = [c for c in ["fold", "bag_id"] if c in ann.columns]
    if not group_cols:
        group_cols = ["sampleID"]

    def mark_high(g: pd.DataFrame) -> pd.DataFrame:
        vals = pd.to_numeric(g[evidence_col], errors="coerce").fillna(-np.inf)
        n_top = max(1, int(np.ceil(len(g) * float(top_frac))))
        order = vals.rank(method="first", ascending=False)
        g = g.copy()
        g["high_evidence"] = order <= n_top
        g["evidence_percentile_within_bag"] = vals.rank(pct=True, method="average")
        return g

    ann = ann.groupby(group_cols, group_keys=False, observed=True).apply(mark_high)
    ann["expanded_by_model_clone_size"] = pd.to_numeric(ann.get("clone_size", np.nan), errors="coerce") > 1
    ann["expanded_by_raw_cells"] = pd.to_numeric(ann.get("n_cells_raw", np.nan), errors="coerce") > 1
    ann["expanded_by_is_expanded_raw"] = ann.get("is_expanded_raw", False)
    if isinstance(ann["expanded_by_is_expanded_raw"], pd.Series):
        ann["expanded_by_is_expanded_raw"] = ann["expanded_by_is_expanded_raw"].fillna(False).astype(bool)
    return ann


def test_high_vs_low_clones(ann: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    print("[Stats] high-evidence clone enrichment ...")
    rows = []
    high = ann[ann["high_evidence"] == True]
    low = ann[ann["high_evidence"] == False]

    bool_cols = [
        "expanded_by_model_clone_size", "expanded_by_raw_cells", "expanded_by_is_expanded_raw",
        "TCR_sclonal", "TCR_pclonal", "TCR_clonal",
    ]
    for c in bool_cols:
        if c not in ann.columns:
            continue
        hb = parse_bool_series(high[c]) if not pd.api.types.is_bool_dtype(high[c]) else high[c].fillna(False).astype(bool)
        lb = parse_bool_series(low[c]) if not pd.api.types.is_bool_dtype(low[c]) else low[c].fillna(False).astype(bool)
        if len(hb) == 0 or len(lb) == 0:
            continue
        table = np.array([[(~lb).sum(), lb.sum()], [(~hb).sum(), hb.sum()]])
        odds, p = fisher_exact(table)
        rows.append({
            "test": "fisher_high_vs_low",
            "feature": c,
            "high_mean_or_rate": float(hb.mean()),
            "low_mean_or_rate": float(lb.mean()),
            "odds_ratio_high_vs_low": float(odds),
            "p_value": float(p),
            "n_high": int(len(hb)),
            "n_low": int(len(lb)),
            "high_true": int(hb.sum()),
            "low_true": int(lb.sum()),
        })

    num_cols = list(dict.fromkeys([
        "clone_size", "n_cells_raw", "clone_cell_fraction_raw", "TCR_sclone.freq", "TCR_pclone.freq", "TCR_clone.freq",
        "clone_prob", "clone_attention", "clone_attn", "clone_contribution", evidence_col,
    ]))
    for c in num_cols:
        if c not in ann.columns:
            continue
        x = pd.to_numeric(high[c], errors="coerce").dropna()
        y = pd.to_numeric(low[c], errors="coerce").dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "test": "mannwhitney_high_vs_low",
            "feature": c,
            "high_mean_or_rate": float(x.mean()),
            "low_mean_or_rate": float(y.mean()),
            "high_median": float(x.median()),
            "low_median": float(y.median()),
            "u_stat": float(stat),
            "p_value": float(p),
            "n_high": int(len(x)),
            "n_low": int(len(y)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("p_value")
    return out


def _maybe_log_normalize(ad: sc.AnnData) -> sc.AnnData:
    ad_score = ad.copy()
    try:
        max_val = float(ad_score.X[: min(200, ad_score.n_obs)].max())
    except Exception:
        max_val = 0.0
    if max_val > 30:
        sc.pp.normalize_total(ad_score, target_sum=1e4)
        sc.pp.log1p(ad_score)
    return ad_score


def compute_signature_scores(ad: sc.AnnData, ann: pd.DataFrame) -> pd.DataFrame:
    print("[Stats] clone-level signature scores ...")
    ad_score = _maybe_log_normalize(ad)
    available = set(map(str.upper, ad_score.var_names.astype(str)))
    upper_to_real = {str(g).upper(): str(g) for g in ad_score.var_names.astype(str)}
    scored = []
    for name, genes in SIGNATURES.items():
        genes_use = [upper_to_real[g.upper()] for g in genes if g.upper() in available]
        if len(genes_use) < 2:
            print(f"  [Warn] skip signature {name}: only {len(genes_use)} genes found")
            continue
        sc.tl.score_genes(ad_score, gene_list=genes_use, score_name=f"score_{name}", use_raw=False)
        scored.append(f"score_{name}")
    if not scored:
        raise RuntimeError("No signature score could be computed. Check gene symbols in h5ad.")

    obs = ad_score.obs.copy()
    clone_scores = obs.groupby(["sampleID", "clone_id"], observed=True)[scored].mean().reset_index()
    clone_scores["sampleID"] = clone_scores["sampleID"].astype(str)
    clone_scores["clone_id"] = clone_scores["clone_id"].astype(str)

    merge_cols = [
        "fold", "bag_id", "sampleID", "PatientID", "Response", "severity", "y_true", "clone_id", "high_evidence",
        "clone_prob", "clone_attention", "clone_attn", "weighted_noisy_or_evidence", "clone_size", "n_cells_raw",
        "expanded_by_model_clone_size", "expanded_by_raw_cells", "expanded_by_is_expanded_raw",
    ]
    merge_cols = [c for c in merge_cols if c in ann.columns]
    out = ann[merge_cols].drop_duplicates().merge(clone_scores, on=["sampleID", "clone_id"], how="left")
    return out


def test_signature_scores(sig: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [c for c in sig.columns if c.startswith("score_")]
    rows_hl, rows_resp = [], []
    high = sig[sig["high_evidence"] == True]
    low = sig[sig["high_evidence"] == False]
    for c in score_cols:
        x = pd.to_numeric(high[c], errors="coerce").dropna()
        y = pd.to_numeric(low[c], errors="coerce").dropna()
        if len(x) >= 2 and len(y) >= 2:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            rows_hl.append({
                "signature": c,
                "high_mean": float(x.mean()), "low_mean": float(y.mean()),
                "high_median": float(x.median()), "low_median": float(y.median()),
                "delta_high_minus_low": float(x.mean() - y.mean()),
                "u_stat": float(stat), "p_value": float(p),
                "n_high": int(len(x)), "n_low": int(len(y)),
            })
        if "y_true" in high.columns:
            resp = high[high["y_true"].astype(int) == 1]
            non = high[high["y_true"].astype(int) == 0]
            xr = pd.to_numeric(resp[c], errors="coerce").dropna()
            yn = pd.to_numeric(non[c], errors="coerce").dropna()
            if len(xr) >= 2 and len(yn) >= 2:
                stat, p = mannwhitneyu(xr, yn, alternative="two-sided")
                rows_resp.append({
                    "signature": c,
                    "responder_high_mean": float(xr.mean()), "nonresponder_high_mean": float(yn.mean()),
                    "responder_high_median": float(xr.median()), "nonresponder_high_median": float(yn.median()),
                    "delta_responder_minus_nonresponder": float(xr.mean() - yn.mean()),
                    "u_stat": float(stat), "p_value": float(p),
                    "n_responder_high": int(len(xr)), "n_nonresponder_high": int(len(yn)),
                })
    out1 = pd.DataFrame(rows_hl)
    out2 = pd.DataFrame(rows_resp)
    if not out1.empty:
        out1 = out1.sort_values("p_value")
    if not out2.empty:
        out2 = out2.sort_values("p_value")
    return out1, out2


def compute_marker_gene_summary(ad: sc.AnnData, ann: pd.DataFrame) -> pd.DataFrame:
    print("[Stats] marker gene high-vs-low summary ...")
    ad_score = _maybe_log_normalize(ad)
    available = set(map(str.upper, ad_score.var_names.astype(str)))
    upper_to_real = {str(g).upper(): str(g) for g in ad_score.var_names.astype(str)}
    genes = [upper_to_real[g.upper()] for g in MARKER_GENES if g.upper() in available]
    if not genes:
        return pd.DataFrame()

    x = ad_score[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    expr = pd.DataFrame(x, index=ad_score.obs_names, columns=genes)
    meta = ad_score.obs[["sampleID", "clone_id"]].copy()
    meta["sampleID"] = meta["sampleID"].astype(str)
    meta["clone_id"] = meta["clone_id"].astype(str)
    expr = pd.concat([meta.reset_index(drop=True), expr.reset_index(drop=True)], axis=1)
    clone_expr = expr.groupby(["sampleID", "clone_id"], observed=True)[genes].mean().reset_index()
    d = ann[["sampleID", "clone_id", "high_evidence"]].drop_duplicates().merge(clone_expr, on=["sampleID", "clone_id"], how="left")

    rows = []
    high = d[d["high_evidence"] == True]
    low = d[d["high_evidence"] == False]
    for g in genes:
        xh = pd.to_numeric(high[g], errors="coerce").dropna()
        xl = pd.to_numeric(low[g], errors="coerce").dropna()
        if len(xh) < 2 or len(xl) < 2:
            continue
        stat, p = mannwhitneyu(xh, xl, alternative="two-sided")
        rows.append({
            "gene": g,
            "high_mean": float(xh.mean()),
            "low_mean": float(xl.mean()),
            "delta_high_minus_low": float(xh.mean() - xl.mean()),
            "high_median": float(xh.median()),
            "low_median": float(xl.median()),
            "u_stat": float(stat),
            "p_value": float(p),
            "n_high": int(len(xh)),
            "n_low": int(len(xl)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_high_minus_low", ascending=False)
    return out


def compute_bag_clonality(ad: sc.AnnData, ann: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    obs = ad.obs.copy()
    rows = []
    for sid, g in obs.groupby("sampleID", observed=True):
        clone_counts = g.groupby("clone_id", observed=True).size().astype(float)
        if clone_counts.sum() <= 0:
            continue
        p = clone_counts.values / clone_counts.values.sum()
        rows.append({
            "sampleID": str(sid),
            "PatientID": str(g["PatientID"].iloc[0]) if "PatientID" in g.columns else "",
            "Response": str(g["Response"].iloc[0]) if "Response" in g.columns else "",
            "y_true": int(g["y"].astype(int).iloc[0]),
            "n_cells": int(len(g)),
            "n_clones": int(len(clone_counts)),
            "expanded_clone_fraction": float((clone_counts > 1).sum() / max(1, len(clone_counts))),
            "top1_clone_fraction": float(np.max(p)) if len(p) else np.nan,
            "top5_clone_fraction": float(np.sort(p)[::-1][:5].sum()) if len(p) else np.nan,
            "shannon_entropy": float(-np.sum(p * np.log(p + 1e-12))),
            "simpson_clonality": float(np.sum(p ** 2)),
        })
    bag = pd.DataFrame(rows)
    if bag.empty:
        return bag
    extra = []
    for sid, g in ann.groupby("sampleID", observed=True):
        gh = g[g["high_evidence"] == True]
        extra.append({
            "sampleID": str(sid),
            "max_evidence": float(pd.to_numeric(g[evidence_col], errors="coerce").max()),
            "mean_evidence": float(pd.to_numeric(g[evidence_col], errors="coerce").mean()),
            "max_clone_prob": float(pd.to_numeric(g.get("clone_prob", pd.Series(dtype=float)), errors="coerce").max()) if "clone_prob" in g.columns else np.nan,
            "high_evidence_clone_count": int(len(gh)),
            "high_evidence_clone_size_sum": float(pd.to_numeric(gh.get("clone_size", pd.Series(dtype=float)), errors="coerce").sum()) if len(gh) else 0.0,
            "high_evidence_expanded_rate": float(gh.get("expanded_by_model_clone_size", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(gh) else np.nan,
        })
    bag = bag.merge(pd.DataFrame(extra), on="sampleID", how="left")
    return bag


def test_bag_clonality(bag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bag is None or bag.empty:
        return pd.DataFrame()
    num_cols = [
        "expanded_clone_fraction", "top1_clone_fraction", "top5_clone_fraction", "shannon_entropy", "simpson_clonality",
        "n_clones", "max_evidence", "mean_evidence", "max_clone_prob", "high_evidence_clone_count",
        "high_evidence_clone_size_sum", "high_evidence_expanded_rate",
    ]
    resp = bag[bag["y_true"].astype(int) == 1]
    non = bag[bag["y_true"].astype(int) == 0]
    for c in num_cols:
        if c not in bag.columns:
            continue
        x = pd.to_numeric(resp[c], errors="coerce").dropna()
        y = pd.to_numeric(non[c], errors="coerce").dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        sx = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        sy = float(y.std(ddof=1)) if len(y) > 1 else np.nan
        pooled_var = np.nan
        pooled_sd = np.nan
        smd = np.nan
        if np.isfinite(sx) and np.isfinite(sy) and (len(x) + len(y) - 2) > 0:
            pooled_var = (((len(x) - 1) * sx ** 2) + ((len(y) - 1) * sy ** 2)) / max(1, (len(x) + len(y) - 2))
            pooled_sd = float(np.sqrt(max(0.0, pooled_var)))
            if pooled_sd > 1e-12:
                smd = float((x.mean() - y.mean()) / pooled_sd)
        rows.append({
            "feature": c,
            "responder_mean": float(x.mean()), "nonresponder_mean": float(y.mean()),
            "responder_median": float(x.median()), "nonresponder_median": float(y.median()),
            "responder_std": sx, "nonresponder_std": sy,
            "delta_responder_minus_nonresponder": float(x.mean() - y.mean()),
            "pooled_std": pooled_sd,
            "standardized_mean_difference": smd,
            "u_stat": float(stat), "p_value": float(p),
            "n_responder": int(len(x)), "n_nonresponder": int(len(y)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("p_value")
    return out


def rna_wrong_fused_correct_summary(out_dir: str, ann: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    preds = load_predictions(out_dir)
    if not {"rna_only", "fused"}.issubset(preds.keys()):
        return pd.DataFrame()
    rna = preds["rna_only"].copy()
    fused = preds["fused"].copy()

    common = [c for c in ["fold_global", "fold", "bag_id", "sampleID"] if c in rna.columns and c in fused.columns]
    if "bag_id" in common:
        keys = ["bag_id"] + (["fold_global"] if "fold_global" in common else (["fold"] if "fold" in common else []))
    elif "sampleID" in common:
        keys = ["sampleID"] + (["fold_global"] if "fold_global" in common else (["fold"] if "fold" in common else []))
    else:
        return pd.DataFrame()

    merged = rna.merge(fused, on=keys, suffixes=("_rna", "_fused"))
    y_col = "y_true_rna" if "y_true_rna" in merged.columns else "y_true"
    if y_col not in merged.columns:
        return pd.DataFrame()
    merged["rna_correct"] = merged["y_pred_rna"].astype(int) == merged[y_col].astype(int)
    merged["fused_correct"] = merged["y_pred_fused"].astype(int) == merged[y_col].astype(int)
    fixed = merged[(merged["rna_correct"] == False) & (merged["fused_correct"] == True)].copy()
    harmed = merged[(merged["rna_correct"] == True) & (merged["fused_correct"] == False)].copy()

    def summarize_samples(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        sample_col = "sampleID_rna" if "sampleID_rna" in df.columns else ("sampleID" if "sampleID" in df.columns else None)
        if sample_col is None:
            return pd.DataFrame()
        samples = set(df[sample_col].astype(str))
        g = ann[ann["sampleID"].astype(str).isin(samples)]
        rows = []
        for sid, gg in g.groupby("sampleID", observed=True):
            gh = gg[gg["high_evidence"] == True]
            rows.append({
                "category": tag,
                "sampleID": str(sid),
                "n_high_evidence_clones": int(len(gh)),
                "mean_high_evidence": float(pd.to_numeric(gh[evidence_col], errors="coerce").mean()) if len(gh) else np.nan,
                "sum_high_clone_size": float(pd.to_numeric(gh.get("clone_size", pd.Series(dtype=float)), errors="coerce").sum()) if len(gh) else 0.0,
                "mean_high_clone_size": float(pd.to_numeric(gh.get("clone_size", pd.Series(dtype=float)), errors="coerce").mean()) if len(gh) else np.nan,
                "expanded_rate_high": float(gh.get("expanded_by_model_clone_size", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(gh) else np.nan,
                "y_true": int(gg["y_true"].iloc[0]) if "y_true" in gg.columns else np.nan,
                "Response": str(gg.get("Response", gg.get("severity", pd.Series([""]))).iloc[0]) if len(gg) else "",
            })
        return pd.DataFrame(rows)

    out = pd.concat([summarize_samples(fixed, "rna_wrong_fused_correct"),
                     summarize_samples(harmed, "rna_correct_fused_wrong")], axis=0, ignore_index=True)
    return out

def _method_label(m: str) -> str:
    return {"fused": "Fused", "rna_only": "RNA only", "tcr_only": "TCR only"}.get(str(m), str(m))


def _method_color(m: str) -> str:
    return {"fused": PAL["fused"], "rna_only": PAL["rna"], "tcr_only": PAL["tcr"]}.get(str(m), PAL["muted"])


def plot_performance_bar(figdir: str, perf: pd.DataFrame):

    if perf is None or perf.empty:
        return
    df = perf.copy()
    method_order = [m for m in ["fused", "rna_only", "tcr_only"] if m in set(df["method"].astype(str))]
    if not method_order:
        method_order = df["method"].astype(str).tolist()
    metrics = [
        ("AUROC", "AUROC_mean", "AUROC_std"),
        ("AUPRC", "AUPRC_mean", "AUPRC_std"),
        ("Balanced acc.", "balanced_accuracy_mean", "balanced_accuracy_std"),
        ("F1", "F1_mean", "F1_std"),
    ]
    metrics = [m for m in metrics if m[1] in df.columns]
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(max(5.4, 1.25 * len(metrics)), 3.7))
    n_methods = len(method_order)
    xcenters = np.arange(len(metrics))
    group_w = 0.78
    bar_w = group_w / max(1, n_methods)

    for i, method in enumerate(method_order):
        row = df[df["method"].astype(str) == method].iloc[0]
        means = [float(row[m[1]]) for m in metrics]
        xs = xcenters + (i - (n_methods - 1) / 2) * bar_w
        ax.bar(xs, means, width=bar_w * 0.92, color=_method_color(method), edgecolor="white", linewidth=0.7,
               label=_method_label(method), zorder=3)
        for x, mean in zip(xs, means):
            ax.text(x, mean + 0.012, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7.4, color=PAL["text"], rotation=0)

    ax.set_xticks(xcenters)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Cross-validated score")

    ax.set_ylim(0.4, 1.02)
    ax.set_yticks(np.arange(0.4, 1.01, 0.2))
    ax.set_title("LUSC ICB response prediction")
    _clean_axis(ax, grid=True, grid_axis="y")

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, n_methods),
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    _savefig(os.path.join(figdir, "fig1_ablation_performance.pdf"))


def plot_fused_delta(figdir: str, perf: pd.DataFrame):
    if perf is None or perf.empty or "method" not in perf.columns:
        return
    df = perf.copy()
    if not {"fused", "rna_only"}.issubset(set(df["method"].astype(str))):
        return
    fused = df[df["method"].astype(str) == "fused"].iloc[0]
    rna = df[df["method"].astype(str) == "rna_only"].iloc[0]
    metrics = [
        ("AUROC", "AUROC_mean"),
        ("AUPRC", "AUPRC_mean"),
        ("Balanced acc.", "balanced_accuracy_mean"),
        ("F1", "F1_mean"),
        ("Specificity", "specificity_mean"),
        ("Sensitivity", "sensitivity_recall_mean"),
    ]
    rows = []
    for label, col in metrics:
        if col in df.columns:
            rows.append({"metric": label, "delta": float(fused[col]) - float(rna[col])})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    colors = [PAL["fused"] if x >= 0 else PAL["tcr"] for x in d["delta"]]
    ax.barh(d["metric"], d["delta"], color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0, color=PAL["axis"], linewidth=0.8)
    for i, row in d.iterrows():
        x = row["delta"]
        ha = "left" if x >= 0 else "right"
        ax.text(x + (0.004 if x >= 0 else -0.004), i, f"{x:+.3f}", va="center", ha=ha, fontsize=8.0)
    ax.set_xlabel("Fused − RNA-only")
    ax.set_title("Incremental value of TCR context")
    _clean_axis(ax, grid=True, grid_axis="x")
    _savefig(os.path.join(figdir, "fig2_fused_minus_rna_delta.pdf"))


def _robust_curve_scale(vals: np.ndarray) -> np.ndarray:

    y = np.asarray(vals, dtype=float)
    y = np.where(np.isfinite(y), y, np.nan)
    if np.all(~np.isfinite(y)):
        return np.zeros_like(y, dtype=float)


    y = np.sign(y) * np.log1p(np.abs(y))

    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.zeros_like(y, dtype=float)

    q1, q99 = np.nanpercentile(y[finite], [1, 99])
    if (not np.isfinite(q1)) or (not np.isfinite(q99)) or abs(q99 - q1) < 1e-12:
        q1, q99 = np.nanmin(y[finite]), np.nanmax(y[finite])

    if (not np.isfinite(q1)) or (not np.isfinite(q99)) or abs(q99 - q1) < 1e-12:

        return np.zeros_like(y, dtype=float)

    y = np.clip(y, q1, q99)
    y = (y - q1) / (q99 - q1)
    y = np.clip(y, 0.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y


def plot_ranked_evidence(figdir: str, ann: pd.DataFrame, evidence_col: str, top_frac: float = 0.10):
    if ann is None or ann.empty or evidence_col not in ann.columns:
        return
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    d = ann[["bag_id", evidence_col, "high_evidence"]].copy()
    d[evidence_col] = pd.to_numeric(d[evidence_col], errors="coerce")
    d = d.dropna(subset=[evidence_col])


    curves = []
    for bid, g in d.groupby("bag_id", observed=True):
        vals = np.sort(g[evidence_col].values.astype(float))[::-1]
        if len(vals) < 2:
            continue
        x = np.linspace(0, 1, len(vals))
        y = _robust_curve_scale(vals)
        curves.append(pd.DataFrame({"rank_fraction": x, "scaled_evidence": y, "bag_id": bid}))
        ax.plot(x, y, color=PAL["low"], alpha=0.22, linewidth=0.8, zorder=1)

    if curves:
        allc = pd.concat(curves, ignore_index=True)
        grid = np.linspace(0, 1, 101)
        mat = []
        for _, g in allc.groupby("bag_id", observed=True):
            mat.append(np.interp(grid, g["rank_fraction"].values, g["scaled_evidence"].values))
        if mat:
            med = np.nanmedian(np.vstack(mat), axis=0)
            ax.plot(grid, med, color=PAL["high"], linewidth=2.0, zorder=3, label="Median bag")

    top_frac = float(top_frac)
    ax.axvline(top_frac, color=PAL["text"], linestyle="--", linewidth=0.8,
               label=f"Top {top_frac*100:.0f}%")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Clone rank fraction within bag")
    ax.set_ylabel("Within-bag scaled evidence")
    ax.set_title("Ranked clone-level evidence")
    _clean_axis(ax, grid=True, grid_axis="both")
    ax.legend(frameon=False)
    _savefig(os.path.join(figdir, "fig3_ranked_clone_evidence.pdf"))


def plot_expansion_enrichment(figdir: str, ann: pd.DataFrame, tests: pd.DataFrame):
    if ann is None or ann.empty:
        return
    features = [
        ("expanded_by_model_clone_size", "Clone size > 1"),
        ("expanded_by_raw_cells", "Raw cells > 1"),
        ("expanded_by_is_expanded_raw", "TCR expanded label"),
    ]
    rows = []
    for c, label in features:
        if c not in ann.columns:
            continue
        high = ann[ann["high_evidence"] == True]
        low = ann[ann["high_evidence"] == False]
        hb = high[c].fillna(False).astype(bool)
        lb = low[c].fillna(False).astype(bool)
        pval = np.nan
        if tests is not None and not tests.empty and "feature" in tests.columns:
            hit = tests[(tests["test"] == "fisher_high_vs_low") & (tests["feature"] == c)]
            if len(hit):
                pval = float(hit["p_value"].iloc[0])
        rows.append({"feature": c, "label": label, "rate_high": float(hb.mean()), "rate_other": float(lb.mean()), "p": pval})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(4.8, 1.35 * len(d)), 3.5))
    x = np.arange(len(d))
    bar_w = 0.34
    ax.bar(x - bar_w/2, d["rate_other"], width=bar_w, color=PAL["low"], edgecolor="white", linewidth=0.6, label="Other clones", zorder=3)
    ax.bar(x + bar_w/2, d["rate_high"], width=bar_w, color=PAL["high"], edgecolor="white", linewidth=0.6, label="Highlighted clones", zorder=3)
    for i, row in d.iterrows():
        ax.text(i - bar_w/2, row["rate_other"] + 0.008, f"{row['rate_other']*100:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + bar_w/2, row["rate_high"] + 0.008, f"{row['rate_high']*100:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        if np.isfinite(row["p"]):
            ymax = max(row["rate_high"], row["rate_other"])
            y = min(0.98, ymax + 0.12)
            ax.plot([i - bar_w/2, i - bar_w/2, i + bar_w/2, i + bar_w/2], [y-0.02, y, y, y-0.02], color=PAL["text"], linewidth=0.6)
            ax.text(i, y + 0.012, f"{_format_p(row['p'])}\n{_p_to_stars(row['p'])}", ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"].tolist(), rotation=0)
    ax.set_ylabel("Fraction of clones")
    ax.set_ylim(0, min(1.05, max(0.2, float(d[["rate_high", "rate_other"]].max().max()) + 0.28)))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Clonal expansion enrichment in highlighted clones")
    _clean_axis(ax, grid=True, grid_axis="y")

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        handlelength=1.8,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    _savefig(os.path.join(figdir, "fig4_expansion_enrichment.pdf"))


def _draw_vector_onecol_heatmap(
    ax,
    labels: List[str],
    values: np.ndarray,
    annotations: List[str],
    title: str,
    vmax: float,
    xtick_label: str = "Highlighted - other",
    cmap=HEAT_CMAP,
    fontsize: float = 7.0,
):

    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    vmax = max(float(vmax), 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    ax.set_xlim(0, 1)
    ax.set_ylim(n, 0)
    for j, v in enumerate(values):
        color = to_hex(cmap(norm(v)), keep_alpha=False)
        ax.add_patch(Rectangle((0, j), 1, 1, facecolor=color, edgecolor="white", linewidth=0.35))
        txt = annotations[j] if j < len(annotations) else f"{v:+.2f}"
        ax.text(0.5, j + 0.5, txt, ha="center", va="center", fontsize=fontsize, color=PAL["text"])

    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels)
    ax.set_xticks([0.5])

    ax.set_xticklabels([xtick_label])
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_vector_colorbar(fig, cax, vmax: float, label: str, cmap=HEAT_CMAP, n_steps: int = 80):

    vmax = max(float(vmax), 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    ys = np.linspace(-vmax, vmax, n_steps + 1)
    cax.set_xlim(0, 1)
    cax.set_ylim(-vmax, vmax)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        mid = 0.5 * (y0 + y1)
        cax.add_patch(Rectangle((0, y0), 1, y1 - y0, facecolor=to_hex(cmap(norm(mid)), keep_alpha=False), edgecolor="none"))
    ticks = np.linspace(-vmax, vmax, 5)
    cax.set_yticks(ticks)
    if vmax < 0.3:
        ticklabels = [f"{t:.2f}" for t in ticks]
    else:
        ticklabels = [f"{t:.1f}" for t in ticks]
    cax.set_yticklabels(ticklabels, fontsize=7.5)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(label, fontsize=8, rotation=90, labelpad=8)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _prepare_signature_heatmap_data(sig_tests: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str], float]:
    d = sig_tests.copy()
    if "delta_high_minus_low" not in d.columns:
        d["delta_high_minus_low"] = d["high_mean"] - d["low_mean"]
    d = d.sort_values("delta_high_minus_low", ascending=False)
    values = pd.to_numeric(d["delta_high_minus_low"], errors="coerce").values.astype(float)
    labels = [_nice_name(x) for x in d["signature"]]
    annotations = [f"{v:+.3f}\n{_format_p(p)}" for v, p in zip(values, pd.to_numeric(d["p_value"], errors="coerce"))]
    vmax = max(float(np.nanmax(np.abs(values))) if len(values) else 0.2, 1e-6)
    return labels, values, annotations, vmax


def _prepare_marker_heatmap_data(marker_df: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str], float]:
    d = marker_df.copy().sort_values("delta_high_minus_low", ascending=False)

    if len(d) > 18:
        d = pd.concat([d.head(12), d.tail(6)], axis=0)
    values = pd.to_numeric(d["delta_high_minus_low"], errors="coerce").values.astype(float)
    labels = d["gene"].astype(str).tolist()
    annotations = [f"{v:+.2f}" for v in values]
    vmax = max(float(np.nanmax(np.abs(values))) if len(values) else 0.5, 1e-6)
    return labels, values, annotations, vmax


def plot_signature_heatmap(figdir: str, sig_tests: pd.DataFrame):

    if sig_tests is None or sig_tests.empty:
        return
    labels, values, annotations, vmax = _prepare_signature_heatmap_data(sig_tests)
    fig = plt.figure(figsize=(3.75, 0.55 * len(labels) + 1.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_vector_onecol_heatmap(
        ax, labels, values, annotations,
        title="Functional states of highlighted clones",
        vmax=vmax,
        xtick_label="Highlighted - other",
        fontsize=7.0,
    )
    _draw_vector_colorbar(fig, cax, vmax=vmax, label="Mean score difference")
    _savefig(os.path.join(figdir, "fig5_signature_high_vs_low_heatmap.pdf"))


def plot_marker_heatmap(figdir: str, marker_df: pd.DataFrame):

    if marker_df is None or marker_df.empty:
        return
    labels, values, annotations, vmax = _prepare_marker_heatmap_data(marker_df)
    fig = plt.figure(figsize=(3.95, 0.42 * len(labels) + 1.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_vector_onecol_heatmap(
        ax, labels, values, annotations,
        title="Marker expression in highlighted clones",
        vmax=vmax,
        xtick_label="Highlighted - other",
        fontsize=7.0,
    )
    _draw_vector_colorbar(fig, cax, vmax=vmax, label="Mean expression difference\n(highlighted - other)")
    _savefig(os.path.join(figdir, "fig6_marker_gene_heatmap.pdf"))


def plot_function_marker_combined(figdir: str, sig_tests: pd.DataFrame, marker_df: pd.DataFrame):

    if sig_tests is None or sig_tests.empty or marker_df is None or marker_df.empty:
        return

    sig_labels, sig_values, sig_annot, sig_vmax = _prepare_signature_heatmap_data(sig_tests)
    gene_labels, gene_values, gene_annot, gene_vmax = _prepare_marker_heatmap_data(marker_df)

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.18, 0.06, 1.36, 0.06],
        wspace=0.42,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = fig.add_subplot(gs[0, 3])

    _draw_vector_onecol_heatmap(
        ax1, sig_labels, sig_values, sig_annot,
        title="Functional states of highlighted clones",
        vmax=sig_vmax,
        xtick_label="Highlighted - other",
        fontsize=7.1,
    )
    _draw_vector_colorbar(fig, cax1, vmax=sig_vmax, label="Mean score difference")

    _draw_vector_onecol_heatmap(
        ax2, gene_labels, gene_values, gene_annot,
        title="Marker expression in highlighted clones",
        vmax=gene_vmax,
        xtick_label="Highlighted - other",
        fontsize=6.9,
    )
    _draw_vector_colorbar(fig, cax2, vmax=gene_vmax, label="Mean expression difference")

    ax1.text(-0.33, 1.035, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    ax2.text(-0.30, 1.035, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")

    _savefig(os.path.join(figdir, "fig5_function_marker_combined_vector.pdf"))


INTEGRATED_FUNCTION_GENE_GROUPS = [
    ("score_cytotoxicity", "Cytotoxicity", ["NKG7", "GZMH", "GZMB", "PRF1", "GNLY"]),
    ("score_exhaustion_activation", "Activation /\nexhaustion", ["IFNG", "CXCL13", "LAG3", "TOX", "HAVCR2", "PDCD1", "CTLA4", "TIGIT", "ENTPD1", "TNF", "LAYN"]),
    ("score_proliferation", "Proliferation", ["TOP2A", "MKI67"]),
    ("score_naive_memory", "Naive /\nmemory", ["TCF7", "SELL", "CCR7", "IL7R"]),
]


def _signature_delta_lookup(sig_tests: pd.DataFrame) -> Dict[str, Tuple[float, float]]:

    out = {}
    if sig_tests is None or sig_tests.empty:
        return out
    d = sig_tests.copy()
    if "delta_high_minus_low" not in d.columns and {"high_mean", "low_mean"}.issubset(d.columns):
        d["delta_high_minus_low"] = d["high_mean"] - d["low_mean"]
    for _, r in d.iterrows():
        key = str(r.get("signature", ""))
        if not key:
            continue
        delta = pd.to_numeric(pd.Series([r.get("delta_high_minus_low", np.nan)]), errors="coerce").iloc[0]
        pval = pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0]
        out[key] = (float(delta) if np.isfinite(delta) else np.nan,
                    float(pval) if np.isfinite(pval) else np.nan)
    return out


def _prepare_integrated_marker_rows(marker_df: pd.DataFrame):

    if marker_df is None or marker_df.empty:
        return []
    d = marker_df.copy()
    d["gene_upper"] = d["gene"].astype(str).str.upper()
    gene_to_row = {str(r["gene_upper"]): r for _, r in d.iterrows()}
    used = set()
    rows = []
    for sig_key, label, genes in INTEGRATED_FUNCTION_GENE_GROUPS:
        for g in genes:
            gu = g.upper()
            if gu not in gene_to_row or gu in used:
                continue
            r = gene_to_row[gu]
            delta = pd.to_numeric(pd.Series([r.get("delta_high_minus_low", np.nan)]), errors="coerce").iloc[0]
            pval = pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "signature": sig_key,
                "function": label,
                "gene": str(r.get("gene", g)),
                "delta": float(delta) if np.isfinite(delta) else np.nan,
                "p_value": float(pval) if np.isfinite(pval) else np.nan,
            })
            used.add(gu)

    leftovers = d[~d["gene_upper"].isin(used)].copy()
    if not leftovers.empty:
        leftovers = leftovers.sort_values("delta_high_minus_low", ascending=False)
        for _, r in leftovers.iterrows():
            delta = pd.to_numeric(pd.Series([r.get("delta_high_minus_low", np.nan)]), errors="coerce").iloc[0]
            pval = pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "signature": "other_markers",
                "function": "Other\nmarkers",
                "gene": str(r.get("gene", "")),
                "delta": float(delta) if np.isfinite(delta) else np.nan,
                "p_value": float(pval) if np.isfinite(pval) else np.nan,
            })
    return rows


def plot_function_gene_integrated(figdir: str, sig_tests: pd.DataFrame, marker_df: pd.DataFrame):

    rows = _prepare_integrated_marker_rows(marker_df)
    if not rows:
        return

    sig_lookup = _signature_delta_lookup(sig_tests)
    n = len(rows)
    values = np.array([r["delta"] for r in rows], dtype=float)
    gene_vmax = max(float(np.nanmax(np.abs(values))) if len(values) else 0.5, 0.5)

    gene_vmax = min(max(gene_vmax, 0.3), 0.6)
    norm = Normalize(vmin=-gene_vmax, vmax=gene_vmax)

    spans = []
    start = 0
    for i in range(1, n + 1):
        if i == n or rows[i]["signature"] != rows[start]["signature"]:
            spans.append((start, i, rows[start]["signature"], rows[start]["function"]))
            start = i

    fig_h = max(4.2, 0.36 * n + 1.10)
    fig = plt.figure(figsize=(6.4, fig_h))

    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.55, 0.07], wspace=0.22)
    ax_fun = fig.add_subplot(gs[0, 0])
    ax_gene = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    ax_fun.set_xlim(0, 1)
    ax_fun.set_ylim(n, 0)
    ax_fun.set_xticks([])
    ax_fun.set_yticks([])
    for a, b, sig_key, flabel in spans:
        center = (a + b) / 2.0
        # Bracket
        ax_fun.plot([0.94, 0.99], [a + 0.02, a + 0.02], color=PAL["axis"], linewidth=0.7)
        ax_fun.plot([0.99, 0.99], [a + 0.02, b - 0.02], color=PAL["axis"], linewidth=0.7)
        ax_fun.plot([0.94, 0.99], [b - 0.02, b - 0.02], color=PAL["axis"], linewidth=0.7)
        delta, pval = sig_lookup.get(sig_key, (np.nan, np.nan))
        if np.isfinite(delta):
            text = f"{flabel}\nΔ={delta:+.3f}\n{_format_p(pval)}"
        else:
            text = flabel
        ax_fun.text(0.90, center, text, ha="right", va="center", fontsize=7.6, color=PAL["text"], linespacing=1.05)
    ax_fun.set_title("Functional program", loc="right", fontsize=10.0, fontweight="bold", pad=8)
    for spine in ax_fun.spines.values():
        spine.set_visible(False)


    ax_gene.set_xlim(0, 1)
    ax_gene.set_ylim(n, 0)
    for j, r in enumerate(rows):
        v = r["delta"]
        color = to_hex(HEAT_CMAP(norm(v)), keep_alpha=False) if np.isfinite(v) else "#FFFFFF"
        ax_gene.add_patch(Rectangle((0, j), 1, 1, facecolor=color, edgecolor="white", linewidth=0.35))
        ax_gene.text(0.5, j + 0.5, f"{v:+.2f}" if np.isfinite(v) else "", ha="center", va="center", fontsize=7.0, color=PAL["text"])

    for a, b, _, _ in spans[1:]:
        ax_gene.plot([0, 1], [a, a], color="white", linewidth=1.2)
    ax_gene.set_yticks(np.arange(n) + 0.5)
    ax_gene.set_yticklabels([r["gene"] for r in rows], fontsize=8.0)
    ax_gene.set_xticks([0.5])
    ax_gene.set_xticklabels(["Highlighted - other"], fontsize=8.3)
    ax_gene.tick_params(axis="both", length=0)
    ax_gene.set_title("Marker expression", loc="left", fontsize=10.0, fontweight="bold", pad=8)
    for spine in ax_gene.spines.values():
        spine.set_visible(False)

    _draw_vector_colorbar(fig, cax, vmax=gene_vmax, label="Mean expression difference")

    _savefig(os.path.join(figdir, "fig5_function_gene_integrated_vector.pdf"))



MARKER_PROGRAM_GROUPS = [
    ("Cytotoxicity", ["NKG7", "GZMH", "GZMB", "PRF1", "GNLY"]),
    ("Activation /\nexhaustion", ["CXCL13", "PDCD1", "TOX", "LAG3", "HAVCR2", "CTLA4", "IFNG", "TIGIT", "ENTPD1", "TNF", "LAYN"]),

    ("Cycling", ["MKI67", "TOP2A"]),
    ("Naive /\nmemory", ["TCF7", "IL7R", "SELL", "CCR7"]),
]


def _prepare_marker_program_columns(marker_df: pd.DataFrame):

    if marker_df is None or marker_df.empty:
        return [], [], np.array([], dtype=float)
    d = marker_df.copy()
    d["gene_upper"] = d["gene"].astype(str).str.upper()
    gene_to_row = {str(r["gene_upper"]): r for _, r in d.iterrows()}

    genes_out = []
    values_out = []
    spans = []
    pos = 0
    for program, genes in MARKER_PROGRAM_GROUPS:
        start = pos
        for g in genes:
            gu = g.upper()
            if gu not in gene_to_row:
                continue
            r = gene_to_row[gu]
            delta = pd.to_numeric(pd.Series([r.get("delta_high_minus_low", np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(delta):
                continue
            genes_out.append(str(r.get("gene", g)))
            values_out.append(float(delta))
            pos += 1
        if pos > start:
            spans.append((start, pos, program))
    return genes_out, spans, np.asarray(values_out, dtype=float)


def plot_marker_programs_heatmap(figdir: str, marker_df: pd.DataFrame):

    genes, spans, values = _prepare_marker_program_columns(marker_df)
    if len(genes) == 0:
        return

    n = len(genes)
    vmax = max(float(np.nanmax(np.abs(values))) if len(values) else 0.5, 0.5)
    vmax = min(max(vmax, 0.3), 0.6)
    norm = Normalize(vmin=-vmax, vmax=vmax)


    fig_w = max(8.6, 0.44 * n + 2.4)
    fig_h = 2.65
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    ax.set_xlim(0, n)
    ax.set_ylim(1.0, -0.85)


    for j, v in enumerate(values):
        color = to_hex(HEAT_CMAP(norm(v)), keep_alpha=False) if np.isfinite(v) else "#FFFFFF"
        ax.add_patch(Rectangle((j, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.45))


    for start, end, program in spans:
        center = (start + end) / 2.0
        ax.text(center, -0.50, program, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=PAL["text"], linespacing=0.95)
        ax.plot([start, start], [-0.05, 1.0], color=PAL["axis"], linewidth=0.70)
        ax.plot([end, end], [-0.05, 1.0], color=PAL["axis"], linewidth=0.70)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(genes, rotation=45, ha="right", va="top", fontsize=8.0)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)
    ax.set_title("Marker programs of highlighted clones", loc="left", fontsize=10.5, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_vector_colorbar(fig, cax, vmax=vmax, label="Mean expression difference\n(highlighted - other)")
    _savefig(os.path.join(figdir, "fig5_marker_programs_vector.pdf"))

def plot_rna_wrong_fused_correct(figdir: str, fixed: pd.DataFrame):
    if fixed is None or fixed.empty:
        return
    d = fixed.copy()
    if "category" not in d.columns:
        return
    metric = "sum_high_clone_size" if "sum_high_clone_size" in d.columns else "n_high_evidence_clones"
    order = ["rna_wrong_fused_correct", "rna_correct_fused_wrong"]
    labels = {"rna_wrong_fused_correct": "RNA wrong\nFused correct", "rna_correct_fused_wrong": "RNA correct\nFused wrong"}
    rows = []
    for cat in order:
        vals = pd.to_numeric(d.loc[d["category"] == cat, metric], errors="coerce").dropna()
        if len(vals):
            rows.append({"category": cat, "mean": float(vals.mean()), "median": float(vals.median()), "n": int(len(vals))})
    dd = pd.DataFrame(rows)
    if dd.empty:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    x = np.arange(len(dd))
    ax.bar(x, dd["mean"], color=[PAL["fused"], PAL["low"]][:len(dd)], edgecolor="white", linewidth=0.7, zorder=3)
    for i, row in dd.iterrows():
        ax.text(i, row["mean"] + 0.02 * max(1, dd["mean"].max()), f"n={int(row['n'])}\nmean={row['mean']:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in dd["category"]])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Clonal signal in rescued predictions")
    _clean_axis(ax, grid=True, grid_axis="y")
    _savefig(os.path.join(figdir, "fig7_rna_wrong_fused_correct.pdf"))


def plot_bag_response_clonality(figdir: str, bag_tests: pd.DataFrame):
    if bag_tests is None or bag_tests.empty:
        return
    features = ["expanded_clone_fraction", "top1_clone_fraction", "simpson_clonality", "max_evidence", "high_evidence_clone_size_sum"]
    d = bag_tests[bag_tests["feature"].isin(features)].copy()
    if d.empty:
        return


    if "standardized_mean_difference" not in d.columns:
        d["standardized_mean_difference"] = d["delta_responder_minus_nonresponder"]
    d["standardized_mean_difference"] = pd.to_numeric(d["standardized_mean_difference"], errors="coerce")
    d = d.dropna(subset=["standardized_mean_difference"])
    if d.empty:
        return

    order = ["simpson_clonality", "top1_clone_fraction", "expanded_clone_fraction", "max_evidence", "high_evidence_clone_size_sum"]
    d["_order"] = d["feature"].map({f: i for i, f in enumerate(order)})
    d = d.sort_values("_order")

    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    y = np.arange(len(d))
    xvals = d["standardized_mean_difference"].values.astype(float)
    colors = [PAL["fused"] if x >= 0 else PAL["tcr"] for x in xvals]
    ax.barh(y, xvals, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color=PAL["axis"], linewidth=0.75)

    max_abs = float(np.nanmax(np.abs(xvals))) if len(xvals) else 1.0
    lim = max(0.55, max_abs * 1.22)
    ax.set_xlim(-lim, lim)

    for i, row in enumerate(d.itertuples(index=False)):
        x = float(getattr(row, "standardized_mean_difference"))
        pval = float(getattr(row, "p_value")) if hasattr(row, "p_value") else np.nan
        dx = 0.025 * lim

        label = f"{x:+.2f}\n{_format_p(pval)}"
        ax.text(x + (dx if x >= 0 else -dx), i, label,
                ha="left" if x >= 0 else "right", va="center", fontsize=7.2)

    ax.set_yticks(y)
    ax.set_yticklabels([_nice_name(x) for x in d["feature"]], fontsize=8.2)
    ax.set_xlabel("Effect size (responder - non-responder)")
    ax.set_title("Response-associated clonal signal")
    _clean_axis(ax, grid=True, grid_axis="x")
    _savefig(os.path.join(figdir, "fig8_bag_clonality_response.pdf"))


def make_figures(figdir: str, perf: pd.DataFrame, ann: pd.DataFrame, evidence_col: str,
                 tests: pd.DataFrame, sig_tests: pd.DataFrame, marker_df: pd.DataFrame,
                 fixed: pd.DataFrame, bag_tests: pd.DataFrame, top_frac: float = 0.10):
    ensure_dir(figdir)
    print(f"[Figures] writing PDFs to {figdir}")
    plot_performance_bar(figdir, perf)
    plot_fused_delta(figdir, perf)
    plot_ranked_evidence(figdir, ann, evidence_col, top_frac=top_frac)
    plot_expansion_enrichment(figdir, ann, tests)

    for old_pdf in [
        "fig5_signature_high_vs_low_heatmap.pdf",
        "fig6_marker_gene_heatmap.pdf",
        "fig5_function_marker_combined_vector.pdf",
        "fig5_function_gene_integrated_vector.pdf",
        "fig5_marker_programs_vector.pdf",
    ]:
        old_path = os.path.join(figdir, old_pdf)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
    plot_marker_programs_heatmap(figdir, marker_df)
    plot_rna_wrong_fused_correct(figdir, fixed)
    plot_bag_response_clonality(figdir, bag_tests)


def parse_args():
    p = argparse.ArgumentParser(description="LUSC/NSCLC response biology analysis from RNA+TCR MIL outputs.")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Top-level training output dir containing fused/rna_only/tcr_only.")
    p.add_argument("--rna_path", type=str, default=DEFAULT_RNA_PATH)
    p.add_argument("--meta_path", type=str, default=DEFAULT_META_PATH)
    p.add_argument("--tcr_path", type=str, default=DEFAULT_TCR_PATH)
    p.add_argument("--sample_meta_path", type=str, default=DEFAULT_SAMPLE_META_PATH)
    p.add_argument("--bag_level", type=str, default="sample", choices=["sample", "patient"])
    p.add_argument("--celltype_filter", type=str, default="CD8")
    p.add_argument("--cancer_type", type=str, default="LUSC", choices=["ALL", "LUAD", "LUSC"])
    p.add_argument("--clone_key", type=str, default="clonotype")
    p.add_argument("--require_tcr", action="store_true", default=True)
    p.add_argument("--no_require_tcr", dest="require_tcr", action="store_false")
    p.add_argument("--response_positive_terms", type=str, default="pCR,MPR")
    p.add_argument("--response_negative_terms", type=str, default="non-MPR")
    p.add_argument("--min_genes", type=int, default=200)
    p.add_argument("--keep_tcr_ig_genes", action="store_true", help="Do not remove TCR/IG genes before signature analysis.")
    p.add_argument("--method", type=str, default="fused")
    p.add_argument("--evidence_col", type=str, default="auto",
                   help="auto / weighted_noisy_or_evidence / clone_prob / clone_attention / clone_attn / clone_contribution")
    p.add_argument("--top_frac", type=float, default=0.10, help="Top fraction of clones per bag/fold defined as highlighted.")
    return p.parse_args()


def main():
    args = parse_args()
    out_bio = os.path.join(args.out_dir, "biology_lusc_response")
    table_dir = os.path.join(out_bio, "tables")
    figdir = os.path.join(out_bio, "figures_pdf")
    ensure_dir(out_bio)
    ensure_dir(table_dir)
    ensure_dir(figdir)

    perf = load_ablation_table(args.out_dir)
    perf.to_csv(os.path.join(table_dir, "ablation_table_loaded.csv"), index=False)

    ad = load_nsclc_response_data(
        rna_path=args.rna_path,
        meta_path=args.meta_path,
        tcr_path=args.tcr_path,
        sample_meta_path=args.sample_meta_path,
        bag_level=args.bag_level,
        require_tcr=args.require_tcr,
        celltype_filter=args.celltype_filter,
        cancer_type=args.cancer_type,
        clone_key=args.clone_key,
        response_positive_terms=args.response_positive_terms,
        response_negative_terms=args.response_negative_terms,
        min_genes=args.min_genes,
        remove_tcr_ig_genes=not args.keep_tcr_ig_genes,
    )
    make_bag_table(ad).to_csv(os.path.join(table_dir, "bag_table_from_annotations.csv"))

    ce = load_clone_evidence(args.out_dir, method=args.method)
    ce, evidence_col = standardize_evidence_columns(ce, args.evidence_col)
    ce.to_csv(os.path.join(table_dir, "clone_evidence_standardized.csv"), index=False)

    ann = annotate_clone_evidence(ce, ad, evidence_col, args.top_frac)
    ann.to_csv(os.path.join(table_dir, "clone_evidence_annotated.csv"), index=False)

    tests = test_high_vs_low_clones(ann, evidence_col)
    tests.to_csv(os.path.join(table_dir, "clone_enrichment_tests.csv"), index=False)

    sig = compute_signature_scores(ad, ann)
    sig.to_csv(os.path.join(table_dir, "clone_level_signature_scores.csv"), index=False)
    sig_hl, sig_resp = test_signature_scores(sig)
    sig_hl.to_csv(os.path.join(table_dir, "signature_tests_high_vs_low.csv"), index=False)
    sig_resp.to_csv(os.path.join(table_dir, "signature_tests_high_responder_vs_nonresponder.csv"), index=False)

    marker_df = compute_marker_gene_summary(ad, ann)
    marker_df.to_csv(os.path.join(table_dir, "marker_gene_high_vs_low_summary.csv"), index=False)

    bag = compute_bag_clonality(ad, ann, evidence_col)
    bag.to_csv(os.path.join(table_dir, "bag_level_clonality.csv"), index=False)
    bag_tests = test_bag_clonality(bag)
    bag_tests.to_csv(os.path.join(table_dir, "bag_level_clonality_response_tests.csv"), index=False)

    fixed = rna_wrong_fused_correct_summary(args.out_dir, ann, evidence_col)
    fixed.to_csv(os.path.join(table_dir, "rna_wrong_fused_correct_clone_summary.csv"), index=False)

    make_figures(figdir, perf, ann, evidence_col, tests, sig_hl, marker_df, fixed, bag_tests)

    print("\n[DONE] LUSC/NSCLC response downstream analysis saved to:")
    print(f"  {out_bio}")
    print(f"  Tables : {table_dir}")
    print(f"  Figures: {figdir}")
    print("\n[Preview] Performance table")
    print(perf.to_string(index=False))
    print("\n[Preview] Clone enrichment tests")
    if not tests.empty:
        print(tests.head(12).to_string(index=False))
    else:
        print("No clone enrichment tests produced.")
    print("\n[Preview] Signature high-vs-low tests")
    if not sig_hl.empty:
        print(sig_hl.head(12).to_string(index=False))
    else:
        print("No signature tests produced.")


if __name__ == "__main__":
    main()
