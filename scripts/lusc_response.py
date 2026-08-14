#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import re
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")



DEFAULT_RNA_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA.h5ad"
DEFAULT_META_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA_metadata.csv"
DEFAULT_TCR_PATH = "./data/nsclc/GSE243013_T_with_TCR_annotation.csv"
DEFAULT_SAMPLE_META_PATH = "./data/nsclc/meta2.csv"
DEFAULT_TCRBERT_DIR = "./pretrained/tcr-bert/"
DEFAULT_OUT_DIR = "./result/nsclcclassvae2"

RUN_METHODS = ["rna_only", "tcr_only", "fused"]

SEED = 1337
AASET = set(list("ACDEFGHIKLMNPQRSTVWY"))


EMB_D = 128
SHARED_D = 64
PRIV_D = 64
SOLO_LATENT_D = SHARED_D + PRIV_D
MAX_LEN = 20


LAMBDA_RNA_RECON = 0.0
LAMBDA_TCR_RECON = 0.15
LAMBDA_ORTH = 0.3
LAMBDA_KL = 1e-4

LAMBDA_KL_PRIV_MULT = 2.0

GATE_BIAS = 0.0
TCR_SCALE = 0.2
FUSE_SCALE = 0.2
ALPHA_WEIGHT = 0.3
ALPHA_RECON_WEIGHT = 0.3
UNFREEZE_LAST_N = 2

DEFAULT_FUSED_PATIENT_HEAD = "noisy_or"
DEFAULT_CLONE_SIZE_PRIOR_WEIGHT = 0.50
DEFAULT_CLONE_EVIDENCE_SCALE = 3.0

LAMBDA_SOLO_RECON = 0.0
LAMBDA_SOLO_KL = LAMBDA_KL


LAMBDA_RNA_CLS = 0.0
LAMBDA_TCR_CLS = 0.0


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
special_tokens = ["<pad>", "<start>", "<end>", "<mask>"]
vocab = special_tokens + amino_acids
token2idx = {tok: i for i, tok in enumerate(vocab)}
pad_idx = token2idx["<pad>"]
start_idx = token2idx["<start>"]
end_idx = token2idx["<end>"]
vocab_size = len(vocab)


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_aa(seq: str) -> str:
    if pd.isna(seq):
        return ""
    s = str(seq).upper().strip()
    if s in {"", "NONE", "NAN", "NULL", "NA", "<NA>"}:
        return ""
    return "".join(ch for ch in s if ch in AASET)


def obs_to_str(s, fill=""):
    s = s.astype("object")
    s = s.where(pd.notna(s), fill)
    s = s.replace({"nan": fill, "NaN": fill, "None": fill, "<NA>": fill, "NONE": fill})
    return s.astype(str)


def encode_seq(seq, max_len=MAX_LEN):
    s = clean_aa(seq)
    ids = [start_idx] + [token2idx.get(aa, pad_idx) for aa in s[:max_len - 2]] + [end_idx]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


def tcr_prepare_ar_batch(seqs, max_len=MAX_LEN, device=None):
    ids = [encode_seq(s, max_len) for s in seqs]
    return torch.tensor(ids, dtype=torch.long, device=device)


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


def normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _split_terms(s: str) -> List[str]:
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]


def parse_response_to_label(
    x,
    positive_terms: str = "pCR,MPR",
    negative_terms: str = "non-MPR",
) -> Optional[int]:

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


def standardize_cell_barcode(x: str) -> str:
    return normalize_id(x)


def barcode_core(x: str) -> str:

    s = normalize_id(x)

    m = re.match(r"^([ACGTN]+-\d+)", s)
    if m:
        return m.group(1)

    m = re.match(r"^([ACGTN]+)", s)
    if m:
        return m.group(1)
    return s

mode_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
    "..", "model"
))
if mode_path not in sys.path:
    sys.path.append(mode_path)

try:
    from model import GeneMLPEncoder, Cross_attention, RNADecoder, CDR3SeqDecoder
except Exception as e:
    raise ImportError(
        "Cannot import GeneMLPEncoder/Cross_attention/RNADecoder/CDR3SeqDecoder from ../model. "
        "Please run this script from the same project structure as the NSCLC script."
    ) from e


def read_sample_metadata(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_table_auto(path: str, dtype=str) -> pd.DataFrame:

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=dtype)
    else:

        df = pd.read_csv(path, sep=None, engine="python", dtype=dtype)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_nsclc_response_data(
    rna_path: str,
    meta_path: str,
    tcr_path: str,
    sample_meta_path: str,
    bag_level: str = "sample",
    require_tcr: bool = True,
    celltype_filter: str = "CD8",
    cancer_type: str = "LUAD",
    clone_key: str = "clonotype",
    response_positive_terms: str = "pCR,MPR",
    response_negative_terms: str = "non-MPR",
) -> sc.AnnData:

    bag_level = str(bag_level).lower()
    if bag_level not in {"sample", "patient"}:
        raise ValueError("--bag_level must be 'sample' or 'patient'.")

    print("[Load] NSCLC scRNA-seq h5ad ...")
    ad = sc.read_h5ad(rna_path)
    ad.obs_names = ad.obs_names.astype(str).str.strip()
    ad.obs["cell_id_original"] = ad.obs_names.astype(str)

    print("[Load] cell annotation ...")
    meta_df = _read_table_auto(meta_path, dtype=str).fillna("")
    cell_col = find_first_col(meta_df, ["cellID", "cell_id", "cell", "cellName", "barcode"])
    meta_df["cell_id"] = meta_df[cell_col].map(normalize_id)
    meta_df = meta_df.drop_duplicates("cell_id", keep="first")

    sub_col = find_first_col(meta_df, ["sub_cell_type", "sub_celltype", "celltype", "cell_type"], required=False)
    major_col = find_first_col(meta_df, ["major_cell_type", "majorType", "major_celltype", "major"], required=False)
    cancer_col = find_first_col(meta_df, ["cancer_type", "Cancer_Type", "histology"], required=False)
    sample_col_meta = find_first_col(meta_df, ["sampleID", "sample_id", "sample", "Tumor_Sample_Barcode"], required=False)
    patient_col_meta = find_first_col(meta_df, ["PatientID", "patientID", "patient_id", "patient", "donor"], required=False)

    if sub_col is not None and sub_col != "sub_cell_type":
        meta_df["sub_cell_type"] = meta_df[sub_col].astype(str).str.strip()
    elif "sub_cell_type" not in meta_df.columns:
        meta_df["sub_cell_type"] = ""
    if major_col is not None and major_col != "major_cell_type":
        meta_df["major_cell_type"] = meta_df[major_col].astype(str).str.strip()
    elif "major_cell_type" not in meta_df.columns:
        meta_df["major_cell_type"] = ""
    if cancer_col is not None and cancer_col != "cancer_type":
        meta_df["cancer_type"] = meta_df[cancer_col].astype(str).str.strip()
    elif "cancer_type" not in meta_df.columns:
        meta_df["cancer_type"] = ""
    if sample_col_meta is not None and sample_col_meta != "sampleID":
        meta_df["sampleID"] = meta_df[sample_col_meta].astype(str).str.strip()
    elif "sampleID" not in meta_df.columns:
        meta_df["sampleID"] = ""
    if patient_col_meta is not None and patient_col_meta != "PatientID":
        meta_df["PatientID"] = meta_df[patient_col_meta].astype(str).str.strip()
    elif "PatientID" not in meta_df.columns:
        meta_df["PatientID"] = ""

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

    def _to_expanded01(x):
        s = str(x).strip().lower()
        return 1 if s in {"expanded", "true", "t", "1", "yes", "y", "clonal"} else 0

    exp_col = find_first_col(tcr_df, ["expansion", "is_expanded", "expanded", "clonal"], required=False)
    if exp_col is not None:
        tcr_df["is_expanded"] = tcr_df[exp_col].map(_to_expanded01).astype(int)
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
    print(f"[TCR] valid TRA ratio: {(tcr_df['TRA_aa'].str.len() > 0).mean():.3f}" if len(tcr_df) else "[TCR] valid TRA ratio: NA")
    if pd.Series(tcr_df["is_expanded"]).notna().any():
        print(f"[TCR] expanded ratio: {pd.to_numeric(tcr_df['is_expanded'], errors='coerce').fillna(0).mean():.3f}")

    print("[Load] sample metadata (Pathological Response) ...")
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


    obs = pd.DataFrame(index=ad.obs_names)
    obs.index.name = "cell_id"

    meta_keep_cols = [c for c in meta_df.columns if c not in {cell_col, "cell_id"}]
    meta2 = (
        meta_df.drop_duplicates("cell_id", keep="first")
        .set_index("cell_id")[meta_keep_cols]
    )
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

    obs = obs.merge(
        sample_df.set_index("sampleID")[["PatientID", "Response", "y"]],
        left_on="sampleID",
        right_index=True,
        how="left",
        suffixes=("", "_samplemeta"),
    )
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
        raise ValueError("No cells left after NSCLC filters. Check --cancer_type, --celltype_filter, --require_tcr and sample metadata matching.")

    sc.pp.filter_cells(ad, min_genes=200)

    ad.raw = ad.copy()
    tcr_ig_pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", flags=re.IGNORECASE)
    keep_genes = [g for g in ad.var_names if not tcr_ig_pattern.match(str(g))]
    removed = ad.n_vars - len(keep_genes)
    ad = ad[:, keep_genes].copy()
    print(f"[Filter] removed {removed} TCR/IG genes; genes_before_fold_preprocess={ad.n_vars}")


    ad.obs["y"] = ad.obs["y"].astype(int)
    ad.obs["Response"] = obs_to_str(ad.obs["Response"], fill="")
    ad.obs["severity"] = ad.obs["Response"].astype(str)
    ad.obs["Response_binary"] = ad.obs["y"].map({1: "effective_response", 0: "non_MPR"}).astype(str)
    ad.obs["response_binary"] = ad.obs["Response_binary"].astype(str)
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
        ad.obs["is_expanded"] = ad.obs["is_expanded"].where(ad.obs["is_expanded"].notna(), (ad.obs["clone_size"].astype(int) > 1).astype(int))
        ad.obs["is_expanded"] = ad.obs["is_expanded"].fillna(0).astype(int)

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
    print(f"cells={ad.n_obs}, genes_before_fold_preprocess={ad.n_vars}")
    print(f"bags={ad.obs['bag_id'].nunique()}, patients/groups={ad.obs['group_id'].nunique()}")
    print("[Response counts by bag]")
    bag_df = make_bag_table(ad)
    print(bag_df["y"].value_counts().rename({1: "effective(pCR/MPR)", 0: "non-MPR"}))
    print("[Response text by bag]")
    print(bag_df["severity"].value_counts())
    if "sub_cell_type" in ad.obs.columns:
        print("[sub_cell_type after filter]")
        print(ad.obs["sub_cell_type"].astype(str).value_counts().head(30))
    if "cancer_type" in ad.obs.columns:
        print("[cancer_type after filter]")
        print(ad.obs["cancer_type"].astype(str).value_counts(dropna=False))
    print(f"[is_expanded ratio] {pd.to_numeric(ad.obs['is_expanded'], errors='coerce').fillna(0).mean():.3f}")

    return ad


def _as_dense_float32(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _matrix_max(x, n_rows: int = 200) -> float:
    if x.shape[0] == 0 or x.shape[1] == 0:
        return 0.0
    sub = x[: min(n_rows, x.shape[0])]
    val = sub.max()
    if hasattr(val, "toarray"):
        val = val.toarray()
    return float(np.asarray(val).max())


def _fit_train_gene_mask(ad_fold: sc.AnnData, train_bags: List[str], min_cells: int) -> np.ndarray:
    train_mask = ad_fold.obs["bag_id"].astype(str).isin([str(x) for x in train_bags]).values
    x_train = ad_fold.X[train_mask]
    if x_train.shape[0] == 0:
        raise ValueError("No train cells found when fitting fold gene mask.")
    detected = np.asarray((x_train > 0).sum(axis=0)).ravel()
    keep = detected >= int(min_cells)
    if keep.sum() == 0:
        print("[Warn] train-only min_cells removed all genes; keeping all genes for this fold.")
        keep = np.ones(ad_fold.n_vars, dtype=bool)
    return keep.astype(bool)


def _fit_hvg_names_train_only(ad_fold_norm: sc.AnnData, train_bags: List[str], n_top_genes: int) -> List[str]:
    train_mask = ad_fold_norm.obs["bag_id"].astype(str).isin([str(x) for x in train_bags]).values
    ad_train = ad_fold_norm[train_mask].copy()
    n_top = min(int(n_top_genes), int(ad_train.n_vars))
    if n_top <= 0:
        raise ValueError("No genes available for HVG selection.")
    try:
        sc.pp.highly_variable_genes(ad_train, n_top_genes=n_top, subset=False)
        hvg = ad_train.var_names[ad_train.var["highly_variable"].values].astype(str).tolist()
    except Exception as e:
        print(f"[Warn] scanpy HVG failed on train fold ({e}); using variance top genes.")
        x_train = _as_dense_float32(ad_train.X)
        var = np.var(x_train, axis=0)
        idx = np.argsort(var)[::-1][:n_top]
        hvg = ad_train.var_names[idx].astype(str).tolist()
    if len(hvg) == 0:
        x_train = _as_dense_float32(ad_train.X)
        var = np.var(x_train, axis=0)
        idx = np.argsort(var)[::-1][:n_top]
        hvg = ad_train.var_names[idx].astype(str).tolist()
    return hvg


def _scale_from_train(ad_fold_hvg: sc.AnnData, train_bags: List[str], max_value: float = 10.0) -> sc.AnnData:
    train_mask = ad_fold_hvg.obs["bag_id"].astype(str).isin([str(x) for x in train_bags]).values
    x_all = _as_dense_float32(ad_fold_hvg.X)
    x_train = x_all[train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_all = (x_all - mean) / std
    if max_value is not None and float(max_value) > 0:
        x_all = np.clip(x_all, -float(max_value), float(max_value))
    ad_out = ad_fold_hvg.copy()
    ad_out.X = x_all.astype(np.float32)
    ad_out.uns["fold_preprocess"] = {
        "scale_fit": "train_only",
        "scale_max_value": float(max_value),
        "n_genes": int(ad_out.n_vars),
    }
    return ad_out


def prepare_fold_anndata(
    ad_raw: sc.AnnData,
    train_bags: List[str],
    val_bags: List[str],
    test_bags: List[str],
    args,
) -> sc.AnnData:
    fold_bags = list(dict.fromkeys([str(x) for x in list(train_bags) + list(val_bags) + list(test_bags)]))
    ad_fold = ad_raw[ad_raw.obs["bag_id"].astype(str).isin(fold_bags).values].copy()
    if ad_fold.n_obs == 0:
        raise ValueError("No cells found for fold bags.")

    keep_gene = _fit_train_gene_mask(ad_fold, train_bags, min_cells=args.min_cells_train)
    ad_fold = ad_fold[:, keep_gene].copy()

    train_mask = ad_fold.obs["bag_id"].astype(str).isin([str(x) for x in train_bags]).values
    train_max = _matrix_max(ad_fold.X[train_mask], n_rows=200)
    do_norm_log = bool(train_max > float(args.log1p_if_max_gt))
    if args.force_log1p:
        do_norm_log = True
    if args.skip_log1p:
        do_norm_log = False
    if do_norm_log:
        sc.pp.normalize_total(ad_fold, target_sum=float(args.target_sum))
        sc.pp.log1p(ad_fold)

    hvg_names = _fit_hvg_names_train_only(ad_fold, train_bags, n_top_genes=args.n_hvg)
    ad_fold = ad_fold[:, hvg_names].copy()
    ad_fold = _scale_from_train(ad_fold, train_bags, max_value=args.scale_max_value)
    ad_fold.uns["fold_preprocess"].update({
        "do_normalize_log1p": bool(do_norm_log),
        "train_max_before_norm": float(train_max),
        "target_sum": float(args.target_sum),
        "min_cells_train": int(args.min_cells_train),
        "n_hvg_requested": int(args.n_hvg),
        "n_hvg_actual": int(ad_fold.n_vars),
    })
    return ad_fold


class TCRBertEncoder(nn.Module):
    def __init__(self, model_dir: str, out_dim: int, unfreeze_last_n: int = 0):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        H = self.backbone.config.hidden_size
        self.proj = nn.Linear(H, out_dim)
        self.max_len = int(getattr(self.backbone.config, "max_position_embeddings", 64))
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "encoder") and unfreeze_last_n > 0:
            L = len(self.backbone.encoder.layer)
            for i in range(max(0, L - unfreeze_last_n), L):
                for p in self.backbone.encoder.layer[i].parameters():
                    p.requires_grad = True
        self._cache: Dict[str, Dict] = {}

    @staticmethod
    def _fmt(seq: str) -> str:
        s = clean_aa(seq)
        return " ".join(list(s)) if s else "X"

    def encode_tokens(self, seqs: List[str]) -> Tuple[Tensor, Tensor]:
        dev = next(self.parameters()).device
        encs = []
        for s in seqs:
            k = f"AA::{s}"
            if k not in self._cache:
                self._cache[k] = self.tok(
                    self._fmt(s), add_special_tokens=True, padding=False,
                    truncation=True, max_length=self.max_len, return_tensors=None,
                )
            encs.append(self._cache[k])
        batch = self.tok.pad(encs, padding=True, return_tensors="pt")
        toks = {k: v.to(dev) for k, v in batch.items()}
        hidden = self.backbone(**toks).last_hidden_state
        att = toks["attention_mask"].bool()
        ids = toks["input_ids"]
        keep = att.clone()
        if self.tok.cls_token_id is not None:
            keep = keep & (ids != self.tok.cls_token_id)
        if self.tok.sep_token_id is not None:
            keep = keep & (ids != self.tok.sep_token_id)
        if self.tok.pad_token_id is not None:
            keep = keep & (ids != self.tok.pad_token_id)
        hidden = hidden.masked_fill(~keep.unsqueeze(-1), 0.0)
        return self.proj(hidden), ~keep


class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, logvar_min: float = -6.0, logvar_max: float = 2.0):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, x: Tensor):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(self.logvar_min, self.logvar_max)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, mu, logvar, kl


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor):
        score = self.score(x).squeeze(-1)
        att = torch.softmax(score, dim=0)
        pooled = torch.sum(att.unsqueeze(-1) * x, dim=0)
        return pooled, att


def scatter_mean_by_clone(x: Tensor, cell_to_clone: Tensor, n_clones: int) -> Tensor:
    if x.dim() != 2:
        raise ValueError(f"x must be [N, D], got shape={tuple(x.shape)}")
    cell_to_clone = cell_to_clone.to(device=x.device, dtype=torch.long)
    out = x.new_zeros((int(n_clones), x.size(-1)))
    cnt = x.new_zeros((int(n_clones), 1))
    out.index_add_(0, cell_to_clone, x)
    cnt.index_add_(0, cell_to_clone, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / cnt.clamp_min(1.0)


def normalized_log_clone_size(clone_size: Tensor) -> Tensor:
    s = torch.log1p(clone_size.float()).view(-1, 1)
    return s / s.max().clamp_min(1.0)


def append_clone_size_feature(z_clone: Tensor, clone_size: Optional[Tensor], use_clone_size_feature: bool = True) -> Tensor:
    if not use_clone_size_feature:
        return z_clone
    if clone_size is None:
        clone_size = torch.ones(z_clone.size(0), device=z_clone.device, dtype=z_clone.dtype)
    size_feat = normalized_log_clone_size(clone_size.to(z_clone.device))
    return torch.cat([z_clone, size_feat.to(dtype=z_clone.dtype)], dim=-1)


def safe_logit(prob: Tensor, eps: float = 1e-6) -> Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.log(prob) - torch.log1p(-prob)


def clone_size_prior_weights(
    clone_size: Optional[Tensor],
    n_clones: int,
    device,
    dtype,
    use_clone_size_feature: bool = True,
) -> Tensor:
    if (not use_clone_size_feature) or (clone_size is None):
        return torch.full((n_clones,), 1.0 / max(1, n_clones), device=device, dtype=dtype)
    s = torch.log1p(clone_size.to(device=device, dtype=dtype)).view(-1)
    if s.numel() != n_clones or float(s.sum().detach().cpu()) <= 0:
        return torch.full((n_clones,), 1.0 / max(1, n_clones), device=device, dtype=dtype)
    return s / s.sum().clamp_min(1e-6)


class CloneEvidenceMILHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patient_head: str = "noisy_or",
        clone_size_prior_weight: float = DEFAULT_CLONE_SIZE_PRIOR_WEIGHT,
        clone_evidence_scale: float = DEFAULT_CLONE_EVIDENCE_SCALE,
        hidden_dim: int = 96,
        dropout: float = 0.25,
        use_clone_size_feature: bool = True,
    ):
        super().__init__()
        self.patient_head = str(patient_head)
        if self.patient_head not in {"attn_pool", "clone_score", "noisy_or"}:
            raise ValueError(f"Unknown patient_head={self.patient_head}")
        self.clone_size_prior_weight = float(clone_size_prior_weight)
        self.clone_evidence_scale = float(clone_evidence_scale)
        self.use_clone_size_feature = bool(use_clone_size_feature)
        self.pool = AttentionPooling(input_dim)
        self.patient_classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.clone_score_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.clone_attn_score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, z_clone_full: Tensor, clone_size: Optional[Tensor] = None):
        n_clones = z_clone_full.size(0)
        if n_clones <= 0:
            raise ValueError("Bag contains no clones.")

        if self.patient_head == "attn_pool":
            patient, att = self.pool(z_clone_full)
            logit = self.patient_classifier(patient).squeeze(-1)
            clone_probs = torch.sigmoid(self.clone_score_head(z_clone_full).squeeze(-1)).detach()
            return logit, patient, att, clone_probs

        clone_logits = self.clone_score_head(z_clone_full).squeeze(-1)
        clone_probs = torch.sigmoid(clone_logits).clamp(1e-6, 1.0 - 1e-6)

        learned_att = torch.softmax(self.clone_attn_score(z_clone_full).squeeze(-1), dim=0)
        size_prior = clone_size_prior_weights(
            clone_size=clone_size,
            n_clones=n_clones,
            device=z_clone_full.device,
            dtype=z_clone_full.dtype,
            use_clone_size_feature=self.use_clone_size_feature,
        )
        beta = min(max(self.clone_size_prior_weight, 0.0), 1.0)
        att = (1.0 - beta) * learned_att + beta * size_prior
        att = att / att.sum().clamp_min(1e-6)

        if self.patient_head == "clone_score":
            patient_prob = torch.sum(att * clone_probs)
        else:
            log_not = torch.log1p(-clone_probs)
            patient_prob = 1.0 - torch.exp(self.clone_evidence_scale * torch.sum(att * log_not))

        patient = torch.sum(att.unsqueeze(-1) * z_clone_full, dim=0)
        logit = safe_logit(patient_prob)
        return logit, patient, att, clone_probs


class RNAOnlyModel(nn.Module):

    def __init__(
        self,
        gene_dim: int,
        gene_out_dim: int = EMB_D,
        latent_d: int = SOLO_LATENT_D,
        use_clone_size_feature: bool = True,
        patient_head: str = DEFAULT_FUSED_PATIENT_HEAD,
        clone_size_prior_weight: float = DEFAULT_CLONE_SIZE_PRIOR_WEIGHT,
        clone_evidence_scale: float = DEFAULT_CLONE_EVIDENCE_SCALE,
    ):
        super().__init__()
        self.use_clone_size_feature = bool(use_clone_size_feature)
        self.gene_encoder = GeneMLPEncoder(
            input_dim=gene_dim, out_dim=gene_out_dim,
            width=512, depth=3, mult=4, dropout=0.1,
            use_attention=True, n_latents=64, n_heads=8,
            latent_self_attn=True, select_topk=512, abs_topk=True,
        )
        self.gene_norm = nn.LayerNorm(gene_out_dim)


        self.z_vae = VAEBottleneck(gene_out_dim, latent_d)

        self.gene_decoder = RNADecoder(
            d_model=latent_d, gene_dim=gene_dim,
            width=512, depth=3, mult=4, dropout=0.1, use_output_bias=True,
        )
        pool_dim = latent_d + (1 if self.use_clone_size_feature else 0)
        self.mil_head = CloneEvidenceMILHead(
            input_dim=pool_dim,
            patient_head=patient_head,
            clone_size_prior_weight=clone_size_prior_weight,
            clone_evidence_scale=clone_evidence_scale,
            hidden_dim=96,
            dropout=0.25,
            use_clone_size_feature=self.use_clone_size_feature,
        )

    def encode_cells(self, rna: Tensor) -> Dict[str, Tensor]:
        h = self.gene_norm(self.gene_encoder(rna))
        z, mu, logvar, kl_per_cell = self.z_vae(h)
        return {
            "z_cell": z,
            "mu": mu,
            "logvar": logvar,
            "kl_per_cell": kl_per_cell,
        }

    def aux_loss(self, z_cell: Tensor, rna_true: Tensor) -> Tensor:
        pred = self.gene_decoder(z_cell)
        return F.mse_loss(pred, rna_true)

    def _encode_all_cells(self, rna: Tensor, chunk_size: int = 0, compute_aux_loss: bool = True):
        if chunk_size is None or chunk_size <= 0 or rna.size(0) <= chunk_size:
            out = self.encode_cells(rna)
            z_cell = out["z_cell"]
            aux_recon = self.aux_loss(z_cell, rna) if compute_aux_loss else None
            aux_kl = out["kl_per_cell"].mean()
            return z_cell, aux_recon, aux_kl

        z_chunks, recon_vals, kl_vals = [], [], []
        for start in range(0, rna.size(0), chunk_size):
            end = min(start + chunk_size, rna.size(0))
            out_b = self.encode_cells(rna[start:end])
            z_chunks.append(out_b["z_cell"])
            kl_vals.append(out_b["kl_per_cell"].mean())
            if compute_aux_loss:
                recon_vals.append(self.aux_loss(out_b["z_cell"], rna[start:end]))
        z_cell = torch.cat(z_chunks, dim=0)
        aux_recon = torch.stack(recon_vals).mean() if compute_aux_loss else None
        aux_kl = torch.stack(kl_vals).mean()
        return z_cell, aux_recon, aux_kl

    def forward_patient(self, rna: Tensor, tra_list: List[str], trb_list: List[str],
                        cell_to_clone: Optional[Tensor] = None, clone_size: Optional[Tensor] = None,
                        chunk_size: int = 0, compute_aux_loss: bool = True):
        z_cell, aux_recon, aux_kl = self._encode_all_cells(
            rna, chunk_size=chunk_size, compute_aux_loss=compute_aux_loss
        )
        if cell_to_clone is None:
            cell_to_clone = torch.arange(z_cell.size(0), device=z_cell.device, dtype=torch.long)
        else:
            cell_to_clone = cell_to_clone.to(z_cell.device).long()
        n_clones = int(cell_to_clone.max().item()) + 1 if cell_to_clone.numel() > 0 else 0
        z_clone = scatter_mean_by_clone(z_cell, cell_to_clone, n_clones)
        z_clone_full = append_clone_size_feature(z_clone, clone_size, self.use_clone_size_feature)
        logit, patient, att, clone_probs = self.mil_head(z_clone_full, clone_size=clone_size)
        ret = {
            "logit": logit,
            "patient": patient,
            "attn": att,
            "clone_probs": clone_probs,
            "z_clone": z_clone,
            "z_cell": z_cell,
            "kl": aux_kl,
        }
        if compute_aux_loss:
            ret["recon"] = aux_recon
        return ret


class TCROnlyModel(nn.Module):

    def __init__(
        self,
        tcrbert_dir: str,
        emb_dim: int = EMB_D,
        latent_d: int = SOLO_LATENT_D,
        unfreeze_last_n: int = UNFREEZE_LAST_N,
        use_clone_size_feature: bool = True,
        patient_head: str = DEFAULT_FUSED_PATIENT_HEAD,
        clone_size_prior_weight: float = DEFAULT_CLONE_SIZE_PRIOR_WEIGHT,
        clone_evidence_scale: float = DEFAULT_CLONE_EVIDENCE_SCALE,
    ):
        super().__init__()
        self.use_clone_size_feature = bool(use_clone_size_feature)
        self.trb_encoder = TCRBertEncoder(tcrbert_dir, out_dim=emb_dim, unfreeze_last_n=unfreeze_last_n)
        self.tra_encoder = TCRBertEncoder(tcrbert_dir, out_dim=emb_dim, unfreeze_last_n=0)
        self.tcr_norm = nn.LayerNorm(emb_dim)
        self.tcr_mix = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )

        self.z_vae = VAEBottleneck(emb_dim, latent_d)

        self.trb_decoder = CDR3SeqDecoder(
            memory_dim=latent_d, d_model=latent_d,
            vocab_size=vocab_size, max_len=MAX_LEN, pad_idx=pad_idx,
            n_layers=2, n_heads=4, d_ff=512, dropout=0.1, tie_weights=True,
        )
        self.tra_decoder = CDR3SeqDecoder(
            memory_dim=latent_d, d_model=latent_d,
            vocab_size=vocab_size, max_len=MAX_LEN, pad_idx=pad_idx,
            n_layers=2, n_heads=4, d_ff=512, dropout=0.1, tie_weights=True,
        )
        pool_dim = latent_d + (1 if self.use_clone_size_feature else 0)
        self.mil_head = CloneEvidenceMILHead(
            input_dim=pool_dim,
            patient_head=patient_head,
            clone_size_prior_weight=clone_size_prior_weight,
            clone_evidence_scale=clone_evidence_scale,
            hidden_dim=96,
            dropout=0.25,
            use_clone_size_feature=self.use_clone_size_feature,
        )

    def encode_cells(self, tra_list: List[str], trb_list: List[str]) -> Dict[str, Tensor]:
        trb_tok_full, trb_pad = self.trb_encoder.encode_tokens(trb_list)
        trb_tok = self.tcr_norm(trb_tok_full)
        valid_b = (~trb_pad).float().unsqueeze(-1)
        z_trb = (trb_tok * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp_min(1.0)

        tra_tok_full, tra_pad = self.tra_encoder.encode_tokens(tra_list)
        tra_tok = self.tcr_norm(tra_tok_full)
        valid_a = (~tra_pad).float().unsqueeze(-1)
        z_tra = (tra_tok * valid_a).sum(dim=1) / valid_a.sum(dim=1).clamp_min(1.0)

        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))
        z, mu, logvar, kl_per_cell = self.z_vae(z_tcr_raw)
        return {
            "z_cell": z,
            "mu": mu,
            "logvar": logvar,
            "kl_per_cell": kl_per_cell,
        }

    def _cdr3_loss(self, decoder, cdr3_list: List[str], z_cell: Tensor) -> Tensor:
        ids = tcr_prepare_ar_batch(cdr3_list, device=z_cell.device)
        logits, target = decoder(z_cell, ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_idx)

    def aux_loss(self, z_cell: Tensor, tra_list: List[str], trb_list: List[str]) -> Tensor:
        l_trb = self._cdr3_loss(self.trb_decoder, trb_list, z_cell)
        valid_a_idx = [i for i, s in enumerate(tra_list) if len(clean_aa(s)) > 0]
        if len(valid_a_idx) > 0:
            idx = torch.tensor(valid_a_idx, dtype=torch.long, device=z_cell.device)
            l_tra = self._cdr3_loss(self.tra_decoder, [tra_list[i] for i in valid_a_idx], z_cell[idx])
        else:
            l_tra = z_cell.sum() * 0.0
        return l_trb + ALPHA_RECON_WEIGHT * l_tra

    def _encode_all_cells(self, tra_list: List[str], trb_list: List[str], chunk_size: int = 0, compute_aux_loss: bool = True):
        N = len(tra_list)
        if chunk_size is None or chunk_size <= 0 or N <= chunk_size:
            out = self.encode_cells(tra_list, trb_list)
            z_cell = out["z_cell"]
            aux_recon = self.aux_loss(z_cell, tra_list, trb_list) if compute_aux_loss else None
            aux_kl = out["kl_per_cell"].mean()
            return z_cell, aux_recon, aux_kl

        z_chunks, recon_vals, kl_vals = [], [], []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            tra_b = tra_list[start:end]
            trb_b = trb_list[start:end]
            out_b = self.encode_cells(tra_b, trb_b)
            z_chunks.append(out_b["z_cell"])
            kl_vals.append(out_b["kl_per_cell"].mean())
            if compute_aux_loss:
                recon_vals.append(self.aux_loss(out_b["z_cell"], tra_b, trb_b))
        z_cell = torch.cat(z_chunks, dim=0)
        aux_recon = torch.stack(recon_vals).mean() if compute_aux_loss else None
        aux_kl = torch.stack(kl_vals).mean()
        return z_cell, aux_recon, aux_kl

    def forward_patient(self, rna: Tensor, tra_list: List[str], trb_list: List[str],
                        cell_to_clone: Optional[Tensor] = None, clone_size: Optional[Tensor] = None,
                        chunk_size: int = 0, compute_aux_loss: bool = True):
        z_cell, aux_recon, aux_kl = self._encode_all_cells(
            tra_list, trb_list, chunk_size=chunk_size, compute_aux_loss=compute_aux_loss
        )
        if cell_to_clone is None:
            cell_to_clone = torch.arange(z_cell.size(0), device=z_cell.device, dtype=torch.long)
        else:
            cell_to_clone = cell_to_clone.to(z_cell.device).long()
        n_clones = int(cell_to_clone.max().item()) + 1 if cell_to_clone.numel() > 0 else 0
        z_clone = scatter_mean_by_clone(z_cell, cell_to_clone, n_clones)
        z_clone_full = append_clone_size_feature(z_clone, clone_size, self.use_clone_size_feature)
        logit, patient, att, clone_probs = self.mil_head(z_clone_full, clone_size=clone_size)
        ret = {
            "logit": logit,
            "patient": patient,
            "attn": att,
            "clone_probs": clone_probs,
            "z_clone": z_clone,
            "z_cell": z_cell,
            "kl": aux_kl,
        }
        if compute_aux_loss:
            ret["recon"] = aux_recon
        return ret


class FusedModel(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        tcrbert_dir: str,
        gene_out_dim: int = EMB_D,
        shared_dim: int = SHARED_D,
        priv_dim: int = PRIV_D,
        unfreeze_last_n: int = UNFREEZE_LAST_N,
        use_clone_size_feature: bool = True,
        patient_head: str = DEFAULT_FUSED_PATIENT_HEAD,
        clone_size_prior_weight: float = DEFAULT_CLONE_SIZE_PRIOR_WEIGHT,
        clone_evidence_scale: float = DEFAULT_CLONE_EVIDENCE_SCALE,
        tcr_priv_weight: float = 0.0,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.priv_dim = priv_dim
        self.use_clone_size_feature = bool(use_clone_size_feature)
        self.patient_head = str(patient_head)
        self.clone_size_prior_weight = float(clone_size_prior_weight)
        self.clone_evidence_scale = float(clone_evidence_scale)
        self.tcr_priv_weight = float(tcr_priv_weight)

        self.gene_encoder = GeneMLPEncoder(
            input_dim=gene_dim, out_dim=gene_out_dim,
            width=512, depth=3, mult=4, dropout=0.1,
            use_attention=True, n_latents=64, n_heads=8,
            latent_self_attn=True, select_topk=512, abs_topk=True,
        )
        self.gene_norm = nn.LayerNorm(gene_out_dim)

        self.trb_encoder = TCRBertEncoder(tcrbert_dir, out_dim=gene_out_dim, unfreeze_last_n=unfreeze_last_n)
        self.tra_encoder = TCRBertEncoder(tcrbert_dir, out_dim=gene_out_dim, unfreeze_last_n=0)
        self.tcr_norm = nn.LayerNorm(gene_out_dim)
        self.tcr_mix = nn.Sequential(
            nn.LayerNorm(gene_out_dim * 2),
            nn.Linear(gene_out_dim * 2, gene_out_dim),
            nn.GELU(),
            nn.LayerNorm(gene_out_dim),
        )

        self.cross_attention = Cross_attention(
            gene_dim=gene_out_dim, cdr_dim=gene_out_dim, emb_dim=gene_out_dim,
            num_heads=8, use_self_attn=True, dropout=0.1, layers=1,
            tcr_scale=TCR_SCALE, fuse_scale=FUSE_SCALE,
            gate_bias=GATE_BIAS, detach_gate_gene=True,
        )

        self.shared_vae = VAEBottleneck(gene_out_dim, shared_dim)
        self.rna_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)
        self.tcr_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)


        self.rna_dec_shared = RNADecoder(
            d_model=shared_dim, gene_dim=gene_dim,
            width=512, depth=3, mult=4, dropout=0.1, use_output_bias=True,
        )
        self.rna_dec_priv = RNADecoder(
            d_model=priv_dim, gene_dim=gene_dim,
            width=512, depth=3, mult=4, dropout=0.1, use_output_bias=False,
        )

        tcr_dec_dmodel = shared_dim + priv_dim
        def _make_cdr3_decoder(memory_dim: int):
            return CDR3SeqDecoder(
                memory_dim=memory_dim,
                d_model=tcr_dec_dmodel,
                vocab_size=vocab_size,
                max_len=MAX_LEN,
                pad_idx=pad_idx,
                n_layers=2,
                n_heads=4,
                d_ff=512,
                dropout=0.1,
                tie_weights=True,
            )

        self.trb_dec_shared = _make_cdr3_decoder(shared_dim)
        self.trb_dec_priv = _make_cdr3_decoder(priv_dim)
        self.tra_dec_shared = _make_cdr3_decoder(shared_dim)
        self.tra_dec_priv = _make_cdr3_decoder(priv_dim)

        self.rna_all_dim = shared_dim + priv_dim
        self.tcr_all_dim = shared_dim + priv_dim
        self.fused_dim = shared_dim + priv_dim + (priv_dim if self.tcr_priv_weight > 0 else 0)

        fused_pool_dim = self.fused_dim + (1 if self.use_clone_size_feature else 0)
        rna_pool_dim = self.rna_all_dim + (1 if self.use_clone_size_feature else 0)
        tcr_pool_dim = self.tcr_all_dim + (1 if self.use_clone_size_feature else 0)

        self.mil_head = CloneEvidenceMILHead(
            input_dim=fused_pool_dim,
            patient_head=self.patient_head,
            clone_size_prior_weight=clone_size_prior_weight,
            clone_evidence_scale=clone_evidence_scale,
            hidden_dim=96,
            dropout=0.25,
            use_clone_size_feature=self.use_clone_size_feature,
        )

        self.rna_pool = AttentionPooling(rna_pool_dim)
        self.rna_head = nn.Sequential(
            nn.LayerNorm(rna_pool_dim),
            nn.Linear(rna_pool_dim, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )
        self.tcr_pool = AttentionPooling(tcr_pool_dim)
        self.tcr_head = nn.Sequential(
            nn.LayerNorm(tcr_pool_dim),
            nn.Linear(tcr_pool_dim, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def encode_cells(self, rna: Tensor, tra_list: List[str], trb_list: List[str]) -> Dict[str, Tensor]:
        z_rna_raw = self.gene_norm(self.gene_encoder(rna))

        trb_tok_full, trb_pad = self.trb_encoder.encode_tokens(trb_list)
        trb_tok = self.tcr_norm(trb_tok_full)
        valid_b = (~trb_pad).float().unsqueeze(-1)
        z_trb = (trb_tok * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp_min(1.0)

        tra_tok_full, tra_pad = self.tra_encoder.encode_tokens(tra_list)
        tra_tok = self.tcr_norm(tra_tok_full)
        valid_a = (~tra_pad).float().unsqueeze(-1)
        z_tra = (tra_tok * valid_a).sum(dim=1) / valid_a.sum(dim=1).clamp_min(1.0)

        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))
        tcr_tok = torch.cat([trb_tok, ALPHA_WEIGHT * tra_tok], dim=1)
        pad_mask = torch.cat([trb_pad, tra_pad], dim=1)
        z_cross, _, gate_score, _ = self.cross_attention(z_rna_raw, tcr_tok, pad_mask=pad_mask)

        z_shared, _, _, kl_shared = self.shared_vae(z_cross)
        z_rna_priv, _, _, kl_rna_priv = self.rna_priv_vae(z_rna_raw)
        z_tcr_priv, _, _, kl_tcr_priv = self.tcr_priv_vae(z_tcr_raw)

        z_rna_all = torch.cat([z_shared, z_rna_priv], dim=-1)
        z_tcr_all = torch.cat([z_shared, z_tcr_priv], dim=-1)
        if self.tcr_priv_weight > 0:
            z_fused = torch.cat([z_shared, z_rna_priv, self.tcr_priv_weight * z_tcr_priv], dim=-1)
        else:
            z_fused = z_rna_all

        kl = kl_shared.mean() + LAMBDA_KL_PRIV_MULT * kl_rna_priv.mean() + LAMBDA_KL_PRIV_MULT * kl_tcr_priv.mean()

        return {
            "z_shared": z_shared,
            "z_rna_priv": z_rna_priv,
            "z_tcr_priv": z_tcr_priv,
            "z_rna_all": z_rna_all,
            "z_tcr_all": z_tcr_all,
            "z_fused": z_fused,
            "kl": kl,
            "kl_shared": kl_shared.mean(),
            "kl_rna_priv": kl_rna_priv.mean(),
            "kl_tcr_priv": kl_tcr_priv.mean(),
            "gate_score": gate_score,
        }

    def rna_recon_loss(self, z_shared: Tensor, z_rna_priv: Tensor, rna_true: Tensor) -> Tensor:
        pred = self.rna_dec_shared(z_shared) + self.rna_dec_priv(z_rna_priv)
        return F.mse_loss(pred, rna_true)

    def _cdr3_logitadd_loss(
        self,
        dec_shared: nn.Module,
        dec_priv: nn.Module,
        cdr3_list: List[str],
        z_shared: Tensor,
        z_tcr_priv: Tensor,
    ) -> Tensor:
        ids = tcr_prepare_ar_batch(cdr3_list, device=z_shared.device)
        logits_s, target = dec_shared(z_shared, ids)
        logits_p, _ = dec_priv(z_tcr_priv, ids)
        logits = logits_s + logits_p
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_idx)

    def full_tcr_recon_loss(self, tra_list: List[str], trb_list: List[str], z_shared: Tensor, z_tcr_priv: Tensor) -> Tensor:
        l_trb = self._cdr3_logitadd_loss(self.trb_dec_shared, self.trb_dec_priv, trb_list, z_shared, z_tcr_priv)
        valid_a_idx = [i for i, s in enumerate(tra_list) if len(clean_aa(s)) > 0]
        if len(valid_a_idx) > 0:
            idx = torch.tensor(valid_a_idx, dtype=torch.long, device=z_shared.device)
            l_tra = self._cdr3_logitadd_loss(
                self.tra_dec_shared,
                self.tra_dec_priv,
                [tra_list[i] for i in valid_a_idx],
                z_shared[idx],
                z_tcr_priv[idx],
            )
        else:
            l_tra = z_shared.sum() * 0.0
        return l_trb + ALPHA_RECON_WEIGHT * l_tra

    def hsic_orth(self, z_shared, z_priv):
        b = z_shared.size(0)
        if b < 2:
            return torch.tensor(0.0, device=z_shared.device)
        dist_s = torch.cdist(z_shared, z_shared, p=2) ** 2
        dist_p = torch.cdist(z_priv, z_priv, p=2) ** 2
        sigma_s = dist_s[dist_s > 0].mean().clamp_min(1e-4) if dist_s.sum() > 0 else torch.tensor(1.0, device=z_shared.device)
        sigma_p = dist_p[dist_p > 0].mean().clamp_min(1e-4) if dist_p.sum() > 0 else torch.tensor(1.0, device=z_shared.device)
        k_s = torch.exp(-dist_s / (2.0 * sigma_s))
        k_p = torch.exp(-dist_p / (2.0 * sigma_p))
        h = torch.eye(b, device=z_shared.device) - (1.0 / b) * torch.ones((b, b), device=z_shared.device)
        k_sc = torch.mm(h, torch.mm(k_s, h))
        return torch.sum(k_sc * k_p) / ((b - 1) ** 2)

    def aux_losses(self, out, rna_true, tra_list, trb_list):
        l_rna = self.rna_recon_loss(out["z_shared"], out["z_rna_priv"], rna_true)
        l_tcr = self.full_tcr_recon_loss(tra_list, trb_list, out["z_shared"], out["z_tcr_priv"])
        l_orth = self.hsic_orth(out["z_shared"], out["z_rna_priv"]) + self.hsic_orth(out["z_shared"], out["z_tcr_priv"])
        return {"rna_recon": l_rna, "tcr_recon": l_tcr, "orth": l_orth}

    def _encode_all_cells(self, rna: Tensor, tra_list: List[str], trb_list: List[str],
                          chunk_size: int = 0, compute_aux_loss: bool = True):
        if chunk_size is None or chunk_size <= 0 or rna.size(0) <= chunk_size:
            out = self.encode_cells(rna, tra_list, trb_list)
            aux = self.aux_losses(out, rna, tra_list, trb_list) if compute_aux_loss else {}
            return out, aux

        outs = {"z_fused": [], "z_rna_all": [], "z_tcr_all": [], "z_shared": [], "z_rna_priv": [], "z_tcr_priv": []}
        kl_vals = []
        aux_vals = {"rna_recon": [], "tcr_recon": [], "orth": []}
        for start in range(0, rna.size(0), chunk_size):
            end = min(start + chunk_size, rna.size(0))
            out_b = self.encode_cells(rna[start:end], tra_list[start:end], trb_list[start:end])
            for k in outs:
                outs[k].append(out_b[k])
            kl_vals.append(out_b["kl"])
            if compute_aux_loss:
                aux_b = self.aux_losses(out_b, rna[start:end], tra_list[start:end], trb_list[start:end])
                for k in aux_vals:
                    aux_vals[k].append(aux_b[k])
        out = {k: torch.cat(v, dim=0) for k, v in outs.items()}
        out["kl"] = torch.stack(kl_vals).mean()
        aux = {k: torch.stack(v).mean() for k, v in aux_vals.items()} if compute_aux_loss else {}
        return out, aux

    def forward_patient(self, rna: Tensor, tra_list: List[str], trb_list: List[str],
                        cell_to_clone: Optional[Tensor] = None, clone_size: Optional[Tensor] = None,
                        chunk_size: int = 0, compute_aux_loss: bool = True):
        out, aux = self._encode_all_cells(rna, tra_list, trb_list, chunk_size=chunk_size, compute_aux_loss=compute_aux_loss)

        if cell_to_clone is None:
            cell_to_clone = torch.arange(out["z_fused"].size(0), device=out["z_fused"].device, dtype=torch.long)
        else:
            cell_to_clone = cell_to_clone.to(out["z_fused"].device).long()
        n_clones = int(cell_to_clone.max().item()) + 1 if cell_to_clone.numel() > 0 else 0

        z_fused_clone = scatter_mean_by_clone(out["z_fused"], cell_to_clone, n_clones)
        z_rna_clone = scatter_mean_by_clone(out["z_rna_all"], cell_to_clone, n_clones)
        z_tcr_clone = scatter_mean_by_clone(out["z_tcr_all"], cell_to_clone, n_clones)

        z_fused_full = append_clone_size_feature(z_fused_clone, clone_size, self.use_clone_size_feature)
        z_rna_full = append_clone_size_feature(z_rna_clone, clone_size, self.use_clone_size_feature)
        z_tcr_full = append_clone_size_feature(z_tcr_clone, clone_size, self.use_clone_size_feature)

        logit, patient, att, clone_probs = self.mil_head(z_fused_full, clone_size=clone_size)

        rna_patient, rna_att = self.rna_pool(z_rna_full)
        tcr_patient, tcr_att = self.tcr_pool(z_tcr_full)
        rna_logit = self.rna_head(rna_patient).squeeze(-1)
        tcr_logit = self.tcr_head(tcr_patient).squeeze(-1)

        ret = {
            "logit": logit,
            "patient": patient,
            "kl": out["kl"],
            "attn": att,
            "rna_logit": rna_logit,
            "tcr_logit": tcr_logit,
            "rna_attn": rna_att,
            "tcr_attn": tcr_att,
            "clone_probs": clone_probs,
            "z_fused_clone": z_fused_clone,
            "z_rna_clone": z_rna_clone,
            "z_tcr_clone": z_tcr_clone,
        }
        if compute_aux_loss:
            ret.update(aux)
        return ret


def make_clone_id_df(obs: pd.DataFrame, clone_key: str = "TCR_sclone.id") -> pd.Series:
    clone_key_raw = str(clone_key)
    clone_key_upper = clone_key_raw.upper()
    if clone_key_raw in obs.columns:
        s = obs[clone_key_raw].fillna("").astype(str)

        trb = obs.get("TRB_aa", pd.Series("", index=obs.index)).fillna("").astype(str)
        tra = obs.get("TRA_aa", pd.Series("", index=obs.index)).fillna("").astype(str)
        fallback = tra + "|" + trb
        s = s.where(s.str.len() > 0, fallback)
        return s
    if clone_key_upper == "TRB":
        return obs["TRB_aa"].fillna("").astype(str)
    if clone_key_upper in {"TRA_TRB", "TRA+TRB", "PAIRED"}:
        tra = obs["TRA_aa"].fillna("").astype(str)
        trb = obs["TRB_aa"].fillna("").astype(str)
        return tra + "|" + trb

    if "TCR_clone.id" in obs.columns:
        return obs["TCR_clone.id"].fillna("").astype(str)
    return obs["TRB_aa"].fillna("").astype(str)


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
            "group_id": str(g["group_id"].iloc[0]),
            "y": y,
            "severity": str(g["severity"].iloc[0]),
            "response_binary": "effective_response" if y == 1 else "non_MPR",
            "n_cells": int(len(g)),
            "n_clones": int(g["clone_id"].nunique()),
        })
    return pd.DataFrame(rows).set_index("bag_id")


class PatientBagDataset(Dataset):
    def __init__(self, ad: sc.AnnData, bag_ids: List[str]):
        self.ad = ad
        self.bag_ids = [str(x) for x in bag_ids]
        self.X_np = np.asarray(ad.X)
        self.obs = ad.obs.copy()
        if "clone_id" not in self.obs.columns:
            self.obs["clone_id"] = make_clone_id_df(self.obs).values
        self.obs["_pos"] = np.arange(ad.n_obs)
        self.items: Dict[str, Dict] = {}
        self.labels: Dict[str, int] = {}

        for bid in self.bag_ids:
            obs_bid = self.obs[self.obs["bag_id"].astype(str).values == str(bid)].copy()
            if len(obs_bid) == 0:
                raise ValueError(f"No cells for bag_id={bid}")
            y_values = obs_bid["y"].astype(int).unique()
            if len(y_values) != 1:
                raise ValueError(f"Inconsistent labels inside bag_id={bid}: {y_values}")
            y = int(y_values[0])
            self.labels[str(bid)] = y

            clone_sizes = obs_bid.groupby("clone_id", observed=True).size().sort_values(ascending=False)
            clone_to_idx = {str(cid): i for i, cid in enumerate(clone_sizes.index.astype(str).tolist())}
            obs_bid["_clone_idx"] = obs_bid["clone_id"].astype(str).map(clone_to_idx).astype(int)
            obs_bid = obs_bid.sort_values(["_clone_idx", "_pos"])

            pos = obs_bid["_pos"].values.astype(int)
            cell_to_clone = obs_bid["_clone_idx"].values.astype(np.int64)
            clone_size = clone_sizes.values.astype(np.float32)

            self.items[str(bid)] = {
                "rna": self.X_np[pos].astype(np.float32),
                "tra": obs_bid["TRA_aa"].astype(str).tolist(),
                "trb": obs_bid["TRB_aa"].astype(str).tolist(),
                "cell_to_clone": cell_to_clone,
                "clone_size": clone_size,
                "clone_id": clone_sizes.index.astype(str).tolist(),
                "n_cells": int(len(pos)),
                "n_clones": int(len(clone_size)),
                "sampleID": str(obs_bid["sampleID"].iloc[0]),
                "PatientID": str(obs_bid["PatientID"].iloc[0]),
                "severity": str(obs_bid["severity"].iloc[0]),
            }

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, i):
        bid = str(self.bag_ids[i])
        item = self.items[bid]
        return {
            "bag_id": bid,
            "sampleID": item["sampleID"],
            "PatientID": item["PatientID"],
            "severity": item["severity"],
            "rna": torch.from_numpy(item["rna"]).float(),
            "tra": item["tra"],
            "trb": item["trb"],
            "cell_to_clone": torch.from_numpy(item["cell_to_clone"]).long(),
            "clone_size": torch.from_numpy(item["clone_size"]).float(),
            "clone_id": item["clone_id"],
            "y": torch.tensor(float(self.labels[bid]), dtype=torch.float32),
            "n_cells": item["n_cells"],
            "n_clones": item["n_clones"],
        }


def patient_collate(batch):
    return batch


def find_best_threshold(
    y_true,
    y_prob,
    metric: str = "balanced_accuracy",
    min_t: float = 0.05,
    max_t: float = 0.95,
    n_grid: int = 181,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5, np.nan
    thresholds = np.linspace(min_t, max_t, n_grid)
    best_t = 0.5
    best_score = -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            score = sensitivity + specificity - 1.0
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
        ):
            best_score = score
            best_t = float(t)
    return best_t, float(best_score)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "AUROC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "AUPRC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = tn / (tn + fp + 1e-12)
    out["TN"] = int(tn)
    out["FP"] = int(fp)
    out["FN"] = int(fn)
    out["TP"] = int(tp)
    return out


def compute_pooled_oof_metrics(pred_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    y = pred_df["y_true"].astype(int).values
    p = pred_df["prob"].astype(float).values
    if "threshold" in pred_df.columns:

        y_pred = (p >= pred_df["threshold"].astype(float).values).astype(int)
        threshold_repr = float(np.nanmedian(pred_df["threshold"].astype(float).values))
    else:
        y_pred = (p >= 0.5).astype(int)
        threshold_repr = 0.5

    out = {
        "feature": feature,
        "n_predictions": int(len(pred_df)),
        "n_unique_bags": int(pred_df["bag_id"].nunique()) if "bag_id" in pred_df.columns else int(len(pred_df)),
        "n_unique_patients": int(pred_df["PatientID"].nunique()) if "PatientID" in pred_df.columns else np.nan,
        "AUROC_pooled": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "AUPRC_pooled": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "accuracy_pooled": accuracy_score(y, y_pred),
        "balanced_accuracy_pooled": balanced_accuracy_score(y, y_pred),
        "F1_pooled": f1_score(y, y_pred, zero_division=0),
        "precision_pooled": precision_score(y, y_pred, zero_division=0),
        "sensitivity_recall_pooled": recall_score(y, y_pred, zero_division=0),
        "threshold_median": threshold_repr,
    }
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    out["specificity_pooled"] = tn / (tn + fp + 1e-12)
    out["TN_pooled"] = int(tn)
    out["FP_pooled"] = int(fp)
    out["FN_pooled"] = int(fn)
    out["TP_pooled"] = int(tp)
    return pd.DataFrame([out])


def make_pos_weight(y_train, device):
    y_train = np.asarray(y_train).astype(int)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos <= 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, dtype=torch.float32, device=device)


def compute_total_loss(method: str, model, item, device, bce, chunk_size, aux_recon):
    rna = item["rna"].to(device)
    y = item["y"].to(device)
    cell_to_clone = item.get("cell_to_clone", None)
    clone_size = item.get("clone_size", None)
    if cell_to_clone is not None:
        cell_to_clone = cell_to_clone.to(device)
    if clone_size is not None:
        clone_size = clone_size.to(device)

    out = model.forward_patient(
        rna, item["tra"], item["trb"],
        cell_to_clone=cell_to_clone, clone_size=clone_size,
        chunk_size=chunk_size, compute_aux_loss=aux_recon,
    )
    l_cls = bce(out["logit"].view(1), y.view(1))
    stats = {"cls": float(l_cls.detach().cpu()), "rna_cls": 0.0, "tcr_cls": 0.0,
             "kl": 0.0, "recon": 0.0, "tcr_recon": 0.0, "orth": 0.0}

    if method in {"rna_only", "tcr_only"}:

        l = l_cls
        if ("kl" in out) and out["kl"] is not None:
            l = l + LAMBDA_SOLO_KL * out["kl"]
            stats["kl"] = float(out["kl"].detach().cpu())
        if aux_recon and ("recon" in out) and out["recon"] is not None:
            l = l + LAMBDA_SOLO_RECON * out["recon"]
            stats["recon"] = float(out["recon"].detach().cpu())
    elif method == "fused":
        l = l_cls + LAMBDA_KL * out["kl"]
        stats["kl"] = float(out["kl"].detach().cpu())

        if LAMBDA_RNA_CLS > 0 and ("rna_logit" in out):
            l_rna_cls = bce(out["rna_logit"].view(1), y.view(1))
            l = l + LAMBDA_RNA_CLS * l_rna_cls
            stats["rna_cls"] = float(l_rna_cls.detach().cpu())

        if LAMBDA_TCR_CLS > 0 and ("tcr_logit" in out):
            l_tcr_cls = bce(out["tcr_logit"].view(1), y.view(1))
            l = l + LAMBDA_TCR_CLS * l_tcr_cls
            stats["tcr_cls"] = float(l_tcr_cls.detach().cpu())

        if aux_recon and ("rna_recon" in out):
            l = l + LAMBDA_RNA_RECON * out["rna_recon"] + LAMBDA_TCR_RECON * out["tcr_recon"] + LAMBDA_ORTH * out["orth"]
            stats["recon"] = float(out["rna_recon"].detach().cpu())
            stats["tcr_recon"] = float(out["tcr_recon"].detach().cpu())
            stats["orth"] = float(out["orth"].detach().cpu())
    else:
        raise ValueError(f"Unknown method={method}")
    return l, stats


def train_one_epoch(method, model, loader, optimizer, device, pos_weight, chunk_size, aux_recon):
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    totals = {"loss": 0.0, "cls": 0.0, "rna_cls": 0.0, "tcr_cls": 0.0, "recon": 0.0, "tcr_recon": 0.0, "orth": 0.0, "kl": 0.0}
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        loss_batch = 0.0
        batch_stats = {k: 0.0 for k in totals}
        for item in batch:
            l, stats = compute_total_loss(method, model, item, device, bce, chunk_size, aux_recon)
            for k, v in stats.items():
                batch_stats[k] += v
            loss_batch = loss_batch + l
            n += 1
        loss_batch = loss_batch / max(1, len(batch))
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_stats["loss"] += float(loss_batch.detach().cpu()) * len(batch)
        for k in totals:
            totals[k] += batch_stats[k]
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def evaluate_model(model, loader, device, chunk_size, method_label: str, threshold: float = 0.5):
    model.eval()
    rows = []
    for batch in loader:
        for item in batch:
            rna = item["rna"].to(device)
            cell_to_clone = item.get("cell_to_clone", None)
            clone_size = item.get("clone_size", None)
            if cell_to_clone is not None:
                cell_to_clone = cell_to_clone.to(device)
            if clone_size is not None:
                clone_size = clone_size.to(device)
            out = model.forward_patient(
                rna, item["tra"], item["trb"],
                cell_to_clone=cell_to_clone, clone_size=clone_size,
                chunk_size=chunk_size, compute_aux_loss=False,
            )
            p = torch.sigmoid(out["logit"]).item()
            rows.append({
                "bag_id": item["bag_id"],
                "sampleID": item["sampleID"],
                "PatientID": item["PatientID"],
                "severity": item["severity"],
                "y_true": int(item["y"].item()),
                "prob": float(p),
                "y_pred": int(p >= threshold),
                "threshold": float(threshold),
                "n_cells": int(item["n_cells"]),
                "n_clones": int(item["n_clones"]),
            })
    pred = pd.DataFrame(rows)
    y = pred["y_true"].values.astype(int)
    metrics = {method_label: compute_metrics(y, pred["prob"].values.astype(float), threshold=threshold)}
    return pred, metrics


@torch.no_grad()
def export_clone_evidence(model, loader, device, chunk_size, fold_id: int, threshold: float = 0.5, split: str = "test") -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        for item in batch:
            rna = item["rna"].to(device)
            cell_to_clone = item["cell_to_clone"].to(device)
            clone_size = item["clone_size"].to(device)
            out = model.forward_patient(
                rna, item["tra"], item["trb"],
                cell_to_clone=cell_to_clone, clone_size=clone_size,
                chunk_size=chunk_size, compute_aux_loss=False,
            )
            if "clone_probs" not in out or "attn" not in out:
                continue
            clone_probs = out["clone_probs"].detach().cpu().numpy()
            attn = out["attn"].detach().cpu().numpy()
            clone_sizes = item["clone_size"].detach().cpu().numpy()
            clone_ids = item["clone_id"]
            for k, cid in enumerate(clone_ids):
                rows.append({
                    "fold": int(fold_id),
                    "split": split,
                    "bag_id": item["bag_id"],
                    "sampleID": item["sampleID"],
                    "PatientID": item["PatientID"],
                    "severity": item["severity"],
                    "y_true": int(item["y"].item()),
                    "threshold": float(threshold),
                    "clone_rank": int(k),
                    "clone_id": str(cid),
                    "clone_size": float(clone_sizes[k]) if k < len(clone_sizes) else np.nan,
                    "clone_prob": float(clone_probs[k]) if k < len(clone_probs) else np.nan,
                    "clone_attn": float(attn[k]) if k < len(attn) else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_metrics(per_fold: pd.DataFrame, feature: str) -> pd.DataFrame:
    metric_cols = ["AUROC", "AUPRC", "accuracy", "balanced_accuracy", "F1", "precision", "sensitivity_recall", "specificity"]
    row = {"feature": feature, "n_folds": int(len(per_fold))}
    for m in metric_cols:
        vals = per_fold[m].astype(float).values
        row[f"{m}_mean"] = float(np.nanmean(vals))
        row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else np.nan
        row[f"{m}_median"] = float(np.nanmedian(vals))
    return pd.DataFrame([row])


def make_model(method: str, gene_dim: int, args):
    use_clone_size_feature = not args.no_clone_size_feature
    baseline_head = args.fused_patient_head if args.baseline_patient_head == "same_as_fused" else args.baseline_patient_head
    if method == "rna_only":
        return RNAOnlyModel(
            gene_dim=gene_dim,
            latent_d=args.solo_latent_d,
            use_clone_size_feature=use_clone_size_feature,
            patient_head=baseline_head,
            clone_size_prior_weight=args.clone_size_prior_weight,
            clone_evidence_scale=args.clone_evidence_scale,
        )
    if method == "tcr_only":
        return TCROnlyModel(
            tcrbert_dir=args.tcrbert_dir,
            latent_d=args.solo_latent_d,
            unfreeze_last_n=args.unfreeze_last_n,
            use_clone_size_feature=use_clone_size_feature,
            patient_head=baseline_head,
            clone_size_prior_weight=args.clone_size_prior_weight,
            clone_evidence_scale=args.clone_evidence_scale,
        )
    if method == "fused":
        return FusedModel(
            gene_dim=gene_dim,
            tcrbert_dir=args.tcrbert_dir,
            shared_dim=args.shared_d,
            priv_dim=args.priv_d,
            unfreeze_last_n=args.unfreeze_last_n,
            use_clone_size_feature=use_clone_size_feature,
            patient_head=args.fused_patient_head,
            clone_size_prior_weight=args.clone_size_prior_weight,
            clone_evidence_scale=args.clone_evidence_scale,
            tcr_priv_weight=(args.tcr_priv_weight if args.use_tcr_priv_in_predict else 0.0),
        )
    raise ValueError(f"Unknown method={method}")


def group_stratified_inner_split(train_bags: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, args, seed: int):
    unique_groups = np.unique(groups_train)
    if len(unique_groups) < 4 or len(np.unique(y_train)) < 2:
        return train_bags.tolist(), train_bags.tolist(), False

    group_label = pd.DataFrame({"group": groups_train, "y": y_train}).drop_duplicates("group")
    if group_label["y"].value_counts().min() < 2:
        return train_bags.tolist(), train_bags.tolist(), False
    n_splits_inner = min(args.inner_n_splits, int(group_label["y"].value_counts().min()))
    if n_splits_inner < 2:
        return train_bags.tolist(), train_bags.tolist(), False
    splitter = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(splitter.split(train_bags, y_train, groups_train))
    return train_bags[tr_idx].tolist(), train_bags[va_idx].tolist(), True


def run_cv_for_method(method: str, ad_raw: sc.AnnData, args, out_dir: str) -> pd.DataFrame:
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print(f"\n========== Method: {method} | device={device} ==========")

    bag_table = make_bag_table(ad_raw)
    bag_table.to_csv(os.path.join(out_dir, "bag_table.csv"))

    bags = bag_table.index.astype(str).values
    y = bag_table["y"].astype(int).values
    groups = bag_table["group_id"].astype(str).values

    label_counts = pd.Series(y).value_counts()
    if len(label_counts) < 2:
        raise ValueError("Only one class is present after filtering. Cannot run severity prediction.")
    n_splits = min(args.n_splits, int(label_counts.min()))
    if n_splits < 2:
        raise ValueError(f"Not enough bags in the minority class for CV: label_counts={label_counts.to_dict()}")

    all_pred_rows = []
    all_metric_rows = []
    all_hist_rows = []
    all_clone_evidence = []

    fold_global = 0
    for rep in range(args.n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + rep)
        for fold, (trainval_idx, test_idx) in enumerate(splitter.split(bags, y, groups)):
            fold_global += 1
            trainval_bags = bags[trainval_idx]
            test_bags = bags[test_idx]
            y_trainval = y[trainval_idx]
            groups_trainval = groups[trainval_idx]

            train_bags, val_bags, inner_val_used = group_stratified_inner_split(
                trainval_bags, y_trainval, groups_trainval, args, seed=args.seed + rep * 100 + fold
            )

            print(
                f"\n[{method}] repeat={rep + 1}/{args.n_repeats}, fold={fold + 1}/{n_splits}, "
                f"train_bags={len(train_bags)}, val_bags={len(val_bags)}, test_bags={len(test_bags)}, "
                f"inner_val_used={inner_val_used}"
            )

            ad_fold = prepare_fold_anndata(
                ad_raw,
                train_bags=train_bags,
                val_bags=val_bags,
                test_bags=test_bags.tolist(),
                args=args,
            )

            train_ds = PatientBagDataset(ad_fold, train_bags)
            val_ds = PatientBagDataset(ad_fold, val_bags)
            test_ds = PatientBagDataset(ad_fold, test_bags.tolist())

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=patient_collate)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=patient_collate)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=patient_collate)

            set_seed(args.seed + rep * 1000 + fold)
            model = make_model(method, gene_dim=ad_fold.n_vars, args=args).to(device)

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

            y_train = np.array([train_ds.labels[str(b)] for b in train_bags], dtype=int)
            pos_weight = make_pos_weight(y_train, device=device)

            best_score = -np.inf
            best_state = None
            bad_epochs = 0
            hist_rows = []

            for epoch in range(1, args.epochs + 1):
                tr_stats = train_one_epoch(
                    method, model, train_loader, optimizer, device, pos_weight,
                    chunk_size=args.chunk_size, aux_recon=(not args.no_aux_recon),
                )
                val_threshold_for_eval = 0.5
                val_pred, val_metrics_dict = evaluate_model(
                    model, val_loader, device, args.chunk_size, method, threshold=val_threshold_for_eval
                )
                val_metrics = val_metrics_dict[method]

                score = val_metrics["AUROC"]
                if np.isnan(score):
                    score = val_metrics["balanced_accuracy"]

                row = {
                    "repeat": rep + 1,
                    "fold": fold + 1,
                    "fold_global": fold_global,
                    "epoch": epoch,
                    "inner_val_used": bool(inner_val_used),
                    "val_score": float(score),
                    **{f"train_{k}": v for k, v in tr_stats.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
                }
                hist_rows.append(row)

                if score > best_score + 1e-12:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            selected_threshold = 0.5
            threshold_score = np.nan
            if args.tune_threshold and inner_val_used:
                val_pred, _ = evaluate_model(model, val_loader, device, args.chunk_size, method, threshold=0.5)
                selected_threshold, threshold_score = find_best_threshold(
                    val_pred["y_true"].values,
                    val_pred["prob"].values,
                    metric=args.threshold_metric,
                )

            test_pred, test_metrics_dict = evaluate_model(
                model, test_loader, device, args.chunk_size, method, threshold=selected_threshold
            )
            test_metrics = test_metrics_dict[method]
            test_pred["method"] = method
            test_pred["repeat"] = rep + 1
            test_pred["fold"] = fold + 1
            test_pred["fold_global"] = fold_global
            test_pred["selected_threshold"] = float(selected_threshold)
            test_pred["inner_val_used"] = bool(inner_val_used)
            all_pred_rows.append(test_pred)

            metric_row = {
                "method": method,
                "feature": feature_name_for_method(method, args),
                "repeat": rep + 1,
                "fold": fold + 1,
                "fold_global": fold_global,
                "inner_val_used": bool(inner_val_used),
                "selected_threshold": float(selected_threshold),
                "threshold_score": float(threshold_score) if not np.isnan(threshold_score) else np.nan,
                **test_metrics,
            }
            all_metric_rows.append(metric_row)

            hist_df = pd.DataFrame(hist_rows)
            all_hist_rows.append(hist_df)
            hist_df.to_csv(os.path.join(out_dir, f"history_fold{fold_global:03d}.csv"), index=False)

            if args.export_clone_evidence:
                ce = export_clone_evidence(
                    model, test_loader, device, args.chunk_size,
                    fold_id=fold_global, threshold=selected_threshold, split="test"
                )
                ce["method"] = method
                all_clone_evidence.append(ce)

            print(
                f"[{method}] fold_global={fold_global} "
                f"AUROC={test_metrics['AUROC']:.4f} AUPRC={test_metrics['AUPRC']:.4f} "
                f"bal_acc={test_metrics['balanced_accuracy']:.4f} F1={test_metrics['F1']:.4f} "
                f"threshold={selected_threshold:.3f}"
            )

    pred_df = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    metrics_df = pd.DataFrame(all_metric_rows)
    hist_all = pd.concat(all_hist_rows, axis=0, ignore_index=True) if all_hist_rows else pd.DataFrame()

    pred_df.to_csv(os.path.join(out_dir, "patient_level_predictions.csv"), index=False)
    metrics_df.to_csv(os.path.join(out_dir, "patient_level_metrics_per_fold.csv"), index=False)
    hist_all.to_csv(os.path.join(out_dir, "history_all_folds.csv"), index=False)

    summary = summarize_metrics(metrics_df, feature=feature_name_for_method(method, args))
    summary.insert(0, "method", method)
    summary.to_csv(os.path.join(out_dir, "patient_level_metrics_summary.csv"), index=False)

    pooled = compute_pooled_oof_metrics(pred_df, feature=feature_name_for_method(method, args))
    pooled.insert(0, "method", method)
    pooled.to_csv(os.path.join(out_dir, "patient_level_metrics_pooled_oof.csv"), index=False)

    if all_clone_evidence:
        ce_all = pd.concat(all_clone_evidence, axis=0, ignore_index=True)
        ce_all.to_csv(os.path.join(out_dir, "clone_evidence_all_folds.csv"), index=False)

    print(f"\n[{method}] summary:")
    print(summary.to_string(index=False))
    print(f"\n[{method}] pooled OOF:")
    print(pooled.to_string(index=False))
    return summary


def feature_name_for_method(method: str, args) -> str:
    if method == "rna_only":
        return "NSCLC_RNA_only_code3_VAE"
    if method == "tcr_only":
        return "NSCLC_TCR_only_VAE"
    if method == "fused":
        return f"NSCLC_Fused_code3_RNA_TCR_{args.fused_patient_head}_tcrpriv{int(args.use_tcr_priv_in_predict)}"
    return method


def parse_args():
    p = argparse.ArgumentParser(description="NSCLC pathological response prediction with code3 shared/private RNA-TCR clone-aware MIL.")

    # Paths
    p.add_argument("--rna_path", type=str, default=DEFAULT_RNA_PATH)
    p.add_argument("--meta_path", type=str, default=DEFAULT_META_PATH)
    p.add_argument("--tcr_path", type=str, default=DEFAULT_TCR_PATH)
    p.add_argument("--sample_meta_path", type=str, default=DEFAULT_SAMPLE_META_PATH)
    p.add_argument("--tcrbert_dir", type=str, default=DEFAULT_TCRBERT_DIR)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # NSCLC response task
    p.add_argument("--bag_level", type=str, default="sample", choices=["sample", "patient"],
                   help="sample: one tumor sample per bag, with PatientID-grouped CV; patient: aggregate all cells per patient.")
    p.add_argument("--celltype_filter", type=str, default="CD8",
                   help="Regex filter over NSCLC sub_cell_type/celltype/major_cell_type. Default CD8; use ALL to keep all response-labeled cells.")
    p.add_argument("--require_tcr", action="store_true", default=True,
                   help="Keep only cells with TRB CDR3. Default True for paired RNA/TCR framework.")
    p.add_argument("--no_require_tcr", dest="require_tcr", action="store_false")
    p.add_argument("--clone_key", type=str, default="clonotype",
                   help="Clone column for NSCLC TCR table. Try clonotype/clonotype_number/TRB/TRA_TRB. Fallback is TRB.")
    p.add_argument("--cancer_type", type=str, default="LUSC", choices=["ALL", "LUAD", "LUSC"],
                   help="Cancer type filter for NSCLC metadata.")
    p.add_argument("--response_positive_terms", type=str, default="pCR,MPR",
                   help="Comma-separated response labels treated as positive/effective.")
    p.add_argument("--response_negative_terms", type=str, default="non-MPR",
                   help="Comma-separated response labels treated as negative/non-effective.")

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=1)
    p.add_argument("--inner_n_splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", type=str, default="cuda:0")


    p.add_argument("--min_cells_train", type=int, default=3)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--scale_max_value", type=float, default=10.0)
    p.add_argument("--log1p_if_max_gt", type=float, default=30.0)
    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--force_log1p", action="store_true")
    p.add_argument("--skip_log1p", action="store_true")

    p.add_argument("--solo_latent_d", type=int, default=SOLO_LATENT_D)
    p.add_argument("--shared_d", type=int, default=SHARED_D)
    p.add_argument("--priv_d", type=int, default=PRIV_D)
    p.add_argument("--unfreeze_last_n", type=int, default=UNFREEZE_LAST_N)
    p.add_argument("--fused_patient_head", type=str, default=DEFAULT_FUSED_PATIENT_HEAD,
                   choices=["attn_pool", "clone_score", "noisy_or"])
    p.add_argument("--baseline_patient_head", type=str, default="same_as_fused",
                   choices=["same_as_fused", "attn_pool", "clone_score", "noisy_or"])
    p.add_argument("--clone_size_prior_weight", type=float, default=DEFAULT_CLONE_SIZE_PRIOR_WEIGHT)
    p.add_argument("--clone_evidence_scale", type=float, default=DEFAULT_CLONE_EVIDENCE_SCALE)
    p.add_argument("--no_clone_size_feature", action="store_true")
    p.add_argument("--use_tcr_priv_in_predict", action="store_true", default=False,
                   help="If set, append tcr_priv_weight*z_tcr_priv to the fused prediction representation. Default off.")
    p.add_argument("--tcr_priv_weight", type=float, default=0.3,
                   help="Scale for z_tcr_priv when --use_tcr_priv_in_predict is enabled. Ignored when the switch is off.")

    p.add_argument("--solo_recon_weight", type=float, default=1.0,
                   help="RNA-only/TCR-only VAE reconstruction weight. Default 1.0 to match code3 VAE-style baselines.")
    p.add_argument("--solo_kl_weight", type=float, default=LAMBDA_KL,
                   help="RNA-only/TCR-only VAE KL weight. Default equals --lambda_kl.")
    p.add_argument("--lambda_rna_cls", type=float, default=0.0)
    p.add_argument("--lambda_tcr_cls", type=float, default=0.0)
    p.add_argument("--lambda_kl", type=float, default=LAMBDA_KL)
    p.add_argument("--lambda_kl_priv_mult", type=float, default=LAMBDA_KL_PRIV_MULT)
    p.add_argument("--lambda_orth", type=float, default=LAMBDA_ORTH)
    p.add_argument("--no_aux_recon", action="store_true")


    p.add_argument("--tune_threshold", action="store_true")
    p.add_argument("--threshold_metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "f1", "youden"])
    p.add_argument("--export_clone_evidence", action="store_true", default=True)
    p.add_argument("--no_export_clone_evidence", dest="export_clone_evidence", action="store_false")
    p.add_argument("--methods", type=str, default=",".join(RUN_METHODS),
                   help="Comma-separated methods: rna_only,tcr_only,fused")

    return p.parse_args()


def apply_global_args(args):
    global LAMBDA_SOLO_RECON, LAMBDA_SOLO_KL, LAMBDA_RNA_CLS, LAMBDA_TCR_CLS, LAMBDA_KL, LAMBDA_ORTH, LAMBDA_KL_PRIV_MULT
    LAMBDA_SOLO_RECON = float(args.solo_recon_weight)
    LAMBDA_SOLO_KL = float(args.solo_kl_weight)
    LAMBDA_RNA_CLS = float(args.lambda_rna_cls)
    LAMBDA_TCR_CLS = float(args.lambda_tcr_cls)
    LAMBDA_KL = float(args.lambda_kl)
    LAMBDA_KL_PRIV_MULT = float(args.lambda_kl_priv_mult)
    LAMBDA_ORTH = float(args.lambda_orth)


def main():
    args = parse_args()
    apply_global_args(args)
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    with open(os.path.join(args.out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

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
    )

    make_bag_table(ad).to_csv(os.path.join(args.out_dir, "bag_table.csv"))
    print(f"[Config] use_tcr_priv_in_predict={args.use_tcr_priv_in_predict}, effective_tcr_priv_weight={(args.tcr_priv_weight if args.use_tcr_priv_in_predict else 0.0)}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"rna_only", "tcr_only", "fused"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown method={m}; allowed={allowed}")

    summaries = []
    for method in methods:
        method_out = os.path.join(args.out_dir, method)
        summary = run_cv_for_method(method, ad, args, method_out)
        summaries.append(summary)

    ablation = pd.concat(summaries, axis=0, ignore_index=True)
    ablation.to_csv(os.path.join(args.out_dir, "ablation_table.csv"), index=False)
    print("\n========== Ablation table ==========")
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
