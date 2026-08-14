#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel



SAVE_DIR = "./result/luadvae"
os.makedirs(SAVE_DIR, exist_ok=True)

if torch.cuda.is_available():
    cuda_env = os.environ.get("CUDA_DEVICE", "1")
    if str(cuda_env).startswith("cuda:"):
        DEVICE = torch.device(cuda_env)
    else:
        DEVICE = torch.device(f"cuda:{cuda_env}")
else:
    DEVICE = torch.device("cpu")

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


RNA_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA.h5ad"
META_PATH = "./data/nsclc/GSE243013_NSCLC_immune_scRNA_metadata.csv"
TCR_PATH = "./data/nsclc/GSE243013_T_with_TCR_annotation.csv"
SAMPLE_META_PATH = "./data/nsclc/meta2.csv"
TCRBERT_DIR = "./pretrained/tcr-bert/"

CANCER_TYPE = "LUAD"

UNFREEZE_LAST_N = 2


EMB_D = 128
SHARED_D = 64
PRIV_D = 64


EPOCHS_RNA = 50
LR_RNA = 1e-4

EPOCHS_FUSED = 50
LR_FUSED = 1e-4
PATIENCE_FUSED = 20

BATCH_SIZE = 4096


LAMBDA_TCR = 0.3
LAMBDA_ORTH = 0.3
LAMBDA_GATE = 0.0

LAMBDA_KL = 1e-4
KL_WARMUP_EPOCHS = 10
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0

TCR_SCALE = 0.3
FUSE_SCALE = 0.3


ALPHA_WEIGHT = 0.3

ALPHA_RECON_WEIGHT = 0.5


UMAP_TCR_PRIV_WEIGHT = 0.30


RNA_RESO_L1 = 1.5
SHARED_RESO_L2 = 1.5


N_NEI = 40
UMAP_ARGS = dict(min_dist=0.3, spread=1.0, random_state=SEED)



amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
AASET = set(amino_acids)
special_tokens = ["<pad>", "<start>", "<end>", "<mask>"]
vocab = special_tokens + amino_acids
token2idx = {tok: i for i, tok in enumerate(vocab)}
pad_idx = token2idx["<pad>"]
start_idx = token2idx["<start>"]
end_idx = token2idx["<end>"]
mask_idx = token2idx["<mask>"]
vocab_size = len(vocab)
MAX_LEN = 20


def clean_aa(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return "".join(ch for ch in str(seq).upper() if ch in AASET)


def encode_seq(seq, max_len=MAX_LEN):
    s = clean_aa(seq)
    ids = [start_idx] + [token2idx.get(aa, pad_idx) for aa in s[:max_len - 2]] + [end_idx]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


def tcr_prepare_ar_batch(seqs, max_len=MAX_LEN, device=DEVICE):
    ids = [encode_seq(s, max_len) for s in seqs]
    return torch.tensor(ids, dtype=torch.long, device=device)



mode_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
    "..", "model"
))
if mode_path not in sys.path:
    sys.path.append(mode_path)

from model import GeneMLPEncoder, Cross_attention, RNADecoder, CDR3SeqDecoder


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
                    self._fmt(s),
                    add_special_tokens=True,
                    padding=False,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors=None,
                )
            encs.append(self._cache[k])

        batch = self.tok.pad(encs, padding=True, return_tensors="pt")
        toks = {k: v.to(dev) for k, v in batch.items()}

        hidden = self.backbone(**toks).last_hidden_state
        att = toks["attention_mask"].bool()
        ids = toks["input_ids"]

        keep = att & (ids != self.tok.cls_token_id) & (ids != self.tok.sep_token_id)
        hidden = hidden.masked_fill(~keep.unsqueeze(-1), 0.0)

        return self.proj(hidden), ~keep


class RNAEncoderModel(nn.Module):
    def __init__(self, gene_dim, emb_dim=EMB_D):
        super().__init__()

        self.gene_encoder = GeneMLPEncoder(
            input_dim=gene_dim,
            out_dim=emb_dim,
            width=512,
            depth=3,
            mult=4,
            dropout=0.1,
            use_attention=True,
            n_latents=64,
            n_heads=8,
            latent_self_attn=True,
            select_topk=512,
            abs_topk=True,
        )

        self.gene_norm = nn.LayerNorm(emb_dim)

        self.gene_decoder = RNADecoder(
            d_model=emb_dim,
            gene_dim=gene_dim,
            width=512,
            depth=3,
            mult=4,
            dropout=0.1,
            use_output_bias=True,
        )

    def forward(self, rna):
        return self.gene_norm(self.gene_encoder(rna))

    def recon_loss(self, z, rna_true):
        pred = self.gene_decoder(z)
        return F.mse_loss(pred, rna_true)



class VAEBottleneck(nn.Module):

    def __init__(self, in_dim: int, latent_dim: int, logvar_min: float = LOGVAR_MIN, logvar_max: float = LOGVAR_MAX):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(self.logvar_min, self.logvar_max)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu


        kl_per_cell = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, mu, logvar, kl_per_cell


class FusedModel(nn.Module):

    def __init__(
        self,
        gene_dim,
        gene_out_dim=EMB_D,
        shared_dim=SHARED_D,
        priv_dim=PRIV_D,
        unfreeze_last_n=UNFREEZE_LAST_N,
        tcr_scale=TCR_SCALE,
        fuse_scale=FUSE_SCALE,
    ):
        super().__init__()

        self.shared_dim = shared_dim
        self.priv_dim = priv_dim
        self.umap_tcr_priv_weight = UMAP_TCR_PRIV_WEIGHT


        self.gene_encoder = GeneMLPEncoder(
            input_dim=gene_dim,
            out_dim=gene_out_dim,
            width=512,
            depth=3,
            mult=4,
            dropout=0.1,
            use_attention=True,
            n_latents=64,
            n_heads=8,
            latent_self_attn=True,
            select_topk=512,
            abs_topk=True,
        )
        self.gene_norm = nn.LayerNorm(gene_out_dim)


        self.trb_encoder = TCRBertEncoder(
            TCRBERT_DIR,
            out_dim=gene_out_dim,
            unfreeze_last_n=unfreeze_last_n,
        )
        self.tra_encoder = TCRBertEncoder(
            TCRBERT_DIR,
            out_dim=gene_out_dim,
            unfreeze_last_n=0,
        )
        self.tcr_norm = nn.LayerNorm(gene_out_dim)
        self.tcr_mix = nn.Sequential(
            nn.LayerNorm(gene_out_dim * 2),
            nn.Linear(gene_out_dim * 2, gene_out_dim),
            nn.GELU(),
            nn.LayerNorm(gene_out_dim),
        )

        self.cross_attention = Cross_attention(
            gene_dim=gene_out_dim,
            cdr_dim=gene_out_dim,
            emb_dim=gene_out_dim,
            num_heads=8,
            use_self_attn=True,
            dropout=0.1,
            layers=1,
            tcr_scale=tcr_scale,
            fuse_scale=fuse_scale,
            gate_bias=-2.0,
            detach_gate_gene=True,
        )


        self.shared_vae = VAEBottleneck(gene_out_dim, shared_dim)
        self.rna_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)
        self.tcr_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)


        self.rna_all_dim = shared_dim + priv_dim
        self.gene_decoder = RNADecoder(
            d_model=self.rna_all_dim,
            gene_dim=gene_dim,
            width=512,
            depth=3,
            mult=4,
            dropout=0.1,
            use_output_bias=True,
        )

        self.tcr_all_dim = shared_dim + priv_dim
        self.trb_decoder = CDR3SeqDecoder(
            memory_dim=self.tcr_all_dim,
            d_model=self.tcr_all_dim,
            vocab_size=vocab_size,
            max_len=MAX_LEN,
            pad_idx=pad_idx,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            tie_weights=True,
        )
        self.tra_decoder = CDR3SeqDecoder(
            memory_dim=self.tcr_all_dim,
            d_model=self.tcr_all_dim,
            vocab_size=vocab_size,
            max_len=MAX_LEN,
            pad_idx=pad_idx,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            tie_weights=True,
        )

    def encode_backbone(self, rna: Tensor, tra_list: List[str], trb_list: List[str]):

        z_rna_raw = self.gene_norm(self.gene_encoder(rna))


        trb_tok_full, trb_pad_mask = self.trb_encoder.encode_tokens(trb_list)
        trb_tok = self.tcr_norm(trb_tok_full)
        valid_b = (~trb_pad_mask).float().unsqueeze(-1)
        z_trb = (trb_tok * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp_min(1.0)


        tra_tok_full, tra_pad_mask = self.tra_encoder.encode_tokens(tra_list)
        tra_tok = self.tcr_norm(tra_tok_full)
        valid_a = (~tra_pad_mask).float().unsqueeze(-1)
        z_tra = (tra_tok * valid_a).sum(dim=1) / valid_a.sum(dim=1).clamp_min(1.0)


        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))

        tcr_tok = torch.cat([trb_tok, ALPHA_WEIGHT * tra_tok], dim=1)
        pad_mask = torch.cat([trb_pad_mask, tra_pad_mask], dim=1)
        z_cross, tcr_ctx_tok, gate_score, attn = self.cross_attention(
            z_rna_raw,
            tcr_tok,
            pad_mask=pad_mask,
        )


        z_shared, shared_mu, shared_logvar, kl_shared = self.shared_vae(z_cross)
        z_rna_priv, rna_priv_mu, rna_priv_logvar, kl_rna_priv = self.rna_priv_vae(z_rna_raw)
        z_tcr_priv, tcr_priv_mu, tcr_priv_logvar, kl_tcr_priv = self.tcr_priv_vae(z_tcr_raw)

        z_rna_all = torch.cat([z_shared, z_rna_priv], dim=-1)
        z_tcr_all = torch.cat([z_shared, z_tcr_priv], dim=-1)

        if self.umap_tcr_priv_weight > 0:
            z_umap = torch.cat(
                [z_shared, z_rna_priv, self.umap_tcr_priv_weight * z_tcr_priv],
                dim=-1,
            )
        else:
            z_umap = z_rna_all

        return {
            "z_rna_raw": z_rna_raw,
            "z_tcr_raw": z_tcr_raw,
            "z_shared": z_shared,
            "z_rna_priv": z_rna_priv,
            "z_tcr_priv": z_tcr_priv,
            "z_rna_all": z_rna_all,
            "z_tcr_all": z_tcr_all,
            "z_umap": z_umap,
            "z_fused": z_umap,
            "shared_mu": shared_mu,
            "shared_logvar": shared_logvar,
            "rna_priv_mu": rna_priv_mu,
            "rna_priv_logvar": rna_priv_logvar,
            "tcr_priv_mu": tcr_priv_mu,
            "tcr_priv_logvar": tcr_priv_logvar,
            "kl_shared": kl_shared,
            "kl_rna_priv": kl_rna_priv,
            "kl_tcr_priv": kl_tcr_priv,
            "tcr_ctx_tok": tcr_ctx_tok,
            "gate_score": gate_score,
            "pad_mask": pad_mask,
            "attn": attn,
        }

    def forward(self, rna: Tensor, tra_list: List[str], trb_list: List[str]):
        return self.encode_backbone(rna, tra_list, trb_list)

    def rna_recon_loss(self, z_rna_all: Tensor, rna_true: Tensor):
        pred = self.gene_decoder(z_rna_all)
        return F.mse_loss(pred, rna_true)

    def _cdr3_ar_loss(self, decoder: nn.Module, cdr3_list: List[str], z_tcr_all: Tensor) -> Tensor:
        ids = tcr_prepare_ar_batch(cdr3_list, device=z_tcr_all.device)
        logits, target = decoder(z_tcr_all, ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=pad_idx,
        )

    def full_tcr_recon_loss(self, tra_list: List[str], trb_list: List[str], z_tcr_all: Tensor):

        l_trb = self._cdr3_ar_loss(self.trb_decoder, trb_list, z_tcr_all)

        valid_a_idx = [i for i, s in enumerate(tra_list) if len(clean_aa(s)) > 0]
        if len(valid_a_idx) > 0:
            z_a = z_tcr_all[torch.tensor(valid_a_idx, dtype=torch.long, device=z_tcr_all.device)]
            tra_valid = [tra_list[i] for i in valid_a_idx]
            l_tra = self._cdr3_ar_loss(self.tra_decoder, tra_valid, z_a)
        else:
            l_tra = z_tcr_all.sum() * 0.0

        return l_trb + ALPHA_RECON_WEIGHT * l_tra

    def vae_kl_loss(self, out: Dict[str, Tensor]) -> Tensor:
        return (
            out["kl_shared"].mean()
            + out["kl_rna_priv"].mean()
            + out["kl_tcr_priv"].mean()
        )

    def gate_loss(self, out: Dict[str, Tensor]) -> Tensor:

        return out["gate_score"].mean()

    def shared_private_orth_loss(self, z_shared: Tensor, z_priv: Tensor) -> Tensor:

        b = z_shared.size(0)
        if b < 2:
            return torch.tensor(0.0, device=z_shared.device)

        dist_s = torch.cdist(z_shared, z_shared, p=2) ** 2
        dist_p = torch.cdist(z_priv, z_priv, p=2) ** 2

        sigma_s = dist_s[dist_s > 0].mean().clamp_min(1e-4) if dist_s.sum() > 0 else 1.0
        sigma_p = dist_p[dist_p > 0].mean().clamp_min(1e-4) if dist_p.sum() > 0 else 1.0

        k_s = torch.exp(-dist_s / (2.0 * sigma_s))
        k_p = torch.exp(-dist_p / (2.0 * sigma_p))

        h = torch.eye(b, device=z_shared.device) - (1.0 / b) * torch.ones((b, b), device=z_shared.device)
        k_sc = torch.mm(h, torch.mm(k_s, h))
        return torch.sum(k_sc * k_p) / ((b - 1) ** 2)



class RNADataset(Dataset):
    def __init__(self, rna_tensor):
        self.rna = rna_tensor

    def __getitem__(self, idx):
        return self.rna[idx]

    def __len__(self):
        return len(self.rna)


class RNATCRDataset(Dataset):
    def __init__(self, rna_tensor, tra_list, trb_list):
        self.rna = rna_tensor
        self.tra = tra_list
        self.trb = trb_list

    def __getitem__(self, idx):
        return self.rna[idx], self.tra[idx], self.trb[idx]

    def __len__(self):
        return len(self.rna)


def collate_rna_tcr(batch):
    rnas, tras, trbs = zip(*batch)
    return torch.stack(rnas, 0), list(tras), list(trbs)



def load_data():

    print("[Load] NSCLC scRNA-seq h5ad ...")
    ad = sc.read_h5ad(RNA_PATH)
    ad.obs_names = ad.obs_names.astype(str).str.strip()

    print("[Load] cell annotation ...")
    meta_df = pd.read_csv(META_PATH, dtype=str)
    meta_df.columns = [c.strip() for c in meta_df.columns]

    if "cellID" not in meta_df.columns:

        first_col = meta_df.columns[0]
        meta_df = meta_df.rename(columns={first_col: "cellID"})
    meta_df["cell_id"] = meta_df["cellID"].astype(str).str.strip()

    if "sub_cell_type" not in meta_df.columns:
        raise ValueError(f"[meta] missing sub_cell_type; cols={list(meta_df.columns)}")
    meta_df["sub_cell_type"] = meta_df["sub_cell_type"].astype(str).str.strip()

    if "cancer_type" not in meta_df.columns:
        raise ValueError(f"[meta] missing cancer_type; cols={list(meta_df.columns)}")
    meta_df["cancer_type"] = meta_df["cancer_type"].astype(str).str.strip()

    print(f"[meta] {len(meta_df)} rows, columns={list(meta_df.columns)}")
    print(f"[meta] cancer_type distribution:\n{meta_df['cancer_type'].value_counts()}")

    print("[Load] TCR (T_with_TCR_annotation.csv) ...")
    tcr_df = pd.read_csv(TCR_PATH, dtype=str).fillna("")
    tcr_df.columns = [c.strip() for c in tcr_df.columns]

    for col in ["cellID", "sampleID", "TRB_cdr3"]:
        if col not in tcr_df.columns:
            raise ValueError(f"[TCR] missing {col}; cols={list(tcr_df.columns)}")

    tcr_df["cell_id"] = tcr_df["cellID"].astype(str).str.strip()
    tcr_df["sampleID"] = tcr_df["sampleID"].astype(str).str.strip()
    tcr_df["TRB_aa"] = tcr_df["TRB_cdr3"].apply(clean_aa)

    if "TRA_cdr3" in tcr_df.columns:
        tcr_df["TRA_aa"] = tcr_df["TRA_cdr3"].apply(clean_aa)
    else:
        tcr_df["TRA_aa"] = ""


    if "expansion" in tcr_df.columns:
        tcr_df["is_expanded"] = (
            tcr_df["expansion"].astype(str).str.lower().str.strip().eq("expanded").astype(int)
        )
    else:
        tcr_df["is_expanded"] = np.nan


    tcr_df = tcr_df[tcr_df["TRB_aa"].str.len() > 0].copy()
    tcr_df = tcr_df.drop_duplicates("cell_id", keep="first")
    print(f"[TCR] valid TRB entries: {len(tcr_df)}")
    if "TRA_aa" in tcr_df.columns:
        print(f"[TCR] valid TRA ratio: {(tcr_df['TRA_aa'].str.len() > 0).mean():.3f}")
    if "is_expanded" in tcr_df.columns and tcr_df["is_expanded"].notna().any():
        print(f"[TCR] expanded ratio: {tcr_df['is_expanded'].mean():.3f}")

    print("[Load] sample metadata (Pathological Response) ...")
    sample_df = pd.read_csv(SAMPLE_META_PATH, dtype=str)
    sample_df.columns = [c.strip() for c in sample_df.columns]

    if "Tumor_Sample_Barcode" not in sample_df.columns:
        raise ValueError(
            f"[sample meta] need 'Tumor_Sample_Barcode'; cols={list(sample_df.columns)}"
        )
    if "Pathological Response" not in sample_df.columns:
        raise ValueError(
            f"[sample meta] need 'Pathological Response'; cols={list(sample_df.columns)}"
        )

    sample_df["sampleID"] = sample_df["Tumor_Sample_Barcode"].astype(str).str.strip()
    sample_df["Response"] = sample_df["Pathological Response"].astype(str).str.strip()
    sample_df = sample_df[["sampleID", "Response"]].drop_duplicates("sampleID")

    print("[Response] raw distribution (patient-level):")
    print(sample_df["Response"].value_counts(dropna=False))

    obs = pd.DataFrame(index=ad.obs_names)
    obs.index.name = "cell_id"

    meta_df2 = meta_df.drop_duplicates("cell_id").set_index("cell_id")

    meta_keep_cols = [c for c in meta_df2.columns if c != "cellID"]
    obs = obs.merge(meta_df2[meta_keep_cols], left_index=True, right_index=True, how="left")

    tcr_keep_cols = ["TRA_aa", "TRB_aa", "is_expanded", "sampleID"]
    for extra_col in [
        "TRA_v_gene", "TRA_j_gene", "TRA_c_gene", "TRA_cdr3",
        "TRB_v_gene", "TRB_j_gene", "TRB_c_gene", "TRB_cdr3",
        "clonotype", "clonotype_number", "T_new_name", "expansion",
    ]:
        if extra_col in tcr_df.columns and extra_col not in tcr_keep_cols:
            tcr_keep_cols.append(extra_col)
    tcr_df2 = tcr_df.drop_duplicates("cell_id").set_index("cell_id")[tcr_keep_cols]

    obs = obs.merge(
        tcr_df2,
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_tcr"),
    )
    if obs.index.has_duplicates:
        obs = obs[~obs.index.duplicated(keep="first")]

    if "sampleID" not in obs.columns and "sampleID_tcr" in obs.columns:
        obs["sampleID"] = obs["sampleID_tcr"]
    elif "sampleID_tcr" in obs.columns:
        obs["sampleID"] = obs["sampleID"].where(obs["sampleID"].notna(), obs["sampleID_tcr"])
    if "sampleID_tcr" in obs.columns:
        obs = obs.drop(columns=["sampleID_tcr"])

    obs["sampleID"] = obs["sampleID"].astype(str)

    obs = obs.merge(
        sample_df.set_index("sampleID"),
        left_on="sampleID",
        right_index=True,
        how="left",
    )

    obs = obs.reindex(ad.obs_names)
    for col in obs.columns:
        ad.obs[col] = obs[col].values
    ad.obs["cell_id"] = ad.obs_names


    if "major_cell_type" in ad.obs.columns:
        print("\n[Diag] full sub_cell_type distribution in T/NK lineage:")
        tnk_mask_diag = ad.obs["major_cell_type"].astype(str).eq("T/NK cell")
        print(ad.obs.loc[tnk_mask_diag, "sub_cell_type"].value_counts())
    else:
        print("[Diag] WARNING: major_cell_type column not found; will rely on sub_cell_type prefix only")

    print(f"\n[Filter] CANCER_TYPE = {CANCER_TYPE}")
    if CANCER_TYPE.upper() == "ALL":
        is_cancer = pd.Series(True, index=ad.obs_names)
    elif CANCER_TYPE.upper() in {"LUAD", "LUSC"}:
        is_cancer = ad.obs["cancer_type"].astype(str).str.upper() == CANCER_TYPE.upper()
    else:
        raise ValueError(f"unknown CANCER_TYPE={CANCER_TYPE}; expected ALL/LUAD/LUSC")
    is_cancer.index = ad.obs_names

    if "sub_cell_type" not in ad.obs.columns:
        raise ValueError("missing sub_cell_type column in ad.obs after merge")

    sub_str = ad.obs["sub_cell_type"].astype(str)


    if "major_cell_type" in ad.obs.columns:
        is_tnk = ad.obs["major_cell_type"].astype(str).eq("T/NK cell")
    else:
        is_tnk = pd.Series(True, index=ad.obs_names)


    is_t = sub_str.str.startswith("CD4T") | sub_str.str.startswith("CD8T")


    has_trb = ad.obs["TRB_aa"].fillna("").astype(str).str.len() > 0


    has_response = (
            ad.obs["Response"].notna()
            & (~ad.obs["Response"].astype(str).isin(["nan", "NA", ""]))
    )


    print(f"\n[Filter] before any filter:        {ad.n_obs}")
    print(f"[Filter] after cancer ({CANCER_TYPE}): {int(is_cancer.sum())}")
    print(f"[Filter] after major=T/NK cell:    {int((is_cancer & is_tnk).sum())}")
    print(f"[Filter] after CD4T/CD8T prefix:   {int((is_cancer & is_tnk & is_t).sum())}")
    print(f"[Filter] after has_trb:            {int((is_cancer & is_tnk & is_t & has_trb).sum())}")
    print(f"[Filter] after has_response:       {int((is_cancer & is_tnk & is_t & has_trb & has_response).sum())}")

    keep = is_cancer & is_tnk & is_t & has_trb & has_response
    ad = ad[keep].copy()
    print(f"[Filter] FINAL: {ad.n_obs}")


    print("\n[Filter] sub_cell_type distribution after filter:")
    print(ad.obs["sub_cell_type"].value_counts())

    def _binarize_response(x):
        x = str(x).strip()
        if x in ("pCR", "MPR"):
            return "MPR"
        elif x == "non-MPR":
            return "non-MPR"
        else:
            return "NA"

    ad.obs["Response_binary"] = ad.obs["Response"].astype(str).map(_binarize_response)
    print("\n[Final] Response_binary distribution (cell level):")
    print(ad.obs["Response_binary"].value_counts())

    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)

    sample_max = float(np.asarray(ad.X[:200].max() if hasattr(ad.X, "max") else 0.0))
    print(f"[Preproc] sample max value: {sample_max:.3f}")

    if sample_max > 30:
        print("  -> assuming raw counts; running normalize_total + log1p")
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    else:
        print("  -> already log-normalized, skip normalize_total/log1p")

    ad.raw = ad.copy()


    tcr_ig_pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", flags=re.IGNORECASE)
    keep_genes = [g for g in ad.var_names if not tcr_ig_pattern.match(g)]
    removed = ad.n_vars - len(keep_genes)
    ad = ad[:, keep_genes].copy()
    print(f"[Filter] removed {removed} TCR/Ig genes; remaining: {ad.n_vars}")

    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.pp.scale(ad, max_value=10)

    if hasattr(ad.X, "toarray"):
        ad.X = ad.X.toarray()

    for col in ["sub_cell_type", "Response", "cancer_type"]:
        if col in ad.obs.columns:
            ad.obs[col] = (
                ad.obs[col].astype("string").fillna("NA").astype(str).astype("category")
            )

    ad.obs["sampleID"] = ad.obs["sampleID"].astype(str)
    ad.obs["TRA_aa"] = ad.obs["TRA_aa"].fillna("").astype(str)
    ad.obs["TRB_aa"] = ad.obs["TRB_aa"].fillna("").astype(str)


    vc = ad.obs["TRB_aa"].value_counts().to_dict()
    ad.obs["clone_size"] = ad.obs["TRB_aa"].map(vc).fillna(1).astype(int)

    if ad.obs["is_expanded"].isna().any():
        fill = (ad.obs["clone_size"] > 1).astype(int)
        ad.obs["is_expanded"] = ad.obs["is_expanded"].where(
            ad.obs["is_expanded"].notna(), fill
        )
    ad.obs["is_expanded"] = ad.obs["is_expanded"].fillna(0).astype(int)

    print("\n[Final] Response distribution (sample level):")
    resp_sample = ad.obs.groupby("sampleID", observed=True)["Response"].first()
    print(resp_sample.value_counts())

    print("\n[Final] Response distribution (cell level):")
    print(ad.obs["Response"].value_counts())

    print("\n[Final] sub_cell_type distribution:")
    print(ad.obs["sub_cell_type"].value_counts())

    print("\n[Final] cancer_type distribution:")
    print(ad.obs["cancer_type"].value_counts())

    print(f"\n[Final] is_expanded ratio: {ad.obs['is_expanded'].mean():.3f}")

    return ad


def train_rna_encoder(ad, epochs=EPOCHS_RNA):
    print("\n" + "=" * 60)
    print("Stage 1: Train RNA-only encoder")
    print("=" * 60)

    X = torch.tensor(ad.X, dtype=torch.float32)
    ds = RNADataset(X)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    model = RNAEncoderModel(gene_dim=ad.n_vars, emb_dim=EMB_D).to(DEVICE)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_RNA,
        weight_decay=1e-4,
    )

    for ep in range(epochs):
        model.train()
        total = 0.0

        for rna in loader:
            rna = rna.to(DEVICE)
            opt.zero_grad()

            z = model(rna)
            loss = model.recon_loss(z, rna)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
            print(f"[RNA][Epoch {ep + 1:03d}] recon_loss={total / len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "rna_encoder_stage1.pt"))
    return model


@torch.no_grad()
def extract_rna_only_features(ad, rna_model, batch_size: int = 512):

    print("[RNA-only] extracting standalone Stage-1 RNA embedding ...")
    rna_model.eval()
    X = torch.tensor(ad.X, dtype=torch.float32)
    z_list = []

    for i in range(0, X.shape[0], batch_size):
        rna_b = X[i:i + batch_size].to(DEVICE)
        z = rna_model(rna_b)
        z_list.append(z.cpu())

    z_rna_only = torch.cat(z_list, dim=0).numpy()
    print(f"[RNA-only] z_rna_only shape={z_rna_only.shape}")
    return z_rna_only


def save_rna_only_outputs(ad, rna_model):

    z_rna_only = extract_rna_only_features(ad, rna_model)
    ad_rna = ad.copy()
    ad_rna.obsm["X_rna_only_raw"] = z_rna_only

    pca_rna = PCA(n_components=min(30, z_rna_only.shape[1])).fit(z_rna_only)
    ad_rna.obsm["X_rna_only"] = pca_rna.transform(z_rna_only)

    sc.pp.neighbors(ad_rna, use_rep="X_rna_only", n_neighbors=N_NEI, key_added="rnaonly_nb")
    try:
        sc.tl.leiden(ad_rna, resolution=RNA_RESO_L1, key_added="RNAonly_cluster", neighbors_key="rnaonly_nb")
    except Exception:
        sc.tl.louvain(ad_rna, resolution=RNA_RESO_L1, key_added="RNAonly_cluster", neighbors_key="rnaonly_nb")
    ad_rna.obs["RNAonly_cluster"] = ad_rna.obs["RNAonly_cluster"].astype(str)

    sc.tl.umap(
        ad_rna,
        neighbors_key="rnaonly_nb",
        min_dist=UMAP_ARGS["min_dist"],
        spread=UMAP_ARGS["spread"],
        random_state=UMAP_ARGS["random_state"],
    )
    ad_rna.obsm["X_umap_rna_only"] = ad_rna.obsm["X_umap"].copy()

    np.save(os.path.join(SAVE_DIR, "rna_only_embedding.npy"), z_rna_only)
    pd.DataFrame(z_rna_only, index=ad.obs_names).to_csv(
        os.path.join(SAVE_DIR, "rna_only_embedding.csv")
    )

    for c in ad_rna.obs.columns:
        if pd.api.types.is_object_dtype(ad_rna.obs[c]):
            ad_rna.obs[c] = ad_rna.obs[c].astype(str)
    ad_rna.write(os.path.join(SAVE_DIR, "adata_nsclc_icb_rnaonly.h5ad"))

    print("[RNA-only] saved:")
    print(f"  - {os.path.join(SAVE_DIR, 'rna_only_embedding.npy')}")
    print(f"  - {os.path.join(SAVE_DIR, 'rna_only_embedding.csv')}")
    print(f"  - {os.path.join(SAVE_DIR, 'adata_nsclc_icb_rnaonly.h5ad')}")
    return z_rna_only, ad_rna



def train_fusion_encoder(ad, rna_model, epochs=EPOCHS_FUSED):
    print("\n" + "=" * 60)
    print("Stage 2: Train shared/private RNA-TCR fused encoder with full CDR3A/CDR3B reconstruction")
    print("=" * 60)

    X = torch.tensor(ad.X, dtype=torch.float32)
    tra = ad.obs["TRA_aa"].astype(str).tolist()
    trb = ad.obs["TRB_aa"].astype(str).tolist()

    ds = RNATCRDataset(X, tra, trb)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_rna_tcr,
    )

    model = FusedModel(
        gene_dim=ad.n_vars,
        gene_out_dim=EMB_D,
        shared_dim=SHARED_D,
        priv_dim=PRIV_D,
        unfreeze_last_n=UNFREEZE_LAST_N,
        tcr_scale=TCR_SCALE,
        fuse_scale=FUSE_SCALE,
    ).to(DEVICE)

    model.gene_encoder.load_state_dict(rna_model.gene_encoder.state_dict())
    model.gene_norm.load_state_dict(rna_model.gene_norm.state_dict())

    try:
        model.gene_decoder.load_state_dict(rna_model.gene_decoder.state_dict())
        print("[Stage 2] gene_decoder weights loaded from Stage 1.")
    except Exception as e:
        print(f"[Stage 2] warning: gene_decoder weight load failed ({e}). Using fresh init.")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FUSED,
        weight_decay=1e-4,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"[Fusion] trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M")
    print(
        f"[Fusion config] shared_dim={SHARED_D}, priv_dim={PRIV_D}, "
        f"lambda_tcr={LAMBDA_TCR}, lambda_orth={LAMBDA_ORTH}, lambda_gate={LAMBDA_GATE}, "
        f"lambda_kl={LAMBDA_KL}, kl_warmup={KL_WARMUP_EPOCHS}, "
        f"UMAP_TCR_PRIV_WEIGHT={UMAP_TCR_PRIV_WEIGHT}"
    )

    best_loss = float("inf")
    patience_cnt = 0
    best_path = os.path.join(SAVE_DIR, "fusion_encoder_stage2.pt")

    for ep in range(epochs):
        model.train()

        tot = dict(
            loss=0.0,
            rna=0.0,
            tcr=0.0,
            orth=0.0,
            kl=0.0,
            gate=0.0,
        )

        for rna, tra_list, trb_list in loader:
            rna = rna.to(DEVICE)
            opt.zero_grad()

            out = model(rna, tra_list, trb_list)


            l_rna = model.rna_recon_loss(out["z_rna_all"], rna)

            l_tcr = model.full_tcr_recon_loss(tra_list, trb_list, out["z_tcr_all"])

            l_orth = (
                model.shared_private_orth_loss(out["z_shared"], out["z_rna_priv"])
                + model.shared_private_orth_loss(out["z_shared"], out["z_tcr_priv"])
            )

            kl_weight = LAMBDA_KL * min(1.0, float(ep + 1) / float(max(1, KL_WARMUP_EPOCHS)))
            l_kl = model.vae_kl_loss(out)

            l_gate = F.relu(out["gate_score"] - 0.35).mean()

            loss = (
                l_rna
                + LAMBDA_TCR * l_tcr
                + LAMBDA_ORTH * l_orth
                + kl_weight * l_kl
                + LAMBDA_GATE * l_gate
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot["loss"] += loss.item()
            tot["rna"] += l_rna.item()
            tot["tcr"] += l_tcr.item()
            tot["orth"] += l_orth.item()
            tot["kl"] += l_kl.item()
            tot["gate"] += l_gate.item()

        n = len(loader)
        avg_loss = tot["loss"] / n

        print(
            f"[Fusion][Epoch {ep + 1:03d}] "
            f"loss={avg_loss:.4f} | "
            f"rna={tot['rna'] / n:.4f} | "
            f"tcr={tot['tcr'] / n:.4f} | "
            f"orth={tot['orth'] / n:.4f} | "
            f"kl={tot['kl'] / n:.4f} | "
            f"kl_w={kl_weight:.2e} | "
            f"gate={tot['gate'] / n:.4f}"
        )

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_FUSED:
                print(f"[Early Stop] at epoch {ep + 1} (best={best_loss:.4f})")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model


@torch.no_grad()
def extract_features(ad, fusion_model):
    print("[Embed] z_rna + shared/private fused embedding ...")
    fusion_model.eval()

    X = torch.tensor(ad.X, dtype=torch.float32)
    tra = ad.obs["TRA_aa"].astype(str).tolist()
    trb = ad.obs["TRB_aa"].astype(str).tolist()

    z_rna_list = []
    z_fused_list = []
    z_shared_list = []
    z_rna_priv_list = []
    z_tcr_priv_list = []
    z_tcr_list = []
    gate_list = []

    for i in range(0, X.shape[0], 512):
        rna_b = X[i:i + 512].to(DEVICE)
        tra_b = tra[i:i + 512]
        trb_b = trb[i:i + 512]

        out = fusion_model(rna_b, tra_b, trb_b)

        z_rna_list.append(out["z_rna_raw"].cpu())
        z_fused_list.append(out["z_umap"].cpu())
        z_shared_list.append(out["z_shared"].cpu())
        z_rna_priv_list.append(out["z_rna_priv"].cpu())
        z_tcr_priv_list.append(out["z_tcr_priv"].cpu())
        z_tcr_list.append(out["z_tcr_raw"].cpu())
        gate_list.append(out["gate_score"].cpu())

    z_rna = torch.cat(z_rna_list).numpy()
    z_fused = torch.cat(z_fused_list).numpy()
    z_shared = torch.cat(z_shared_list).numpy()
    z_rna_priv = torch.cat(z_rna_priv_list).numpy()
    z_tcr_priv = torch.cat(z_tcr_priv_list).numpy()
    z_tcr = torch.cat(z_tcr_list).numpy()
    gate_score = torch.cat(gate_list).numpy()

    norm_rna = np.linalg.norm(z_rna, ord=2, axis=1)
    norm_tcr = np.linalg.norm(z_tcr, ord=2, axis=1)
    norm_fused = np.linalg.norm(z_fused, ord=2, axis=1)
    norm_shared = np.linalg.norm(z_shared, ord=2, axis=1)
    norm_rna_priv = np.linalg.norm(z_rna_priv, ord=2, axis=1)
    norm_tcr_priv = np.linalg.norm(z_tcr_priv, ord=2, axis=1)

    diag_df = pd.DataFrame(
        {
            "fusion_gate": gate_score,
            "norm_rna": norm_rna,
            "norm_tcr": norm_tcr,
            "norm_fused": norm_fused,
            "norm_shared": norm_shared,
            "norm_rna_priv": norm_rna_priv,
            "norm_tcr_priv": norm_tcr_priv,
            "umap_tcr_priv_weight": UMAP_TCR_PRIV_WEIGHT,
        },
        index=ad.obs_names,
    )

    print(f"[Embed] z_rna shape={z_rna.shape}, z_fused/umap shape={z_fused.shape}")
    print(
        f"[Diag] gate mean={diag_df['fusion_gate'].mean():.4f} | "
        f"shared norm={diag_df['norm_shared'].mean():.4f} | "
        f"rna_priv norm={diag_df['norm_rna_priv'].mean():.4f} | "
        f"tcr_priv norm={diag_df['norm_tcr_priv'].mean():.4f} | "
        f"UMAP_TCR_PRIV_WEIGHT={UMAP_TCR_PRIV_WEIGHT}"
    )

    return z_rna, z_fused, z_tcr_priv, diag_df


def parallel_cluster(ad, z_rna, z_fused, diag_df,
                     rna_reso=RNA_RESO_L1,
                     shared_reso=SHARED_RESO_L2):
    print("" + "=" * 60)
    print("Parallel Clustering: RNA vs shared/private RNA-TCR fused embedding")
    print("=" * 60)

    ad_work = ad.copy()

    for col in diag_df.columns:
        ad_work.obs[col] = diag_df.loc[ad_work.obs_names, col].values

    pca_rna = PCA(n_components=min(30, z_rna.shape[1])).fit(z_rna)
    ad_work.obsm["X_rna"] = pca_rna.transform(z_rna)

    sc.pp.neighbors(ad_work, use_rep="X_rna", n_neighbors=N_NEI, key_added="rna_nb")

    try:
        sc.tl.leiden(ad_work, resolution=rna_reso, key_added="RNA_cluster", neighbors_key="rna_nb")
    except Exception:
        sc.tl.louvain(ad_work, resolution=rna_reso, key_added="RNA_cluster", neighbors_key="rna_nb")

    ad_work.obs["RNA_cluster"] = ad_work.obs["RNA_cluster"].astype(str)
    print(f"[L1] RNA_cluster n_clusters = {ad_work.obs['RNA_cluster'].nunique()}")

    pca_fused = PCA(n_components=min(30, z_fused.shape[1])).fit(z_fused)
    ad_work.obsm["X_final"] = pca_fused.transform(z_fused)

    sc.pp.neighbors(ad_work, use_rep="X_final", n_neighbors=N_NEI, key_added="fused_nb")

    try:
        sc.tl.leiden(ad_work, resolution=shared_reso, key_added="fine_cluster", neighbors_key="fused_nb")
    except Exception:
        sc.tl.louvain(ad_work, resolution=shared_reso, key_added="fine_cluster", neighbors_key="fused_nb")

    ad_work.obs["fine_cluster"] = ad_work.obs["fine_cluster"].astype(str)
    print(f"[L2] fine_cluster (from z_fused) n_clusters = {ad_work.obs['fine_cluster'].nunique()}")

    return ad_work


def cluster_composition_analysis(ad, group_col="Response",
                                 cluster_col="fine_cluster",
                                 min_samples_per_group=3):
    print("\n" + "=" * 60)
    print(f"Cluster composition: {cluster_col} vs {group_col}")
    print("=" * 60)

    df = ad.obs[["sampleID", cluster_col, group_col]].copy()
    df[cluster_col] = df[cluster_col].astype(str)
    df[group_col] = df[group_col].astype(str)

    comp = pd.crosstab(df["sampleID"], df[cluster_col], normalize="index")
    sample_group = df.drop_duplicates("sampleID").set_index("sampleID")[group_col]
    comp = comp.join(sample_group.rename("group"), how="left")

    comp = comp[
        comp["group"].notna()
        & (comp["group"] != "NA")
        & (comp["group"] != "nan")
    ]

    print(f"[Comp] {len(comp)} samples with valid group label")
    print(f"[Comp] group distribution:\n{comp['group'].value_counts()}")

    groups = sorted(comp["group"].unique().tolist())
    results = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            c1 = comp[comp["group"] == g1]
            c2 = comp[comp["group"] == g2]

            if len(c1) < min_samples_per_group or len(c2) < min_samples_per_group:
                continue

            for cl in comp.columns:
                if cl == "group":
                    continue

                v1 = c1[cl].values
                v2 = c2[cl].values

                try:
                    stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                except ValueError:
                    continue

                results.append(
                    {
                        "cluster": cl,
                        "group1": g1,
                        "mean1": float(v1.mean()),
                        "group2": g2,
                        "mean2": float(v2.mean()),
                        "diff": float(v1.mean() - v2.mean()),
                        "U_stat": stat,
                        "p_value": p,
                    }
                )

    res_df = pd.DataFrame(results)

    if len(res_df) == 0:
        print("[Comp] no valid comparison")
        return comp, res_df

    from statsmodels.stats.multitest import multipletests

    _, res_df["p_adj"], _, _ = multipletests(res_df["p_value"], method="fdr_bh")
    res_df = res_df.sort_values("p_value")

    print("\n[Top 15 differential clusters]")
    print(res_df.head(15).to_string(index=False))

    res_df.to_csv(
        os.path.join(SAVE_DIR, f"composition_{cluster_col}_vs_{group_col}.csv"),
        index=False,
    )
    comp.to_csv(os.path.join(SAVE_DIR, f"composition_matrix_{cluster_col}.csv"))

    sig = res_df[res_df["p_adj"] < 0.05]
    print(f"\n[Summary] {len(sig)} clusters with p_adj<0.05")

    return comp, res_df



def make_umap_from_rep(ad, use_rep, tag):
    print(f"\n[UMAP] building for '{use_rep}' -> '{tag}'")

    key_name = f"neigh_{tag}"

    sc.pp.neighbors(
        ad,
        use_rep=use_rep,
        n_neighbors=N_NEI,
        random_state=UMAP_ARGS["random_state"],
        key_added=key_name,
    )

    sc.tl.umap(
        ad,
        neighbors_key=key_name,
        min_dist=UMAP_ARGS["min_dist"],
        spread=UMAP_ARGS["spread"],
        random_state=UMAP_ARGS["random_state"],
    )

    ad.obsm[f"X_umap_{tag}"] = ad.obsm["X_umap"].copy()
    return ad


def plot_umap(ad, umap_key, color, out_png, title_prefix=""):
    ad.obsm["X_umap"] = ad.obsm[umap_key]

    sc.pl.umap(
        ad,
        color=color,
        legend_loc=("on data" if color in {"RNA_cluster", "fine_cluster"} else "right margin"),
        legend_fontsize="xx-small",
        show=False,
        title=f"{title_prefix}{color}",
    )

    plt.savefig(os.path.join(SAVE_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.close()


def make_all_umaps(ad):
    ad = make_umap_from_rep(ad, use_rep="X_rna", tag="rna")
    ad = make_umap_from_rep(ad, use_rep="X_final", tag="final")

    rna_colors = [
        "sub_cell_type",
        "RNA_cluster",
        "Response",
        "cancer_type",
        "is_expanded",
    ]

    fused_colors = [
        "sub_cell_type",
        "fine_cluster",
        "RNA_cluster",
        "Response",
        "cancer_type",
        "is_expanded",
        "fusion_gate",
    ]

    for color in rna_colors:
        if color in ad.obs.columns:
            plot_umap(
                ad,
                "X_umap_rna",
                color,
                f"umap_rna_{color}.png",
                title_prefix="[RNA UMAP] ",
            )

    for color in fused_colors:
        if color in ad.obs.columns:
            plot_umap(
                ad,
                "X_umap_final",
                color,
                f"umap_final_{color}.png",
                title_prefix="[Shared-private RNA-TCR UMAP] ",
            )

    return ad


def main():
    print(f"[Config] DEVICE={DEVICE}")
    print(f"[Config] SAVE_DIR={SAVE_DIR}")
    print(f"[Config] CANCER_TYPE={CANCER_TYPE}")
    print(f"[Config] EMB_D={EMB_D}, SHARED_D={SHARED_D}, PRIV_D={PRIV_D}")
    print(f"[Config] LAMBDA_TCR={LAMBDA_TCR}, LAMBDA_ORTH={LAMBDA_ORTH}, LAMBDA_GATE={LAMBDA_GATE}, LAMBDA_KL={LAMBDA_KL}, KL_WARMUP_EPOCHS={KL_WARMUP_EPOCHS}")
    print(f"[Config] TCR_SCALE={TCR_SCALE}, FUSE_SCALE={FUSE_SCALE}, ALPHA_WEIGHT={ALPHA_WEIGHT}, ALPHA_RECON_WEIGHT={ALPHA_RECON_WEIGHT}")
    print(f"[Config] UMAP_TCR_PRIV_WEIGHT={UMAP_TCR_PRIV_WEIGHT}")


    ad = load_data()


    rna_model = train_rna_encoder(ad, epochs=EPOCHS_RNA)

    z_rna_only, ad_rna_only = save_rna_only_outputs(ad, rna_model)


    fusion_model = train_fusion_encoder(ad, rna_model, epochs=EPOCHS_FUSED)


    z_rna_fusion, z_fused, z_tcr_priv, diag_df = extract_features(ad, fusion_model)

    ad = parallel_cluster(
        ad,
        z_rna_only,
        z_fused,
        diag_df,
        rna_reso=RNA_RESO_L1,
        shared_reso=SHARED_RESO_L2,
    )

    ad.obsm["X_rna_only_raw"] = z_rna_only
    ad.obsm["X_rna_fusion_raw"] = z_rna_fusion
    ad.obsm["X_tcr_priv_raw"] = z_tcr_priv
    np.save(os.path.join(SAVE_DIR, "tcr_priv_embedding.npy"), z_tcr_priv)


    comp_coarse, res_coarse = cluster_composition_analysis(
        ad,
        group_col="Response",
        cluster_col="RNA_cluster",
    )

    comp_fine, res_fine = cluster_composition_analysis(
        ad,
        group_col="Response",
        cluster_col="fine_cluster",
    )


    ad = make_all_umaps(ad)


    for c in ad.obs.columns:
        if pd.api.types.is_object_dtype(ad.obs[c]):
            ad.obs[c] = ad.obs[c].astype(str)

    ad.write(os.path.join(SAVE_DIR, "adata_nsclc_icb.h5ad"))

    print("\n[DONE]")
    print(f"  - CANCER_TYPE: {CANCER_TYPE}")
    print("  - Stage 1: RNA-only encoder saved separately before fusion")
    print("  - Stage 2: shared/private paired TRA/TRB RNA-TCR fusion model with 3-branch VAE bottlenecks")
    print("  - RNA reconstruction: concat(z_shared, z_rna_priv)")
    print("  - TCR reconstruction: full TRA/TRB CDR3 autoregressive reconstruction with concat(z_shared, z_tcr_priv)")
    print(
        "  - UMAP embedding: "
        "concat(z_shared, z_rna_priv, UMAP_TCR_PRIV_WEIGHT * z_tcr_priv)"
    )
    print("  - RNA_cluster  : leiden on z_rna")
    print("  - fine_cluster : leiden on z_fused / z_umap")
    print(f"  - Output dir: {SAVE_DIR}")
    print("  - CSV: composition_*.csv")
    print("  - Plots: umap_rna_*.png / umap_final_*.png")
    print("  - Deleted plots: umap_final_norm_{shared,rna_priv,tcr_priv}.png and top_clusters_{fine,coarse}_Response.png")
    print("  - h5ad: adata_nsclc_icb.h5ad")
    print("  - RNA-only h5ad: adata_nsclc_icb_rnaonly.h5ad")
    print("  - TCR-private embedding: tcr_priv_embedding.npy and ad.obsm[\"X_tcr_priv_raw\"]")


if __name__ == "__main__":
    main()