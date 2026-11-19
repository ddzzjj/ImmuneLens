#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict influenza GILGFVFTL cell binder calls in donors 1 and 2.

The original RNA-only, TCR-only and shared/private fused encoders, together
with their decoders and losses, are retained. Each mode uses the same
clone-disjoint train/validation/test split from epidata_cell_binder_v2. Mixed
label clones are retained. Encoder training, RNA feature selection, and
classifier training use training clones only. Validation and test report
clone-balanced cell AP: each clone contributes total weight one.

Run:
   python -m immunlens.training.influenza_binder_prediction --task donor1_flu_gilgf --mode all
   python -m immunlens.training.influenza_binder_prediction --task all --mode all
"""

import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/immunlens_influenza_mpl")
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import re
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 0. Args
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="all",
    choices=["rna_only", "tcr_only", "fused", "all"],
)
TASKS = ("donor1_flu_gilgf", "donor2_flu_gilgf")
parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
parser.add_argument("--dataset-dir", default=os.environ.get(
    "IMMUNLENS_EPIBIND_DATASET_DIR", "results/epidata_cell_binder_v2"))
parser.add_argument("--rna-dir",
                    default=os.environ.get("IMMUNLENS_EPIBIND_RNA_DIR", "data/10x"))
parser.add_argument("--out-dir",
                    default=os.environ.get("IMMUNLENS_EPIBIND_OUT_DIR", "results/epibind10x2"))
parser.add_argument("--export-attention-only", action="store_true",
                    help="Load an existing fused encoder and export test-cell attention without training")
parser.add_argument("--dry-run", action="store_true",
                    help="Validate influenza tasks and clone splits without fitting models")
parser.add_argument("--encoder-epochs", type=int, default=None)
parser.add_argument("--classifier-epochs", type=int, default=30)
parser.add_argument(
    "--cuda",
    type=str,
    default=os.environ.get("CUDA_DEVICE", "cuda:2"),
)
args, _ = parser.parse_known_args()

DEVICE = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Mode] {args.mode}")
print(f"[Device] {DEVICE}")


# ============================================================
# 1. Paths
# ============================================================

TCRBERT_DIR = os.environ.get("IMMUNLENS_TCRBERT_DIR", "models/tcrbert")
UNFREEZE_LAST_N = 2


# ============================================================
# 2. Hyperparameters
# ============================================================

EPOCHS_RNA = 20
LR_RNA = 1e-4

EPOCHS_TCR = 20
LR_TCR = 1e-4

EPOCHS_FUSED = 20
LR_FUSED = 1e-4

EPOCHS_SUP = 30
LR_SUP = 1e-3
WEIGHT_DECAY_SUP = 1e-4
EARLY_STOP_PATIENCE_SUP = 10
MIN_DELTA_SUP = 1e-4

BATCH_SIZE = 1024

# ---- Embedding dims (代码1) ----
EMB_D = 128
SHARED_D = 64
PRIV_D = 64

# ---- Fused loss weights ----
# 代码1 用 0.2; 抗原任务高度依赖 TCR, 这里保留 1.0 让 tcr_priv 充分承载 TCR 信息。
LAMBDA_TCR = 1.0
LAMBDA_ORTH = 0.3
LAMBDA_GATE = 0.0

# ---- VAE / KL (代码1) ----
LAMBDA_KL = 1e-4
LAMBDA_KL_PRIV_MULT = 2.0      # private 信息瓶颈更紧
KL_WARMUP_EPOCHS = 10
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0

# ---- Cross-attention TCR injection (代码1) ----
TCR_SCALE = 0.2
FUSE_SCALE = 0.2
GATE_BIAS = 0.0                # 代码1: shared 要重构 TCR, 起始中性不按死

# ---- α/β 链融合 (代码1) ----
ALPHA_WEIGHT = 0.3
ALPHA_RECON_WEIGHT = 0.3

# ---- 下游 fused embedding 是否包含 tcr_priv ----
# 抗原结合依赖 TCR, 默认 True (与代码1做分析时默认丢弃 tcr_priv 的诉求不同)。
TCR_PRIV_IN_EMBED = True

MAX_LEN = 20


# ============================================================
# 3. TCR tokenization (自回归全 CDR3 重构用, 与代码1一致)
# ============================================================

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


def clean_cdr3(seq: str) -> str:
    return "".join([c for c in str(seq).upper() if c in AASET])


def encode_seq(seq: str, max_len: int = MAX_LEN) -> List[int]:
    s = clean_cdr3(seq)
    ids = [start_idx] + [token2idx.get(aa, pad_idx) for aa in s[: max_len - 2]] + [end_idx]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


def tcr_prepare_ar_batch(seqs: List[str], max_len: int = MAX_LEN, device: torch.device = DEVICE) -> Tensor:
    ids = [encode_seq(s, max_len) for s in seqs]
    return torch.tensor(ids, dtype=torch.long, device=device)


# ============================================================
# 4. Import RNA / fusion / CDR3 modules
# ============================================================

from immunlens.models.model import GeneMLPEncoder, Cross_attention, RNADecoder, CDR3SeqDecoder
from immunlens.preprocessing.influenza_binder_data import (
    read_fixed_task, prepare_task_rna)


# ============================================================
# 5. VAE bottleneck (代码1 风格: 返回 z, mu, logvar, kl_per_cell)
# ============================================================

class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int,
                 logvar_min: float = LOGVAR_MIN, logvar_max: float = LOGVAR_MAX):
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


def kl_warmup_weight(epoch: int) -> float:
    if KL_WARMUP_EPOCHS <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(KL_WARMUP_EPOCHS))


# ============================================================
# 6. TCR-BERT token encoder (代码1 风格)
# ============================================================

class TCRBertEncoder(nn.Module):
    def __init__(self, model_dir: str, out_dim: int, unfreeze_last_n: int = 0):
        super().__init__()
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(model_dir)

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
        s = clean_cdr3(seq)
        return " ".join(list(s)) if s else "X"

    def encode_tokens(self, seqs: List[str]) -> Tuple[Tensor, Tensor]:
        dev = next(self.parameters()).device
        encs = []

        for s in seqs:
            k = f"AA::{clean_cdr3(s)}"
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

        keep = att.clone()
        if self.tok.cls_token_id is not None:
            keep = keep & (ids != self.tok.cls_token_id)
        if self.tok.sep_token_id is not None:
            keep = keep & (ids != self.tok.sep_token_id)

        valid_seq = torch.tensor(
            [len(clean_cdr3(s)) > 0 for s in seqs], dtype=torch.bool, device=dev
        )
        keep = keep & valid_seq[:, None]

        hidden = hidden.masked_fill(~keep.unsqueeze(-1), 0.0)
        return self.proj(hidden), ~keep


# ============================================================
# 7. Datasets
# ============================================================

class UnsupervisedDataset(Dataset):
    def __init__(self, rna_tensor: Tensor, cdr3a_list: List[str], cdr3b_list: List[str]):
        assert rna_tensor.shape[0] == len(cdr3a_list) == len(cdr3b_list)
        self.rna = rna_tensor
        self.cdr3a = cdr3a_list
        self.cdr3b = cdr3b_list

    def __getitem__(self, idx):
        return self.rna[idx], self.cdr3a[idx], self.cdr3b[idx]

    def __len__(self):
        return len(self.rna)


def collate_unsup_batch(batch):
    rnas, cdr3a, cdr3b = zip(*batch)
    return torch.stack(rnas, 0), list(cdr3a), list(cdr3b)


class TCRPairDataset(Dataset):
    def __init__(self, cdr3a_list: List[str], cdr3b_list: List[str]):
        assert len(cdr3a_list) == len(cdr3b_list)
        self.cdr3a = cdr3a_list
        self.cdr3b = cdr3b_list

    def __len__(self):
        return len(self.cdr3a)

    def __getitem__(self, idx):
        return self.cdr3a[idx], self.cdr3b[idx]


def collate_tcr_pair(batch):
    a, b = zip(*batch)
    return list(a), list(b)


# ============================================================
# 8. RNA-only model (代码1 Stage-1 风格: VAE 全重构, 不 mask)
# ============================================================

class RNAOnlyModel(nn.Module):
    def __init__(self, gene_dim: int, latent_dim: int = EMB_D):
        super().__init__()
        self.latent_dim = int(latent_dim)

        self.gene_encoder = GeneMLPEncoder(
            input_dim=gene_dim,
            out_dim=EMB_D,
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
        self.gene_norm = nn.LayerNorm(EMB_D)

        self.z_mu = nn.Linear(EMB_D, latent_dim)
        self.z_logvar = nn.Linear(EMB_D, latent_dim)

        self.gene_decoder = RNADecoder(
            d_model=latent_dim,
            gene_dim=gene_dim,
            width=512,
            depth=3,
            mult=4,
            dropout=0.1,
            use_output_bias=True,
        )

    def encode(self, rna: Tensor):
        h = self.gene_norm(self.gene_encoder(rna))
        mu = self.z_mu(h)
        logvar = self.z_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)

        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        kl_per_cell = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, mu, logvar, kl_per_cell

    def forward(self, rna: Tensor):
        z, _, _, _ = self.encode(rna)
        return z

    def recon_loss(self, z: Tensor, rna_true: Tensor):
        pred = self.gene_decoder(z)
        return F.mse_loss(pred, rna_true)

    @staticmethod
    def kl_loss(kl_per_cell: Tensor):
        return kl_per_cell.mean()


# ============================================================
# 9. TCR-only model (代码1 风格: β 锚定 + 低权 α, 自回归全 CDR3 重构)
# ============================================================

class TCROnlyModel(nn.Module):
    def __init__(self, latent_dim: int = EMB_D):
        super().__init__()
        self.latent_dim = int(latent_dim)

        self.trb_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=EMB_D, unfreeze_last_n=UNFREEZE_LAST_N)
        self.tra_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=EMB_D, unfreeze_last_n=0)
        self.tcr_norm = nn.LayerNorm(EMB_D)
        self.tcr_mix = nn.Sequential(
            nn.LayerNorm(EMB_D * 2),
            nn.Linear(EMB_D * 2, EMB_D),
            nn.GELU(),
            nn.LayerNorm(EMB_D),
        )

        self.tcr_vae = VAEBottleneck(EMB_D, latent_dim)

        self.trb_decoder = CDR3SeqDecoder(
            memory_dim=latent_dim, d_model=latent_dim, vocab_size=vocab_size,
            max_len=MAX_LEN, pad_idx=pad_idx, n_layers=2, n_heads=4, d_ff=512,
            dropout=0.1, tie_weights=True,
        )
        self.tra_decoder = CDR3SeqDecoder(
            memory_dim=latent_dim, d_model=latent_dim, vocab_size=vocab_size,
            max_len=MAX_LEN, pad_idx=pad_idx, n_layers=2, n_heads=4, d_ff=512,
            dropout=0.1, tie_weights=True,
        )

    def _pool(self, tok: Tensor, pad_mask: Tensor) -> Tensor:
        valid = (~pad_mask).float().unsqueeze(-1)
        return (tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def encode_backbone(self, cdr3a_list: List[str], cdr3b_list: List[str]):
        trb_tok, trb_pad = self.trb_encoder.encode_tokens(cdr3b_list)
        trb_tok = self.tcr_norm(trb_tok)
        z_trb = self._pool(trb_tok, trb_pad)

        tra_tok, tra_pad = self.tra_encoder.encode_tokens(cdr3a_list)
        tra_tok = self.tcr_norm(tra_tok)
        z_tra = self._pool(tra_tok, tra_pad)

        h_tcr = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))
        z, mu, logvar, kl = self.tcr_vae(h_tcr)
        return {"z": z, "mu": mu, "logvar": logvar, "kl": kl}

    def encode(self, cdr3a_list: List[str], cdr3b_list: List[str], return_dict: bool = False):
        out = self.encode_backbone(cdr3a_list, cdr3b_list)
        return out if return_dict else out["z"]

    def _cdr3_ce_loss(self, decoder: nn.Module, seqs: List[str], z: Tensor) -> Tensor:
        ids = tcr_prepare_ar_batch(seqs, device=z.device)
        logits, target = decoder(z, ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_idx
        )

    def tcr_recon_loss(self, cdr3a_list: List[str], cdr3b_list: List[str], z: Tensor) -> Tensor:
        l_trb = self._cdr3_ce_loss(self.trb_decoder, cdr3b_list, z)
        valid_a_idx = [i for i, s in enumerate(cdr3a_list) if len(clean_cdr3(s)) > 0]
        if len(valid_a_idx) > 0:
            idx = torch.tensor(valid_a_idx, dtype=torch.long, device=z.device)
            l_tra = self._cdr3_ce_loss(self.tra_decoder, [cdr3a_list[i] for i in valid_a_idx], z[idx])
        else:
            l_tra = z.sum() * 0.0
        return l_trb + ALPHA_RECON_WEIGHT * l_tra

    @staticmethod
    def kl_loss(out: Dict[str, Tensor]):
        return out["kl"].mean()


# ============================================================
# 10. Fused model (代码1: shared/private 四解码器互补相加)
# ============================================================

class FusedModel(nn.Module):
    def __init__(self, gene_dim: int,
                 gene_out_dim: int = EMB_D, shared_dim: int = SHARED_D, priv_dim: int = PRIV_D,
                 unfreeze_last_n: int = UNFREEZE_LAST_N,
                 tcr_scale: float = TCR_SCALE, fuse_scale: float = FUSE_SCALE):
        super().__init__()
        self.shared_dim = shared_dim
        self.priv_dim = priv_dim

        # RNA encoder
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

        # TCR encoders: β 锚定(可解冻) + 低权 α(冻结)
        self.trb_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=gene_out_dim, unfreeze_last_n=unfreeze_last_n)
        self.tra_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=gene_out_dim, unfreeze_last_n=0)
        self.tcr_norm = nn.LayerNorm(gene_out_dim)
        self.tcr_mix = nn.Sequential(
            nn.LayerNorm(gene_out_dim * 2),
            nn.Linear(gene_out_dim * 2, gene_out_dim),
            nn.GELU(),
            nn.LayerNorm(gene_out_dim),
        )

        # RNA-guided TCR token cross-attention -> shared backbone
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
            gate_bias=GATE_BIAS,
            detach_gate_gene=True,
        )

        # 三路 VAE bottleneck
        self.shared_vae = VAEBottleneck(gene_out_dim, shared_dim)
        self.rna_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)
        self.tcr_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)

        # -------- RNA: shared / priv 各一套 RNADecoder, 输出空间相加 --------
        self.rna_dec_shared = RNADecoder(
            d_model=shared_dim, gene_dim=gene_dim, width=512, depth=3, mult=4,
            dropout=0.1, use_output_bias=True,
        )
        self.rna_dec_priv = RNADecoder(
            d_model=priv_dim, gene_dim=gene_dim, width=512, depth=3, mult=4,
            dropout=0.1, use_output_bias=False,
        )

        # -------- TCR: α/β 各两套(shared / priv)自回归 CDR3 解码器, logits 相加 --------
        tcr_dec_dmodel = shared_dim + priv_dim

        def _make_cdr3_decoder(memory_dim):
            return CDR3SeqDecoder(
                memory_dim=memory_dim, d_model=tcr_dec_dmodel, vocab_size=vocab_size,
                max_len=MAX_LEN, pad_idx=pad_idx, n_layers=2, n_heads=4, d_ff=512,
                dropout=0.1, tie_weights=True,
            )

        self.trb_dec_shared = _make_cdr3_decoder(shared_dim)
        self.trb_dec_priv = _make_cdr3_decoder(priv_dim)
        self.tra_dec_shared = _make_cdr3_decoder(shared_dim)
        self.tra_dec_priv = _make_cdr3_decoder(priv_dim)

    def _pool(self, tok: Tensor, pad_mask: Tensor) -> Tensor:
        valid = (~pad_mask).float().unsqueeze(-1)
        return (tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def encode_backbone(self, rna: Tensor, cdr3a_list: List[str], cdr3b_list: List[str]):
        # RNA latent
        z_rna_raw = self.gene_norm(self.gene_encoder(rna))

        # β / α token latent
        trb_tok, trb_pad = self.trb_encoder.encode_tokens(cdr3b_list)
        trb_tok = self.tcr_norm(trb_tok)
        z_trb = self._pool(trb_tok, trb_pad)

        tra_tok, tra_pad = self.tra_encoder.encode_tokens(cdr3a_list)
        tra_tok = self.tcr_norm(tra_tok)
        z_tra = self._pool(tra_tok, tra_pad)

        # TCR-private cell-level rep
        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))

        # shared 拿到 paired token 上下文: β 锚 + 低权 α
        tcr_tok = torch.cat([trb_tok, ALPHA_WEIGHT * tra_tok], dim=1)
        pad_mask = torch.cat([trb_pad, tra_pad], dim=1)
        z_cross, tcr_ctx_tok, gate_score, attn = self.cross_attention(
            z_rna_raw, tcr_tok, pad_mask=pad_mask,
        )

        # VAE bottlenecks
        z_shared, shared_mu, shared_logvar, kl_shared = self.shared_vae(z_cross)
        z_rna_priv, rna_priv_mu, rna_priv_logvar, kl_rna_priv = self.rna_priv_vae(z_rna_raw)
        z_tcr_priv, tcr_priv_mu, tcr_priv_logvar, kl_tcr_priv = self.tcr_priv_vae(z_tcr_raw)

        return {
            "z_rna_raw": z_rna_raw,
            "z_tcr_raw": z_tcr_raw,
            "z_shared": z_shared,
            "z_rna_priv": z_rna_priv,
            "z_tcr_priv": z_tcr_priv,
            "shared_mu": shared_mu,
            "rna_priv_mu": rna_priv_mu,
            "tcr_priv_mu": tcr_priv_mu,
            "kl_shared": kl_shared,
            "kl_rna_priv": kl_rna_priv,
            "kl_tcr_priv": kl_tcr_priv,
            "gate_score": gate_score,
            "attn": attn,
            "trb_pad_mask": trb_pad,
            "tra_pad_mask": tra_pad,
        }

    def forward(self, rna: Tensor, cdr3a_list: List[str], cdr3b_list: List[str]):
        return self.encode_backbone(rna, cdr3a_list, cdr3b_list)

    # -------- RNA 重构: 输出空间相加 --------
    def rna_recon_loss(self, z_shared: Tensor, z_rna_priv: Tensor, rna_true: Tensor):
        pred = self.rna_dec_shared(z_shared) + self.rna_dec_priv(z_rna_priv)
        return F.mse_loss(pred, rna_true)

    # -------- TCR 重构: logit 空间相加 --------
    def _cdr3_logitadd_loss(self, dec_shared, dec_priv, cdr3_list, z_shared, z_tcr_priv) -> Tensor:
        ids = tcr_prepare_ar_batch(cdr3_list, device=z_shared.device)
        logits_s, target = dec_shared(z_shared, ids)
        logits_p, _ = dec_priv(z_tcr_priv, ids)
        logits = logits_s + logits_p
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_idx
        )

    def full_tcr_recon_loss(self, cdr3a_list, cdr3b_list, z_shared, z_tcr_priv):
        l_trb = self._cdr3_logitadd_loss(
            self.trb_dec_shared, self.trb_dec_priv, cdr3b_list, z_shared, z_tcr_priv
        )
        valid_a_idx = [i for i, s in enumerate(cdr3a_list) if len(clean_cdr3(s)) > 0]
        if len(valid_a_idx) > 0:
            idx = torch.tensor(valid_a_idx, dtype=torch.long, device=z_shared.device)
            l_tra = self._cdr3_logitadd_loss(
                self.tra_dec_shared, self.tra_dec_priv,
                [cdr3a_list[i] for i in valid_a_idx], z_shared[idx], z_tcr_priv[idx],
            )
        else:
            l_tra = z_shared.sum() * 0.0
        return l_trb + ALPHA_RECON_WEIGHT * l_tra

    # -------- KL: private 分支加权更紧 --------
    def vae_kl_loss(self, out: Dict[str, Tensor], priv_mult: float = LAMBDA_KL_PRIV_MULT) -> Tensor:
        return (
            out["kl_shared"].mean()
            + priv_mult * out["kl_rna_priv"].mean()
            + priv_mult * out["kl_tcr_priv"].mean()
        )

    def gate_loss(self, out: Dict[str, Tensor]) -> Tensor:
        return out["gate_score"].mean()

    # -------- HSIC 正交 --------
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


# ============================================================
# 11. Training functions (全重构, 无 mask; 代码1 风格)
# ============================================================

def train_rna_only(model: RNAOnlyModel, loader: DataLoader, optimizer, epoch: int):
    model.train()
    kl_weight = LAMBDA_KL * kl_warmup_weight(epoch)
    total = total_recon = total_kl = 0.0

    for rna, _, _ in loader:
        rna = rna.to(DEVICE)
        optimizer.zero_grad()

        z, mu, logvar, kl_per_cell = model.encode(rna)
        loss_recon = model.recon_loss(z, rna)
        loss_kl = model.kl_loss(kl_per_cell)
        loss = loss_recon + kl_weight * loss_kl

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        total_recon += loss_recon.item()
        total_kl += loss_kl.item()

    n = len(loader)
    print(
        f"[RNA-only][Epoch {epoch+1:03d}] "
        f"loss={total/n:.4f} | recon={total_recon/n:.4f} | KL={total_kl/n:.4f} | kl_w={kl_weight:.2e}"
    )


def train_tcr_only(model: TCROnlyModel, loader: DataLoader, optimizer, epoch: int):
    model.train()
    kl_weight = LAMBDA_KL * kl_warmup_weight(epoch)
    total = total_recon = total_kl = 0.0

    for cdr3a, cdr3b in loader:
        optimizer.zero_grad()

        out = model.encode(cdr3a, cdr3b, return_dict=True)
        z = out["z"]
        loss_recon = model.tcr_recon_loss(cdr3a, cdr3b, z)
        loss_kl = model.kl_loss(out)
        loss = loss_recon + kl_weight * loss_kl

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        total_recon += loss_recon.item()
        total_kl += loss_kl.item()

    n = len(loader)
    print(
        f"[TCR-only][Epoch {epoch+1:03d}] "
        f"loss={total/n:.4f} | recon={total_recon/n:.4f} | KL={total_kl/n:.4f} | kl_w={kl_weight:.2e}"
    )


def train_fused(model: FusedModel, loader: DataLoader, optimizer, epoch: int):
    model.train()
    kl_weight = LAMBDA_KL * kl_warmup_weight(epoch)
    tot = dict(loss=0.0, rna=0.0, tcr=0.0, orth=0.0, kl=0.0, gate=0.0)

    for rna, cdr3a, cdr3b in loader:
        rna = rna.to(DEVICE)
        optimizer.zero_grad()

        out = model(rna, cdr3a, cdr3b)

        l_rna = model.rna_recon_loss(out["z_shared"], out["z_rna_priv"], rna)
        l_tcr = model.full_tcr_recon_loss(cdr3a, cdr3b, out["z_shared"], out["z_tcr_priv"])
        l_orth = (
            model.shared_private_orth_loss(out["z_shared"], out["z_rna_priv"])
            + model.shared_private_orth_loss(out["z_shared"], out["z_tcr_priv"])
        )
        l_kl = model.vae_kl_loss(out, priv_mult=LAMBDA_KL_PRIV_MULT)
        l_gate = model.gate_loss(out)

        loss = (
            l_rna
            + LAMBDA_TCR * l_tcr
            + LAMBDA_ORTH * l_orth
            + kl_weight * l_kl
            + LAMBDA_GATE * l_gate
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot["loss"] += loss.item()
        tot["rna"] += l_rna.item()
        tot["tcr"] += l_tcr.item()
        tot["orth"] += l_orth.item()
        tot["kl"] += l_kl.item()
        tot["gate"] += l_gate.item()

    n = len(loader)
    print(
        f"[Fused][Epoch {epoch+1:03d}] "
        f"loss={tot['loss']/n:.4f} | rna={tot['rna']/n:.4f} | tcr={tot['tcr']/n:.4f} | "
        f"orth={tot['orth']/n:.4f} | kl={tot['kl']/n:.4f} | kl_w={kl_weight:.2e} | gate={tot['gate']/n:.4f}"
    )


# ============================================================
# 12. Embedding computation
# ============================================================

@torch.no_grad()
def compute_rna_embedding(model: RNAOnlyModel, X_rna: np.ndarray, batch_size: int = 512):
    model.eval()
    X = torch.tensor(X_rna, dtype=torch.float32)
    zs = []
    for i in range(0, X.shape[0], batch_size):
        z, _, _, _ = model.encode(X[i:i + batch_size].to(DEVICE))
        zs.append(z.cpu())
    z_all = F.normalize(torch.cat(zs, dim=0), dim=-1)
    return z_all.numpy()


@torch.no_grad()
def compute_tcr_embedding(model: TCROnlyModel, cdr3a_list: List[str], cdr3b_list: List[str],
                          batch_size: int = 512):
    model.eval()
    zs = []
    for i in range(0, len(cdr3b_list), batch_size):
        z = model.encode(cdr3a_list[i:i + batch_size], cdr3b_list[i:i + batch_size])
        zs.append(z.cpu())
    z_all = F.normalize(torch.cat(zs, dim=0), dim=-1)
    return z_all.numpy()


@torch.no_grad()
def compute_fused_eval_embedding(model: FusedModel, X_rna: np.ndarray,
                                 cdr3a_list: List[str], cdr3b_list: List[str],
                                 batch_size: int = 512):
    """
    抗原预测 cell embedding = normalize(concat(norm(shared), norm(rna_priv)[, norm(tcr_priv)]))
    默认包含 tcr_priv (TCR 是抗原结合的关键信号)。
    """
    model.eval()
    X = torch.tensor(X_rna, dtype=torch.float32)
    z_list = []
    for i in range(0, X.shape[0], batch_size):
        out = model.encode_backbone(
            X[i:i + batch_size].to(DEVICE),
            cdr3a_list[i:i + batch_size],
            cdr3b_list[i:i + batch_size],
        )
        parts = [F.normalize(out["shared_mu"], dim=-1), F.normalize(out["rna_priv_mu"], dim=-1)]
        if TCR_PRIV_IN_EMBED:
            parts.append(F.normalize(out["tcr_priv_mu"], dim=-1))
        z_eval = F.normalize(torch.cat(parts, dim=-1), dim=-1)
        z_list.append(z_eval.cpu())
    return torch.cat(z_list, dim=0).numpy()


@torch.no_grad()
def save_fused_gate_for_donor(model: FusedModel, X_rna: np.ndarray,
                              cdr3a_list: List[str], cdr3b_list: List[str],
                              barcodes: List[str], out_path: str, batch_size: int = 512):
    model.eval()
    X = torch.tensor(X_rna, dtype=torch.float32)
    gate_list, batch_sizes = [], []
    for i in range(0, X.shape[0], batch_size):
        out = model.encode_backbone(
            X[i:i + batch_size].to(DEVICE),
            cdr3a_list[i:i + batch_size],
            cdr3b_list[i:i + batch_size],
        )
        gate_list.append(out["gate_score"].detach().cpu().numpy())
        batch_sizes.append(out["gate_score"].shape[0])

    if len(gate_list) == 0:
        return

    gate_all = np.concatenate(gate_list, axis=0)
    np.savez_compressed(
        out_path,
        gate_score=gate_all,
        batch_sizes=np.asarray(batch_sizes, dtype=np.int32),
        barcodes=np.array(barcodes, dtype=object),
        cdr3a=np.array(cdr3a_list, dtype=object),
        cdr3b=np.array(cdr3b_list, dtype=object),
    )
    print(f"[Gate] saved: {out_path}")


@torch.no_grad()
def save_fused_test_attention(model: FusedModel, cells: pd.DataFrame,
                              x: np.ndarray, out_dir: str, batch_size: int = 128):
    """Save RNA-to-TCR attention at actual CDR3 residue positions for test cells."""
    test_indices = np.flatnonzero(cells.split.to_numpy() == "test")
    test = cells.iloc[test_indices].reset_index(drop=True)
    predictions = pd.read_csv(os.path.join(out_dir, "test_cell_predictions.csv"))
    lookup = predictions.set_index("barcode")
    if not lookup.index.is_unique or not test.barcode.isin(lookup.index).all():
        raise ValueError("Test cell predictions do not match the test cells")
    score = lookup.loc[test.barcode, "score_positive"].to_numpy(dtype=np.float32)
    pred_label = lookup.loc[test.barcode, "pred_label"].to_numpy(dtype=np.int8)

    model.eval()
    beta_rows, alpha_rows, gate_rows = [], [], []
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        batch = test.iloc[start:stop]
        out = model.encode_backbone(
            torch.from_numpy(x[test_indices[start:stop]]).to(DEVICE),
            batch.cdr3a.astype(str).tolist(), batch.cdr3b.astype(str).tolist(),
        )
        attention = out["attn"][1]
        if attention is None:
            raise RuntimeError("Fused cross-attention did not return RNA-to-TCR weights")
        attention = attention.squeeze(1).detach().cpu().numpy()
        beta_valid = (~out["trb_pad_mask"]).cpu().numpy()
        alpha_valid = (~out["tra_pad_mask"]).cpu().numpy()
        beta_width = beta_valid.shape[1]
        if attention.shape != (len(batch), beta_width + alpha_valid.shape[1]):
            raise ValueError(f"Unexpected attention shape: {attention.shape}")
        for row, (beta_seq, alpha_seq) in enumerate(zip(batch.cdr3b, batch.cdr3a)):
            beta = attention[row, :beta_width][beta_valid[row]].astype(np.float32)
            alpha = attention[row, beta_width:][alpha_valid[row]].astype(np.float32)
            if len(beta) != len(clean_cdr3(beta_seq)) or len(alpha) != len(clean_cdr3(alpha_seq)):
                raise ValueError("TCR-BERT tokens do not align one-to-one with CDR3 residues")
            beta_rows.append(beta)
            alpha_rows.append(alpha)
        gate_rows.append(out["gate_score"].detach().cpu().numpy().reshape(-1))

    def pad_rows(rows):
        width = max(map(len, rows), default=0)
        padded = np.full((len(rows), width), np.nan, dtype=np.float32)
        for row, values in enumerate(rows):
            padded[row, :len(values)] = values
        return padded

    out_path = os.path.join(out_dir, "test_fused_attention.npz")
    np.savez_compressed(
        out_path,
        att_trb_residue=pad_rows(beta_rows),
        att_tra_residue=pad_rows(alpha_rows),
        trb_length=np.asarray([len(row) for row in beta_rows], dtype=np.int16),
        tra_length=np.asarray([len(row) for row in alpha_rows], dtype=np.int16),
        gate_score=np.concatenate(gate_rows).astype(np.float32),
        barcode=np.asarray(test.barcode.astype(str).tolist(), dtype=str),
        clone_id=np.asarray(test.clone_id.astype(str).tolist(), dtype=str),
        cdr3a=np.asarray(test.cdr3a.astype(str).tolist(), dtype=str),
        cdr3b=np.asarray(test.cdr3b.astype(str).tolist(), dtype=str),
        label=test.label.to_numpy(dtype=np.int8),
        score_positive=score,
        pred_label=pred_label,
        attention_definition=np.asarray("RNA query to concatenated beta and alpha CDR3 tokens; heads averaged"),
    )
    print(f"[Attention] saved {len(test)} test cells: {out_path}", flush=True)
    return out_path


# ============================================================
# 15. Fixed-antigen, clone-disjoint experiment
# ============================================================





def fit_encoder(mode: str, cells: pd.DataFrame, x: np.ndarray, out_dir: str):
    train_mask = cells.split.to_numpy() == "train"
    train = cells.loc[train_mask]
    if mode == "rna_only":
        data = UnsupervisedDataset(torch.from_numpy(x[train_mask]),
                                   [""] * len(train), [""] * len(train))
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_unsup_batch, num_workers=0)
        encoder = RNAOnlyModel(gene_dim=x.shape[1], latent_dim=EMB_D).to(DEVICE)
        opt = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                lr=LR_RNA, weight_decay=1e-4)
        epochs = EPOCHS_RNA if args.encoder_epochs is None else args.encoder_epochs
        for epoch in range(epochs):
            train_rna_only(encoder, loader, opt, epoch)
    elif mode == "tcr_only":
        unique = train[["clone_id", "cdr3a", "cdr3b"]].drop_duplicates("clone_id")
        data = TCRPairDataset(unique.cdr3a.tolist(), unique.cdr3b.tolist())
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_tcr_pair, num_workers=0)
        encoder = TCROnlyModel(latent_dim=EMB_D).to(DEVICE)
        opt = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                lr=LR_TCR, weight_decay=1e-4)
        epochs = EPOCHS_TCR if args.encoder_epochs is None else args.encoder_epochs
        for epoch in range(epochs):
            train_tcr_only(encoder, loader, opt, epoch)
    elif mode == "fused":
        data = UnsupervisedDataset(torch.from_numpy(x[train_mask]),
                                   train.cdr3a.tolist(), train.cdr3b.tolist())
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_unsup_batch, num_workers=0)
        encoder = FusedModel(gene_dim=x.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                lr=LR_FUSED, weight_decay=1e-4)
        epochs = EPOCHS_FUSED if args.encoder_epochs is None else args.encoder_epochs
        for epoch in range(epochs):
            train_fused(encoder, loader, opt, epoch)
    else:
        raise ValueError(mode)
    torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))
    return encoder


def embed_cells(mode: str, encoder, cells: pd.DataFrame, x: np.ndarray):
    if mode == "rna_only":
        return compute_rna_embedding(encoder, x)
    if mode == "tcr_only":
        return compute_tcr_embedding(encoder, cells.cdr3a.tolist(), cells.cdr3b.tolist())
    return compute_fused_eval_embedding(encoder, x, cells.cdr3a.tolist(), cells.cdr3b.tolist())


class FixedAntigenClassifier(nn.Module):
    def __init__(self, cell_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(cell_dim), nn.Linear(cell_dim, 256),
                                 nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, z):
        return self.mlp(z).squeeze(-1)


def score_cells(model, z: np.ndarray, batch_size: int = 4096):
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(0, len(z), batch_size):
            values.append(model(torch.from_numpy(z[i:i + batch_size]).to(DEVICE)).sigmoid().cpu().numpy())
    return np.concatenate(values)


def clone_balanced_weights(cells: pd.DataFrame) -> np.ndarray:
    """Give each clone total weight one while retaining its cell-level labels."""
    sizes = cells.groupby("clone_id").clone_id.transform("size").to_numpy()
    return (1.0 / sizes).astype(np.float32)


def choose_threshold(y: np.ndarray, score: np.ndarray, weights: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y, score, sample_weight=weights)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)]) if len(thresholds) else 0.5


def cell_metrics(cells: pd.DataFrame, score: np.ndarray, threshold: float):
    y = cells.label.to_numpy(dtype=int)
    weights = clone_balanced_weights(cells)
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights)
    return dict(n_cells=int(len(y)), n_clones=int(cells.clone_id.nunique()),
                n_positive_cells=int(y.sum()),
                ap_baseline=float(np.average(y, weights=weights)),
                ap=float(average_precision_score(y, score, sample_weight=weights)),
                auroc=float(roc_auc_score(y, score, sample_weight=weights)),
                cell_ap_baseline=float(y.mean()),
                cell_ap=float(average_precision_score(y, score)),
                cell_auroc=float(roc_auc_score(y, score)),
                threshold=float(threshold),
                positive_f1=float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
                macro_f1=float(f1_score(y, pred, average="macro", sample_weight=weights,
                                        zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred, sample_weight=weights)),
                positive_precision=float(precision_score(y, pred, sample_weight=weights,
                                                         zero_division=0)),
                positive_recall=float(recall_score(y, pred, sample_weight=weights,
                                                   zero_division=0)),
                specificity=float(cm[0, 0] / cm[0].sum()),
                confusion_matrix_cells=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def fit_classifier(z: np.ndarray, cells: pd.DataFrame, out_dir: str):
    train_ids = np.flatnonzero(cells.split.to_numpy() == "train")
    val_ids = np.flatnonzero(cells.split.to_numpy() == "val")
    test_ids = np.flatnonzero(cells.split.to_numpy() == "test")
    labels = cells.label.to_numpy(dtype=int)
    model = FixedAntigenClassifier(z.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_SUP, weight_decay=WEIGHT_DECAY_SUP)
    train_cells = cells.iloc[train_ids]
    weights = clone_balanced_weights(train_cells)
    weighted_positive = float(weights @ labels[train_ids])
    weighted_negative = float(weights.sum() - weighted_positive)
    pos_weight = torch.tensor(np.sqrt(weighted_negative / weighted_positive),
                              dtype=torch.float32, device=DEVICE)
    weights /= weights.mean()
    data = TensorDataset(torch.from_numpy(z[train_ids]),
                         torch.from_numpy(labels[train_ids].astype(np.float32)),
                         torch.from_numpy(weights))
    loader = DataLoader(data, batch_size=512, shuffle=True, num_workers=0)
    best_ap, stale, best_epoch = -1.0, 0, 0
    for epoch in range(args.classifier_epochs):
        model.train()
        for features, target, sample_weight in loader:
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            sample_weight = sample_weight.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(features), target, reduction="none", pos_weight=pos_weight)
            loss = (loss * sample_weight).mean()
            loss.backward()
            opt.step()
        val_score = score_cells(model, z[val_ids])
        val_ap = float(average_precision_score(
            labels[val_ids], val_score,
            sample_weight=clone_balanced_weights(cells.iloc[val_ids])))
        print(f"[Classifier] epoch={epoch+1} val_clone_balanced_cell_AP={val_ap:.4f}", flush=True)
        if val_ap > best_ap + MIN_DELTA_SUP:
            best_ap, best_epoch, stale = val_ap, epoch + 1, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "classifier.pt"))
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE_SUP:
                break
    model.load_state_dict(torch.load(os.path.join(out_dir, "classifier.pt"),
                                     map_location=DEVICE, weights_only=True))
    val_score = score_cells(model, z[val_ids])
    threshold = choose_threshold(labels[val_ids], val_score,
                                 clone_balanced_weights(cells.iloc[val_ids]))
    test_score = score_cells(model, z[test_ids])
    for split, ids, score in (("val", val_ids, val_score), ("test", test_ids, test_score)):
        output = cells.iloc[ids].copy()
        output["score_positive"] = score
        output["pred_label"] = (score >= threshold).astype(int)
        output.to_csv(os.path.join(out_dir, f"{split}_cell_predictions.csv"), index=False)
    return dict(best_epoch=best_epoch,
                validation=cell_metrics(cells.iloc[val_ids], val_score, threshold),
                test=cell_metrics(cells.iloc[test_ids], test_score, threshold),
                training_cells_by_class=np.bincount(labels[train_ids], minlength=2).tolist(),
                training_clones=int(train_cells.clone_id.nunique()),
                training_positive_weight=float(weighted_positive),
                training_negative_weight=float(weighted_negative))


def main():
    if args.classifier_epochs < 1:
        raise ValueError("Classifier epochs must be positive")
    task_names = TASKS if args.task == "all" else (args.task,)
    mode_names = ("rna_only", "tcr_only", "fused") if args.mode == "all" else (args.mode,)
    if args.export_attention_only and args.mode not in {"all", "fused"}:
        raise ValueError("--export-attention-only requires --mode fused or --mode all")
    summary = []
    for task in task_names:
        cells, clones, manifest = read_fixed_task(task, args.dataset_dir)
        task_dir = os.path.join(args.out_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        print(f"[Task] {task}: {manifest['donor']} / {manifest['antigen']}; "
              f"{len(clones)} clones, {len(cells)} cells, "
              f"{int(cells.label.sum())} positive cells", flush=True)
        if args.dry_run:
            for split in ("train", "val", "test"):
                part = cells.loc[cells.split.eq(split)]
                print(f"  {split}: {len(part)} cells, {int(part.label.sum())} positive, "
                      f"{part.clone_id.nunique()} clones", flush=True)
            continue
        x = prepare_task_rna(cells, task_dir, args.rna_dir,
                             reuse_preprocessing=args.export_attention_only)
        if args.export_attention_only:
            out_dir = os.path.join(task_dir, "fused")
            encoder = FusedModel(gene_dim=x.shape[1]).to(DEVICE)
            encoder.load_state_dict(torch.load(os.path.join(out_dir, "encoder.pt"),
                                               map_location=DEVICE, weights_only=True))
            save_fused_test_attention(encoder, cells, x, out_dir)
            del encoder, x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        for mode in mode_names:
            out_dir = os.path.join(task_dir, mode)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[Mode] {mode}", flush=True)
            encoder = fit_encoder(mode, cells, x, out_dir)
            z_cell = embed_cells(mode, encoder, cells, x)
            report = fit_classifier(z_cell, cells, out_dir)
            if mode == "fused":
                save_fused_test_attention(encoder, cells, x, out_dir)
            report.update(task=task, donor=manifest["donor"], antigen=manifest["antigen"],
                          mode=mode, n_cells=len(cells), n_clones=len(clones),
                          note="One target pMHC per donor; train-only encoder and RNA preprocessing; "
                               "clone-disjoint splits; cell labels retained within mixed clones; "
                               "cell predictions weighted so each clone contributes one total vote; "
                               "validation clone-balanced cell AP selects "
                               "checkpoint and validation positive F1 selects threshold.")
            with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
                json.dump(report, handle, indent=2)
            with open(os.path.join(out_dir, "config.json"), "w") as handle:
                json.dump(vars(args), handle, indent=2)
            summary.append(dict(task=task, donor=manifest["donor"], antigen=manifest["antigen"],
                                mode=mode, **{f"test_{k}": v for k, v in report["test"].items()
                                              if k in {"ap", "auroc", "positive_f1", "macro_f1",
                                                       "balanced_accuracy", "positive_recall", "positive_precision"}}))
            print(f"[Result] {task} {mode}: {report['test']}", flush=True)
            del encoder, z_cell
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del x
    if not args.export_attention_only and not args.dry_run:
        pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "comparison.csv"), index=False)
        print(f"[Summary] {os.path.join(args.out_dir, 'comparison.csv')}", flush=True)


if __name__ == "__main__":
    main()
