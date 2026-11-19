
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsupervised RNA/TCR training on the Zhang vaccine ECCITE-seq dataset.

Stage 1 reconstructs RNA. Stage 2 initializes its RNA encoder and decoder
from Stage 1, then trains a shared/private RNA-TCR model using RNA and paired
TRA/TRB reconstruction. An independent TCR-only reconstruction control is
optional. Dextramer labels annotate outputs; they never enter any training loss.

The optional Stage 2 regularizer uses paired-chain TCRdist3 distances computed
from CDR3 and V/J calls. It ranks mutual nearest clones ahead of clones in the
farthest TCRdist quartile and regularizes representation variance. It requires
``pip install tcrdist3`` in the training environment. Antigen and dextramer
labels are never used to build comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import typing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/virus_unsup_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/virus_unsup_mpl")

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from immunlens.preprocessing.antigen_binding_data import (
    clean_aa, read_expression, read_paired_cells)
from immunlens.models.model import (
    FiLMBlock, GEGLU2, ResidualMLPBlock2, PerceiverAggregator,
    GeneMLPEncoder, RNADecoder, _FFN, _CrossBlock, Cross_attention, CDR3SeqDecoder,
)

DEVICE = torch.device("cpu")
TCRBERT_DIR = os.environ.get("IMMUNLENS_TCRBERT_DIR", "models/tcrbert")

UNFREEZE_LAST_N = 2
EMB_D = 128
SHARED_D = 64
PRIV_D = 64
LR_RNA = 1e-4
LR_TCR = 1e-4
LR_FUSED = 1e-4
PATIENCE_FUSED = 10
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
UMAP_TCR_PRIV_WEIGHT = 0.1

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
AASET = set(amino_acids)
special_tokens = ["<pad>", "<start>", "<end>", "<mask>"]
vocab = special_tokens + amino_acids
token2idx = {token: i for i, token in enumerate(vocab)}
pad_idx = token2idx["<pad>"]
start_idx = token2idx["<start>"]
end_idx = token2idx["<end>"]
vocab_size = len(vocab)
MAX_LEN = 20




def clean_cdr3(seq: str) -> str:
    return clean_aa(seq)


def encode_seq(seq, max_len=MAX_LEN):
    aa = clean_aa(seq)
    ids = [start_idx] + [token2idx.get(x, pad_idx) for x in aa[:max_len - 2]] + [end_idx]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


def tcr_prepare_ar_batch(seqs, max_len=MAX_LEN, device=None):
    if device is None:
        device = DEVICE
    ids = [encode_seq(seq, max_len) for seq in seqs]
    return torch.tensor(ids, dtype=torch.long, device=device)


# Shared RNA encoder, attention and decoders live in immunlens.models.model.
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


# ================== Stage 1: RNA-only Model ==================
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


# ================== VAE bottleneck ==================
class VAEBottleneck(nn.Module):
    """
    Generic VAE bottleneck for one latent branch.

    training: z = mu + eps * std
    eval    : z = mu, deterministic for feature extraction / UMAP
    """
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

        # per-cell KL, then caller decides how to aggregate
        kl_per_cell = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, mu, logvar, kl_per_cell


# ================== Stage 2: Shared / Private RNA-TCR Fused Model ==================
class FusedModel(nn.Module):
    """
    Shared/private RNA-TCR fused model with full TRA/TRB CDR3 reconstruction.

      z_shared   : RNA-TCR shared space from RNA-guided TCR token cross-attention
      z_rna_priv : RNA private space
      z_tcr_priv : TCR private space

    RNA reconstruction:
        concat(z_shared, z_rna_priv)

    TCR reconstruction:
        concat(z_shared, z_tcr_priv) -> full autoregressive reconstruction of
        TRB CDR3 and TRA CDR3 only. V/D/J gene reconstruction is intentionally
        not used because the current input setting does not rely on VDJ genes.

    UMAP embedding:
        concat(z_shared, z_rna_priv, UMAP_TCR_PRIV_WEIGHT * z_tcr_priv)
    """

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

        # TCR encoders: beta remains the anchor; alpha provides paired-chain context.
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

        # RNA-guided TCR token correction into the shared backbone.
        # Gate is uncapped sigmoid inside Cross_attention; use LAMBDA_GATE in loss.
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

        # Three probabilistic bottlenecks.
        self.shared_vae = VAEBottleneck(gene_out_dim, shared_dim)
        self.rna_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)
        self.tcr_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)

        # RNA decoder: shared + RNA private.
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

        # Full CDR3 decoders: shared + TCR private.
        # Separate decoders let alpha and beta chains keep chain-specific grammar.
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
        # RNA latent
        z_rna_raw = self.gene_norm(self.gene_encoder(rna))

        # Beta-chain token latent
        trb_tok_full, trb_pad_mask = self.trb_encoder.encode_tokens(trb_list)
        trb_tok = self.tcr_norm(trb_tok_full)
        valid_b = (~trb_pad_mask).float().unsqueeze(-1)
        z_trb = (trb_tok * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp_min(1.0)

        # Alpha-chain token latent
        tra_tok_full, tra_pad_mask = self.tra_encoder.encode_tokens(tra_list)
        tra_tok = self.tcr_norm(tra_tok_full)
        valid_a = (~tra_pad_mask).float().unsqueeze(-1)
        z_tra = (tra_tok * valid_a).sum(dim=1) / valid_a.sum(dim=1).clamp_min(1.0)

        # Paired-chain TCR cell-level representation for TCR-private space
        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))

        # Shared space gets paired-token context: beta anchor + low-weight alpha tokens.
        tcr_tok = torch.cat([trb_tok, ALPHA_WEIGHT * tra_tok], dim=1)
        pad_mask = torch.cat([trb_pad_mask, tra_pad_mask], dim=1)
        z_cross, tcr_ctx_tok, gate_score, attn = self.cross_attention(
            z_rna_raw,
            tcr_tok,
            pad_mask=pad_mask,
        )

        # VAE bottlenecks: stochastic in training, deterministic mu in eval.
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
        """
        Full alpha/beta CDR3 reconstruction loss.
        No V/J gene reconstruction is used.
        """
        l_trb = self._cdr3_ar_loss(self.trb_decoder, trb_list, z_tcr_all)

        # Compute TRA loss only for rows with a non-empty alpha CDR3.
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
        # Light penalty; gate remains unconstrained sigmoid [0,1].
        return out["gate_score"].mean()

    def shared_private_orth_loss(self, z_shared: Tensor, z_priv: Tensor) -> Tensor:
        """HSIC-based shared/private disentanglement loss."""
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


# ================== Dataset ==================
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

def _tcrdist_chain(label: str, prefix: str):
    """Take the first complete chain call from a clonotype_label half."""
    for call in label.split(";"):
        fields = call.split("|")
        if (len(fields) >= 3 and fields[1].startswith(prefix + "V")
                and fields[2].startswith(prefix + "J")):
            cdr3 = clean_aa(fields[0])
            if len(cdr3) >= 5:
                v = fields[1] if "*" in fields[1] else fields[1] + "*01"
                j = fields[2] if "*" in fields[2] else fields[2] + "*01"
                return cdr3, v, j
    return None


def _tcrdist_record(label: str):
    if "_" not in label:
        return None
    alpha, beta = label.split("_", 1)
    a, b = _tcrdist_chain(alpha, "TRA"), _tcrdist_chain(beta, "TRB")
    if a is None or b is None:
        return None
    return dict(cdr3_a_aa=a[0], v_a_gene=a[1], j_a_gene=a[2],
                cdr3_b_aa=b[0], v_b_gene=b[1], j_b_gene=b[2])


def build_tcrdist_graph(train_obs: pd.DataFrame, neighbors: int,
                        distance_quantile: float, max_clones: int, seed: int = 2026):
    """Build conservative, mutual paired-chain TCRdist edges between clones."""
    try:
        from tcrdist.repertoire import TCRrep
    except ImportError as exc:
        raise RuntimeError(
            "TCRdist regularization requires tcrdist3 in this Python environment; "
            "install with: pip install tcrdist3"
        ) from exc

    clone_ids = train_obs["clone_id"].astype(str).to_numpy()
    all_ids, inverse = np.unique(clone_ids, return_inverse=True)
    chosen = np.arange(len(all_ids))
    if len(chosen) > max_clones:
        chosen = np.sort(np.random.default_rng(seed).choice(
            chosen, size=max_clones, replace=False))
        print(f"[TCRdist] sampling {max_clones:,} of {len(all_ids):,} "
              "donor-specific clones for the distance graph")
    unique_ids = all_ids[chosen]
    clone_to_cells = [np.flatnonzero(inverse == i) for i in chosen]
    records = []
    for clone_id, positions in zip(unique_ids, clone_to_cells):
        row = train_obs.iloc[int(positions[0])]
        record = _tcrdist_record(str(row["clonotype_label"]))
        if record is None:
            continue
        if (record["cdr3_a_aa"] != str(row["cdr3a"])
                or record["cdr3_b_aa"] != str(row["cdr3b"])):
            continue  # Do not pair a V/J call with a different selected chain.
        # TCRrep reserves "clone_id" and replaces it during deduplication.
        records.append({"source_clone_id": clone_id, "count": 1, **record})
    if len(records) < 3:
        raise ValueError("Fewer than three paired clones have usable CDR3 and V/J calls")

    tcr = TCRrep(cell_df=pd.DataFrame(records), organism="human",
                 chains=["alpha", "beta"],
                 db_file="alphabeta_gammadelta_db.tsv")
    retained = tcr.clone_df["source_clone_id"].astype(str).tolist()
    if len(retained) < 3:
        raise ValueError("TCRdist3 recognized fewer than three paired clones")
    alpha = np.asarray(tcr.pw_alpha, dtype=np.float32)
    beta = np.asarray(tcr.pw_beta, dtype=np.float32)
    if alpha.shape != beta.shape or alpha.shape != (len(retained), len(retained)):
        raise RuntimeError("Unexpected TCRdist3 chain-distance matrix shapes")
    distance = alpha + beta  # The same TCRdist units are used on both chains.
    if not np.isfinite(distance).all():
        raise RuntimeError("TCRdist3 produced a non-finite paired-chain distance")

    # Repeated paired receptors, even across donors, are excluded: this loss is
    # intended to connect distinct sequence solutions, not identical TCRs.
    keys = list(zip(tcr.clone_df.cdr3_a_aa, tcr.clone_df.v_a_gene,
                    tcr.clone_df.cdr3_b_aa, tcr.clone_df.v_b_gene))
    n = len(retained)
    for i in range(n):
        distance[i, i] = np.inf
        for j in range(i + 1, n):
            if keys[i] == keys[j]:
                distance[i, j] = distance[j, i] = np.inf
    finite = distance[np.triu_indices(n, 1)]
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No distinct paired TCR sequences are available for TCRdist")
    cutoff = float(np.quantile(finite, distance_quantile))
    nearest = np.argsort(distance, axis=1)[:, :neighbors]
    nearest_sets = [set(row.tolist()) for row in nearest]
    id_to_index = {clone_id: i for i, clone_id in enumerate(unique_ids)}
    if not set(retained).issubset(id_to_index):
        raise RuntimeError("TCRdist3 returned source clone IDs absent from training data")
    # A far receptor is only a relative TCRdist reference, not a known
    # different-virus cell. Use the farthest quartile for each anchor so every
    # recognized clone has candidates without consulting antigen labels.
    far_candidates = [np.empty(0, dtype=np.int32) for _ in unique_ids]
    for i, clone_id in enumerate(retained):
        finite_indices = np.flatnonzero(np.isfinite(distance[i]))
        if len(finite_indices):
            far_cutoff = np.quantile(distance[i, finite_indices], 0.75)
            far_indices = finite_indices[distance[i, finite_indices] >= far_cutoff]
            far_candidates[id_to_index[clone_id]] = np.asarray(
                [id_to_index[retained[j]] for j in far_indices], dtype=np.int32
            )
    edges, edge_distances = [], []
    for i in range(n):
        for j in nearest[i]:
            j = int(j)
            d = float(distance[i, j])
            if i < j and i in nearest_sets[j] and np.isfinite(d) and d <= cutoff:
                weight = float(np.exp(-d / max(cutoff, 1.0)))
                edges.append((id_to_index[retained[i]], id_to_index[retained[j]], weight))
                edge_distances.append(d)
    if not edges:
        raise ValueError(
            "No mutual TCRdist neighbor pairs passed the distance cutoff; "
            "increase --tcrdist_neighbors or --tcrdist_quantile"
        )
    if any(not np.any(far_candidates[a] != b) for a, b, _ in edges):
        raise ValueError("A TCRdist positive anchor has no distinct far-distance reference")
    print(f"[TCRdist] input clones={len(unique_ids)}, usable V/J={len(records)}, "
          f"recognized={n}, mutual edges={len(edges)}, cutoff={cutoff:.2f}")
    return {"edges": edges, "edge_distances": edge_distances,
            "clone_to_cells": clone_to_cells, "clone_ids": unique_ids,
            "far_candidates": far_candidates,
            "cutoff": cutoff, "recognized_clones": n}






@torch.no_grad()
def extract_fused(model, x, cdr3a, cdr3b, batch_size, device):
    """Return the deterministic fusion embedding without L2 scaling."""
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        out = model.encode_backbone(
            torch.from_numpy(x[start:stop]).to(device),
            cdr3a[start:stop], cdr3b[start:stop],
        )
        chunks.append(out["z_umap"].cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def extract_rna_only(model, x, batch_size, device):
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        rna = torch.from_numpy(x[start:start + batch_size]).to(device)
        chunks.append(model(rna).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def extract_tcr_only(model, cdr3a, cdr3b, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(cdr3b), batch_size):
        z = model.encode(cdr3a[start:start + batch_size],
                         cdr3b[start:start + batch_size])
        chunks.append(z.cpu())
    return F.normalize(torch.cat(chunks, dim=0), dim=-1).numpy()


def train_tcr_only(cdr3a, cdr3b, epochs, batch_size, device, seed):
    """Train the independent paired-chain TCR reconstruction control."""
    pairs = sorted(set(zip(cdr3a, cdr3b)))
    dataset = TCRPairDataset([a for a, _ in pairs], [b for _, b in pairs])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=0, collate_fn=collate_tcr_pair)
    model = TCROnlyModel().to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LR_TCR, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        kl_weight = LAMBDA_KL * min(1.0, (epoch + 1) / max(1, KL_WARMUP_EPOCHS))
        for tra, trb in loader:
            optimizer.zero_grad()
            out = model.encode(tra, trb, return_dict=True)
            loss = model.tcr_recon_loss(tra, trb, out["z"]) + kl_weight * model.kl_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"[TCR-only][Epoch {epoch + 1:03d}] loss={total / len(loader):.4f}")
    return model


def training_loader(dataset, clone_ids, batch_size, sampling, seed, collate_fn=None):
    """Use equal expected weight per clone when requested."""
    generator = torch.Generator().manual_seed(seed)
    if sampling == "clone_balanced":
        counts = pd.Series(clone_ids).value_counts()
        weights = torch.as_tensor(
            [1.0 / counts[clone_id] for clone_id in clone_ids], dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True, generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=0, collate_fn=collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      generator=generator, num_workers=0, collate_fn=collate_fn)


def train_rna(x, clone_ids, sampling, seed, epochs, batch_size, device, checkpoint):
    """Stage 1 RNA reconstruction with zero loader workers."""
    model = RNAEncoderModel(gene_dim=x.shape[1], emb_dim=EMB_D).to(device)
    loader = training_loader(RNADataset(torch.from_numpy(x)), clone_ids,
                             batch_size, sampling, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_RNA, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for rna in loader:
            rna = rna.to(device)
            optimizer.zero_grad()
            loss = model.recon_loss(model(rna), rna)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"[RNA][Epoch {epoch + 1:03d}] recon_loss={total / len(loader):.4f}")
    torch.save(model.state_dict(), checkpoint)
    return model


def tcrdist_auxiliary_loss(model, x, cdr3a, cdr3b, graph, rng,
                           pairs_per_step, device, variance_floor,
                           variance_weight):
    """Rank TCRdist neighbors ahead of far clones in the exported fused space."""
    edges = graph["edges"]
    cells = graph["clone_to_cells"]
    far_candidates = graph["far_candidates"]
    chosen = rng.integers(len(edges), size=pairs_per_step)
    left, right, references, weights = [], [], [], []
    for edge_index in chosen:
        a, b, weight = edges[int(edge_index)]
        far_clone = int(rng.choice(far_candidates[a][far_candidates[a] != b]))
        left.append(int(rng.choice(cells[a])))
        right.append(int(rng.choice(cells[b])))
        references.append(int(rng.choice(cells[far_clone])))
        weights.append(weight)
    # Uniform clone samples make the variance estimate cover the training
    # repertoire instead of only the selected positive-pair neighborhoods.
    background = [int(rng.choice(cells[int(clone)]))
                  for clone in rng.integers(len(cells), size=pairs_per_step)]
    positions = left + right + references + background
    latent = model.encode_backbone(
        torch.from_numpy(x[positions]).to(device),
        [cdr3a[i] for i in positions], [cdr3b[i] for i in positions],
    )
    # VAE samples inject noise in train mode. The exported embedding uses mu,
    # so use the same deterministic representation and Euclidean geometry here.
    pieces = [latent["shared_mu"], latent["rna_priv_mu"]]
    if model.umap_tcr_priv_weight > 0:
        pieces.append(model.umap_tcr_priv_weight * latent["tcr_priv_mu"])
    fused = torch.cat(pieces, dim=-1)
    anchor = fused[:pairs_per_step]
    positive = fused[pairs_per_step:2 * pairs_per_step]
    reference = fused[2 * pairs_per_step:3 * pairs_per_step]
    positive_distance = torch.linalg.vector_norm(anchor - positive, dim=-1)
    reference_distance = torch.linalg.vector_norm(anchor - reference, dim=-1)
    # Normalizing by a detached scale makes the margin insensitive to overall
    # embedding magnitude. A far TCR is only a ranking reference, not a known
    # different-virus negative: it is never forced beyond a fixed distance.
    scale = reference_distance.detach().mean().clamp_min(1e-3)
    relative_gap = (reference_distance - positive_distance) / scale
    weights_t = torch.as_tensor(weights, dtype=fused.dtype, device=fused.device)
    ranking = (F.relu(0.2 - relative_gap) * weights_t).sum() / weights_t.sum()
    # Angular variance remains a separate guard against all receptors pointing
    # in the same direction; the random clone samples make it less biased.
    normalized = F.normalize(fused, dim=-1)
    variance = F.relu(variance_floor - normalized.std(dim=0, unbiased=False)).mean()
    return (ranking + variance_weight * variance,
            ranking.detach(), variance.detach(), relative_gap.detach().mean())


def train_fusion(x, cdr3a, cdr3b, clone_ids, sampling, seed,
                 rna_model, epochs, batch_size, device, checkpoint,
                 tcrdist_graph=None, tcrdist_weight=0.0,
                 tcrdist_pairs_per_step=16, tcrdist_warmup_epochs=5,
                 tcrdist_variance_floor=0.05,
                 tcrdist_variance_weight=1.0):
    """Stage 2 shared/private fusion with zero loader workers."""
    model = FusedModel(gene_dim=x.shape[1]).to(device)
    model.gene_encoder.load_state_dict(rna_model.gene_encoder.state_dict())
    model.gene_norm.load_state_dict(rna_model.gene_norm.state_dict())
    model.gene_decoder.load_state_dict(rna_model.gene_decoder.state_dict())
    loader = training_loader(
        RNATCRDataset(torch.from_numpy(x), cdr3a, cdr3b),
        clone_ids, batch_size, sampling, seed,
        collate_fn=collate_rna_tcr,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR_FUSED, weight_decay=1e-4,
    )
    best_loss = float("inf")
    stale = 0
    pair_rng = np.random.default_rng(seed + 173)
    for epoch in range(epochs):
        model.train()
        totals = {k: 0.0 for k in ("loss", "rna", "tcr", "orth", "kl", "gate",
                                    "tcrdist", "tcrdist_rank", "tcrdist_var",
                                    "tcrdist_gap")}
        kl_weight = LAMBDA_KL * min(
            1.0, (epoch + 1) / max(1, KL_WARMUP_EPOCHS)
        )
        pair_weight = tcrdist_weight * min(
            1.0, (epoch + 1) / max(1, tcrdist_warmup_epochs)
        )
        for rna, tra, trb in loader:
            rna = rna.to(device)
            optimizer.zero_grad()
            out = model(rna, tra, trb)
            l_rna = model.rna_recon_loss(out["z_rna_all"], rna)
            l_tcr = model.full_tcr_recon_loss(tra, trb, out["z_tcr_all"])
            l_orth = (model.shared_private_orth_loss(out["z_shared"], out["z_rna_priv"])
                      + model.shared_private_orth_loss(out["z_shared"], out["z_tcr_priv"]))
            l_kl = model.vae_kl_loss(out)
            l_gate = F.relu(out["gate_score"] - 0.35).mean()
            loss = (l_rna + LAMBDA_TCR * l_tcr
                    + LAMBDA_ORTH * l_orth + kl_weight * l_kl
                    + LAMBDA_GATE * l_gate)
            if tcrdist_graph is not None and pair_weight > 0:
                l_pair, l_rank, l_variance, l_gap = tcrdist_auxiliary_loss(
                    model, x, cdr3a, cdr3b, tcrdist_graph, pair_rng,
                    tcrdist_pairs_per_step, device, tcrdist_variance_floor,
                    tcrdist_variance_weight,
                )
                loss = loss + pair_weight * l_pair
                totals["tcrdist"] += l_pair.item()
                totals["tcrdist_rank"] += l_rank.item()
                totals["tcrdist_var"] += l_variance.item()
                totals["tcrdist_gap"] += l_gap.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in zip(totals, (loss, l_rna, l_tcr, l_orth, l_kl, l_gate)):
                totals[name] += value.item()
        average = totals["loss"] / len(loader)
        print(f"[Fusion][Epoch {epoch + 1:03d}] loss={average:.4f} | "
              f"rna={totals['rna'] / len(loader):.4f} | "
              f"tcr={totals['tcr'] / len(loader):.4f} | "
              f"orth={totals['orth'] / len(loader):.4f} | "
              f"kl={totals['kl'] / len(loader):.4f} | kl_w={kl_weight:.2e} | "
              f"tcrdist={totals['tcrdist'] / len(loader):.4f} | "
              f"rank={totals['tcrdist_rank'] / len(loader):.4f} | "
              f"gap={totals['tcrdist_gap'] / len(loader):.4f} | "
              f"var={totals['tcrdist_var'] / len(loader):.4f} | "
              f"pair_w={pair_weight:.3f}")
        if average < best_loss - 1e-4:
            best_loss = average
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE_FUSED:
                print(f"[Fusion] early stop at epoch {epoch + 1}")
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model


def main():
    global DEVICE, TCRBERT_DIR, UMAP_TCR_PRIV_WEIGHT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.environ.get(
        "IMMUNLENS_ECCITE_DATA", "data/eccite/PBMC_vaccine_ECCITE_TCR.h5ad"))
    p.add_argument("--tcrbert_dir",
                   default=os.environ.get("IMMUNLENS_TCRBERT_DIR", "models/tcrbert"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--methods", default="rna_only,tcr_only,fused")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min_genes", type=int, default=200)
    p.add_argument("--training_scope", choices=("all_paired",), default="all_paired",
                   help="train all paired cells without using dextramer labels")
    p.add_argument("--max_train_cells", type=int, default=0,
                   help="randomly subsample training cells; 0 uses all eligible cells")
    p.add_argument("--train_sampling", choices=("clone_balanced", "cell"),
                   default="clone_balanced",
                   help="sample equal expected examples per donor-specific clone or per cell")
    p.add_argument("--max_embed_cells", type=int, default=0,
                   help="randomly subsample exported cells; 0 embeds all eligible cells")
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--rna_epochs", type=int, default=20,
                   help="Stage 1 RNA reconstruction epochs")
    p.add_argument("--fused_epochs", type=int, default=20,
                   help="Stage 2 fusion epochs")
    p.add_argument("--tcr_private_weight", type=float, default=0.6,
                   help="weight of the TCR-private branch in the f"
                        "used embedding")
    p.add_argument("--tcrdist_weight", type=float, default=0.0,
                   help="optional Stage 2 TCRdist neighbor-loss weight; 0 disables it")
    p.add_argument("--tcrdist_neighbors", type=int, default=3,
                   help="mutual nearest receptors eligible as positive pairs")
    p.add_argument("--tcrdist_quantile", type=float, default=0.05,
                   help="maximum paired distance as a quantile of distinct-clone distances")
    p.add_argument("--tcrdist_pairs_per_step", type=int, default=16,
                   help="different-clone neighbor pairs sampled per fusion step")
    p.add_argument("--tcrdist_warmup_epochs", type=int, default=5)
    p.add_argument("--tcrdist_max_clones", type=int, default=5000,
                   help="upper limit for dense paired-chain TCRdist computation")
    p.add_argument("--tcrdist_variance_floor", type=float, default=0.05)
    p.add_argument("--tcrdist_variance_weight", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embed_batch_size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.rna_epochs < 1 or args.fused_epochs < 1 or args.pretrain_epochs < 1:
        p.error("all epoch counts must be positive")
    if args.tcr_private_weight < 0:
        p.error("--tcr_private_weight must be nonnegative")
    if (args.tcrdist_weight < 0 or args.tcrdist_neighbors < 1
            or not 0 < args.tcrdist_quantile <= 1
            or args.tcrdist_pairs_per_step < 2
            or args.tcrdist_warmup_epochs < 1
            or args.tcrdist_max_clones < 3
            or args.tcrdist_variance_floor < 0
            or args.tcrdist_variance_weight < 0):
        p.error("invalid TCRdist settings; weights/floor must be nonnegative, "
                "counts positive, and quantile in (0, 1]")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use a new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    DEVICE = device
    TCRBERT_DIR = args.tcrbert_dir
    UMAP_TCR_PRIV_WEIGHT = args.tcr_private_weight

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods or any(m not in {"rna_only", "tcr_only", "fused"} for m in methods):
        raise ValueError("methods must be a comma-separated subset of rna_only,tcr_only,fused")

    rows, cdr3a, cdr3b, obs, matrix_shape = read_paired_cells(args.data, args.min_genes)
    if args.max_embed_cells > 0 and len(rows) > args.max_embed_cells:
        chosen = np.sort(np.random.default_rng(args.seed).choice(len(rows), args.max_embed_cells, replace=False))
        rows = rows[chosen]
        cdr3a = [cdr3a[i] for i in chosen]
        cdr3b = [cdr3b[i] for i in chosen]
        obs = obs.iloc[chosen].reset_index(drop=True)
    train_n = len(rows) if args.max_train_cells <= 0 else min(args.max_train_cells, len(rows))
    train_pos = np.sort(np.random.default_rng(args.seed).choice(len(rows), train_n, replace=False))
    print(f"[Data] eligible cells={len(rows)} (from {matrix_shape[0]} total); unsupervised train cells={train_n}")
    print("[Labels] Dextramer calls are annotation only and excluded from every training loss")

    x, gene_names = read_expression(args.data, rows.tolist())
    mean = x[train_pos].mean(axis=0)
    std = np.maximum(x[train_pos].std(axis=0), 1e-3)
    x = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    train_x = x[train_pos]
    train_a = [cdr3a[i] for i in train_pos]
    train_b = [cdr3b[i] for i in train_pos]
    train_clone_ids = obs.iloc[train_pos].clone_id.astype(str).tolist()
    print(f"[Sampling] {args.train_sampling}; {len(set(train_clone_ids)):,} "
          "donor-specific clones in training")

    tcrdist_graph = None
    if "fused" in methods and args.tcrdist_weight > 0:
        # Build edges only from receptor sequences and V/J genes. Antigen
        # and dextramer annotations never influence the graph.
        tcrdist_graph = build_tcrdist_graph(
            obs.iloc[train_pos].reset_index(drop=True),
            args.tcrdist_neighbors, args.tcrdist_quantile,
            args.tcrdist_max_clones, args.seed,
        )
        graph_ids = tcrdist_graph["clone_ids"]
        pd.DataFrame([
            {"clone_id_a": graph_ids[a], "clone_id_b": graph_ids[b],
             "tcrdist_alpha_beta": distance, "weight": weight}
            for (a, b, weight), distance in zip(
                tcrdist_graph["edges"], tcrdist_graph["edge_distances"])
        ]).to_csv(out / "tcrdist_neighbor_pairs.csv", index=False)

    embeddings = {}
    rna_model = None
    if "rna_only" in methods or "fused" in methods:
        print("\n[Stage 1] RNA reconstruction")
        rna_model = train_rna(
            train_x, train_clone_ids, args.train_sampling, args.seed,
            args.rna_epochs, args.batch_size, device,
            out / "rna_only_encoder.pt",
        )
        if "rna_only" in methods:
            z = extract_rna_only(rna_model, x, args.embed_batch_size, device)
            embeddings["rna_only"] = z.astype(np.float32)
            print(f"[Embedding] rna_only: {z.shape}")

    if "fused" in methods:
        # Copying state_dict from CPU to the Stage 2 model leaves GPU memory for
        # the paired-chain transformer and autoregressive decoders.
        rna_model.cpu()
        if device.type == "cuda": torch.cuda.empty_cache()
        print("\n[Stage 2] shared/private RNA-TCR fusion")
        model = train_fusion(
            train_x, train_a, train_b, train_clone_ids,
            args.train_sampling, args.seed, rna_model,
            args.fused_epochs, args.batch_size, device,
            out / "fused_encoder.pt",
            tcrdist_graph=tcrdist_graph,
            tcrdist_weight=args.tcrdist_weight,
            tcrdist_pairs_per_step=args.tcrdist_pairs_per_step,
            tcrdist_warmup_epochs=args.tcrdist_warmup_epochs,
            tcrdist_variance_floor=args.tcrdist_variance_floor,
            tcrdist_variance_weight=args.tcrdist_variance_weight,
        )
        z = extract_fused(model, x, cdr3a, cdr3b, args.embed_batch_size, device)
        embeddings["fused"] = z
        print(f"[Embedding] fused: {z.shape}; TCR-private weight={args.tcr_private_weight}")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    del rna_model

    if "tcr_only" in methods:
        print("\n[Control] independent TCR-only reconstruction")
        model = train_tcr_only(train_a, train_b, args.pretrain_epochs,
                               args.batch_size, device, args.seed)
        z = extract_tcr_only(model, cdr3a, cdr3b, args.embed_batch_size)
        embeddings["tcr_only"] = z.astype(np.float32)
        torch.save(model.state_dict(), out / "tcr_only_encoder.pt")
        print(f"[Embedding] tcr_only: {z.shape}")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    obs.index = pd.Index(obs["barcode"].astype(str), name="barcode")
    obs = obs.drop(columns=["barcode"])
    primary = embeddings.get("fused", next(iter(embeddings.values())))
    var = pd.DataFrame(index=[f"latent_{i}" for i in range(primary.shape[1])])
    adata = ad.AnnData(X=primary, obs=obs, var=var)
    for method, z in embeddings.items():
        adata.obsm[f"X_{method}"] = z
    adata.uns["description"] = (
        "Two-stage RNA/TCR training on paired vaccine ECCITE-seq cells; "
        "dextramer labels annotate outputs but do not enter the training loss."
    )
    adata.uns["training_scope"] = args.training_scope
    adata.uns["training_methods"] = methods
    adata.uns["primary_X"] = "fused" if "fused" in embeddings else methods[0]
    adata.uns["fusion_embedding"] = "concat(shared, rna_private, tcr_private_weight * tcr_private)"
    adata.uns["tcr_private_weight"] = args.tcr_private_weight
    adata.uns["tcrdist_regularization"] = {
        "enabled": bool(tcrdist_graph is not None),
        "weight": args.tcrdist_weight,
        "mutual_neighbors": args.tcrdist_neighbors,
        "distance_quantile": args.tcrdist_quantile,
        "n_pairs": len(tcrdist_graph["edges"]) if tcrdist_graph is not None else 0,
        "pair_source": "paired TRA/TRB CDR3 and V/J calls only",
        "loss": "relative Euclidean ranking in deterministic fused space plus angular variance",
        "reference": "farthest TCRdist quartile for each anchor",
        "ranking_margin": 0.2,
    }
    adata.uns["label_definition"] = {
        "virus_label": "SARS-CoV-2 for published Dex-positive calls",
        "No_recorded_virus_hit": "no measured dextramer binding call; not a proven nonreactive cell",
    }
    adata.write_h5ad(out / "virus_unsupervised_embeddings.h5ad", compression="gzip")
    np.savez_compressed(out / "rna_scaler.npz", mean=mean, std=std)
    with open(out / "genes.json", "w") as f:
        json.dump(gene_names, f)
    config = vars(args).copy()
    config.update(device=str(device), n_export_cells=len(rows), n_train_cells=train_n,
                  matrix_shape=list(matrix_shape), labels_used_for_selection=False,
                  training_labels_used=False,
                  fused_model_source=str(Path(__file__).resolve()))
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Saved] {out / 'virus_unsupervised_embeddings.h5ad'}")


if __name__ == "__main__":
    main()
