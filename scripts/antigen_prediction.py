#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import re
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from transformers import AutoTokenizer, AutoModel


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="fused",
    choices=["rna_only", "tcr_only", "fused"],
)
parser.add_argument(
    "--require_paired_ab",
    type=int,
    default=0,
    help="1 = only keep cells with both cdr3a and cdr3b",
)
parser.add_argument(
    "--cuda",
    type=str,
    default=os.environ.get("CUDA_DEVICE", "cuda:2"),
)
args, _ = parser.parse_known_args()

TRAIN_MODE = args.mode
REQUIRE_PAIRED_AB = bool(args.require_paired_ab)

DEVICE = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Mode] {TRAIN_MODE}")
print(f"[Device] {DEVICE}")
print(f"[Require paired AB] {REQUIRE_PAIRED_AB}")


BASE_10X_DIR = "./data/10x/"
PER_CELL_ROOT = "./data/10x/antigen_global_multiclass"
DONORS = [f"donor{i}" for i in range(1, 5)]

TCRBERT_DIR = "./pretrained/tcr-bert/"
UNFREEZE_LAST_N = 2

BASE_SAVE_ROOT = "./result/epibind"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{TRAIN_MODE}_pairedAB{int(REQUIRE_PAIRED_AB)}_seed{SEED}_{RUN_TAG}"
SAVE_ROOT = os.path.join(BASE_SAVE_ROOT, RUN_NAME)
os.makedirs(SAVE_ROOT, exist_ok=True)


EPOCHS_RNA = 50
LR_RNA = 1e-4

EPOCHS_TCR = 50
LR_TCR = 1e-4

EPOCHS_FUSED = 50
LR_FUSED = 1e-4

EPOCHS_SUP = 50
LR_SUP = 1e-3
WEIGHT_DECAY_SUP = 1e-4
EARLY_STOP_PATIENCE_SUP = 20
MIN_DELTA_SUP = 1e-4

BATCH_SIZE = 1024


EMB_D = 128
SHARED_D = 64
PRIV_D = 64

LAMBDA_TCR = 1.0
LAMBDA_ORTH = 0.3
LAMBDA_GATE = 0.0


LAMBDA_KL = 1e-4
LAMBDA_KL_PRIV_MULT = 2.0
KL_WARMUP_EPOCHS = 10
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0

TCR_SCALE = 0.2
FUSE_SCALE = 0.2
GATE_BIAS = 0.0
ALPHA_WEIGHT = 0.3
ALPHA_RECON_WEIGHT = 0.3


TCR_PRIV_IN_EMBED = True

MAX_LEN = 20



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



mode_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
    "..",
    "model",
))

if mode_path not in sys.path:
    sys.path.append(mode_path)

from model import (  # noqa: E402
    GeneMLPEncoder,
    Cross_attention,
    RNADecoder,
    CDR3SeqDecoder,
)


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

        self.trb_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=gene_out_dim, unfreeze_last_n=unfreeze_last_n)
        self.tra_encoder = TCRBertEncoder(TCRBERT_DIR, out_dim=gene_out_dim, unfreeze_last_n=0)
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
            gate_bias=GATE_BIAS,
            detach_gate_gene=True,
        )


        self.shared_vae = VAEBottleneck(gene_out_dim, shared_dim)
        self.rna_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)
        self.tcr_priv_vae = VAEBottleneck(gene_out_dim, priv_dim)


        self.rna_dec_shared = RNADecoder(
            d_model=shared_dim, gene_dim=gene_dim, width=512, depth=3, mult=4,
            dropout=0.1, use_output_bias=True,
        )
        self.rna_dec_priv = RNADecoder(
            d_model=priv_dim, gene_dim=gene_dim, width=512, depth=3, mult=4,
            dropout=0.1, use_output_bias=False,
        )


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


        trb_tok, trb_pad = self.trb_encoder.encode_tokens(cdr3b_list)
        trb_tok = self.tcr_norm(trb_tok)
        z_trb = self._pool(trb_tok, trb_pad)

        tra_tok, tra_pad = self.tra_encoder.encode_tokens(cdr3a_list)
        tra_tok = self.tcr_norm(tra_tok)
        z_tra = self._pool(tra_tok, tra_pad)


        z_tcr_raw = self.tcr_mix(torch.cat([z_trb, ALPHA_WEIGHT * z_tra], dim=-1))

        tcr_tok = torch.cat([trb_tok, ALPHA_WEIGHT * tra_tok], dim=1)
        pad_mask = torch.cat([trb_pad, tra_pad], dim=1)
        z_cross, tcr_ctx_tok, gate_score, attn = self.cross_attention(
            z_rna_raw, tcr_tok, pad_mask=pad_mask,
        )


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
        }

    def forward(self, rna: Tensor, cdr3a_list: List[str], cdr3b_list: List[str]):
        return self.encode_backbone(rna, cdr3a_list, cdr3b_list)


    def rna_recon_loss(self, z_shared: Tensor, z_rna_priv: Tensor, rna_true: Tensor):
        pred = self.rna_dec_shared(z_shared) + self.rna_dec_priv(z_rna_priv)
        return F.mse_loss(pred, rna_true)

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

    def vae_kl_loss(self, out: Dict[str, Tensor], priv_mult: float = LAMBDA_KL_PRIV_MULT) -> Tensor:
        return (
            out["kl_shared"].mean()
            + priv_mult * out["kl_rna_priv"].mean()
            + priv_mult * out["kl_tcr_priv"].mean()
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


def preprocess_rna(adata: sc.AnnData, selected_genes: List[str] = None):
    ad = adata.copy()

    tcr_ig_pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", flags=re.IGNORECASE)
    keep_genes = [g for g in ad.var_names if not tcr_ig_pattern.match(g)]
    removed_count = ad.n_vars - len(keep_genes)
    ad = ad[:, keep_genes].copy()

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if selected_genes is None:
        X = ad.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        gene_var = X.var(axis=0)
        k = min(2000, gene_var.shape[0])
        top_idx = np.argsort(gene_var)[-k:]
        selected_genes = ad.var_names[top_idx].tolist()
        print(f"  [RNA] removed {removed_count} TCR/Ig genes; selected HVGs={len(selected_genes)}")
    else:
        missing = [g for g in selected_genes if g not in ad.var_names]
        if missing:
            raise ValueError(f"Missing selected genes in donor RNA: {missing[:10]} ... total={len(missing)}")

    ad = ad[:, selected_genes].copy()
    ad.var["highly_variable"] = True
    sc.pp.scale(ad, max_value=10)

    X = ad.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return X.astype(np.float32), ad.obs_names.astype(str).str.strip().tolist(), selected_genes


def load_rna_for_donor(donor: str, selected_genes: List[str] = None):
    rna_path = os.path.join(BASE_10X_DIR, donor, "rna.h5ad")
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"{donor}: missing RNA file: {rna_path}")
    print(f"[{donor}] RNA: {rna_path}")
    ad = sc.read_h5ad(rna_path)
    ad.obs_names = ad.obs_names.astype(str).str.strip()
    print(f"[{donor}] RNA raw: cells={ad.n_obs}, genes={ad.n_vars}")
    X, barcodes, selected_genes = preprocess_rna(ad, selected_genes=selected_genes)
    print(f"[{donor}] RNA processed: cells={X.shape[0]}, genes={X.shape[1]}")
    return X, barcodes, selected_genes


def load_per_cell_multiclass(donor: str):
    donor_dir = os.path.join(PER_CELL_ROOT, donor)
    path = os.path.join(donor_dir, f"{donor}_per_cell_multiclass.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{donor}: missing per-cell table: {path}")

    df = pd.read_csv(path, dtype=str)
    if "barcode" not in df.columns or "cdr3b" not in df.columns:
        raise RuntimeError(f"{donor}: per-cell table must contain barcode and cdr3b")
    if "cdr3a" not in df.columns:
        df["cdr3a"] = ""

    df["barcode"] = df["barcode"].astype(str).str.strip()
    df["cdr3a"] = df["cdr3a"].fillna("").astype(str).str.strip()
    df["cdr3b"] = df["cdr3b"].fillna("").astype(str).str.strip()
    df["donor"] = donor

    meta_cols = {
        "donor", "barcode", "cdr3a", "cdr3b", "TRAV", "TRAJ", "TRBV", "TRBJ",
        "has_binding", "binding_label", "binding_name",
        "high_count_binding_label", "high_count_binding_name",
    }
    antigen_cols = [c for c in df.columns if c not in meta_cols]
    for c in antigen_cols:
        df[c] = df[c].fillna("0").astype(float).astype(int)

    df["cdr3a"] = df["cdr3a"].map(clean_cdr3)
    df["cdr3b"] = df["cdr3b"].map(clean_cdr3)

    if REQUIRE_PAIRED_AB:
        df = df[(df["cdr3a"].str.len() >= 5) & (df["cdr3b"].str.len() >= 5)].reset_index(drop=True)
    else:
        df = df[df["cdr3b"].str.len() >= 5].reset_index(drop=True)

    return df, antigen_cols

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, z_cell: np.ndarray,
                 bc2idx: Dict[str, int], ag2id: Dict[str, int]):
        self.z_cell = torch.tensor(z_cell, dtype=torch.float32)
        zs, ag_ids, ys = [], [], []
        miss_bc = miss_ag = 0

        for _, row in df.iterrows():
            bc = str(row["barcode"]).strip()
            ag = str(row["antigen"]).strip()
            y = int(row["label"])
            if bc not in bc2idx:
                miss_bc += 1
                continue
            if ag not in ag2id:
                miss_ag += 1
                continue
            zs.append(self.z_cell[bc2idx[bc]])
            ag_ids.append(ag2id[ag])
            ys.append(y)

        if miss_bc > 0:
            print(f"  PairDataset: skipped {miss_bc} rows due to missing barcode")
        if miss_ag > 0:
            print(f"  PairDataset: skipped {miss_ag} rows due to missing antigen")

        if len(zs) == 0:
            self.z_pairs = torch.empty((0, self.z_cell.shape[1]), dtype=torch.float32)
            self.ag_ids = torch.empty((0,), dtype=torch.long)
            self.y = torch.empty((0,), dtype=torch.float32)
        else:
            self.z_pairs = torch.stack(zs, dim=0)
            self.ag_ids = torch.tensor(ag_ids, dtype=torch.long)
            self.y = torch.tensor(ys, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.z_pairs[idx], self.ag_ids[idx], self.y[idx]


class PairClassifier(nn.Module):
    def __init__(self, cell_dim: int, num_antigen: int, ag_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.ag_emb = nn.Embedding(num_antigen, ag_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cell_dim + ag_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_cell: Tensor, ag_ids: Tensor):
        ag = self.ag_emb(ag_ids)
        x = torch.cat([z_cell, ag], dim=-1)
        return self.mlp(x).squeeze(-1)


def make_pos_weight(y: np.ndarray):
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos <= 0:
        return 1.0
    return neg / max(pos, 1.0)


def stratified_split_indices(y: np.ndarray, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=SEED):
    assert np.isclose(train_frac + val_frac + test_frac, 1.0)
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    idx_all = np.arange(len(y))
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(y):
        idx_c = idx_all[y == c]
        rng.shuffle(idx_c)
        n = len(idx_c)
        if n < 5:
            train_idx.extend(idx_c.tolist())
            continue
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
        if n_val <= 0:
            n_val = 1
        n_train = max(1, n - n_val - n_test)
        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:n_train + n_val].tolist())
        test_idx.extend(idx_c[n_train + n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def compute_metrics_from_logits(logits: np.ndarray, y_true: np.ndarray):
    prob = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, prob)
    acc = accuracy_score(y_true, (prob >= 0.5).astype(np.float32))
    return dict(auc=auc, ap=ap, acc=acc)


@torch.no_grad()
def eval_pair_classifier(model: PairClassifier, loader: DataLoader,
                         bce_loss: nn.BCEWithLogitsLoss, split_name: str = "val"):
    model.eval()
    all_logits, all_y = [], []
    total_loss, n_batch = 0.0, 0

    for z_cell, ag_ids, y in loader:
        z_cell = z_cell.to(DEVICE)
        ag_ids = ag_ids.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(z_cell, ag_ids)
        loss = bce_loss(logits, y)
        total_loss += loss.item()
        n_batch += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    if n_batch == 0:
        print(f"[Pair-{split_name}] empty loader")
        return dict(loss=np.nan, auc=np.nan, ap=np.nan, acc=np.nan)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    mets = compute_metrics_from_logits(logits, y_true)
    avg_loss = total_loss / n_batch
    print(
        f"[Pair-{split_name}] loss={avg_loss:.4f} | "
        f"AUC={mets['auc']:.4f} | AP={mets['ap']:.4f} | ACC={mets['acc']:.4f}"
    )
    return dict(loss=avg_loss, **mets)


@torch.no_grad()
def collect_logits_and_labels(model: PairClassifier, loader: DataLoader):
    model.eval()
    logits_all, y_all = [], []
    for z_cell, ag_ids, y in loader:
        z_cell = z_cell.to(DEVICE)
        ag_ids = ag_ids.to(DEVICE)
        logits = model(z_cell, ag_ids)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
    if not logits_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(logits_all), np.concatenate(y_all)


def train_pair_classifier(model: PairClassifier, train_loader: DataLoader, val_loader: DataLoader,
                          pos_weight: float, epochs: int = EPOCHS_SUP):
    model.to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR_SUP, weight_decay=WEIGHT_DECAY_SUP)

    best_val_ap, best_state, best_epoch, epochs_no_improve = -1.0, None, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        total, n_batch = 0.0, 0
        for z_cell, ag_ids, y in train_loader:
            z_cell = z_cell.to(DEVICE)
            ag_ids = ag_ids.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            logits = model(z_cell, ag_ids)
            loss = bce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n_batch += 1

        print(f"[Pair-Train][Epoch {ep:03d}] loss={total/max(n_batch,1):.4f}")
        val_res = eval_pair_classifier(model, val_loader, bce, split_name="val")
        cur_ap = val_res["ap"]
        if np.isnan(cur_ap):
            cur_ap = 0.0

        if cur_ap > best_val_ap + MIN_DELTA_SUP:
            best_val_ap, best_epoch = cur_ap, ep
            best_state = {"model": model.state_dict(), "epoch": ep, "val_ap": best_val_ap}
            epochs_no_improve = 0
            print(f"[EarlyStop] AP improved to {best_val_ap:.4f}")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStop] AP not improved: {epochs_no_improve}/{EARLY_STOP_PATIENCE_SUP}")
            if epochs_no_improve >= EARLY_STOP_PATIENCE_SUP:
                print("[EarlyStop] stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"[Pair] loaded best state: epoch={best_state['epoch']}, val AP={best_state['val_ap']:.4f}")
    else:
        print("[Pair] warning: no best state")

    return dict(best_epoch=best_epoch, best_val_ap=best_val_ap)


def main():
    print(f"Device: {DEVICE}")
    print(f"Donors: {DONORS}")
    print(f"Save root: {SAVE_ROOT}")
    print(f"TRAIN_MODE: {TRAIN_MODE}")
    print(f"TCR encoder: TCR-BERT @ {TCRBERT_DIR}")
    print("[Arch] code1 shared/private 4-decoder additive (fused); VAE full-recon baselines")

    cfg_path = os.path.join(SAVE_ROOT, "config.txt")
    with open(cfg_path, "w") as f:
        f.write(f"TRAIN_MODE={TRAIN_MODE}\n")
        f.write(f"REQUIRE_PAIRED_AB={REQUIRE_PAIRED_AB}\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"EMB_D={EMB_D}, SHARED_D={SHARED_D}, PRIV_D={PRIV_D}\n")
        f.write(f"TCR_LM=TCRBERT  DIR={TCRBERT_DIR}  UNFREEZE_LAST_N={UNFREEZE_LAST_N}\n")
        f.write(f"LAMBDA_TCR={LAMBDA_TCR}, LAMBDA_ORTH={LAMBDA_ORTH}, LAMBDA_GATE={LAMBDA_GATE}\n")
        f.write(f"LAMBDA_KL={LAMBDA_KL}, LAMBDA_KL_PRIV_MULT={LAMBDA_KL_PRIV_MULT}, KL_WARMUP={KL_WARMUP_EPOCHS}\n")
        f.write(f"TCR_SCALE={TCR_SCALE}, FUSE_SCALE={FUSE_SCALE}, GATE_BIAS={GATE_BIAS}\n")
        f.write(f"ALPHA_WEIGHT={ALPHA_WEIGHT}, ALPHA_RECON_WEIGHT={ALPHA_RECON_WEIGHT}\n")
        f.write(f"TCR_PRIV_IN_EMBED={TCR_PRIV_IN_EMBED}\n")
        f.write("fused RNA recon = rna_dec_shared(z_shared)+rna_dec_priv(z_rna_priv) [output-add]\n")
        f.write("fused TCR recon = dec_shared(z_shared)+dec_priv(z_tcr_priv) per chain [logit-add, AR full CDR3]\n")
        f.write("fused embedding = normalize(concat(norm(shared), norm(rna_priv)[, norm(tcr_priv)]))\n")

    donor_data: Dict[str, Dict] = {}
    selected_genes = None
    gene_dim = None

    for donor in DONORS:
        try:
            X_rna, barcodes, selected_genes = load_rna_for_donor(donor, selected_genes=selected_genes)
            if gene_dim is None:
                gene_dim = X_rna.shape[1]
            elif X_rna.shape[1] != gene_dim:
                raise ValueError(f"{donor}: gene_dim mismatch")

            df_percell, antigen_cols = load_per_cell_multiclass(donor)
            bc2idx_rna = {bc: i for i, bc in enumerate(barcodes)}
            df_percell = df_percell[df_percell["barcode"].isin(bc2idx_rna)].reset_index(drop=True)

            if len(df_percell) == 0:
                print(f"[{donor}] no matched per-cell rows; skip")
                continue

            donor_data[donor] = dict(X=X_rna, barcodes=barcodes, percell=df_percell, antigen_cols=antigen_cols)
            print(f"[{donor}] matched per-cell rows: {len(df_percell)}")
        except Exception as e:
            print(f"[{donor}] failed: {e}")

    if len(donor_data) == 0:
        print("No valid donor data.")
        return


    print("\n========== Stage 1: unsupervised representation learning ==========")

    if TRAIN_MODE in {"rna_only", "fused"}:
        X_list, cdr3a_all, cdr3b_all = [], [], []
        for donor, dct in donor_data.items():
            X, barcodes, df = dct["X"], dct["barcodes"], dct["percell"]
            bc2idx_rna = {bc: i for i, bc in enumerate(barcodes)}
            idxs = [bc2idx_rna[bc] for bc in df["barcode"]]
            X_list.append(X[idxs])
            cdr3a_all.extend(df["cdr3a"].tolist())
            cdr3b_all.extend(df["cdr3b"].tolist())
            print(f"[{donor}] unsup cells: {len(idxs)}")
        X_all = np.concatenate(X_list, axis=0)
        X_all_t = torch.tensor(X_all, dtype=torch.float32)

    if TRAIN_MODE == "tcr_only":
        pairs = []
        for donor, dct in donor_data.items():
            df = dct["percell"]
            for a, b in zip(df["cdr3a"], df["cdr3b"]):
                a, b = clean_cdr3(a), clean_cdr3(b)
                if len(b) < 5:
                    continue
                if REQUIRE_PAIRED_AB and len(a) < 5:
                    continue
                pairs.append((a, b))
        pairs = sorted(set(pairs))
        cdr3a_all = [p[0] for p in pairs]
        cdr3b_all = [p[1] for p in pairs]
        print(f"TCR-only unique TRA/TRB pairs: {len(pairs)}")


    if TRAIN_MODE == "rna_only":
        ds = UnsupervisedDataset(X_all_t, [""] * X_all_t.shape[0], [""] * X_all_t.shape[0])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_unsup_batch)
        encoder = RNAOnlyModel(gene_dim=gene_dim, latent_dim=EMB_D).to(DEVICE)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()),
                                lr=LR_RNA, weight_decay=1e-4)
        for ep in range(EPOCHS_RNA):
            train_rna_only(encoder, loader, opt, ep)

    elif TRAIN_MODE == "tcr_only":
        ds = TCRPairDataset(cdr3a_all, cdr3b_all)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_tcr_pair)
        encoder = TCROnlyModel(latent_dim=EMB_D).to(DEVICE)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()),
                                lr=LR_TCR, weight_decay=1e-4)
        for ep in range(EPOCHS_TCR):
            train_tcr_only(encoder, loader, opt, ep)

    elif TRAIN_MODE == "fused":
        ds = UnsupervisedDataset(X_all_t, cdr3a_all, cdr3b_all)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_unsup_batch)
        encoder = FusedModel(gene_dim=gene_dim).to(DEVICE)

        n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in encoder.parameters())
        print(f"[Fusion] trainable params: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M")

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()),
                                lr=LR_FUSED, weight_decay=1e-4)
        for ep in range(EPOCHS_FUSED):
            train_fused(encoder, loader, opt, ep)
    else:
        raise ValueError(TRAIN_MODE)

    print("\n========== Stage 2: supervised cell-antigen prediction ==========")
    all_rows = []

    for donor in DONORS:
        if donor not in donor_data:
            continue

        print("\n" + "=" * 80)
        print(f"[Donor] {donor}")

        X_rna = donor_data[donor]["X"]
        barcodes = donor_data[donor]["barcodes"]
        df_percell = donor_data[donor]["percell"]
        antigen_cols = donor_data[donor]["antigen_cols"]
        bc2idx_rna = {bc: i for i, bc in enumerate(barcodes)}

        keep_idxs, keep_bcs, keep_cdr3a, keep_cdr3b = [], [], [], []
        for bc, a, b in zip(df_percell["barcode"], df_percell["cdr3a"], df_percell["cdr3b"]):
            if bc not in bc2idx_rna:
                continue
            a, b = clean_cdr3(a), clean_cdr3(b)
            if len(b) < 5:
                continue
            if REQUIRE_PAIRED_AB and len(a) < 5:
                continue
            keep_idxs.append(bc2idx_rna[bc])
            keep_bcs.append(bc)
            keep_cdr3a.append(a)
            keep_cdr3b.append(b)

        if len(keep_idxs) == 0:
            print(f"[{donor}] no usable cells; skip")
            continue

        X_sub = X_rna[keep_idxs]
        print(f"[{donor}] usable cells: {X_sub.shape[0]}")

        if TRAIN_MODE == "rna_only":
            z_cell = compute_rna_embedding(encoder, X_sub)
        elif TRAIN_MODE == "tcr_only":
            z_cell = compute_tcr_embedding(encoder, keep_cdr3a, keep_cdr3b)
        elif TRAIN_MODE == "fused":
            z_cell = compute_fused_eval_embedding(encoder, X_sub, keep_cdr3a, keep_cdr3b)
        else:
            raise ValueError(TRAIN_MODE)

        cell_dim = z_cell.shape[1]
        bc2idx_embed = {bc: i for i, bc in enumerate(keep_bcs)}
        print(f"[{donor}] mode={TRAIN_MODE}, cell_dim={cell_dim}")

        df_sub = df_percell[df_percell["barcode"].isin(keep_bcs)].reset_index(drop=True)
        df_long = df_sub.melt(id_vars=["donor", "barcode"], value_vars=antigen_cols,
                              var_name="antigen", value_name="label")
        df_long["label"] = df_long["label"].astype(int)

        y_all = df_long["label"].values.astype(int)
        num_pos = int(y_all.sum())
        print(f"[{donor}] pair rows={len(df_long)}, positive={num_pos} ({num_pos / len(df_long):.4%})")

        train_idx, val_idx, test_idx = stratified_split_indices(
            y_all, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=SEED
        )
        df_train = df_long.iloc[train_idx].reset_index(drop=True)
        df_val = df_long.iloc[val_idx].reset_index(drop=True)
        df_test = df_long.iloc[test_idx].reset_index(drop=True)

        all_antigens = sorted(df_long["antigen"].unique().tolist())
        ag2id = {ag: i for i, ag in enumerate(all_antigens)}
        num_antigen = len(all_antigens)

        train_ds = PairDataset(df_train, z_cell, bc2idx_embed, ag2id)
        val_ds = PairDataset(df_val, z_cell, bc2idx_embed, ag2id)
        test_ds = PairDataset(df_test, z_cell, bc2idx_embed, ag2id)

        if len(train_ds) == 0:
            print(f"[{donor}] empty train set; skip")
            continue

        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=False,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, drop_last=False,
                                num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, drop_last=False,
                                 num_workers=4, pin_memory=True)

        y_tr = train_ds.y.numpy()
        pos_w = make_pos_weight(y_tr)
        print(f"[{donor}] train N={len(y_tr)}, pos={int(y_tr.sum())} ({y_tr.mean():.2%}), pos_weight={pos_w:.4f}")

        clf = PairClassifier(cell_dim=cell_dim, num_antigen=num_antigen, ag_dim=64, hidden=256)
        sup_info = train_pair_classifier(clf, train_loader, val_loader, pos_weight=pos_w, epochs=EPOCHS_SUP)

        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
        print(f"\n[{donor}] Test evaluation")
        test_res = eval_pair_classifier(clf.to(DEVICE), test_loader, bce, split_name="test")

        shuffled_ap = shuffled_auc = np.nan
        tcr_shuf_test_res = tcr_shuf_sup_info = tcr_shuf_pos_w = None
        rna_shuf_test_res = rna_shuf_sup_info = rna_shuf_pos_w = None

        if TRAIN_MODE == "fused":
            logits, y_true = collect_logits_and_labels(clf, test_loader)
            if len(logits) > 0:
                rng = np.random.RandomState(SEED)
                y_shuf = y_true.copy()
                rng.shuffle(y_shuf)
                mets_shuf = compute_metrics_from_logits(logits, y_shuf)
                shuffled_ap, shuffled_auc = mets_shuf["ap"], mets_shuf["auc"]
                print(f"[{donor}] label-shuffle sanity: AP={shuffled_ap:.4f}, AUC={shuffled_auc:.4f}")


            print(f"[{donor}] fused control: TCR-shuffle")
            rng_tcr = np.random.RandomState(SEED)
            perm_tcr = rng_tcr.permutation(len(keep_cdr3b))
            cdr3a_tcr_shuf = [keep_cdr3a[i] for i in perm_tcr]
            cdr3b_tcr_shuf = [keep_cdr3b[i] for i in perm_tcr]
            z_tcr_shuf = compute_fused_eval_embedding(encoder, X_sub, cdr3a_tcr_shuf, cdr3b_tcr_shuf)

            train_ds_ts = PairDataset(df_train, z_tcr_shuf, bc2idx_embed, ag2id)
            val_ds_ts = PairDataset(df_val, z_tcr_shuf, bc2idx_embed, ag2id)
            test_ds_ts = PairDataset(df_test, z_tcr_shuf, bc2idx_embed, ag2id)

            if len(train_ds_ts) > 0:
                tl = DataLoader(train_ds_ts, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
                vl = DataLoader(val_ds_ts, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
                te = DataLoader(test_ds_ts, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
                tcr_shuf_pos_w = make_pos_weight(train_ds_ts.y.numpy())
                clf_ts = PairClassifier(cell_dim=z_tcr_shuf.shape[1], num_antigen=num_antigen, ag_dim=64, hidden=256)
                tcr_shuf_sup_info = train_pair_classifier(clf_ts, tl, vl, pos_weight=tcr_shuf_pos_w, epochs=EPOCHS_SUP)
                bce_ts = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([tcr_shuf_pos_w], device=DEVICE))
                print(f"\n[{donor}] Test evaluation: TCR-shuffle")
                tcr_shuf_test_res = eval_pair_classifier(clf_ts.to(DEVICE), te, bce_ts, split_name="test_tcr_shuffle")

            # ---- RNA-shuffle control ----
            print(f"[{donor}] fused control: RNA-shuffle")
            rng_rna = np.random.RandomState(SEED + 1)
            perm_rna = rng_rna.permutation(X_sub.shape[0])
            X_sub_rna_shuf = X_sub[perm_rna, :]
            z_rna_shuf = compute_fused_eval_embedding(encoder, X_sub_rna_shuf, keep_cdr3a, keep_cdr3b)

            train_ds_rs = PairDataset(df_train, z_rna_shuf, bc2idx_embed, ag2id)
            val_ds_rs = PairDataset(df_val, z_rna_shuf, bc2idx_embed, ag2id)
            test_ds_rs = PairDataset(df_test, z_rna_shuf, bc2idx_embed, ag2id)

            if len(train_ds_rs) > 0:
                tl = DataLoader(train_ds_rs, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
                vl = DataLoader(val_ds_rs, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
                te = DataLoader(test_ds_rs, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
                rna_shuf_pos_w = make_pos_weight(train_ds_rs.y.numpy())
                clf_rs = PairClassifier(cell_dim=z_rna_shuf.shape[1], num_antigen=num_antigen, ag_dim=64, hidden=256)
                rna_shuf_sup_info = train_pair_classifier(clf_rs, tl, vl, pos_weight=rna_shuf_pos_w, epochs=EPOCHS_SUP)
                bce_rs = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([rna_shuf_pos_w], device=DEVICE))
                print(f"\n[{donor}] Test evaluation: RNA-shuffle")
                rna_shuf_test_res = eval_pair_classifier(clf_rs.to(DEVICE), te, bce_rs, split_name="test_rna_shuffle")

            gate_path = os.path.join(SAVE_ROOT, f"{donor}_fused_gate_scores.npz")
            save_fused_gate_for_donor(encoder, X_sub, keep_cdr3a, keep_cdr3b, keep_bcs, gate_path, batch_size=512)

        donor_summary_path = os.path.join(SAVE_ROOT, f"{donor}_summary.txt")
        with open(donor_summary_path, "w") as f:
            f.write(f"donor={donor}\nTRAIN_MODE={TRAIN_MODE}\nREQUIRE_PAIRED_AB={REQUIRE_PAIRED_AB}\n")
            f.write(f"gene_dim={gene_dim}\ncell_dim={cell_dim}\nnum_antigen={num_antigen}\n")
            f.write(f"BEST_SUP_EPOCH_aligned={sup_info['best_epoch']}\n")
            f.write(f"BEST_SUP_VAL_AP_aligned={sup_info['best_val_ap']:.4f}\n")
            f.write(f"pos_weight_aligned={pos_w:.4f}\n\nTest metrics aligned:\n")
            f.write(f"loss={test_res['loss']:.4f}, AUC={test_res['auc']:.4f}, "
                    f"AP={test_res['ap']:.4f}, ACC={test_res['acc']:.4f}\n")

            if TRAIN_MODE == "fused":
                f.write(f"\nLabel-shuffle sanity:\nshuffled_AP={shuffled_ap:.4f}, shuffled_AUC={shuffled_auc:.4f}\n")
                if tcr_shuf_test_res is not None:
                    f.write("\nTCR-shuffle control:\n")
                    f.write(f"BEST_SUP_EPOCH_tcr_shuffle={tcr_shuf_sup_info['best_epoch']}\n")
                    f.write(f"BEST_SUP_VAL_AP_tcr_shuffle={tcr_shuf_sup_info['best_val_ap']:.4f}\n")
                    f.write(f"pos_weight_tcr_shuffle={tcr_shuf_pos_w:.4f}\n")
                    f.write(f"loss={tcr_shuf_test_res['loss']:.4f}, AUC={tcr_shuf_test_res['auc']:.4f}, "
                            f"AP={tcr_shuf_test_res['ap']:.4f}, ACC={tcr_shuf_test_res['acc']:.4f}\n")
                if rna_shuf_test_res is not None:
                    f.write("\nRNA-shuffle control:\n")
                    f.write(f"BEST_SUP_EPOCH_rna_shuffle={rna_shuf_sup_info['best_epoch']}\n")
                    f.write(f"BEST_SUP_VAL_AP_rna_shuffle={rna_shuf_sup_info['best_val_ap']:.4f}\n")
                    f.write(f"pos_weight_rna_shuffle={rna_shuf_pos_w:.4f}\n")
                    f.write(f"loss={rna_shuf_test_res['loss']:.4f}, AUC={rna_shuf_test_res['auc']:.4f}, "
                            f"AP={rna_shuf_test_res['ap']:.4f}, ACC={rna_shuf_test_res['acc']:.4f}\n")

        print(f"[{donor}] summary saved: {donor_summary_path}")

        row = dict(
            donor=donor, mode=TRAIN_MODE, require_paired_ab=REQUIRE_PAIRED_AB, tcr_lm="TCRBERT",
            gene_dim=gene_dim, cell_dim=cell_dim, num_antigen=num_antigen,
            best_sup_epoch_aligned=sup_info["best_epoch"], best_sup_val_ap_aligned=sup_info["best_val_ap"],
            pos_weight_aligned=pos_w, test_loss_aligned=test_res["loss"], test_auc_aligned=test_res["auc"],
            test_ap_aligned=test_res["ap"], test_acc_aligned=test_res["acc"],
        )
        if TRAIN_MODE == "fused":
            row.update(shuffled_AP=shuffled_ap, shuffled_AUC=shuffled_auc)
            if tcr_shuf_test_res is not None:
                row.update(
                    best_sup_epoch_tcr_shuffle=tcr_shuf_sup_info["best_epoch"],
                    best_sup_val_ap_tcr_shuffle=tcr_shuf_sup_info["best_val_ap"],
                    pos_weight_tcr_shuffle=tcr_shuf_pos_w,
                    test_loss_tcr_shuffle=tcr_shuf_test_res["loss"], test_auc_tcr_shuffle=tcr_shuf_test_res["auc"],
                    test_ap_tcr_shuffle=tcr_shuf_test_res["ap"], test_acc_tcr_shuffle=tcr_shuf_test_res["acc"],
                )
            if rna_shuf_test_res is not None:
                row.update(
                    best_sup_epoch_rna_shuffle=rna_shuf_sup_info["best_epoch"],
                    best_sup_val_ap_rna_shuffle=rna_shuf_sup_info["best_val_ap"],
                    pos_weight_rna_shuffle=rna_shuf_pos_w,
                    test_loss_rna_shuffle=rna_shuf_test_res["loss"], test_auc_rna_shuffle=rna_shuf_test_res["auc"],
                    test_ap_rna_shuffle=rna_shuf_test_res["ap"], test_acc_rna_shuffle=rna_shuf_test_res["acc"],
                )

        all_rows.append(row)

    if len(all_rows) > 0:
        df_sum = pd.DataFrame(all_rows)
        sum_path = os.path.join(SAVE_ROOT, f"summary_{TRAIN_MODE}_all_donors.tsv")
        df_sum.to_csv(sum_path, sep="\t", index=False)
        print(f"\n[Summary] saved: {sum_path}")
        print(df_sum.to_string(index=False))
    else:
        print("\n[Summary] no valid donor result")

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()