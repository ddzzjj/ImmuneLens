import typing
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FiLMBlock(nn.Module):
    """Pre-norm FiLM-conditioned residual MLP block."""

    def __init__(self, width: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = width * mult
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        x = self.norm(h)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = (1.0 + gamma) * x + beta
        x = self.drop(x)
        return h + x


class GEGLU2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class ResidualMLPBlock2(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.ffn1 = GEGLU2(dim, hidden)
        self.drop1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.ffn1(y)
        y = self.drop1(y)
        y = self.ffn2(y)
        y = self.drop2(y)
        return x + y


class PerceiverAggregator(nn.Module):
    """Aggregate selected gene tokens with learnable latent queries."""

    def __init__(
        self,
        width: int,
        n_latents: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        latent_self_attn: bool = True,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, width) / (width ** 0.5))
        self.q_norm = nn.LayerNorm(width)
        self.kv_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
        self.cross_ffn = ResidualMLPBlock2(width, mult=4, dropout=dropout)
        self.latent_self_attn = latent_self_attn
        if latent_self_attn:
            self.sa_norm = nn.LayerNorm(width)
            self.self_attn = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
            self.self_ffn = ResidualMLPBlock2(width, mult=4, dropout=dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        b, _, c = tokens.shape
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        qn = self.q_norm(q)
        kv = self.kv_norm(tokens)
        z, _ = self.cross_attn(qn, kv, kv, need_weights=False)
        z = z + q
        z = self.cross_ffn(z)
        if self.latent_self_attn:
            zn = self.sa_norm(z)
            zs, _ = self.self_attn(zn, zn, zn, need_weights=False)
            z = z + zs
            z = self.self_ffn(z)
        return z.mean(dim=1)


class GeneMLPEncoder(nn.Module):
    """Low-memory gene encoder with MLP path + optional top-k gene token Perceiver path."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int = 256,
        width: int = 512,
        depth: int = 3,
        mult: int = 4,
        dropout: float = 0.1,
        feature_drop: float = 0.0,
        use_attention: bool = True,
        n_latents: int = 64,
        n_heads: int = 8,
        latent_self_attn: bool = True,
        select_topk: typing.Union[int, None] = 512,
        abs_topk: bool = True,
        detach_gene_embed: bool = False,
    ):
        super().__init__()
        self.feature_drop = float(feature_drop)
        self.use_attention = use_attention
        self.select_topk = select_topk
        self.abs_topk = abs_topk
        self.detach_gene_embed = detach_gene_embed

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, width)

        if use_attention:
            self.gene_embed = nn.Embedding(input_dim, width)
            nn.init.normal_(self.gene_embed.weight, std=1.0 / (width ** 0.5))
            self.attn_agg = PerceiverAggregator(
                width=width,
                n_latents=n_latents,
                n_heads=n_heads,
                dropout=dropout,
                latent_self_attn=latent_self_attn,
            )
            self.fuse = nn.Linear(width * 2, width)

        self.blocks = nn.ModuleList([
            ResidualMLPBlock2(width, mult=mult, dropout=dropout) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_dropout(self, x: Tensor) -> Tensor:
        if not self.training or self.feature_drop <= 0.0:
            return x
        keep = torch.rand(x.shape[1], device=x.device) > self.feature_drop
        scale = 1.0 / keep.float().mean().clamp_min(1e-6)
        return x * keep.float() * scale

    def _build_tokens_topk(self, x: Tensor) -> Tensor:
        b, g = x.shape
        if (self.select_topk is None) or (self.select_topk >= g):
            emb = self.gene_embed.weight.detach() if self.detach_gene_embed else self.gene_embed.weight
            return x.unsqueeze(-1) * emb.unsqueeze(0)

        k = int(self.select_topk)
        score = x.abs() if self.abs_topk else x
        _, topk_idx = torch.topk(score, k=k, dim=1, largest=True, sorted=False)
        emb_weight = self.gene_embed.weight.detach() if self.detach_gene_embed else self.gene_embed.weight
        topk_emb = emb_weight.index_select(0, topk_idx.reshape(-1)).view(b, k, -1)
        topk_x = x.gather(1, topk_idx).unsqueeze(-1)
        return topk_x * topk_emb

    def forward(self, x: Tensor) -> Tensor:
        x = self._feature_dropout(x)
        y = self.in_norm(x)
        y = self.in_proj(y)
        if self.use_attention:
            tokens = self._build_tokens_topk(x)
            z = self.attn_agg(tokens)
            y = self.fuse(torch.cat([y, z], dim=-1))
        for blk in self.blocks:
            y = blk(y)
        y = self.out_norm(y)
        return self.out_proj(y)


class RNADecoder(nn.Module):
    """FiLM-conditioned RNA decoder."""

    def __init__(
        self,
        d_model: int = 256,
        gene_dim: int = 2000,
        width: int = 512,
        depth: int = 3,
        mult: int = 4,
        dropout: float = 0.1,
        use_output_bias: bool = True,
        film_share_mlp: bool = False,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, width)
        self.in_drop = nn.Dropout(dropout)

        def make_film():
            return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, width * 2))

        if film_share_mlp:
            self.film = make_film()
            self.films = None
        else:
            self.films = nn.ModuleList([make_film() for _ in range(depth)])
            self.film = None

        self.blocks = nn.ModuleList([
            FiLMBlock(width, mult=mult, dropout=dropout) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, gene_dim)
        if use_output_bias:
            self.out_bias = nn.Parameter(torch.zeros(1, gene_dim))
        else:
            self.register_parameter("out_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.7)

    def forward(self, memory: Tensor) -> Tensor:
        h = self.in_norm(memory)
        h = F.gelu(self.in_proj(h))
        h = self.in_drop(h)
        for i, blk in enumerate(self.blocks):
            gamma_beta = self.films[i](memory) if self.films is not None else self.film(memory)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma) * 0.5
            beta = beta * 0.5
            h = blk(h, gamma, beta)
        out = self.out_proj(self.out_norm(h))
        if self.out_bias is not None:
            out = out + self.out_bias
        return out


class _FFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _CrossBlock(nn.Module):
    """
    RNA-guided TCR token attention block.

    delta = MHA(Q=RNA, K=TCR_tokens, V=TCR_tokens)
    gate  = sigmoid(MLP([RNA.detach(), delta]))
    RNA'  = LN(RNA + fuse_scale * gate * delta)

    Gate is intentionally not hard-capped. Use a light gate penalty in the main loss.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_self_attn: bool = True,
        tcr_scale: float = 0.2,
        fuse_scale: float = 0.2,
        gate_bias: float = -2.0,
        detach_gate_gene: bool = True,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.tcr_scale = float(tcr_scale)
        self.fuse_scale = float(fuse_scale)
        self.detach_gate_gene = bool(detach_gate_gene)
        self.dropout = nn.Dropout(dropout)
        d = d_model

        self.gene_norm = nn.LayerNorm(d)
        self.tcr_norm = nn.LayerNorm(d)

        if use_self_attn:
            self.tcr_self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
            self.tcr_self_ln1 = nn.LayerNorm(d)
            self.tcr_self_ffn = _FFN(d, mult=4, dropout=dropout)
            self.tcr_self_ln2 = nn.LayerNorm(d)

        self.q_rna = nn.Linear(d, d)
        self.k_tcr = nn.Linear(d, d)
        self.v_tcr = nn.Linear(d, d)
        self.attn_tcr_to_rna = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.delta_norm = nn.LayerNorm(d)

        self.gate_probe = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )
        nn.init.constant_(self.gate_probe[-1].bias, gate_bias)
        self.out_ln = nn.LayerNorm(d)

    def forward(
        self,
        gene: Tensor,
        tcr: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Optional[Tensor], Tensor]]:
        if self.use_self_attn:
            res = tcr
            tcr_sa, _ = self.tcr_self_attn(tcr, tcr, tcr, key_padding_mask=pad_mask, need_weights=False)
            tcr = self.tcr_self_ln1(res + self.dropout(tcr_sa))
            res = tcr
            tcr = self.tcr_self_ln2(res + self.tcr_self_ffn(tcr))

        gene_q = self.gene_norm(gene)
        tcr_kv = self.tcr_norm(tcr) * self.tcr_scale
        q = self.q_rna(gene_q).unsqueeze(1)
        k = self.k_tcr(tcr_kv)
        v = self.v_tcr(tcr_kv)
        delta, attn_t2rna = self.attn_tcr_to_rna(q, k, v, key_padding_mask=pad_mask, need_weights=True)
        delta = self.delta_norm(delta.squeeze(1))

        gate_gene = gene.detach() if self.detach_gate_gene else gene
        raw_gate = self.gate_probe(torch.cat([gate_gene, delta], dim=-1))
        gate = torch.sigmoid(raw_gate)
        gene = self.out_ln(gene + self.fuse_scale * gate * delta)
        return gene, tcr, gate.squeeze(-1), (None, attn_t2rna)


class Cross_attention(nn.Module):
    """RNA-guided TCR token fusion with an uncapped sigmoid gate."""

    def __init__(
        self,
        gene_dim: int,
        cdr_dim: int,
        emb_dim: int,
        num_heads: int = 8,
        use_self_attn: bool = True,
        dropout: float = 0.1,
        layers: int = 1,
        tcr_scale: float = 0.2,
        fuse_scale: float = 0.2,
        gate_bias: float = -2.0,
        detach_gate_gene: bool = True,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        d = emb_dim
        self.proj_gene_in = nn.Linear(gene_dim, d) if gene_dim != d else nn.Identity()
        self.proj_tcr_in = nn.Linear(cdr_dim, d) if cdr_dim != d else nn.Identity()
        self.layers = nn.ModuleList([
            _CrossBlock(
                d_model=d,
                num_heads=num_heads,
                dropout=dropout,
                use_self_attn=use_self_attn,
                tcr_scale=tcr_scale,
                fuse_scale=fuse_scale,
                gate_bias=gate_bias,
                detach_gate_gene=detach_gate_gene,
            )
            for _ in range(layers)
        ])
        self.proj_gene_out = nn.Linear(d, gene_dim) if gene_dim != d else nn.Identity()
        self.proj_tcr_out = nn.Linear(d, emb_dim)
        self._printed_heads = False

    def forward(
        self,
        gene: Tensor,
        tcr: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Optional[Tensor], Tensor]]:
        gene = self.proj_gene_in(gene)
        tcr = self.proj_tcr_in(tcr)
        if not self._printed_heads:
            print("heads(tcr_to_rna):", getattr(self.layers[0].attn_tcr_to_rna, "num_heads", None))
            self._printed_heads = True

        gate_scores = []
        attn_last = None
        for block in self.layers:
            gene, tcr, gate_l, (_, attn_t2rna) = block(gene, tcr, pad_mask)
            gate_scores.append(gate_l)
            attn_last = attn_t2rna

        gate_score_mean = torch.stack(gate_scores, dim=1).mean(dim=1)
        fused_out = self.proj_gene_out(gene)
        tcr_to_rna = self.proj_tcr_out(tcr)
        return fused_out, tcr_to_rna, gate_score_mean, (None, attn_last)


class CDR3SeqDecoder(nn.Module):
    """
    Autoregressive full CDR3 sequence decoder.

    Given a cell-level memory vector, reconstruct the complete CDR3 sequence using
    teacher forcing. The caller should pass token IDs containing <start>, AA tokens,
    <end>, and <pad>. Loss should be computed against the returned shifted target with
    ignore_index=pad_idx.
    """

    def __init__(
        self,
        memory_dim: int,
        d_model: int = 128,
        vocab_size: int = 24,
        max_len: int = 20,
        pad_idx: int = 0,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.mem_proj = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=True)
        if tie_weights:
            self.out.weight = self.tok_embed.weight

    @staticmethod
    def _causal_mask(t: int, device=None):
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, memory: Tensor, target_ids: Tensor) -> Tuple[Tensor, Tensor]:
        if target_ids.size(1) < 2:
            raise ValueError("target_ids must include at least <start> and one target token")
        dec_in = target_ids[:, :-1]
        dec_tg = target_ids[:, 1:]
        b, t = dec_in.shape
        pos = torch.arange(t, device=dec_in.device).unsqueeze(0).expand(b, t)
        x = self.tok_embed(dec_in) + self.pos_embed(pos)
        mem = self.mem_proj(memory).unsqueeze(1)
        causal = self._causal_mask(t, device=dec_in.device)
        pad_mask = dec_in == self.pad_idx
        h = self.decoder(tgt=x, memory=mem, tgt_mask=causal, tgt_key_padding_mask=pad_mask)
        h = self.norm(h)
        logits = self.out(h)
        return logits, dec_tg
