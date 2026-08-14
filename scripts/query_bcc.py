# -*- coding: utf-8 -*-


import os, re, json, glob, pickle, argparse, warnings
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import seaborn as sns
from matplotlib.lines import Line2D

STANDARD_COLORS = {
    "CD4_T": "#d62728",
    "CD8_T": "#9467bd",
    "T_gdT": "#8c564b",
    "B_lineage": "#17becf",
    "Plasma": "#e377c2",
    "DC": "#7f7f7f",
    "Monocyte": "#bcbd22",
    "Macrophage": "#b15928",
    "Neutrophil": "#1f9e89",
    "NK_ILC": "#6baed6",
    "Mast": "#ffd92f",
}

def get_color_for_label(label: str):

    if label in STANDARD_COLORS:
        return STANDARD_COLORS[label]
    clean_label = label.replace("BCC_", "")
    if clean_label in STANDARD_COLORS:
        return STANDARD_COLORS[clean_label]
    return "#333333"

def get_args():
    p = argparse.ArgumentParser(description="Project BCC dataset onto reference atlas (Styled).")
    p.add_argument("--ref-dir", required=True, help="")
    p.add_argument("--ref-tag", default=None, help="")
    p.add_argument("--bcc-h5ad", required=True, help="BCC h5ad")
    p.add_argument("--bcc-meta", required=True, help="BCC metadata")
    p.add_argument("--cell-id-col", default="cell.id", help="BCC meta cell ID col")
    p.add_argument("--cluster-col", default="cluster", help="BCC meta cluster col")
    p.add_argument("--use-h", action="store_true", help="Use h space")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--dpi", type=int, default=200, help="")
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--bandwidth-scale", type=float, default=0.5)
    p.add_argument("--topk-nn-report", type=int, default=5)
    p.add_argument("--save-dir", required=True)
    return p.parse_args()

args = get_args()
os.makedirs(args.save_dir, exist_ok=True)


def _latest_ckpt_tag(ref_dir: str) -> str:
    pats = sorted(glob.glob(os.path.join(ref_dir, "checkpoint_*.pt")), key=os.path.getmtime)
    if not pats:
        raise FileNotFoundError(f"No checkpoint found in {ref_dir}")
    return re.sub(r"^checkpoint_|\.pt$", "", os.path.basename(pats[-1]))

def load_ref_assets(ref_dir: str, tag: Optional[str] = None) -> Dict:
    if tag is None:
        tag = _latest_ckpt_tag(ref_dir)

    files = {
        "ckpt": f"checkpoint_{tag}.pt",
        "genes": f"genes_hvg_order_{tag}.txt",
        "stats": f"scale_stats_{tag}.npz",
        "pca": f"pca_{tag}.pkl",
        "ref_npz": f"reference_pca_umap_{tag}.npz",
        "proto": f"prototypes_big_{tag}.npz",
        "cfg": f"config_{tag}.json",
    }
    paths = {k: os.path.join(ref_dir, v) for k, v in files.items()}

    state = torch.load(paths["ckpt"], map_location="cpu")
    with open(paths["genes"]) as f:
        ref_genes = [x.strip() for x in f if x.strip()]
    st = np.load(paths["stats"])
    with open(paths["pca"], "rb") as f:
        pca = pickle.load(f)
    ref = np.load(paths["ref_npz"])

    protos, proto_classes = None, None
    if os.path.exists(paths["proto"]):
        pr = np.load(paths["proto"], allow_pickle=False)
        protos, proto_classes = pr["protos"], [str(x) for x in pr["classes"]]

    cfg = {}
    if os.path.exists(paths["cfg"]):
        with open(paths["cfg"]) as f:
            cfg = json.load(f)

    return {
        "tag": tag,
        "ckpt": state,
        "ref_genes": ref_genes,
        "scale_mean": st["mean"],
        "scale_std": st["std"],
        "pca": pca,
        "X_pca_ref": ref["X_pca_full"],
        "U_ref": ref["UMAP_2D"],
        "protos": protos,
        "proto_classes": proto_classes,
        "cfg": cfg,
    }


COMMON_BIG = set(STANDARD_COLORS.keys())

def _prep(label: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(label).strip().lower())

def map_to_big(label: str) -> Optional[str]:
    q = _prep(label)
    if any(k in q for k in ["tumor","caf","fibro","myofibro","endothel","melano","keratino","epi","epithe","stromal"]):
        return None

    if "b_cell" in q: return "B_lineage"
    if "plasma" in q: return "Plasma"
    if "treg" in q or "cd4" in q or "tcell_prolif" in q: return "CD4_T"
    if "cd8" in q: return "CD8_T"
    if "nk" in q or "ilc" in q: return "NK_ILC"
    if "pdc" in q or q.startswith("dc") or re.fullmatch(r"dcs?", q): return "DC"
    if "macrophage" in q: return "Macrophage"
    if "mono" in q: return "Monocyte"
    if "neutro" in q: return "Neutrophil"
    if "mast" in q: return "Mast"
    return None

def add_bcc_prefix(cls: str) -> str:
    return f"BCC_{cls}"


def load_bcc(bcc_h5ad, bcc_meta, cell_id_col, cluster_col):
    print(f"[LOAD] BCC: {bcc_h5ad}")
    ad = sc.read_h5ad(bcc_h5ad)
    ad.var_names_make_unique()

    df = pd.read_csv(bcc_meta, sep=None, engine="python", dtype=str)
    df[cell_id_col] = df[cell_id_col].astype(str)
    df = df.drop_duplicates(subset=[cell_id_col]).set_index(cell_id_col)

    inter = ad.obs_names.isin(df.index)
    ad = ad[inter].copy()
    ad.obs["cluster"] = df.loc[ad.obs_names, cluster_col].values

    ad.obs["ImmuneMajor"] = ad.obs["cluster"].map(map_to_big)
    ad = ad[ad.obs["ImmuneMajor"].isin(COMMON_BIG)].copy()
    ad.obs["ImmuneMajor_BCC"] = ad.obs["ImmuneMajor"].apply(add_bcc_prefix)
    return ad

def build_model(ckpt_state, gene_dim, device, use_h_flag):

    par = ckpt_state.get("args", {})

    mode_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "..", "model")
    )
    if mode_path not in sys.path:
        sys.path.append(mode_path)

    from model import GeneMLPEncoder, RNADecoder

    class RNAMaskedAutoEncoder(nn.Module):
        def __init__(self, gene_dim):
            super().__init__()
            self.encoder = GeneMLPEncoder(
                input_dim=gene_dim,
                out_dim=par.get("enc_n_latents", 256),
                width=par.get("enc_width", 512),
                depth=par.get("enc_depth", 3),
                mult=par.get("enc_mult", 4),
                dropout=par.get("enc_dropout", 0.1),
                use_attention=par.get("enc_use_attn", False),
                n_latents=par.get("enc_n_latents", 256),
                n_heads=par.get("enc_n_heads", 8),
                latent_self_attn=par.get("enc_latent_self_attn", False),
                select_topk=(None if (par.get("enc_topk", 512) <= 0) else par.get("enc_topk", 512)),
                abs_topk=par.get("enc_topk_abs", False),
                detach_gene_embed=par.get("enc_detach_embed", False),
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(par.get("enc_n_latents", 256)),
                nn.Linear(par.get("enc_n_latents", 256), par.get("enc_width", 512)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(par.get("enc_width", 512), 256),
            )
            self.norm = nn.LayerNorm(par.get("enc_n_latents", 256))
            self.mask_token = nn.Parameter(torch.zeros(1, gene_dim))

        @torch.no_grad()
        def encode(self, x, return_h=False):
            z = self.norm(self.encoder(x))
            if return_h:
                return z, self.proj(z)
            return z

    model = RNAMaskedAutoEncoder(gene_dim).to(device)
    model.load_state_dict(ckpt_state["model"], strict=False)
    model.eval()
    return model

def align_scale_matrix(ad, ref_genes, mean, std):
    present = [g for g in ref_genes if g in ad.var_names]
    X_full = np.zeros((ad.n_obs, len(ref_genes)), dtype=np.float32)

    if hasattr(ad.X, "toarray"):
        X_sub = ad[:, present].X.toarray()
    else:
        X_sub = ad[:, present].X

    idx_map = {g: i for i, g in enumerate(ref_genes)}
    cols = [idx_map[g] for g in present]
    X_full[:, cols] = X_sub
    return X_full

@torch.no_grad()
def encode_in_batches(model, X, device, use_h, bs=2048):
    outs = []
    for i in range(0, X.shape[0], bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
        if use_h:
            _, h = model.encode(xb, return_h=True)
            outs.append(h.cpu())
        else:
            z = model.encode(xb)
            outs.append(z.cpu())
    return torch.cat(outs, dim=0).numpy()

def interpolate_umap(X_pca_ref, U_ref, pca, X_lat_q, l2norm, k, bw_scale, clip_q=0.98):
    X_pca_q = pca.transform(X_lat_q)
    if l2norm:
        X_pca_q /= (norm(X_pca_q, axis=1, keepdims=True) + 1e-8)
        X_pca_ref /= (norm(X_pca_ref, axis=1, keepdims=True) + 1e-8)

    k = min(k, X_pca_ref.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_pca_ref)
    dists, idxs = nbrs.kneighbors(X_pca_q)

    sigma = dists[:, -1:] * bw_scale + 1e-12
    if clip_q and 0.5 < clip_q < 1.0:
        thr = np.quantile(dists, clip_q, axis=1, keepdims=True)
        dists = np.minimum(dists, thr)

    w = np.exp(-dists / sigma)
    w /= (w.sum(axis=1, keepdims=True) + 1e-12)
    U_q = np.einsum("nk,nkd->nd", w, U_ref[idxs])
    return X_pca_q, U_q, dists.mean(axis=1)

def cosine_sim(a, b):
    a = a / (norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def plot_overlay_styled(U_ref, U_q, labels_q, save_path, dpi=200):

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    xlim = (U_ref[:, 0].min(), U_ref[:, 0].max())
    ylim = (U_ref[:, 1].min(), U_ref[:, 1].max())

    fig = plt.figure(figsize=(11, 7), dpi=dpi)
    ax = plt.gca()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        U_ref[:, 0], U_ref[:, 1],
        s=0.5,
        c="#E0E0E0",
        alpha=0.3,
        linewidths=0,
        rasterized=True,
        zorder=1
    )

    cats = pd.Series(labels_q).astype(str)
    cats_unique = sorted(cats.unique())

    for c in cats_unique:
        idx = (cats == c).values
        col = get_color_for_label(c)
        ax.scatter(
            U_q[idx, 0], U_q[idx, 1],
            s=2.5,
            c=col,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            zorder=3
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    present = set(cats_unique)
    ordered = [f"BCC_{k}" for k in STANDARD_COLORS.keys() if f"BCC_{k}" in present]
    extras = sorted(list(present - set(ordered)))
    ordered += extras

    handles, labels = [], []
    for c in ordered:
        col = get_color_for_label(c)
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="None",
            markersize=7.0,
            markerfacecolor=col,
            markeredgecolor="none",
            alpha=1.0
        ))
        labels.append(c)


    legend = ax.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        labelspacing=0.45,
        borderaxespad=0.0
    )

    plt.tight_layout()


    plt.savefig(save_path, bbox_inches="tight", format="pdf", dpi=dpi)
    plt.close()
    print(f"[PLOT] Saved styled overlay: {save_path}")

def plot_enhanced_diagnostics(df_res, dataset_tag, ref_tag, save_dir):

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
    plt.rcParams["pdf.compression"] = 9


    if len(df_res["original_label"].unique()) > 0:
        plt.figure(figsize=(12, 7))
        order = sorted(df_res["original_label"].unique())
        palette = {c: get_color_for_label(c) for c in order}

        ax = sns.boxplot(
            data=df_res, x="original_label", y="proto_top1_score",
            order=order, palette=palette, showfliers=False, width=0.6,
            boxprops=dict(alpha=0.7, edgecolor="black", linewidth=1.2)
        )

        sns.stripplot(
            data=df_res, x="original_label", y="proto_top1_score",
            order=order, palette=palette, size=1.8, alpha=0.3, jitter=True, ax=ax
        )

        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.ylabel("Confidence (Top1 Cosine Similarity)", fontsize=12, fontweight="bold")
        plt.title(f"{dataset_tag} Prediction Confidence Distribution", fontsize=14, pad=15)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        outp = os.path.join(save_dir, f"confidence_v2_{dataset_tag}_{ref_tag}.pdf")
        plt.savefig(outp, format="pdf", dpi=300, bbox_inches="tight")
        plt.close()


    plt.figure(figsize=(9, 7))
    hb = plt.hexbin(
        df_res["proto_top1_score"], df_res["proj_uncertainty"],
        gridsize=45, cmap="YlGnBu", bins="log", mincnt=1, alpha=0.85
    )
    cb = plt.colorbar(hb, label="Log10(Cell Count)")
    cb.outline.set_visible(False)

    sns.regplot(
        data=df_res, x="proto_top1_score", y="proj_uncertainty",
        scatter=False, color="#e74c3c", line_kws={"linewidth": 2.5, "linestyle": "--"}
    )

    plt.xlabel("Confidence (Top1 Cosine Similarity)", fontsize=11, fontweight="bold")
    plt.ylabel("Uncertainty (Mean KNN Distance)", fontsize=11, fontweight="bold")
    plt.title(f"Uncertainty Landscape: {dataset_tag}", fontsize=13, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    plt.tight_layout()

    outp = os.path.join(save_dir, f"uncertainty_v2_{dataset_tag}_{ref_tag}.pdf")
    plt.savefig(outp, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


    cm = pd.crosstab(df_res["original_label"], df_res["proto_top1"], normalize="index")
    if cm.size > 0:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues", linewidths=.5,
            cbar_kws={"label": "Proportion"}, annot_kws={"size": 8}
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Alignment Matrix: {dataset_tag} vs Reference Prototypes", fontsize=13)
        plt.tight_layout()

        outp = os.path.join(save_dir, f"alignment_v2_{dataset_tag}_{ref_tag}.pdf")
        plt.savefig(outp, format="pdf", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    dev = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    ref = load_ref_assets(args.ref_dir, args.ref_tag)

    ad_bcc = load_bcc(args.bcc_h5ad, args.bcc_meta, args.cell_id_col, args.cluster_col)

    print("[PP] Preprocessing BCC...")
    sc.pp.normalize_total(ad_bcc, target_sum=1e4)
    sc.pp.log1p(ad_bcc)
    sc.pp.scale(ad_bcc, max_value=10)

    X_full = align_scale_matrix(ad_bcc, ref["ref_genes"], ref["scale_mean"], ref["scale_std"])

    use_h = args.use_h if args.use_h else ref["cfg"].get("eval_with_h", True)
    model = build_model(ref["ckpt"], len(ref["ref_genes"]), dev, use_h)

    print("[Inference] Encoding...")
    X_lat_q = encode_in_batches(model, X_full, dev, use_h, args.batch_size)

    print("[Projection] Interpolating UMAP...")
    l2 = ref["cfg"].get("pca_l2norm", False)
    _, U_q, unc = interpolate_umap(
        ref["X_pca_ref"], ref["U_ref"], ref["pca"], X_lat_q,
        l2, args.knn, args.bandwidth_scale
    )


    pdf_path = os.path.join(args.save_dir, f"BCC_Pure_Projection_{ref['tag']}.pdf")
    plot_overlay_styled(ref["U_ref"], U_q, ad_bcc.obs["ImmuneMajor_BCC"], pdf_path, dpi=args.dpi)

    print("[Report] Calculating prediction results...")
    top1_labels, top1_scores = ["NA"] * ad_bcc.n_obs, np.zeros(ad_bcc.n_obs, dtype=float)

    if ref["protos"] is not None:
        S = cosine_sim(X_lat_q, ref["protos"])
        top1_idx = np.argmax(S, axis=1)
        top1_labels = [ref["proto_classes"][i] for i in top1_idx]
        top1_scores = S[np.arange(S.shape[0]), top1_idx]

    df_res = pd.DataFrame({
        "cell_id": ad_bcc.obs_names,
        "original_label": ad_bcc.obs["ImmuneMajor_BCC"],
        "proto_top1": top1_labels,
        "proto_top1_score": top1_scores,
        "proj_uncertainty": unc
    })


    print("[Report] Plotting enhanced diagnostics (PDF)...")
    plot_enhanced_diagnostics(df_res, "BCC", ref["tag"], args.save_dir)

    csv_path = os.path.join(args.save_dir, f"Combined_BCC_query_results_{ref['tag']}.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"[DONE] Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
