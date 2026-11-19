#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUAD RNA-defined state -> fused GEX+TCR substate analysis.
python -m immunelens.analysis.luad_subcluster_analysis \
    --h5ad results/luadvae/adata_nsclc_icb.h5ad \
    --out_dir ./result/luadp2 \
    --rna_cluster 5 \
    --focus_fines 12,19,6,9,24,10,27 \
    --nonmpr_fines 12,19 \
    --mpr_fines 9

Main figure logic
-----------------
1) GEX-only UMAP with RNA clusters.
2) Compare CXCL13/PDCD1/TOX/TIGIT/ENTPD1/GZMB expression across ALL RNA
   clusters and highlight the pre-selected RNA cluster (default: cluster 5).
3) Show the global fused GEX+TCR UMAP.
4) Show how cells from the selected RNA cluster map into globally defined
   fused fine clusters (Sankey/alluvial).
5) Test patient-level clinical-response association of those fused states.
6) Show TCR/clonotype structure captured by the fused representation.
7) For clinically associated fused states, show transcriptional differences
   with a DEG dotplot.

Important interpretation
------------------------
- The selected RNA cluster is NOT re-clustered for the main analysis.
- Fused states are the GLOBAL fine_cluster labels restricted to cells from the
  selected RNA cluster. This directly asks whether a single RNA-defined state
  is resolved across multiple globally defined fused states.
- Six predefined tumor-reactive / exhaustion-associated markers are shown
  transparently across RNA clusters instead of collapsed into a composite score.
- Clinical association is tested at the PATIENT/SAMPLE level.
- The TCR panel is best interpreted as a representation sanity check/supporting
  analysis, because TCR information contributes to the fused representation.
- The final transcriptional panel is exploratory marker characterization.
  F12 vs F9 and F19 vs F9 are analyzed separately rather than pooling F12/F19.

Example
-------
python -m immunelens.analysis.luad_subcluster_analysis \
    --h5ad results/luadvae/adata_nsclc_icb.h5ad \
    --out_dir ./result/luad_fused_substate \
    --rna_cluster 5 \
    --focus_fines 12,19,6,9,24,10,27 \
    --nonmpr_fines 12,19 \
    --mpr_fines 9

If --focus_fines is omitted, the top N fused fine clusters by cell count within
selected RNA cluster are used (default N=7).
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path


# =============================================================================
# Configuration
# =============================================================================

RNA_CLUSTER_COL = "RNA_cluster"
FINE_COL = "fine_cluster"
SAMPLE_COL = "sampleID"
GROUP_COL = "Response_binary"
TRB_COL = "TRB_aa"

RNA_REP = "X_rna"
FUSED_REP = "X_final"

POS_GROUP = "MPR"
NEG_GROUP = "non-MPR"

ANCHOR_GENES = ["CXCL13", "PDCD1", "TOX", "TIGIT", "ENTPD1", "GZMB"]

N_NEIGHBORS = 30
RANDOM_SEED = 1337

N_DEG_PER_DIRECTION = 5
MIN_MARKER_LOGFC = 0.25
MIN_MARKER_PCT = 0.15
MIN_MARKER_PCT_DIFF = 0.10
MARKER_FDR = 0.01

INK = "#1A1A1A"
RNA_COLOR = "#A6B1BC"
FUSED_COLOR = "#9E2B25"
MPR_COLOR = "#2E6E8E"
NONMPR_COLOR = "#B2483B"
OTHER_COLOR = "#D8D8D8"

FOCUS_PALETTE = [
    "#A24B57", "#D58A4C", "#5E8C61", "#4C6E91",
    "#7A5CA6", "#3F8CA6", "#B78B3D", "#8A6E5A",
    "#5B7C99", "#8F6AA8", "#6A8F7B", "#B46B5B",
]


# =============================================================================
# Helpers
# =============================================================================

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_pdf(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def clean_series(s):
    return s.astype(str).replace({
        "nan": "NA", "NaN": "NA", "None": "NA", "": "NA",
    })


def natural_sort(values):
    def key(x):
        x = str(x)
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    return sorted([str(x) for x in values], key=key)


def parse_csv_list(x):
    if x is None:
        return None
    vals = [v.strip() for v in str(x).split(",") if v.strip()]
    return vals if vals else None


def despine(ax, keep=("left", "bottom")):
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(s in keep)


def ensure_response_binary(ad):
    if GROUP_COL in ad.obs.columns:
        ad.obs[GROUP_COL] = clean_series(ad.obs[GROUP_COL])
        return ad

    if "Response" not in ad.obs.columns:
        raise ValueError("Need Response_binary or Response in ad.obs")

    ad.obs[GROUP_COL] = ad.obs["Response"].astype(str).map(
        lambda x: "MPR" if x in {"MPR", "pCR"}
        else ("non-MPR" if x == "non-MPR" else "NA")
    )
    return ad


def get_matrix_and_names(ad):
    if ad.raw is not None:
        return ad.raw.X, pd.Index(ad.raw.var_names.astype(str))
    return ad.X, pd.Index(ad.var_names.astype(str))


def gene_vector(ad, gene):
    X, names = get_matrix_and_names(ad)
    if gene not in names:
        return np.full(ad.n_obs, np.nan)
    v = X[:, names.get_loc(gene)]
    if sparse.issparse(v):
        v = v.toarray()
    return np.asarray(v).reshape(-1)


def expression_df(ad, genes):
    return pd.DataFrame(
        {g: gene_vector(ad, g) for g in genes},
        index=ad.obs_names,
    )


def get_umap(ad, rep_key, preferred_key=None):
    if preferred_key and preferred_key in ad.obsm:
        print(f"[UMAP] using ad.obsm['{preferred_key}']")
        return np.asarray(ad.obsm[preferred_key])

    aw = ad.copy()
    sc.pp.neighbors(
        aw, use_rep=rep_key, n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_SEED,
    )
    sc.tl.umap(aw, random_state=RANDOM_SEED)
    return np.asarray(aw.obsm["X_umap"])


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR, preserving NaNs."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not np.any(ok):
        return q

    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    tmp = np.empty(m, dtype=float)
    tmp[order] = adj
    q[ok] = tmp
    return q


def q_to_star(q):
    if not np.isfinite(q):
        return "NA"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def build_focus_colors(focus_fines):
    return {
        f: FOCUS_PALETTE[i % len(FOCUS_PALETTE)]
        for i, f in enumerate(focus_fines)
    }


def choose_focus_fines(ad_rna, requested=None, top_n=7):
    fine = clean_series(ad_rna.obs[FINE_COL])
    if requested:
        missing = [x for x in requested if x not in set(fine)]
        if missing:
            raise ValueError(
                f"Requested focus fused clusters are absent from selected "
                f"RNA cluster: {missing}"
            )
        return requested

    counts = fine.value_counts()
    vals = [str(x) for x in counts.head(top_n).index]
    print(f"[Focus fused states] auto top {top_n}: {vals}")
    return vals


def is_unwanted_marker(g):
    g = str(g).upper().strip()
    prefixes = (
        "TRAV", "TRBV", "TRAJ", "TRBJ",
        "TRGV", "TRGJ", "TRDV", "TRDJ",
        "IGH", "IGK", "IGL",
        "RPL", "RPS", "MT-",
    )
    if g.startswith(prefixes):
        return True
    if g in {
        "SCGB1A1", "SCGB3A1", "SFTPC",
        "ACTB", "GAPDH", "B2M", "MALAT1",
        "EEF1A1", "EEF1B2", "TMSB10", "TMSB4X",
        "FTL", "FTH1", "UBC", "KIAA1551",
    }:
        return True
    if g.startswith(("AL", "AC", "LINC")):
        return True
    return False


# =============================================================================
# Figure 1: GEX-only RNA clusters
# =============================================================================

def plot_rna_clusters(ad, coords, selected_rna, out_dir):
    obs = clean_series(ad.obs[RNA_CLUSTER_COL])
    labels = natural_sort(obs.unique())

    cmap = plt.get_cmap("tab20")
    colors = {lab: cmap(i % 20) for i, lab in enumerate(labels)}
    colors[selected_rna] = FUSED_COLOR

    fig, ax = plt.subplots(figsize=(5.1, 4.2))

    # Plot non-selected first, selected last.
    order = [x for x in labels if x != selected_rna] + [selected_rna]
    for lab in order:
        m = obs.values == lab
        is_selected = lab == selected_rna
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=3.2 if is_selected else 2.0,
            c=[colors[lab]],
            alpha=0.95 if is_selected else 0.60,
            linewidths=0,
            rasterized=True,
        )
        if m.sum():
            ax.text(
                np.median(coords[m, 0]), np.median(coords[m, 1]),
                lab,
                ha="center", va="center",
                fontsize=8 if is_selected else 7,
                fontweight="bold" if is_selected else "normal",
            )

    ax.set_title("GEX-only RNA clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    save_pdf(fig, out_dir, "01_GEX_only_RNA_clusters")


# =============================================================================
# Figure 2: CXCL13 / PDCD1 across ALL RNA clusters
# =============================================================================

def plot_anchor_gene_violins(
    ad,
    selected_rna,
    out_dir,
    max_cells_per_cluster=5000,
):
    """
    Compact heatmap of selected marker expression across RNA clusters.

    Rows:
        CXCL13, PDCD1, TOX, TIGIT, ENTPD1, GZMB

    Columns:
        RNA clusters

    Color:
        gene-wise scaled mean expression across clusters
        (0 = lowest cluster for that gene, 1 = highest)

    selected_rna:
        highlighted with a rectangle.
    """

    genes = [
        "CXCL13",
        "PDCD1",
        "TOX",
        "TIGIT",
        "ENTPD1",
        "GZMB",
    ]

    obs = ad.obs.copy()
    obs[RNA_CLUSTER_COL] = clean_series(obs[RNA_CLUSTER_COL])

    clusters = sorted(
        obs[RNA_CLUSTER_COL].unique(),
        key=lambda x: int(x) if str(x).isdigit() else 999999
    )

    # ---------------------------------------------------------
    # Expression matrix
    # ---------------------------------------------------------
    ex = expression_df(ad, genes)

    mean_mat = np.zeros(
        (len(genes), len(clusters)),
        dtype=float
    )

    for j, cl in enumerate(clusters):

        idx = obs.index[
            obs[RNA_CLUSTER_COL].astype(str).values == str(cl)
        ]

        for i, gene in enumerate(genes):

            vals = pd.to_numeric(
                ex.loc[idx, gene],
                errors="coerce"
            ).fillna(0).values

            if len(vals) == 0:
                mean_mat[i, j] = np.nan
            else:
                mean_mat[i, j] = float(np.mean(vals))

    # ---------------------------------------------------------
    # Gene-wise scaling across clusters
    #
    # Each gene is scaled independently to [0, 1].
    # This is only for visualization.
    # ---------------------------------------------------------
    scaled = np.zeros_like(mean_mat)

    for i in range(len(genes)):

        vals = mean_mat[i, :]

        finite = np.isfinite(vals)

        if finite.sum() == 0:
            scaled[i, :] = 0.0
            continue

        lo = np.nanmin(vals)
        hi = np.nanmax(vals)

        if abs(hi - lo) < 1e-12:
            scaled[i, :] = 0.5
        else:
            scaled[i, :] = (
                vals - lo
            ) / (
                hi - lo
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            max(6.5, 0.27 * len(clusters) + 1.5),
            2.5
        )
    )

    im = ax.imshow(
        scaled,
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------
    ax.set_xticks(
        np.arange(len(clusters))
    )

    ax.set_xticklabels(
        clusters,
        fontsize=7
    )

    ax.set_yticks(
        np.arange(len(genes))
    )

    ax.set_yticklabels(
        genes,
        fontsize=8
    )

    ax.set_xlabel(
        "RNA cluster"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Tumor-reactive / exhaustion-associated markers"
    )

    ax.tick_params(
        length=0
    )

    # ---------------------------------------------------------
    # Thin white cell borders
    # ---------------------------------------------------------
    ax.set_xticks(
        np.arange(-0.5, len(clusters), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(genes), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.6
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # ---------------------------------------------------------
    # Highlight selected RNA cluster
    # ---------------------------------------------------------
    selected_rna = str(selected_rna)

    if selected_rna in clusters:

        xpos = clusters.index(selected_rna)

        rect = Rectangle(
            (xpos - 0.5, -0.5),
            1.0,
            len(genes),
            fill=False,
            edgecolor=FUSED_COLOR,
            linewidth=1.8,
            clip_on=False,
        )

        ax.add_patch(rect)

    # ---------------------------------------------------------
    # Colorbar
    # ---------------------------------------------------------
    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.025
    )

    cb.set_label(
        "Relative mean expression"
    )

    cb.set_ticks(
        [0, 0.5, 1]
    )

    cb.set_ticklabels(
        ["Low", "Mid", "High"]
    )

    # ---------------------------------------------------------
    # Clean style
    # ---------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_pdf(
        fig,
        out_dir,
        "02_marker_heatmap_across_RNA_clusters"
    )
# =============================================================================
# Figure 3: global fused GEX+TCR UMAP
# =============================================================================

def plot_fused_umap(ad, coords, selected_rna, focus_fines, focus_colors, out_dir):
    fine = clean_series(ad.obs[FINE_COL])
    labels = natural_sort(fine.unique())
    selected_mask = clean_series(ad.obs[RNA_CLUSTER_COL]).values == selected_rna

    fig, ax = plt.subplots(figsize=(5.6, 4.5))

    # Background: all global fused states in light gray.
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=1.6, c=OTHER_COLOR, alpha=0.45,
        linewidths=0, rasterized=True,
    )

    # Highlight focus global fused states.
    for lab in focus_fines:
        m = fine.values == lab
        if not np.any(m):
            continue
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=2.8, c=[focus_colors[lab]], alpha=0.80,
            linewidths=0, rasterized=True,
        )
        ax.text(
            np.median(coords[m, 0]), np.median(coords[m, 1]),
            f"F{lab}",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
        )

    # Outline cells belonging to the selected RNA state.
    ax.scatter(
        coords[selected_mask, 0], coords[selected_mask, 1],
        s=4.2,
        facecolors="none", edgecolors=INK,
        linewidths=0.18, alpha=0.32,
        rasterized=True,
    )

    ax.set_title(
        f"Global fused GEX+TCR states\n"
        f"RNA cluster {selected_rna} cells outlined"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5,
               markerfacecolor=focus_colors[f], markeredgecolor="none",
               label=f"F{f}")
        for f in focus_fines
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left", title="Focus fused states")

    save_pdf(fig, out_dir, "03_Global_fused_UMAP")


# =============================================================================
# Figure 4: Sankey / alluvial RNA state -> global fused states
# =============================================================================

def stack_segments(items, top=0.90, bottom=0.10, gap=0.02):
    items = [x for x in items if x[1] > 0]
    total = sum(v for _, v in items)
    usable = top - bottom - gap * max(0, len(items) - 1)
    out = {}
    cur = top
    for name, value in items:
        h = usable * value / total if total > 0 else 0
        out[name] = (cur, cur - h)
        cur = cur - h - gap
    return out


def ribbon_path(x0, x1, y0t, y0b, y1t, y1b):
    dx = x1 - x0
    c0 = x0 + 0.46 * dx
    c1 = x1 - 0.46 * dx
    verts = [
        (x0, y0t), (c0, y0t), (c1, y1t), (x1, y1t),
        (x1, y1b), (c1, y1b), (c0, y0b), (x0, y0b),
        (x0, y0t),
    ]
    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def plot_sankey(ad_rna, selected_rna, focus_fines, focus_colors, out_dir):
    fine = clean_series(ad_rna.obs[FINE_COL])
    counts = fine.value_counts()
    focus_counts = {f: int(counts.get(f, 0)) for f in focus_fines}
    total = len(ad_rna)
    other = total - sum(focus_counts.values())

    right_items = [(f, focus_counts[f]) for f in focus_fines]
    if other > 0:
        right_items.append(("Other", other))

    right_pos = stack_segments(right_items, gap=0.02)
    left_stack = stack_segments(right_items, top=0.90, bottom=0.10, gap=0.02)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    xl, xr = 0.15, 0.72
    w = 0.035

    ax.add_patch(Rectangle(
        (xl - w / 2, 0.10), w, 0.80,
        facecolor=RNA_COLOR, edgecolor="white",
    ))

    for f, value in right_items:
        if value <= 0:
            continue
        st, sb = left_stack[f]
        rt, rb = right_pos[f]

        fc = OTHER_COLOR if f == "Other" else focus_colors[f]
        ax.add_patch(PathPatch(
            ribbon_path(xl + w / 2, xr - w / 2, st, sb, rt, rb),
            facecolor=fc, alpha=0.42, edgecolor="none",
        ))
        ax.add_patch(Rectangle(
            (xr - w / 2, rb), w, rt - rb,
            facecolor=fc, edgecolor="white",
        ))
        label = "Other" if f == "Other" else f"F{f}"
        ax.text(
            xr + 0.045, (rt + rb) / 2,
            f"{label}\n{value:,} ({100 * value / total:.1f}%)",
            ha="left", va="center", fontsize=8,
        )

    ax.text(xl, 0.94, f"RNA cluster {selected_rna}",
            ha="center", fontweight="bold")
    ax.text(xl, 0.04, "GEX-only state",
            ha="center", fontsize=7, color="#555555")
    ax.text(xr, 0.04, "Global fused states",
            ha="center", fontsize=7, color="#555555")

    ax.set_xlim(0.02, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Fused partitioning of the selected RNA-defined state")

    save_pdf(fig, out_dir, "04_RNA_to_fused_Sankey")


# =============================================================================
# Figure 5: patient-level clinical response association
# =============================================================================

def patient_fraction_table(ad_rna, focus_fines):
    required = [SAMPLE_COL, GROUP_COL, FINE_COL]
    for c in required:
        if c not in ad_rna.obs.columns:
            raise ValueError(f"Missing ad.obs['{c}']")

    o = ad_rna.obs[required].copy()
    o[FINE_COL] = clean_series(o[FINE_COL])
    o[GROUP_COL] = clean_series(o[GROUP_COL])
    o[SAMPLE_COL] = clean_series(o[SAMPLE_COL])
    o = o[o[GROUP_COL].isin([POS_GROUP, NEG_GROUP])].copy()

    if o.empty:
        raise ValueError("No MPR/non-MPR cells in selected RNA state.")

    # Each patient contributes a composition vector that sums to 1 across ALL
    # fused states within the selected RNA state. We then extract focus states.
    comp = pd.crosstab(o[SAMPLE_COL], o[FINE_COL], normalize="index")
    for f in focus_fines:
        if f not in comp.columns:
            comp[f] = 0.0

    # Assumes each sample has one response label.
    grp_check = o.groupby(SAMPLE_COL)[GROUP_COL].nunique()
    if (grp_check > 1).any():
        bad = grp_check[grp_check > 1].index.tolist()
        raise ValueError(f"Samples with inconsistent response labels: {bad[:5]}")

    group = o.drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL)[GROUP_COL]

    long = comp[focus_fines].reset_index().melt(
        id_vars=SAMPLE_COL,
        var_name="cluster",
        value_name="fraction",
    )
    return long.merge(
        group.rename("response"),
        left_on=SAMPLE_COL,
        right_index=True,
        how="left",
    )


def plot_response(ad_rna, focus_fines, out_dir):
    long = patient_fraction_table(ad_rna, focus_fines)
    rng = np.random.default_rng(RANDOM_SEED)

    tests = []
    for cl in focus_fines:
        d = long[long["cluster"] == cl]
        neg = d.loc[d["response"] == NEG_GROUP, "fraction"].dropna().values
        pos = d.loc[d["response"] == POS_GROUP, "fraction"].dropna().values
        p = np.nan
        if len(neg) >= 2 and len(pos) >= 2:
            _, p = mannwhitneyu(neg, pos, alternative="two-sided")
        tests.append({
            "cluster": cl,
            "p": p,
            "median_nonMPR": np.nanmedian(neg) if len(neg) else np.nan,
            "median_MPR": np.nanmedian(pos) if len(pos) else np.nan,
            "n_nonMPR": len(neg),
            "n_MPR": len(pos),
        })

    test_df = pd.DataFrame(tests)
    test_df["q"] = bh_fdr(test_df["p"].values)
    test_df["star"] = [q_to_star(q) for q in test_df["q"]]

    print("\n[Clinical association: patient-level fractions]")
    print(test_df.to_string(index=False))

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.88 * len(focus_fines) + 1.0), 3.8)
    )

    ymax = 0.0
    cluster_max = {}

    for i, cl in enumerate(focus_fines):
        d = long[long["cluster"] == cl]
        neg = d.loc[d["response"] == NEG_GROUP, "fraction"].dropna().values
        pos = d.loc[d["response"] == POS_GROUP, "fraction"].dropna().values

        vals_all = np.concatenate([neg, pos]) if len(neg) + len(pos) else np.array([0])
        cluster_max[cl] = float(np.nanmax(vals_all)) if len(vals_all) else 0.0

        for vals, xpos, col in [
            (neg, i - 0.18, NONMPR_COLOR),
            (pos, i + 0.18, MPR_COLOR),
        ]:
            if len(vals) == 0:
                continue
            ax.boxplot(
                vals,
                positions=[xpos],
                widths=0.30,
                patch_artist=True,
                showfliers=False,
                showcaps=False,
                boxprops=dict(facecolor=col, alpha=0.55,
                              edgecolor=INK, linewidth=0.6),
                medianprops=dict(color=INK, linewidth=0.8),
                whiskerprops=dict(color=INK, linewidth=0.6),
            )
            ax.scatter(
                xpos + rng.uniform(-0.055, 0.055, len(vals)),
                vals,
                s=11,
                c=col,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            ymax = max(ymax, float(np.nanmax(vals)))

    # FDR stars across tested fused states.
    for i, cl in enumerate(focus_fines):
        row = test_df[test_df["cluster"] == cl].iloc[0]
        y = cluster_max.get(cl, 0) + max(0.015, ymax * 0.04)
        ax.text(i, y, row["star"], ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(focus_fines)))
    ax.set_xticklabels([f"F{x}" for x in focus_fines])
    ax.set_ylabel("Per-patient fraction\nwithin RNA cluster")
    ax.set_ylim(0, ymax * 1.25 + 0.05)
    ax.set_title("Clinical response association of fused substates")

    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="",
               markerfacecolor=NONMPR_COLOR, markeredgecolor="white",
               label=NEG_GROUP),
        Line2D([], [], marker="s", linestyle="",
               markerfacecolor=MPR_COLOR, markeredgecolor="white",
               label=POS_GROUP),
    ], frameon=False, loc="upper right")

    ax.text(
        0.01, 0.98, "Stars: BH-FDR across displayed fused states",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color="#555555",
    )

    despine(ax)
    save_pdf(fig, out_dir, "05_Clinical_response_association")

    return test_df

def plot_response_3group_stacked(
    ad_rna,
    focus_fines,
    out_dir,
):
    """
    Stacked barplot of pathological response composition
    within each fused substate.

    Groups:
        non-MPR
        MPR
        pCR

    Each bar sums to 1 within a fused cluster.
    """

    if "Response" not in ad_rna.obs.columns:
        raise ValueError(
            "Need ad.obs['Response'] with non-MPR / MPR / pCR labels."
        )

    o = ad_rna.obs[
        [FINE_COL, "Response"]
    ].copy()

    o[FINE_COL] = clean_series(o[FINE_COL])
    o["Response"] = clean_series(o["Response"])

    valid_groups = [
        "non-MPR",
        "MPR",
        "pCR",
    ]

    o = o[
        o[FINE_COL].isin(focus_fines)
        &
        o["Response"].isin(valid_groups)
    ].copy()

    if o.empty:
        raise ValueError(
            "No cells with non-MPR / MPR / pCR labels "
            "in selected fused states."
        )

    # ---------------------------------------------------------
    # Composition within each fused cluster
    # ---------------------------------------------------------
    tab = pd.crosstab(
        o[FINE_COL],
        o["Response"],
        normalize="index",
    )

    for g in valid_groups:
        if g not in tab.columns:
            tab[g] = 0.0

    tab = tab.reindex(
        focus_fines
    ).fillna(0.0)

    tab = tab[
        valid_groups
    ]

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------
    colors = {
        "non-MPR": "#B2483B",
        "MPR": "#6A8FA5",
        "pCR": "#244F73",
    }

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            max(
                5.6,
                0.75 * len(focus_fines) + 1.6
            ),
            3.8
        )
    )

    x = np.arange(
        len(focus_fines)
    )

    bottom = np.zeros(
        len(focus_fines)
    )

    for g in valid_groups:

        vals = tab[g].values

        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.68,
            color=colors[g],
            edgecolor="white",
            linewidth=0.6,
            label=g,
        )

        # percentage labels
        for i, v in enumerate(vals):

            if v >= 0.06:
                ax.text(
                    i,
                    bottom[i] + v / 2,
                    f"{100 * v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )

        bottom += vals

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"F{f}"
            for f in focus_fines
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Fraction of cells\nwithin fused state"
    )

    ax.set_title(
        "Pathological response composition of fused substates"
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        title="Response",
    )

    despine(ax)

    save_pdf(
        fig,
        out_dir,
        "05b_Pathological_response_3group_stacked"
    )
# =============================================================================
# Figure 6: clonotype / TRB structure
# =============================================================================

def recluster_to_n(ad_sub, rep_key, target_n):
    aw = ad_sub.copy()
    n_nb = min(N_NEIGHBORS, max(5, aw.n_obs // 10))
    n_nb = min(n_nb, max(2, aw.n_obs - 1))

    sc.pp.neighbors(
        aw,
        use_rep=rep_key,
        n_neighbors=n_nb,
        random_state=RANDOM_SEED,
    )

    best = None
    for res in np.linspace(0.05, 3.0, 80):
        key = "tmp_cluster"
        try:
            sc.tl.leiden(
                aw, resolution=float(res), key_added=key,
                random_state=RANDOM_SEED,
            )
        except Exception:
            sc.tl.louvain(
                aw, resolution=float(res), key_added=key,
                random_state=RANDOM_SEED,
            )

        labels = aw.obs[key].astype(str).values
        n = len(np.unique(labels))
        item = (abs(n - target_n), labels.copy(), n)
        if best is None or item[0] < best[0]:
            best = item
        if n == target_n:
            return labels

    if best is None:
        raise RuntimeError("Could not recluster RNA representation.")
    return best[1]


def same_trb_fraction(X, obs, k=N_NEIGHBORS):
    trb = clean_series(obs[TRB_COL])
    valid = trb != "NA"

    X = np.asarray(X)[valid.values]
    trb = trb.loc[valid].values

    if len(trb) < 10:
        return np.nan

    kk = min(k + 1, len(trb))
    if kk <= 1:
        return np.nan

    nn = (
        NearestNeighbors(n_neighbors=kk)
        .fit(X)
        .kneighbors(X, return_distance=False)[:, 1:]
    )

    vals = [np.mean(trb[idx] == trb[i]) for i, idx in enumerate(nn)]
    return float(np.mean(vals))


def plot_trb_structure(ad_focus, focus_fines, out_dir):
    if TRB_COL not in ad_focus.obs.columns:
        raise ValueError(f"Missing ad.obs['{TRB_COL}']")

    o = ad_focus.obs.copy()
    trb = clean_series(o[TRB_COL])
    clone_size = trb.map(trb.value_counts()).fillna(1)
    use = (trb != "NA") & (clone_size >= 2)

    n_target = len(focus_fines)
    labels_rna = recluster_to_n(ad_focus, RNA_REP, n_target)
    labels_fused = clean_series(ad_focus.obs[FINE_COL]).values

    if use.sum() >= 10:
        ami_rna = adjusted_mutual_info_score(
            trb[use], labels_rna[use.values]
        )
        ami_fused = adjusted_mutual_info_score(
            trb[use], labels_fused[use.values]
        )
    else:
        ami_rna = np.nan
        ami_fused = np.nan

    nn_rna = same_trb_fraction(ad_focus.obsm[RNA_REP], o)
    nn_fused = same_trb_fraction(ad_focus.obsm[FUSED_REP], o)

    print("\n[TRB structure]")
    print(f"AMI RNA matched clustering: {ami_rna}")
    print(f"AMI fused states:          {ami_fused}")
    print(f"Same-TRB {N_NEIGHBORS}-NN RNA:   {nn_rna}")
    print(f"Same-TRB {N_NEIGHBORS}-NN fused: {nn_fused}")

    fig = plt.figure(figsize=(5.6, 2.9))
    gs = fig.add_gridspec(1, 2, wspace=0.48)

    ax1 = fig.add_subplot(gs[0, 0])
    vals = [ami_rna, ami_fused]
    plot_vals = [0 if not np.isfinite(v) else v for v in vals]
    ax1.bar([0, 1], plot_vals,
            color=[RNA_COLOR, FUSED_COLOR],
            width=0.62, edgecolor="white")
    for x, v in enumerate(vals):
        ax1.text(x, plot_vals[x] + 0.01,
                 "NA" if not np.isfinite(v) else f"{v:.2f}",
                 ha="center")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["RNA-only\nmatched", "Fused\nstates"])
    ax1.set_ylabel("AMI with TRB clone")
    ax1.set_title(f"TRB agreement\n({n_target}-state granularity)")
    ax1.set_ylim(0, max(plot_vals + [0.1]) * 1.30)
    despine(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = [nn_rna, nn_fused]
    plot_vals2 = [0 if not np.isfinite(v) else v for v in vals2]
    ax2.bar([0, 1], plot_vals2,
            color=[RNA_COLOR, FUSED_COLOR],
            width=0.62, edgecolor="white")
    for x, v in enumerate(vals2):
        ax2.text(x, plot_vals2[x] + 0.01,
                 "NA" if not np.isfinite(v) else f"{v:.2f}",
                 ha="center")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["RNA-only\nembedding", "Fused\nembedding"])
    ax2.set_ylabel(f"Same-TRB fraction\namong {N_NEIGHBORS}-NN")
    ax2.set_title("TRB neighborhood")
    ax2.set_ylim(0, max(plot_vals2 + [0.1]) * 1.30)
    despine(ax2)

    fig.suptitle("TCR structure within the selected RNA-defined state", y=1.03)
    save_pdf(fig, out_dir, "06_TRB_structure")


# =============================================================================
# Figure 7: pairwise transcriptional characterization
#             F12 vs F9 and F19 vs F9 (or CLI-specified states)
# =============================================================================

def _pairwise_marker_table(
    ad_focus,
    query_cluster,
    ref_cluster,
    n_per_direction=N_DEG_PER_DIRECTION,
):
    """
    Cell-level exploratory marker analysis for one fused-state pair.

    Returns top markers enriched in query_cluster and top markers enriched in
    ref_cluster. This is intended for transcriptional characterization, not
    patient-level inferential DEG.
    """
    query_cluster = str(query_cluster)
    ref_cluster = str(ref_cluster)

    ad = ad_focus.copy()
    ad.obs["focus_cluster"] = clean_series(ad.obs[FINE_COL])
    ad = ad[
        ad.obs["focus_cluster"].isin([query_cluster, ref_cluster]).values
    ].copy()

    if ad.n_obs == 0:
        raise ValueError(
            f"No cells remain for F{query_cluster} vs F{ref_cluster}."
        )

    vc = ad.obs["focus_cluster"].value_counts()
    if vc.get(query_cluster, 0) < 2 or vc.get(ref_cluster, 0) < 2:
        raise ValueError(
            f"Too few cells for F{query_cluster} vs F{ref_cluster}: "
            f"{vc.to_dict()}"
        )

    ad.obs["focus_cluster"] = ad.obs["focus_cluster"].astype("category")
    use_raw = ad.raw is not None
    key = f"deg_F{query_cluster}_vs_F{ref_cluster}"

    sc.tl.rank_genes_groups(
        ad,
        groupby="focus_cluster",
        groups=[query_cluster, ref_cluster],
        reference="rest",
        method="wilcoxon",
        use_raw=use_raw,
        pts=True,
        key_added=key,
    )

    selected = {}
    detail_tables = {}

    for target, other in [(query_cluster, ref_cluster), (ref_cluster, query_cluster)]:
        df = sc.get.rank_genes_groups_df(
            ad,
            group=target,
            key=key,
        ).copy()

        df = df.rename(columns={
            "names": "gene",
            "logfoldchanges": "logFC",
            "pvals_adj": "padj",
        })
        df["gene"] = df["gene"].astype(str)
        df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
        df["padj"] = pd.to_numeric(df["padj"], errors="coerce")

        df = df[~df["gene"].apply(is_unwanted_marker)].copy()
        df = df[
            (df["padj"] < MARKER_FDR)
            & (df["logFC"] >= MIN_MARKER_LOGFC)
        ].copy()

        target_mask = ad.obs["focus_cluster"].astype(str).values == target
        rows = []

        for gene in df["gene"].head(800):
            v = gene_vector(ad, gene)
            if np.all(~np.isfinite(v)):
                continue

            pct_in = float(np.mean(v[target_mask] > 0))
            pct_out = float(np.mean(v[~target_mask] > 0))

            if pct_in < MIN_MARKER_PCT:
                continue
            if pct_in - pct_out < MIN_MARKER_PCT_DIFF:
                continue

            row = df.loc[df["gene"] == gene].iloc[0]
            logfc = float(row["logFC"])
            score = logfc * max(pct_in - pct_out, 1e-6)

            rows.append({
                "gene": gene,
                "target": target,
                "other": other,
                "logFC": logfc,
                "padj": float(row["padj"]),
                "pct_in": pct_in,
                "pct_out": pct_out,
                "pct_diff": pct_in - pct_out,
                "score": score,
            })

        d = pd.DataFrame(rows)
        if d.empty:
            selected[target] = []
            detail_tables[target] = d
            print(
                f"[Pairwise DEG] F{target} vs F{other}: "
                "no markers passed filters"
            )
            continue

        d = d.sort_values(
            ["score", "logFC", "padj"],
            ascending=[False, False, True],
        )
        detail_tables[target] = d
        selected[target] = d["gene"].head(n_per_direction).tolist()
        print(
            f"[Pairwise DEG] F{target} enriched vs F{other}: "
            + ", ".join(selected[target])
        )

    return selected, detail_tables


def select_pairwise_response_markers(
    ad_focus,
    nonmpr_fines,
    mpr_fines,
    n_per_direction=N_DEG_PER_DIRECTION,
):
    """
    Run separate pairwise comparisons of each non-MPR-associated fused state
    against the MPR-associated reference state.

    Default CLI values produce:
        F12 vs F9
        F19 vs F9

    The output genes are the union of top markers from each pairwise comparison,
    with duplicate genes removed while preserving order.
    """
    if not nonmpr_fines or not mpr_fines:
        raise ValueError(
            "Need non-empty --nonmpr_fines and --mpr_fines for pairwise marker analysis."
        )

    if len(mpr_fines) != 1:
        raise ValueError(
            "Pairwise DEG currently expects exactly one MPR-associated reference "
            f"state; got {mpr_fines}."
        )

    ref = str(mpr_fines[0])
    genes = []
    owners = []
    seen = set()
    all_details = []

    for q in nonmpr_fines:
        q = str(q)
        selected, detail = _pairwise_marker_table(
            ad_focus,
            query_cluster=q,
            ref_cluster=ref,
            n_per_direction=n_per_direction,
        )

        # Query-enriched genes first.
        for gene in selected.get(q, []):
            if gene not in seen:
                genes.append(gene)
                owners.append(f"F{q}>F{ref}")
                seen.add(gene)

        # Reference-enriched genes specific to this comparison.
        for gene in selected.get(ref, []):
            if gene not in seen:
                genes.append(gene)
                owners.append(f"F{ref}>F{q}")
                seen.add(gene)

        for target, d in detail.items():
            if not d.empty:
                dd = d.copy()
                dd["comparison"] = f"F{q}_vs_F{ref}"
                all_details.append(dd)

    detail_df = (
        pd.concat(all_details, ignore_index=True)
        if all_details else pd.DataFrame()
    )

    return genes, owners, detail_df


def plot_response_direction_markers(
    ad_focus,
    nonmpr_fines,
    mpr_fines,
    out_dir,
):
    """
    Compact dotplot based on separate pairwise marker analyses.

    With defaults:
        F12 vs F9
        F19 vs F9

    Rows show F12, F19, F9 expression for the union of top markers selected
    from the two pairwise comparisons.
    """
    display_clusters = list(dict.fromkeys(nonmpr_fines + mpr_fines))

    available = set(clean_series(ad_focus.obs[FINE_COL]))
    missing = [x for x in display_clusters if x not in available]
    if missing:
        raise ValueError(f"DEG display fused states absent: {missing}")

    genes, owners, detail_df = select_pairwise_response_markers(
        ad_focus,
        nonmpr_fines=nonmpr_fines,
        mpr_fines=mpr_fines,
    )

    if not detail_df.empty:
        out_tsv = os.path.join(out_dir, "07_pairwise_DEG_results.tsv")
        detail_df.to_csv(out_tsv, sep="\t", index=False)
        print(f"[Saved] {out_tsv}")

    fig, ax = plt.subplots(
        figsize=(
            max(7.2, 0.58 * max(len(genes), 1) + 3.2),
            max(3.5, 0.60 * len(display_clusters) + 2.1),
        )
    )

    if len(genes) == 0:
        ax.text(
            0.5, 0.5,
            "No pairwise markers passed filters",
            ha="center", va="center",
        )
        ax.axis("off")
        save_pdf(fig, out_dir, "07_Pairwise_F12_F19_vs_F9_markers")
        return

    obs = ad_focus.obs.copy()
    obs[FINE_COL] = clean_series(obs[FINE_COL])
    ex = expression_df(ad_focus, genes)

    rows = []
    for cl in display_clusters:
        idx = obs.index[obs[FINE_COL].astype(str).values == str(cl)]
        for gene in genes:
            vals = pd.to_numeric(
                ex.loc[idx, gene], errors="coerce"
            ).fillna(0).values
            rows.append({
                "cluster": str(cl),
                "gene": gene,
                "mean": float(np.mean(vals)) if len(vals) else 0.0,
                "pct": float(np.mean(vals > 0)) if len(vals) else 0.0,
            })

    dot = pd.DataFrame(rows)
    dot["relative_mean"] = 0.0

    for gene in genes:
        m = dot["gene"] == gene
        vals = dot.loc[m, "mean"].astype(float)
        lo = float(vals.min())
        hi = float(vals.max())
        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or abs(hi - lo) < 1e-12
        ):
            dot.loc[m, "relative_mean"] = 0.5
        else:
            dot.loc[m, "relative_mean"] = (vals - lo) / (hi - lo)

    xmap = {gene: i for i, gene in enumerate(genes)}
    ymap = {str(cl): i for i, cl in enumerate(display_clusters)}

    x = dot["gene"].map(xmap).values
    y = dot["cluster"].map(ymap).values
    sizes = 18 + 300 * dot["pct"].values

    sca = ax.scatter(
        x,
        y,
        s=sizes,
        c=dot["relative_mean"].values,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolor="#555555",
        linewidth=0.4,
    )

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.set_yticks(range(len(display_clusters)))
    ax.set_yticklabels([f"F{x}" for x in display_clusters])
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.set_ylim(len(display_clusters) - 0.5, -0.5)

    # Comparison labels above contiguous gene blocks.
    if owners:
        start = 0
        owner = owners[0]
        for i in range(1, len(owners) + 1):
            boundary = i == len(owners) or owners[i] != owner
            if not boundary:
                continue

            mid = (start + i - 1) / 2
            ax.text(
                mid,
                -0.78,
                owner,
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=INK,
                clip_on=False,
            )

            if i < len(owners):
                ax.axvline(
                    i - 0.5,
                    color="#999999",
                    lw=0.8,
                    ls="--",
                )
                start = i
                owner = owners[i]

    ax.grid(True, color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(
        "Pairwise transcriptional characterization of response-associated fused states"
    )
    despine(ax, keep=("left", "bottom"))

    cb = fig.colorbar(sca, ax=ax, fraction=0.025, pad=0.018)
    cb.set_label("Relative mean expression\nwithin each gene")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])

    pct_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
    handles = [
        ax.scatter(
            [], [],
            s=18 + 300 * p,
            facecolor="#B0B0B0",
            edgecolor="#555555",
            linewidth=0.4,
        )
        for p in pct_levels
    ]
    ax.legend(
        handles,
        ["10%", "25%", "50%", "75%", "100%"],
        title="% cells\nexpressing",
        frameon=False,
        bbox_to_anchor=(1.13, 1.02),
        loc="upper left",
    )

    save_pdf(
        fig,
        out_dir,
        "07_Pairwise_F12_F19_vs_F9_markers",
    )

# =============================================================================
# Optional summary printout: CXCL13 / PDCD1 cluster statistics
# =============================================================================

def print_anchor_gene_summary(ad, selected_rna):
    obs = clean_series(ad.obs[RNA_CLUSTER_COL])
    ex = expression_df(ad, ANCHOR_GENES)

    rows = []
    for cl in natural_sort(obs.unique()):
        idx = obs == cl
        row = {"cluster": cl, "n_cells": int(idx.sum())}
        for g in ANCHOR_GENES:
            vals = pd.to_numeric(ex.loc[idx, g], errors="coerce").dropna().values
            row[f"{g}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{g}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{g}_pct_pos"] = float(np.mean(vals > 0)) if len(vals) else np.nan
        rows.append(row)

    d = pd.DataFrame(rows)
    d["selected"] = d["cluster"].astype(str) == str(selected_rna)
    print("\n[CXCL13 / PDCD1 across RNA clusters]")
    print(d.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--h5ad", required=True, help="Input h5ad file")
    ap.add_argument("--out_dir", default="./fig_luad_fused_substate",
                    help="Output directory")

    # Explicitly fixed RNA cluster: no automatic six-gene selection anymore.
    ap.add_argument(
        "--rna_cluster", default="5",
        help="Pre-selected RNA cluster to follow into fused space",
    )

    ap.add_argument(
        "--focus_fines", default=None,
        help=("Comma-separated global fused fine clusters to display, e.g. "
              "12,19,6,9,24,10,27. If omitted, top-N by cell count within "
              "the selected RNA cluster are used."),
    )
    ap.add_argument(
        "--top_n_fines", type=int, default=7,
        help="Number of top fused states if --focus_fines is omitted",
    )

    # Defaults reproduce the response-associated states from the current story.
    ap.add_argument(
        "--nonmpr_fines", default="12,19",
        help="Comma-separated non-MPR-associated fused states; each is compared separately to the MPR reference",
    )
    ap.add_argument(
        "--mpr_fines", default="9",
        help="Comma-separated MPR-associated reference state; pairwise analysis currently expects one state",
    )

    ap.add_argument(
        "--skip_deg", action="store_true",
        help="Skip the final pairwise marker dotplot",
    )
    ap.add_argument(
        "--skip_trb", action="store_true",
        help="Skip the TRB/clonotype panel",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)

    focus_requested = parse_csv_list(args.focus_fines)
    nonmpr_fines = parse_csv_list(args.nonmpr_fines) or []
    mpr_fines = parse_csv_list(args.mpr_fines) or []
    selected_rna = str(args.rna_cluster)

    ad = sc.read_h5ad(args.h5ad)
    ad = ensure_response_binary(ad)

    required_obs = [RNA_CLUSTER_COL, FINE_COL, SAMPLE_COL, GROUP_COL]
    for c in required_obs:
        if c not in ad.obs.columns:
            raise ValueError(f"Missing ad.obs['{c}']")

    for c in [RNA_CLUSTER_COL, FINE_COL, SAMPLE_COL, GROUP_COL,
              TRB_COL, "sub_cell_type", "Response"]:
        if c in ad.obs.columns:
            ad.obs[c] = clean_series(ad.obs[c])

    for rep in [RNA_REP, FUSED_REP]:
        if rep not in ad.obsm:
            raise ValueError(f"Missing ad.obsm['{rep}']")

    # Check anchor genes.
    _, all_gene_names = get_matrix_and_names(ad)
    missing_anchor = [g for g in ANCHOR_GENES if g not in all_gene_names]
    if missing_anchor:
        raise ValueError(f"Missing anchor genes: {missing_anchor}")

    print(f"[Selection] fixed RNA cluster = {selected_rna}")

    ad_rna = ad[clean_series(ad.obs[RNA_CLUSTER_COL]).values == selected_rna].copy()
    if ad_rna.n_obs == 0:
        raise ValueError(f"RNA cluster {selected_rna} contains no cells.")
    print(f"[Selection] cells in RNA cluster {selected_rna}: {ad_rna.n_obs}")

    focus_fines = choose_focus_fines(
        ad_rna,
        requested=focus_requested,
        top_n=args.top_n_fines,
    )
    focus_colors = build_focus_colors(focus_fines)
    print(f"[Focus fused states] {focus_fines}")

    ad_focus = ad_rna[
        clean_series(ad_rna.obs[FINE_COL]).isin(focus_fines).values
    ].copy()
    print(f"[Focus fused states] cells: {ad_focus.n_obs}")

    # Print transparent marker summary across ALL RNA clusters.
    print_anchor_gene_summary(ad, selected_rna)

    # UMAPs.
    rna_umap = get_umap(ad, RNA_REP, "X_umap_rna")
    fused_umap = get_umap(ad, FUSED_REP, "X_umap_final")

    # -------------------------------------------------------------------------
    # Main figure sequence
    # -------------------------------------------------------------------------
    plot_rna_clusters(
        ad, rna_umap, selected_rna, args.out_dir
    )

    plot_anchor_gene_violins(
        ad,
        selected_rna,
        args.out_dir,
    )

    plot_fused_umap(
        ad, fused_umap, selected_rna,
        focus_fines, focus_colors, args.out_dir
    )

    plot_sankey(
        ad_rna, selected_rna,
        focus_fines, focus_colors, args.out_dir
    )

    response_stats = plot_response(
        ad_rna, focus_fines, args.out_dir
    )
    plot_response_3group_stacked(
        ad_rna,
        focus_fines,
        args.out_dir,
    )
    if not args.skip_trb:
        if TRB_COL not in ad_focus.obs.columns:
            print(f"[Skip TRB] ad.obs['{TRB_COL}'] not found")
        else:
            plot_trb_structure(
                ad_focus, focus_fines, args.out_dir
            )

    if not args.skip_deg:
        requested_deg_states = list(dict.fromkeys(nonmpr_fines + mpr_fines))
        missing_deg = [x for x in requested_deg_states if x not in set(focus_fines)]
        if missing_deg:
            print(
                f"[Skip DEG] requested clinically associated fused states "
                f"are not all in focus_fines: {missing_deg}"
            )
        else:
            plot_response_direction_markers(
                ad_focus,
                nonmpr_fines=nonmpr_fines,
                mpr_fines=mpr_fines,
                out_dir=args.out_dir,
            )

    print("\n[Clinical association table]")
    print(response_stats.to_string(index=False))
    print(f"\nDone. PDF figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
