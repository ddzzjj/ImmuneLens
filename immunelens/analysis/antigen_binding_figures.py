#!/usr/bin/env python3
"""Plot ECCITE embeddings and characterize selected fused Leiden clusters.

The model never sees dextramer calls during training. A no-hit call is a
measured non-hit for this panel, not proof that the cell is nonreactive.
Gene expression is read from the original log-normalized 2,000-gene matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/virus_eccite_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/virus_eccite_mpl")

import anndata as ad
import igraph as ig
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import umap


METHODS = ("tcr_only", "rna_only", "fused")
DEX_COLORS = {"Dex+": "#c73d3b", "No measured hit": "#a4acb3"}
EPITOPE_ORDER = ("No measured hit", "YLQPRTFLL", "SPRRARSVA",
                 "Other Spike epitopes", "Mixed/unspecified Dex+")
EPITOPE_COLORS = {
    "No measured hit": "#c6cbd0",
    "YLQPRTFLL": "#c83f3d",
    "SPRRARSVA": "#327eb7",
    "Other Spike epitopes": "#7954a2",
    "Mixed/unspecified Dex+": "#d19c35",
}
DEFAULT_EXPRESSION = os.environ.get(
    "IMMUNELENS_ECCITE_DATA", "data/eccite/PBMC_vaccine_ECCITE_TCR.h5ad")
# Preserve editable text in PDF; dense point clouds are rasterized by default.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_input(path: str):
    data = ad.read_h5ad(path)
    required = {"donor", "clone_id", "dex_label", "sample", "cell_state"}
    missing = required - set(data.obs)
    if missing:
        raise ValueError(f"Missing ECCITE metadata: {sorted(missing)}")
    obs = data.obs.copy()
    obs["donor"] = obs["donor"].astype(str)
    obs["clone_id"] = obs["clone_id"].astype(str)
    obs["dex_positive"] = obs["dex_label"].astype(str).str.lower().eq("positive")
    obs["dex_display"] = np.where(obs["dex_positive"], "Dex+", "No measured hit")
    obs["cell_state"] = obs["cell_state"].astype(str)
    obs["hit_antigens"] = obs.get("hit_antigens", pd.Series("", index=obs.index)).astype(str)
    methods = [m for m in METHODS if f"X_{m}" in data.obsm]
    if not methods:
        raise ValueError("No X_rna_only, X_tcr_only or X_fused embeddings found")
    embeddings = {m: np.asarray(data.obsm[f"X_{m}"], dtype=np.float32)
                  for m in methods}
    return obs, embeddings, methods


def clone_means(obs: pd.DataFrame, embeddings: dict[str, np.ndarray]):
    codes, clone_ids = pd.factorize(obs["clone_id"], sort=True)
    counts = np.bincount(codes, minlength=len(clone_ids))
    grouped = obs.groupby("clone_id", sort=True, observed=True)
    clone_obs = grouped.agg(
        donor=("donor", "first"),
        n_cells=("clone_id", "size"),
        n_dex_positive=("dex_positive", "sum"),
        cell_state=("cell_state", lambda x: x.mode().iloc[0] if len(x.mode()) else "<missing>"),
    ).reindex(clone_ids)
    clone_obs.index.name = "clone_id"
    clone_obs["dex_positive"] = clone_obs["n_dex_positive"] > 0
    clone_obs["dex_display"] = np.where(clone_obs["dex_positive"],
                                        "Dex+ clone", "No measured hit")
    epitope_by_clone = obs.loc[obs.dex_positive].groupby("clone_id")["hit_antigens"].agg(
        lambda x: ";".join(sorted(set(x) - {"", "nan"})))
    clone_obs["epitope"] = epitope_by_clone.reindex(clone_obs.index).fillna("")
    clone_obs.loc[clone_obs.epitope.str.contains(";"), "epitope"] = "Mixed"
    result = {}
    for method, z in embeddings.items():
        means = np.zeros((len(clone_ids), z.shape[1]), dtype=np.float32)
        np.add.at(means, codes, z)
        result[method] = means / counts[:, None]
    return clone_obs, result


def coords_for(embeddings, seed, n_neighbors, min_dist):
    result = {}
    for method, z in embeddings.items():
        if len(z) < 4:
            raise ValueError("At least four observations are required for UMAP")
        model = umap.UMAP(n_neighbors=min(n_neighbors, len(z) - 1),
                          min_dist=min_dist, metric="euclidean", random_state=seed)
        result[method] = model.fit_transform(z)
        print(f"[UMAP] {method}: {len(z):,} observations")
    return result


def leiden_for(embeddings, seed, n_neighbors, resolution):
    """Cluster each learned embedding; dextramer labels are never used here."""
    result = {}
    for method, z in embeddings.items():
        n = len(z)
        if n < 3:
            raise ValueError("At least three observations are required for Leiden")
        k = min(n_neighbors, n - 1)
        distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)
        rows = np.repeat(np.arange(n), k + 1)
        cols = indices.ravel()
        dist = distances.ravel()
        nonself = rows != cols
        rows, cols, dist = rows[nonself], cols[nonself], dist[nonself]
        # Scale by each observation's neighborhood radius; retain the stronger
        # direction when two cells select one another.
        radius = np.maximum(distances[:, -1], 1e-8)
        weights = np.exp(-dist / radius[rows])
        graph_weights = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        graph_weights = graph_weights.maximum(graph_weights.T)
        upper = sparse.triu(graph_weights, k=1).tocoo()
        graph = ig.Graph(n=n, edges=list(zip(upper.row.tolist(), upper.col.tolist())),
                         directed=False)
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights=upper.data.tolist(), resolution_parameter=resolution, seed=seed,
        )
        result[method] = np.asarray(partition.membership, dtype=int)
        print(f"[Leiden] {method}: {len(partition)} clusters ({n:,} observations)")
    return result


def save_leiden_proportions(obs, leiden, methods, output, clone_level=False):
    positive = obs["dex_positive"].to_numpy(dtype=bool)
    rows = []
    for method in methods:
        for cluster in np.unique(leiden[method]):
            mask = leiden[method] == cluster
            count = int(mask.sum())
            hits = int(positive[mask].sum())
            rows.append(dict(method=method, leiden=int(cluster), n_total=count,
                             n_dex_positive=hits, n_no_measured_hit=count - hits,
                             fraction_dex_positive=hits / count,
                             fraction_no_measured_hit=1 - hits / count))
    table = pd.DataFrame(rows)
    stem = "dex_clones_leiden_composition" if clone_level else "dex_cells_leiden_composition"
    table.to_csv(output / f"{stem}.csv", index=False)
    for method in methods:
        subset = table.loc[table.method == method].sort_values("leiden")
        fig, ax = plt.subplots(figsize=(max(8, min(24, len(subset) * .32)), 4.1))
        x = np.arange(len(subset))
        ax.bar(x, subset.fraction_dex_positive, color=DEX_COLORS["Dex+"],
               label="Measured Dex+")
        ax.bar(x, subset.fraction_no_measured_hit,
               bottom=subset.fraction_dex_positive,
               color=DEX_COLORS["No measured hit"], label="No measured hit")
        ax.set_xticks(x, subset.leiden.astype(str),
                      rotation=90 if len(x) > 20 else 0,
                      fontsize=7 if len(x) > 30 else 9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_xlabel("Leiden cluster")
        level = "donor-specific clones" if clone_level else "ECCITE cells"
        ax.set_title(f"{method}: {level}")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(output / f"{stem}_{method}.png", dpi=250,
                    bbox_inches="tight")
        fig.savefig(output / f"{stem}_{method}.pdf", bbox_inches="tight",
                    transparent=True)
        plt.close(fig)
    return table


def category_palette(values, preferred=None):
    names = pd.Series(values).fillna("<missing>").astype(str).unique().tolist()
    names = sorted(names)
    if "No measured hit" in names:
        names.remove("No measured hit")
        names.insert(0, "No measured hit")
    cmap = plt.get_cmap("tab20", max(len(names), 1))
    return {name: (preferred[name] if preferred and name in preferred else cmap(i))
            for i, name in enumerate(names)}


def draw_categories(ax, xy, values, palette):
    values = pd.Series(values).fillna("<missing>").astype(str).to_numpy()
    for name, color in palette.items():
        mask = values == name
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], s=2.5, alpha=.8,
                       color=color, linewidths=0, zorder=0)


def draw_leiden(ax, xy, labels, max_labels=20):
    n_clusters = int(labels.max()) + 1
    palette = []
    for name in ("tab20", "tab20b", "tab20c"):
        palette.extend(plt.get_cmap(name).colors)
    while len(palette) < n_clusters:
        i = len(palette) - 60
        palette.append(hsv_to_rgb(((i * 0.61803398875) % 1,
                                   0.62 + 0.12 * (i % 3),
                                   0.77 + 0.1 * (i % 2))))
    counts = np.bincount(labels, minlength=n_clusters)
    annotate = set(np.argsort(counts)[-max_labels:])
    for cluster in range(n_clusters):
        mask = labels == cluster
        color = palette[cluster]
        ax.scatter(xy[mask, 0], xy[mask, 1], s=2.5, alpha=.8,
                   color=color, linewidths=0, zorder=0)
        if cluster in annotate:
            center = np.median(xy[mask], axis=0)
            ax.text(center[0], center[1], str(cluster), fontsize=6,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=.5))
    return n_clusters


def shared_legend(fig, palette, center_x, ncol, y=.075):
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=5,
                      markerfacecolor=color, markeredgecolor="none", label=name)
               for name, color in palette.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(center_x, y),
               ncol=ncol, fontsize=7, frameon=False, handletextpad=.25,
               columnspacing=.65, borderaxespad=0)


def draw_umap_panel(ax, xy, kind, values, palettes, labels, vector_points):
    if not vector_points:
        ax.set_rasterization_zorder(1)
    if kind == "leiden":
        n_clusters = draw_leiden(ax, xy, labels)
        ax.text(.99, .99, f"{n_clusters} clusters", transform=ax.transAxes,
                ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    else:
        draw_categories(ax, xy, values[kind], palettes[kind])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def save_panel_legends(palettes, panel_dir):
    for kind, title, ncol in (("donor", "Donor", 4),
                              ("cell_state", "Cell state", 3),
                              ("dex_display", "Measured dextramer", 2)):
        palette = palettes[kind]
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=6,
                          markerfacecolor=color, markeredgecolor="none", label=name)
                   for name, color in palette.items()]
        rows = int(np.ceil(len(handles) / ncol))
        fig = plt.figure(figsize=(4.4, max(.55, .35 * rows)))
        fig.patch.set_alpha(0)
        fig.legend(handles=handles, loc="center", ncol=ncol,
                   title=title, frameon=False, fontsize=8, title_fontsize=9)
        fig.savefig(panel_dir / f"legend_{kind}.pdf", transparent=True,
                    bbox_inches="tight", pad_inches=.02)
        plt.close(fig)


def save_umap_grid(obs, coords, leiden, methods, output, clone_level=False,
                   vector_points=False, pdf_dpi=250):
    """Four UMAP columns and independent transparent PDF panels.

    By default, points in each panel become one raster layer in the PDF while
    titles, cluster numbers, color keys and other text stay as vector objects.
    """
    values = {
        "dex_display": obs["dex_display"].astype(str).replace({"Dex+ clone": "Dex+"}),
        "donor": obs["donor"].astype(str),
        "cell_state": obs["cell_state"].astype(str),
    }
    palettes = {
        "dex_display": category_palette(values["dex_display"], DEX_COLORS),
        "donor": category_palette(values["donor"]),
        "cell_state": category_palette(values["cell_state"]),
    }
    fields = ("leiden", "donor", "cell_state", "dex_display")
    titles = {"leiden": "Leiden", "donor": "Donor",
              "cell_state": "Cell state", "dex_display": "Measured dextramer"}
    fig, axes = plt.subplots(len(methods), len(fields),
                             figsize=(19.2, 4.45 * len(methods) + 2.1),
                             squeeze=False)
    fig.subplots_adjust(left=.045, right=.992, top=.92, bottom=.18,
                        wspace=.035, hspace=.12)
    for i, method in enumerate(methods):
        xy = coords[method]
        for j, kind in enumerate(fields):
            ax = axes[i, j]
            draw_umap_panel(ax, xy, kind, values, palettes,
                            leiden[method], vector_points)
            if i == 0:
                ax.set_title(titles[kind], fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(method.replace("_", "-"), rotation=90,
                              fontsize=10, labelpad=13)
    centers = [axes[0, j].get_position().x0 + axes[0, j].get_position().width / 2
               for j in range(len(fields))]
    shared_legend(fig, palettes["donor"], centers[1], ncol=4)
    shared_legend(fig, palettes["cell_state"], centers[2], ncol=3)
    shared_legend(fig, palettes["dex_display"], centers[3], ncol=2)
    fig.text(centers[0], .075, "Numbers label the largest clusters in each method",
             ha="center", va="top", fontsize=7)
    fig.suptitle("One point per donor-specific clone" if clone_level else
                 "Measured vaccine ECCITE cells", fontsize=14)
    stem = "dex_clones_combined_umap" if clone_level else "dex_cells_combined_umap"
    fig.savefig(output / f"{stem}.pdf", dpi=pdf_dpi,
                bbox_inches="tight", transparent=True)
    fig.savefig(output / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    panel_dir = output / "panels" / ("clones" if clone_level else "cells")
    panel_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for kind in fields:
            panel, ax = plt.subplots(figsize=(4.4, 4.4))
            panel.patch.set_alpha(0)
            ax.set_facecolor("none")
            draw_umap_panel(ax, coords[method], kind, values, palettes,
                            leiden[method], vector_points)
            ax.set_title(f"{method}: {titles[kind]}", fontsize=10)
            panel.savefig(panel_dir / f"{method}_{kind}.pdf", dpi=pdf_dpi,
                          bbox_inches="tight", pad_inches=.03, transparent=True)
            plt.close(panel)
    save_panel_legends(palettes, panel_dir)


def within_donor_enrichment(clone_obs, embeddings, k):
    donors = clone_obs["donor"].astype(str).to_numpy()
    positive = clone_obs["dex_positive"].to_numpy(dtype=bool)
    rows = []
    for donor in np.unique(donors):
        ix = np.flatnonzero(donors == donor)
        y = positive[ix]
        if y.sum() < 2 or (~y).sum() < 1 or len(ix) < 3:
            continue
        used = min(k, len(ix) - 1)
        expected = (int(y.sum()) - 1) / (len(ix) - 1)
        for method, all_z in embeddings.items():
            z = all_z[ix]
            nn = NearestNeighbors(n_neighbors=used, metric="euclidean").fit(z)
            neighbors = nn.kneighbors(X=None, return_distance=False)
            observed = float(np.mean(y[neighbors[y]]))
            rows.append(dict(donor=donor, method=method, n_clones=len(ix),
                             n_positive_clones=int(y.sum()), k=used,
                             positive_neighbor_fraction=observed,
                             random_expected_fraction=expected,
                             excess=observed - expected))
    if not rows:
        raise ValueError("No donor has enough measured Dex+ and no-hit clones")
    return pd.DataFrame(rows)


def cross_donor_enrichment(clone_obs, embeddings, k):
    donors = clone_obs["donor"].astype(str).to_numpy()
    positive = clone_obs["dex_positive"].to_numpy(dtype=bool)
    rows = []
    for donor in np.unique(donors):
        query = np.flatnonzero((donors == donor) & positive)
        other = np.flatnonzero(donors != donor)
        if not len(query) or not len(other):
            continue
        used = min(k, len(other))
        expected = float(positive[other].mean())
        for method, z in embeddings.items():
            nn = NearestNeighbors(n_neighbors=used, metric="euclidean").fit(z[other])
            neighbors = nn.kneighbors(z[query], return_distance=False)
            observed = float(np.mean(positive[other[neighbors]]))
            rows.append(dict(donor=donor, method=method, n_positive_queries=len(query),
                             k=used, positive_neighbor_fraction=observed,
                             random_expected_fraction=expected,
                             excess=observed - expected))
    return pd.DataFrame(rows)


def save_enrichment_plot(summary, output):
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    methods = [m for m in METHODS if m in summary.method.values]
    x = np.arange(len(methods))
    width = .36
    for offset, scope, color in [(-width / 2, "within_donor", "#5193bc"),
                                 (width / 2, "cross_donor", "#db8659")]:
        vals = [summary.loc[(summary.method == m) & (summary.scope == scope),
                            "mean_excess"].iloc[0] for m in methods]
        ax.bar(x + offset, vals, width, color=color,
               label=scope.replace("_", " "))
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Dex+ neighbor enrichment above donor-matched baseline")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "dex_neighbor_enrichment.png", dpi=250)
    fig.savefig(output / "dex_neighbor_enrichment.pdf")
    plt.close(fig)


def save_cluster_figure(fig, output: Path, stem: str, pdf_dpi: int = 250):
    fig.tight_layout()
    fig.savefig(output / f"{stem}.pdf", dpi=pdf_dpi,
                bbox_inches="tight", transparent=True)
    fig.savefig(output / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def epitope_display(obs: pd.DataFrame) -> pd.Series:
    """Group the three extremely rare Spike dextramer calls for legibility."""
    hit = obs.hit_antigens.fillna("").astype(str)
    display = pd.Series("No measured hit", index=obs.index)
    display.loc[obs.dex_positive & hit.eq("YLQPRTFLL")] = "YLQPRTFLL"
    display.loc[obs.dex_positive & hit.eq("SPRRARSVA")] = "SPRRARSVA"
    rare = {"RLNEVAKNL", "VLNDILSRL", "NLNESLIDL"}
    display.loc[obs.dex_positive & hit.isin(rare)] = "Other Spike epitopes"
    unknown = obs.dex_positive & display.eq("No measured hit")
    display.loc[unknown] = "Mixed/unspecified Dex+"
    return display


def draw_epitope_umap(ax, xy: np.ndarray, labels: np.ndarray,
                      title: str, pdf_dpi: int):
    # A single rasterized point layer keeps PDF labels editable and files small.
    for group in EPITOPE_ORDER:
        ix = np.flatnonzero(labels == group)
        if not len(ix):
            continue
        background = group == "No measured hit"
        ax.scatter(xy[ix, 0], xy[ix, 1],
                   s=2.0 if background else 5.0,
                   alpha=.28 if background else .9,
                   color=EPITOPE_COLORS[group], linewidths=0,
                   rasterized=True, zorder=0 if background else 1)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal", adjustable="datalim")


def save_epitope_umaps(obs: pd.DataFrame, coords: dict[str, np.ndarray],
                       methods: list[str], output: Path, pdf_dpi: int):
    output.mkdir(parents=True, exist_ok=True)
    labels = epitope_display(obs)
    counts = labels.value_counts().reindex(EPITOPE_ORDER, fill_value=0)
    counts.rename_axis("epitope_display").rename("n_cells").to_csv(
        output / "epitope_cell_counts.csv")
    names = {"tcr_only": "TCR-only", "rna_only": "GEX-only", "fused": "Fused"}
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5.1 * len(methods), 5.8), squeeze=False)
    for ax, method in zip(axes[0], methods):
        draw_epitope_umap(ax, coords[method], labels.to_numpy(),
                          names.get(method, method), pdf_dpi)
        panel, panel_ax = plt.subplots(figsize=(5.0, 5.0))
        draw_epitope_umap(panel_ax, coords[method], labels.to_numpy(),
                          names.get(method, method), pdf_dpi)
        save_cluster_figure(panel, output, f"epitope_umap_{method}", pdf_dpi)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6,
                      markerfacecolor=EPITOPE_COLORS[group], markeredgecolor="none",
                      label=f"{group} (n={int(counts[group]):,})")
               for group in EPITOPE_ORDER if counts[group] > 0]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Measured Spike dextramer epitopes", fontsize=13, y=.99)
    fig.subplots_adjust(left=.02, right=.99, top=.83, bottom=.18, wspace=.03)
    fig.savefig(output / "epitope_umap_three_modalities.pdf", dpi=pdf_dpi,
                bbox_inches="tight", transparent=True)
    fig.savefig(output / "epitope_umap_three_modalities.png", dpi=250,
                bbox_inches="tight")
    plt.close(fig)


def save_top_dex_clones(obs: pd.DataFrame, clone_obs: pd.DataFrame,
                        coords: dict[str, np.ndarray], methods: list[str],
                        output: Path, top_n: int, pdf_dpi: int):
    output.mkdir(parents=True, exist_ok=True)
    positive = clone_obs.loc[clone_obs.n_dex_positive.gt(0)].copy()
    # clone_means gives a lexically sorted clone_id index, so stable sorting
    # makes tied clone sizes deterministic without duplicating the index name.
    top = positive.sort_values(["n_dex_positive", "n_cells"],
                               ascending=[False, False], kind="stable").head(top_n).copy()
    top["clone_id"] = top.index.astype(str)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    # clone_means records a single epitope, or 'Mixed' for more than one.
    top["epitope_display"] = np.select(
        [top.epitope.eq("YLQPRTFLL"), top.epitope.eq("SPRRARSVA"),
         top.epitope.isin(["RLNEVAKNL", "VLNDILSRL", "NLNESLIDL"])],
        ["YLQPRTFLL", "SPRRARSVA", "Other Spike epitopes"],
        default="Mixed/unspecified Dex+")
    sequences = obs.groupby("clone_id", observed=True)[["cdr3a", "cdr3b"]].first()
    table = top[["rank", "clone_id", "donor", "n_dex_positive", "n_cells",
                 "epitope", "epitope_display"]].copy()
    table["cdr3a"] = sequences.reindex(top.index).cdr3a.to_numpy()
    table["cdr3b"] = sequences.reindex(top.index).cdr3b.to_numpy()
    table.to_csv(output / f"top{top_n}_measured_dex_clones.csv", index=False)
    clone_ids = obs.clone_id.astype(str).to_numpy()
    other = ~np.isin(clone_ids, top.index.to_numpy())
    if any(len(coords[m]) != len(obs) for m in methods):
        raise ValueError("Top-clone UMAPs must use cell-level coordinates")
    cmap = plt.get_cmap("tab20")
    color_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9]
    colors = [cmap(color_order[i % len(color_order)]) for i in range(len(top))]
    top_rows = list(table.itertuples(index=False))
    handles = []
    for row, color in zip(top_rows, colors):
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6,
                              markerfacecolor=color, markeredgecolor="none",
                              label=f"{row.cdr3b} ({row.donor})"))
    names = {"tcr_only": "TCR-only", "rna_only": "GEX-only", "fused": "Fused"}

    def draw_panel(ax, method):
        xy = coords[method]
        # Match the gray point layer in the existing dex_display UMAP panels.
        ax.scatter(xy[other, 0], xy[other, 1], s=2.5, alpha=.8,
                   color=DEX_COLORS["No measured hit"], linewidths=0,
                   rasterized=True, zorder=0)
        for row, color in zip(top_rows, colors):
            mask = clone_ids == row.clone_id
            ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=.9,
                       color=color, linewidths=0, rasterized=False, zorder=2)
        ax.set_title(names.get(method, method), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_aspect("equal", adjustable="datalim")

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5.0 * len(methods), 5.8),
                             squeeze=False)
    for ax, method in zip(axes[0], methods):
        draw_panel(ax, method)
    fig.suptitle(f"Top {len(top)} measured Dex+ TCR clones", fontsize=13, y=.99)
    fig.subplots_adjust(left=.02, right=.98, top=.84, bottom=.07, wspace=.03)
    stem = f"top{top_n}_tcr_clones_three_modalities"
    fig.savefig(output / f"{stem}.pdf", dpi=pdf_dpi,
                bbox_inches="tight", transparent=True)
    fig.savefig(output / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    for method in methods:
        panel, ax = plt.subplots(figsize=(5.4, 5.4))
        draw_panel(ax, method)
        panel.subplots_adjust(left=.03, right=.97, top=.94, bottom=.03)
        stem = f"top{top_n}_tcr_clones_{method}"
        panel.savefig(output / f"{stem}.pdf", dpi=pdf_dpi,
                      bbox_inches="tight", transparent=True)
        panel.savefig(output / f"{stem}.png", dpi=250, bbox_inches="tight")
        plt.close(panel)

    legend_fig = plt.figure(figsize=(9.5, 3.8))
    legend_fig.legend(handles=handles, title="TRB CDR3 (donor)",
                      loc="center", ncol=2, frameon=False,
                      fontsize=9, title_fontsize=10, columnspacing=1.6)
    legend_fig.savefig(output / f"top{top_n}_tcr_clones_legend.pdf",
                       bbox_inches="tight", transparent=True)
    legend_fig.savefig(output / f"top{top_n}_tcr_clones_legend.png",
                       dpi=250, bbox_inches="tight")
    plt.close(legend_fig)
    print(f"[Top clones] {len(top)} measured Dex+ clones; "
          f"epitopes={top.epitope_display.value_counts().to_dict()}")


def save_epitope_figures(obs: pd.DataFrame, clone_obs: pd.DataFrame,
                         cell_coords: dict[str, np.ndarray],
                         methods: list[str], output: Path,
                         top_output: Path, top_n: int, pdf_dpi: int):
    output.mkdir(parents=True, exist_ok=True)
    save_epitope_umaps(obs, cell_coords, methods, output, pdf_dpi)
    save_top_dex_clones(obs, clone_obs, cell_coords, methods,
                        top_output, top_n, pdf_dpi)
    print(f"[Epitope figures] {output}")


def save_ylq_positive_neighbors(obs: pd.DataFrame,
                                embeddings: dict[str, np.ndarray],
                                methods: list[str], output: Path,
                                k: int = 20):
    """Measure YLQ+ recovery among same-donor, different-clone neighbors.

    Queries are YLQPRTFLL Dex+ cells. The baseline is the YLQ+ fraction of
    eligible cells in each query's donor after excluding its own clonotype.
    All modalities use the same queries and eligible candidate cells.
    """
    from scipy.spatial.distance import cdist

    donor = obs["donor"].astype(str).to_numpy()
    clone = obs["clone_id"].astype(str).to_numpy()
    positive = (obs["dex_positive"].to_numpy(dtype=bool) &
                (obs["hit_antigens"].astype(str).to_numpy() == "YLQPRTFLL"))
    ordered = [m for m in METHODS if m in methods]
    per_donor = []
    all_fractions = {method: [] for method in ordered}
    all_baselines = []
    for donor_name in sorted(np.unique(donor)):
        pool = np.flatnonzero(donor == donor_name)
        query = pool[positive[pool]]
        if len(query) == 0:
            continue
        same_clone = clone[query, None] == clone[pool][None, :]
        eligible_count = len(pool) - same_clone.sum(axis=1)
        if np.any(eligible_count < k):
            raise ValueError(f"Donor {donor_name} has fewer than {k} eligible neighbors")
        eligible_positive = (positive[pool][None, :] & ~same_clone).sum(axis=1)
        baseline = eligible_positive / eligible_count
        all_baselines.extend(baseline)

        for method in ordered:
            z = np.asarray(embeddings[method], dtype=np.float32)
            if len(z) != len(obs) or not np.isfinite(z).all():
                raise ValueError(f"Invalid X_{method} embedding")
            distances = cdist(z[query], z[pool], metric="euclidean")
            distances[same_clone] = np.inf
            neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
            fractions = positive[pool][neighbors].mean(axis=1)
            all_fractions[method].extend(fractions)
            per_donor.append(dict(donor=donor_name, method=method,
                                  n_query_cells=len(query), k=k,
                                  positive_neighbor_fraction=float(fractions.mean()),
                                  baseline_fraction=float(baseline.mean()),
                                  fold_enrichment=float(fractions.mean() /
                                                        baseline.mean())
                                  if baseline.mean() > 0 else np.nan))

    if not all_baselines:
        raise ValueError("No YLQPRTFLL Dex+ query cells found")
    baseline_mean = float(np.mean(all_baselines))
    summary = pd.DataFrame([
        dict(method=method, n_query_cells=len(values), k=k,
             positive_neighbor_fraction=float(np.mean(values)),
             baseline_fraction=baseline_mean,
             fold_enrichment=float(np.mean(values) / baseline_mean))
        for method, values in all_fractions.items()
    ])
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "ylq_positive_neighbor_summary.csv", index=False)
    pd.DataFrame(per_donor).to_csv(output / "ylq_positive_neighbor_by_donor.csv",
                                    index=False)

    names = {"tcr_only": "TCR", "rna_only": "GEX", "fused": "Fused"}
    colors = {"tcr_only": "#397fb2", "rna_only": "#59a68f",
              "fused": "#d48258"}
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    x = np.arange(len(ordered))
    values = 100 * summary.set_index("method").loc[ordered, "positive_neighbor_fraction"]
    ax.bar(x, values, width=.57, color=[colors[m] for m in ordered])
    ax.axhline(100 * baseline_mean, color="#555555", linestyle="--", linewidth=1.2,
               label=f"Donor baseline ({baseline_mean:.1%})")
    for xi, value in zip(x, values):
        ax.text(xi, value + 1.5, f"{value:.1f}%", ha="center", va="bottom")
    ax.set_xticks(x, [names[m] for m in ordered])
    ax.set_ylim(0, min(100, max(values.max() + 11, 100 * baseline_mean + 11)))
    ax.set_ylabel("Same-epitope neighbors (%)")
    ax.set_title("YLQPRTFLL Dex+ cells")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    fig.text(.5, .015, f"{k} nearest cells within donor; same clone excluded",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output / "ylq_positive_neighbor_fraction.pdf", bbox_inches="tight")
    fig.savefig(output / "ylq_positive_neighbor_fraction.png", dpi=250,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[YLQ+ {k}-NN] {len(all_baselines):,} query cells; {output}")
    print(summary.to_string(index=False))


def save_positive_cosine_distances(obs: pd.DataFrame,
                                   embeddings: dict[str, np.ndarray],
                                   methods: list[str], output: Path,
                                   seed: int):
    """Plot raw-embedding cosine distances for the same Dex+/Dex+ pairs.

    Pairs from the same donor-specific clonotype are excluded. All measured
    Dex+ epitopes are pooled, so this describes overall Dex+ compactness.
    """
    from scipy.ndimage import gaussian_filter1d

    positive = np.flatnonzero(obs.dex_positive.to_numpy(dtype=bool))
    if len(positive) < 2:
        raise ValueError("At least two Dex+ cells are required")
    clones = obs.clone_id.astype(str).to_numpy()[positive]
    if len(np.unique(clones)) < 2:
        raise ValueError("At least two distinct Dex+ clonotypes are required")
    rng = np.random.default_rng(seed)
    n_sample = 500_000
    n = len(positive)
    if n <= 1500:
        first, second = np.triu_indices(n, k=1)
        eligible = np.flatnonzero(clones[first] != clones[second])
        chosen = rng.choice(eligible, size=min(n_sample, len(eligible)),
                            replace=False)
        pair_i, pair_j = positive[first[chosen]], positive[second[chosen]]
    else:
        # Large datasets: sample uniformly from unordered cell pairs and
        # reject same-clone pairs without materializing the full n x n matrix.
        chunks_i, chunks_j = [], []
        collected = 0
        while collected < n_sample:
            draw = max(1000, 2 * (n_sample - collected))
            first = rng.integers(n, size=draw)
            second = rng.integers(n, size=draw)
            keep = (first != second) & (clones[first] != clones[second])
            first, second = first[keep], second[keep]
            take = min(n_sample - collected, len(first))
            chunks_i.append(positive[first[:take]])
            chunks_j.append(positive[second[:take]])
            collected += take
        pair_i, pair_j = np.concatenate(chunks_i), np.concatenate(chunks_j)

    output.mkdir(parents=True, exist_ok=True)
    ordered = [m for m in METHODS if m in methods]
    names = {"tcr_only": "TCR-only", "rna_only": "RNA-only", "fused": "Fused"}
    palette = {"tcr_only": "#397fb2", "rna_only": "#59a68f",
               "fused": "#d48258"}
    distributions, rows = {}, []
    for method in ordered:
        z = np.asarray(embeddings[method], dtype=np.float32)
        if len(z) != len(obs) or not np.isfinite(z).all():
            raise ValueError(f"Invalid X_{method} embedding")
        norms = np.linalg.norm(z, axis=1)
        if np.any(norms[positive] == 0):
            raise ValueError(f"Zero-norm Dex+ vector in X_{method}")
        unit = z / np.maximum(norms[:, None], np.finfo(np.float32).tiny)
        distances = np.empty(len(pair_i), dtype=np.float32)
        for start in range(0, len(pair_i), 50_000):
            stop = min(start + 50_000, len(pair_i))
            similarity = np.einsum("ij,ij->i", unit[pair_i[start:stop]],
                                   unit[pair_j[start:stop]])
            distances[start:stop] = 1 - np.clip(similarity, -1, 1)
        q25, median, q75 = np.quantile(distances, [.25, .5, .75])
        rows.append(dict(method=method, n_positive_cells=n,
                         n_positive_clones=len(np.unique(clones)),
                         n_pairs=len(pair_i), metric="cosine",
                         embedding="raw", q25_distance=q25,
                         median_distance=median, q75_distance=q75,
                         mean_distance=float(np.mean(distances))))
        distributions[method] = distances

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "dex_positive_pair_cosine_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for x, method in enumerate(ordered):
        values = distributions[method]
        counts, edges = np.histogram(values, bins=160, range=(0, 2), density=True)
        density = gaussian_filter1d(counts.astype(float), sigma=1.5)
        centers = (edges[:-1] + edges[1:]) / 2
        half_width = .38 * density / max(density.max(), 1e-12)
        ax.fill_betweenx(centers, x - half_width, x + half_width,
                         facecolor=palette[method], edgecolor="none", alpha=.85)
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        ax.plot([x, x], [q25, q75], color="black", linewidth=3.5,
                solid_capstyle="round", zorder=3)
        ax.scatter(x, median, s=35, facecolors="white", edgecolors="black",
                   linewidths=.7, zorder=4)
        ax.text(x, q75 + .025, f"{median:.3f}", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(range(len(ordered)), [names[m] for m in ordered])
    ax.set_xlim(-.6, len(ordered) - .4)
    ax.set_ylim(0, min(2.0, max(float(np.quantile(v, .995))
                                for v in distributions.values()) * 1.12 + .03))
    ax.set_ylabel("Cosine distance (1 - cosine similarity)")
    ax.set_title("Dex+ to Dex+ | different clonotypes\nRaw embeddings; same cell pairs in all modalities")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    stem = "dex_positive_pair_cosine_three_modalities"
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Dex+ cosine distance] {n:,} cells, {len(pair_i):,} pairs; {output}")
    print(summary.to_string(index=False))


def save_ylq_relative_distances(obs: pd.DataFrame,
                                embeddings: dict[str, np.ndarray],
                                methods: list[str], output: Path,
                                seed: int):
    """Compare YLQ-to-YLQ and YLQ-to-other distances within each modality.

    PCA-15 is fitted on all cells separately for each embedding. The same
    sampled cell pairs are used in every modality; pairs sharing a clonotype
    are excluded. Other cells include no measured hit and other epitopes.
    """
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA

    output.mkdir(parents=True, exist_ok=True)
    is_ylq = (obs.dex_positive.to_numpy() &
              obs.hit_antigens.eq("YLQPRTFLL").to_numpy())
    positive = np.flatnonzero(is_ylq)
    other = np.flatnonzero(~is_ylq)
    if len(positive) < 2 or len(other) == 0:
        raise ValueError("Both YLQ+ cells and other cells are required")
    clone = obs.clone_id.astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    n_sample = 500_000

    # Sample uniformly from all distinct-clone, unordered positive pairs.
    first, second = np.triu_indices(len(positive), k=1)
    eligible = np.flatnonzero(clone[positive[first]] !=
                              clone[positive[second]])
    if len(eligible) == 0:
        raise ValueError("No cross-clone YLQ+ cell pairs")
    chosen = rng.choice(eligible, size=min(n_sample, len(eligible)),
                        replace=False)
    within_i = positive[first[chosen]]
    within_j = positive[second[chosen]]
    del first, second, eligible, chosen

    # Uniformly sample YLQ+ / non-YLQ pairs, again excluding the same clone.
    between_i, between_j = [], []
    collected = 0
    while collected < n_sample:
        draw = max(1000, (n_sample - collected) * 2)
        i = positive[rng.integers(len(positive), size=draw)]
        j = other[rng.integers(len(other), size=draw)]
        keep = clone[i] != clone[j]
        i, j = i[keep], j[keep]
        take = min(n_sample - collected, len(i))
        between_i.append(i[:take])
        between_j.append(j[:take])
        collected += take
    between_i = np.concatenate(between_i)
    between_j = np.concatenate(between_j)

    # One shared set of random, different-clone cell pairs establishes a
    # modality-specific distance scale without forcing the between-group
    # medians in the second figure to equal one.
    random_i, random_j = [], []
    collected = 0
    while collected < n_sample:
        draw = max(1000, (n_sample - collected) * 2)
        i = rng.integers(len(obs), size=draw)
        j = rng.integers(len(obs), size=draw)
        keep = (i != j) & (clone[i] != clone[j])
        i, j = i[keep], j[keep]
        take = min(n_sample - collected, len(i))
        random_i.append(i[:take])
        random_j.append(j[:take])
        collected += take
    random_i = np.concatenate(random_i)
    random_j = np.concatenate(random_j)

    names = {"tcr_only": "TCR-only", "rna_only": "GEX-only", "fused": "Fused"}
    ordered = [m for m in METHODS if m in methods]
    distributions, rows = {}, []
    for method in ordered:
        z = np.asarray(embeddings[method], dtype=np.float32)
        if not np.isfinite(z).all():
            raise ValueError(f"Non-finite values in X_{method}")
        if min(z.shape) < 15:
            raise ValueError(f"X_{method} has fewer than 15 PCA dimensions")
        scores = PCA(n_components=15, svd_solver="full").fit_transform(z)
        within = np.linalg.norm(scores[within_i] - scores[within_j], axis=1)
        between = np.linalg.norm(scores[between_i] - scores[between_j], axis=1)
        random_distances = np.linalg.norm(scores[random_i] - scores[random_j],
                                          axis=1)
        random_median = float(np.median(random_distances))
        median_between = float(np.median(between))
        ratio = float(np.median(within) / median_between)
        # Chance that an independent YLQ+/YLQ+ pair is closer than a
        # YLQ+/other pair; 0.5 means no separation.
        sorted_between = np.sort(between)
        probability = float(np.mean(
            (len(between) - np.searchsorted(sorted_between, within,
                                             side="right")) / len(between)))
        rows.append({
            "method": method,
            "n_ylq_cells": len(positive),
            "n_other_cells": len(other),
            "n_within_pairs_sampled": len(within),
            "n_between_pairs_sampled": len(between),
            "pca_components": 15,
            "within_median_distance": float(np.median(within)),
            "between_median_distance": median_between,
            "random_pair_median_distance": random_median,
            "within_over_random_median": float(np.median(within) / random_median),
            "between_over_random_median": float(median_between / random_median),
            "within_to_between_median_ratio": ratio,
            "probability_within_closer_than_between": probability,
        })
        distributions[method] = (within / random_median,
                                 between / random_median)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "ylq_within_vs_other_pca15_summary.csv",
                   index=False)

    palette = {"tcr_only": "#397fb2", "rna_only": "#59a68f",
               "fused": "#d48258"}
    ymax = max(float(np.quantile(values, .999))
               for groups in distributions.values() for values in groups) * 1.06
    for group_idx, stem, title in (
            (0, "ylq_positive_pair_distance_three_modalities",
             "YLQ+ to YLQ+ | different clones"),
            (1, "ylq_positive_to_other_distance_three_modalities",
             "YLQ+ to other cells | different clones")):
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for x, method in enumerate(ordered):
            values = distributions[method][group_idx]
            counts, edges = np.histogram(values, bins=160, density=True)
            density = gaussian_filter1d(counts.astype(float), sigma=1.5)
            centers = (edges[:-1] + edges[1:]) / 2
            half_width = .38 * density / max(density.max(), 1e-12)
            ax.fill_betweenx(centers, x - half_width, x + half_width,
                             facecolor=palette[method], edgecolor="none",
                             alpha=.85)
            q25, median, q75 = np.quantile(values, [.25, .5, .75])
            ax.plot([x, x], [q25, q75], color="black", linewidth=3.5,
                    solid_capstyle="round", zorder=3)
            ax.scatter(x, median, s=35, facecolors="white",
                       edgecolors="black", linewidths=.7, zorder=4)
            ax.text(x, q75 + .045, f"{median:.2f}", ha="center",
                    va="bottom", fontsize=9)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=.9)
        ax.set_xticks(range(len(ordered)), [names[m] for m in ordered])
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Distance / median random-cell-pair distance")
        ax.set_title(title + "\nPCA 15D; 500,000 sampled pairs per group",
                     fontsize=11, pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.png", dpi=250,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"[YLQ+ relative distance] {len(positive):,} YLQ+ cells, "
          f"{len(other):,} other cells; {output}")
    print(summary.to_string(index=False))


def plot_cluster_categories(obs: pd.DataFrame, labels: np.ndarray,
                            targets: tuple[int, ...], output: Path):
    """Report measured categories; epitope denominators include Dex+ cells only."""
    work = obs.copy()
    work["cluster_group"] = pd.Categorical(
        np.where(np.isin(labels, targets), labels.astype(str), "Other"),
        categories=[*(str(x) for x in targets), "Other"], ordered=True)
    panels = (
        ("cell_state", "cell_state", work, "All cells"),
        ("donor", "donor", work, "All cells"),
        ("spike_epitope", "hit_antigens", work.loc[work.dex_positive].copy(),
         "Measured Dex+ cells only"),
    )
    for stem, column, frame, denominator in panels:
        if frame.empty:
            continue
        frame[column] = frame[column].fillna("<missing>").astype(str)
        if stem == "spike_epitope":
            frame.loc[frame[column].eq(""), column] = "Unspecified"
        counts = pd.crosstab(frame.cluster_group, frame[column], dropna=False)
        counts = counts.reindex(work.cluster_group.cat.categories, fill_value=0)
        counts = counts.loc[:, counts.sum().sort_values(ascending=False).index]
        totals = counts.sum(axis=1)
        fractions = counts.div(totals.replace(0, np.nan), axis=0).fillna(0)
        detail = counts.stack().rename("n_cells").reset_index()
        detail.columns = ["cluster_group", column, "n_cells"]
        detail["fraction_within_group"] = [
            fractions.loc[group, value] for group, value in
            zip(detail.cluster_group, detail[column])]
        detail["denominator"] = denominator
        detail.to_csv(output / f"{stem}_composition.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.1, 4.6))
        bottom = np.zeros(len(fractions))
        cmap = plt.get_cmap("tab20")
        for i, category in enumerate(fractions.columns):
            values = fractions[category].to_numpy()
            ax.bar(np.arange(len(fractions)), values, bottom=bottom,
                   color=cmap(i % 20), label=category, width=.7)
            bottom += values
        ax.set_xticks(np.arange(len(fractions)),
                      [f"{g}\n(n={int(totals.loc[g]):,})" for g in fractions.index])
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Fraction within group")
        ax.set_xlabel("Fused Leiden cluster")
        ax.set_title(f"{stem.replace('_', ' ').title()} | {denominator}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title=column.replace("_", " "), bbox_to_anchor=(1.02, 1),
                  loc="upper left", frameon=False, fontsize=8)
        save_cluster_figure(fig, output, stem + "_composition")


def plot_independent_clones(obs: pd.DataFrame, labels: np.ndarray,
                            targets: tuple[int, ...], output: Path):
    """Deduplicate cells by donor-specific paired clonotype within each cluster."""
    work = obs[["donor", "clone_id", "dex_positive"]].copy()
    work["cluster_group"] = pd.Categorical(
        np.where(np.isin(labels, targets), labels.astype(str), "Other"),
        categories=[*(str(x) for x in targets), "Other"], ordered=True)
    # clone_id in the source is donor::paired CDR3. Verify the donor component.
    if not all(clone.startswith(donor + "::") for clone, donor in
               zip(work.clone_id.astype(str), work.donor.astype(str))):
        raise ValueError("clone_id is not donor-specific; cannot count independent clones")
    clones = work.groupby(["cluster_group", "clone_id"], observed=True).agg(
        donor=("donor", "first"), n_cells=("clone_id", "size"),
        n_dex_positive_cells=("dex_positive", "sum")).reset_index()
    clones["dex_positive_clone"] = clones.n_dex_positive_cells.gt(0)
    clones.to_csv(output / "independent_clones_detail.csv", index=False)
    summary = clones.groupby("cluster_group", observed=True).agg(
        n_unique_donor_specific_clones=("clone_id", "size"),
        n_dex_positive_clones=("dex_positive_clone", "sum"),
        n_cells=("n_cells", "sum"),
        n_dex_positive_cells=("n_dex_positive_cells", "sum"),
        median_cells_per_clone=("n_cells", "median"),
    ).reindex([*(str(x) for x in targets), "Other"]).fillna(0)
    summary["fraction_dex_positive_clones"] = (
        summary.n_dex_positive_clones /
        summary.n_unique_donor_specific_clones.replace(0, np.nan))
    summary["cells_per_clone"] = summary.n_cells / summary.n_unique_donor_specific_clones.replace(0, np.nan)
    summary.to_csv(output / "independent_clones_summary.csv")
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    x = np.arange(len(summary))
    hits = summary.n_dex_positive_clones.to_numpy()
    no_hits = summary.n_unique_donor_specific_clones.to_numpy() - hits
    ax.bar(x, hits, color=DEX_COLORS["Dex+"], label="At least one measured Dex+ cell")
    ax.bar(x, no_hits, bottom=hits, color=DEX_COLORS["No measured hit"],
           label="No measured hit")
    ax.set_xticks(x, summary.index)
    ax.set_ylabel("Unique donor-specific paired TCR clones")
    ax.set_xlabel("Fused Leiden cluster")
    ax.set_title("Independent paired TCR clones per cluster")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save_cluster_figure(fig, output, "independent_clones")
    # A clone may occupy several clusters; this CSV makes overlap explicit.
    overlap = clones.groupby("clone_id").agg(
        n_cluster_groups=("cluster_group", "nunique"),
        groups=("cluster_group", lambda x: ";".join(sorted(map(str, x.unique())))))
    overlap.loc[overlap.n_cluster_groups.gt(1)].to_csv(
        output / "clones_in_multiple_groups.csv")


def plot_cluster_markers(obs: pd.DataFrame, labels: np.ndarray,
                         targets: tuple[int, ...], source_path: str,
                         output: Path, top_n: int = 6):
    """Rank markers by equal-weight donor mean expression differences.

    Each target is compared with the other selected clusters in the same
    donor. Scores are differences in the source matrix's log expression units,
    not log fold changes or formal differential-expression p values.
    """
    source = ad.read_h5ad(source_path, backed="r")
    try:
        if not source.obs_names.is_unique or not obs.index.is_unique:
            raise ValueError("Barcodes must be unique for expression alignment")
        rows = source.obs_names.get_indexer(obs.index)
        if np.any(rows < 0):
            raise ValueError(f"{int(np.sum(rows < 0))} embedding barcodes absent from expression data")
        if np.any(np.diff(rows) < 0):
            raise ValueError("Expression barcodes are not in embedding order")
        x = source.X[rows, :]
        x = x.tocsr() if sparse.issparse(x) else np.asarray(x, dtype=np.float32)
        genes = np.asarray(source.var_names.astype(str))
        if len(np.unique(genes)) != len(genes):
            raise ValueError("Expression gene names must be unique")
    finally:
        source.file.close()
    donor = obs.donor.astype(str).to_numpy()
    groups = [*(str(c) for c in targets), "Other"]
    group_labels = np.where(np.isin(labels, targets), labels.astype(str), "Other")
    mean_by_group = {}
    for group in groups:
        per_donor = []
        for d in np.unique(donor):
            ix = np.flatnonzero((donor == d) & (group_labels == group))
            if len(ix):
                per_donor.append(np.asarray(x[ix].mean(axis=0)).ravel())
        mean_by_group[group] = np.mean(per_donor, axis=0) if per_donor else np.full(len(genes), np.nan)
    marker_rows = []
    for target in targets:
        target_mask = labels == target
        reference_mask = np.isin(labels, [c for c in targets if c != target])
        differences = []
        positive_fractions = []
        for d in np.unique(donor):
            target_ix = np.flatnonzero((donor == d) & target_mask)
            reference_ix = np.flatnonzero((donor == d) & reference_mask)
            if len(target_ix) < 5 or len(reference_ix) < 5:
                continue
            a = np.asarray(x[target_ix].mean(axis=0)).ravel()
            b = np.asarray(x[reference_ix].mean(axis=0)).ravel()
            differences.append(a - b)
            positive_fractions.append(np.asarray((x[target_ix] > 0).mean(axis=0)).ravel())
        if len(differences) < 2:
            raise ValueError(f"Cluster {target} has fewer than two donors with >=5 cells in both comparison groups")
        diffs = np.vstack(differences)
        scores = diffs.mean(axis=0)
        consistency = (diffs > 0).mean(axis=0)
        detection = np.mean(positive_fractions, axis=0)
        for j, gene in enumerate(genes):
            marker_rows.append((target, gene, float(scores[j]),
                                float(consistency[j]), float(detection[j]), len(differences)))
    markers = pd.DataFrame(marker_rows, columns=[
        "cluster", "gene", "mean_log_expression_difference",
        "fraction_donors_higher", "mean_fraction_cells_expressing", "n_donors_compared"])
    markers.sort_values(["cluster", "mean_log_expression_difference"],
                        ascending=[True, False]).to_csv(output / "marker_genes_all.csv", index=False)
    chosen = []
    chosen_rows = []
    for target in targets:
        block = markers.loc[(markers.cluster == target) &
                            (markers.fraction_donors_higher >= .6) &
                            (markers.mean_fraction_cells_expressing.between(.05, .95)) &
                            (~markers.gene.str.match(r"^(RPL|RPS|MT-|HIST)")) &
                            (~markers.gene.isin(["GAPDH", "ACTB", "MALAT1"]))]
        block = block.sort_values("mean_log_expression_difference", ascending=False)
        selected_for_target = 0
        for _, row in block.iterrows():
            gene = row.gene
            if gene not in chosen:
                chosen.append(gene)
                chosen_rows.append(row)
                selected_for_target += 1
            if selected_for_target >= top_n:
                break
    if not chosen:
        raise ValueError("No expressed, donor-consistent marker genes found")
    pd.DataFrame(chosen_rows).to_csv(output / "marker_genes_selected.csv", index=False)
    matrix = np.vstack([mean_by_group[g] for g in groups]).T
    def draw_marker_heatmap(gene_list, stem, title):
        indexes = [int(np.flatnonzero(genes == gene)[0]) for gene in gene_list]
        values = matrix[indexes]
        row_mean = np.nanmean(values, axis=1, keepdims=True)
        row_sd = np.nanstd(values, axis=1, keepdims=True)
        values = np.clip((values - row_mean) / np.maximum(row_sd, 1e-6), -2, 2)
        fig, ax = plt.subplots(figsize=(7, max(5, .34 * len(gene_list) + 1.7)))
        im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_yticks(range(len(gene_list)), gene_list, fontsize=8)
        ax.set_xlabel("Fused Leiden cluster")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Row z score of mean log expression", shrink=.7)
        save_cluster_figure(fig, output, stem)
    draw_marker_heatmap(chosen, "marker_gene_heatmap",
                        "Candidate markers | donor-balanced mean expression")
    curated = ["CD8A", "CD8B", "CCR7", "IL7R", "TCF7", "LTB", "SELL",
               "GZMB", "GZMK", "GNLY", "PRF1", "NKG7", "CCL5", "CX3CR1",
               "MKI67", "STMN1", "CD69", "HLA-DRA", "LAG3"]
    curated = [gene for gene in curated if gene in set(genes)]
    pd.DataFrame({"gene": curated}).to_csv(output / "known_cd8_marker_panel.csv", index=False)
    if curated:
        draw_marker_heatmap(curated, "known_cd8_marker_heatmap",
                            "Known CD8 state markers | donor-balanced mean expression")


def characterize_clusters(obs: pd.DataFrame, labels: np.ndarray,
                          targets: tuple[int, ...], source_path: str,
                          output: Path):
    missing = set(targets) - set(np.unique(labels))
    if missing:
        raise ValueError(f"Fused Leiden clusters absent: {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    plot_cluster_categories(obs, labels, targets, output)
    plot_independent_clones(obs, labels, targets, output)
    plot_cluster_markers(obs, labels, targets, source_path, output)
    pd.DataFrame({"barcode": obs.index.astype(str), "donor": obs.donor.astype(str),
                  "fused_leiden": labels, "dex_positive": obs.dex_positive,
                  "clone_id": obs.clone_id.astype(str)}).to_csv(
                      output / "fused_cell_cluster_assignments.csv", index=False)
    print(f"[Cluster characterization] {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=.3)
    parser.add_argument("--leiden_neighbors", type=int, default=15)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pdf_dpi", type=int, default=250,
                        help="point-cloud resolution in the compact PDF")
    parser.add_argument("--vector_points", action="store_true",
                        help="make every dot editable; may overwhelm Illustrator")
    parser.add_argument("--skip_umap", action="store_true")
    parser.add_argument("--cluster_only", action="store_true",
                        help="Only make fused-cluster characterization; skip existing plots")
    parser.add_argument("--epitope_only", action="store_true",
                        help="Only make three-modality epitope and Top-clone UMAPs")
    parser.add_argument("--top_clones_only", action="store_true",
                        help="Only color Top measured Dex+ TCR clones on three cell UMAPs")
    parser.add_argument("--epitope_distance_only", action="store_true",
                        help="Only compare YLQ+ within-epitope and other-cell distances in PCA-15 space")
    parser.add_argument("--positive_distance_only", action="store_true",
                        help="Only plot Dex+/Dex+ cosine distances in raw embeddings")
    parser.add_argument("--positive_neighbor_only", action="store_true",
                        help="Only plot YLQ+ fraction among 20 same-donor, different-clone neighbors")
    parser.add_argument("--top_clones", type=int, default=15,
                        help="Number of measured Dex+ donor-specific clones to highlight")
    parser.add_argument("--expression_data", default=DEFAULT_EXPRESSION,
                        help="Original ECCITE h5ad containing the log-normalized gene matrix")
    parser.add_argument("--clusters", default="2,13,15",
                        help="Comma-separated fused Leiden cluster numbers")
    args = parser.parse_args()
    if args.neighbors < 1 or args.umap_neighbors < 2 or args.leiden_neighbors < 2:
        parser.error("neighbor counts must be positive")
    if args.leiden_resolution <= 0:
        parser.error("--leiden_resolution must be positive")
    if args.pdf_dpi < 72:
        parser.error("--pdf_dpi must be at least 72")
    if args.top_clones < 1:
        parser.error("--top_clones must be positive")
    if sum((args.epitope_only, args.cluster_only, args.top_clones_only,
            args.epitope_distance_only, args.positive_distance_only,
            args.positive_neighbor_only)) > 1:
        parser.error("Only one *_only mode may be selected")
    if (args.epitope_only or args.top_clones_only) and args.skip_umap:
        parser.error("UMAP-only modes cannot be combined with --skip_umap")
    try:
        targets = tuple(dict.fromkeys(int(s.strip()) for s in args.clusters.split(",")))
    except ValueError:
        parser.error("--clusters must be comma-separated integers")
    if not targets:
        parser.error("--clusters cannot be empty")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    obs, embeddings, methods = load_input(args.input)
    if "fused" not in embeddings:
        raise ValueError("Cluster characterization requires X_fused")
    cluster_output = output / ("clusters_" + "_".join(map(str, targets)))
    epitope_output = output / "spike_epitope_umaps"
    top_output = output / f"top{args.top_clones}_tcr_clones"
    distance_output = output / "ylq_two_distance_groups_pca15"
    positive_distance_output = output / "dex_positive_pair_cosine"
    positive_neighbor_output = output / "ylq_positive_neighbors_k20"
    if args.positive_neighbor_only:
        save_ylq_positive_neighbors(obs, embeddings, methods,
                                    positive_neighbor_output)
        return
    if args.positive_distance_only:
        save_positive_cosine_distances(obs, embeddings, methods,
                                       positive_distance_output, args.seed)
        return
    if args.epitope_distance_only:
        save_ylq_relative_distances(obs, embeddings, methods,
                                    distance_output, args.seed)
        return
    if args.cluster_only:
        labels = leiden_for({"fused": embeddings["fused"]}, args.seed,
                            args.leiden_neighbors, args.leiden_resolution)["fused"]
        characterize_clusters(obs, labels, targets, args.expression_data, cluster_output)
        return
    clones, clone_embeddings = clone_means(obs, embeddings)
    if args.top_clones_only:
        cell_coords = coords_for(embeddings, args.seed,
                                 args.umap_neighbors, args.umap_min_dist)
        save_top_dex_clones(obs, clones, cell_coords, methods, top_output,
                            args.top_clones, args.pdf_dpi)
        return
    if args.epitope_only:
        cell_coords = coords_for(embeddings, args.seed,
                                 args.umap_neighbors, args.umap_min_dist)
        save_epitope_figures(obs, clones, cell_coords,
                             methods, epitope_output, top_output,
                             args.top_clones, args.pdf_dpi)
        return
    save_positive_cosine_distances(obs, embeddings, methods,
                                   positive_distance_output, args.seed)
    save_ylq_positive_neighbors(obs, embeddings, methods,
                                positive_neighbor_output)
    cell_leiden = leiden_for(embeddings, args.seed, args.leiden_neighbors,
                             args.leiden_resolution)
    clone_leiden = leiden_for(clone_embeddings, args.seed, args.leiden_neighbors,
                              args.leiden_resolution)
    save_leiden_proportions(obs, cell_leiden, methods, output)
    save_leiden_proportions(clones, clone_leiden, methods, output, clone_level=True)
    if not args.skip_umap:
        cell_coords = coords_for(embeddings, args.seed,
                                 args.umap_neighbors, args.umap_min_dist)
        clone_coords = coords_for(clone_embeddings, args.seed,
                                  args.umap_neighbors, args.umap_min_dist)
        save_umap_grid(obs, cell_coords, cell_leiden, methods, output,
                       vector_points=args.vector_points, pdf_dpi=args.pdf_dpi)
        save_umap_grid(clones, clone_coords, clone_leiden, methods, output,
                       clone_level=True, vector_points=args.vector_points,
                       pdf_dpi=args.pdf_dpi)
        save_epitope_figures(obs, clones, cell_coords,
                             methods, epitope_output, top_output,
                             args.top_clones, args.pdf_dpi)
    within = within_donor_enrichment(clones, clone_embeddings, args.neighbors)
    cross = cross_donor_enrichment(clones, clone_embeddings, args.neighbors)
    within.to_csv(output / "dex_neighbors_within_donor.csv", index=False)
    cross.to_csv(output / "dex_neighbors_cross_donor.csv", index=False)
    detail = pd.concat([within.assign(scope="within_donor"),
                        cross.assign(scope="cross_donor")], ignore_index=True)
    summary = detail.groupby(["scope", "method"], as_index=False).agg(
        n_donors=("donor", "nunique"),
        mean_positive_neighbor_fraction=("positive_neighbor_fraction", "mean"),
        mean_random_expected_fraction=("random_expected_fraction", "mean"),
        mean_excess=("excess", "mean"),
        median_excess=("excess", "median"),
    )
    summary.to_csv(output / "dex_neighbor_summary.csv", index=False)
    save_enrichment_plot(summary, output)
    clones.to_csv(output / "dex_clone_annotations.csv")
    report = dict(n_cells=len(obs), n_dex_positive_cells=int(obs.dex_positive.sum()),
                  n_clones=len(clones), n_dex_positive_clones=int(clones.dex_positive.sum()),
                  n_donors=int(obs.donor.nunique()), methods=methods,
                  note="No-hit means no measured binding call for this panel, not verified nonreactivity")
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    characterize_clusters(obs, cell_leiden["fused"], targets,
                          args.expression_data, cluster_output)
    print(summary.to_string(index=False))
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
