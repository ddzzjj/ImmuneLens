#!/usr/bin/env python3
"""Leave-one-donor-out unsupervised RNA/TCR training and Dex evaluation.

Each fold fits RNA scaling and all three models on seven donors. Dextramer
calls are read only after training, to score distinct donor-specific clones.
The held-out donor is encoded by the trained models without adaptation.
The default input H5AD has a 2,000-gene panel selected globally when converted
from RDS; this script cannot undo that earlier unsupervised feature selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from immunelens.training import antigen_binding_representation as core
from immunelens.preprocessing.antigen_binding_data import (
    read_paired_cells, read_expression)


def clone_means(obs: pd.DataFrame, embeddings: dict[str, np.ndarray]):
    codes, clone_ids = pd.factorize(obs["clone_id"].astype(str), sort=True)
    counts = np.bincount(codes, minlength=len(clone_ids))
    grouped = obs.groupby("clone_id", sort=True, observed=True)
    clones = grouped.agg(
        donor=("donor", "first"),
        cdr3a=("cdr3a", "first"),
        cdr3b=("cdr3b", "first"),
        n_cells=("clone_id", "size"),
        n_dex_positive=("dex_positive", "sum"),
    ).reindex(clone_ids)
    if (grouped["donor"].nunique() > 1).any():
        raise ValueError("clone_id must be donor-specific")
    clones["dex_positive"] = clones["n_dex_positive"] > 0
    means = {}
    for method, z in embeddings.items():
        total = np.zeros((len(clone_ids), z.shape[1]), dtype=np.float32)
        np.add.at(total, codes, z)
        means[method] = total / counts[:, None]
    return clones, means


def score_fold(clones: pd.DataFrame, means: dict[str, np.ndarray],
               heldout: str, neighbors: int) -> pd.DataFrame:
    donor = clones["donor"].astype(str).to_numpy()
    positive = clones["dex_positive"].to_numpy(dtype=bool)
    test = np.flatnonzero(donor == heldout)
    gallery = np.flatnonzero(donor != heldout)
    if len(test) < 3 or len(gallery) < 2 or not positive[test].any():
        raise ValueError(f"Insufficient held-out clones or Dex+ calls for donor {heldout}")
    receptor = list(zip(clones["cdr3a"].astype(str), clones["cdr3b"].astype(str)))
    trained_receptors = {receptor[i] for i in gallery}
    novel = np.asarray([receptor[i] not in trained_receptors for i in test])
    subsets = {"all": np.ones(len(test), dtype=bool),
               "unseen_paired_cdr3": novel}
    rows = []
    for method, z in means.items():
        # Query the unseen donor against training donors. All reference labels
        # are used here only after the encoder has been trained and frozen.
        k = min(neighbors, len(gallery))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(z[gallery])
        ix = nn.kneighbors(z[test], return_distance=False)
        fractions = positive[gallery][ix].mean(axis=1)
        y_all = positive[test]
        expected = float(positive[gallery].mean())
        for name, mask in subsets.items():
            y, scores = y_all[mask], fractions[mask]
            if not y.any():
                continue
            rows.append(dict(
                heldout_donor=heldout, scope="cross_donor", method=method,
                query_subset=name, n_test_clones=len(y),
                n_test_dex_clones=int(y.sum()),
                n_gallery_clones=len(gallery), k=k,
                positive_query_neighbor_fraction=float(scores[y].mean()),
                no_hit_query_neighbor_fraction=float(scores[~y].mean()) if (~y).any() else np.nan,
                random_expected_fraction=expected,
                excess=float(scores[y].mean() - expected),
                average_precision=float(average_precision_score(y, scores))
                if (~y).any() else np.nan,
                test_dex_prevalence=float(y.mean()),
            ))

        # Also check whether the unseen donor's Dex+ clones group together.
        k = min(neighbors, len(test) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(z[test])
        ix = nn.kneighbors(X=None, return_distance=False)
        fractions = positive[test][ix].mean(axis=1)
        expected = float((int(y_all.sum()) - 1) / (len(test) - 1))
        for name, mask in subsets.items():
            y, scores = y_all[mask], fractions[mask]
            if not y.any():
                continue
            rows.append(dict(
                heldout_donor=heldout, scope="within_heldout_donor", method=method,
                query_subset=name, n_test_clones=len(y),
                n_test_dex_clones=int(y.sum()),
                n_gallery_clones=len(test) - 1, k=k,
                positive_query_neighbor_fraction=float(scores[y].mean()),
                no_hit_query_neighbor_fraction=float(scores[~y].mean()) if (~y).any() else np.nan,
                random_expected_fraction=expected,
                excess=float(scores[y].mean() - expected),
                average_precision=float(average_precision_score(y, scores))
                if (~y).any() else np.nan,
                test_dex_prevalence=float(y.mean()),
            ))
    return pd.DataFrame(rows)


def fit_one_fold(args, heldout: str, fold_index: int, rows: np.ndarray,
                 cdr3a: list[str], cdr3b: list[str], obs: pd.DataFrame,
                 expression: np.ndarray, genes: list[str], gene_selection_note: str,
                 output: Path,
                 device: torch.device):
    donors = obs["donor"].astype(str).to_numpy()
    test_pos = np.flatnonzero(donors == heldout)
    train_pos = np.flatnonzero(donors != heldout)
    if len(test_pos) == 0 or len(train_pos) == 0:
        raise ValueError(f"Empty training or test split for {heldout}")
    if np.intersect1d(train_pos, test_pos).size:
        raise AssertionError("Train/test donors overlap")
    fold_seed = args.seed + fold_index
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fold_seed)
    print(f"\n[Fold {fold_index + 1}] held-out donor={heldout}; "
          f"train={len(train_pos):,} cells, test={len(test_pos):,} cells")
    print("[Labels] Dex labels are excluded from training and preprocessing")

    # All preprocessing statistics are fitted on the seven training donors.
    mean = expression[train_pos].mean(axis=0)
    std = np.maximum(expression[train_pos].std(axis=0), 1e-3)
    x = np.clip((expression - mean) / std, -10, 10).astype(np.float32)
    train_x = x[train_pos]
    train_a = [cdr3a[i] for i in train_pos]
    train_b = [cdr3b[i] for i in train_pos]
    train_clones = obs.iloc[train_pos]["clone_id"].astype(str).tolist()
    graph = None
    if args.tcrdist_weight > 0:
        graph = core.build_tcrdist_graph(
            obs.iloc[train_pos].reset_index(drop=True),
            args.tcrdist_neighbors, args.tcrdist_quantile,
            args.tcrdist_max_clones, fold_seed,
        )

    rna = core.train_rna(
        train_x, train_clones, args.train_sampling, fold_seed,
        args.rna_epochs, args.batch_size, device, output / "rna_only_encoder.pt",
    )
    embeddings = {
        "rna_only": core.extract_rna_only(rna, x, args.embed_batch_size, device),
    }
    rna.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fused = core.train_fusion(
        train_x, train_a, train_b, train_clones,
        args.train_sampling, fold_seed, rna,
        args.fused_epochs, args.batch_size, device,
        output / "fused_encoder.pt", tcrdist_graph=graph,
        tcrdist_weight=args.tcrdist_weight,
        tcrdist_pairs_per_step=args.tcrdist_pairs_per_step,
        tcrdist_warmup_epochs=args.tcrdist_warmup_epochs,
        tcrdist_variance_floor=args.tcrdist_variance_floor,
        tcrdist_variance_weight=args.tcrdist_variance_weight,
    )
    embeddings["fused"] = core.extract_fused(
        fused, x, cdr3a, cdr3b, args.embed_batch_size, device,
    )
    del fused, rna
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tcr = core.train_tcr_only(
        train_a, train_b, args.pretrain_epochs,
        args.batch_size, device, fold_seed,
    )
    embeddings["tcr_only"] = core.extract_tcr_only(
        tcr, cdr3a, cdr3b, args.embed_batch_size,
    )
    torch.save(tcr.state_dict(), output / "tcr_only_encoder.pt")
    del tcr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The first access to dex_label happens only after training all models.
    annotated = obs.copy()
    annotated["dex_positive"] = annotated["dex_label"].astype(str).str.lower().eq("positive")
    clones, means = clone_means(annotated, embeddings)
    metrics = score_fold(clones, means, heldout, args.neighbors)
    metrics.to_csv(output / "metrics.csv", index=False)

    exported_obs = annotated.copy()
    exported_obs["split"] = np.where(donors == heldout, "heldout", "training")
    exported_obs["original_row"] = rows
    exported_obs.index = pd.Index(exported_obs["barcode"].astype(str), name="barcode")
    exported_obs = exported_obs.drop(columns=["barcode"])
    primary = embeddings["fused"]
    data = ad.AnnData(X=primary, obs=exported_obs,
                      var=pd.DataFrame(index=[f"latent_{i}" for i in range(primary.shape[1])]))
    for method, z in embeddings.items():
        data.obsm[f"X_{method}"] = z
    data.uns["heldout_donor"] = heldout
    data.uns["training_donors"] = sorted(set(donors) - {heldout})
    data.uns["label_usage"] = "Dextramer labels used for evaluation only"
    data.uns["tcr_private_weight"] = args.tcr_private_weight
    data.uns["input_gene_selection"] = gene_selection_note
    data.write_h5ad(output / "fold_embeddings.h5ad", compression="gzip")
    np.savez_compressed(output / "rna_scaler.npz", mean=mean, std=std)
    (output / "genes.json").write_text(json.dumps(genes))
    config = vars(args).copy()
    config.update(heldout_donor=heldout, fold_seed=fold_seed,
                  training_donors=sorted(set(donors) - {heldout}),
                  n_train_cells=len(train_pos), n_heldout_cells=len(test_pos),
                  labels_used_in_training=False, scaler_fit_on="training donors only",
                  input_gene_selection=gene_selection_note,
                  source_model=str(Path(core.__file__).resolve()))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(metrics[["scope", "method", "query_subset", "positive_query_neighbor_fraction",
                   "random_expected_fraction", "excess", "average_precision"]].to_string(index=False))
    del x, train_x, embeddings, means, data
    gc.collect()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get(
        "IMMUNELENS_ECCITE_DATA", "data/eccite/PBMC_vaccine_ECCITE_TCR.h5ad"))
    parser.add_argument("--tcrbert_dir",
                        default=os.environ.get("IMMUNELENS_TCRBERT_DIR", "models/tcrbert"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heldout_donors", default="all",
                        help="all or comma-separated donor IDs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--train_sampling", choices=("clone_balanced", "cell"),
                        default="clone_balanced")
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--rna_epochs", type=int, default=20)
    parser.add_argument("--fused_epochs", type=int, default=20)
    parser.add_argument("--tcr_private_weight", type=float, default=0.6)
    parser.add_argument("--tcrdist_weight", type=float, default=0.0)
    parser.add_argument("--tcrdist_neighbors", type=int, default=3)
    parser.add_argument("--tcrdist_quantile", type=float, default=0.05)
    parser.add_argument("--tcrdist_pairs_per_step", type=int, default=16)
    parser.add_argument("--tcrdist_warmup_epochs", type=int, default=5)
    parser.add_argument("--tcrdist_max_clones", type=int, default=5000)
    parser.add_argument("--tcrdist_variance_floor", type=float, default=0.05)
    parser.add_argument("--tcrdist_variance_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_batch_size", type=int, default=64)
    args = parser.parse_args()
    if min(args.neighbors, args.rna_epochs, args.fused_epochs,
           args.pretrain_epochs, args.batch_size, args.embed_batch_size) < 1:
        parser.error("neighbor counts, epochs and batch sizes must be positive")
    if (args.tcr_private_weight < 0 or args.tcrdist_weight < 0
            or args.tcrdist_neighbors < 1 or not 0 < args.tcrdist_quantile <= 1
            or args.tcrdist_pairs_per_step < 2 or args.tcrdist_warmup_epochs < 1
            or args.tcrdist_max_clones < 3 or args.tcrdist_variance_floor < 0
            or args.tcrdist_variance_weight < 0):
        parser.error("invalid fusion or TCRdist settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA requested but unavailable: {args.device}")
    if not Path(args.data).is_file():
        parser.error(f"input does not exist: {args.data}")
    device = torch.device(args.device)
    core.DEVICE = device
    core.TCRBERT_DIR = args.tcrbert_dir
    core.UMAP_TCR_PRIV_WEIGHT = args.tcr_private_weight
    rows, cdr3a, cdr3b, obs, _ = read_paired_cells(args.data, args.min_genes)
    obs["donor"] = obs["donor"].astype(str)
    donors = sorted(obs["donor"].unique().tolist())
    selected = donors if args.heldout_donors == "all" else [
        x.strip() for x in args.heldout_donors.split(",") if x.strip()]
    if not selected or len(set(selected)) != len(selected) or any(d not in donors for d in selected):
        parser.error(f"--heldout_donors must be unique IDs from {donors}")
    expression, genes = read_expression(args.data, rows.tolist())
    source = ad.read_h5ad(args.data, backed="r")
    try:
        gene_selection_note = str(source.uns.get("gene_selection", "unspecified"))
    finally:
        source.file.close()
    print(f"[Gene panel] {gene_selection_note}")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for heldout in selected:
        index = donors.index(heldout)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", heldout)
        fold_out = output / f"fold_{index + 1:02d}_{slug}"
        if fold_out.exists() and any(fold_out.iterdir()):
            required = ("metrics.csv", "config.json", "fold_embeddings.h5ad")
            if not all((fold_out / name).is_file() for name in required):
                raise FileExistsError(f"Incomplete fold directory: {fold_out}; use a fresh output path")
            previous = json.loads((fold_out / "config.json").read_text())
            requested = vars(args).copy()
            if any(previous.get(k) != v for k, v in requested.items()):
                raise ValueError(f"Existing fold has different parameters: {fold_out}")
            print(f"[Resume] Reusing completed fold {heldout}")
            all_metrics.append(pd.read_csv(fold_out / "metrics.csv"))
            continue
        fold_out.mkdir(parents=True, exist_ok=True)
        all_metrics.append(fit_one_fold(
            args, heldout, index, rows, cdr3a, cdr3b, obs,
            expression, genes, gene_selection_note, fold_out, device,
        ))
    detail = pd.concat(all_metrics, ignore_index=True)
    detail.to_csv(output / "leave_one_donor_out_detail.csv", index=False)
    summary = detail.groupby(["scope", "method", "query_subset"], as_index=False).agg(
        n_heldout_donors=("heldout_donor", "nunique"),
        mean_positive_neighbor_fraction=("positive_query_neighbor_fraction", "mean"),
        mean_no_hit_neighbor_fraction=("no_hit_query_neighbor_fraction", "mean"),
        mean_random_expected_fraction=("random_expected_fraction", "mean"),
        mean_excess=("excess", "mean"),
        mean_average_precision=("average_precision", "mean"),
        mean_test_dex_prevalence=("test_dex_prevalence", "mean"),
    )
    summary.to_csv(output / "leave_one_donor_out_summary.csv", index=False)
    print("\n[Summary] Each held-out donor has equal weight")
    print(summary.to_string(index=False))
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()

