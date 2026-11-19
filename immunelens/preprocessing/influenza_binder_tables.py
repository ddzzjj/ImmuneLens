#!/usr/bin/env python3
"""Build fixed-pMHC cell binder tasks with clone-disjoint splits.

All valid paired TRA/TRB cells are retained, including clones whose cells have
different binder calls. The prediction target is each cell's measured call;
the split group is its exact paired CDR3 alpha/beta sequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


TASKS = {
    "donor1_flu_gilgf": ("donor1", "A0201_GILGFVFTL_Flu-MP_Influenza"),
    "donor1_cmv_klgg": ("donor1", "A0301_KLGGALQAK_IE-1_CMV"),
    "donor2_flu_gilgf": ("donor2", "A0201_GILGFVFTL_Flu-MP_Influenza"),
    "donor2_cmv_klgg": ("donor2", "A0301_KLGGALQAK_IE-1_CMV"),
    "donor3_cmv_klgg": ("donor3", "A0301_KLGGALQAK_IE-1_CMV"),
    "donor4_cmv_klgg": ("donor4", "A0301_KLGGALQAK_IE-1_CMV"),
}


def balanced_clone_split(clones: pd.DataFrame, seed: int, n_candidates: int = 2000):
    """Stratify clones, then balance cell labels without using model predictions."""
    groups = [np.flatnonzero(clones.has_positive.to_numpy() == label)
              for label in (0, 1)]
    positive = clones.n_positive_cells.to_numpy()
    negative = clones.n_negative_cells.to_numpy()
    target = np.array([0.7, 0.1, 0.2])
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_split = None
    best_candidate = None
    for candidate in range(n_candidates):
        split = np.empty(len(clones), dtype=np.int8)
        for group in groups:
            shuffled = rng.permutation(group)
            train_end = round(len(group) * 0.7)
            val_end = round(len(group) * 0.8)
            split[shuffled[:train_end]] = 0
            split[shuffled[train_end:val_end]] = 1
            split[shuffled[val_end:]] = 2
        positive_share = np.bincount(split, weights=positive, minlength=3) / positive.sum()
        negative_share = np.bincount(split, weights=negative, minlength=3) / negative.sum()
        score = max(np.max(np.abs(positive_share - target)),
                    np.max(np.abs(negative_share - target)))
        if score < best_score:
            best_score, best_split, best_candidate = score, split.copy(), candidate
    return np.array(["train", "val", "test"])[best_split], best_candidate, best_score


def build_task(data_dir: Path, output_dir: Path, task: str, seed: int) -> dict:
    donor, antigen = TASKS[task]
    binder = antigen + "_binder"
    source = data_dir / donor / "epi.csv"
    frame = pd.read_csv(source, usecols=["barcode", "cell_clono_cdr3_aa", binder])
    if frame.barcode.duplicated().any():
        raise ValueError(f"{donor}: duplicate source barcodes")
    source_cells = len(frame)
    source_positive_cells = int(frame[binder].eq(1).sum())

    chains = frame.cell_clono_cdr3_aa.fillna("").astype(str)
    paired = chains.str.count("TRA:").eq(1) & chains.str.count("TRB:").eq(1)
    frame = frame.loc[paired].copy()
    parts = frame.cell_clono_cdr3_aa.str.split(";", expand=True)
    if parts.shape[1] != 2:
        raise ValueError(f"{donor}: unexpected paired-chain format")
    first, second = parts[0].str.strip(), parts[1].str.strip()
    alpha = first.where(first.str.startswith("TRA:"), second).str.removeprefix("TRA:")
    beta = first.where(first.str.startswith("TRB:"), second).str.removeprefix("TRB:")
    valid = alpha.str.len().ge(5) & beta.str.len().ge(5)
    frame = frame.loc[valid].copy()
    frame["cdr3a"] = alpha.loc[valid]
    frame["cdr3b"] = beta.loc[valid]
    frame["clone_id"] = frame.cdr3a + "|" + frame.cdr3b
    frame["label"] = pd.to_numeric(frame[binder], errors="raise").astype(int)
    if not frame.label.isin([0, 1]).all():
        raise ValueError(f"{donor}: target binder column is not binary")

    backed = ad.read_h5ad(data_dir / donor / "rna.h5ad", backed="r")
    try:
        missing = pd.Index(frame.barcode).difference(backed.obs_names)
    finally:
        backed.file.close()
    if len(missing):
        raise ValueError(f"{donor}: {len(missing)} paired barcodes missing from RNA")

    clones = frame.groupby("clone_id", sort=False).agg(
        cdr3a=("cdr3a", "first"), cdr3b=("cdr3b", "first"),
        n_cells=("label", "size"), n_positive_cells=("label", "sum"),
    ).reset_index()
    clones["n_negative_cells"] = clones.n_cells - clones.n_positive_cells
    clones["positive_fraction"] = clones.n_positive_cells / clones.n_cells
    clones["has_positive"] = clones.n_positive_cells.gt(0).astype(int)
    clones["mixed_label"] = clones.n_positive_cells.gt(0) & clones.n_negative_cells.gt(0)
    if clones.has_positive.value_counts().min() < 20:
        raise ValueError(f"{task}: fewer than 20 clones in a split stratum")

    clones["split"], split_candidate, split_balance_error = balanced_clone_split(clones, seed)
    frame = frame.merge(clones[["clone_id", "split"]], on="clone_id",
                        how="left", validate="many_to_one")
    frame.insert(0, "donor", donor)
    frame.insert(1, "antigen", antigen)
    clones.insert(0, "donor", donor)
    clones.insert(1, "antigen", antigen)
    if frame.split.isna().any() or frame.groupby("clone_id").split.nunique().max() != 1:
        raise AssertionError(f"{task}: clone leakage or missing split")
    if frame.barcode.duplicated().any():
        raise AssertionError(f"{task}: duplicate output barcode")

    summary = []
    for split in ("train", "val", "test"):
        cell_part = frame.loc[frame.split.eq(split)]
        clone_part = clones.loc[clones.split.eq(split)]
        summary.append({
            "split": split,
            "n_cells": len(cell_part),
            "n_positive_cells": int(cell_part.label.sum()),
            "positive_cell_fraction": float(cell_part.label.mean()),
            "n_clones": len(clone_part),
            "n_clones_with_positive_call": int(clone_part.has_positive.sum()),
            "n_mixed_label_clones": int(clone_part.mixed_label.sum()),
        })
    if any(row["n_positive_cells"] == 0 or row["n_positive_cells"] == row["n_cells"]
           for row in summary):
        raise ValueError(f"{task}: a split has only one cell label")

    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    frame[["donor", "antigen", "barcode", "cdr3a", "cdr3b", "clone_id",
           "label", "split"]].sort_values(["split", "clone_id", "barcode"]).to_csv(
               task_dir / "cells.csv", index=False)
    clones.sort_values(["split", "clone_id"]).to_csv(task_dir / "clones.csv", index=False)
    pd.DataFrame(summary).to_csv(task_dir / "split_summary.csv", index=False)
    manifest = {
        "task": task, "donor": donor, "antigen": antigen, "source": str(source),
        "seed": seed, "source_cells": source_cells,
        "source_positive_cells": source_positive_cells,
        "paired_valid_cells": len(frame),
        "paired_valid_positive_cells": int(frame.label.sum()),
        "included_clones": len(clones),
        "mixed_label_clones_retained": int(clones.mixed_label.sum()),
        "mixed_label_cells_retained": int(clones.loc[clones.mixed_label, "n_cells"].sum()),
        "mixed_label_positive_cells_retained": int(clones.loc[clones.mixed_label, "n_positive_cells"].sum()),
        "prediction_unit": "cell; original target pMHC binder call",
        "generalization_group": "exact paired CDR3alpha|CDR3beta sequence within donor",
        "split_definition": "clone-stratified 70% train, 10% val, 20% test by any-positive status; choose the most cell-label-balanced of 2000 deterministic candidates",
        "split_candidate": split_candidate,
        "maximum_positive_or_negative_cell_share_error": float(split_balance_error),
        "clone_label_policy": "No binary clone label assigned; mixed-label clones retained",
        "split_counts": summary,
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{task}: {len(frame):,} cells, {int(frame.label.sum()):,} positive, "
          f"{len(clones):,} clones, {int(clones.mixed_label.sum()):,} mixed clones")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("IMMUNELENS_EPIBIND_RNA_DIR", "data/10x")))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(os.environ.get(
            "IMMUNELENS_EPIBIND_DATASET_DIR", "results/epidata_cell_binder_v2")))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [build_task(args.data_dir, args.out_dir, name, args.seed) for name in TASKS]
    (args.out_dir / "dataset_manifest.json").write_text(
        json.dumps({"tasks": manifests}, indent=2) + "\n")


if __name__ == "__main__":
    main()
