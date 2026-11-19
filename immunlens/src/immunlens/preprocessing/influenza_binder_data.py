"""Validate clone-disjoint influenza tasks and prepare train-only RNA features."""

import json
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def read_fixed_task(task: str, dataset_dir: str):
    root = os.path.join(dataset_dir, task)
    cells = pd.read_csv(os.path.join(root, "cells.csv"))
    clones = pd.read_csv(os.path.join(root, "clones.csv"))
    with open(os.path.join(root, "manifest.json")) as handle:
        manifest = json.load(handle)
    required = {"donor", "antigen", "barcode", "cdr3a", "cdr3b", "clone_id", "label", "split"}
    if not required.issubset(cells.columns):
        raise ValueError(f"{task}: missing cell columns {required - set(cells.columns)}")
    clone_required = {"donor", "antigen", "clone_id", "cdr3a", "cdr3b", "split",
                      "n_cells", "n_positive_cells", "n_negative_cells", "mixed_label"}
    if not clone_required.issubset(clones.columns):
        raise ValueError(f"{task}: incomplete clone table")
    if cells.barcode.duplicated().any() or clones.clone_id.duplicated().any():
        raise ValueError(f"{task}: duplicate cell or clone")
    if cells.donor.nunique() != 1 or cells.antigen.nunique() != 1:
        raise ValueError(f"{task}: expected one donor and one antigen")
    if cells.donor.iloc[0] != manifest["donor"] or cells.antigen.iloc[0] != manifest["antigen"]:
        raise ValueError(f"{task}: manifest disagrees with cell table")
    if cells.antigen.iloc[0] != "A0201_GILGFVFTL_Flu-MP_Influenza":
        raise ValueError(f"{task}: expected influenza GILGFVFTL")
    if cells.donor.iloc[0] != task.split("_", 1)[0]:
        raise ValueError(f"{task}: donor does not match task name")
    if set(cells.split) != {"train", "val", "test"}:
        raise ValueError(f"{task}: missing split")
    if not cells.label.isin([0, 1]).all() or len(cells.groupby(["split", "label"])) != 6:
        raise ValueError(f"{task}: every split needs both cell labels")
    verified = cells.merge(clones[["clone_id", "split", "cdr3a", "cdr3b"]],
                           on="clone_id", validate="many_to_one", suffixes=("", "_clone"))
    if len(verified) != len(cells) or not (
        verified.split.eq(verified.split_clone).all()
        and verified.cdr3a.eq(verified.cdr3a_clone).all()
        and verified.cdr3b.eq(verified.cdr3b_clone).all()
    ):
        raise ValueError(f"{task}: cell and clone tables disagree")
    if clones.groupby("clone_id").split.nunique().max() != 1:
        raise ValueError(f"{task}: clone crosses splits")
    observed = cells.groupby("clone_id").label.agg(n_cells="size", n_positive_cells="sum")
    expected = clones.set_index("clone_id")[["n_cells", "n_positive_cells"]]
    if not observed.sort_index().equals(expected.sort_index()):
        raise ValueError(f"{task}: clone cell counts disagree")
    cells["label"] = cells.label.astype(int)
    return cells, clones, manifest


def prepare_task_rna(cells: pd.DataFrame, out_dir: str, rna_dir: str, reuse_preprocessing: bool = False):
    donor = cells.donor.iloc[0]
    path = os.path.join(rna_dir, donor, "rna.h5ad")
    backed = ad.read_h5ad(path, backed="r")
    barcodes = cells.barcode.astype(str).tolist()
    missing = pd.Index(barcodes).difference(backed.obs_names)
    if len(missing):
        backed.file.close()
        raise ValueError(f"{donor}: {len(missing)} barcodes missing from RNA")
    one = backed[barcodes, :].to_memory()
    backed.file.close()
    raw = one.X.tocsr().astype(np.float32) if sparse.issparse(one.X) else sparse.csr_matrix(one.X, dtype=np.float32)
    library = np.asarray(raw.sum(axis=1)).ravel()
    if (library <= 0).any():
        raise ValueError("RNA contains zero-library cells")
    normalized = raw.multiply((1e4 / library)[:, None]).tocsr()
    np.log1p(normalized.data, out=normalized.data)
    preprocessing_path = os.path.join(out_dir, "rna_preprocessing.npz")
    if reuse_preprocessing:
        with np.load(preprocessing_path, allow_pickle=False) as saved:
            genes = saved["genes"].astype(str)
            mu = saved["mean"]
            sd = saved["std"]
        selected = one.var_names.get_indexer(genes)
        if (selected < 0).any():
            raise ValueError("Saved RNA genes are missing from the donor RNA file")
        x = normalized[:, selected].toarray().astype(np.float32)
        x = np.clip((x - mu) / sd, -10.0, 10.0).astype(np.float32)
        print(f"[RNA] reused {len(genes)} saved genes for {donor}", flush=True)
        return x
    train_mask = cells.split.to_numpy() == "train"
    train = normalized[train_mask]
    pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", re.IGNORECASE)
    present = np.asarray((train > 0).sum(axis=0)).ravel()
    allowed = np.array([not pattern.match(g) for g in one.var_names]) & (present >= 3)
    mean = np.asarray(train.mean(axis=0)).ravel()
    variance = np.asarray(train.power(2).mean(axis=0)).ravel() - mean ** 2
    variance[~allowed] = -np.inf
    n_genes = min(2000, int(allowed.sum()))
    if n_genes < 1:
        raise ValueError("No genes passed the train-only filter")
    selected = np.sort(np.argsort(variance)[-n_genes:])
    genes = one.var_names[selected].tolist()
    x = normalized[:, selected].toarray().astype(np.float32)
    mu = x[train_mask].mean(axis=0)
    sd = x[train_mask].std(axis=0)
    sd[sd < 1e-6] = 1.0
    x = np.clip((x - mu) / sd, -10.0, 10.0).astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "rna_preprocessing.npz"),
                        genes=np.asarray(genes), mean=mu, std=sd)
    print(f"[RNA] {donor}: {len(cells)} cells; {len(genes)} train-selected genes", flush=True)
    return x
