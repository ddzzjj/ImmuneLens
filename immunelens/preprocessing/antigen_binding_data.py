"""Read and validate paired-chain ECCITE-seq cells and expression."""

import anndata as ad
import numpy as np
import pandas as pd

from immunelens.preprocessing.sequences import clean_aa


def read_expression(path, cells):
    data = ad.read_h5ad(path, backed="r")
    try:
        selected = data.X[np.asarray(cells, dtype=int), :]
        expression = selected.toarray() if hasattr(selected, "toarray") else np.asarray(selected)
        names = data.var_names.astype(str).tolist()
    finally:
        data.file.close()
    expression = np.nan_to_num(np.asarray(expression, dtype=np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
    return expression, names


def read_paired_cells(path: str, min_genes: int):
    data = ad.read_h5ad(path, backed="r")
    try:
        obs = data.obs.copy()
        matrix_shape = data.shape
    finally:
        data.file.close()
    required = {"donor", "clone_id", "clonotype", "clonotype_label",
                "cdr3a", "cdr3b", "n_genes", "dex_label", "virus_label"}
    missing = required - set(obs)
    if missing:
        raise ValueError(f"ECCITE H5AD is missing columns: {sorted(missing)}")
    alpha = obs["cdr3a"].astype(object).fillna("").astype(str).map(clean_aa)
    beta = obs["cdr3b"].astype(object).fillna("").astype(str).map(clean_aa)
    donor = obs["donor"].astype(object).fillna("").astype(str)
    valid = ((alpha.str.len() >= 5) & (beta.str.len() >= 5)
             & (pd.to_numeric(obs["n_genes"], errors="coerce") >= min_genes)
             & donor.ne("") & donor.ne("nan"))
    rows = np.flatnonzero(valid.to_numpy())
    if len(rows) == 0:
        raise RuntimeError("No paired TRA/TRB cells passed the filters")
    selected = obs.iloc[rows].copy().reset_index().rename(columns={"barcode": "barcode"})
    if "barcode" not in selected:
        selected = selected.rename(columns={selected.columns[0]: "barcode"})
    selected["cdr3a"] = selected["cdr3a"].astype(object)
    selected["cdr3b"] = selected["cdr3b"].astype(object)
    selected["cdr3a"] = alpha.iloc[rows].to_numpy()
    selected["cdr3b"] = beta.iloc[rows].to_numpy()
    return rows, selected.cdr3a.tolist(), selected.cdr3b.tolist(), selected, matrix_shape
