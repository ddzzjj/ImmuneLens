#!/usr/bin/env python3
# -*- coding: utf-8 -*-



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
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D



H5AD_PATH = "./result/luadvae/adata_nsclc_icb.h5ad"
OUT_DIR = "./fig_rna5_fused_story_final_with_fused_leiden"

RNA_CLUSTER_COL = "RNA_cluster"
FINE_COL = "fine_cluster"
SAMPLE_COL = "sampleID"
GROUP_COL = "Response_binary"
TRB_COL = "TRB_aa"

RNA_REP = "X_rna"
FUSED_REP = "X_final"

POS_GROUP = "MPR"
NEG_GROUP = "non-MPR"

TOP_FUSED_N = 4
MIN_RNA_CLUSTER_CELLS = 500
MIN_CD8_PROP = 0.30

N_NEIGHBORS = 30
MIN_CLONE_SIZE_FOR_METRICS = 2
N_PERM = 200
RANDOM_SEED = 1337

SAVE_PNG = True
SAVE_PDF = True

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 160)



PROGRAMS = {
    "exhaustion": ["CXCL13", "PDCD1", "TOX", "LAG3", "HAVCR2", "TIGIT", "ENTPD1"],
    "cytotoxic": ["GZMB", "PRF1", "GNLY", "NKG7", "GZMA", "IFNG"],
    "proliferation": ["MKI67", "TOP2A", "STMN1", "TYMS", "UBE2C"],
    "memory": ["TCF7", "IL7R", "CCR7", "LEF1", "SELL"],
    "trm": ["ITGAE", "CXCR6", "ZNF683", "CD69", "ITGA1"],
    "activation": ["CD69", "TNFRSF9", "ICOS", "CD38", "HLA-DRA"],
}

FOCUS_GENES = [
    "CXCL13", "PDCD1", "TOX",
    "LAG3", "HAVCR2", "TIGIT", "ENTPD1",
    "GZMB", "PRF1", "NKG7", "GNLY", "IFNG",
    "MKI67", "TCF7", "IL7R"
]

PROGRAM_DISPLAY = [
    ("score_exhaustion", "Exhaustion"),
    ("score_cytotoxic", "Cytotoxic"),
    ("score_proliferation", "Proliferation"),
    ("score_memory", "Memory"),
    ("score_trm", "TRM"),
    ("score_activation", "Activation"),
]



INK = "#1A1A1A"
LIGHT_GREY = "#D9DDE1"
VERY_LIGHT_GREY = "#EEEEEE"

RNA_COLOR = "#A6B1BC"
FUSED_COLOR = "#9E2B25"

MPR_COLOR = "#2E6E8E"
NONMPR_COLOR = "#B2483B"
OTHER_COLOR = "#D8D8D8"

RESP_COLORS = {
    "MPR": MPR_COLOR,
    "non-MPR": NONMPR_COLOR,
}

CLUSTER_PALETTE = [
    "#4C6E91",
    "#D58A4C",
    "#5E8C61",
    "#A24B57",
    "#6B5B95",
    "#3D8E8C",
    "#B0823B",
    "#876B8E",
]

METHOD_COLORS = {
    "RNAonly_matched": RNA_COLOR,
    "fused_existing": FUSED_COLOR,
    "fused_matched": "#E0A458",
}


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def despine(ax, keep=("left", "bottom")):
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(s in keep)


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    if SAVE_PDF:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    if SAVE_PNG:
        fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)


def clabel(x):
    return f"F{x}" if str(x) != "Other" else "Other"


def qstar(q):
    if q is None or pd.isna(q):
        return "ns"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 5e-2:
        return "*"
    return "ns"


def sort_key(x):
    x = str(x)
    try:
        return (0, int(x))
    except Exception:
        return (1, x)



def clean_series(s):
    return s.astype(str).replace({
        "nan": "NA",
        "NaN": "NA",
        "None": "NA",
        "": "NA",
    })

def parse_response_rate(x):

    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s in ["", "NA", "nan", "NaN", "None"]:
        return np.nan

    s = s.replace("%", "")
    try:
        v = float(s)
    except Exception:
        return np.nan

    if 0 <= v <= 1:
        v = v * 100.0

    return v


def check_response_label_logic(ad, tag="global"):

    print("\n" + "=" * 100)
    print(f"[Response label check] {tag}")
    print("=" * 100)

    obs = ad.obs.copy()


    candidate_cols = [
        "Response",
        "Response_binary",
        "Path_response",
        "pathological_response",
        "pathological_response_rate",
        "radiological_response",
        "PatientID",
        "sampleID",
    ]

    existing = [c for c in candidate_cols if c in obs.columns]

    print("\n[Existing response-related columns]")
    print(existing)


    for c in existing:
        if c in ["PatientID", "sampleID"]:
            continue

        print(f"\n[{c} value counts]")
        print(obs[c].astype(str).value_counts(dropna=False).head(30).to_string())

    if "Response" in obs.columns and "Response_binary" in obs.columns:
        print("\n[Cross-tab: Response vs Response_binary]")
        tab = pd.crosstab(
            obs["Response"].astype(str),
            obs["Response_binary"].astype(str),
            dropna=False
        )
        print(tab.to_string())

    if "Path_response" in obs.columns and "Response_binary" in obs.columns:
        print("\n[Cross-tab: Path_response vs Response_binary]")
        tab = pd.crosstab(
            obs["Path_response"].astype(str),
            obs["Response_binary"].astype(str),
            dropna=False
        )
        print(tab.to_string())

    if "pathological_response" in obs.columns and "Response_binary" in obs.columns:
        print("\n[Cross-tab: pathological_response vs Response_binary]")
        tab = pd.crosstab(
            obs["pathological_response"].astype(str),
            obs["Response_binary"].astype(str),
            dropna=False
        )
        print(tab.to_string())

    id_cols = [c for c in ["PatientID", "sampleID"] if c in obs.columns]

    if "Response_binary" in obs.columns:
        for id_col in id_cols:
            tmp = (
                obs[[id_col, "Response_binary"]]
                .drop_duplicates()
                .copy()
            )

            n_resp = tmp.groupby(id_col)["Response_binary"].nunique()
            bad_ids = n_resp[n_resp > 1]

            print(f"\n[{id_col}-level Response_binary consistency]")
            print(f"n_{id_col} = {n_resp.shape[0]}")
            print(f"n_inconsistent_{id_col} = {bad_ids.shape[0]}")

            if bad_ids.shape[0] > 0:
                print(f"\n[WARNING] Inconsistent Response_binary within {id_col}:")
                show = tmp[tmp[id_col].isin(bad_ids.index)].sort_values(id_col)
                print(show.head(80).to_string(index=False))


    if "pathological_response_rate" in obs.columns and "Response_binary" in obs.columns:
        tmp = obs[["Response_binary", "pathological_response_rate"]].copy()
        tmp["pathological_response_rate_num"] = tmp["pathological_response_rate"].map(parse_response_rate)
        tmp = tmp.dropna(subset=["pathological_response_rate_num"])

        print("\n[Pathological response rate numeric summary]")
        if len(tmp) == 0:
            print("No valid numeric pathological_response_rate found.")
        else:
            summary = (
                tmp.groupby("Response_binary")["pathological_response_rate_num"]
                .agg(["count", "mean", "median", "min", "max"])
                .sort_index()
            )
            print(summary.to_string())

            if "MPR" in summary.index and "non-MPR" in summary.index:
                mpr_mean = summary.loc["MPR", "mean"]
                non_mean = summary.loc["non-MPR", "mean"]
                print(f"\n[Direction check] mean pathological_response_rate:")
                print(f"MPR     = {mpr_mean:.3f}")
                print(f"non-MPR = {non_mean:.3f}")

                if mpr_mean <= non_mean:
                    print("[WARNING] MPR mean response rate is not higher than non-MPR. Check label direction.")
                else:
                    print("[OK] MPR has higher pathological response rate than non-MPR.")


    show_cols = [c for c in [
        "PatientID",
        "sampleID",
        "Response",
        "Response_binary",
        "Path_response",
        "pathological_response",
        "pathological_response_rate",
    ] if c in obs.columns]

    if len(show_cols) > 0:
        print("\n[Patient/sample-level response table preview]")
        preview = (
            obs[show_cols]
            .drop_duplicates()
            .sort_values(show_cols[:1])
        )
        print(preview.head(100).to_string(index=False))
def check_selected_rna_fused_response(ad, selected_rna, focus_fines):

    print("\n" + "=" * 100)
    print("[Selected RNA / fused response check]")
    print("=" * 100)

    obs = ad.obs.copy()
    obs[RNA_CLUSTER_COL] = clean_series(obs[RNA_CLUSTER_COL])
    obs[FINE_COL] = clean_series(obs[FINE_COL])
    obs[GROUP_COL] = clean_series(obs[GROUP_COL])
    obs[SAMPLE_COL] = clean_series(obs[SAMPLE_COL])

    selected_rna = str(selected_rna)
    focus_fines = [str(x) for x in focus_fines]

    sub_rna = obs[obs[RNA_CLUSTER_COL].astype(str) == selected_rna].copy()

    print(f"\n[Selected RNA cluster]")
    print(f"RNA_cluster = {selected_rna}")
    print(f"n_cells     = {len(sub_rna)}")
    print(f"n_patients  = {sub_rna[SAMPLE_COL].nunique()}")

    print("\n[Response cell counts inside selected RNA cluster]")
    print(sub_rna[GROUP_COL].value_counts(dropna=False).to_string())

    print("\n[Fine_cluster counts inside selected RNA cluster]")
    print(sub_rna[FINE_COL].value_counts().head(20).to_string())

    sub_focus = sub_rna[sub_rna[FINE_COL].isin(focus_fines)].copy()

    print("\n[Focus fused clusters]")
    print(focus_fines)
    print(f"focus cells    = {len(sub_focus)}")
    print(f"focus patients = {sub_focus[SAMPLE_COL].nunique()}")

    print("\n[Cell-level Response_binary by fused cluster]")
    tab_cell = pd.crosstab(
        sub_focus[FINE_COL],
        sub_focus[GROUP_COL],
        margins=True
    )
    print(tab_cell.to_string())

    print("\n[Cell-level Response_binary fraction by fused cluster]")
    tab_cell_frac = pd.crosstab(
        sub_focus[FINE_COL],
        sub_focus[GROUP_COL],
        normalize="index"
    )
    print(tab_cell_frac.round(4).to_string())


    print("\n[Patient counts by fused cluster and response]")
    patient_presence = (
        sub_focus[[SAMPLE_COL, GROUP_COL, FINE_COL]]
        .drop_duplicates()
        .groupby([FINE_COL, GROUP_COL])
        .size()
        .reset_index(name="n_patients")
        .sort_values([FINE_COL, GROUP_COL])
    )
    print(patient_presence.to_string(index=False))


    print("\n[Top patient contributors per fused cluster]")
    for f in focus_fines:
        tmp = sub_focus[sub_focus[FINE_COL] == f]
        if len(tmp) == 0:
            continue

        print(f"\n--- F{f} ---")
        out = (
            tmp.groupby([SAMPLE_COL, GROUP_COL])
            .size()
            .reset_index(name="n_cells")
            .sort_values("n_cells", ascending=False)
            .head(15)
        )
        print(out.to_string(index=False))

    print("\n[Dominant patient fraction per fused cluster]")
    rows = []
    for f in focus_fines:
        tmp = sub_focus[sub_focus[FINE_COL] == f]
        if len(tmp) == 0:
            continue

        vc = tmp[SAMPLE_COL].value_counts()
        rows.append({
            "fine_cluster": f,
            "n_cells": int(len(tmp)),
            "n_patients": int(tmp[SAMPLE_COL].nunique()),
            "top_patient": str(vc.index[0]),
            "top_patient_cells": int(vc.iloc[0]),
            "top_patient_frac": float(vc.iloc[0] / len(tmp)),
        })

    dom = pd.DataFrame(rows)
    print(dom.to_string(index=False))

    if len(dom) > 0 and (dom["top_patient_frac"] > 0.5).any():
        print("\n[WARNING] Some fused clusters are dominated by one patient.")
    else:
        print("\n[OK] No focus fused cluster is dominated by a single patient >50%.")

def ensure_response_binary(ad):
    if GROUP_COL in ad.obs.columns:
        ad.obs[GROUP_COL] = clean_series(ad.obs[GROUP_COL])
        return ad

    if "Response" not in ad.obs.columns:
        raise ValueError("Need Response_binary or Response in ad.obs")

    def binarize(x):
        x = str(x).strip()
        if x in ["pCR", "MPR"]:
            return "MPR"
        if x == "non-MPR":
            return "non-MPR"
        return "NA"

    ad.obs[GROUP_COL] = ad.obs["Response"].map(binarize).astype(str)
    return ad


def to_1d(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def get_expression_df(ad, genes):
    if ad.raw is not None:
        var_names = pd.Index(ad.raw.var_names.astype(str))
        X = ad.raw.X
    else:
        var_names = pd.Index(ad.var_names.astype(str))
        X = ad.X

    out = {}
    for g in genes:
        if g in var_names:
            out[g] = to_1d(X[:, var_names.get_loc(g)])
        else:
            out[g] = np.full(ad.n_obs, np.nan)

    return pd.DataFrame(out, index=ad.obs_names)


def get_num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def zscore(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    sd = x.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean(skipna=True)) / (sd + 1e-9)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def is_cd8_like(obs):
    if "sub_cell_type" not in obs.columns:
        return pd.Series(False, index=obs.index)

    s = obs["sub_cell_type"].astype(str)
    return s.str.startswith("CD8T") | s.str.contains("CD8", case=False, na=False)


def add_scores(ad):
    all_genes = sorted(set(FOCUS_GENES + sum(PROGRAMS.values(), [])))
    expr = get_expression_df(ad, all_genes)

    used = {}

    for g in ["CXCL13", "PDCD1", "TOX"]:
        ad.obs[f"expr_{g}"] = expr[g].values

    for name, genes in PROGRAMS.items():
        present = [g for g in genes if g in expr.columns and not np.all(pd.isna(expr[g]))]
        ad.obs[f"score_{name}"] = expr[present].mean(axis=1).values if present else np.nan
        used[name] = f"{len(present)}/{len(genes)}"

    return ad, used


def score_rna_clusters(ad):
    rows = []

    for cl, idx in ad.obs.groupby(RNA_CLUSTER_COL, observed=True).groups.items():
        sub = ad.obs.loc[idx].copy()

        rows.append({
            "RNA_cluster": str(cl),
            "n_cells": int(len(sub)),
            "n_patients": int(sub[SAMPLE_COL].nunique()) if SAMPLE_COL in sub.columns else np.nan,
            "CD8_prop": float(is_cd8_like(sub).mean()),
            "score_exhaustion": float(get_num(sub, "score_exhaustion").mean()),
            "expr_CXCL13": float(get_num(sub, "expr_CXCL13").mean()),
            "expr_PDCD1": float(get_num(sub, "expr_PDCD1").mean()),
            "expr_TOX": float(get_num(sub, "expr_TOX").mean()),
            "expanded_ratio": float(get_num(sub, "is_expanded", 0).mean()),
        })

    df = pd.DataFrame(rows)

    valid = df["n_cells"] >= MIN_RNA_CLUSTER_CELLS
    valid_cd8 = valid & (df["CD8_prop"] >= MIN_CD8_PROP)
    if valid_cd8.sum() > 0:
        valid = valid_cd8

    df["eligible"] = valid.values
    df["selection_score"] = np.nan

    dfv = df.loc[valid].copy()
    dfv["selection_score"] = (
        1.0 * zscore(dfv["score_exhaustion"]).values
        + 0.6 * zscore(dfv["CD8_prop"]).values
        + 0.2 * zscore(np.log1p(dfv["n_cells"])).values
    )

    df.loc[dfv.index, "selection_score"] = dfv["selection_score"].values
    df = df.sort_values("selection_score", ascending=False, na_position="last")

    selected = df.iloc[0].copy()
    return selected, df


def patient_composition_within_selected_rna(ad_rna, focus_fines):
    obs = ad_rna.obs[[SAMPLE_COL, GROUP_COL, FINE_COL]].copy()
    obs[SAMPLE_COL] = clean_series(obs[SAMPLE_COL])
    obs[GROUP_COL] = clean_series(obs[GROUP_COL])
    obs[FINE_COL] = clean_series(obs[FINE_COL])

    obs = obs[
        (obs[SAMPLE_COL] != "NA") &
        (obs[GROUP_COL].isin([POS_GROUP, NEG_GROUP])) &
        (obs[FINE_COL] != "NA")
    ].copy()

    comp_all = pd.crosstab(obs[SAMPLE_COL], obs[FINE_COL], normalize="index")
    for f in focus_fines:
        if str(f) not in comp_all.columns:
            comp_all[str(f)] = 0.0

    comp_focus = comp_all[[str(f) for f in focus_fines]].copy()

    group = obs.drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL)[GROUP_COL]
    long = comp_focus.reset_index().melt(
        id_vars=SAMPLE_COL,
        var_name="cluster",
        value_name="prop"
    )
    long = long.merge(group.rename("group"), left_on=SAMPLE_COL, right_index=True, how="left")

    return long, comp_focus


def patient_level_test_from_long(comp_long):
    rows = []

    for cl in comp_long["cluster"].astype(str).unique():
        sub = comp_long[comp_long["cluster"].astype(str) == str(cl)]
        pos = sub.loc[sub["group"] == POS_GROUP, "prop"].values
        neg = sub.loc[sub["group"] == NEG_GROUP, "prop"].values

        if len(pos) < 2 or len(neg) < 2:
            continue

        try:
            _, p = mannwhitneyu(pos, neg, alternative="two-sided")
        except Exception:
            p = np.nan

        rows.append({
            "cluster": str(cl),
            "mean_MPR": float(np.mean(pos)),
            "mean_nonMPR": float(np.mean(neg)),
            "diff_MPR_minus_nonMPR": float(np.mean(pos) - np.mean(neg)),
            "p_value": float(p),
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["p_adj"] = bh_fdr(out["p_value"].fillna(1.0).values)
    return out


def summarize_fused_clusters(ad_focus, clinical_df):
    obs = ad_focus.obs.copy()
    rows = []

    for cl, sub in obs.groupby("fused_existing", observed=True):
        sub = sub.copy()

        trb = clean_series(sub[TRB_COL]) if TRB_COL in sub.columns else pd.Series("NA", index=sub.index)
        trb_valid = trb[trb != "NA"]
        vc = trb_valid.value_counts()

        if len(vc) > 0:
            p = vc.values / vc.values.sum()
            shannon = float(-(p * np.log(p + 1e-12)).sum())
            clonality = float(1.0 - shannon / np.log(len(p))) if len(p) > 1 else 1.0
            top_clone_frac = float(vc.iloc[0] / len(sub))
            n_unique_trb = int(len(vc))
        else:
            clonality = np.nan
            top_clone_frac = np.nan
            n_unique_trb = 0

        rows.append({
            "cluster": str(cl),
            "n_cells": int(len(sub)),
            "n_patients": int(sub[SAMPLE_COL].nunique()) if SAMPLE_COL in sub.columns else np.nan,
            "expanded_ratio": float(get_num(sub, "is_expanded", 0).mean()),
            "mean_clone_size": float(get_num(sub, "clone_size", 1).mean()),
            "top_clone_frac": top_clone_frac,
            "n_unique_TRB": n_unique_trb,
            "clonality": clonality,
            "score_exhaustion": float(get_num(sub, "score_exhaustion").mean()),
            "score_cytotoxic": float(get_num(sub, "score_cytotoxic").mean()),
            "score_proliferation": float(get_num(sub, "score_proliferation").mean()),
            "score_ifn": float(get_num(sub, "score_ifn").mean()),
            "score_memory": float(get_num(sub, "score_memory").mean()),
        })

    out = pd.DataFrame(rows)

    if clinical_df is not None and len(clinical_df) > 0:
        out = out.merge(clinical_df, on="cluster", how="left")

    if "p_value" in out.columns:
        out = out.sort_values("p_value")
    else:
        out = out.sort_values("n_cells", ascending=False)

    return out


def safe_metric(labels, truth):
    labels = np.asarray(labels).astype(str)
    truth = np.asarray(truth).astype(str)

    mask = (labels != "NA") & (truth != "NA")
    labels = labels[mask]
    truth = truth[mask]

    if len(labels) < 20:
        return None
    if len(np.unique(labels)) < 2 or len(np.unique(truth)) < 2:
        return None

    return {
        "n_cells": int(len(labels)),
        "n_clusters": int(len(np.unique(labels))),
        "n_labels": int(len(np.unique(truth))),
        "ARI": float(adjusted_rand_score(truth, labels)),
        "AMI": float(adjusted_mutual_info_score(truth, labels)),
        "NMI": float(normalized_mutual_info_score(truth, labels)),
        "homogeneity": float(homogeneity_score(truth, labels)),
        "completeness": float(completeness_score(truth, labels)),
        "v_measure": float(v_measure_score(truth, labels)),
    }


def metric_table(obs, label_dict):
    rows = []

    for method, labels in label_dict.items():
        labels = np.asarray(labels).astype(str)

        for label_col, cell_set in [
            (TRB_COL, "all_cells"),
            (TRB_COL, "clone_size>=2"),
            (SAMPLE_COL, "all_cells"),
            (GROUP_COL, "all_cells"),
        ]:
            tmp = obs.copy()
            tmp["_label"] = labels

            if cell_set == "clone_size>=2":
                trb = clean_series(tmp[TRB_COL])
                clone_size = trb.map(trb.value_counts()).fillna(1).astype(int)
                tmp = tmp.loc[(trb != "NA") & (clone_size >= MIN_CLONE_SIZE_FOR_METRICS)].copy()

            m = safe_metric(tmp["_label"].values, clean_series(tmp[label_col]).values)
            if m is None:
                continue

            m.update({
                "method": method,
                "label_col": label_col,
                "cell_set": cell_set,
            })
            rows.append(m)

    return pd.DataFrame(rows)


def same_neighbor_score(X, obs, k=30):
    obs = obs.copy()
    trb = clean_series(obs[TRB_COL])
    valid = trb != "NA"

    obs_valid = obs.loc[valid].copy()
    X_valid = X[valid.values]

    if len(obs_valid) < 100:
        return None

    k_use = min(k + 1, len(obs_valid))
    nbrs = NearestNeighbors(n_neighbors=k_use, metric="euclidean")
    nbrs.fit(X_valid)
    _, nn = nbrs.kneighbors(X_valid)
    nn = nn[:, 1:]

    trb_arr = clean_series(obs_valid[TRB_COL]).values
    sample_arr = clean_series(obs_valid[SAMPLE_COL]).values

    same_trb = []
    same_sample = []

    for i in range(nn.shape[0]):
        same_trb.append(float(np.mean(trb_arr[nn[i]] == trb_arr[i])))
        same_sample.append(float(np.mean(sample_arr[nn[i]] == sample_arr[i])))

    return {
        "n_cells": int(len(obs_valid)),
        "k": int(k_use - 1),
        "same_TRB": float(np.mean(same_trb)),
        "same_sample": float(np.mean(same_sample)),
    }


def recluster_match_n(ad_sub, rep_key, target_n, key_added):
    ad_work = ad_sub.copy()

    sc.pp.neighbors(
        ad_work,
        use_rep=rep_key,
        n_neighbors=min(N_NEIGHBORS, max(5, ad_work.n_obs // 10)),
        key_added=f"{key_added}_nb",
        random_state=RANDOM_SEED,
    )

    candidate_res = np.round(np.linspace(0.02, 5.0, 160), 3).tolist()

    best = None
    exact = None

    for res in candidate_res:
        tmp_key = f"{key_added}_tmp"

        try:
            sc.tl.leiden(
                ad_work,
                resolution=res,
                key_added=tmp_key,
                neighbors_key=f"{key_added}_nb",
                random_state=RANDOM_SEED,
            )
        except Exception:
            sc.tl.louvain(
                ad_work,
                resolution=res,
                key_added=tmp_key,
                neighbors_key=f"{key_added}_nb",
                random_state=RANDOM_SEED,
            )

        labels = clean_series(ad_work.obs[tmp_key])
        ncl = labels.nunique()
        score = abs(ncl - target_n)

        item = {
            "resolution": float(res),
            "n_clusters": int(ncl),
            "score": int(score),
            "labels": labels.values.copy(),
        }

        if best is None or score < best["score"]:
            best = item

        if ncl == target_n and exact is None:
            exact = item

    final = exact if exact is not None else best

    if final["n_clusters"] != target_n:
        print(f"[warn] {key_added}: target_n={target_n}, got={final['n_clusters']}")

    return final


def permutation_ami(labels, truth, n_perm=200, seed=1337):
    rng = np.random.default_rng(seed)

    labels = np.asarray(labels).astype(str)
    truth = np.asarray(truth).astype(str)

    mask = (labels != "NA") & (truth != "NA")
    labels = labels[mask]
    truth = truth[mask]

    if len(labels) < 50:
        return None
    if len(np.unique(labels)) < 2 or len(np.unique(truth)) < 2:
        return None

    obs_ami = adjusted_mutual_info_score(truth, labels)

    perm = []
    for _ in range(n_perm):
        lab = labels.copy()
        rng.shuffle(lab)
        perm.append(adjusted_mutual_info_score(truth, lab))

    perm = np.asarray(perm)

    return {
        "observed_AMI": float(obs_ami),
        "perm_mean": float(perm.mean()),
        "perm_std": float(perm.std()),
        "z_score": float((obs_ami - perm.mean()) / (perm.std() + 1e-12)),
        "empirical_p": float((np.sum(perm >= obs_ami) + 1) / (n_perm + 1)),
    }


def compute_umap_from_rep(ad, rep_key, umap_key=None):
    if umap_key is not None and umap_key in ad.obsm:
        return np.asarray(ad.obsm[umap_key])

    adw = ad.copy()
    sc.pp.neighbors(
        adw,
        use_rep=rep_key,
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_SEED,
    )
    sc.tl.umap(adw, random_state=RANDOM_SEED)
    return np.asarray(adw.obsm["X_umap"])


def compute_rna_umap(ad):
    return compute_umap_from_rep(ad, RNA_REP, umap_key="X_umap_rna")


def compute_fused_umap(ad):
    if "X_umap_fused" in ad.obsm:
        return np.asarray(ad.obsm["X_umap_fused"])
    if "X_umap_final" in ad.obsm:
        return np.asarray(ad.obsm["X_umap_final"])
    return compute_umap_from_rep(ad, FUSED_REP, umap_key=None)


def build_label_palette(labels, highlight_labels=None):
    labels = sorted([str(x) for x in labels], key=sort_key)
    highlight_labels = set([str(x) for x in highlight_labels]) if highlight_labels is not None else set()

    base = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        base.extend([cmap(i) for i in range(cmap.N)])

    pal = {}
    for i, lab in enumerate(labels):
        pal[lab] = base[i % len(base)]

    return pal


def plot_rnaonly_leiden_umap(ad, coords, selected_rna, out_dir):
    obs = ad.obs.copy()
    obs[RNA_CLUSTER_COL] = clean_series(obs[RNA_CLUSTER_COL])

    labels = sorted(obs[RNA_CLUSTER_COL].astype(str).unique(), key=sort_key)
    pal = build_label_palette(labels)
    pal[str(selected_rna)] = FUSED_COLOR

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    for lab in labels:
        mask = obs[RNA_CLUSTER_COL].astype(str).values == str(lab)
        size = 2.0 if lab != str(selected_rna) else 2.8
        alpha = 0.62 if lab != str(selected_rna) else 0.95

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            c=[pal[lab]],
            linewidths=0,
            alpha=alpha,
            rasterized=True,
        )

    for lab in labels:
        mask = obs[RNA_CLUSTER_COL].astype(str).values == str(lab)
        if mask.sum() == 0:
            continue

        cx = np.median(coords[mask, 0])
        cy = np.median(coords[mask, 1])

        ax.text(
            cx,
            cy,
            str(lab),
            ha="center",
            va="center",
            fontsize=7 if lab != str(selected_rna) else 9,
            fontweight="bold" if lab == str(selected_rna) else "normal",
            color="white" if lab == str(selected_rna) else INK,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=FUSED_COLOR if lab == str(selected_rna) else "white",
                edgecolor="none",
                alpha=0.85 if lab == str(selected_rna) else 0.65,
            ),
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("RNA-only Leiden clusters")
    ax.text(
        0.02,
        0.02,
        f"Selected RNA cluster {selected_rna}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=FUSED_COLOR,
        fontweight="bold",
    )

    save_fig(fig, out_dir, "Fig1a_RNAonly_leiden_umap")


def plot_rna_cluster_selection_rank(rna_rank, selected_rna, out_dir):
    df = rna_rank.copy()
    df["RNA_cluster"] = df["RNA_cluster"].astype(str)
    df = df.sort_values("selection_score", ascending=False, na_position="last").head(8).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    x = np.arange(len(df))
    colors = [
        FUSED_COLOR if c == str(selected_rna) else RNA_COLOR
        for c in df["RNA_cluster"].values
    ]

    ax.bar(
        x,
        df["selection_score"].values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.72,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["RNA_cluster"].values, rotation=90)
    ax.set_ylabel("Selection score")
    ax.set_xlabel("RNA-only cluster")
    ax.set_title("Top-ranked\nRNA clusters")

    selected_idx = df.index[df["RNA_cluster"] == str(selected_rna)][0]
    selected_y = df.loc[selected_idx, "selection_score"]

    ax.text(
        selected_idx,
        selected_y + 0.08,
        f"RNA {selected_rna}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=FUSED_COLOR,
        fontweight="bold",
    )

    ymax = max(df["selection_score"].max() * 1.18, 0.5)
    ax.set_ylim(0, ymax)

    despine(ax)
    save_fig(fig, out_dir, "Fig1b_RNA_cluster_selection_rank")


def plot_rna_score_umap(ad, coords, selected_rna, out_dir):
    obs = ad.obs.copy()
    obs[RNA_CLUSTER_COL] = clean_series(obs[RNA_CLUSTER_COL])

    selected_mask = obs[RNA_CLUSTER_COL].astype(str).values == str(selected_rna)

    score_col = "score_exhaustion"
    score = pd.to_numeric(obs[score_col], errors="coerce").fillna(0).values
    vmin = np.nanpercentile(score, 2)
    vmax = np.nanpercentile(score, 99)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    sca = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=2.0,
        c=score,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.92,
        rasterized=True,
    )

    ax.scatter(
        coords[selected_mask, 0],
        coords[selected_mask, 1],
        s=3.0,
        facecolors="none",
        edgecolors=FUSED_COLOR,
        linewidths=0.18,
        alpha=0.7,
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Tumor-reactive / exhaustion score on RNA-only UMAP")
    cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("program score")

    ax.text(
        0.02,
        0.02,
        f"RNA cluster {selected_rna} outlined",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=FUSED_COLOR,
        fontweight="bold",
    )

    save_fig(fig, out_dir, "Fig1c_RNAonly_score_umap")


def plot_fused_leiden_umap(ad, coords, selected_rna, focus_fines, out_dir):

    obs = ad.obs.copy()
    obs[RNA_CLUSTER_COL] = clean_series(obs[RNA_CLUSTER_COL])
    obs[FINE_COL] = clean_series(obs[FINE_COL])

    focus_fines = [str(x) for x in focus_fines]

    labels = sorted(obs[FINE_COL].astype(str).unique(), key=sort_key)
    pal = build_label_palette(labels)

    for i, f in enumerate(focus_fines):
        if f in pal:
            pal[f] = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]

    selected_mask = obs[RNA_CLUSTER_COL].astype(str).values == str(selected_rna)

    fig, ax = plt.subplots(figsize=(5.0, 4.3))

    for lab in labels:
        mask = obs[FINE_COL].astype(str).values == str(lab)

        if lab in focus_fines:
            size = 2.8
            alpha = 0.90
            zorder = 4
        else:
            size = 1.8
            alpha = 0.62
            zorder = 2

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            c=[pal[lab]],
            linewidths=0,
            alpha=alpha,
            rasterized=True,
            zorder=zorder,
        )

    for lab in labels:
        mask = obs[FINE_COL].astype(str).values == str(lab)
        if mask.sum() == 0:
            continue

        cx = np.median(coords[mask, 0])
        cy = np.median(coords[mask, 1])

        is_focus = lab in focus_fines

        ax.text(
            cx,
            cy,
            str(lab),
            ha="center",
            va="center",
            fontsize=8 if is_focus else 6.5,
            fontweight="bold" if is_focus else "normal",
            color="white" if is_focus else INK,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=pal[lab] if is_focus else "white",
                edgecolor="none",
                alpha=0.88 if is_focus else 0.60,
            ),
            zorder=10,
        )

    ax.scatter(
        coords[selected_mask, 0],
        coords[selected_mask, 1],
        s=3.2,
        facecolors="none",
        edgecolors=FUSED_COLOR,
        linewidths=0.12,
        alpha=0.28,
        rasterized=True,
        zorder=5,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Fused Leiden clusters")
    ax.text(
        0.02,
        0.02,
        f"RNA cluster {selected_rna} outlined; F12/F19/F6/F9 highlighted",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=FUSED_COLOR,
        fontweight="bold",
    )

    save_fig(fig, out_dir, "Fig1d_Fused_leiden_umap")

def _ribbon_path(x0, x1, y0_top, y0_bot, y1_top, y1_bot):
    dx = x1 - x0
    c0 = x0 + dx * 0.46
    c1 = x1 - dx * 0.46

    verts = [
        (x0, y0_top),
        (c0, y0_top),
        (c1, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (c1, y1_bot),
        (c0, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    return Path(verts, codes)


def _stack_segments(items, y_top=0.90, y_bottom=0.10, gap=0.025):
    total = sum(v for _, v in items)
    n = len(items)
    usable = (y_top - y_bottom) - gap * max(0, n - 1)

    out = {}
    cursor = y_top

    for name, value in items:
        h = usable * value / total if total > 0 else 0
        top = cursor
        bottom = cursor - h
        out[name] = (top, bottom)
        cursor = bottom - gap

    return out


def plot_sankey_selected_rna_three_layer(ad_rna, selected_rna, focus_fines, summary, out_dir):
    obs = ad_rna.obs.copy()
    obs[FINE_COL] = clean_series(obs[FINE_COL])

    fine_counts = obs[FINE_COL].value_counts()
    total = int(len(obs))
    focus_fines = [str(f) for f in focus_fines]

    diff_map = {}
    if summary is not None and len(summary) > 0 and "diff_MPR_minus_nonMPR" in summary.columns:
        diff_map = summary.set_index("cluster")["diff_MPR_minus_nonMPR"].to_dict()

    fine_items = []
    for f in focus_fines:
        cnt = int(fine_counts.get(f, 0))
        if cnt <= 0:
            continue

        diff = diff_map.get(f, np.nan)
        if pd.isna(diff):
            group = "Other"
        elif diff < 0:
            group = "non-MPR-associated"
        else:
            group = "MPR-associated"

        fine_items.append({
            "fine": f,
            "count": cnt,
            "group": group,
            "diff": diff,
        })

    other_count = int(total - sum(x["count"] for x in fine_items))
    if other_count > 0:
        fine_items.append({
            "fine": "Other",
            "count": other_count,
            "group": "Other",
            "diff": np.nan,
        })

    group_order = ["non-MPR-associated", "MPR-associated", "Other"]
    group_items = []
    for g in group_order:
        cnt = sum(x["count"] for x in fine_items if x["group"] == g)
        if cnt > 0:
            group_items.append((g, cnt))

    right_order = []
    for g in group_order:
        for item in fine_items:
            if item["group"] == g and item["fine"] not in right_order:
                right_order.append(item["fine"])

    right_items = []
    for f in right_order:
        cnt = sum(x["count"] for x in fine_items if x["fine"] == f)
        if cnt > 0:
            right_items.append((f, cnt))

    group_colors = {
        "non-MPR-associated": NONMPR_COLOR,
        "MPR-associated": MPR_COLOR,
        "Other": OTHER_COLOR,
    }

    fine_colors = {}
    for i, f in enumerate(focus_fines):
        fine_colors[f] = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
    fine_colors["Other"] = OTHER_COLOR

    x_left = 0.10
    x_mid = 0.50
    x_right = 0.82
    node_w = 0.035

    y_top = 0.90
    y_bottom = 0.10

    group_seg = _stack_segments(group_items, y_top=y_top, y_bottom=y_bottom, gap=0.04)
    right_seg = _stack_segments(right_items, y_top=y_top, y_bottom=y_bottom, gap=0.026)
    left_seg = group_seg.copy()

    mid_subseg = {}
    for g, _ in group_items:
        g_fines = [(x["fine"], x["count"]) for x in fine_items if x["group"] == g]
        g_top, g_bot = group_seg[g]
        mid_subseg[g] = _stack_segments(g_fines, y_top=g_top, y_bottom=g_bot, gap=0.004)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    ax.add_patch(Rectangle(
        (x_left - node_w / 2, y_bottom),
        node_w,
        y_top - y_bottom,
        facecolor=RNA_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
    ))

    for g, cnt in group_items:
        lt, lb = left_seg[g]
        mt, mb = group_seg[g]
        color = group_colors[g]

        patch = PathPatch(
            _ribbon_path(
                x_left + node_w / 2,
                x_mid - node_w / 2,
                lt,
                lb,
                mt,
                mb,
            ),
            facecolor=color,
            edgecolor="none",
            alpha=0.18 if g == "Other" else 0.24,
            zorder=1,
        )
        ax.add_patch(patch)

    for g, cnt in group_items:
        mt, mb = group_seg[g]
        ax.add_patch(Rectangle(
            (x_mid - node_w / 2, mb),
            node_w,
            mt - mb,
            facecolor=group_colors[g],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
            alpha=0.88 if g != "Other" else 0.65,
        ))

        label = g.replace("-associated", "")
        ax.text(
            x_mid - 0.045,
            (mt + mb) / 2,
            label,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold" if g != "Other" else "normal",
        )

    for item in fine_items:
        f = item["fine"]
        g = item["group"]

        mt, mb = mid_subseg[g][f]
        rt, rb = right_seg[f]

        patch = PathPatch(
            _ribbon_path(
                x_mid + node_w / 2,
                x_right - node_w / 2,
                mt,
                mb,
                rt,
                rb,
            ),
            facecolor=fine_colors[f],
            edgecolor="none",
            alpha=0.43 if f != "Other" else 0.22,
            zorder=2,
        )
        ax.add_patch(patch)

    for f, cnt in right_items:
        rt, rb = right_seg[f]
        color = fine_colors[f]

        ax.add_patch(Rectangle(
            (x_right - node_w / 2, rb),
            node_w,
            rt - rb,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=9,
            alpha=0.95 if f != "Other" else 0.65,
        ))

        pct = cnt / total * 100
        ax.text(
            x_right + 0.045,
            (rt + rb) / 2,
            f"{clabel(f)}\n{cnt:,} ({pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=8,
            linespacing=1.05,
        )

    ax.text(
        x_left,
        y_top + 0.045,
        f"RNA {selected_rna}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x_left,
        y_bottom - 0.035,
        f"{total:,} cells",
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
    )

    ax.text(x_left, 0.035, "RNA-only state", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(x_mid, 0.035, "Response direction", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(x_right, 0.035, "Fused subclusters", ha="center", va="center", fontsize=7, color="#555555")

    ax.set_xlim(0.02, 1.05)
    ax.set_ylim(0.00, 1.00)
    ax.axis("off")
    ax.set_title("Fused partitioning of the selected RNA-defined state", fontsize=9, pad=8)

    save_fig(fig, out_dir, "Fig1e_RNA_to_fused_alluvial")


def _draw_box_strip(ax, vals, pos, color, rng, width=0.28, jitter=0.055):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    ax.boxplot(
        vals,
        positions=[pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        showcaps=False,
        boxprops=dict(facecolor=color, edgecolor=INK, linewidth=0.6, alpha=0.55),
        medianprops=dict(color=INK, linewidth=0.9),
        whiskerprops=dict(color=INK, linewidth=0.6),
    )

    jx = pos + rng.uniform(-jitter, jitter, size=len(vals))
    ax.scatter(
        jx,
        vals,
        s=9,
        c=color,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.9,
        zorder=3,
    )


def plot_response_association(comp_long, summary, focus_fines, out_dir):
    rng = np.random.default_rng(RANDOM_SEED)
    order = [str(x) for x in focus_fines]

    qmap = summary.set_index("cluster")["p_adj"].to_dict() if "p_adj" in summary.columns else {}
    diff_map = summary.set_index("cluster")["diff_MPR_minus_nonMPR"].to_dict() if "diff_MPR_minus_nonMPR" in summary.columns else {}

    fig, ax = plt.subplots(figsize=(5.8, 3.2))

    width = 0.32
    ymax = 0.0

    for i, cl in enumerate(order):
        sub = comp_long[comp_long["cluster"].astype(str) == str(cl)]
        neg = sub.loc[sub["group"] == NEG_GROUP, "prop"].values
        pos = sub.loc[sub["group"] == POS_GROUP, "prop"].values

        _draw_box_strip(ax, neg, i - width / 1.8, RESP_COLORS[NEG_GROUP], rng, width=width)
        _draw_box_strip(ax, pos, i + width / 1.8, RESP_COLORS[POS_GROUP], rng, width=width)

        cur_max = 0
        if len(neg):
            cur_max = max(cur_max, np.nanmax(neg))
        if len(pos):
            cur_max = max(cur_max, np.nanmax(pos))
        ymax = max(ymax, cur_max)

        q = qmap.get(cl, np.nan)
        ax.text(i, cur_max + 0.025, qstar(q), ha="center", va="bottom", fontsize=8)

        if cl in diff_map:
            ax.text(
                i,
                -0.055,
                f"Δ={diff_map[cl]:+.2f}",
                ha="center",
                va="top",
                fontsize=6.5,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([clabel(c) for c in order])
    ax.set_ylabel("Per-patient fraction\nwithin selected RNA cluster")
    ax.set_title("Clinical response association of fused subclusters")
    ax.set_ylim(0, ymax * 1.22 + 0.04)

    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=6, markerfacecolor=RESP_COLORS[NEG_GROUP], markeredgecolor="white", label=NEG_GROUP),
        Line2D([], [], marker="s", linestyle="", markersize=6, markerfacecolor=RESP_COLORS[POS_GROUP], markeredgecolor="white", label=POS_GROUP),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    despine(ax)

    save_fig(fig, out_dir, "Fig1f_clinical_response")


def plot_function_gene(ad_focus, summary, focus_fines, out_dir):

    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, to_hex
    from matplotlib.cm import ScalarMappable

    order = [str(x) for x in focus_fines]
    order = [x for x in order if x in set(summary["cluster"].astype(str))]

    fig = plt.figure(figsize=(9.4, 3.45))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.05, 2.85],
        wspace=0.34
    )

    ax1 = fig.add_subplot(gs[0, 0])

    d = summary.set_index("cluster").loc[order]

    program_display_local = [
        ("score_exhaustion", "Exhaustion"),
        ("score_cytotoxic", "Cytotoxic"),
        ("score_proliferation", "Proliferation"),
        ("score_memory", "Memory"),
    ]


    if "score_trm" in d.columns:
        program_display_local.append(("score_trm", "TRM"))

    program_cols = [c for c, _ in program_display_local if c in d.columns]
    program_names = [n for c, n in program_display_local if c in d.columns]

    raw = d[program_cols].astype(float)


    z = raw.apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-9),
        axis=0
    )

    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-1.25, vmax=1.25)

    n_rows, n_cols = z.shape

    for i in range(n_rows):
        for j in range(n_cols):
            val_z = float(z.values[i, j])
            val_raw = float(raw.values[i, j])

            color_hex = to_hex(cmap(norm(val_z)), keep_alpha=False)

            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=color_hex,
                edgecolor="white",
                linewidth=0.6,
                alpha=1.0,
                zorder=1,
            )
            ax1.add_patch(rect)

            ax1.text(
                j,
                i,
                f"{val_raw:.2f}",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color="white" if abs(val_z) > 0.95 else INK,
                zorder=3,
            )

    ax1.set_xlim(-0.5, n_cols - 0.5)
    ax1.set_ylim(n_rows - 0.5, -0.5)

    ax1.set_xticks(np.arange(n_cols))
    ax1.set_xticklabels(program_names, rotation=45, ha="right", fontsize=8.5)
    ax1.set_yticks(np.arange(n_rows))
    ax1.set_yticklabels([clabel(c) for c in order], fontsize=8.5)

    ax1.set_title("Functional programs", pad=5, fontsize=10)
    ax1.tick_params(length=0, pad=3)
    despine(ax1, keep=())

    sm1 = ScalarMappable(norm=norm, cmap=cmap)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, fraction=0.055, pad=0.025)
    cb1.set_label("z-score", labelpad=4, fontsize=8.5)
    cb1.ax.tick_params(labelsize=8, length=2)

    ax2 = fig.add_subplot(gs[0, 1])

    if ad_focus.raw is not None:
        var_names = set(ad_focus.raw.var_names.astype(str))
    else:
        var_names = set(ad_focus.var_names.astype(str))

    gene_order = [
        "CXCL13", "PDCD1", "TOX", "LAG3", "HAVCR2", "TIGIT", "ENTPD1",
        "GZMB", "PRF1", "NKG7", "GNLY", "IFNG",
        "MKI67", "TCF7", "IL7R"
    ]

    genes_present = [g for g in gene_order if g in var_names]

    expr = get_expression_df(ad_focus, genes_present)
    obs = ad_focus.obs.copy()
    obs["fused_existing"] = clean_series(obs["fused_existing"])

    rows = []
    for cl in order:
        idx = obs.index[obs["fused_existing"].astype(str) == str(cl)]
        sub_expr = expr.loc[idx]

        for g in genes_present:
            vals = pd.to_numeric(sub_expr[g], errors="coerce").fillna(0).values
            rows.append({
                "cluster": str(cl),
                "gene": g,
                "mean_expr": float(np.mean(vals)),
                "pct_expr": float(np.mean(vals > 0)),
            })

    dot = pd.DataFrame(rows)

    if len(dot) == 0:
        ax2.text(0.5, 0.5, "No marker genes found", ha="center", va="center")
        ax2.axis("off")
    else:
        dot["mean_z"] = 0.0

        for g in genes_present:
            m = dot["gene"] == g
            vals = dot.loc[m, "mean_expr"]
            dot.loc[m, "mean_z"] = (vals - vals.mean()) / (vals.std() + 1e-9)

        xmap = {g: i for i, g in enumerate(genes_present)}
        ymap = {cl: i for i, cl in enumerate(order)}

        dot["x"] = dot["gene"].map(xmap)
        dot["y"] = dot["cluster"].map(ymap)

        sizes = 18 + 310 * dot["pct_expr"].values

        sca = ax2.scatter(
            dot["x"].values,
            dot["y"].values,
            s=sizes,
            c=dot["mean_z"].values,
            cmap="Reds",
            vmin=-1.5,
            vmax=1.8,
            edgecolor="#555555",
            linewidth=0.45,
            alpha=1.0,
        )

        ax2.set_xticks(np.arange(len(genes_present)))
        ax2.set_xticklabels(genes_present, rotation=45, ha="right", fontsize=8.5)
        ax2.set_yticks(np.arange(len(order)))
        ax2.set_yticklabels([clabel(c) for c in order], fontsize=8.5)
        ax2.set_title("Marker gene expression", pad=5, fontsize=10)

        ax2.set_xlim(-0.5, len(genes_present) - 0.5)
        ax2.set_ylim(len(order) - 0.5, -0.5)

        ax2.grid(True, axis="both", color="#E9E9E9", linewidth=0.7)
        ax2.set_axisbelow(True)
        ax2.tick_params(length=0, pad=3)
        despine(ax2, keep=("left", "bottom"))

        cb2 = fig.colorbar(sca, ax=ax2, fraction=0.024, pad=0.018)
        cb2.set_label("Mean expression\n(z-score)", labelpad=4, fontsize=8.5)
        cb2.ax.tick_params(labelsize=8, length=2)

        size_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
        handles = []
        labels = []

        for pct in size_levels:
            handles.append(
                ax2.scatter(
                    [],
                    [],
                    s=18 + 310 * pct,
                    facecolor="#F06A4A",
                    edgecolor="#555555",
                    linewidth=0.45,
                    alpha=1.0,
                )
            )
            labels.append(f"{int(pct * 100)}%")

        ax2.legend(
            handles,
            labels,
            title="% cells\nexpressing",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.13, 1.02),
            borderaxespad=0,
            handletextpad=0.9,
            labelspacing=0.85,
            fontsize=7.6,
            title_fontsize=8.0,
        )

    save_fig(fig, out_dir, "Fig1g_function_and_genes")

def plot_trb_clone_structure_global(overall_mt, s_rna, s_fused, out_dir):
    fig = plt.figure(figsize=(5.4, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])

    methods = ["RNAonly_matched", "fused_existing"]
    method_labels = ["RNA-only\nmatched", "Fused\nsubclusters"]
    colors = [RNA_COLOR, FUSED_COLOR]

    ami_vals = []

    for method in methods:
        sub = overall_mt[
            (overall_mt["method"] == method) &
            (overall_mt["label_col"] == TRB_COL) &
            (overall_mt["cell_set"] == "clone_size>=2")
        ]

        if len(sub) == 0:
            ami_vals.append(np.nan)
        else:
            ami_vals.append(float(sub["AMI"].iloc[0]))

    x = np.arange(len(methods))

    ax1.bar(
        x,
        ami_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.62,
    )

    for xi, v in zip(x, ami_vals):
        if np.isfinite(v):
            ax1.text(
                xi,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels)
    ax1.set_ylabel("AMI with TRB clone")
    ax1.set_title("Cluster-level clonotype agreement")
    ax1.set_ylim(0, max([v for v in ami_vals if np.isfinite(v)] + [0.1]) * 1.28)
    despine(ax1)

    ax2 = fig.add_subplot(gs[0, 1])

    nn_vals = [
        s_rna["same_TRB"] if s_rna is not None else np.nan,
        s_fused["same_TRB"] if s_fused is not None else np.nan,
    ]

    nn_labels = ["RNA-only\nembedding", "Fused\nembedding"]
    nn_colors = [RNA_COLOR, FUSED_COLOR]

    x2 = np.arange(2)

    ax2.bar(
        x2,
        nn_vals,
        color=nn_colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.62,
    )

    for xi, v in zip(x2, nn_vals):
        if np.isfinite(v):
            ax2.text(
                xi,
                v + 0.018,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(nn_labels)
    ax2.set_ylabel(f"same-TRB fraction\namong {N_NEIGHBORS}-NN")
    ax2.set_title("Embedding-level clonotype neighborhood")
    ax2.set_ylim(0, max([v for v in nn_vals if np.isfinite(v)] + [0.1]) * 1.28)
    despine(ax2)

    fig.suptitle(
        "Fused representation enhances TRB clonotype-associated structure",
        fontsize=9,
        y=1.05,
    )

    save_fig(fig, out_dir, "Fig1h_TRB_clonotype_structure")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=H5AD_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--top_fused_n", type=int, default=TOP_FUSED_N)
    ap.add_argument("--no_png", action="store_true")
    ap.add_argument("--no_pdf", action="store_true")
    return ap.parse_args()


def main():
    global TOP_FUSED_N, SAVE_PNG, SAVE_PDF

    args = parse_args()
    TOP_FUSED_N = args.top_fused_n
    SAVE_PNG = not args.no_png
    SAVE_PDF = not args.no_pdf

    set_style()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 100)
    print("RNA-only marker-selected state -> fused subcluster story with fused Leiden")
    print("=" * 100)

    if not os.path.exists(args.h5ad):
        raise FileNotFoundError(f"Cannot find: {args.h5ad}")

    ad = sc.read_h5ad(args.h5ad)
    ad = ensure_response_binary(ad)
    check_response_label_logic(ad, tag="after loading")
    for c in [RNA_CLUSTER_COL, FINE_COL, SAMPLE_COL, GROUP_COL, TRB_COL, "sub_cell_type"]:
        if c in ad.obs.columns:
            ad.obs[c] = clean_series(ad.obs[c])

    for rep in [RNA_REP, FUSED_REP]:
        if rep not in ad.obsm:
            raise ValueError(f"Missing ad.obsm['{rep}']")

    ad, used = add_scores(ad)

    print("\n[Basic]")
    print(
        f"cells={ad.n_obs}, "
        f"samples={ad.obs[SAMPLE_COL].nunique()}, "
        f"RNA_clusters={ad.obs[RNA_CLUSTER_COL].nunique()}, "
        f"fine_clusters={ad.obs[FINE_COL].nunique()}"
    )
    print(f"score gene usage={used}")


    selected, rna_rank = score_rna_clusters(ad)
    selected_rna = str(selected["RNA_cluster"])

    print("\n" + "=" * 100)
    print("1) RNA-only cluster scoring")
    print("=" * 100)

    show_rna_cols = [
        "RNA_cluster",
        "n_cells",
        "n_patients",
        "CD8_prop",
        "score_exhaustion",
        "expr_CXCL13",
        "expr_PDCD1",
        "expr_TOX",
        "expanded_ratio",
        "selection_score",
    ]

    print("\n[Top RNA-only clusters]")
    print(rna_rank[show_rna_cols].head(10).to_string(index=False))

    print("\n[Selected RNA-only cluster]")
    print(selected[show_rna_cols].to_string())

    print("\n[UMAP] computing RNA-only UMAP ...")
    rna_umap = compute_rna_umap(ad)

    plot_rnaonly_leiden_umap(ad, rna_umap, selected_rna, args.out_dir)
    plot_rna_cluster_selection_rank(rna_rank, selected_rna, args.out_dir)
    plot_rna_score_umap(ad, rna_umap, selected_rna, args.out_dir)


    ad_rna = ad[ad.obs[RNA_CLUSTER_COL].astype(str) == selected_rna].copy()
    ad_rna.obs[FINE_COL] = clean_series(ad_rna.obs[FINE_COL])

    fine_counts = ad_rna.obs[FINE_COL].value_counts()
    focus_fines = fine_counts.head(TOP_FUSED_N).index.astype(str).tolist()

    ad_focus = ad_rna[ad_rna.obs[FINE_COL].astype(str).isin(focus_fines)].copy()
    ad_focus.obs["fused_existing"] = clean_series(ad_focus.obs[FINE_COL]).values
    check_selected_rna_fused_response(ad, selected_rna, focus_fines)
    print("\n" + "=" * 100)
    print("2) Selected RNA cluster -> fused fine clusters")
    print("=" * 100)
    print(f"selected RNA_cluster = {selected_rna}")
    print(f"RNA_cluster cells    = {ad_rna.n_obs}")
    print(f"focus fine_clusters  = {focus_fines}")
    print(f"focus cells          = {ad_focus.n_obs}")
    print(f"patients             = {ad_focus.obs[SAMPLE_COL].nunique()}")

    print("\n[Fine clusters inside selected RNA cluster]")
    fine_table = fine_counts.reset_index()
    fine_table.columns = ["fine_cluster", "n_cells"]
    fine_table["fraction_in_selected_RNA"] = fine_table["n_cells"] / ad_rna.n_obs
    print(fine_table.head(12).to_string(index=False))

    print("\n[UMAP] computing fused UMAP ...")
    fused_umap = compute_fused_umap(ad)
    plot_fused_leiden_umap(ad, fused_umap, selected_rna, focus_fines, args.out_dir)

    comp_long, comp_focus = patient_composition_within_selected_rna(ad_rna, focus_fines)
    clinical_df = patient_level_test_from_long(comp_long)
    summary = summarize_fused_clusters(ad_focus, clinical_df)

    print("\n" + "=" * 100)
    print("3) Clinical / clone / function summary of selected fused subclusters")
    print("=" * 100)
    print("[Clinical denominator] all cells in selected RNA_cluster for each patient")

    show_cols = [
        "cluster",
        "n_cells",
        "n_patients",
        "mean_MPR",
        "mean_nonMPR",
        "diff_MPR_minus_nonMPR",
        "p_value",
        "p_adj",
        "expanded_ratio",
        "mean_clone_size",
        "top_clone_frac",
        "n_unique_TRB",
        "clonality",
        "score_exhaustion",
        "score_cytotoxic",
        "score_proliferation",
        "score_memory",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]
    print(summary[show_cols].to_string(index=False))

    plot_sankey_selected_rna_three_layer(ad_rna, selected_rna, focus_fines, summary, args.out_dir)
    plot_response_association(comp_long, summary, focus_fines, args.out_dir)

    print("\n" + "=" * 100)
    print("4) Functional program / marker genes")
    print("=" * 100)

    if ad.raw is not None:
        var_names_all = set(ad.raw.var_names.astype(str))
    else:
        var_names_all = set(ad.var_names.astype(str))

    present_genes = [g for g in FOCUS_GENES if g in var_names_all]
    print("[Programs shown]")
    print(", ".join([x[1] for x in PROGRAM_DISPLAY]))
    print("[Marker genes shown if present]")
    print(", ".join(present_genes))

    plot_function_gene(ad_focus, summary, focus_fines, args.out_dir)


    print("\n" + "=" * 100)
    print("5) RNA-only vs fused clonotype structure")
    print("=" * 100)

    target_n = len(focus_fines)

    rna_best = recluster_match_n(
        ad_focus,
        rep_key=RNA_REP,
        target_n=target_n,
        key_added="RNAonly_matched",
    )

    fused_best = recluster_match_n(
        ad_focus,
        rep_key=FUSED_REP,
        target_n=target_n,
        key_added="fused_matched",
    )

    ad_focus.obs["RNAonly_matched"] = rna_best["labels"]
    ad_focus.obs["fused_matched"] = fused_best["labels"]

    print(f"RNAonly_matched: resolution={rna_best['resolution']}, n_clusters={rna_best['n_clusters']}")
    print(f"fused_matched  : resolution={fused_best['resolution']}, n_clusters={fused_best['n_clusters']}")

    obs_focus = ad_focus.obs.copy()

    labels = {
        "fused_existing": obs_focus["fused_existing"].values,
        "RNAonly_matched": obs_focus["RNAonly_matched"].values,
        "fused_matched": obs_focus["fused_matched"].values,
    }

    mt = metric_table(obs_focus, labels)

    mt_show = mt[
        ((mt["label_col"] == TRB_COL) & (mt["cell_set"] == "clone_size>=2")) |
        ((mt["label_col"] == SAMPLE_COL) & (mt["cell_set"] == "all_cells")) |
        ((mt["label_col"] == GROUP_COL) & (mt["cell_set"] == "all_cells"))
    ].copy()

    mt_cols = [
        "method",
        "label_col",
        "cell_set",
        "n_cells",
        "n_clusters",
        "n_labels",
        "ARI",
        "AMI",
        "NMI",
        "homogeneity",
        "completeness",
        "v_measure",
    ]

    print("\n[Cluster-label agreement]")
    print(mt_show[mt_cols].sort_values(["label_col", "AMI"], ascending=[True, False]).to_string(index=False))

    X_rna_focus = np.asarray(ad_focus.obsm[RNA_REP])
    X_fused_focus = np.asarray(ad_focus.obsm[FUSED_REP])

    s_rna = same_neighbor_score(X_rna_focus, obs_focus, k=N_NEIGHBORS)
    s_fused = same_neighbor_score(X_fused_focus, obs_focus, k=N_NEIGHBORS)

    print("\n[Embedding-level same-neighbor, overall focus cells]")
    if s_rna is not None:
        print(
            f"{RNA_REP}: n={s_rna['n_cells']}, k={s_rna['k']}, "
            f"same_TRB={s_rna['same_TRB']:.4f}, same_sample={s_rna['same_sample']:.4f}"
        )
    if s_fused is not None:
        print(
            f"{FUSED_REP}: n={s_fused['n_cells']}, k={s_fused['k']}, "
            f"same_TRB={s_fused['same_TRB']:.4f}, same_sample={s_fused['same_sample']:.4f}"
        )

    trb = clean_series(obs_focus[TRB_COL])
    clone_size = trb.map(trb.value_counts()).fillna(1).astype(int)
    obs_rec = obs_focus.loc[(trb != "NA") & (clone_size >= MIN_CLONE_SIZE_FOR_METRICS)].copy()

    perm_rows = []
    for method in ["fused_existing", "RNAonly_matched", "fused_matched"]:
        out = permutation_ami(
            obs_rec[method].values,
            clean_series(obs_rec[TRB_COL]).values,
            n_perm=N_PERM,
            seed=RANDOM_SEED,
        )
        if out is not None:
            out["method"] = method
            perm_rows.append(out)

    perm_df = pd.DataFrame(perm_rows)

    print("\n[Permutation: AMI(cluster, TRB) above random labels]")
    if len(perm_df) > 0:
        print(
            perm_df[
                ["method", "observed_AMI", "perm_mean", "perm_std", "z_score", "empirical_p"]
            ].sort_values("observed_AMI", ascending=False).to_string(index=False)
        )

    plot_trb_clone_structure_global(
        overall_mt=mt,
        s_rna=s_rna,
        s_fused=s_fused,
        out_dir=args.out_dir,
    )

    print("\n" + "=" * 100)
    print("Output figures")
    print("=" * 100)
    print(f"[out_dir] {args.out_dir}")
    print("Fig1a_RNAonly_leiden_umap.pdf/png")
    print("Fig1b_RNA_cluster_selection_rank.pdf/png")
    print("Fig1c_RNAonly_score_umap.pdf/png")
    print("Fig1d_Fused_leiden_umap.pdf/png")
    print("Fig1e_RNA_to_fused_alluvial.pdf/png")
    print("Fig1f_clinical_response.pdf/png")
    print("Fig1g_function_and_genes.pdf/png")
    print("Fig1h_TRB_clonotype_structure.pdf/png")

    print("\n[DONE]")


if __name__ == "__main__":
    main()