"""Load and annotate NSCLC/LUAD RNA, TCR and response records."""

import re

import numpy as np
import pandas as pd
import scanpy as sc

from immunlens.preprocessing.sequences import clean_aa


def load_data(RNA_PATH, META_PATH, TCR_PATH, SAMPLE_META_PATH, CANCER_TYPE):
    """
    NSCLC neoadjuvant ICB 数据加载.
      RNA_PATH         : GSE243013_NSCLC_immune_scRNA.h5ad
      META_PATH        : GSE243013_NSCLC_immune_scRNA_metadata.csv
                         (cellID / sub_cell_type / cancer_type / sampleID(若有) ...)
      TCR_PATH         : GSE243013_T_with_TCR_annotation.csv
                         (cellID / sampleID / TRA_cdr3 / TRB_cdr3 / expansion / sub_cell_type)
      SAMPLE_META_PATH : meta2.csv
                         (Tumor_Sample_Barcode -> Pathological Response = pCR / MPR / non-MPR)

    Cancer type 开关: CANCER_TYPE = "ALL" | "LUAD" | "LUSC".
    Tissue 过滤: 不做.
    CD8 过滤: 宽松 (sub_cell_type contains "CD8").
    """
    print("[Load] NSCLC scRNA-seq h5ad ...")
    ad = sc.read_h5ad(RNA_PATH)
    ad.obs_names = ad.obs_names.astype(str).str.strip()

    print("[Load] cell annotation ...")
    meta_df = pd.read_csv(META_PATH, dtype=str)
    meta_df.columns = [c.strip() for c in meta_df.columns]

    if "cellID" not in meta_df.columns:
        # 兜底: 第一列认作 cellID
        first_col = meta_df.columns[0]
        meta_df = meta_df.rename(columns={first_col: "cellID"})
    meta_df["cell_id"] = meta_df["cellID"].astype(str).str.strip()

    if "sub_cell_type" not in meta_df.columns:
        raise ValueError(f"[meta] missing sub_cell_type; cols={list(meta_df.columns)}")
    meta_df["sub_cell_type"] = meta_df["sub_cell_type"].astype(str).str.strip()

    if "cancer_type" not in meta_df.columns:
        raise ValueError(f"[meta] missing cancer_type; cols={list(meta_df.columns)}")
    meta_df["cancer_type"] = meta_df["cancer_type"].astype(str).str.strip()

    print(f"[meta] {len(meta_df)} rows, columns={list(meta_df.columns)}")
    print(f"[meta] cancer_type distribution:\n{meta_df['cancer_type'].value_counts()}")

    print("[Load] TCR (T_with_TCR_annotation.csv) ...")
    tcr_df = pd.read_csv(TCR_PATH, dtype=str).fillna("")
    tcr_df.columns = [c.strip() for c in tcr_df.columns]

    for col in ["cellID", "sampleID", "TRB_cdr3"]:
        if col not in tcr_df.columns:
            raise ValueError(f"[TCR] missing {col}; cols={list(tcr_df.columns)}")

    tcr_df["cell_id"] = tcr_df["cellID"].astype(str).str.strip()
    tcr_df["sampleID"] = tcr_df["sampleID"].astype(str).str.strip()
    tcr_df["TRB_aa"] = tcr_df["TRB_cdr3"].apply(clean_aa)

    if "TRA_cdr3" in tcr_df.columns:
        tcr_df["TRA_aa"] = tcr_df["TRA_cdr3"].apply(clean_aa)
    else:
        tcr_df["TRA_aa"] = ""

    # expansion 直接转 0/1 (expanded -> 1, non-expanded -> 0)
    if "expansion" in tcr_df.columns:
        tcr_df["is_expanded"] = (
            tcr_df["expansion"].astype(str).str.lower().str.strip().eq("expanded").astype(int)
        )
    else:
        tcr_df["is_expanded"] = np.nan

    # beta chain remains the required anchor
    tcr_df = tcr_df[tcr_df["TRB_aa"].str.len() > 0].copy()
    tcr_df = tcr_df.drop_duplicates("cell_id", keep="first")
    print(f"[TCR] valid TRB entries: {len(tcr_df)}")
    if "TRA_aa" in tcr_df.columns:
        print(f"[TCR] valid TRA ratio: {(tcr_df['TRA_aa'].str.len() > 0).mean():.3f}")
    if "is_expanded" in tcr_df.columns and tcr_df["is_expanded"].notna().any():
        print(f"[TCR] expanded ratio: {tcr_df['is_expanded'].mean():.3f}")

    print("[Load] sample metadata (Pathological Response) ...")
    sample_df = pd.read_csv(SAMPLE_META_PATH, dtype=str)
    sample_df.columns = [c.strip() for c in sample_df.columns]

    if "Tumor_Sample_Barcode" not in sample_df.columns:
        raise ValueError(
            f"[sample meta] need 'Tumor_Sample_Barcode'; cols={list(sample_df.columns)}"
        )
    if "Pathological Response" not in sample_df.columns:
        raise ValueError(
            f"[sample meta] need 'Pathological Response'; cols={list(sample_df.columns)}"
        )

    sample_df["sampleID"] = sample_df["Tumor_Sample_Barcode"].astype(str).str.strip()
    sample_df["Response"] = sample_df["Pathological Response"].astype(str).str.strip()
    sample_df = sample_df[["sampleID", "Response"]].drop_duplicates("sampleID")

    print("[Response] raw distribution (patient-level):")
    print(sample_df["Response"].value_counts(dropna=False))

    # ---- merge: cellID + meta + tcr + sample-level Response ----
    obs = pd.DataFrame(index=ad.obs_names)
    obs.index.name = "cell_id"

    meta_df2 = meta_df.drop_duplicates("cell_id").set_index("cell_id")
    # 留下除 cellID 外的所有 meta 列
    meta_keep_cols = [c for c in meta_df2.columns if c != "cellID"]
    obs = obs.merge(meta_df2[meta_keep_cols], left_index=True, right_index=True, how="left")

    # tcr 里的 sub_cell_type 可能与 meta 重名, 用 suffixes 区分
    tcr_keep_cols = ["TRA_aa", "TRB_aa", "is_expanded", "sampleID"]
    for extra_col in [
        "TRA_v_gene", "TRA_j_gene", "TRA_c_gene", "TRA_cdr3",
        "TRB_v_gene", "TRB_j_gene", "TRB_c_gene", "TRB_cdr3",
        "clonotype", "clonotype_number", "T_new_name", "expansion",
    ]:
        if extra_col in tcr_df.columns and extra_col not in tcr_keep_cols:
            tcr_keep_cols.append(extra_col)
    tcr_df2 = tcr_df.drop_duplicates("cell_id").set_index("cell_id")[tcr_keep_cols]

    obs = obs.merge(
        tcr_df2,
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_tcr"),
    )
    if obs.index.has_duplicates:
        obs = obs[~obs.index.duplicated(keep="first")]

    # sampleID 合并: 优先用 meta 里的 sampleID, 兜底用 tcr 里的
    if "sampleID" not in obs.columns and "sampleID_tcr" in obs.columns:
        obs["sampleID"] = obs["sampleID_tcr"]
    elif "sampleID_tcr" in obs.columns:
        obs["sampleID"] = obs["sampleID"].where(obs["sampleID"].notna(), obs["sampleID_tcr"])
    if "sampleID_tcr" in obs.columns:
        obs = obs.drop(columns=["sampleID_tcr"])

    obs["sampleID"] = obs["sampleID"].astype(str)

    obs = obs.merge(
        sample_df.set_index("sampleID"),
        left_on="sampleID",
        right_index=True,
        how="left",
    )

    obs = obs.reindex(ad.obs_names)
    for col in obs.columns:
        ad.obs[col] = obs[col].values
    ad.obs["cell_id"] = ad.obs_names

    # ---- Diagnostic: 完整 sub_cell_type 清单 (T/NK 大类) ----
    if "major_cell_type" in ad.obs.columns:
        print("\n[Diag] full sub_cell_type distribution in T/NK lineage:")
        tnk_mask_diag = ad.obs["major_cell_type"].astype(str).eq("T/NK cell")
        print(ad.obs.loc[tnk_mask_diag, "sub_cell_type"].value_counts())
    else:
        print("[Diag] WARNING: major_cell_type column not found; will rely on sub_cell_type prefix only")

    # ---- Cancer type 开关 ----
    print(f"\n[Filter] CANCER_TYPE = {CANCER_TYPE}")
    if CANCER_TYPE.upper() == "ALL":
        is_cancer = pd.Series(True, index=ad.obs_names)
    elif CANCER_TYPE.upper() in {"LUAD", "LUSC"}:
        is_cancer = ad.obs["cancer_type"].astype(str).str.upper() == CANCER_TYPE.upper()
    else:
        raise ValueError(f"unknown CANCER_TYPE={CANCER_TYPE}; expected ALL/LUAD/LUSC")
    is_cancer.index = ad.obs_names

    # ---- T 细胞过滤 ----
    # 融合模型只能处理有 TCR 的细胞，所以保留 CD4T / CD8T (含 Treg、Tfh、Tex、Texp、Tem 等子类型)
    # 自动排除纯 NK (例如 NK_CD16hi_FGFBP2)，因为没有 TCR
    if "sub_cell_type" not in ad.obs.columns:
        raise ValueError("missing sub_cell_type column in ad.obs after merge")

    sub_str = ad.obs["sub_cell_type"].astype(str)

    # T/NK 大类过滤 (如果该列存在)
    if "major_cell_type" in ad.obs.columns:
        is_tnk = ad.obs["major_cell_type"].astype(str).eq("T/NK cell")
    else:
        is_tnk = pd.Series(True, index=ad.obs_names)

    # 严格 T 细胞 (CD4T / CD8T 开头)，排除纯 NK
    is_t = sub_str.str.startswith("CD4T") | sub_str.str.startswith("CD8T")

    # TCR 必须存在
    has_trb = ad.obs["TRB_aa"].fillna("").astype(str).str.len() > 0

    # Response 必须存在
    has_response = (
            ad.obs["Response"].notna()
            & (~ad.obs["Response"].astype(str).isin(["nan", "NA", ""]))
    )

    # ---- Diagnostic: 分步骤报告过滤损失 ----
    print(f"\n[Filter] before any filter:        {ad.n_obs}")
    print(f"[Filter] after cancer ({CANCER_TYPE}): {int(is_cancer.sum())}")
    print(f"[Filter] after major=T/NK cell:    {int((is_cancer & is_tnk).sum())}")
    print(f"[Filter] after CD4T/CD8T prefix:   {int((is_cancer & is_tnk & is_t).sum())}")
    print(f"[Filter] after has_trb:            {int((is_cancer & is_tnk & is_t & has_trb).sum())}")
    print(f"[Filter] after has_response:       {int((is_cancer & is_tnk & is_t & has_trb & has_response).sum())}")

    keep = is_cancer & is_tnk & is_t & has_trb & has_response
    ad = ad[keep].copy()
    print(f"[Filter] FINAL: {ad.n_obs}")

    # ---- Diagnostic: 过滤后子类型分布 ----
    print("\n[Filter] sub_cell_type distribution after filter:")
    print(ad.obs["sub_cell_type"].value_counts())

    # ---- Response_binary: 把 pCR/MPR 合并成 MPR, 用于二分类对比 ----
    def _binarize_response(x):
        x = str(x).strip()
        if x in ("pCR", "MPR"):
            return "MPR"
        elif x == "non-MPR":
            return "non-MPR"
        else:
            return "NA"

    ad.obs["Response_binary"] = ad.obs["Response"].astype(str).map(_binarize_response)
    print("\n[Final] Response_binary distribution (cell level):")
    print(ad.obs["Response_binary"].value_counts())

    # ---- RNA preprocessing ----
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)

    sample_max = float(np.asarray(ad.X[:200].max() if hasattr(ad.X, "max") else 0.0))
    print(f"[Preproc] sample max value: {sample_max:.3f}")

    if sample_max > 30:
        print("  -> assuming raw counts; running normalize_total + log1p")
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    else:
        print("  -> already log-normalized, skip normalize_total/log1p")

    ad.raw = ad.copy()

    # 去除 TCR/Ig genes
    tcr_ig_pattern = re.compile(r"^(TR[ABGD][VJC]|IG[HKL][VJC])", flags=re.IGNORECASE)
    keep_genes = [g for g in ad.var_names if not tcr_ig_pattern.match(g)]
    removed = ad.n_vars - len(keep_genes)
    ad = ad[:, keep_genes].copy()
    print(f"[Filter] removed {removed} TCR/Ig genes; remaining: {ad.n_vars}")

    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.pp.scale(ad, max_value=10)

    if hasattr(ad.X, "toarray"):
        ad.X = ad.X.toarray()

    # ---- 字段类型规整 ----
    for col in ["sub_cell_type", "Response", "cancer_type"]:
        if col in ad.obs.columns:
            ad.obs[col] = (
                ad.obs[col].astype("string").fillna("NA").astype(str).astype("category")
            )

    ad.obs["sampleID"] = ad.obs["sampleID"].astype(str)
    ad.obs["TRA_aa"] = ad.obs["TRA_aa"].fillna("").astype(str)
    ad.obs["TRB_aa"] = ad.obs["TRB_aa"].fillna("").astype(str)

    # is_expanded: 优先用 TCR 文件提供的, 缺失再用 clone_size > 1 兜底
    vc = ad.obs["TRB_aa"].value_counts().to_dict()
    ad.obs["clone_size"] = ad.obs["TRB_aa"].map(vc).fillna(1).astype(int)

    if ad.obs["is_expanded"].isna().any():
        fill = (ad.obs["clone_size"] > 1).astype(int)
        ad.obs["is_expanded"] = ad.obs["is_expanded"].where(
            ad.obs["is_expanded"].notna(), fill
        )
    ad.obs["is_expanded"] = ad.obs["is_expanded"].fillna(0).astype(int)

    print("\n[Final] Response distribution (sample level):")
    resp_sample = ad.obs.groupby("sampleID", observed=True)["Response"].first()
    print(resp_sample.value_counts())

    print("\n[Final] Response distribution (cell level):")
    print(ad.obs["Response"].value_counts())

    print("\n[Final] sub_cell_type distribution:")
    print(ad.obs["sub_cell_type"].value_counts())

    print("\n[Final] cancer_type distribution:")
    print(ad.obs["cancer_type"].value_counts())

    print(f"\n[Final] is_expanded ratio: {ad.obs['is_expanded'].mean():.3f}")

    return ad
