"""Shared amino-acid sequence cleaning for paired TCR data."""

import pandas as pd

AASET = set("ACDEFGHIKLMNPQRSTVWY")


def clean_aa(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return "".join(ch for ch in str(seq).upper() if ch in AASET)
