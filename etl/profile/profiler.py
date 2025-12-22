import pandas as pd
import numpy as np
from typing import Dict, Any
import re
import pandas as pd

DATE_REGEX = re.compile(
    r"""
    ^
    (\d{4}[-/]\d{1,2}[-/]\d{1,2}) |     # YYYY-MM-DD or YYYY/MM/DD
    (\d{1,2}[-/]\d{1,2}[-/]\d{2,4}) |   # DD-MM-YYYY or MM/DD/YYYY
    (\d{4}\d{2}\d{2})                  # YYYYMMDD
    $
    """,
    re.VERBOSE,
)


def infer_semantic_type(series: pd.Series) -> str:
    non_null = series.dropna()

    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(non_null):
        return "boolean"

    if pd.api.types.is_numeric_dtype(non_null):
        return "numeric"

    # 🚨 STRONG GUARD: regex before datetime parsing
    if non_null.dtype == object:
        sample = non_null.astype(str).head(20)

        date_like_ratio = sum(
            bool(DATE_REGEX.match(val.strip())) for val in sample
        ) / len(sample)

        if date_like_ratio >= 0.8:
            parsed = pd.to_datetime(sample, errors="coerce")

            if parsed.notna().mean() >= 0.8:
                return "datetime"

    unique_ratio = non_null.nunique() / len(non_null)

    if unique_ratio < 0.05:
        return "categorical"

    return "text"




def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile a dataframe without mutating it.
    """
    profile: Dict[str, Any] = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_cells_pct": float(df.isna().mean().mean() * 100)
        },
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        col_profile = {
            "dtype": str(series.dtype),
            "semantic_type": infer_semantic_type(series),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_count": int(non_null.nunique()),
            "sample_values": non_null.astype(str).head(3).tolist(),
            "numeric_string_ratio": float(non_null.astype(str).str.count(r'^-?\d+\.?\d*$').sum() / len(non_null) if len(non_null) > 0 else 0)
        }

        # numeric stats (only if numeric-like)
        if col_profile["semantic_type"] == "numeric":
            col_profile.update({
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0
            })

        profile["columns"][col] = col_profile

    return profile
