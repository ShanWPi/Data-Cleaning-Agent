import pandas as pd
from typing import List, Optional


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_markers = ["", "na", "n/a", "null", "none", "nan", "?"]

    for col in df.columns:
        df[col] = (
            df[col]
            .replace(missing_markers, pd.NA)
        )

    return df


def trim_whitespace(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    df = df.copy()

    target_cols = columns if columns else df.select_dtypes(include="object").columns

    for col in target_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def convert_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df
