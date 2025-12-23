import pandas as pd
from typing import List, Optional
import re


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

    missing_markers = {
        "": pd.NA,
        " ": pd.NA,
        "null": pd.NA,
        "n/a": pd.NA,
        "na": pd.NA,
    }

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.strip()
                .str.lower()
                .replace(missing_markers)
            )

    return df


def trim_whitespace(
    df: pd.DataFrame,
    column: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = df.copy()

    if column:
        target_cols = [column]
    elif columns:
        target_cols = columns
    else:
        target_cols = df.select_dtypes(include="object").columns

    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].where(
                df[col].isna(),
                df[col].str.strip()
            )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def convert_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    return df.drop(columns=[column])

import re

def normalize_currency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(r"[^\d\.]", "", regex=True)
    )

    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def normalize_percentage(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df

    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df[column] = pd.to_numeric(df[column], errors="coerce") / 100
    return df
