import pandas as pd
from typing import Dict, Any


class ValidationError(Exception):
    pass


def validate_transformation(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    max_row_loss_pct: float = 30.0,
    max_null_increase_pct: float = 50.0,
) -> None:
    """
    Validates transformation safety.
    Raises ValidationError if unsafe.
    """

    # ---------------------------
    # 1. Empty dataset check
    # ---------------------------
    if df_after.empty:
        raise ValidationError("Output dataset is empty")

    # ---------------------------
    # 2. Row loss check
    # ---------------------------
    rows_before = len(df_before)
    rows_after = len(df_after)

    if rows_before > 0:
        row_loss_pct = ((rows_before - rows_after) / rows_before) * 100
        if row_loss_pct > max_row_loss_pct:
            raise ValidationError(
                f"Too many rows dropped: {row_loss_pct:.2f}%"
            )

    # ---------------------------
    # 3. Column disappearance
    # ---------------------------
    missing_columns = set(df_before.columns) - set(df_after.columns)
    if missing_columns:
        raise ValidationError(
            f"Columns removed unexpectedly: {missing_columns}"
        )

    # ---------------------------
    # 4. Column null explosion
    # ---------------------------
    for col in df_before.columns:
        before_null_pct = df_before[col].isna().mean() * 100
        after_null_pct = df_after[col].isna().mean() * 100

        if (after_null_pct - before_null_pct) > max_null_increase_pct:
            raise ValidationError(
                f"Column '{col}' nulls increased too much "
                f"({before_null_pct:.2f}% → {after_null_pct:.2f}%)"
            )
