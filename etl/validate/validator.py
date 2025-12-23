import pandas as pd
from typing import Dict, Any, Optional, Set


class ValidationError(Exception):
    pass


def _get_planned_dropped_columns(plan: Optional[Dict[str, Any]]) -> Set[str]:
    """
    Extracts columns that were explicitly planned to be dropped.
    """
    if not plan or "steps" not in plan:
        return set()

    dropped = set()
    for step in plan["steps"]:
        if step.get("type") == "tool" and step.get("name") == "drop_column":
            col = step.get("args", {}).get("column")
            if col:
                dropped.add(col)

    return dropped


def validate_transformation(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    plan: Optional[Dict[str, Any]] = None,
    max_row_loss_pct: float = 30.0,
    max_null_increase_pct: float = 50.0,
) -> None:
    """
    Validates transformation safety.

    IMPORTANT:
    - Allows no-op transformations
    - Allows explicitly planned column drops
    - Protects against destructive transformations
    """

    # ---------------------------
    # 0. Allow no-op transformations
    # ---------------------------
    if df_before.equals(df_after):
        return

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
    # 3. Column disappearance check (planner-aware)
    # ---------------------------
    before_cols = set(df_before.columns)
    after_cols = set(df_after.columns)

    removed_columns = before_cols - after_cols
    allowed_drops = _get_planned_dropped_columns(plan)

    unexpected_drops = removed_columns - allowed_drops
    if unexpected_drops:
        raise ValidationError(
            f"Columns removed unexpectedly: {unexpected_drops}"
        )

    # ---------------------------
    # 4. Column null explosion
    # ---------------------------
    common_cols = before_cols & after_cols

    for col in common_cols:
        before_null_pct = df_before[col].isna().mean() * 100
        after_null_pct = df_after[col].isna().mean() * 100

        if (after_null_pct - before_null_pct) > max_null_increase_pct:
            raise ValidationError(
                f"Column '{col}' nulls increased too much "
                f"({before_null_pct:.2f}% → {after_null_pct:.2f}%)"
            )

    # ---------------------------
    # 5. Passed validation
    # ---------------------------
    return
