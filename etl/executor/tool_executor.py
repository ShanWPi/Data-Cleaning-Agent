import pandas as pd
from typing import Dict, Any

from etl.transform import cleaners


# 🔒 Tool registry (single source of truth)
TOOL_REGISTRY = {
    "clean_column_names": cleaners.clean_column_names,
    "standardize_missing": cleaners.standardize_missing,
    "trim_whitespace": cleaners.trim_whitespace,
    "remove_duplicates": cleaners.remove_duplicates,
    "convert_numeric": cleaners.convert_numeric,
    "parse_datetime": cleaners.parse_datetime,
}


class ToolExecutionError(Exception):
    pass


def execute_tool_step(
    df: pd.DataFrame,
    step: Dict[str, Any]
) -> pd.DataFrame:
    """
    Executes a single tool step on the dataframe.
    """

    tool_name = step.get("name")
    args = step.get("args", {})

    if tool_name not in TOOL_REGISTRY:
        raise ToolExecutionError(f"Tool not registered: {tool_name}")

    tool_fn = TOOL_REGISTRY[tool_name]

    try:
        return tool_fn(df, **args)
    except Exception as e:
        raise ToolExecutionError(
            f"Error executing tool '{tool_name}': {e}"
        ) from e


def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any]
) -> pd.DataFrame:
    """
    Executes all tool steps in sequence.
    """

    if "steps" not in plan:
        raise ToolExecutionError("Plan has no steps")

    current_df = df.copy()

    for idx, step in enumerate(plan["steps"], start=1):
        if step.get("type") != "tool":
            raise ToolExecutionError(
                f"Step {idx}: only tool steps are supported"
            )

        current_df = execute_tool_step(current_df, step)

    return current_df
