import pandas as pd
from typing import Dict, Any

from etl.extract.reader import read_csv_safe
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.llm.planner import generate_plan
from etl.executor.tool_executor import execute_plan
from etl.validate.validator import validate_transformation


class PipelineError(Exception):
    pass


def run_pipeline(
    input_csv_path: str,
    output_csv_path: str
) -> Dict[str, Any]:
    """
    End-to-end ETL pipeline.

    Returns execution metadata.
    """

    # ---------------------------
    # 1. READ
    # ---------------------------
    try:
        df_raw, read_meta = read_csv_safe(input_csv_path)
    except Exception as e:
        raise PipelineError(f"CSV read failed: {e}")

    # ---------------------------
    # 2. PROFILE
    # ---------------------------
    try:
        profile = profile_dataframe(df_raw)
        profile = ensure_json_serializable(profile)
    except Exception as e:
        raise PipelineError(f"Profiling failed: {e}")

    # ---------------------------
    # 3. PLAN (LLM)
    # ---------------------------
    try:
        plan = generate_plan(profile)
    except Exception as e:
        raise PipelineError(f"LLM planning failed: {e}")

    # ---------------------------
    # 4. EXECUTE
    # ---------------------------
    try:
        try:
            df_clean = execute_plan(df_raw, plan)
            validate_transformation(df_raw, df_clean)
        except Exception as e:
            raise PipelineError(f"Validation failed: {e}")
    except Exception as e:
        raise PipelineError(f"Plan execution failed: {e}")

    # ---------------------------
    # 5. WRITE OUTPUT
    # ---------------------------
    try:
        df_clean.to_csv(output_csv_path, index=False)
    except Exception as e:
        raise PipelineError(f"Writing output failed: {e}")

    # ---------------------------
    # 6. RETURN METADATA
    # ---------------------------
    return {
        "input_rows": len(df_raw),
        "output_rows": len(df_clean),
        "steps_executed": len(plan.get("steps", [])),
        "plan": plan,
        "read_metadata": read_meta,
    }
