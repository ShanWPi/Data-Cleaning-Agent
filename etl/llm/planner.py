import json
import os
from typing import Dict, Any, List

from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ============================================================
# 1. CONFIGURATION (LOCKED CONTRACTS)
# ============================================================

ALLOWED_TOOLS: List[str] = [
    "clean_column_names",
    "standardize_missing",
    "trim_whitespace",
    "remove_duplicates",
    "convert_numeric",
    "parse_datetime",
]

PLAN_SCHEMA_EXAMPLE = {
    "steps": [
        {
            "type": "tool",
            "name": "clean_column_names",
            "args": {}
        }
    ]
}


# ============================================================
# 2. PROMPTS (ALL PROMPTING LIVES HERE)
# ============================================================

SYSTEM_PROMPT = """
You are a data cleaning planner for a production ETL system.

STRICT RULES (MUST FOLLOW):
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include code blocks
- Do NOT include comments
- Do NOT include text outside JSON

YOU ARE NOT ALLOWED TO:
- Write Python code
- Execute transformations
- Access files
- Guess business meaning

YOUR TASK:
Given dataset profiling metadata, generate a SAFE, MINIMAL,
and CONSERVATIVE data cleaning plan.

Only suggest steps that are clearly justified by the metadata.
"""


def build_user_prompt(profile_json: str) -> str:
    return f"""
Dataset profile (JSON):
{profile_json}

Allowed tools:
{ALLOWED_TOOLS}

Required output schema (example):
{PLAN_SCHEMA_EXAMPLE}

Rules:
- Use ONLY the allowed tools
- Each step must be necessary
- Do NOT invent tools
- Do NOT add unnecessary steps

Generate the cleaning plan now.
"""


# ============================================================
# 3. LLM CLIENT (GROQ)
# ============================================================

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)


def call_groq_llm(system_prompt: str, user_prompt: str) -> str:
    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,   # deterministic
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 4. OUTPUT VALIDATION (NON-NEGOTIABLE)
# ============================================================

def validate_plan_schema(plan: Dict[str, Any]) -> None:
    if step["name"] == "convert_numeric":
        if "column" not in step["args"]:
            raise ValueError("convert_numeric requires 'column'")

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")

    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise ValueError("Plan must contain a 'steps' list")

    for idx, step in enumerate(plan["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"Step {idx} is not an object")

        if step.get("type") != "tool":
            raise ValueError(f"Step {idx}: only 'tool' type is allowed")

        tool_name = step.get("name")
        if tool_name not in ALLOWED_TOOLS:
            raise ValueError(f"Step {idx}: tool not allowed: {tool_name}")

        if "args" not in step or not isinstance(step["args"], dict):
            raise ValueError(f"Step {idx}: 'args' must be a dict")


# ============================================================
# 5. PUBLIC ENTRY POINT (PIPELINE CALLS ONLY THIS)
# ============================================================

def generate_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: profiler output (dict)
    Output: validated cleaning plan (dict)
    """

    # 1. Ensure JSON-safe input
    try:
        profile_json = json.dumps(profile, indent=2)
    except TypeError as e:
        raise ValueError(f"Profile is not JSON-serializable: {e}")

    # 2. Build prompts
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(profile_json)

    # 3. Call Groq
    llm_output = call_groq_llm(system_prompt, user_prompt)

    # 4. Parse JSON
    try:
        plan = json.loads(llm_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output is not valid JSON.\nRaw output:\n{llm_output}"
        ) from e

    # 5. Validate schema
    validate_plan_schema(plan)

    return plan
