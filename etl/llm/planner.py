import json
import os
from typing import Dict, Any, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# CONFIG
# ======================================================

ALLOWED_TOOLS = [
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

# ======================================================
# PROMPTS
# ======================================================

SYSTEM_PROMPT = """
You are a data-cleaning planner for a production ETL system. 

STRICT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No comments
- No text outside JSON

YOU MAY ONLY:
- Select tools from the allowed list
- Provide COMPLETE arguments for tools

DO NOT:
- Guess column names
- Repeat failed actions
- Be aggressive unless justified by metadata
"""

def build_user_prompt(
    profile_json: str,
    feedback: Optional[Dict[str, Any]] = None
) -> str:

    feedback_block = ""
    if feedback:
        feedback_block = f"""
Previous attempt FAILED.

Failure details:
{json.dumps(feedback, indent=2)}

Rules:
- Do NOT repeat the failed step
- Fix missing or incorrect arguments
- Be more conservative
"""

    return f"""
Dataset profile (JSON):
{profile_json}

{feedback_block}

Allowed tools:
{ALLOWED_TOOLS}

Output schema example:
{PLAN_SCHEMA_EXAMPLE}

Generate a minimal, safe cleaning plan.
"""

# ======================================================
# OpenAI Client
# ======================================================

def get_openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return api_key


def call_openai(system_prompt: str, user_prompt: str) -> str:
    api_key = get_openai_key()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )

    # extract content
    try:
        content = response.choices[0].message.content
    except Exception:
        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:
            raise ValueError(f"OpenAI returned no content: {response}")

    return content.strip()


def call_groq(system_prompt: str, user_prompt: str) -> str:
    """Compatibility wrapper for older code expecting `call_groq()`.

    Delegates to `call_openai`.
    """
    return call_openai(system_prompt, user_prompt)

# ======================================================
# VALIDATION
# ======================================================

def validate_plan(plan: Dict[str, Any]) -> None:
    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise ValueError("Plan must contain 'steps' list")

    for step in plan["steps"]:
        if step.get("type") != "tool":
            raise ValueError("Only tool steps allowed")

        name = step.get("name")
        args = step.get("args")

        if name not in ALLOWED_TOOLS:
            raise ValueError(f"Tool not allowed: {name}")

        if not isinstance(args, dict):
            raise ValueError("Args must be a dict")

        # if name in {"convert_numeric", "parse_datetime"}:
        #     if "column" not in args:
        #         raise ValueError(f"{name} requires 'column'")

# ======================================================
# PUBLIC API
# ======================================================

def generate_plan(
    profile: Dict[str, Any],
    feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    profile_json = json.dumps(profile, indent=2)

    user_prompt = build_user_prompt(profile_json, feedback)
    llm_output = call_groq(SYSTEM_PROMPT, user_prompt)

    try:
        plan = json.loads(llm_output)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON:\n{llm_output}")

    validate_plan(plan)
    return plan
