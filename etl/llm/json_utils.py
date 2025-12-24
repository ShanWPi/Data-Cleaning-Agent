import logging
import json
import ast

logger = logging.getLogger(__name__)


class LLMJSONError(Exception):
    pass


def parse_llm_json(text: str) -> dict:
    """
    Attempts strict JSON parsing first.
    Falls back to safe Python literal parsing.
    """

    logger.debug("Entering parse_llm_json")
    # 1️⃣ Strict JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2️⃣ Safe Python literal fallback
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    raise LLMJSONError(f"LLM returned unparseable JSON:\n{text}")
