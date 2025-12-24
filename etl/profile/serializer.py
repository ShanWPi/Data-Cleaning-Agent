import logging
import json

logger = logging.getLogger(__name__)


def ensure_json_serializable(obj: dict) -> dict:
    logger.debug("Entering ensure_json_serializable")
    try:
        json.dumps(obj)
        return obj
    except TypeError as e:
        raise ValueError(f"Profiler output is not JSON-serializable: {e}")
