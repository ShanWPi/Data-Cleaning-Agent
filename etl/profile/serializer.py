import json

def ensure_json_serializable(obj: dict) -> dict:
    try:
        json.dumps(obj)
        return obj
    except TypeError as e:
        raise ValueError(f"Profiler output is not JSON-serializable: {e}")
