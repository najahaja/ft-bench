"""Robust JSON extraction from model outputs."""
import json
import re
from typing import Any, Dict, Optional, Tuple


def normalize_booleans(text: str) -> str:
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_output(raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    text = strip_markdown_fences(raw_text)
    text = normalize_booleans(text)
    json_str = extract_first_json_object(text)
    if json_str is None:
        return False, None, "No JSON object found in output"
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return False, None, f"Parsed JSON is not a dict: {type(obj)}"
        return True, obj, None
    except json.JSONDecodeError as e:
        return False, None, f"JSONDecodeError: {e}"
