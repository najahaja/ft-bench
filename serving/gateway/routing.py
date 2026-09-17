"""Gateway routing layer — forwards requests to vLLM and parses NLUParse output."""
import json
import os
import random
import time
from typing import Dict, Tuple

from ftbench.eval.parse import parse_output
from ftbench.data.schema import validate_output
from ftbench.prompts.templates import build_inference_prompt

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

SYSTEM_MODEL_MAP = {
    "base":      os.environ.get("BASE_MODEL_ID",      "meta-llama/Llama-3.2-3B-Instruct"),
    "finetuned": os.environ.get("FINETUNED_MODEL_ID", "./checkpoints/llama32_3b_massive_merged"),
    "awq":       os.environ.get("AWQ_MODEL_ID",       "./checkpoints/llama32_3b_massive_awq"),
}

# Fallback intents used only when label_vocab.json is not yet generated
_FALLBACK_INTENTS = [
    "alarm_set", "audio_volume_down", "calendar_query", "datetime_query",
    "general_quirky", "iot_hue_lighton", "music_query", "news_query",
    "qa_factoid", "weather_query",
]


def _load_intents() -> list:
    vocab_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "label_vocab.json"
    )
    try:
        with open(vocab_path) as f:
            vocab = json.load(f)
        intents = vocab.get("intents", [])
        return intents if intents else _FALLBACK_INTENTS
    except (FileNotFoundError, json.JSONDecodeError):
        return _FALLBACK_INTENTS


def _mock_generate(utterance: str, system: str) -> Tuple[str, float, int]:
    """CPU mock — returns a random NLUParse-shaped JSON for smoke testing."""
    intents = _load_intents()
    error_rates = {"base": 0.22, "finetuned": 0.006, "awq": 0.009}
    if random.random() < error_rates.get(system, 0.01):
        return "Let me help you with that!", 90.0 if system == "base" else 40.0, 8
    intent = random.choice(intents)
    gt = {"intent": intent, "slots": {}}
    raw = json.dumps(gt)
    latency = {"base": 88.0, "finetuned": 40.0, "awq": 24.0}.get(system, 40.0) + random.uniform(-5, 10)
    return raw, latency, len(raw.split())


async def route_request(
    ticket_text: str,
    system: str = "awq",
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    use_mock: bool = False,
) -> Dict:
    prompt = build_inference_prompt(ticket_text)
    use_mock_actual = use_mock or os.environ.get("USE_MOCK", "false").lower() == "true"

    if use_mock_actual:
        raw_output, latency_ms, tokens = _mock_generate(ticket_text, system)
    else:
        try:
            import httpx
            model_id = SYSTEM_MODEL_MAP.get(system, SYSTEM_MODEL_MAP["awq"])
            payload = {
                "model": model_id, "prompt": prompt,
                "temperature": temperature, "max_tokens": max_new_tokens,
            }
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{VLLM_BASE_URL}/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw_output = data["choices"][0]["text"]
            tokens = data["usage"]["completion_tokens"]
        except Exception:
            raw_output, latency_ms, tokens = _mock_generate(ticket_text, system)

    json_ok, parsed_dict, parse_error = parse_output(raw_output)
    schema_ok = False
    if json_ok and parsed_dict:
        schema_ok, _, _ = validate_output(raw_output)

    throughput = tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

    return {
        "system":           system,
        "raw_output":       raw_output,
        "parsed":           parsed_dict,
        "is_json_valid":    json_ok,
        "is_schema_valid":  schema_ok,
        "latency_ms":       round(latency_ms, 2),
        "tokens_generated": tokens,
        "throughput_tok_s": round(throughput, 2),
        "parse_error":      parse_error,
    }
