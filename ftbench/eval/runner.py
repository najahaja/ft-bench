"""
Shared evaluation loop - the reason ftbench is an installable package.
All three systems (base, finetuned, quantized) use this identical loop.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from ftbench.data.schema import validate_output
from ftbench.eval.metrics import compute_metrics
from ftbench.eval.parse import parse_output
from ftbench.prompts.templates import build_inference_prompt


def run_eval(
    samples: List[Dict[str, Any]],
    generate_fn: Callable[[str], str],
    system_name: str = "unknown",
    output_path: Optional[str] = None,
    few_shot_examples: Optional[List[dict]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        gt = sample["ground_truth"]
        prompt = build_inference_prompt(text, few_shot_examples)

        t0 = time.perf_counter()
        raw_output = generate_fn(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        json_ok, parsed_dict, parse_error = parse_output(raw_output)
        schema_ok, validated, schema_error = False, None, None
        if json_ok and parsed_dict is not None:
            schema_ok, validated, schema_error = validate_output(raw_output)

        tokens_generated = len(raw_output.split())

        record = {
            "id": sample.get("id", str(i)),
            "system": system_name,
            "text": text,
            "ground_truth": gt,
            "raw_output": raw_output,
            "parsed": parsed_dict,
            "is_json_valid": json_ok,
            "is_schema_valid": schema_ok,
            "parse_error": parse_error,
            "schema_error": schema_error,
            "latency_ms": round(latency_ms, 2),
            "tokens_generated": tokens_generated,
        }
        results.append(record)

        if verbose and not schema_ok:
            print(f"  [{i+1}/{len(samples)}] FAIL - parse_error={parse_error}")

    if output_path:
        import json
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    return results
