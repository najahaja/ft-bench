"""
Task-specific metrics for the FT-Bench NLU parsing benchmark.

Evaluated fields: intent (exact match) + slot token-F1.
No categories, no sentiments, no priorities — those fields do not exist
in the MASSIVE/NLUParse schema.
"""
from typing import Any, Dict, List, Optional

import numpy as np


def _intent_match(pred: Optional[dict], gt: dict) -> bool:
    if pred is None:
        return False
    return str(pred.get("intent", "")).strip() == str(gt.get("intent", "")).strip()


def _slot_f1(pred: Optional[dict], gt: dict) -> float:
    """
    Micro-averaged token-F1 over slot values.
    Treats each slot value as a bag of tokens; averages across all slots in gt.
    """
    if pred is None:
        return 0.0
    gt_slots: dict = gt.get("slots", {})
    pred_slots: dict = pred.get("slots", {}) if pred else {}

    if not gt_slots and not pred_slots:
        return 1.0

    total_tp = total_pred = total_true = 0
    all_keys = set(gt_slots.keys()) | set(pred_slots.keys())
    for key in all_keys:
        gt_tokens  = set(gt_slots.get(key,   "").lower().split())
        pred_tokens = set(pred_slots.get(key, "").lower().split())
        tp = len(gt_tokens & pred_tokens)
        total_tp   += tp
        total_pred += len(pred_tokens)
        total_true += len(gt_tokens)

    prec = total_tp / total_pred if total_pred > 0 else 0.0
    rec  = total_tp / total_true if total_true > 0 else 0.0
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def _exact_match(pred: Optional[dict], gt: dict) -> bool:
    """Intent correct AND all slot keys/values correct simultaneously."""
    if not _intent_match(pred, gt):
        return False
    return (pred or {}).get("slots", {}) == gt.get("slots", {})


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute all benchmark metrics over a list of prediction records.
    Each record must have keys: is_json_valid, is_schema_valid, parsed, ground_truth,
    latency_ms, tokens_generated.
    """
    n = len(records)
    if n == 0:
        return {}

    json_valid   = sum(1 for r in records if r.get("is_json_valid",   False))
    schema_valid = sum(1 for r in records if r.get("is_schema_valid", False))

    intent_hits  = 0
    slot_f1_sum  = 0.0
    exact_hits   = 0

    for r in records:
        pred = r.get("parsed")
        gt   = r.get("ground_truth", {})
        if _intent_match(pred, gt):
            intent_hits += 1
        slot_f1_sum += _slot_f1(pred, gt)
        if _exact_match(pred, gt):
            exact_hits += 1

    latencies = np.array([r.get("latency_ms", 0.0) for r in records])
    tokens     = sum(r.get("tokens_generated", 0) for r in records)
    total_time_s = latencies.sum() / 1000.0

    return {
        "n":                  n,
        "json_valid_rate":    round(json_valid   / n * 100, 2),
        "schema_valid_rate":  round(schema_valid / n * 100, 2),
        "intent_accuracy":    round(intent_hits  / n * 100, 2),
        "slot_f1":            round(slot_f1_sum  / n * 100, 2),
        "exact_match":        round(exact_hits   / n * 100, 2),
        "latency_p50_ms":     round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms":     round(float(np.percentile(latencies, 95)), 2),
        "mean_latency_ms":    round(float(np.mean(latencies)), 2),
        "throughput_tok_s":   round(tokens / total_time_s, 2) if total_time_s > 0 else 0.0,
    }
