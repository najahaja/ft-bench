"""Tests for ftbench.eval.metrics — NLUParse-shaped records."""
import pytest
from ftbench.eval.metrics import compute_metrics, _intent_match, _slot_f1, _exact_match

GT = {"intent": "alarm_set", "slots": {"date": "tomorrow", "time": "7am"}}
PRED_CORRECT = {"intent": "alarm_set", "slots": {"date": "tomorrow", "time": "7am"}}
PRED_WRONG_INTENT = {"intent": "music_query", "slots": {"date": "tomorrow", "time": "7am"}}
PRED_PARTIAL_SLOTS = {"intent": "alarm_set", "slots": {"date": "tomorrow"}}
PRED_NONE = None


def _make_record(pred, gt=GT, json_valid=True, schema_valid=True, latency_ms=10.0, tokens=20):
    return {
        "is_json_valid": json_valid and pred is not None,
        "is_schema_valid": schema_valid and pred is not None,
        "parsed": pred,
        "ground_truth": gt,
        "latency_ms": latency_ms,
        "tokens_generated": tokens,
    }


def test_intent_match_correct():
    assert _intent_match(PRED_CORRECT, GT) is True


def test_intent_match_wrong():
    assert _intent_match(PRED_WRONG_INTENT, GT) is False


def test_intent_match_none():
    assert _intent_match(None, GT) is False


def test_slot_f1_perfect():
    assert _slot_f1(PRED_CORRECT, GT) == 1.0


def test_slot_f1_partial():
    f1 = _slot_f1(PRED_PARTIAL_SLOTS, GT)
    assert 0.0 < f1 < 1.0


def test_slot_f1_none():
    assert _slot_f1(None, GT) == 0.0


def test_exact_match_correct():
    assert _exact_match(PRED_CORRECT, GT) is True


def test_exact_match_wrong_intent():
    assert _exact_match(PRED_WRONG_INTENT, GT) is False


def test_exact_match_partial_slots():
    assert _exact_match(PRED_PARTIAL_SLOTS, GT) is False


def test_compute_metrics_all_correct():
    records = [_make_record(PRED_CORRECT) for _ in range(10)]
    m = compute_metrics(records)
    assert m["intent_accuracy"] == 100.0
    assert m["slot_f1"] == 100.0
    assert m["exact_match"] == 100.0
    assert m["n"] == 10


def test_compute_metrics_all_wrong():
    records = [_make_record(PRED_WRONG_INTENT) for _ in range(5)]
    m = compute_metrics(records)
    assert m["intent_accuracy"] == 0.0


def test_compute_metrics_empty():
    assert compute_metrics([]) == {}


def test_no_ticket_fields_in_metrics():
    """Confirm TicketAnalysis fields are not returned by compute_metrics."""
    records = [_make_record(PRED_CORRECT)]
    m = compute_metrics(records)
    for bad_key in ("category_accuracy", "sentiment_accuracy", "priority_accuracy", "escalate_accuracy"):
        assert bad_key not in m, f"{bad_key} must not appear in metrics output"
