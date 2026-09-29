"""Tests for common utilities, stats, and templates to maintain high test coverage."""
from ftbench.common.seed import seed_everything
from ftbench.common.io import read_jsonl, write_jsonl, read_json, write_json, push_to_hub
from ftbench.common.gpu import get_gpu_stats, peak_vram_mb
from ftbench.eval.stats import bootstrap_ci
from ftbench.prompts.templates import build_inference_prompt, build_training_prompt


def test_seed_everything():
    seed_everything(123)
    import random
    val1 = random.random()
    seed_everything(123)
    val2 = random.random()
    assert val1 == val2


def test_io_helpers(tmp_path):
    records = [{"id": "1", "val": "abc"}, {"id": "2", "val": "xyz"}]
    jsonl_path = tmp_path / "test.jsonl"
    write_jsonl(records, jsonl_path)
    loaded = read_jsonl(jsonl_path)
    assert loaded == records

    json_path = tmp_path / "test.json"
    data = {"hello": "world"}
    write_json(data, json_path)
    assert read_json(json_path) == data

    # push_to_hub fallback on missing token/hub
    push_to_hub(str(json_path), "nonexistent/repo")


def test_gpu_helpers():
    stats = get_gpu_stats()
    assert stats is None or isinstance(stats, dict)
    vram = peak_vram_mb()
    assert vram is None or isinstance(vram, (int, float))


def test_bootstrap_ci():
    records = [
        {"intent_match": True, "slot_f1": 1.0, "exact_match": True, "json_valid": True},
        {"intent_match": False, "slot_f1": 0.0, "exact_match": False, "json_valid": True},
        {"intent_match": True, "slot_f1": 0.8, "exact_match": True, "json_valid": True},
    ]
    mean, lower, upper = bootstrap_ci(records, "exact_match", n_bootstrap=50, seed=42)
    assert 0.0 <= lower <= mean <= upper <= 100.0


def test_prompt_templates():
    p = build_inference_prompt("book a flight from seattle to dallas")
    assert "book a flight from seattle to dallas" in p
    assert "[SYSTEM]" in p
    assert "[USER]" in p

    t = build_training_prompt("wake me up at 7am", {"intent": "alarm_set", "slots": {"time": "7am"}})
    assert "wake me up at 7am" in t
    assert "alarm_set" in t
