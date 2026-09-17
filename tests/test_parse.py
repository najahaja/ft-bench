"""Tests for ftbench.data.schema parse utilities (extract_json_block, validate_output)."""
import json
import pytest
from ftbench.data.schema import extract_json_block, validate_output

VALID = {"intent": "weather_query", "slots": {"place_name": "tokyo"}}


def test_extract_raw_json():
    block = extract_json_block(json.dumps(VALID))
    assert json.loads(block) == VALID


def test_extract_fenced_json():
    text = f"```json\n{json.dumps(VALID)}\n```"
    block = extract_json_block(text)
    assert json.loads(block) == VALID


def test_extract_json_with_prose():
    text = f"Sure! Here is the result: {json.dumps(VALID)} Hope that helps."
    block = extract_json_block(text)
    assert json.loads(block) == VALID


def test_validate_valid():
    ok, model, err = validate_output(json.dumps(VALID))
    assert ok is True
    assert model.intent == "weather_query"
    assert model.slots == {"place_name": "tokyo"}


def test_validate_empty_slots():
    d = {"intent": "general_quirky", "slots": {}}
    ok, model, err = validate_output(json.dumps(d))
    assert ok is True


def test_validate_missing_intent():
    d = {"slots": {"place_name": "london"}}
    ok, model, err = validate_output(json.dumps(d))
    assert ok is False


def test_validate_missing_slots():
    d = {"intent": "alarm_set"}
    ok, model, err = validate_output(json.dumps(d))
    assert ok is False


def test_validate_garbled():
    ok, model, err = validate_output("not json at all")
    assert ok is False
    assert model is None
