"""Tests for ftbench.data.schema — NLUParse validation and JSON extraction."""
import json
import pytest
from ftbench.data.schema import NLUParse, validate_output, extract_json_block

# vocab file will be placeholder (empty lists) during unit tests;
# the validator skips enforcement when lists are empty.
VALID_DICT = {
    "intent": "alarm_set",
    "slots": {"date": "tomorrow", "time": "7am"},
}


def test_valid_schema():
    model = NLUParse(**VALID_DICT)
    assert model.intent == "alarm_set"
    assert model.slots == {"date": "tomorrow", "time": "7am"}


def test_empty_slots_valid():
    model = NLUParse(intent="general_quirky", slots={})
    assert model.slots == {}


def test_validate_output_clean_json():
    ok, model, err = validate_output(json.dumps(VALID_DICT))
    assert ok is True
    assert model is not None
    assert err is None


def test_validate_output_markdown_fenced():
    raw = f"```json\n{json.dumps(VALID_DICT)}\n```"
    ok, model, err = validate_output(raw)
    assert ok is True
    assert model is not None


def test_validate_output_invalid():
    ok, model, err = validate_output("Sorry, I cannot help with that.")
    assert ok is False
    assert model is None
    assert err is not None


def test_extract_json_block():
    text = f"Here you go: {json.dumps(VALID_DICT)} That is the parse."
    block = extract_json_block(text)
    data = json.loads(block)
    assert data["intent"] == "alarm_set"


def test_no_ticket_analysis_fields():
    """Confirm TicketAnalysis fields (sentiment, priority, category, escalate) are absent."""
    import ftbench.data.schema as schema_mod
    assert not hasattr(schema_mod, "TicketAnalysis"), \
        "TicketAnalysis must not exist in schema.py"
    assert not hasattr(schema_mod, "SENTIMENT_VOCAB"), \
        "SENTIMENT_VOCAB must not exist in schema.py"
    assert not hasattr(schema_mod, "PRIORITY_VOCAB"), \
        "PRIORITY_VOCAB must not exist in schema.py"
    assert not hasattr(schema_mod, "CATEGORY_VOCAB"), \
        "CATEGORY_VOCAB must not exist in schema.py"
