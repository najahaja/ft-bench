import os
os.environ["USE_MOCK"] = "true"

import pytest
from fastapi.testclient import TestClient
from serving.gateway.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert "awq" in resp.json()["available_systems"]


def test_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    assert "awq" in resp.json()["systems"]


def test_generate_base():
    resp = client.post("/generate", json={"utterance": "Cancel my order #12345", "system": "base"})
    assert resp.status_code == 200
    data = resp.json()
    assert "latency_ms" in data
    assert "is_json_valid" in data


def test_generate_awq():
    resp = client.post("/generate", json={"utterance": "My refund has not arrived!", "system": "awq"})
    assert resp.status_code == 200
    assert resp.json()["is_json_valid"] is True


def test_metrics_endpoint():
    client.post("/generate", json={"utterance": "test", "system": "awq"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"ftbench_requests_total" in resp.content


def test_history():
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "logs" in resp.json()
