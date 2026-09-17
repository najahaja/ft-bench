"""FT-Bench FastAPI Gateway."""
import os
import time

from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from serving.gateway.models import GenerateRequest, GenerateResponse, HealthResponse
from serving.gateway.db import log_request, get_recent_logs
from serving.gateway.routing import route_request

app = FastAPI(
    title="FT-Bench Gateway",
    description="Production inference gateway for Open-Weight LLM fine-tuning benchmark",
    version="0.1.0",
)

REQUEST_COUNT = Counter("ftbench_requests_total", "Total inference requests", ["system", "status"])
LATENCY = Histogram("ftbench_latency_seconds", "Request latency", ["system"],
                    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
TOKENS_OUT = Counter("ftbench_tokens_generated_total", "Output tokens generated", ["system"])
ACTIVE = Gauge("ftbench_active_requests", "In-flight requests")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        available_systems=["base", "finetuned", "awq"],
        backend="vLLM" if os.environ.get("USE_VLLM") else "Mock/HF Pipeline",
    )


@app.get("/models")
async def list_models():
    return {
        "systems": {
            "base": {"description": "Zero-shot prompted base model (Llama 3.2 3B Instruct)"},
            "finetuned": {"description": "QLoRA fine-tuned specialist (BF16 merged)"},
            "awq": {"description": "Fine-tuned + AWQ 4-bit quantized (PagedAttention via vLLM)"},
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    ACTIVE.inc()
    t0 = time.perf_counter()
    try:
        result = await route_request(
            ticket_text=req.ticket_text,
            system=req.system,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        REQUEST_COUNT.labels(system=req.system, status="200").inc()
        LATENCY.labels(system=req.system).observe(elapsed)
        TOKENS_OUT.labels(system=req.system).inc(result["tokens_generated"])

        log_request(
            system=result["system"], ticket_text=req.ticket_text,
            raw_output=result["raw_output"], is_json_valid=result["is_json_valid"],
            is_schema_valid=result["is_schema_valid"], latency_ms=result["latency_ms"],
            tokens_generated=result["tokens_generated"], throughput_tok_s=result["throughput_tok_s"],
            parse_error=result.get("parse_error"),
        )
        return GenerateResponse(**{k: v for k, v in result.items() if k != "parse_error"})
    finally:
        ACTIVE.dec()


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/history")
async def history(limit: int = 50):
    return {"logs": get_recent_logs(limit=limit)}
