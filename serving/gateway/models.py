from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    utterance: str = Field(..., description="Raw utterance for NLU parsing")
    system: Literal["base", "finetuned", "awq"] = Field(default="awq")
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_new_tokens: int = Field(256, ge=16, le=1024)


class GenerateResponse(BaseModel):
    system: str
    raw_output: str
    parsed: Optional[Dict[str, Any]]
    is_json_valid: bool
    is_schema_valid: bool
    latency_ms: float
    tokens_generated: int
    throughput_tok_s: float


class HealthResponse(BaseModel):
    status: str
    available_systems: list
    backend: str
