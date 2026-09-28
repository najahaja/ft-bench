"""
vLLM serving entry point for all 3 ft-bench systems.

Serves:
  System A: meta-llama/Llama-3.2-3B-Instruct       (FP16, zero-shot)
  System B: najahaja/ftbench-qlora-llama3.2-3b      (FP16, fine-tuned)
  System C: najahaja/ftbench-qlora-llama3.2-3b-awq  (INT4 AWQ)

Usage:
  python serving/vllm/server.py --system base     --port 8001
  python serving/vllm/server.py --system finetuned --port 8002
  python serving/vllm/server.py --system awq      --port 8003
"""

import argparse
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

SYSTEM_MODEL_MAP = {
    "base":      "meta-llama/Llama-3.2-3B-Instruct",
    "finetuned": "najahaja/ftbench-qlora-llama3.2-3b",
    "awq":       "najahaja/ftbench-qlora-llama3.2-3b-awq",
}

QUANTIZATION_MAP = {
    "base":      None,
    "finetuned": None,
    "awq":       "awq",
}


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


app = FastAPI(title="ft-bench vLLM Server")
llm = None
sampling_params_default = None


@app.on_event("startup")
def load_model():
    global llm, sampling_params_default
    model_id = app.state.model_id
    quantization = app.state.quantization
    llm = LLM(
        model=model_id,
        quantization=quantization,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params_default = SamplingParams(temperature=0.0, max_tokens=128)
    print(f"[+] Model loaded: {model_id}")


@app.post("/generate")
def generate(req: GenerateRequest):
    params = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
    outputs = llm.generate([req.prompt], params)
    return {"output": outputs[0].outputs[0].text}


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.model_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["base", "finetuned", "awq"], required=True)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app.state.model_id = SYSTEM_MODEL_MAP[args.system]
    app.state.quantization = QUANTIZATION_MAP[args.system]

    uvicorn.run(app, host="0.0.0.0", port=args.port)
