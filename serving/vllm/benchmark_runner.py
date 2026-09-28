"""
Latency and throughput benchmarking script for all 3 ft-bench vLLM servers.

Measures:
  - Latency: p50, p95, p99 per system
  - Throughput: tokens/sec per system
  - GPU utilization snapshot

Usage:
  python serving/vllm/benchmark_runner.py --host localhost --systems base finetuned awq
"""

import time
import json
import argparse
import statistics
import requests
from pathlib import Path


def benchmark_system(host: str, port: int, prompts: list, n_warmup: int = 5) -> dict:
    """Run latency benchmark against a running vLLM server."""
    url = f"http://{host}:{port}/generate"

    # Warmup
    for p in prompts[:n_warmup]:
        requests.post(url, json={"prompt": p, "max_tokens": 128})

    latencies_ms = []
    total_tokens = 0
    t_start = time.time()

    for p in prompts:
        t0 = time.perf_counter()
        resp = requests.post(url, json={"prompt": p, "max_tokens": 128})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        output = resp.json().get("output", "")
        total_tokens += len(output.split())

    t_total = time.time() - t_start
    latencies_ms.sort()
    n = len(latencies_ms)

    return {
        "n_requests":      n,
        "total_time_s":    round(t_total, 2),
        "throughput_rps":  round(n / t_total, 2),
        "throughput_tok_s": round(total_tokens / t_total, 2),
        "latency_p50_ms":  round(latencies_ms[int(n * 0.50)], 1),
        "latency_p95_ms":  round(latencies_ms[int(n * 0.95)], 1),
        "latency_p99_ms":  round(latencies_ms[int(n * 0.99)], 1),
        "latency_mean_ms": round(statistics.mean(latencies_ms), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--systems", nargs="+", default=["base", "finetuned", "awq"])
    parser.add_argument("--test-prompts", default="data/test.jsonl")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--output", default="eval/results/benchmark_latency.json")
    args = parser.parse_args()

    from ftbench.common.io import read_jsonl
    samples = read_jsonl(args.test_prompts)[:args.n_prompts]
    prompts = [s["prompt"] for s in samples]

    PORTS = {"base": 8001, "finetuned": 8002, "awq": 8003}
    results = {}

    for system in args.systems:
        print(f"[+] Benchmarking {system} on port {PORTS[system]}...")
        results[system] = benchmark_system(args.host, PORTS[system], prompts)
        print(f"    p50={results[system]['latency_p50_ms']}ms  "
              f"p95={results[system]['latency_p95_ms']}ms  "
              f"tps={results[system]['throughput_tok_s']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Results saved to {args.output}")
