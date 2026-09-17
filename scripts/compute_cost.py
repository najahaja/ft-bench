#!/usr/bin/env python3
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from ftbench.common.io import read_json


def cost_per_1k_self_hosted(cost_per_hour, latency_p50_ms):
    req_per_second = 1000.0 / latency_p50_ms
    req_per_hour = req_per_second * 3600
    return (cost_per_hour / req_per_hour) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cost.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["cost"]

    gpu_rate = cfg["gpu_cost_per_hour_usd"]
    results = []
    for system, result_dir in [("base", "eval/results/base"),
                                ("finetuned", "eval/results/finetuned"),
                                ("quantized", "eval/results/quantized")]:
        metrics_path = os.path.join(result_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            results.append({"system": system, "note": "No results"})
            continue
        m = read_json(metrics_path)["metrics"]
        p50 = m.get("latency_p50_ms", 50.0)
        tok_s = m.get("throughput_tok_s", 0.0)
        cost_1k = cost_per_1k_self_hosted(gpu_rate, p50)
        results.append({
            "system": system, "latency_p50_ms": p50,
            "throughput_tok_s": tok_s,
            "cost_per_1k_req_usd": round(cost_1k, 4),
        })

    for api_name, api_cfg in cfg.get("api_costs", {}).items():
        cost_1k = ((250 * api_cfg["input_per_million_tokens"]) +
                   (cfg["avg_output_tokens"] * api_cfg["output_per_million_tokens"])) / 1000
        results.append({
            "system": api_name, "latency_p50_ms": "network",
            "cost_per_1k_req_usd": round(cost_1k, 4),
        })

    print("\n" + "=" * 75 + "\n  FT-BENCH COST COMPARISON\n" + "=" * 75)
    print(f"{'System':<20} | {'p50 Lat (ms)':<14} | {'$/1k Req':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['system']:<20} | {str(r.get('latency_p50_ms', 'N/A')):<14} | ${r.get('cost_per_1k_req_usd', 'N/A')}")
    print("=" * 75)

    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/cost_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Cost table saved to eval/results/cost_comparison.json")

if __name__ == "__main__":
    main()
