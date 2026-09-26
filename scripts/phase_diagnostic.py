#!/usr/bin/env python3
"""
FT-Bench Phase Diagnostic
==========================
Run anywhere to see which phases are done / in-progress / pending.

  In Kaggle  : paste into a cell (after cloning repo + pip install -e .)
  In WSL     : python3 scripts/phase_diagnostic.py   (from llm-platform/)

Checks real artifacts on disk + HF Hub. The checklist file can be stale;
files and HTTP responses cannot.
"""

import json, os, sys, subprocess
from pathlib import Path

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN   = "\033[96m"; RESET  = "\033[0m";  BOLD = "\033[1m"

def jsonl_lines(p):
    p = Path(p)
    return sum(1 for _ in open(p)) if p.exists() else 0

def load_json(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def hf_exists(model_id, token=None):
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{model_id}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False

def git_log(n=5):
    try:
        r = subprocess.run(["git","log",f"-{n}","--oneline"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip().splitlines()
    except Exception:
        return []

# Try to get HF token from env or Kaggle secrets
HF_TOKEN = os.environ.get("HF_TOKEN","")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

results = []

def phase(number, name, checks):
    passed = sum(1 for c,_ in checks if c)
    total  = len(checks)
    if passed == total:
        status = "DONE"
        badge  = f"{GREEN}OK  Phase {number} -- {name}{RESET}"
    elif passed == 0:
        status = "PENDING"
        badge  = f"{RED}X   Phase {number} -- {name}{RESET}"
    else:
        status = "IN-PROGRESS"
        badge  = f"{YELLOW}>>  Phase {number} -- {name}  ({passed}/{total}){RESET}"
    lines = [badge]
    for cond, ev in checks:
        mark = "[OK]" if cond else "[X] "
        lines.append(f"{CYAN}    -> {mark} {ev}{RESET}")
    results.append((number, name, status, passed, total))
    return "\n".join(lines)


print(f"\n{BOLD}{'='*65}")
print("  FT-BENCH PHASE DIAGNOSTIC")
print(f"{'='*65}{RESET}\n")

# ── Phase 0 ──
docs_failures_size = Path("docs/failures.md").stat().st_size if Path("docs/failures.md").exists() else 0
print(phase(0, "Design & Documentation", [
    (Path("docs/00-design.md").exists(),             "docs/00-design.md"),
    (Path("docs/01-experiment-design.md").exists(),  "docs/01-experiment-design.md"),
    (Path("docs/decisions.md").exists(),             "docs/decisions.md"),
    (docs_failures_size > 500,
     "docs/failures.md (size=" + str(docs_failures_size) + " bytes)"),
]))

# ── Phase 1 ──
try:
    import ftbench; ftbench_ok = True
except ImportError:
    ftbench_ok = False

try:
    r = subprocess.run(
        [sys.executable,"-m","pytest","tests/","-q","--tb=no","--no-header"],
        capture_output=True, text=True, timeout=60)
    pytest_ok  = "passed" in r.stdout and "failed" not in r.stdout
    pytest_sum = (r.stdout.strip().splitlines() or ["no output"])[-1]
except Exception as e:
    pytest_ok = False; pytest_sum = str(e)

print(phase(1, "Repo Scaffold + Core Package", [
    (ftbench_ok,              "ftbench importable: " + str(ftbench_ok)),
    (Path("ftbench").is_dir(),"ftbench/ directory exists"),
    (pytest_ok,               "pytest: " + pytest_sum),
    (Path(".gitignore").exists(), ".gitignore exists"),
]))

# ── Phase 2 ──
train_n = jsonl_lines("data/train.jsonl")
val_n   = jsonl_lines("data/val.jsonl")
test_n  = jsonl_lines("data/test.jsonl")
vocab   = load_json("configs/label_vocab.json") or {}
n_int   = len(vocab.get("intents",[]))
n_sl    = len(vocab.get("slots",[]))

print(phase(2, "Dataset Pipeline (MASSIVE)", [
    (train_n == 11481, "data/train.jsonl: " + str(train_n) + " (expected 11481)"),
    (val_n   == 2033,  "data/val.jsonl:   " + str(val_n)   + " (expected 2033)"),
    (test_n  == 2974,  "data/test.jsonl:  " + str(test_n)  + " (expected 2974)"),
    (n_int   == 60,    "intents: " + str(n_int) + " (expected 60)"),
    (n_sl    == 55,    "slots:   " + str(n_sl)  + " (expected 55)"),
]))

# ── Phase 3 ──
print(phase(3, "Serving Gateway Skeleton", [
    (Path("serving/gateway/main.py").exists(),    "serving/gateway/main.py"),
    (Path("serving/gateway/routing.py").exists(), "serving/gateway/routing.py"),
    (Path("serving/gateway/db.py").exists(),      "serving/gateway/db.py"),
    (pytest_ok, "pytest: " + pytest_sum),
]))

# ── Phase 4 ──
base_m   = load_json("eval/results/base/metrics.json") or {}
base_n   = base_m.get("n_samples", 0)
base_rec = jsonl_lines("eval/results/base/records.jsonl")
base_ci  = "intent_accuracy_95ci" in base_m
base_em  = base_m.get("metrics", {}).get("exact_match", -1)
base_acc = base_m.get("metrics", {}).get("intent_accuracy", "?")

print(phase(4, "Zero-Shot Baseline, System A", [
    (base_n == 2974,  "metrics.json n_samples=" + str(base_n) + " (expected 2974)"),
    (base_ci,         "intent_accuracy_95ci present  acc=" + str(base_acc) + "%"),
    (base_em >= 0,    "exact_match=" + str(base_em) + "%"),
    (base_rec == 2974,"records.jsonl " + str(base_rec) + " lines (need 2974 for error analysis)"),
]))

# ── Phase 5 ──
adapter_hub = hf_exists("najahaja/ftbench-qlora-llama3.2-3b-adapter", HF_TOKEN)
merged_hub  = hf_exists("najahaja/ftbench-qlora-llama3.2-3b",         HF_TOKEN)
phase5_md   = Path("results/phase5_qlora_results.md").exists()

print(phase(5, "QLoRA Fine-Tuning, System B", [
    (adapter_hub, "HF Hub adapter: najahaja/ftbench-qlora-llama3.2-3b-adapter  -> " + str(adapter_hub)),
    (merged_hub,  "HF Hub merged:  najahaja/ftbench-qlora-llama3.2-3b  -> " + str(merged_hub)),
    (phase5_md,   "results/phase5_qlora_results.md"),
    (Path("docs/failures.md").exists(), "docs/failures.md (crash history)"),
]))

# ── Phase 6 ──
ft_m    = load_json("eval/results/finetuned/metrics.json") or {}
ft_n    = ft_m.get("n_samples", 0)
ft_rec  = jsonl_lines("eval/results/finetuned/records.jsonl")
ft_em   = ft_m.get("metrics", {}).get("exact_match", -1)
ft_ci   = "intent_accuracy_95ci" in ft_m
err_f   = Path("eval/results/error_analysis.json").exists()

print(phase(6, "Fine-Tuned Evaluation, System B", [
    (ft_n == 2974, "metrics.json n_samples=" + str(ft_n) + " (expected 2974)"),
    (ft_em > 50,   "exact_match=" + str(ft_em) + "% (expected ~70%, must be >> 2.56% base)"),
    (ft_ci,        "Bootstrap CIs present: " + str(ft_ci)),
    (ft_rec == 2974,"records.jsonl " + str(ft_rec) + " lines (need 2974 for error analysis)"),
    (err_f,        "eval/results/error_analysis.json (4-quadrant): " + str(err_f)),
]))

# ── Phase 7 ──
awq_hub  = hf_exists("najahaja/ftbench-qlora-llama3.2-3b-awq", HF_TOKEN)
vram_f   = Path("eval/results/quantized/vram_stats.json").exists()

print(phase(7, "AWQ Quantization, System C", [
    (awq_hub,  "HF Hub AWQ: najahaja/ftbench-qlora-llama3.2-3b-awq  -> " + str(awq_hub)),
    (vram_f,   "eval/results/quantized/vram_stats.json (FP16 vs INT4): " + str(vram_f)),
]))

# ── Phase 8 ──
awq_m      = load_json("eval/results/quantized/metrics.json") or {}
awq_n      = awq_m.get("n_samples", 0)
awq_rec    = jsonl_lines("eval/results/quantized/records.jsonl")
awq_ci     = "exact_match_95ci" in awq_m
awq_em     = awq_m.get("metrics", {}).get("exact_match", -1)
win_cond   = load_json("eval/results/quantized/win_condition.json")
smoke_only = 0 < awq_n < 100

smoke_note = "  [SMOKE TEST ONLY -- need full 2974-sample run]" if smoke_only else ""
rec_note   = "  [SMOKE TEST -- need full run]" if 0 < awq_rec < 100 else ""
wc_val     = win_cond.get("decision", "MISSING") if win_cond else "MISSING"

print(phase(8, "Quantized Evaluation, System C", [
    (awq_n == 2974,  "metrics.json n_samples=" + str(awq_n) + smoke_note),
    (awq_rec == 2974,"records.jsonl " + str(awq_rec) + " lines" + rec_note),
    (awq_ci,         "exact_match_95ci present: " + str(awq_ci)),
    (awq_em > 30,    "exact_match=" + str(awq_em) + "% (should be close to FT ~70%)"),
    (win_cond is not None, "win_condition.json (sec8 decision): " + wc_val),
    (err_f,          "eval/results/error_analysis.json (4-quadrant): " + str(err_f)),
]))

# ── Phase 9 ──
vllm_files = list(Path("serving/vllm").glob("*.py")) if Path("serving/vllm").exists() else []
bench_env  = Path("eval/results/benchmark_env.json").exists()

print(phase(9, "vLLM Serving (all 3 systems)", [
    (len(vllm_files) > 0, "serving/vllm/*.py: " + str(len(vllm_files)) + " file(s)"),
    (bench_env,           "eval/results/benchmark_env.json: " + str(bench_env)),
]))

# ── Phase 10 ──
bench_jsons  = list(Path("eval/results").glob("benchmark*.json")) if Path("eval/results").exists() else []
locust_csvs  = list(Path("load_testing/reports").glob("*.csv")) if Path("load_testing/reports").exists() else []

print(phase(10, "Benchmark: Latency / Throughput / GPU", [
    (len(bench_jsons) > 0,  "eval/results/benchmark*.json: " + str(len(bench_jsons)) + " file(s)"),
    (len(locust_csvs) > 0,  "load_testing/reports/*.csv: "  + str(len(locust_csvs))  + " file(s)"),
    (bench_env,             "benchmark_env.json (GPU/vLLM snapshot): " + str(bench_env)),
]))

# ── Phase 11 ──
cost_f = Path("eval/results/cost_comparison.json").exists()
print(phase(11, "Cost Analysis", [
    (cost_f, "eval/results/cost_comparison.json: " + str(cost_f)),
]))

# ── Phase 12 ──
prom_cfgs   = list(Path("monitoring/prometheus").glob("*.yml"))
graf_dashs  = (list(Path("monitoring/grafana/dashboards").glob("*.json"))
               if Path("monitoring/grafana/dashboards").exists() else [])

print(phase(12, "Monitoring (Prometheus + Grafana)", [
    (len(prom_cfgs) > 0,  "monitoring/prometheus/*.yml: "         + str(len(prom_cfgs))  + " file(s)"),
    (len(graf_dashs) > 0, "monitoring/grafana/dashboards/*.json: " + str(len(graf_dashs)) + " file(s)"),
]))

# ── Phase 13 ──
dash_pys = list(Path("dashboard").glob("*.py")) if Path("dashboard").exists() else []
print(phase(13, "Streamlit Dashboard", [
    (len(dash_pys) > 0, "dashboard/*.py: " + str(len(dash_pys)) + " file(s)"),
]))

# ── Phase 14 ──
dc_exists = Path("docker-compose.yml").exists()
df_count  = len(list(Path(".").glob("Dockerfile*")))
print(phase(14, "Docker", [
    (dc_exists,    "docker-compose.yml: " + str(dc_exists)),
    (df_count > 0, "Dockerfile*: "       + str(df_count) + " file(s)"),
]))

# ── Phase 15 ──
ci_ymls = list(Path(".github/workflows").glob("*.yml")) if Path(".github/workflows").exists() else []
print(phase(15, "CI/CD (GitHub Actions)", [
    (len(ci_ymls) > 0, ".github/workflows/*.yml: " + str([f.name for f in ci_ymls])),
]))

# ── Phase 16 ──
readme = Path("README.md")
readme_has_results = False
if readme.exists():
    c = readme.read_text()
    readme_has_results = "exact_match" in c.lower() or "| system" in c.lower()

print(phase(16, "Documentation & Final Results", [
    (readme.exists(),        "README.md exists"),
    (readme_has_results,     "README.md contains results table: " + str(readme_has_results)),
    (Path("docs/decisions.md").exists(), "docs/decisions.md"),
    (Path("docs/failures.md").exists(),  "docs/failures.md"),
]))

# ── Phase 17 ──
interview_docs = (
    list(Path("docs").glob("*interview*")) +
    list(Path("docs").glob("*resume*"))
) if Path("docs").exists() else []

print(phase(17, "Interview Prep & Resume Bullets", [
    (len(interview_docs) > 0, "docs/interview* or docs/resume*: " + str(len(interview_docs)) + " file(s)"),
    (awq_n == 2974, "Real AWQ metrics exist (required before writing bullets)"),
]))

# ── Summary table ──
print(f"\n{BOLD}{'='*65}")
print("  SUMMARY")
print(f"{'='*65}{RESET}")
print(f"{'Phase':<8} {'Name':<40} {'Status':<13} Checks")
print("-"*65)

for num, name, status, passed, total in results:
    if   status == "DONE":        badge = f"{GREEN}DONE       {RESET}"
    elif status == "IN-PROGRESS": badge = f"{YELLOW}IN-PROGRESS{RESET}"
    else:                         badge = f"{RED}PENDING    {RESET}"
    print(f"  P{num:<6} {name:<40} {badge}  {passed}/{total}")

print()
first_nd = next((num for num,_,st,_,_ in results if st != "DONE"), None)
if first_nd is not None:
    nm = next(n for p,n,_,_,_ in results if p == first_nd)
    print(f"{BOLD}{CYAN}You are in: Phase {first_nd} -- {nm}{RESET}")
else:
    print(f"{BOLD}{GREEN}All phases complete!{RESET}")

print(f"\n{BOLD}Recent git commits:{RESET}")
for line in git_log(5):
    print("  " + line)
print(f"\n{BOLD}{'='*65}{RESET}\n")
