"""
FT-Bench Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="FT-Bench Results Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── Load results ──────────────────────────────────────────────
@st.cache_data
def load_metrics():
    base_path      = "eval/results/base/metrics.json"
    finetuned_path = "eval/results/finetuned/metrics.json"
    awq_path       = "eval/results/quantized/metrics.json"
    results = {}
    for name, path in [("Base", base_path), ("Fine-Tuned", finetuned_path), ("AWQ (4-bit)", awq_path)]:
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results

results = load_metrics()

# ── Header ────────────────────────────────────────────────────
st.title("🤖 FT-Bench: QLoRA Fine-Tuning vs AWQ Quantization")
st.markdown(
    "**Llama-3.2-3B-Instruct** | ATIS NLU Dataset | 2,974 test samples"
)
st.divider()

if not results:
    st.warning("No metrics found. Run evaluations first.")
    st.stop()

# ── Key Metrics Row ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def get_metric(name, key, default=0.0):
    r = results.get(name, {})
    m = r.get("metrics", r)
    return m.get(key, default)

col1.metric("Base Exact Match",      f"{get_metric('Base','exact_match'):.2f}%")
col2.metric("Fine-Tuned Exact Match",f"{get_metric('Fine-Tuned','exact_match'):.2f}%",
            delta=f"+{get_metric('Fine-Tuned','exact_match') - get_metric('Base','exact_match'):.2f}pp")
col3.metric("AWQ Exact Match",       f"{get_metric('AWQ (4-bit)','exact_match'):.2f}%",
            delta=f"{get_metric('AWQ (4-bit)','exact_match') - get_metric('Fine-Tuned','exact_match'):.2f}pp")
col4.metric("AWQ Accuracy Loss",
            f"{abs(get_metric('AWQ (4-bit)','exact_match') - get_metric('Fine-Tuned','exact_match')):.2f}pp")

st.divider()

# ── Comparison Chart ──────────────────────────────────────────
st.subheader("📊 Head-to-Head Metric Comparison")

metrics_keys = ["intent_accuracy", "slot_f1", "exact_match", "json_valid_rate"]
metric_labels = ["Intent Accuracy", "Slot F1", "Exact Match", "JSON Valid Rate"]
systems = list(results.keys())

fig = go.Figure()
for system in systems:
    vals = [get_metric(system, k) for k in metrics_keys]
    fig.add_trace(go.Bar(name=system, x=metric_labels, y=vals))

fig.update_layout(
    barmode="group",
    yaxis_title="Score (%)",
    legend_title="System",
    height=450,
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# ── Cost/Benefit Table ────────────────────────────────────────
st.subheader("💰 Cost vs. Performance Trade-off")
cost_path = "eval/results/cost_comparison.json"
if os.path.exists(cost_path):
    with open(cost_path) as f:
        cost_data = json.load(f)
    df = pd.DataFrame([
        {"System": k, **v}
        for k, v in cost_data.get("systems", {}).items()
    ])
    st.dataframe(df, use_container_width=True)
else:
    st.info("cost_comparison.json not found — run P11 first.")

# ── Error Analysis ────────────────────────────────────────────
st.subheader("🔍 Error Analysis (4-Quadrant)")
ea_path = "eval/results/error_analysis.json"
if os.path.exists(ea_path):
    with open(ea_path) as f:
        ea = json.load(f)
    quadrant = ea.get("base_to_finetuned", ea)
    labels = list(quadrant.keys())
    vals   = [quadrant[k] if isinstance(quadrant[k], (int, float)) else quadrant[k].get("count", 0)
              for k in labels]
    fig2 = px.pie(names=labels, values=vals, title="Base → Fine-Tuned Error Shifts",
                  template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("error_analysis.json not found.")

st.divider()
st.caption("FT-Bench · github.com/najahaja/ft-bench")
