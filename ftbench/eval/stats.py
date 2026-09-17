"""Bootstrap confidence intervals for evaluation metrics."""
import random
from typing import Any, Dict, List, Tuple
import numpy as np
from ftbench.eval.metrics import compute_metrics


def bootstrap_ci(
    records: List[Dict[str, Any]],
    metric_key: str,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    point = compute_metrics(records).get(metric_key, 0.0)
    samples = []
    n = len(records)
    for _ in range(n_bootstrap):
        resample = [rng.choice(records) for _ in range(n)]
        val = compute_metrics(resample).get(metric_key, 0.0)
        samples.append(val)
    lower_p = (1 - ci) / 2 * 100
    upper_p = (1 + ci) / 2 * 100
    return (
        round(point, 3),
        round(float(np.percentile(samples, lower_p)), 3),
        round(float(np.percentile(samples, upper_p)), 3),
    )
