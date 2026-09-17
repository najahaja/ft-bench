"""GPU memory helpers via pynvml (gracefully degrades when unavailable)."""
from typing import Optional, Dict, Any


def get_gpu_stats(device_index: int = 0) -> Optional[Dict[str, Any]]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "vram_used_mb": mem.used // (1024 ** 2),
            "vram_total_mb": mem.total // (1024 ** 2),
            "vram_util_pct": round(mem.used / mem.total * 100, 1),
            "gpu_util_pct": util.gpu,
        }
    except Exception:
        return None


def peak_vram_mb(device_index: int = 0) -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
    except Exception:
        pass
    return None
