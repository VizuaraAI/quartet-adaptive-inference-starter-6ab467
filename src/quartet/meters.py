"""Latency, energy and peak-memory meters.

Energy is the objective that routing papers usually leave out, so measuring it
honestly is a large part of this project's contribution. Three sources, in
order of preference:

1. NVML (`pynvml`) sampled in a background thread, integrated over the action.
2. Intel RAPL via `/sys/class/powercap/intel-rapl` for CPU-only runs.
3. An analytic fallback: tokens x measured joules-per-token, calibrated once by
   `scripts/calibrate_energy.sh`. The fallback is flagged in every record so it
   never silently contaminates a reported number.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

RAPL_ROOT = "/sys/class/powercap/intel-rapl"

#: Joules per generated token, per model size. Replace these with your own
#: calibration numbers before reporting anything from the fallback path.
FALLBACK_J_PER_TOKEN = {"small": 0.045, "large": 0.31}


@dataclass
class Measurement:
    latency_ms: float = 0.0
    energy_j: float = 0.0
    peak_memory_mib: float = 0.0
    source: str = "none"


def _nvml_handle():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None, None


def _rapl_microjoules() -> float | None:
    if not os.path.isdir(RAPL_ROOT):
        return None
    total = 0.0
    found = False
    for name in sorted(os.listdir(RAPL_ROOT)):
        path = os.path.join(RAPL_ROOT, name, "energy_uj")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                total += float(handle.read().strip())
            found = True
        except (OSError, ValueError):
            continue
    return total if found else None


class _PowerSampler(threading.Thread):
    """Samples GPU power at a fixed interval and integrates it over time."""

    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.energy_j = 0.0
        self.peak_memory_mib = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        pynvml, handle = _nvml_handle()
        if handle is None:
            return
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            now = time.perf_counter()
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
            except Exception:
                break
            self.energy_j += watts * (now - last)
            self.peak_memory_mib = max(self.peak_memory_mib, mem)
            last = now

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1.0)


@contextmanager
def measure(enabled: bool = True):
    """Time, energy and peak memory for whatever runs inside the block.

    Usage::

        with measure() as m:
            out = model.generate(...)
        print(m.latency_ms, m.energy_j, m.source)
    """
    result = Measurement()
    sampler = None
    rapl_start = None
    torch = None

    if enabled:
        pynvml, handle = _nvml_handle()
        if handle is not None:
            sampler = _PowerSampler()
            sampler.start()
            result.source = "nvml"
        else:
            rapl_start = _rapl_microjoules()
            if rapl_start is not None:
                result.source = "rapl"
    try:
        import torch as _torch

        torch = _torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    start = time.perf_counter()
    try:
        yield result
    finally:
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        if sampler is not None:
            sampler.stop()
            result.energy_j = sampler.energy_j
            result.peak_memory_mib = sampler.peak_memory_mib
        elif rapl_start is not None:
            rapl_end = _rapl_microjoules()
            if rapl_end is not None and rapl_end >= rapl_start:
                result.energy_j = (rapl_end - rapl_start) / 1e6
        if torch is not None and torch.cuda.is_available():
            result.peak_memory_mib = max(
                result.peak_memory_mib,
                torch.cuda.max_memory_allocated() / (1024**2),
            )
        if result.source == "none":
            result.source = "fallback"


def fallback_energy(num_tokens: int, model_size: str = "small") -> float:
    """Analytic energy estimate for machines with no power counters."""
    return num_tokens * FALLBACK_J_PER_TOKEN.get(model_size, 0.045)
