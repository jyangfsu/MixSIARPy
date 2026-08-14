"""Inference-backend discovery, selection and benchmark utilities."""

from __future__ import annotations

from importlib.util import find_spec
import os
import platform
import time

import numpy as np


BACKENDS = ("pymc", "nutpie", "numpyro", "blackjax")
_DEPENDENCIES = {
    "pymc": "pymc",
    "nutpie": "nutpie",
    "numpyro": "numpyro",
    "blackjax": "blackjax",
}


def backend_status():
    """Return availability and platform notes for every supported sampler."""
    status = {}
    for name in BACKENDS:
        dependency = _DEPENDENCIES[name]
        available = find_spec(dependency) is not None
        note = "available" if available else f"install optional dependency {dependency!r}"
        if name in {"numpyro", "blackjax"} and os.name == "nt":
            note += "; JAX is generally better supported through WSL2/Linux"
        status[name] = {
            "available": available,
            "dependency": dependency,
            "note": note,
        }
    return status


def resolve_backend(backend="pymc", device="auto"):
    """Resolve ``auto`` or validate an explicitly requested NUTS backend.

    ``auto`` prefers Nutpie, then JAX samplers, then native PyMC. On Windows,
    Nutpie is preferred and JAX samplers are not automatically selected.
    Explicit requests never silently fall back to another sampler.
    """
    requested = str(backend).lower()
    if requested not in (*BACKENDS, "auto"):
        choices = ", ".join(("auto", *BACKENDS))
        raise ValueError(f"Unknown inference backend {backend!r}; choose one of {choices}")
    status = backend_status()
    requested_device = str(device).lower()
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
    if requested == "auto":
        if requested_device == "gpu":
            candidates = ("numpyro", "blackjax")
            for name in candidates:
                if status[name]["available"]:
                    return name
            raise ImportError(
                "No GPU-capable inference backend is installed. Install "
                "MixSIARPy's JAX dependencies and a CUDA-enabled JAX build."
            )
        candidates = ("nutpie", "pymc") if os.name == "nt" else (
            "nutpie", "numpyro", "blackjax", "pymc"
        )
        return next(name for name in candidates if status[name]["available"])
    if not status[requested]["available"]:
        extra = "fast" if requested == "nutpie" else "jax"
        raise ImportError(
            f"Backend {requested!r} is not installed. Install its dependency "
            f"with `pip install mixsiarpy[{extra}]` or choose backend='pymc'. "
            f"{status[requested]['note']}"
        )
    return requested


def resolve_device(backend, device="auto"):
    """Resolve CPU/GPU execution and return ``(device_name, context)``.

    GPU execution is supported by the NumPyro and BlackJAX backends. The
    returned context pins JAX computation to the selected device for sampling.
    """
    requested = str(device).lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
    if backend not in {"numpyro", "blackjax"}:
        if requested == "gpu":
            raise ValueError(
                f"device='gpu' requires backend='numpyro' or 'blackjax'; "
                f"backend={backend!r} is CPU based"
            )
        from contextlib import nullcontext
        return "cpu", nullcontext()

    try:
        import jax
    except ImportError as exc:
        raise ImportError("JAX is required for GPU-backed inference") from exc
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        # CPU-only JAX installations may raise instead of returning [].
        gpu_devices = []
    if requested == "gpu" and not gpu_devices:
        windows_note = (
            " Native Windows JAX GPU support is limited; use WSL2/Linux with "
            "a CUDA-enabled jax installation."
            if os.name == "nt" else ""
        )
        raise RuntimeError("No JAX GPU device was detected." + windows_note)
    selected = gpu_devices[0] if (requested in {"auto", "gpu"} and gpu_devices) else jax.devices("cpu")[0]
    return selected.platform, jax.default_device(selected)


def benchmark_summary(results):
    """Create a tidy performance/diagnostic table from backend fit results.

    Parameters
    ----------
    results : mapping
        Mapping of backend label to ArviZ ``InferenceData``.
    """
    import arviz as az
    import pandas as pd

    rows = []
    for label, fit in results.items():
        summary = az.summary(fit, kind="diagnostics")
        stats = fit.sample_stats
        elapsed = float(fit.attrs.get("sampling_seconds", np.nan))
        bulk = summary["ess_bulk"].min() if "ess_bulk" in summary else np.nan
        tail = summary["ess_tail"].min() if "ess_tail" in summary else np.nan
        divergences = int(stats["diverging"].sum()) if "diverging" in stats else 0
        rows.append({
            "backend": fit.attrs.get("inference_backend", label),
            "requested_backend": fit.attrs.get("requested_backend", label),
            "build_seconds": float(fit.attrs.get("build_seconds", np.nan)),
            "sampling_seconds": elapsed,
            "total_seconds": float(fit.attrs.get("total_seconds", np.nan)),
            "min_ess_bulk": float(bulk),
            "min_ess_tail": float(tail),
            "min_ess_bulk_per_second": float(bulk / elapsed) if elapsed > 0 else np.nan,
            "max_rhat": float(summary["r_hat"].max()) if "r_hat" in summary else np.nan,
            "divergences": divergences,
        })
    return pd.DataFrame(rows).set_index("backend", drop=False)


def runtime_metadata():
    """Return portable runtime metadata suitable for InferenceData attrs."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "recorded_at_unix": float(time.time()),
    }
