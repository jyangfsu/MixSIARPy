"""Native Python tools for Bayesian tracer mixing models."""

from .data import load_discr_data, load_mix_data, load_source_data
from .model import build_model, run_model
from .results import (
    combine_sources,
    compare_models,
    continuous_effect_prediction,
    compositional_regression_prediction,
    diagnostics,
    plot_continuous_effect,
    save_diagnostics,
    summary_stat,
)
from .geometry import calc_area
from .output import save_results
from .resources import get_resource_path, list_resources
from .backends import backend_status, benchmark_summary, resolve_backend, resolve_device

__all__ = [
    "load_mix_data",
    "load_source_data",
    "load_discr_data",
    "build_model",
    "run_model",
    "summary_stat",
    "compare_models",
    "combine_sources",
    "continuous_effect_prediction",
    "compositional_regression_prediction",
    "plot_continuous_effect",
    "calc_area",
    "diagnostics",
    "save_diagnostics",
    "save_results",
    "get_resource_path",
    "list_resources",
    "backend_status",
    "benchmark_summary",
    "resolve_backend",
    "resolve_device",
]

__version__ = "0.1.0"
