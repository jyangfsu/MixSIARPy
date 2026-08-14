"""Run one checkpointed MixSIARPy/PyMC CPU numerical-agreement fit."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import platform
import sys
import time
import traceback

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from mixsiarpy import load_discr_data, load_mix_data, load_source_data, run_model
from mixsiarpy.model import RUN_PRESETS
from numerical_configs import CONFIGS


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


parser = ArgumentParser()
parser.add_argument("unit", choices=sorted(CONFIGS))
parser.add_argument("--preset", default="normal", choices=RUN_PRESETS)
parser.add_argument("--draws", type=int)
parser.add_argument("--tune", type=int)
parser.add_argument("--chains", type=int)
parser.add_argument("--target-accept", type=float, default=0.99)
parser.add_argument("--round-name")
parser.add_argument("--seed", type=int, default=20260814)
parser.add_argument("--output-root", type=Path,
                    default=ROOT / "validation" / "numerical_agreement")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

cfg = dict(CONFIGS[args.unit])
round_name = args.round_name or args.preset
round_dir = args.output_root / args.unit / round_name / "python"
round_dir.mkdir(parents=True, exist_ok=True)
done = round_dir / "DONE"
failed = round_dir / "FAILED"
if done.exists() and not args.force:
    print(f"SKIP complete: {round_dir}")
    raise SystemExit(0)
failed.unlink(missing_ok=True)

sample = dict(RUN_PRESETS[args.preset])
if args.draws is not None:
    sample["draws"] = args.draws
if args.tune is not None:
    sample["tune"] = args.tune
if args.chains is not None:
    sample["chains"] = args.chains

frozen = {
    "unit": args.unit, "preset": args.preset, "model": cfg,
    "sampling": sample, "target_accept": args.target_accept,
    "seed": args.seed, "backend": "pymc", "device": "cpu",
}
(round_dir / "config.json").write_text(
    json.dumps(frozen, default=jsonable, indent=2), encoding="utf-8")

try:
    data = ROOT / "mixsiarpy" / "data"
    mix = load_mix_data(
        data / cfg["mix"], cfg["iso"], cfg.get("factors"),
        cfg.get("random"), cfg.get("nested"), cfg.get("continuous"),
    )
    source = load_source_data(
        data / cfg["source"], cfg.get("source_factor"),
        cfg.get("conc_dep", False), cfg["source_type"], mix,
    )
    discr = load_discr_data(data / cfg["discr"], mix)
    started = time.perf_counter()
    fit = run_model(
        sample, mix, source, discr, alpha_prior=cfg["alpha_prior"],
        process_err=cfg["process_err"], resid_err=cfg["resid_err"],
        random_seed=args.seed, backend="pymc", device="cpu", cores=1,
        target_accept=args.target_accept, progressbar=True,
    )
    elapsed = time.perf_counter() - started
    fit.to_netcdf(round_dir / "posterior.nc")
    # Diagnose prespecified scientific outputs and fitted hyperparameters, not
    # log-likelihood arrays, observed nodes, or sampler auxiliaries.
    prefixes = (
        "p_global", "p_fac", "p_both", "ilr_", "fac", "resid_prop",
        "Sigma", "src_mu", "src_var", "src_rho",
    )
    variables = [v for v in fit.posterior.data_vars if v.startswith(prefixes)]
    if args.unit == "alligator_length_ind" and "p_ind" in fit.posterior:
        variables.append("p_ind")
    summary = az.summary(fit, var_names=variables, hdi_prob=.95).reset_index()
    summary.rename(columns={"index": "parameter"}).to_csv(
        round_dir / "summary.csv", index=False)
    diag = az.summary(fit, var_names=variables, kind="diagnostics").reset_index()
    diag.rename(columns={"index": "parameter"}).to_csv(
        round_dir / "diagnostics.csv", index=False)
    divergences = int(fit.sample_stats["diverging"].sum())
    treedepth_hits = int(fit.sample_stats.get("reached_max_treedepth", 0).sum())
    min_bfmi = float(np.min(az.bfmi(fit)))
    usable = summary[(summary["sd"] > 0) & np.isfinite(summary["sd"])].copy()
    usable["mcse_over_sd"] = usable["mcse_mean"] / usable["sd"]
    max_rhat = float(diag["r_hat"].dropna().max())
    min_bulk = float(diag["ess_bulk"].dropna().min())
    min_tail = float(diag["ess_tail"].dropna().min())
    max_mcse_sd = float(usable["mcse_over_sd"].dropna().max())
    strict = (max_rhat <= 1.01 and min_bulk >= 400 and min_tail >= 400
              and max_mcse_sd <= 0.05 and divergences == 0
              and treedepth_hits == 0 and min_bfmi > 0.30)
    longer = (not strict and max_rhat <= 1.05 and min_bulk >= 100
              and min_tail >= 100 and divergences == 0
              and treedepth_hits == 0 and min_bfmi > 0.30)
    convergence_status = (
        "CONVERGED" if strict else
        "NEEDS_LONGER_RUN" if longer else "NOT_CONVERGED"
    )
    metadata = {
        "unit": args.unit, "python": platform.python_version(),
        "pymc": pm.__version__, "arviz": az.__version__,
        "module_path": str(Path(sys.modules["mixsiarpy"].__file__).resolve()),
        "elapsed_seconds": elapsed, "chains": sample["chains"],
        "tune": sample["tune"], "requested_draws": sample["draws"],
        "thin": sample.get("thin", 1),
        "retained_draws_per_chain": int(fit.posterior.sizes["draw"]),
        "monitored_variables": variables,
        "max_rhat": max_rhat, "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail, "max_mcse_over_sd": max_mcse_sd,
        "divergences": divergences,
        "treedepth_hits": treedepth_hits, "min_bfmi": min_bfmi,
        "min_bulk_ess_per_second": min_bulk / elapsed,
        "convergence_status": convergence_status,
    }
    (round_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    done.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for marker in ("CONVERGED", "NEEDS_LONGER_RUN", "NOT_CONVERGED"):
        (round_dir / marker).unlink(missing_ok=True)
    (round_dir / convergence_status).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
except Exception:
    report = traceback.format_exc()
    (round_dir / "traceback.txt").write_text(report, encoding="utf-8")
    failed.write_text(report, encoding="utf-8")
    print(report, file=sys.stderr)
    raise
