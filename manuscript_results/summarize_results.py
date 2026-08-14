"""Create manuscript tables only from completed, auditable experiment files."""
from pathlib import Path
import json
import subprocess
import sys

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).parents[1]
OUT = ROOT / "manuscript_results"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def recover_r_diagnostics():
    directory = ROOT / "validation" / "results" / "wolves_r"
    draws = pd.read_csv(directory / "posterior_draws.csv")
    if len(draws) % 3:
        raise ValueError("R posterior rows cannot be divided into the three chains")
    n = len(draws) // 3
    array = draws.to_numpy().reshape(3, n, draws.shape[1])
    ds = xr.Dataset({name: (("chain", "draw"), array[:, :, i])
                     for i, name in enumerate(draws.columns)})
    summary = az.summary(ds, kind="diagnostics").reset_index().rename(
        columns={"index": "parameter"})
    summary.to_csv(directory / "diagnostics_recovered.csv", index=False)
    return summary


def main():
    r_diag = recover_r_diagnostics()
    rows = [{
        "analysis": "wolves_R_MixSIAR_JAGS",
        "chains": 3,
        "draws_per_chain": 1000,
        "max_rhat": r_diag["r_hat"].max(),
        "min_ess_bulk": r_diag["ess_bulk"].min(),
        "min_ess_tail": r_diag["ess_tail"].min(),
        "divergences": "not applicable",
    }]
    py_meta = ROOT / "validation" / "results" / "wolves_python" / "metadata.json"
    if py_meta.exists():
        d = json.loads(py_meta.read_text())
        rows.append({"analysis": "wolves_Python_PyMC", **{
            k: d[k] for k in ("chains", "max_rhat", "min_ess_bulk",
                              "min_ess_tail", "divergences")},
            "draws_per_chain": d["draws_per_chain"],
        })
    crop_meta = OUT / "crop_water" / "metadata.json"
    if crop_meta.exists():
        d = json.loads(crop_meta.read_text())
        rows.append({
            "analysis": "crop_water_Python_PyMC", "chains": 4,
            "draws_per_chain": 2000, "max_rhat": d["max_rhat"],
            "min_ess_bulk": d["min_ess_bulk"],
            "min_ess_tail": d["min_ess_tail"],
            "divergences": d["divergences"],
        })
    pd.DataFrame(rows).to_csv(TABLES / "convergence.csv", index=False)

    r_summary = ROOT / "validation" / "results" / "wolves_r" / "summary.csv"
    py_summary = ROOT / "validation" / "results" / "wolves_python" / "summary.csv"
    if r_summary.exists() and py_summary.exists():
        # Execute the package comparison script after both summaries exist.
        subprocess.run(
            [sys.executable, str(ROOT / "mixsiarpy" / "validation" / "compare_wolves.py")],
            cwd=ROOT, check=True,
        )
        comparison = pd.read_csv(
            ROOT / "validation" / "results" / "wolves_comparison.csv")
        metrics = pd.DataFrame([{
            "matched_parameters": len(comparison),
            "median_abs_standardized_mean_difference": comparison[
                "standardized_mean_difference"].abs().median(),
            "max_abs_standardized_mean_difference": comparison[
                "standardized_mean_difference"].abs().max(),
            "parameters_with_interval_overlap": int((comparison["interval_overlap"] > 0).sum()),
            "interval_overlap_percent": 100 * (comparison["interval_overlap"] > 0).mean(),
        }])
        metrics.to_csv(TABLES / "wolves_agreement_metrics.csv", index=False)


if __name__ == "__main__":
    main()
