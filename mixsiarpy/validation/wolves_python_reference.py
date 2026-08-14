"""Reproducible PyMC wolves run for comparison with wolves_r_reference.R."""
from pathlib import Path
import json
import platform
import sys

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from mixsiarpy import load_discr_data, load_mix_data, load_source_data, run_model

DATA = ROOT / "mixsiarpy" / "data"
OUTPUT = ROOT / "validation" / "results" / "wolves_python"
OUTPUT.mkdir(parents=True, exist_ok=True)

mix = load_mix_data(DATA / "wolves_consumer.csv", ["d13C", "d15N"],
                    ["Region", "Pack"], [True, True], [False, True])
source = load_source_data(DATA / "wolves_sources.csv", "Region", False, "means", mix)
discr = load_discr_data(DATA / "wolves_discrimination.csv", mix)

fit = run_model(
    {"draws": 2000, "tune": 3000, "chains": 4, "thin": 1},
    mix, source, discr, random_seed=20260813, cores=1,
    target_accept=0.99, progressbar=True,
)
fit.to_netcdf(OUTPUT / "posterior.nc")

variables = [v for v in ("p_global", "p_fac1", "p_fac2", "fac1_sig",
                          "fac2_sig", "resid_prop") if v in fit.posterior]
summary = az.summary(fit, var_names=variables, hdi_prob=0.95).reset_index()
summary.rename(columns={"index": "parameter"}).to_csv(OUTPUT / "summary.csv", index=False)

divergences = int(fit.sample_stats["diverging"].sum())
report = {
    "python": platform.python_version(), "pymc": pm.__version__, "arviz": az.__version__,
    "chains": 4, "draws_per_chain": 2000, "tune": 3000, "target_accept": 0.99,
    "max_rhat": float(summary["r_hat"].max()),
    "min_ess_bulk": float(summary["ess_bulk"].min()),
    "min_ess_tail": float(summary["ess_tail"].min()),
    "divergences": divergences,
}
(OUTPUT / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
