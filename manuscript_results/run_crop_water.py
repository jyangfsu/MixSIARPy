"""Manuscript run for the crop-water generalized compositional regression."""
from pathlib import Path
import json
import platform
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from mixsiarpy import (
    compositional_regression_prediction, load_discr_data, load_mix_data,
    load_source_data, run_model,
)

DATA = ROOT / "mixsiarpy" / "data"
OUT = ROOT / "manuscript_results" / "crop_water"
OUT.mkdir(parents=True, exist_ok=True)

formula = "species * biomass + diversity"
mix = load_mix_data(
    DATA / "crop_water_consumer.csv", ["delta2H", "delta18O"],
    factors=None, fac_random=None, fac_nested=None,
    cont_effects=["biomass"], composition_formula=formula,
)
source = load_source_data(
    DATA / "crop_water_sources.csv", None, False, "raw", mix,
)
discr = load_discr_data(DATA / "crop_water_discrimination.csv", mix)

fit = run_model(
    {"draws": 2000, "tune": 3000, "chains": 4, "thin": 1},
    mix, source, discr, random_seed=20260813, cores=1,
    target_accept=0.99, backend="pymc", device="cpu",
)
fit.to_netcdf(OUT / "posterior.nc")

variables = [v for v in ("p_global", "comp_beta", "resid_prop")
             if v in fit.posterior]
summary = az.summary(fit, var_names=variables, hdi_prob=0.95).reset_index()
summary.rename(columns={"index": "parameter"}).to_csv(OUT / "summary.csv", index=False)

observed = mix["data"]
biomass_values = np.quantile(observed["biomass"], [0.1, 0.5, 0.9])
rows = pd.DataFrame([
    (species, biomass, diversity)
    for species in sorted(observed["species"].unique())
    for biomass in biomass_values
    for diversity in sorted(observed["diversity"].unique())
], columns=["species", "biomass", "diversity"])
prediction = compositional_regression_prediction(
    fit, mix, rows, source["source_names"]
)
prediction.to_netcdf(OUT / "predictions.nc")
q = prediction.quantile([0.025, 0.5, 0.975], dim=("chain", "draw"))
records = []
for i, row in rows.iterrows():
    for source_name in source["source_names"]:
        v = q.sel(prediction=i, source=source_name)
        records.append({
            **row.to_dict(), "source": source_name,
            "q2.5": float(v.sel(quantile=0.025)),
            "median": float(v.sel(quantile=0.5)),
            "q97.5": float(v.sel(quantile=0.975)),
        })
pd.DataFrame(records).to_csv(OUT / "prediction_summary.csv", index=False)

diagnostics = {
    "python": platform.python_version(), "pymc": pm.__version__,
    "arviz": az.__version__, "n_plants": int(mix["N"]),
    "n_sources": int(source["n_sources"]), "formula": formula,
    "max_rhat": float(summary["r_hat"].max()),
    "min_ess_bulk": float(summary["ess_bulk"].min()),
    "min_ess_tail": float(summary["ess_tail"].min()),
    "divergences": int(fit.sample_stats["diverging"].sum()),
    "sampling_seconds": float(fit.attrs.get("sampling_seconds", np.nan)),
}
(OUT / "metadata.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
print(json.dumps(diagnostics, indent=2))
