"""Posterior contrasts for defensible interpretation of crop-water predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).parents[1]
OUT = ROOT / "manuscript_results" / "crop_water"
pred = xr.open_dataarray(OUT / "predictions.nc")
rows = pd.read_csv(OUT / "prediction_summary.csv")[
    ["species", "biomass", "diversity"]].drop_duplicates().reset_index(drop=True)

records = []
for species in sorted(rows.species.unique()):
    for diversity in sorted(rows.diversity.unique()):
        sel = rows[(rows.species == species) & (rows.diversity == diversity)]
        lo_i = int(sel.biomass.idxmin())
        hi_i = int(sel.biomass.idxmax())
        delta = pred.sel(prediction=hi_i) - pred.sel(prediction=lo_i)
        for source in pred.source.values:
            x = delta.sel(source=source).values.ravel()
            records.append({
                "contrast": "high_minus_low_biomass", "species": species,
                "diversity": diversity, "source": str(source),
                "mean": float(x.mean()), "q2.5": float(np.quantile(x, .025)),
                "median": float(np.median(x)), "q97.5": float(np.quantile(x, .975)),
                "probability_positive": float((x > 0).mean()),
            })

for species in sorted(rows.species.unique()):
    # Compare mixture vs monoculture at the median biomass grid point.
    for label in ("mixture", "monoculture"):
        subset = rows[(rows.species == species) & (rows.diversity == label)]
        target = subset.iloc[(subset.biomass - subset.biomass.median()).abs().argmin()]
        if label == "mixture":
            mix_i = int(target.name)
        else:
            mono_i = int(target.name)
    delta = pred.sel(prediction=mix_i) - pred.sel(prediction=mono_i)
    for source in pred.source.values:
        x = delta.sel(source=source).values.ravel()
        records.append({
            "contrast": "mixture_minus_monoculture_at_median_biomass",
            "species": species, "diversity": "mixture-monoculture",
            "source": str(source), "mean": float(x.mean()),
            "q2.5": float(np.quantile(x, .025)), "median": float(np.median(x)),
            "q97.5": float(np.quantile(x, .975)),
            "probability_positive": float((x > 0).mean()),
        })

pd.DataFrame(records).to_csv(OUT / "posterior_contrasts.csv", index=False)
