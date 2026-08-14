"""Wolves example using the normal MCMC preset.

The data and model match wolves.py; this separate script mirrors the original
R repository's longer-run example and keeps every analysis step visible.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from mixsiarpy import (
    build_model, calc_area, get_resource_path, load_discr_data, load_mix_data,
    load_source_data, run_model, save_results,
)
from mixsiarpy.plotting import plot_data, plot_prior

###############################################################################
# USER OPTIONS
###############################################################################

DATA = get_resource_path("data")
INFERENCE_BACKEND = "pymc"  # auto, pymc, nutpie, numpyro, blackjax
COMPUTE_DEVICE = "auto"     # auto, cpu, gpu
p = argparse.ArgumentParser()
p.add_argument("--sample", action="store_true")
p.add_argument("--run", default="normal")
p.add_argument("--backend", default=None)
p.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
p.add_argument("--output")
a = p.parse_args()
output = Path(a.output) if a.output else Path("outputs/wolves_normal")
output.mkdir(parents=True, exist_ok=True)
###############################################################################
# LOAD MIXTURE, REGION-SPECIFIC SOURCES, AND DISCRIMINATION DATA
###############################################################################

mix = load_mix_data(
    DATA / "wolves_consumer.csv",
    ["d13C", "d15N"],
    ["Region", "Pack"],
    [True, True],
    [False, True],
)
source = load_source_data(DATA / "wolves_sources.csv", "Region", False, "means", mix)
discr = load_discr_data(DATA / "wolves_discrimination.csv", mix)
plot_data(mix, source, discr, output / "isospace.pdf")
plot_prior(1, source, output / "prior_distribution.png")
plt.close("all")
print("Normalized polygon area:", calc_area(source, mix, discr))
###############################################################################
# BUILD, RUN THE NORMAL PRESET, AND SAVE COMPLETE RESULTS
###############################################################################

resid_err = True
process_err = True
alpha_prior = 1
build_model(mix, source, discr, alpha_prior, process_err, resid_err)
if a.sample:
    fit = run_model(
        a.run,
        mix,
        source,
        discr,
        alpha_prior=alpha_prior,
        process_err=process_err,
        resid_err=resid_err,
        random_seed=42,
        backend=a.backend or INFERENCE_BACKEND,
        device=a.device or COMPUTE_DEVICE,
        target_accept=0.95,
    )
    print(save_results(fit, output, source["source_names"], mix))
