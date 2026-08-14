"""Mantis-shrimp example with alternative priors and combined sources.

The alternative biological priors are written out explicitly, following the
original R analysis, rather than hidden in a helper function.
"""

from pathlib import Path
import argparse

import numpy as np, matplotlib.pyplot as plt
from mixsiarpy import (
    build_model, combine_sources, get_resource_path, load_discr_data,
    load_mix_data, load_source_data, run_model, save_results,
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
output = Path(a.output) if a.output else Path("outputs/mantis")
output.mkdir(parents=True, exist_ok=True)
###############################################################################
# LOAD MIXTURE, FACTOR-SPECIFIC SOURCES AND DISCRIMINATION DATA
###############################################################################

mix = load_mix_data(
    DATA / "mantis_consumer.csv", ["d13C", "d15N"], ["Habitat"], [False], [False]
)
source = load_source_data(DATA / "mantis_source.csv", "Habitat", True, "means", mix)
discr = load_discr_data(DATA / "mantis_discrimination.csv", mix)
plot_data(mix, source, discr, output / "isospace.pdf")
plt.close("all")
###############################################################################
# DEFINE AND PLOT THE FOUR PRIORS USED IN THE R EXAMPLE
###############################################################################

alpha_unif = np.ones(source["n_sources"])
alpha_spec = np.array([1, 1, 4, 4, 1, 4], float)
alpha_spec *= len(alpha_spec) / alpha_spec.sum()
alpha_grass = np.array([0.35, 1.61, 0.43, 51.65 + 0.26, 5.18, 40.5]) * 6 / 100
alpha_coral = (
    np.array([14.31 + 24.74, 0.01, 15.48, 13.81 + 4.71, 8.44, 18.51]) * 6 / 100
)
for name, alpha in (
    ("uninformative", alpha_unif),
    ("specialist", alpha_spec),
    ("seagrass", alpha_grass),
    ("coral", alpha_coral),
):
    plot_prior(alpha, source, output / f"prior_{name}.png")
    plt.close("all")
###############################################################################
# FIT THE SELECTED PRIOR MODEL AND COMBINE ECOLOGICALLY SIMILAR SOURCES
###############################################################################

alpha_prior = alpha_spec
resid_err = True
process_err = True
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
    save_results(fit, output, source["source_names"], mix)
    combined = combine_sources(
        fit,
        {
            "hard": ["clam", "crab", "snail"],
            "soft": ["alphworm", "brittlestar", "fish"],
        },
        source["source_names"],
    )
    combined.to_netcdf(output / "combined_sources.nc")
