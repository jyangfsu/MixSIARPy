"""Alligator example with Length and Individual effects.

This linear script exposes the continuous Length slope and ID random effect
in the same order as the original R example.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from mixsiarpy import (
    build_model, continuous_effect_prediction, get_resource_path,
    load_discr_data, load_mix_data, load_source_data, plot_continuous_effect,
    run_model, save_results,
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
p.add_argument("--run", default="short")
p.add_argument("--backend", default=None)
p.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
p.add_argument("--output")
a = p.parse_args()
output = Path(a.output) if a.output else Path("outputs/alligator_length_ind")
output.mkdir(parents=True, exist_ok=True)
###############################################################################
# LOAD DATA: ID IS RANDOM; LENGTH IS A STANDARDIZED CONTINUOUS EFFECT
###############################################################################

mix = load_mix_data(
    DATA / "alligator_consumer.csv",
    ["d13C", "d15N"],
    ["ID"],
    [True],
    [False],
    ["Length"],
)
source = load_source_data(
    DATA / "alligator_sources_simplemean.csv", None, False, "means", mix
)
discr = load_discr_data(DATA / "alligator_TEF.csv", mix)
plot_data(mix, source, discr, output / "isospace.pdf")
plot_prior(1, source, output / "prior_distribution.png")
plt.close("all")
###############################################################################
# BUILD THE PROCESS-ERROR MODEL, SAMPLE, AND PLOT THE LENGTH RESPONSE
###############################################################################

resid_err = False
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
    save_results(fit, output, source["source_names"], mix)
    prediction = continuous_effect_prediction(fit, mix, source["source_names"])
    plot_continuous_effect(prediction, output / "continuous_effect.png")
