"""Killer-whale example comparing uninformative and fecal-data priors.

This is intentionally a linear, R-style analysis script.  It keeps both prior
choices visible so a user can see exactly which biological information enters
each model.
"""

from pathlib import Path
import argparse

import numpy as np, matplotlib.pyplot as plt
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
p.add_argument("--run", default="test")
p.add_argument("--backend", default=None)
p.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
p.add_argument("--informative", action="store_true")
p.add_argument("--output")
a = p.parse_args()
output = Path(a.output) if a.output else Path("outputs/killerwhale")
output.mkdir(parents=True, exist_ok=True)
###############################################################################
# LOAD MIXTURE, SOURCE AND DISCRIMINATION DATA
###############################################################################

mix = load_mix_data(DATA / "killerwhale_consumer.csv", ["d13C", "d15N"])
source = load_source_data(DATA / "killerwhale_sources.csv", None, False, "means", mix)
discr = load_discr_data(DATA / "killerwhale_discrimination.csv", mix)
plot_data(mix, source, discr, output / "isospace.pdf")
plt.close("all")
print("Normalized polygon area:", calc_area(source, mix, discr))
###############################################################################
# DEFINE THE ALTERNATIVE DIRICHLET PRIORS
###############################################################################

# Construct informative alpha from 14 fecal samples: 10, 1, 0, 0 and 3.
# Rescaling preserves the observed relative frequencies while making the sum
# of alpha equal the number of sources, as in the original R example.
kw_alpha = np.array([10, 1, 0, 0, 3], float)
kw_alpha *= len(kw_alpha) / kw_alpha.sum()
kw_alpha[kw_alpha == 0] = 0.01
alpha_prior = kw_alpha if a.informative else 1
plot_prior(
    alpha_prior,
    source,
    output / ("prior_informative.png" if a.informative else "prior_uninformative.png"),
)
plt.close("all")
###############################################################################
# BUILD, SAMPLE AND SAVE
###############################################################################

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
