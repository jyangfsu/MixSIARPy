"""Palmyra example with raw sources and a Taxa fixed effect.

The order and comments follow the original MixSIAR R example.  Edit only the
USER OPTIONS section for an ordinary Spyder run.  The workflow is deliberately
linear: no main function and no hidden common-example configuration.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from mixsiarpy import (
    build_model,
    calc_area,
    continuous_effect_prediction,
    get_resource_path,
    load_discr_data,
    load_mix_data,
    load_source_data,
    plot_continuous_effect,
    run_model,
    save_results,
)
from mixsiarpy.plotting import plot_data, plot_prior


###############################################################################
# USER OPTIONS
###############################################################################

# Installed example data. Replace these paths with your own CSV files.
DATA_DIR = get_resource_path("data")
MIX_FILENAME = DATA_DIR / 'palmyra_consumer.csv'
SOURCE_FILENAME = DATA_DIR / 'palmyra_sources.csv'
DISCR_FILENAME = DATA_DIR / 'palmyra_discrimination.csv'

# Results are saved relative to the current Spyder/terminal working directory.
OUTPUT_DIR = Path("outputs/palmyra")

# Set False when you only want to inspect data, figures and the model graph.
SAMPLE = True
RUN = "test"  # test, very short, short, normal, long, very long, extreme
RANDOM_SEED = 42
TARGET_ACCEPT = 0.95

# Inference engine and compute device. GPU requires NumPyro/BlackJAX and CUDA JAX.
INFERENCE_BACKEND = "pymc"  # auto, pymc, nutpie, numpyro, blackjax
COMPUTE_DEVICE = "auto"     # auto, cpu, gpu

# Error model switches:
# Process x Residual: True, True; Residual only: False, True;
# Process only (MixSIR): True, False.
PROCESS_ERR = True
RESID_ERR = True

# Dirichlet prior on source proportions. Scalar 1 gives alpha=1 per source.
ALPHA_PRIOR = 1

PLOT_ISOSPACE = True
PLOT_PRIOR = True
CALCULATE_POLYGON_AREA = True


###############################################################################
# OPTIONAL COMMAND-LINE OVERRIDES
###############################################################################

# parse_known_args keeps Spyder's own kernel arguments from causing errors.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--build-only", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--backend", default=None)
parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
args, _unknown = parser.parse_known_args()

output_dir = (args.output or OUTPUT_DIR).expanduser().resolve()
run = args.run or RUN
sample = SAMPLE and not args.build_only
random_seed = RANDOM_SEED if args.seed is None else args.seed
inference_backend = args.backend or INFERENCE_BACKEND
compute_device = args.device or COMPUTE_DEVICE
output_dir.mkdir(parents=True, exist_ok=True)


###############################################################################
# 1. LOAD MIXTURE / CONSUMER DATA
###############################################################################

# factors contains categorical fixed/random effects. fac_random=False means a
# fixed effect; True means a hierarchical random effect. cont_effects is fitted
# as a linear slope in ILR space after standardization.
mix = load_mix_data(
    filename=MIX_FILENAME,
    iso_names=['d13C', 'd15N'],
    factors=['Taxa'],
    fac_random=[False],
    fac_nested=[False],
    cont_effects=None,
)


###############################################################################
# 2. LOAD SOURCE DATA
###############################################################################

# data_type="raw" uses replicate measurements. data_type="means" expects
# Mean<tracer>, SD<tracer> and n columns. conc_dep=True uses Conc<tracer>.
source = load_source_data(
    filename=SOURCE_FILENAME,
    source_factors=None,
    conc_dep=False,
    data_type='raw',
    mix=mix,
)


###############################################################################
# 3. LOAD DISCRIMINATION / TROPHIC ENRICHMENT DATA
###############################################################################

discr = load_discr_data(filename=DISCR_FILENAME, mix=mix)


###############################################################################
# 4. INSPECT INPUT DATA AND PRIOR
###############################################################################

if PLOT_ISOSPACE:
    plot_data(mix, source, discr, output_dir / "isospace.pdf")
    plt.close("all")

if CALCULATE_POLYGON_AREA and mix["n_iso"] == 2:
    area = calc_area(source=source, mix=mix, discr=discr)
    print(f"Normalized polygon area: {area}")

if PLOT_PRIOR:
    plot_prior(ALPHA_PRIOR, source, output_dir / "prior_distribution.png")
    plt.close("all")


###############################################################################
# 5. BUILD THE PYMC MODEL
###############################################################################

model = build_model(
    mix=mix,
    source=source,
    discr=discr,
    alpha_prior=ALPHA_PRIOR,
    process_err=PROCESS_ERR,
    resid_err=RESID_ERR,
)
print(
    f"palmyra: model built ({mix['N']} mixtures, "
    f"{source['n_sources']} sources, {mix['n_iso']} tracers)"
)


###############################################################################
# 6. SAMPLE AND SAVE COMPLETE RESULTS
###############################################################################

fit = None
if sample:
    fit = run_model(
        run=run,
        mix=mix,
        source=source,
        discr=discr,
        alpha_prior=ALPHA_PRIOR,
        process_err=PROCESS_ERR,
        resid_err=RESID_ERR,
        random_seed=random_seed,
        backend=inference_backend,
        device=compute_device,
        target_accept=TARGET_ACCEPT,
    )
    save_results(fit, output_dir, source["source_names"], mix)

    print(f"Complete results saved to: {output_dir}")
else:
    print("Build-only mode: sampling was skipped.")

# In Spyder, mix/source/discr/model/fit remain in Variable Explorer.
