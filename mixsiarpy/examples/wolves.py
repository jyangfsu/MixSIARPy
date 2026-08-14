"""Run the MixSIAR wolves example without a GUI.

This example deliberately follows the order of the original R script. Edit
the USER OPTIONS block below when running in Spyder, or override the common
options from a terminal. No shared example configuration is hidden elsewhere.
"""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from mixsiarpy import (
    build_model,
    calc_area,
    get_resource_path,
    load_discr_data,
    load_mix_data,
    load_source_data,
    run_model,
    save_results,
)
from mixsiarpy.plotting import plot_data, plot_prior


###############################################################################
# USER OPTIONS
###############################################################################

# Data files. Replace any of these paths to run the model with your own data.
DATA_DIR = get_resource_path("data")
MIX_FILENAME = DATA_DIR / "wolves_consumer.csv"
SOURCE_FILENAME = DATA_DIR / "wolves_sources.csv"
DISCR_FILENAME = DATA_DIR / "wolves_discrimination.csv"

# Results are written relative to the current Spyder/terminal working directory.
OUTPUT_DIR = Path("outputs/wolves")

# Run the sampler? Set False to load data, draw input plots and build only.
SAMPLE = True

# PyMC sampling presets: test, very short, short, normal, long, very long, extreme.
RUN = "normal"

# The three MixSIAR error structures are selected with these two switches:
#   Process × Residual (default): PROCESS_ERR=True,  RESID_ERR=True
#   Residual only:               PROCESS_ERR=False, RESID_ERR=True
#   Process only (MixSIR):       PROCESS_ERR=True,  RESID_ERR=False
PROCESS_ERR = True
RESID_ERR = True

# Dirichlet prior for source proportions. Scalar 1 is the generalist prior.
# A source-specific prior could be: ALPHA_PRIOR = [1, 1, 1]
ALPHA_PRIOR = [1, 1, 1]

# Sampling controls.
RANDOM_SEED = 42
TARGET_ACCEPT = 0.95

# Inference options for this CPU example. Other choices are "nutpie",
# "numpyro", "blackjax", or "auto". See wolves_gpu.py for explicit GPU use.
INFERENCE_BACKEND = "pymc"
COMPUTE_DEVICE = "cpu"

# Input checks and figures.
PLOT_ISOSPACE = True
PLOT_PRIOR = True
CALCULATE_POLYGON_AREA = True


###############################################################################
# COMMAND-LINE OVERRIDES
###############################################################################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", choices=(
    "test", "very short", "short", "normal", "long", "very long", "extreme"
), default=None, help="override the RUN sampling preset")
parser.add_argument("--output", type=Path, default=None,
                    help="override OUTPUT_DIR")
parser.add_argument("--build-only", action="store_true",
                    help="override SAMPLE and skip MCMC sampling")
parser.add_argument("--seed", type=int, default=None,
                    help="override RANDOM_SEED")
parser.add_argument("--backend", default=None,
                    help="auto, pymc, nutpie, numpyro, or blackjax")
parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=None)


def main(argv=None):
    """Execute the wolves workflow in the same order as the original R script."""
    args = parser.parse_args(argv)
    output_dir = (args.output or OUTPUT_DIR).expanduser().resolve()
    run = args.run or RUN
    sample = SAMPLE and not args.build_only
    random_seed = RANDOM_SEED if args.seed is None else args.seed
    inference_backend = args.backend or INFERENCE_BACKEND
    compute_device = args.device or COMPUTE_DEVICE
    output_dir.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Load mixture/consumer data
    ###########################################################################
    # Region and Pack are random effects. Pack is nested within Region.
    mix = load_mix_data(
        filename=MIX_FILENAME,
        iso_names=["d13C", "d15N"],
        factors=["Region", "Pack"],
        fac_random=[True, True],
        fac_nested=[False, True],
        cont_effects=None,
    )

    ###########################################################################
    # Load source data
    ###########################################################################
    # Source means, SDs and sample sizes vary among Regions.
    source = load_source_data(
        filename=SOURCE_FILENAME,
        source_factors="Region",
        conc_dep=False,
        data_type="means",       # choose "means" or "raw"
        mix=mix,
    )

    ###########################################################################
    # Load discrimination / trophic enrichment factor data
    ###########################################################################
    discr = load_discr_data(filename=DISCR_FILENAME, mix=mix)

    ###########################################################################
    # Inspect the isotope space
    ###########################################################################
    if PLOT_ISOSPACE:
        plot_data(mix, source, discr, output_dir / "isospace.pdf")
        plt.close("all")

    if CALCULATE_POLYGON_AREA and mix["n_iso"] == 2:
        area = calc_area(source=source, mix=mix, discr=discr)
        print(f"Normalized polygon area: {area}")

    ###########################################################################
    # Define and inspect the source-proportion prior
    ###########################################################################
    if PLOT_PRIOR:
        plot_prior(ALPHA_PRIOR, source, output_dir / "prior_distribution.png")
        plt.close("all")

    ###########################################################################
    # Build the PyMC model
    ###########################################################################
    model = build_model(
        mix=mix,
        source=source,
        discr=discr,
        alpha_prior=ALPHA_PRIOR,
        process_err=PROCESS_ERR,
        resid_err=RESID_ERR,
    )
    print(
        f"Wolves model built ({mix['N']} mixtures, "
        f"{source['n_sources']} sources, {mix['n_iso']} tracers)"
    )

    ###########################################################################
    # Run MCMC and save complete output
    ###########################################################################
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
            target_accept=TARGET_ACCEPT,
            backend=inference_backend,
            device=compute_device,
        )
        save_results(fit, output_dir, source["source_names"], mix)
        print(f"Complete results saved to: {output_dir}")
    else:
        print(f"Build-only files saved to: {output_dir}")

    # Keeping these objects makes them available in Spyder's Variable Explorer.
    return fit, model, mix, source, discr


if __name__ == "__main__":
    fit, model, mix, source, discr = main()
