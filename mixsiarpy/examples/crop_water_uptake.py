"""Generalized compositional regression of crop water uptake.

Schmutz & Schoeb (2023) measured delta2H and delta18O in crop xylem water
and soil water.  The three sources are shallow (0-10 cm), middle (15-30 cm)
and deep (50-75 cm) soil water.  Unlike the original two-stage analysis, this
example estimates source proportions and their relationships with species,
plant biomass and planting diversity in one Bayesian model.

Data: https://doi.org/10.5281/zenodo.7505603 (CC BY 4.0).
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mixsiarpy import (
    build_model,
    compositional_regression_prediction,
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

DATA_DIR = get_resource_path("data")
MIX_FILENAME = DATA_DIR / "crop_water_consumer.csv"
SOURCE_FILENAME = DATA_DIR / "crop_water_sources.csv"
DISCR_FILENAME = DATA_DIR / "crop_water_discrimination.csv"
OUTPUT_DIR = Path("outputs/crop_water_uptake")

SAMPLE = True
RUN = "test"
RANDOM_SEED = 42
TARGET_ACCEPT = 0.95

# Inference engine and compute device. GPU requires NumPyro/BlackJAX and CUDA JAX.
INFERENCE_BACKEND = "pymc"  # auto, pymc, nutpie, numpyro, blackjax
COMPUTE_DEVICE = "auto"     # auto, cpu, gpu
PROCESS_ERR = True
RESID_ERR = True
ALPHA_PRIOR = 1

# Generalized ILR composition model.  ``species * biomass`` expands to
# species + biomass + species:biomass.  Traditional MixSIAR permits only one
# continuous effect and cannot fit this interaction together with diversity.
COMPOSITION_FORMULA = "species * biomass + diversity"


###############################################################################
# OPTIONAL COMMAND-LINE OVERRIDES (also safe in Spyder)
###############################################################################

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
# 1. LOAD PLANT (MIXTURE) DATA
###############################################################################

mix = load_mix_data(
    filename=MIX_FILENAME,
    iso_names=["delta2H", "delta18O"],
    factors=None,
    fac_random=None,
    fac_nested=None,
    cont_effects=["biomass"],
    composition_formula=COMPOSITION_FORMULA,
)


###############################################################################
# 2. LOAD SOIL-WATER SOURCES
###############################################################################

source = load_source_data(
    filename=SOURCE_FILENAME,
    source_factors=None,
    conc_dep=False,
    data_type="raw",
    mix=mix,
)


###############################################################################
# 3. WATER UPTAKE HAS NO TROPHIC DISCRIMINATION
###############################################################################

discr = load_discr_data(filename=DISCR_FILENAME, mix=mix)


###############################################################################
# 4. INSPECT DATA AND PRIOR
###############################################################################

plot_data(mix, source, discr, output_dir / "isospace.pdf")
plt.close("all")
plot_prior(ALPHA_PRIOR, source, output_dir / "prior_distribution.png")
plt.close("all")


###############################################################################
# 5. BUILD AND SAMPLE THE GENERALIZED COMPOSITIONAL REGRESSION
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
    f"crop_water_uptake: model built ({mix['N']} plants, "
    f"{source['n_sources']} soil-water sources, formula: {COMPOSITION_FORMULA})"
)

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

    # Predict a biomass gradient for every crop in mixture plots.  These are
    # joint posterior predictions, not a regression fitted to estimated means.
    observed = mix["data"]
    biomass_grid = np.linspace(observed.biomass.min(), observed.biomass.max(), 60)
    prediction_rows = pd.DataFrame(
        [(species, biomass, "mixture")
         for species in sorted(observed.species.unique())
         for biomass in biomass_grid],
        columns=["species", "biomass", "diversity"],
    )
    prediction = compositional_regression_prediction(
        fit, mix, prediction_rows, source["source_names"]
    )
    prediction.to_netcdf(output_dir / "compositional_regression_predictions.nc")

    quantiles = prediction.quantile(
        [0.025, 0.5, 0.975], dim=("chain", "draw")
    )
    records = []
    for index, row in prediction_rows.iterrows():
        for source_name in source["source_names"]:
            values = quantiles.sel(prediction=index, source=source_name)
            records.append({
                **row.to_dict(), "source": source_name,
                "q2.5": float(values.sel(quantile=0.025)),
                "median": float(values.sel(quantile=0.5)),
                "q97.5": float(values.sel(quantile=0.975)),
            })
    prediction_table = pd.DataFrame(records)
    prediction_table.to_csv(
        output_dir / "compositional_regression_predictions.csv", index=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    colors = ["#2A9D8F", "#E9C46A", "#457B9D"]
    for ax, species in zip(axes.ravel(), sorted(observed.species.unique())):
        subset = prediction_table[prediction_table.species == species]
        for color, source_name in zip(colors, source["source_names"]):
            line = subset[subset.source == source_name]
            ax.plot(line.biomass, line["median"], color=color, label=source_name)
            ax.fill_between(
                line.biomass, line["q2.5"], line["q97.5"],
                color=color, alpha=0.18,
            )
        ax.set_title(species)
        ax.grid(color="#E8E8E8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Plant biomass")
    axes[1, 1].set_xlabel("Plant biomass")
    axes[1, 2].set_xlabel("Plant biomass")
    axes[0, 0].set_ylabel("Water-source proportion")
    axes[1, 0].set_ylabel("Water-source proportion")
    axes[0, 2].legend(frameon=False, loc="upper right")
    fig.suptitle("Predicted soil-water uptake in crop mixtures", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "water_uptake_by_species_and_biomass.png", dpi=300)
    plt.close(fig)
    print(f"Complete results saved to: {output_dir}")
else:
    print("Build-only mode: sampling was skipped.")

# In Spyder, mix/source/discr/model/fit remain in Variable Explorer.
