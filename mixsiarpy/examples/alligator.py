"""Alligator eight-model comparison, following the official R example.

Every candidate formula is listed below.  The loop only repeats the same
visible load/build/sample steps; no model specification is hidden elsewhere.
"""

from pathlib import Path
import argparse

from mixsiarpy import (
    build_model, compare_models, get_resource_path, load_discr_data,
    load_mix_data, load_source_data, run_model, save_results,
)

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
a = p.parse_args()
output = Path("outputs/alligator")
output.mkdir(parents=True, exist_ok=True)
###############################################################################
# CANDIDATE EFFECT STRUCTURES FROM THE ORIGINAL R ANALYSIS
###############################################################################

specifications = [
    ("null", None, None, None, None),
    ("habitat", ["habitat"], [False], [False], None),
    ("sex", ["sex"], [False], [False], None),
    ("size_class", ["sclass"], [False], [False], None),
    ("length", None, None, None, ["Length"]),
    ("sex_size", ["sex", "sclass"], [False, False], [False, False], None),
    ("sex_length", ["sex"], [False], [False], ["Length"]),
    ("sex_size_combined", ["sex_sclass"], [False], [False], None),
]
fits = []
names = []
###############################################################################
# LOAD, BUILD AND OPTIONALLY SAMPLE EACH CANDIDATE MODEL
###############################################################################

for name, factors, random, nested, continuous in specifications:
    print(f"\n--- {name} ---")
    model_output = output / name
    model_output.mkdir(parents=True, exist_ok=True)
    mix = load_mix_data(
        DATA / "alligator_consumer.csv",
        ["d13C", "d15N"],
        factors,
        random,
        nested,
        continuous,
    )
    source = load_source_data(
        DATA / "alligator_sources_simplemean.csv", None, False, "means", mix
    )
    discr = load_discr_data(DATA / "alligator_TEF.csv", mix)
    process_err = True
    resid_err = True
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
        save_results(fit, model_output, source["source_names"], mix)
        fits.append(fit)
        names.append(name)
###############################################################################
# COMPARE MODELS BY PSIS-LOO AFTER ALL MODELS HAVE BEEN SAMPLED
###############################################################################

if a.sample:
    compare_models(fits, names).to_csv(output / "model_comparison_loo.csv")
