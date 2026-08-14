# Validation status

## Current evidence

- Data-loader and model-graph tests cover the wolves, geese, alligator, and snail structures.
- A small sampling smoke test verifies that PyMC produces an ArviZ posterior with the expected source dimension.
- All distributed R example scripts have explicit Python counterparts.
- Manual comparisons have shown broadly similar global and pack-level wolves estimates in selected runs, but the available R run was severely non-converged and is not a valid parity benchmark.

These checks establish implementation progress, not complete numerical equivalence.

## Release validation matrix

Before a stable release, every official example should be evaluated for:

| Dimension | Required check |
|---|---|
| Input semantics | factor levels, nesting, source arrays, discrimination and concentration dimensions |
| Model graph | priors, ILR transform, fixed/random/continuous effects and selected error structure |
| Sampling quality | R-hat, bulk/tail ESS, divergences and Monte Carlo standard error |
| Posterior agreement | means, medians and interval overlap for named estimands |
| Predictive behavior | posterior predictive distributions and pointwise log likelihood |
| Output contract | tables, NetCDF variables, plots and metadata |

## Comparison protocol

1. Pin the R, MixSIAR, JAGS, Python, PyMC and ArviZ versions.
2. Use identical input CSV files and document all prior and error-model choices.
3. Run enough chains and iterations for both implementations to converge; rerun or reparameterize failed fits rather than treating them as reference data.
4. Export named posterior draws from both systems.
5. Compare summaries using Monte Carlo uncertainty, interval overlap, distributional distances and posterior predictive checks. Do not demand identical random draws from different samplers.
6. Archive the scripts, environment files, diagnostics and comparison tables for every example.

## Known gaps

- The complete cross-language validation matrix has not yet been executed.
- DIC from the historical R/JAGS workflow is not currently reproduced; Python provides modern ArviZ LOO/WAIC measures where supported.
- More tests are needed for all combinations of source-data modes, factor structures, concentration dependence, continuous effects and error structures.
- Performance and memory benchmarks have not yet been collected.

## Executable benchmark

The first reproducible cross-language benchmark is under `validation/`. The detected R/JAGS environment is usable, but the initial wolves reference run exceeded the 20-minute interactive execution limit and produced no accepted baseline. See `validation/README.md` for the exact offline commands and acceptance criteria.
