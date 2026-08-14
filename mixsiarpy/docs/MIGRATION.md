# Migration notes

The supported implementation lives in `mixsiarpy/`. Early Python prototypes have been removed from the distributable repository; `reference_r/` is a frozen copy of MixSIAR 3.1.12 used to compare behavior.

The Python implementation uses PyMC as its inference backend and returns an ArviZ `InferenceData` object. It does not generate or execute a JAGS model. Consequently, exact sample-by-sample equality is neither expected nor a useful validation criterion. Validation should instead compare model definitions, posterior estimands, predictive behavior, and Monte Carlo-compatible posterior summaries under converged runs.

Example programs intentionally keep their complete configuration local to each file. This mirrors the R scripts and reduces the learning cost for users translating an analysis.
