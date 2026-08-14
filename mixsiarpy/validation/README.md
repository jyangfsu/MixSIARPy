# Cross-language validation

These scripts run long, converged comparisons outside the fast test suite. Generated results are written to `validation/results/` and are intentionally ignored by Git.

## Wolves benchmark

From the repository root:

```powershell
& "C:\Program Files\R\R-4.4.1\bin\Rscript.exe" mixsiarpy\validation\wolves_r_reference.R
python mixsiarpy\validation\wolves_python_reference.py
python mixsiarpy\validation\compare_wolves.py
```

The reference environment detected on 13 August 2026 was R 4.4.1, MixSIAR 3.1.12 and JAGS 4.3.1. The R configuration uses three 100,000-iteration chains, discards 50,000 iterations and thins by 50. On the current machine it did not finish within a 20-minute interactive execution limit; this is an incomplete run, not a convergence failure.

Do not run the Python benchmark or comparison until the R script finishes and `validation/results/wolves_r/gelman.csv` exists. Inspect convergence before comparing posteriors. A benchmark is acceptable only if all important proportions and scale parameters have R-hat below 1.05, effective sample sizes are adequate, and the Python fit has zero divergences. Increase iterations or improve parameterization when these conditions are not met.

The scripts are validation infrastructure rather than proof of equivalence. Extend the comparison to the remaining official examples after the wolves benchmark passes.
