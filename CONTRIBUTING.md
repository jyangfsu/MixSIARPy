# Contributing to MixSIARPy

Thank you for helping improve MixSIARPy. Changes should preserve statistical transparency and make comparison with the R reference implementation straightforward.

## Development setup

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[bayes,test,dev]"
python -m pytest
```

## Pull requests

1. Keep public behavior changes focused and documented.
2. Add or update tests for loaders, model structure, error modes, and result dimensions.
3. If statistical behavior changes, state the corresponding R function or example and provide the seeds, chain settings, diagnostics, and comparison metric.
4. Do not commit posterior samples, caches, virtual environments, or generated figures.
5. Do not claim parity from non-converged chains. Report R-hat, effective sample size, divergences, and Monte Carlo uncertainty.

## Example scripts

Examples remain self-contained by design. Mirror the order and named choices of the corresponding script in `reference_r/example_scripts/`; do not move their configuration into a shared hidden helper.

## Reporting issues

Include the operating system, Python version, package versions, the smallest reproducible input, and the complete traceback. For inference problems, also include the run preset, random seed, chain count, divergences, and diagnostic summary.
