# R MixSIAR to MixSIARPy model map

This document maps the frozen MixSIAR 3.1.12 reference implementation to the native PyMC model. Names use dots in R/JAGS and underscores in Python/ArviZ.

| R/JAGS | Python/PyMC | Prior or construction | Status |
|---|---|---|---|
| `p.global` | `p_global` | Dirichlet(`alpha`) | implemented |
| `ilr.global` | `ilr_global` | Egozcue ILR of `p_global` | implemented |
| `ilr.fac1`, `ilr.fac2` | `ilr_fac1`, `ilr_fac2` | fixed: reference level 0 and N(0,1); random: N(0, factor SD) | implemented |
| `fac1.sig`, `fac2.sig` | `fac1_sig`, `fac2_sig` | Uniform(0,20) | implemented |
| `p.fac1`, `p.fac2` | `p_fac1`, `p_fac2` | inverse ILR, including nested parent offset | implemented |
| post-processed `p.both` | `p_both` | inverse ILR of global + both factor effects | implemented and exported |
| `ilr.cont1` | `ilr_cont1` | N(0, precision .001), equivalent SD sqrt(1000) | implemented |
| `p.ind` | `p_ind` | inverse ILR for every observation | implemented |
| `src_mu`, `src_tau` | `src_mu`, `src_var` / source precision nodes | summary or raw hierarchical source model | implemented; parameterization differs internally |
| `resid.prop` | `resid_prop` | Uniform(0,20) multiplicative scale | implemented |
| `Sigma` | `Sigma` | Wishart(I, J+1) precision or Gamma(.001,.001) for one tracer | implemented via stable parameterization |
| `mix.mu` | `mix_mean` | concentration-weighted expected mixture | implemented |
| `loglik` | `log_likelihood.mix` | pointwise mixture log likelihood | implemented after sampling |

## Intentional differences

- JAGS is replaced by PyMC/NUTS; random draws and sampler diagnostics are therefore not expected to be identical.
- R reports DIC. MixSIARPy uses ArviZ PSIS-LOO and WAIC and does not currently reproduce DIC.
- The multivariate raw-source correlation prior uses `LKJCorr(eta=1)`. For two tracers this matches a uniform correlation prior; for more than two tracers it guarantees a valid correlation matrix and is not algebraically identical to independent pairwise uniform correlations.
- MixSIARPy stores additional deterministic variables and labeled xarray dimensions. Extra variables do not alter the likelihood.
- For two fixed effects or one fixed plus one random effect, R constructs `p.both` during output processing and only for observed factor combinations. MixSIARPy calculates the full Cartesian array in the model; downstream comparisons must mask combinations absent from the mixture data.
