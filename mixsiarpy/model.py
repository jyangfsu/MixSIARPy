"""Native PyMC implementation of the MixSIAR likelihood."""

import numpy as np
import os
import copy
import warnings
import time


def _import_pymc():
    # ArviZ <0.18 imports this historical SciPy alias; keep older environments usable.
    import scipy.signal as signal

    if not hasattr(signal, "gaussian"):
        signal.gaussian = signal.windows.gaussian
    import pymc as pm
    import pytensor.tensor as pt

    return pm, pt


RUN_PRESETS = {
    "test": dict(draws=200, tune=200, thin=1, chains=2),
    "very short": dict(draws=1000, tune=1000, thin=1, chains=3),
    "short": dict(draws=2000, tune=2000, thin=1, chains=3),
    "normal": dict(draws=5000, tune=5000, thin=1, chains=3),
    "long": dict(draws=10000, tune=10000, thin=2, chains=4),
    "very long": dict(draws=20000, tune=20000, thin=5, chains=4),
    "extreme": dict(draws=50000, tune=50000, thin=10, chains=4),
}


def _source_parameters(source):
    if source["data_type"] == "means":
        return np.asarray(source["MU_array"], float), np.asarray(
            source["SIG2_array"], float
        )
    arr = np.asarray(source["SOURCE_array"], float)
    return np.nanmean(arr, axis=-1), np.nanvar(arr, axis=-1, ddof=1)


def _ilr_basis(k):
    """Egozcue sequential binary partition used by MixSIAR."""
    basis = np.zeros((k, k - 1))
    for col in range(k - 1):
        r = col + 1
        basis[:r, col] = 1 / np.sqrt(r * (r + 1))
        basis[r, col] = -r / np.sqrt(r * (r + 1))
    return basis


def _normalized_inputs(mix, source, discr):
    """Apply run_model.R's pooled tracer normalization without mutating inputs."""
    mix, source, discr = copy.deepcopy(mix), copy.deepcopy(source), copy.deepcopy(discr)
    x = mix["data_iso"]
    for iso in range(mix["n_iso"]):
        if source["data_type"] == "raw":
            vals = source["SOURCE_array"][:, iso, ...]
            pooled = np.r_[x[:, iso], vals[np.isfinite(vals)]]
            center, scale = pooled.mean(), pooled.std(ddof=1)
            source["SOURCE_array"][:, iso, ...] = (vals - center) / scale
        else:
            mu, var, nn = source["MU_array"], source["SIG2_array"], source["n_array"]
            means = mu[:, iso, ...].ravel()
            variances = var[:, iso, ...].ravel()
            counts = nn.ravel()
            total = counts.sum() + len(x)
            center = (np.sum(counts * means) + np.sum(x[:, iso])) / total
            ss = np.sum((counts - 1) * variances + counts * means**2)
            ss += np.sum(x[:, iso] ** 2) - total * center**2
            scale = np.sqrt(ss / (total - 1))
            source["MU_array"][:, iso, ...] = (mu[:, iso, ...] - center) / scale
            source["SIG2_array"][:, iso, ...] = var[:, iso, ...] / scale**2
        mix["data_iso"][:, iso] = (x[:, iso] - center) / scale
        discr["mu"].iloc[:, iso] = discr["mu"].iloc[:, iso] / scale
        discr["sig2"].iloc[:, iso] = discr["sig2"].iloc[:, iso] / scale**2
    return mix, source, discr


def _fit_sources(pm, pt, source, mix, coords):
    """Fit MixSIAR's hierarchical raw or mean/SD/n source model."""
    k, j = source["n_sources"], mix["n_iso"]
    factor_n = source["S_factor_levels"] if source["by_factor"] is not None else None
    src_dims = (
        ("source", "isotope", "source_factor") if factor_n else ("source", "isotope")
    )
    if source["data_type"] == "means":
        observed_mu = np.asarray(source["MU_array"], float)
        observed_var = np.asarray(source["SIG2_array"], float)
        sample_n = np.asarray(source["n_array"], float)
        n_for_iso = sample_n[:, None, :] if factor_n else sample_n[:, None]
        src_mu = pm.Normal(
            "src_mu",
            mu=observed_mu,
            sigma=pt.sqrt(observed_var / n_for_iso),
            dims=src_dims,
        )
        # Exact Ward et al. / MixSIAR construction:
        # tmp.X ~ chi-square(n); variance = s2*(n-1)/tmp.X.
        tmp_x = pm.ChiSquared("src_tmp_x", nu=n_for_iso, shape=observed_mu.shape)
        src_var = pm.Deterministic(
            "src_var", observed_var * (n_for_iso - 1) / tmp_x, dims=src_dims
        )
        return src_mu, src_var, None

    raw = np.asarray(source["SOURCE_array"], float)
    # R priors are on normalized data. Here the raw source likelihood is fully
    # Bayesian, with the same diffuse mean and Gamma(.001,.001) precision priors.
    src_mu = pm.Normal("src_mu", 0, sigma=np.sqrt(1000), dims=src_dims)
    tau = pm.Gamma("src_tau_diag", alpha=0.001, beta=0.001, dims=src_dims)
    src_sd = pt.sqrt(1 / tau)
    if j == 1:
        for si in range(k):
            for fi in range(factor_n or 1):
                values = raw[si, 0, fi, :] if factor_n else raw[si, 0, :]
                values = values[np.isfinite(values)]
                index = (si, 0, fi) if factor_n else (si, 0)
                pm.Normal(
                    f"source_obs_{si}_{fi}",
                    src_mu[index],
                    src_sd[index],
                    observed=values,
                )
        return src_mu, 1 / tau, None

    # LKJ(eta=1) is uniform over valid correlation matrices. For two tracers it
    # is exactly MixSIAR's rho~Uniform(-1,1), while guaranteeing positive-definite covariance.
    covariances = []
    for si in range(k):
        cov_by_factor = []
        for fi in range(factor_n or 1):
            sd_here = src_sd[si, :, fi] if factor_n else src_sd[si, :]
            packed = pm.LKJCorr(f"src_rho_{si}_{fi}", n=j, eta=1)
            corr = pt.eye(j)
            tri = np.triu_indices(j, 1)
            corr = pt.set_subtensor(corr[tri], packed)
            corr = pt.set_subtensor(corr[(tri[1], tri[0])], packed)
            cov = sd_here[:, None] * corr * sd_here[None, :]
            values = raw[si, :, fi, :].T if factor_n else raw[si, :, :].T
            values = values[np.all(np.isfinite(values), axis=1)]
            pm.MvNormal(
                f"source_obs_{si}_{fi}",
                mu=src_mu[si, :, fi] if factor_n else src_mu[si, :],
                cov=cov,
                observed=values,
            )
            cov_by_factor.append(cov)
        covariances.append(pt.stack(cov_by_factor) if factor_n else cov_by_factor[0])
    src_cov = pt.stack(covariances)
    src_var = pm.Deterministic(
        "src_var",
        (
            pt.diagonal(src_cov, axis1=-2, axis2=-1).dimshuffle(0, 2, 1)
            if factor_n
            else pt.diagonal(src_cov, axis1=-2, axis2=-1)
        ),
        dims=src_dims,
    )
    return src_mu, src_var, src_cov


def _wishart_identity_bartlett(pm, pt, name, dim, nu):
    """Sampling-stable Bartlett decomposition of Wishart(I, nu)."""
    diagonal = []
    for i in range(dim):
        chi = pm.ChiSquared(f"{name}_chi_{i+1}", nu=nu - i)
        diagonal.append(pt.sqrt(chi))
    lower = pt.diag(pt.stack(diagonal))
    for i in range(1, dim):
        for col in range(i):
            z = pm.Normal(f"{name}_z_{i+1}_{col+1}", 0, 1)
            lower = pt.set_subtensor(lower[i, col], z)
    return pm.Deterministic(name, lower @ lower.T)


def build_model(mix, source, discr, alpha_prior=1, process_err=True, resid_err=True):
    """Build a PyMC model covering MixSIAR error and covariate structures."""
    if not process_err and not resid_err:
        raise ValueError("At least one of process_err and resid_err must be true")
    if mix["N"] == 1 and resid_err:
        raise ValueError("A single mixture observation requires process-only error")
    try:
        pm, pt = _import_pymc()
    except ImportError as exc:
        raise ImportError(
            "Install mixsiarpy[bayes] to build and sample models"
        ) from exc
    mix, source, discr = _normalized_inputs(mix, source, discr)
    k, j, n = source["n_sources"], mix["n_iso"], mix["N"]
    alpha = (
        np.ones(k)
        if np.isscalar(alpha_prior) and alpha_prior == 1
        else np.asarray(alpha_prior, float)
    )
    if alpha.shape != (k,) or np.any(alpha <= 0):
        raise ValueError(f"alpha_prior must contain {k} positive values")
    dmu, dvar = discr["mu"].to_numpy(float), discr["sig2"].to_numpy(float)
    if dmu.shape != (k, j):
        raise ValueError("Source names/count do not match discrimination data")
    conc = (
        np.ones((k, j)) if source["conc"] is None else np.asarray(source["conc"], float)
    )
    coords = {
        "obs": np.arange(n),
        "source": source["source_names"],
        "isotope": mix["iso_names"],
        "ilr": np.arange(k - 1),
    }
    if mix.get("regression") is not None:
        coords["regression_term"] = mix["regression"]["columns"]
    for idx, fac in enumerate(mix["FAC"], 1):
        coords[f"fac{idx}_level"] = fac["labels"]
    if source["S_factor_levels"] is not None:
        coords["source_factor"] = np.arange(source["S_factor_levels"])
    with pm.Model(coords=coords) as model:
        src_mu, src_var, src_cov = _fit_sources(pm, pt, source, mix, coords)
        if source["by_factor"] is not None:
            fac_idx = mix["FAC"][source["by_factor"] - 1]["codes"]
            mu_obs = src_mu.dimshuffle(2, 0, 1)[fac_idx]
            var_obs = src_var.dimshuffle(2, 0, 1)[fac_idx]
            cov_obs = (
                src_cov.dimshuffle(1, 0, 2, 3)[fac_idx] if src_cov is not None else None
            )
        else:
            mu_obs = pt.broadcast_to(src_mu, (n, k, j))
            var_obs = pt.broadcast_to(src_var, (n, k, j))
            cov_obs = (
                pt.broadcast_to(src_cov, (n, k, j, j)) if src_cov is not None else None
            )
        p_global = pm.Dirichlet("p_global", a=alpha, dims="source")
        basis = pt.as_tensor_variable(_ilr_basis(k))
        ilr_global = pm.Deterministic(
            "ilr_global", basis.T @ pt.log(p_global), dims="ilr"
        )
        ilr_total = pt.broadcast_to(ilr_global, (n, k - 1))
        factor_effects = []
        for idx, fac in enumerate(mix["FAC"]):
            if fac["re"]:
                sigma = pm.Uniform(f"fac{idx+1}_sig", 0, 20)
                effect = pm.Normal(
                    f"ilr_fac{idx+1}", 0, sigma, dims=(f"fac{idx+1}_level", "ilr")
                )
            else:
                free = pm.Normal(
                    f"ilr_fac{idx+1}_free",
                    0,
                    1,
                    shape=(max(fac["levels"] - 1, 0), k - 1),
                )
                effect = pt.concatenate([pt.zeros((1, k - 1)), free], axis=0)
                pm.Deterministic(
                    f"ilr_fac{idx+1}", effect, dims=(f"fac{idx+1}_level", "ilr")
                )
            factor_effects.append(effect)
            ilr_total = ilr_total + effect[fac["codes"]]
        if mix.get("regression") is not None:
            design = np.asarray(mix["regression"]["matrix"], float)
            beta = pm.Normal(
                "ilr_beta", 0, sigma=2.5, dims=("regression_term", "ilr")
            )
            ilr_total = ilr_total + design @ beta
        elif mix["n_ce"]:
            beta = pm.Normal("ilr_cont1", 0, sigma=np.sqrt(1000), dims="ilr")
            ilr_total = ilr_total + np.asarray(mix["CE"][0])[:, None] * beta
        p_ind = pm.Deterministic(
            "p_ind",
            pm.math.softmax(ilr_total @ basis.T, axis=-1),
            dims=("obs", "source"),
        )
        # MixSIAR factor-level proportions, including the parent offset for nesting.
        for idx, (fac, effect) in enumerate(zip(mix["FAC"], factor_effects)):
            level_ilr = ilr_global + effect
            if (
                len(mix["FAC"]) == 2
                and mix["fac_nested"][idx]
                and fac["lookup"] is not None
            ):
                parent = factor_effects[1 - idx]
                level_ilr = level_ilr + parent[np.asarray(fac["lookup"]) - 1]
            pm.Deterministic(
                f"p_fac{idx+1}",
                pm.math.softmax(level_ilr @ basis.T, axis=-1),
                dims=(f"fac{idx+1}_level", "source"),
            )
        if len(factor_effects) == 2:
            both_ilr = (
                ilr_global[None, None, :]
                + factor_effects[0][:, None, :]
                + factor_effects[1][None, :, :]
            )
            pm.Deterministic(
                "p_both",
                pm.math.softmax(both_ilr @ basis.T, axis=-1),
                dims=("fac1_level", "fac2_level", "source"),
            )
        # Concentration dependence modifies each tracer's mixture mean, but the
        # original MixSIAR/JAGS process-error model defines p2 from the dietary
        # proportions themselves: p2[i, k] = p.ind[i, k]^2.  Do not reuse the
        # concentration-adjusted mean weights in the process covariance.
        weighted = p_ind[:, :, None] * conc[None, :, :]
        mean_weights = weighted / weighted.sum(axis=1, keepdims=True)
        expected = pm.Deterministic(
            "mix_mean",
            (mean_weights * (mu_obs + dmu)).sum(axis=1),
            dims=("obs", "isotope"),
        )
        p2 = p_ind**2
        if process_err and source["data_type"] == "raw" and j > 1:
            tdf_cov = np.zeros((k, j, j))
            idx = np.arange(j)
            tdf_cov[:, idx, idx] = dvar
            process_cov = (p2[:, :, None, None] * (cov_obs + tdf_cov[None, :, :, :])).sum(
                axis=1
            )
        elif process_err:
            process_var = (p2[:, :, None] * (var_obs + dvar)).sum(axis=1)

        if process_err and resid_err:  # MixSIAR multiplicative residual x process error
            resid_prop = pm.Uniform("resid_prop", 0, 20, dims="isotope")
            if source["data_type"] == "raw" and j > 1:
                scale_mat = pt.sqrt(resid_prop[:, None] * resid_prop[None, :])
                total_cov = process_cov * scale_mat[None, :, :]
                pm.MvNormal(
                    "mix",
                    mu=expected,
                    cov=total_cov,
                    observed=mix["data_iso"],
                    dims=("obs", "isotope"),
                )
            else:
                pm.Normal(
                    "mix",
                    mu=expected,
                    sigma=pt.sqrt(process_var * resid_prop),
                    observed=mix["data_iso"],
                    dims=("obs", "isotope"),
                )
        elif resid_err:  # residual-only precision prior from MixSIAR
            if j > 1:
                precision = _wishart_identity_bartlett(pm, pt, "Sigma", j, j + 1)
                delta = pt.as_tensor_variable(mix["data_iso"]) - expected
                logdet = pt.log(pt.linalg.det(precision))
                quad = pt.sum((delta @ precision) * delta, axis=1)
                loglik = 0.5 * (logdet - j * np.log(2 * np.pi) - quad)
                pm.Deterministic("loglik", loglik, dims="obs")
                pm.Potential("mix", pt.sum(loglik))
            else:
                precision = pm.Gamma("Sigma", 0.001, 0.001)
                pm.Normal(
                    "mix",
                    mu=expected,
                    tau=precision,
                    observed=mix["data_iso"],
                    dims=("obs", "isotope"),
                )
        else:  # process-only (MixSIR): diagonal even for raw source covariance
            if source["data_type"] == "raw" and j > 1:
                process_var = pt.diagonal(process_cov, axis1=-2, axis2=-1)
            pm.Normal(
                "mix",
                mu=expected,
                sigma=pt.sqrt(process_var),
                observed=mix["data_iso"],
                dims=("obs", "isotope"),
            )
    return model


def run_model(
    run,
    mix,
    source,
    discr,
    model_filename=None,
    alpha_prior=1,
    process_err=True,
    resid_err=True,
    random_seed=None,
    backend="pymc",
    device="auto",
    **sample_kwargs,
):
    """Build and sample a model with a selectable scalable NUTS backend.

    Parameters
    ----------
    backend : {"pymc", "nutpie", "numpyro", "blackjax", "auto"}
        NUTS implementation. ``auto`` selects the fastest installed backend
        appropriate for the current platform.
    device : {"auto", "cpu", "gpu"}
        Execution device. GPU is available for NumPyro/BlackJAX when JAX sees
        a compatible accelerator. Explicit GPU requests never silently fall
        back to CPU.
    """
    pm, _ = _import_pymc()
    from .backends import resolve_backend, resolve_device, runtime_metadata

    requested_backend = str(backend).lower()
    inference_backend = resolve_backend(requested_backend, device=device)
    actual_device, device_context = resolve_device(inference_backend, device)
    if isinstance(run, str):
        if run not in RUN_PRESETS:
            raise ValueError(f"Unknown run preset: {run}")
        cfg = RUN_PRESETS[run].copy()
    else:
        old = dict(run)
        cfg = {
            "draws": old.pop("draws", old.pop("chainLength", 1000)),
            "tune": old.pop("tune", old.pop("burn", 500)),
            "chains": old.pop("chains", 2),
            "thin": old.pop("thin", 1),
        }
        cfg.update(old)
    thin = int(cfg.pop("thin", 1))
    # On Windows, spawned workers import PyMC/ArviZ in a fresh interpreter and
    # bypass the SciPy compatibility shim above. Sequential chains are robust
    # and statistically equivalent; callers can still explicitly pass cores>1
    # after upgrading their ArviZ/SciPy combination.
    if os.name == "nt":
        sample_kwargs.setdefault("cores", 1)
    # Bartlett-Wishart residual precision must start from its valid matrix
    # initial value; jittering its latent diagonal can produce invalid starts.
    sample_kwargs.setdefault("init", "adapt_diag")
    total_started = time.perf_counter()
    build_started = time.perf_counter()
    model = build_model(mix, source, discr, alpha_prior, process_err, resid_err)
    build_seconds = time.perf_counter() - build_started
    def sample_model():
        sampling_started = time.perf_counter()
        with device_context:
            with model:
                sampled = pm.sample(
                    random_seed=random_seed,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                    nuts_sampler=inference_backend,
                    **cfg,
                    **sample_kwargs,
                )
        return sampled, time.perf_counter() - sampling_started

    try:
        result, sampling_seconds = sample_model()
    except Exception as exc:
        message = str(exc).lower()
        pytensor_compile_failure = (
            type(exc).__name__ == "CompileError"
            and ("lazylinker" in message or "compilation failed" in message)
        )
        if not pytensor_compile_failure:
            raise
        # Some Windows/Anaconda Python combinations have an incompatible
        # MinGW C linker. PyTensor's supported Python linker is slower but
        # statistically equivalent and keeps Spyder workflows functional.
        import pytensor

        warnings.warn(
            "PyTensor C compilation failed; retrying with the pure-Python "
            "linker. Sampling will be slower. For best performance use a "
            "clean Python 3.11/3.12 environment with a compatible compiler.",
            RuntimeWarning,
            stacklevel=2,
        )
        pytensor.config.cxx = ""
        pytensor.config.linker = "py"
        result, sampling_seconds = sample_model()
    # Potential-based residual-only likelihood is stored as a posterior
    # deterministic. Promote it to ArviZ's standard log_likelihood group.
    if "loglik" in result.posterior:
        import xarray as xr

        dataset = xr.Dataset({"mix": result.posterior["loglik"]})
        if "log_likelihood" in result.groups():
            result.log_likelihood = dataset
        else:
            result.add_groups({"log_likelihood": dataset})
    elif "log_likelihood" in result.groups() and "mix" in result.log_likelihood:
        # R MixSIAR model comparison uses mixture log-likelihood only, not the
        # source-data likelihood that is also part of the hierarchical model.
        result.log_likelihood = result.log_likelihood[["mix"]]
    if thin > 1:
        result = result.sel(draw=slice(None, None, thin))
    # NetCDF attributes do not support booleans portably.
    metadata = runtime_metadata()
    result.attrs.update(
        {
            "backend": "pymc",
            "inference_backend": inference_backend,
            "requested_backend": requested_backend,
            "compute_device": actual_device,
            "build_seconds": float(build_seconds),
            "sampling_seconds": float(sampling_seconds),
            "total_seconds": float(time.perf_counter() - total_started),
            "process_error": int(process_err),
            "residual_error": int(resid_err),
            **metadata,
        }
    )
    return result
