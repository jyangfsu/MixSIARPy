import numpy as np
import xarray as xr


def summary_stat(result, var_names=None, hdi_prob=0.95):
    """Return posterior mean, SD, HDI, ESS and R-hat as a DataFrame."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("arviz is required") from exc
    return az.summary(result, var_names=var_names, hdi_prob=hdi_prob)


def compare_models(models, names=None, ic="loo"):
    """Compare fitted models with PSIS-LOO (default) or WAIC."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("arviz is required") from exc
    names = names or [f"model_{i+1}" for i in range(len(models))]
    if len(names) != len(models):
        raise ValueError("names and models must have equal length")
    return az.compare(dict(zip(names, models)), ic=ic)


def combine_sources(result, groups, source_names=None, variable="p_global"):
    """Combine posterior source proportions without refitting the model.

    ``groups`` maps new group names to source names or integer positions. The
    groups must form a non-overlapping partition of all original sources.
    """
    if variable not in result.posterior:
        raise ValueError(f"Posterior variable {variable!r} was not found")
    values = result.posterior[variable]
    source_dim = "source" if "source" in values.dims else values.dims[-1]
    labels = list(
        source_names
        or values.coords.get(source_dim, np.arange(values.sizes[source_dim])).values
    )
    used = []
    arrays = []
    names = []
    for name, members in groups.items():
        idx = []
        for member in members:
            pos = (
                int(member)
                if isinstance(member, (int, np.integer))
                else labels.index(member)
            )
            idx.append(pos)
            used.append(pos)
        arrays.append(values.isel({source_dim: idx}).sum(source_dim))
        names.append(name)
    if sorted(used) != list(range(len(labels))) or len(set(used)) != len(used):
        raise ValueError("groups must include every source exactly once")
    combined = xr.concat(arrays, dim=xr.IndexVariable("combined_source", names))
    return combined.transpose(
        *[d for d in values.dims if d != source_dim], "combined_source"
    )


def continuous_effect_prediction(
    result, mix, source_names=None, values=None, factor1_level=None, factor2_level=None
):
    """Predict source proportions over a continuous covariate on its original scale."""
    if not mix.get("n_ce") or "ilr_cont1" not in result.posterior:
        raise ValueError("The fitted model has no continuous effect")
    original = np.asarray(mix["CE_orig"][0], float)
    values = (
        np.linspace(original.min(), original.max(), 100)
        if values is None
        else np.asarray(values, float)
    )
    scaled = (values - mix["CE_center"][0]) / mix["CE_scale"][0]
    post = result.posterior
    eta = post["ilr_global"]
    if factor1_level is not None:
        level_dim = [
            d for d in post["ilr_fac1"].dims if d not in ("chain", "draw", "ilr")
        ][0]
        eta = eta + post["ilr_fac1"].isel({level_dim: factor1_level})
    if factor2_level is not None:
        level_dim = [
            d for d in post["ilr_fac2"].dims if d not in ("chain", "draw", "ilr")
        ][0]
        eta = eta + post["ilr_fac2"].isel({level_dim: factor2_level})
    ilr = eta.expand_dims(prediction_value=values) + post["ilr_cont1"].expand_dims(
        prediction_value=values
    ) * xr.DataArray(scaled, dims="prediction_value")
    from .model import _ilr_basis

    logits = xr.dot(
        ilr,
        xr.DataArray(
            _ilr_basis(len(source_names or post.coords["source"])),
            dims=("source", "ilr"),
        ),
        dims="ilr",
    )
    exp = np.exp(logits - logits.max("source"))
    pred = exp / exp.sum("source")
    if source_names is not None:
        pred = pred.assign_coords(source=source_names)
    return pred.transpose("chain", "draw", "prediction_value", "source")


def compositional_regression_prediction(result, mix, new_data, source_names=None):
    """Predict compositions for new rows using a generalized ILR regression."""
    import pandas as pd
    from .data import _composition_design
    if mix.get("regression") is None or "ilr_beta" not in result.posterior:
        raise ValueError("The fitted model has no generalized compositional regression")
    frame = pd.DataFrame(new_data).copy()
    meta = mix["regression"]["metadata"]
    # Build using training transformations, including fixed reference levels.
    encoded = {}
    for name, info in meta.items():
        if name not in frame:
            raise ValueError(f"Prediction data are missing {name!r}")
        if info["kind"] == "numeric":
            encoded[name] = pd.DataFrame({
                name: (frame[name].to_numpy(float) - info["center"]) / info["scale"]
            })
        else:
            values = frame[name].astype(str)
            unknown = sorted(set(values) - set(info["levels"]))
            if unknown:
                raise ValueError(f"Unknown levels for {name!r}: {unknown}")
            cat = pd.Categorical(values, categories=info["levels"])
            encoded[name] = pd.get_dummies(
                cat, prefix=name, drop_first=True, dtype=float
            ).reset_index(drop=True)
    pieces = []
    for term in mix["regression"]["terms"]:
        parts = term.split(":")
        if len(parts) == 1:
            pieces.append(encoded[parts[0]])
        else:
            left, right = encoded[parts[0]], encoded[parts[1]]
            pieces.append(pd.DataFrame({
                f"{a}:{b}": left[a].to_numpy() * right[b].to_numpy()
                for a in left.columns for b in right.columns
            }))
    design = pd.concat(pieces, axis=1)
    design = design.loc[:, ~design.columns.duplicated()]
    design = design.reindex(columns=mix["regression"]["columns"], fill_value=0.0)
    post = result.posterior
    x = xr.DataArray(
        design.to_numpy(float),
        dims=("prediction", "regression_term"),
        coords={"prediction": np.arange(len(frame)),
                "regression_term": mix["regression"]["columns"]},
    )
    ilr = post["ilr_global"] + xr.dot(post["ilr_beta"], x, dims="regression_term")
    from .model import _ilr_basis
    names = list(source_names or post.coords["source"].values)
    logits = xr.dot(
        ilr,
        xr.DataArray(_ilr_basis(len(names)), dims=("source", "ilr")),
        dims="ilr",
    )
    exp = np.exp(logits - logits.max("source"))
    pred = (exp / exp.sum("source")).assign_coords(source=names)
    for column in frame.columns:
        pred = pred.assign_coords({column: ("prediction", frame[column].to_numpy())})
    return pred.transpose("chain", "draw", "prediction", "source")


def plot_continuous_effect(prediction, filename=None, hdi_prob=0.95):
    """Plot posterior median and HDI for continuous-effect predictions."""
    import matplotlib.pyplot as plt

    alpha = (1 - hdi_prob) / 2
    q = prediction.quantile([alpha, 0.5, 1 - alpha], dim=("chain", "draw"))
    x = prediction.coords["prediction_value"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    for source in prediction.coords["source"].values:
        y = q.sel(source=source)
        ax.plot(x, y.sel(quantile=0.5), label=str(source))
        ax.fill_between(x, y.sel(quantile=alpha), y.sel(quantile=1 - alpha), alpha=0.2)
    ax.set(xlabel="Continuous covariate", ylabel="Source proportion", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


def diagnostics(result, var_names=None, rhat_limit=1.05, ess_limit=400):
    """Return MixSIAR-style convergence checks and a compact status report."""
    import arviz as az

    table = az.summary(result, var_names=var_names, kind="diagnostics")
    table["rhat_ok"] = table["r_hat"].fillna(np.inf) <= rhat_limit
    table["ess_bulk_ok"] = table["ess_bulk"].fillna(0) >= ess_limit
    table["ess_tail_ok"] = table["ess_tail"].fillna(0) >= ess_limit
    divergences = (
        int(result.sample_stats["diverging"].sum())
        if "diverging" in result.sample_stats
        else 0
    )
    report = {
        "parameters": len(table),
        "rhat_failures": int((~table.rhat_ok).sum()),
        "ess_bulk_failures": int((~table.ess_bulk_ok).sum()),
        "ess_tail_failures": int((~table.ess_tail_ok).sum()),
        "divergences": divergences,
        "converged": bool(
            table.rhat_ok.all()
            and table.ess_bulk_ok.all()
            and table.ess_tail_ok.all()
            and divergences == 0
        ),
    }
    return table, report


def save_diagnostics(result, directory, var_names=None):
    """Save diagnostic table and human-readable report."""
    from pathlib import Path
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table, report = diagnostics(result, var_names=var_names)
    table.to_csv(directory / "diagnostics.csv")
    (directory / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = ["MixSIARPy convergence diagnostics", ""] + [
        f"{k}: {v}" for k, v in report.items()
    ]
    (directory / "diagnostics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return table, report
