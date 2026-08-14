"""Save complete MixSIARPy results and R-style posterior figures."""

from pathlib import Path
import numpy as np


def save_results(fit, output, source_names, mix):
    import arviz as az
    import matplotlib.pyplot as plt
    from .results import save_diagnostics
    from .plotting import plot_proportion_density

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "posterior_complete.tmp.nc"
    fit.to_netcdf(temporary)
    temporary.replace(output / "posterior_complete.nc")
    az.summary(fit).to_csv(output / "summary_all_parameters.csv")
    global_summary = az.summary(fit, var_names=["p_global"], hdi_prob=0.95)
    global_summary.index = [f"p_global[{x}]" for x in source_names]
    global_summary.to_csv(output / "summary_source_proportions.csv")
    if "ilr_beta" in fit.posterior:
        az.summary(fit, var_names=["ilr_beta"], hdi_prob=0.95).to_csv(
            output / "summary_compositional_regression.csv"
        )
    save_diagnostics(fit, output)
    for index, fac in enumerate(mix.get("FAC", []), 1):
        variable = f"p_fac{index}"
        if variable not in fit.posterior:
            continue
        data = fit.posterior[variable]
        level_dim = [d for d in data.dims if d not in ("chain", "draw")][-2]
        source_dim = data.dims[-1]
        labels = fac["labels"]
        data = data.assign_coords({level_dim: labels, source_dim: source_names})
        az.summary(data, hdi_prob=0.95).to_csv(output / f"summary_{variable}.csv")
        q = data.quantile([0.025, 0.5, 0.975], dim=("chain", "draw"))
        fig, axes = plt.subplots(
            len(labels), 1, figsize=(8.2, max(3.2, 2.35 * len(labels))),
            squeeze=False, sharex=True
        )
        source_colors = ["#E76F61", "#00A65A", "#4C8DFF", "#9C6ADE"]
        for lev, ax in zip(labels, axes[:, 0]):
            row = q.sel({level_dim: lev})
            y = np.arange(len(source_names))
            med = row.sel(quantile=0.5).values
            lo = row.sel(quantile=0.025).values
            hi = row.sel(quantile=0.975).values
            for source_index in range(len(source_names)):
                ax.errorbar(
                    med[source_index], y[source_index],
                    xerr=[[med[source_index] - lo[source_index]],
                          [hi[source_index] - med[source_index]]],
                    fmt="o", color=source_colors[source_index % len(source_colors)],
                    markersize=5.5, capsize=3, linewidth=1.5,
                )
            ax.set(
                yticks=y,
                yticklabels=source_names,
                xlim=(0, 1),
                title=str(lev),
            )
            ax.invert_yaxis()
            ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1, 0].set_xlabel(
            "Source proportion (median and 95% credible interval)"
        )
        fig.subplots_adjust(
            left=0.22, right=0.97, top=0.96, bottom=0.10, hspace=0.55
        )
        fig.savefig(
            output / f"intervals_{variable}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        for position, label in enumerate(labels):
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))
            fig = plot_proportion_density(
                data.isel({level_dim: position}).values,
                source_names,
                str(label),
                output / f"posterior_density_{variable}_{safe}.png",
            )
            plt.close(fig)
    if "p_both" in fit.posterior:
        both = fit.posterior["p_both"]
        both.to_netcdf(output / "posterior_p_both.nc")
        az.summary(both, hdi_prob=0.95).to_csv(output / "summary_p_both.csv")
        level_dims = [d for d in both.dims if d not in ("chain", "draw", "source")]
        for i in range(both.sizes[level_dims[0]]):
            for j in range(both.sizes[level_dims[1]]):
                label1 = str(both.coords[level_dims[0]].values[i])
                label2 = str(both.coords[level_dims[1]].values[j])
                safe = "_".join(
                    "".join(c if c.isalnum() or c in "-_" else "_" for c in x)
                    for x in (label1, label2)
                )
                fig = plot_proportion_density(
                    both.isel({level_dims[0]: i, level_dims[1]: j}).values,
                    source_names,
                    f"{label1} × {label2}",
                    output / f"posterior_density_p_both_{safe}.png",
                )
                plt.close(fig)
    if mix.get("n_ce") and "ilr_cont1" in fit.posterior:
        from .results import continuous_effect_prediction, plot_continuous_effect

        prediction = continuous_effect_prediction(fit, mix, source_names)
        prediction.to_netcdf(output / "continuous_effect_predictions.nc")
        fig = plot_continuous_effect(prediction, output / "continuous_effect.png")
        plt.close(fig)
    vars_ = [
        v
        for v in (
            "p_global", "ilr_beta", "fac1_sig", "fac2_sig", "resid_prop", "Sigma"
        )
        if v in fit.posterior
    ]
    axes = az.plot_trace(
        fit, var_names=vars_, figsize=(14, max(4.5, 3.0 * len(vars_))),
        compact=True,
    )
    trace_figure = np.asarray(axes).ravel()[0].figure
    trace_figure.subplots_adjust(
        left=0.07, right=0.98, top=0.97, bottom=0.05,
        hspace=0.72, wspace=0.20,
    )
    trace_figure.savefig(
        output / "trace_core_parameters.png", dpi=300, bbox_inches="tight"
    )
    plt.close("all")
    axes = az.plot_posterior(fit, var_names=["p_global"], hdi_prob=0.95)
    np.asarray(axes).ravel()[0].figure.savefig(
        output / "posterior_source_proportions.png", dpi=200
    )
    plt.close("all")
    for kind, function in (("loo", az.loo), ("waic", az.waic)):
        try:
            (output / f"{kind}.txt").write_text(str(function(fit)), encoding="utf-8")
        except Exception as exc:
            (output / f"{kind}.txt").write_text(
                f"Could not calculate {kind}: {exc}\n", encoding="utf-8"
            )
    return global_summary
