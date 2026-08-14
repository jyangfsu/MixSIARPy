from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gaussian_kde


_FACTOR_COLORS = [
    "#E76F61",  # coral
    "#00A65A",  # green
    "#4C8DFF",  # blue
    "#9C6ADE",
    "#00A6A6",
    "#C58B00",
]
_FACTOR_MARKERS = ["o", "^", "s", "+", "s", "*", "o", "v", "D", "P", "X"]
_SOURCE_LINESTYLES = ["-", "--", ":", "-."]


def _tracer_label(name):
    """Return a publication-style isotope label when the name is recognizable."""
    lower = str(name).lower()
    isotopes = {"c": "13", "n": "15", "s": "34", "o": "18"}
    for element, mass in isotopes.items():
        if element in lower:
            return rf"$\delta^{{{mass}}}${element.upper()} (‰)"
    return str(name)


def _save_isospace_figure(fig, filename, suffix=""):
    """Save an editable PDF/SVG and a high-resolution PNG."""
    if filename is None:
        return
    path = Path(filename)
    stem = path.with_name(path.stem + suffix)
    requested = stem.with_suffix(path.suffix or ".pdf")
    fig.savefig(requested, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def plot_prior(alpha_prior, source, filename=None, points=400):
    """Plot MixSIAR marginal Dirichlet priors against the alpha=1 reference."""
    k = source["n_sources"]
    alpha = (
        np.ones(k)
        if np.isscalar(alpha_prior) and alpha_prior == 1
        else np.asarray(alpha_prior, float)
    )
    if alpha.shape != (k,) or np.any(alpha <= 0):
        raise ValueError(f"alpha_prior needs {k} positive values")
    x = np.linspace(0.001, 0.999, points)
    fig, axes = plt.subplots(k, 2, figsize=(10, max(3, 2.6 * k)), squeeze=False)
    reference = np.ones(k)
    for i, name in enumerate(source["source_names"]):
        axes[i, 0].plot(
            x, beta.pdf(x, alpha[i], alpha.sum() - alpha[i]), color="red", lw=1.5
        )
        axes[i, 0].fill_between(
            x, beta.pdf(x, alpha[i], alpha.sum() - alpha[i]), color="red", alpha=0.35
        )
        axes[i, 1].plot(x, beta.pdf(x, 1, k - 1), color="0.35", lw=1.5)
        axes[i, 1].fill_between(x, beta.pdf(x, 1, k - 1), color="0.6", alpha=0.5)
        for ax in axes[i]:
            ax.set(xlim=(0, 1), ylabel="Density", title=str(name))
    axes[-1, 0].set_xlabel(f"Your prior: {alpha.tolist()}")
    axes[-1, 1].set_xlabel(f"Reference prior: {[1]*k}")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


def plot_proportion_density(data, source_names, title, filename=None):
    """Plot scaled marginal posterior densities of source proportions.

    Each curve shows uncertainty in one source contribution for the named
    population or factor level. Curves are scaled independently to one, so
    compare their locations and widths rather than absolute peak heights.
    """
    values = np.asarray(data).reshape(-1, len(source_names))
    x = np.linspace(0, 1, 400)
    colors = ["#F8766D", "#00BA38", "#619CFF", "#C77CFF", "#00BFC4", "#B79F00"]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for i, name in enumerate(source_names):
        col = values[:, i]
        kde = gaussian_kde(col)
        y = kde(x)
        y = y / y.max()
        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, lw=1.6)
        ax.fill_between(x, y, color=color, alpha=0.20, label=str(name))
        ax.axvline(np.median(col), ymax=0.10, color=color, lw=1.4)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="Proportion of Diet",
        ylabel="Scaled Posterior Density",
        title=str(title),
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=min(3, len(source_names)), frameon=False,
        columnspacing=1.2, handlelength=2.0,
    )
    ax.grid(color="#E7E7E7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.82, left=0.12, right=0.97, bottom=0.12)
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


def plot_data(mix, source, discr, filename=None):
    """Plot MixSIAR-style isotope space with factor-aware visual encoding.

    Mixture observations use factor 1 as color and factor 2 as marker. Source
    means include discrimination means; error bars are ±1 combined source and
    discrimination SD, matching the R MixSIAR definition.
    """
    if sorted(discr["mu"].index.astype(str)) != sorted(source["source_names"]):
        raise ValueError("Source names differ between source and discrimination files")
    smu = source["S_MU"].to_numpy()[:, : mix["n_iso"]]
    ssig = source["S_SIG"].to_numpy()[:, : mix["n_iso"]]
    # For factor-specific sources show every source/factor combination.
    repeat = len(smu) // source["n_sources"]
    dmu = np.repeat(discr["mu"].to_numpy(), repeat, axis=0)
    dsd = np.sqrt(np.repeat(discr["sig2"].to_numpy(), repeat, axis=0) + ssig**2)
    means = smu + dmu
    pairs = (
        [(0, 0)]
        if mix["n_iso"] == 1
        else [
            (a, b) for a in range(mix["n_iso"] - 1) for b in range(a + 1, mix["n_iso"])
        ]
    )
    figs = []
    for a, b in pairs:
        fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
        if a == b:
            ax.hist(
                mix["data_iso"][:, a], bins="auto", color="#6384A8",
                edgecolor="white", alpha=0.75, label="Mixture"
            )
            ax.errorbar(
                means[:, a],
                np.zeros(len(means)),
                xerr=dsd[:, a],
                fmt="D", color="#202020", capsize=3, label="Sources",
            )
            ax.set_xlabel(_tracer_label(mix["iso_names"][a]))
            ax.set_yticks([])
        else:
            _plot_mixture_factors(ax, mix, a, b)
            _plot_sources(ax, source, means, dsd, a, b)
            ax.set(
                xlabel=_tracer_label(mix["iso_names"][a]),
                ylabel=_tracer_label(mix["iso_names"][b]),
            )
        ax.grid(True, color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
        ax.tick_params(direction="out", length=3.5, width=0.8)
        figs.append(fig)
        suffix = "" if len(pairs) == 1 else f"_{a+1}_{b+1}"
        _save_isospace_figure(fig, filename, suffix)
    return figs[0] if len(figs) == 1 else figs


def _plot_mixture_factors(ax, mix, x_index, y_index):
    """Draw mixture points and separate color/marker legends."""
    from matplotlib.lines import Line2D

    factors = mix.get("FAC", [])
    if not factors:
        ax.scatter(
            mix["data_iso"][:, x_index], mix["data_iso"][:, y_index],
            s=27, color="#555555", edgecolor="white", linewidth=0.35,
        )
        return

    color_codes = factors[0]["codes"]
    marker_codes = factors[1]["codes"] if len(factors) > 1 else np.zeros(mix["N"], int)
    for row in range(mix["N"]):
        marker = _FACTOR_MARKERS[marker_codes[row] % len(_FACTOR_MARKERS)]
        color = _FACTOR_COLORS[color_codes[row] % len(_FACTOR_COLORS)]
        line_marker = marker in {"+", "*"}
        filled = marker not in {"+", "*"} and marker_codes[row] < 4
        scatter_options = {
            "marker": marker, "s": 31, "linewidth": 0.85, "zorder": 3
        }
        if line_marker:
            scatter_options["color"] = color
        else:
            scatter_options.update(
                facecolor=color if filled else "none", edgecolor=color
            )
        ax.scatter(
            mix["data_iso"][row, x_index], mix["data_iso"][row, y_index],
            **scatter_options,
        )

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=_FACTOR_COLORS[i % len(_FACTOR_COLORS)],
               markeredgecolor="none", label=label)
        for i, label in enumerate(factors[0]["labels"])
    ]
    legend1 = ax.legend(
        handles=color_handles, loc="upper left", frameon=False,
        handletextpad=0.45, borderaxespad=0.5, fontsize=8.5,
    )
    ax.add_artist(legend1)
    if len(factors) > 1:
        marker_handles = [
            Line2D([0], [0], marker=_FACTOR_MARKERS[i % len(_FACTOR_MARKERS)],
                   linestyle="none", color="black", markersize=5,
                   markerfacecolor="black" if i < 4 else "none", label=label)
            for i, label in enumerate(factors[1]["labels"])
        ]
        ax.legend(
            handles=marker_handles, loc="upper left", bbox_to_anchor=(0, 0.82),
            frameon=False, handletextpad=0.45, borderaxespad=0.5, fontsize=8.5,
        )


def _plot_sources(ax, source, means, combined_sd, x_index, y_index):
    """Draw source means, uncertainty and direct labels.

    When sources vary by a factor, color represents that factor exactly as in
    the R plot. Otherwise color represents source identity. The latter is a
    readability improvement over R's all-black source symbols and is especially
    useful for examples such as Lake, which have no categorical mixture effect.
    """
    factor_count = source["S_factor_levels"] or 1
    for source_index, source_name in enumerate(source["source_names"]):
        start = source_index * factor_count
        for factor_index in range(factor_count):
            row = start + factor_index
            color = (
                _FACTOR_COLORS[factor_index % len(_FACTOR_COLORS)]
                if source["by_factor"] is not None
                else _FACTOR_COLORS[source_index % len(_FACTOR_COLORS)]
            )
            ax.errorbar(
                means[row, x_index], means[row, y_index],
                xerr=combined_sd[row, x_index], yerr=combined_sd[row, y_index],
                fmt="D", markersize=7.5, color=color, markeredgecolor="white",
                markeredgewidth=0.45, elinewidth=1.2, capsize=0,
                linestyle=_SOURCE_LINESTYLES[source_index % len(_SOURCE_LINESTYLES)],
                zorder=5,
            )
        label_row = start
        label_color = (
            "#111111" if source["by_factor"] is not None
            else _FACTOR_COLORS[source_index % len(_FACTOR_COLORS)]
        )
        ax.annotate(
            str(source_name),
            (means[label_row, x_index], means[label_row, y_index]),
            xytext=(-10, 13), textcoords="offset points", ha="right",
            va="bottom", fontsize=9, color=label_color, fontweight="semibold",
            zorder=6,
        )


def plot_posteriors(result, var_names=("p_global",)):
    import arviz as az

    return az.plot_posterior(result, var_names=list(var_names))


def plot_diagnostics(result, var_names=None):
    import arviz as az

    return az.plot_trace(result, var_names=var_names)
