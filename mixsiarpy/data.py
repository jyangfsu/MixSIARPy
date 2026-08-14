"""CSV loading and validation compatible with MixSIAR's public data API."""

from pathlib import Path
import numpy as np
import pandas as pd


def _names(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def load_mix_data(
    filename,
    iso_names,
    factors=None,
    fac_random=None,
    fac_nested=None,
    cont_effects=None,
    composition_formula=None,
):
    data = pd.read_csv(Path(filename))
    iso_names, factors = _names(iso_names), _names(factors)
    cont_effects = _names(cont_effects)
    fac_random = [] if fac_random is None else list(fac_random)
    if not iso_names:
        raise ValueError("At least one isotope/tracer must be selected")
    if len(factors) != len(fac_random):
        raise ValueError("fac_random must have one value for every factor")
    if len(factors) > 2:
        raise ValueError("MixSIAR supports at most two categorical effects")
    if len(cont_effects) > 1 and composition_formula is None:
        raise ValueError("MixSIAR supports at most one continuous effect")
    if len(factors) == 2:
        if fac_nested is None or len(fac_nested) != 2:
            raise ValueError(
                "fac_nested must have two values when two factors are used"
            )
        fac_nested = list(map(bool, fac_nested))
        if all(fac_nested):
            raise ValueError("Two factors cannot be nested within each other")
    else:
        fac_nested = [False] * len(factors)
    selected = iso_names + factors + cont_effects
    missing = [x for x in selected if x not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in mixture data: {missing}")
    if data[iso_names].isna().any().any():
        raise ValueError("Mixture isotope values may not be missing")

    fac = []
    for name, random in zip(factors, fac_random):
        codes, levels = pd.factorize(data[name], sort=True)
        labels = [
            f"{name} {x}" if pd.api.types.is_numeric_dtype(data[name]) else str(x)
            for x in levels
        ]
        fac.append(
            {
                "values": codes + 1,
                "codes": codes,
                "levels": len(levels),
                "labels": labels,
                "lookup": None,
                "re": bool(random),
                "name": name,
            }
        )
    # R MixSIAR requires the fixed effect first in a mixed FE/RE pair.
    if len(fac) == 2 and sum(fac_random) == 1 and fac[0]["re"]:
        fac.reverse()
        factors.reverse()
        fac_random.reverse()
        fac_nested.reverse()
    if len(fac) == 2:
        for i, nested in enumerate(fac_nested):
            if nested:
                other = 1 - i
                lookup = []
                for level in range(fac[i]["levels"]):
                    parents = np.unique(fac[other]["codes"][fac[i]["codes"] == level])
                    if len(parents) != 1:
                        raise ValueError(
                            f"Factor {fac[i]['name']} is not strictly nested"
                        )
                    lookup.append(int(parents[0]) + 1)
                fac[i]["lookup"] = np.asarray(lookup)

    ce_orig, ce, centers, scales = [], [], [], []
    for name in cont_effects:
        values = data[name].to_numpy(float)
        scale = values.std(ddof=1)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Continuous effect {name!r} has zero/invalid SD")
        ce_orig.append(values)
        centers.append(values.mean())
        scales.append(scale)
        ce.append((values - values.mean()) / scale)

    regression = None
    if composition_formula is not None:
        regression = _composition_design(data, composition_formula)
    n_re = sum(bool(x) for x in fac_random)
    return {
        "data": data,
        "data_iso": data[iso_names].to_numpy(float),
        "n_iso": len(iso_names),
        "n_re": n_re,
        "n_ce": len(ce),
        "FAC": fac,
        "CE": ce,
        "CE_orig": ce_orig,
        "CE_center": centers,
        "CE_scale": scales,
        "cont_effects": cont_effects,
        "MU_names": ["Mean" + x for x in iso_names],
        "SIG_names": ["SD" + x for x in iso_names],
        "iso_names": iso_names,
        "N": len(data),
        "n_fe": len(factors) - n_re,
        "n_effects": len(factors),
        "factors": factors,
        "fac_random": fac_random,
        "fac_nested": fac_nested,
        "fere": len(factors) == 2 and n_re < 2,
        "composition_formula": composition_formula,
        "regression": regression,
    }


def _composition_design(data, formula):
    """Create a full-rank design matrix for ILR compositional regression.

    The small formula language supports numeric/categorical columns, ``:``
    interactions and R-style ``*`` expansion.  An intercept is handled by
    ``p_global`` and is therefore not included in the returned matrix.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("composition_formula must be a non-empty string")
    rhs = formula.split("~", 1)[-1]
    requested = []
    for item in (x.strip() for x in rhs.split("+") if x.strip()):
        if "*" in item:
            names = [x.strip() for x in item.split("*")]
            if len(names) != 2:
                raise ValueError("Only two-way '*' interactions are supported")
            requested.extend([names[0], names[1], ":".join(names)])
        else:
            requested.append(item)
    terms = list(dict.fromkeys(requested))
    variables = list(dict.fromkeys(v for t in terms for v in t.split(":")))
    missing = [v for v in variables if v not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in composition_formula: {missing}")

    encoded, metadata = {}, {}
    for name in variables:
        series = data[name]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(float)
            center, scale = float(values.mean()), float(values.std(ddof=1))
            if not np.isfinite(scale) or scale == 0:
                raise ValueError(f"Regression variable {name!r} has zero/invalid SD")
            encoded[name] = pd.DataFrame({name: (values - center) / scale})
            metadata[name] = {"kind": "numeric", "center": center, "scale": scale}
        else:
            levels = sorted(series.astype(str).unique())
            if len(levels) < 2:
                raise ValueError(f"Regression variable {name!r} has only one level")
            cat = pd.Categorical(series.astype(str), categories=levels)
            frame = pd.get_dummies(cat, prefix=name, drop_first=True, dtype=float)
            encoded[name] = frame.reset_index(drop=True)
            metadata[name] = {"kind": "categorical", "levels": levels}

    columns = []
    for term in terms:
        parts = term.split(":")
        if len(parts) == 1:
            frame = encoded[parts[0]].copy()
        elif len(parts) == 2:
            left, right = encoded[parts[0]], encoded[parts[1]]
            frame = pd.DataFrame(
                {
                    f"{a}:{b}": left[a].to_numpy() * right[b].to_numpy()
                    for a in left.columns for b in right.columns
                }
            )
        else:
            raise ValueError("Only two-way ':' interactions are supported")
        columns.append(frame)
    design = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=data.index)
    if design.columns.duplicated().any():
        design = design.loc[:, ~design.columns.duplicated()]
    return {
        "matrix": design.to_numpy(float),
        "columns": list(design.columns),
        "metadata": metadata,
        "terms": terms,
        "formula": formula,
    }


def load_source_data(
    filename, source_factors=None, conc_dep=False, data_type="raw", mix=None
):
    if mix is None:
        raise ValueError("mix is required")
    source_factors = _names(source_factors)
    if len(source_factors) > 1:
        raise ValueError("Only one source factor is supported")
    factor = source_factors[0] if source_factors else None
    df = pd.read_csv(Path(filename))
    if factor and factor not in df.columns:
        raise ValueError(f"Source factor {factor!r} is not a column")
    if factor and factor not in mix["factors"]:
        raise ValueError(f"Source factor {factor!r} is not a mixture factor")
    data_type = data_type.lower()
    if data_type not in {"raw", "means"}:
        raise ValueError("data_type must be 'raw' or 'means'")
    source_col = df.columns[0]
    source_names = sorted(df[source_col].astype(str).unique())
    factor_levels = sorted(df[factor].unique()) if factor else []
    expected = set(source_names)
    if factor:
        for level, group in df.groupby(factor, observed=True):
            if set(group[source_col].astype(str).unique()) != expected:
                raise ValueError(f"Sources differ at factor level {level!r}")
    df = df.assign(**{source_col: df[source_col].astype(str)})
    df = df.sort_values([source_col] + ([factor] if factor else [])).reset_index(
        drop=True
    )
    conc = None
    if conc_dep:
        cols = ["Conc" + x for x in mix["iso_names"]]
        missing = [x for x in cols if x not in df]
        if missing:
            raise ValueError(f"Concentration columns missing: {missing}")
        conc = df.groupby(source_col, sort=True)[cols].mean().to_numpy(float)
        if np.any(~np.isfinite(conc)) or np.any(conc <= 0):
            raise ValueError("Concentration values must be finite and positive")

    shape = (len(source_names), len(mix["iso_names"]))
    if data_type == "raw":
        missing = [x for x in mix["iso_names"] if x not in df]
        if missing:
            raise ValueError(f"Raw isotope columns missing: {missing}")
        if df[mix["iso_names"]].isna().any().any():
            raise ValueError("Raw source isotope values may not be missing")
        keys = [source_col] + ([factor] if factor else [])
        stats = df.groupby(keys, sort=True)[mix["iso_names"]].agg(["mean", "std"])
        means = stats.xs("mean", axis=1, level=1).to_numpy(float)
        sds = stats.xs("std", axis=1, level=1).to_numpy(float)
        if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
            raise ValueError("Every source group needs >=2 replicates and non-zero SD")
        if factor:
            shape3 = shape + (len(factor_levels),)
            mu, sig = means.reshape(shape3[0], shape3[2], shape3[1]).transpose(
                0, 2, 1
            ), sds.reshape(shape3[0], shape3[2], shape3[1]).transpose(0, 2, 1)
            counts = (
                df.groupby(keys, sort=True)
                .size()
                .to_numpy()
                .reshape(shape3[0], shape3[2])
            )
            arr = np.full(shape3 + (int(counts.max()),), np.nan)
            for si, src in enumerate(source_names):
                for fi, lev in enumerate(factor_levels):
                    vals = df[(df[source_col] == src) & (df[factor] == lev)][
                        mix["iso_names"]
                    ].to_numpy(float)
                    arr[si, :, fi, : len(vals)] = vals.T
        else:
            mu, sig = means, sds
            counts = df.groupby(source_col, sort=True).size().to_numpy()
            arr = np.full(shape + (int(counts.max()),), np.nan)
            for si, src in enumerate(source_names):
                vals = df[df[source_col] == src][mix["iso_names"]].to_numpy(float)
                arr[si, :, : len(vals)] = vals.T
        mu_array = sig2_array = n_array = None
        source_array, n_rep = arr, counts
    else:
        required = mix["MU_names"] + mix["SIG_names"] + ["n"]
        missing = [x for x in required if x not in df]
        if missing:
            raise ValueError(f"Source summary columns missing: {missing}")
        means, sds = df[mix["MU_names"]].to_numpy(float), df[mix["SIG_names"]].to_numpy(
            float
        )
        sample_sizes = df["n"].to_numpy(float)
        if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
            raise ValueError("Source SDs must be finite and positive")
        if np.any(~np.isfinite(means)):
            raise ValueError("Source means must be finite")
        if np.any(~np.isfinite(sample_sizes)) or np.any(sample_sizes <= 1):
            raise ValueError("Source sample sizes must be finite and greater than 1")
        if factor:
            nf = len(factor_levels)
            mu_array = means.reshape(shape[0], nf, shape[1]).transpose(0, 2, 1)
            sig2_array = (sds**2).reshape(shape[0], nf, shape[1]).transpose(0, 2, 1)
            n_array = df["n"].to_numpy(float).reshape(shape[0], nf)
        else:
            mu_array, sig2_array, n_array = means, sds**2, df["n"].to_numpy(float)
        mu, sig = means, sds
        source_array = n_rep = None
    s_mu = pd.DataFrame(mu.reshape(-1, shape[1]), columns=mix["iso_names"])
    s_sig = pd.DataFrame(sig.reshape(-1, shape[1]), columns=mix["iso_names"])
    return {
        "n_sources": len(source_names),
        "source_names": source_names,
        "S_MU": s_mu,
        "S_SIG": s_sig,
        "S_factor1": factor_levels or None,
        "S_factor_levels": len(factor_levels) if factor else None,
        "conc": conc,
        "MU_array": mu_array,
        "SIG2_array": sig2_array,
        "n_array": n_array,
        "SOURCE_array": source_array,
        "n_rep": n_rep,
        "by_factor": mix["factors"].index(factor) + 1 if factor else None,
        "data_type": data_type,
        "conc_dep": bool(conc_dep),
    }


def load_discr_data(filename, mix=None):
    if mix is None:
        raise ValueError("mix is required")
    # The first CSV column contains source names.
    df = pd.read_csv(Path(filename), index_col=0).sort_index()
    required = mix["MU_names"] + mix["SIG_names"]
    missing = [x for x in required if x not in df]
    if missing:
        raise ValueError(f"Discrimination columns missing: {missing}")
    mu = df[mix["MU_names"]].astype(float)
    sig = df[mix["SIG_names"]].astype(float)
    if (sig < 0).any().any():
        raise ValueError("Discrimination SDs cannot be negative")
    return {"mu": mu, "sig2": sig**2}
