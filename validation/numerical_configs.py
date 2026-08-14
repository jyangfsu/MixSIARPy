"""Frozen configurations for the 20 numerical-agreement fitted units."""

COMMON = dict(alpha_prior=1, process_err=True, resid_err=True)

CONFIGS = {
    "snail": dict(example="snail", mix="snail_consumer.csv",
        source="snail_sources.csv", discr="snail_discrimination.csv",
        iso=["d13C"], source_type="raw", **COMMON),
    "palmyra": dict(example="palmyra", mix="palmyra_consumer.csv",
        source="palmyra_sources.csv", discr="palmyra_discrimination.csv",
        iso=["d13C", "d15N"], factors=["Taxa"], random=[False], nested=[False],
        source_type="raw", **COMMON),
    "stormpetrel": dict(example="stormpetrel", mix="stormpetrel_consumer.csv",
        source="stormpetrel_sources.csv", discr="stormpetrel_discrimination.csv",
        iso=["d13C", "d15N"], factors=["Region"], random=[False], nested=[False],
        source_type="raw", **COMMON),
    "lake": dict(example="lake", mix="lake_consumer.csv", source="lake_sources.csv",
        discr="lake_discrimination.csv", iso=["d13C", "d15N"],
        continuous=["Secchi:Mixed"], source_type="raw",
        alpha_prior=1, process_err=False, resid_err=True),
    "geese": dict(example="geese", mix="geese_consumer.csv", source="geese_sources.csv",
        discr="geese_discrimination.csv", iso=["d13C", "d15N"],
        factors=["Group"], random=[False], nested=[False], source_type="means",
        conc_dep=True, alpha_prior=1, process_err=False, resid_err=True),
    "isopod": dict(example="isopod", mix="isopod_consumer.csv", source="isopod_sources.csv",
        discr="isopod_discrimination.csv",
        iso=["c16.4w3", "c18.2w6", "c18.3w3", "c18.4w3", "c20.4w6",
             "c20.5w3", "c22.5w3", "c22.6w3"], factors=["Site"],
        random=[True], nested=[False], source_type="means",
        alpha_prior=1, process_err=False, resid_err=True),
    "cladocera": dict(example="cladocera", mix="cladocera_consumer.csv",
        source="cladocera_sources.csv", discr="cladocera_discrimination.csv",
        iso=["c14.0", "c16.0", "c16.1w9", "c16.1w7", "c16.2w4", "c16.3w3",
             "c16.4w3", "c17.0", "c18.0", "c18.1w9", "c18.1w7", "c18.2w6",
             "c18.3w6", "c18.3w3", "c18.4w3", "c18.5w3", "c20.0", "c22.0",
             "c20.4w6", "c20.5w3", "c22.6w3", "BrFA"], factors=["id"],
        random=[False], nested=[False], source_type="means",
        alpha_prior=1, process_err=True, resid_err=False),
    "wolves": dict(example="wolves", mix="wolves_consumer.csv", source="wolves_sources.csv",
        discr="wolves_discrimination.csv", iso=["d13C", "d15N"],
        factors=["Region", "Pack"], random=[True, True], nested=[False, True],
        source_factor="Region", source_type="means", **COMMON),
    "killerwhale_uninformative": dict(example="killerwhale", mix="killerwhale_consumer.csv",
        source="killerwhale_sources.csv", discr="killerwhale_discrimination.csv",
        iso=["d13C", "d15N"], source_type="means", **COMMON),
    "killerwhale_informative": dict(example="killerwhale", mix="killerwhale_consumer.csv",
        source="killerwhale_sources.csv", discr="killerwhale_discrimination.csv",
        iso=["d13C", "d15N"], source_type="means",
        alpha_prior=[3.5714285714, 0.3571428571, 0.01, 0.01, 1.0714285714],
        process_err=True, resid_err=True),
    "mantis": dict(example="mantis", mix="mantis_consumer.csv", source="mantis_source.csv",
        discr="mantis_discrimination.csv", iso=["d13C", "d15N"],
        factors=["Habitat"], random=[False], nested=[False], source_factor="Habitat",
        source_type="means", conc_dep=True,
        alpha_prior=[0.4, 0.4, 1.6, 1.6, 0.4, 1.6], process_err=True, resid_err=True),
    "alligator_length_ind": dict(example="alligator_length_ind",
        mix="alligator_consumer.csv", source="alligator_sources_simplemean.csv",
        discr="alligator_TEF.csv", iso=["d13C", "d15N"], factors=["ID"],
        random=[True], nested=[False], continuous=["Length"], source_type="means",
        alpha_prior=1, process_err=True, resid_err=False),
}

_ALLIGATOR = [
    ("alligator_01", None, None),
    ("alligator_02", ["habitat"], None),
    ("alligator_03", ["sex"], None),
    ("alligator_04", ["sclass"], None),
    ("alligator_05", None, ["Length"]),
    ("alligator_06", ["sex", "sclass"], None),
    ("alligator_07", ["sex"], ["Length"]),
    ("alligator_08", ["sex_sclass"], None),
]
for unit, factors, continuous in _ALLIGATOR:
    cfg = dict(example="alligator", mix="alligator_consumer.csv",
        source="alligator_sources_simplemean.csv", discr="alligator_TEF.csv",
        iso=["d13C", "d15N"], source_type="means", **COMMON)
    if factors:
        cfg.update(factors=factors, random=[False] * len(factors),
                   nested=[False] * len(factors))
    if continuous:
        cfg["continuous"] = continuous
    CONFIGS[unit] = cfg

ORDER = [
    "snail", "killerwhale_uninformative", "killerwhale_informative", "palmyra",
    "stormpetrel", "geese", "lake", "isopod", "cladocera", "mantis",
    "alligator_01", "alligator_02", "alligator_03", "alligator_04",
    "alligator_05", "alligator_06", "alligator_07", "alligator_08",
    "alligator_length_ind", "wolves",
]
