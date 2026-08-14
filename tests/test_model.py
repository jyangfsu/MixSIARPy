from test_data import wolves
from mixsiarpy.model import build_model, run_model
import numpy as np
import pytest
from pathlib import Path


def test_build_wolves_model():
    mix, source, discr = wolves()
    model = build_model(mix, source, discr)
    assert "p_global" in model.named_vars
    assert "p_ind" in model.named_vars
    assert "ilr_global" in model.named_vars
    assert "ilr_fac1" in model.named_vars and "p_fac1" in model.named_vars
    assert "ilr_fac2" in model.named_vars and "p_fac2" in model.named_vars
    assert "p_both" in model.named_vars
    assert "src_mu" in model.named_vars and "src_tmp_x" in model.named_vars


def test_tiny_sampling():
    mix, source, discr = wolves()
    result = run_model(
        {"draws": 5, "tune": 5, "chains": 1},
        mix,
        source,
        discr,
        random_seed=4,
        progressbar=False,
        cores=1,
    )
    assert result.posterior["p_global"].shape[-1] == 3


def test_fixed_effect_uses_reference_level():
    from pathlib import Path
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    mix = load_mix_data(
        data / "geese_consumer.csv", ["d13C", "d15N"], ["Group"], [False], [False]
    )
    source = load_source_data(data / "geese_sources.csv", None, True, "means", mix)
    discr = load_discr_data(data / "geese_discrimination.csv", mix)
    model = build_model(mix, source, discr, process_err=False, resid_err=True)
    assert model.named_vars["ilr_fac1_free"].eval().shape == (
        mix["FAC"][0]["levels"] - 1,
        source["n_sources"] - 1,
    )


def test_raw_sources_are_likelihood_nodes():
    from pathlib import Path
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    mix = load_mix_data(data / "snail_consumer.csv", ["d13C"])
    source = load_source_data(data / "snail_sources.csv", None, False, "raw", mix)
    discr = load_discr_data(data / "snail_discrimination.csv", mix)
    model = build_model(mix, source, discr)
    assert any(name.startswith("source_obs_") for name in model.named_vars)


def test_combine_sources_and_continuous_prediction():
    import arviz as az
    import numpy as np
    from mixsiarpy import combine_sources, continuous_effect_prediction

    posterior = {
        "p_global": np.array([[[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]]),
        "ilr_global": np.zeros((1, 2, 2)),
        "ilr_cont1": np.ones((1, 2, 2)) * 0.1,
    }
    fit = az.from_dict(
        posterior=posterior,
        coords={"source": ["a", "b", "c"], "ilr": [0, 1]},
        dims={"p_global": ["source"], "ilr_global": ["ilr"], "ilr_cont1": ["ilr"]},
    )
    combined = combine_sources(fit, {"ab": ["a", "b"], "c": ["c"]})
    assert np.allclose(combined.sum("combined_source"), 1)
    mix = {
        "n_ce": 1,
        "CE_orig": [np.array([0.0, 2.0])],
        "CE_center": [1.0],
        "CE_scale": [1.0],
    }
    pred = continuous_effect_prediction(
        fit, mix, source_names=["a", "b", "c"], values=[0, 1, 2]
    )
    assert pred.shape == (1, 2, 3, 3)
    assert np.allclose(pred.sum("source"), 1)


def test_random_effect_priors_and_shapes():
    mix, source, discr = wolves()
    model = build_model(mix, source, discr)
    # PyTensor stores Uniform bounds as the final two random-variable inputs.
    assert [
        float(x.eval()) for x in model.named_vars["fac1_sig"].owner.inputs[-2:]
    ] == [0, 20]
    assert [
        float(x.eval()) for x in model.named_vars["fac2_sig"].owner.inputs[-2:]
    ] == [0, 20]
    assert model.named_vars["ilr_fac1"].eval().shape == (3, 2)
    assert model.named_vars["ilr_fac2"].eval().shape == (8, 2)
    assert model.named_vars["p_fac1"].eval().shape == (3, 3)
    assert model.named_vars["p_fac2"].eval().shape == (8, 3)
    assert model.named_vars["p_both"].eval().shape == (3, 8, 3)


def test_fixed_and_random_effect_are_reordered_like_r():
    from pathlib import Path
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    # Deliberately supply random first; MixSIAR moves the fixed effect to factor 1.
    mix = load_mix_data(
        data / "alligator_consumer.csv",
        ["d13C", "d15N"],
        ["ID", "sex"],
        [True, False],
        [False, False],
    )
    assert mix["factors"] == ["sex", "ID"]
    assert mix["fac_random"] == [False, True]
    assert not mix["FAC"][0]["re"] and mix["FAC"][1]["re"]
    source = load_source_data(
        data / "alligator_sources_simplemean.csv", None, False, "means", mix
    )
    discr = load_discr_data(data / "alligator_TEF.csv", mix)
    model = build_model(mix, source, discr)
    fixed = model.named_vars["ilr_fac1"].eval()
    assert np.allclose(fixed[0], 0)
    assert model.named_vars["ilr_fac1_free"].eval().shape == (1, 1)
    assert model.named_vars["ilr_fac2"].eval().shape == (mix["FAC"][1]["levels"], 1)
    assert model.named_vars["p_both"].eval().shape == (2, mix["FAC"][1]["levels"], 2)


def test_continuous_effect_standardization_and_model_shape():
    from pathlib import Path
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    mix = load_mix_data(
        data / "alligator_consumer.csv", ["d13C", "d15N"], None, None, None, ["Length"]
    )
    assert np.isclose(np.mean(mix["CE"][0]), 0)
    assert np.isclose(np.std(mix["CE"][0], ddof=1), 1)
    assert np.allclose(
        mix["CE_orig"][0],
        np.asarray(mix["CE"][0]) * mix["CE_scale"][0] + mix["CE_center"][0],
    )
    source = load_source_data(
        data / "alligator_sources_simplemean.csv", None, False, "means", mix
    )
    discr = load_discr_data(data / "alligator_TEF.csv", mix)
    model = build_model(mix, source, discr)
    assert model.named_vars["ilr_cont1"].eval().shape == (1,)
    assert model.named_vars["p_ind"].eval().shape == (mix["N"], 2)


def test_nested_effect_lookup_and_parent_offset():
    mix, source, discr = wolves()
    assert mix["fac_nested"] == [False, True]
    assert np.array_equal(mix["FAC"][1]["lookup"], [1, 1, 1, 2, 2, 2, 2, 3])
    model = build_model(mix, source, discr)
    # At any sampled point the nested factor-level proportions remain compositions.
    assert np.allclose(model.named_vars["p_fac2"].eval().sum(axis=-1), 1)


def test_effect_input_validation():
    from pathlib import Path
    from mixsiarpy import load_mix_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data" / "alligator_consumer.csv"
    with pytest.raises(ValueError, match="at most two"):
        load_mix_data(
            data,
            ["d13C", "d15N"],
            ["sex", "sclass", "habitat"],
            [False, False, False],
            [False, False, False],
        )
    with pytest.raises(ValueError, match="at most one continuous"):
        load_mix_data(data, ["d13C", "d15N"], None, None, None, ["Length", "year"])


def test_generalized_compositional_regression_design_and_model():
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    mix = load_mix_data(
        data / "crop_water_consumer.csv",
        ["delta2H", "delta18O"],
        cont_effects=["biomass"],
        composition_formula="species * biomass + diversity",
    )
    assert mix["regression"]["matrix"].shape == (mix["N"], 12)
    assert "species_Bean:biomass" in mix["regression"]["columns"]
    source = load_source_data(
        data / "crop_water_sources.csv", None, False, "raw", mix
    )
    discr = load_discr_data(data / "crop_water_discrimination.csv", mix)
    model = build_model(mix, source, discr)
    assert model.named_vars["ilr_beta"].eval().shape == (12, 2)
    assert "ilr_cont1" not in model.named_vars


@pytest.mark.parametrize(
    "process_err,resid_err,required,forbidden",
    [
        (True, True, {"resid_prop", "mix"}, {"Sigma"}),
        (False, True, {"Sigma", "loglik", "mix"}, {"resid_prop"}),
        (True, False, {"mix"}, {"Sigma", "resid_prop"}),
    ],
)
def test_three_error_structure_graphs(process_err, resid_err, required, forbidden):
    mix, source, discr = wolves()
    model = build_model(
        mix, source, discr, process_err=process_err, resid_err=resid_err
    )
    names = set(model.named_vars)
    assert required <= names
    assert not (forbidden & names)


def test_multiplicative_error_prior_matches_r():
    mix, source, discr = wolves()
    model = build_model(mix, source, discr, process_err=True, resid_err=True)
    resid = model.named_vars["resid_prop"]
    assert [float(x.eval()) for x in resid.owner.inputs[-2:]] == [0, 20]
    assert resid.eval().shape == (mix["n_iso"],)


def test_residual_only_single_isotope_uses_gamma_precision():
    from pathlib import Path
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    mix = load_mix_data(data / "snail_consumer.csv", ["d13C"])
    source = load_source_data(data / "snail_sources.csv", None, False, "raw", mix)
    discr = load_discr_data(data / "snail_discrimination.csv", mix)
    model = build_model(mix, source, discr, process_err=False, resid_err=True)
    assert type(model.named_vars["Sigma"].owner.op).__name__ == "GammaRV"


def test_invalid_error_combinations_and_single_observation(tmp_path):
    mix, source, discr = wolves()
    with pytest.raises(ValueError, match="At least one"):
        build_model(mix, source, discr, process_err=False, resid_err=False)

    # MixSIAR permits only the process-error (MixSIR) structure for N=1.
    import pandas as pd

    one = pd.read_csv(
        Path(__file__).parents[1] / "mixsiarpy" / "data" / "alligator_consumer.csv"
    ).iloc[:1]
    filename = tmp_path / "one_consumer.csv"
    one.to_csv(filename, index=False)
    from mixsiarpy import load_mix_data, load_source_data, load_discr_data

    data = Path(__file__).parents[1] / "mixsiarpy" / "data"
    one_mix = load_mix_data(filename, ["d13C", "d15N"])
    one_source = load_source_data(
        data / "alligator_sources_simplemean.csv", None, False, "means", one_mix
    )
    one_discr = load_discr_data(data / "alligator_TEF.csv", one_mix)
    with pytest.raises(ValueError, match="single mixture"):
        build_model(one_mix, one_source, one_discr, process_err=True, resid_err=True)
    process_model = build_model(
        one_mix, one_source, one_discr, process_err=True, resid_err=False
    )
    assert "mix" in process_model.named_vars
