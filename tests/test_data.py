from pathlib import Path
import numpy as np
import pytest
from mixsiarpy import load_mix_data, load_source_data, load_discr_data, calc_area

DATA = Path(__file__).parents[1] / "mixsiarpy" / "data"


def wolves():
    mix = load_mix_data(
        DATA / "wolves_consumer.csv",
        ["d13C", "d15N"],
        ["Region", "Pack"],
        [True, True],
        [False, True],
    )
    source = load_source_data(
        DATA / "wolves_sources.csv", "Region", False, "means", mix
    )
    discr = load_discr_data(DATA / "wolves_discrimination.csv", mix)
    return mix, source, discr


def test_wolves_shapes_and_nesting():
    mix, source, discr = wolves()
    assert (mix["N"], mix["n_iso"], source["n_sources"]) == (66, 2, 3)
    assert mix["FAC"][1]["lookup"].tolist() == [1, 1, 1, 2, 2, 2, 2, 3]
    assert source["MU_array"].shape == (3, 2, 3)
    assert discr["mu"].shape == (3, 2)
    assert np.all(np.isfinite(calc_area(source, mix, discr)))


def test_mix_validation():
    f = DATA / "wolves_consumer.csv"
    with pytest.raises(ValueError):
        load_mix_data(f, [], [], [], [])
    with pytest.raises(ValueError):
        load_mix_data(f, ["d13C"], ["Region", "Pack"], [True, True], [True, True])


def test_alligator_no_factor():
    mix = load_mix_data(DATA / "alligator_consumer.csv", ["d13C", "d15N"])
    source = load_source_data(
        DATA / "alligator_sources_simplemean.csv", None, False, "means", mix
    )
    discr = load_discr_data(DATA / "alligator_TEF.csv", mix)
    assert source["MU_array"].shape == (2, 2)
    assert calc_area(source, mix, discr) == 0  # two sources form a line, not a polygon


def test_means_source_without_factor_and_concentration():
    mix = load_mix_data(
        DATA / "geese_consumer.csv", ["d15N", "d13C"], ["Group"], [False], [False]
    )
    source = load_source_data(DATA / "geese_sources.csv", None, True, "means", mix)
    assert source["MU_array"].shape == (4, 2)
    assert source["SIG2_array"].shape == (4, 2)
    assert source["n_array"].shape == (4,)
    assert source["SOURCE_array"] is None
    assert source["conc"].shape == (4, 2)
    assert source["source_names"] == sorted(source["source_names"])


def test_means_source_by_factor():
    _, source, _ = wolves()
    assert source["by_factor"] == 1
    assert source["S_factor_levels"] == 3
    assert source["MU_array"].shape == (3, 2, 3)
    assert source["SIG2_array"].shape == (3, 2, 3)
    assert source["n_array"].shape == (3, 3)
    assert np.allclose(source["MU_array"][0, 0], [-26.88, -27.15, -27.47])


def test_raw_source_without_factor_and_with_concentration():
    mix = load_mix_data(DATA / "stormpetrel_consumer.csv", ["d13C", "d15N"])
    source = load_source_data(DATA / "stormpetrel_sources.csv", None, True, "raw", mix)
    assert source["MU_array"] is None
    assert source["SOURCE_array"].shape[:2] == (source["n_sources"], 2)
    assert source["n_rep"].shape == (source["n_sources"],)
    assert source["conc"].shape == (source["n_sources"], 2)
    for i, count in enumerate(source["n_rep"]):
        assert np.isfinite(source["SOURCE_array"][i, :, :count]).all()
        assert np.isnan(source["SOURCE_array"][i, :, count:]).all()


def test_raw_source_by_factor_layout(tmp_path):
    source_file = tmp_path / "raw_factor.csv"
    source_file.write_text(
        "Source,Region,d13C,d15N\n"
        "B,2,-18,11\nB,2,-17,12\nB,1,-20,9\nB,1,-19,10\n"
        "A,2,-26,5\nA,2,-25,6\nA,1,-28,3\nA,1,-27,4\n",
        encoding="utf-8",
    )
    mix, _, _ = wolves()
    source = load_source_data(source_file, "Region", False, "raw", mix)
    assert source["source_names"] == ["A", "B"]
    assert source["SOURCE_array"].shape == (2, 2, 2, 2)
    assert np.array_equal(source["n_rep"], np.full((2, 2), 2))
    assert np.allclose(source["SOURCE_array"][0, 0, 0], [-28, -27])
    assert np.allclose(source["SOURCE_array"][1, 1, 1], [11, 12])


def test_large_n_is_fixed_source_approximation(tmp_path):
    import pandas as pd

    frame = pd.read_csv(DATA / "alligator_sources_simplemean.csv")
    frame["n"] = 1_000_000
    filename = tmp_path / "fixed_approx.csv"
    frame.to_csv(filename, index=False)
    mix = load_mix_data(DATA / "alligator_consumer.csv", ["d13C", "d15N"])
    source = load_source_data(filename, None, False, "means", mix)
    mean_sd = np.sqrt(source["SIG2_array"] / source["n_array"][:, None])
    assert np.max(mean_sd) < 0.003


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("n", 1, "sample sizes"),
        ("SDd13C", -1, "SDs"),
        ("Concd13C", 0, "Concentration"),
    ],
)
def test_invalid_summary_source_parameters(tmp_path, column, value, message):
    import pandas as pd

    frame = pd.read_csv(DATA / "geese_sources.csv")
    frame.loc[0, column] = value
    filename = tmp_path / "invalid.csv"
    frame.to_csv(filename, index=False)
    mix = load_mix_data(
        DATA / "geese_consumer.csv", ["d15N", "d13C"], ["Group"], [False], [False]
    )
    with pytest.raises(ValueError, match=message):
        load_source_data(filename, None, True, "means", mix)


def test_wolves_isospace_preserves_factor_semantics(tmp_path):
    import matplotlib.pyplot as plt
    from mixsiarpy.plotting import plot_data

    mix, source, discr = wolves()
    output = tmp_path / "wolves_isospace.pdf"
    figure = plot_data(mix, source, discr, output)
    axis = figure.axes[0]
    legends = [item for item in axis.get_children()
               if type(item).__name__ == "Legend"]
    labels = {text.get_text() for legend in legends for text in legend.get_texts()}
    assert {"Region 1", "Region 2", "Region 3"} <= labels
    assert {"Pack 1", "Pack 8"} <= labels
    assert {text.get_text() for text in axis.texts} == {
        "Deer", "Marine Mammals", "Salmon"
    }
    assert output.exists()
    assert output.with_suffix(".svg").exists()
    assert output.with_suffix(".png").exists()
    plt.close(figure)
