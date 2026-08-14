import numpy as np
from scipy.spatial import ConvexHull


def calc_area(source, mix, discr):
    if mix["n_iso"] != 2:
        raise ValueError("Normalized polygon area requires exactly two tracers")
    dmu, dvar = discr["mu"].to_numpy(), discr["sig2"].to_numpy()
    if source["by_factor"] is None:
        points = source["S_MU"].to_numpy() + dmu
        spread = np.sqrt(source["S_SIG"].to_numpy() ** 2 + dvar)
        if len(points) < 3:
            return 0.0
        return float(ConvexHull(points).volume / np.prod(spread.mean(axis=0)))
    values = []
    for level in range(source["S_factor_levels"]):
        points = source["MU_array"][:, :, level] + dmu
        spread = np.sqrt(source["SIG2_array"][:, :, level] + dvar)
        values.append(ConvexHull(points).volume / np.prod(spread.mean(axis=0)))
    return np.asarray(values)
