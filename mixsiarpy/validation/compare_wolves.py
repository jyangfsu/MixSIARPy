"""Compare named R and Python wolves posterior summaries."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).parents[2]
RESULTS = ROOT / "validation" / "results"


def canonical_r(name):
    replacements = {"p.global": "p_global", "p.fac1": "p_fac1",
                    "p.fac2": "p_fac2", "fac1.sig": "fac1_sig",
                    "fac2.sig": "fac2_sig", "resid.prop": "resid_prop"}
    for old, new in replacements.items():
        if name.startswith(old):
            value = new + name[len(old):]
            # R/JAGS uses integer indices; PyMC summaries use coordinate labels.
            sources = {"1": "Deer", "2": "Marine Mammals", "3": "Salmon"}
            if new == "p_global":
                value = re.sub(r"\[(\d+)\]", lambda m: f"[{sources[m.group(1)]}]", value)
            elif new == "p_fac1":
                value = re.sub(
                    r"\[(\d+),(\d+)\]",
                    lambda m: f"[Region {m.group(1)},{sources[m.group(2)]}]", value,
                )
            elif new == "p_fac2":
                value = re.sub(
                    r"\[(\d+),(\d+)\]",
                    lambda m: f"[Pack {m.group(1)},{sources[m.group(2)]}]", value,
                )
            return value
    return name


r = pd.read_csv(RESULTS / "wolves_r" / "summary.csv")
r["canonical"] = r["parameter"].map(canonical_r)
p = pd.read_csv(RESULTS / "wolves_python" / "summary.csv")
p["canonical"] = p["parameter"].str.replace(", ", ",", regex=False)
joined = r.merge(p, on="canonical", suffixes=("_r", "_py"))
joined["mean_difference"] = joined["mean_py"] - joined["mean_r"]
joined["standardized_mean_difference"] = joined["mean_difference"] / (
    (joined["sd_r"] ** 2 + joined["sd_py"] ** 2) ** 0.5
)
joined["interval_overlap"] = (
    joined[["q975", "hdi_97.5%"]].min(axis=1)
    - joined[["q025", "hdi_2.5%"]].max(axis=1)
).clip(lower=0)
joined.to_csv(RESULTS / "wolves_comparison.csv", index=False)
print(joined[["canonical", "mean_r", "mean_py", "standardized_mean_difference",
              "interval_overlap"]].to_string(index=False))
