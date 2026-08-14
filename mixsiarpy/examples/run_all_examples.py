"""Build every official example model without sampling."""

import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
EXAMPLES = [
    "alligator.py",
    "alligator_length_ind.py",
    "cladocera.py",
    "crop_water_uptake.py",
    "geese.py",
    "isopod.py",
    "killerwhale.py",
    "lake.py",
    "mantis.py",
    "palmyra.py",
    "snail.py",
    "stormpetrel.py",
    "wolves.py",
    "wolves_normal.py",
]
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="pymc")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    options = parser.parse_args()
    for script in EXAMPLES:
        print(f"\n=== {script} ===")
        subprocess.run(
            [sys.executable, str(HERE / script), "--backend", options.backend,
             "--device", options.device], check=True, cwd=HERE.parent
        )
