"""Integration checks for every Python port of the official R examples."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLE_DIR = ROOT / "mixsiarpy" / "examples"
EXAMPLES = [
    "alligator.py",
    "alligator_length_ind.py",
    "cladocera.py",
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


def test_python_ports_cover_all_official_r_examples():
    r_names = {
        path.stem.removeprefix("mixsiar_script_")
        for path in (ROOT / "mixsiarpy" / "reference_r" / "example_scripts").glob(
            "mixsiar_script_*.R"
        )
    }
    python_names = {Path(name).stem for name in EXAMPLES}
    assert python_names == r_names


@pytest.mark.parametrize("script", EXAMPLES)
def test_official_example_builds_without_sampling(script, tmp_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    command = [sys.executable, str(EXAMPLE_DIR / script)]
    if script == "wolves.py":
        command.append("--build-only")
    # Scripts supporting --output must not write into the repository in tests.
    source = (EXAMPLE_DIR / script).read_text(encoding="utf-8")
    if 'add_argument("--output")' in source:
        command.extend(["--output", str(tmp_path / Path(script).stem)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
