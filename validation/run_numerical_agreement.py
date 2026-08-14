"""Resumable adaptive R/JAGS versus PyMC CPU benchmark orchestrator."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from numerical_configs import CONFIGS, ORDER

R_EXE = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
DEFAULT_OUT = ROOT / "validation" / "numerical_agreement_v2"

# R values are the exact official MixSIAR presets.  The normal round retains
# 1,000 draws per chain for both engines.  Later rounds are engine-appropriate
# extensions and are compared using ESS/s, not raw iteration counts.
ROUNDS = {
    "normal": dict(chains=3, chain_length=100000, burn=50000, thin=50,
                   py_chains=3, py_tune=5000, py_draws=1000),
    "long": dict(chains=3, chain_length=300000, burn=200000, thin=100,
                 py_chains=3, py_tune=10000, py_draws=3000),
    "very_long": dict(chains=3, chain_length=1000000, burn=500000, thin=500,
                      py_chains=4, py_tune=20000, py_draws=5000),
}
SMOKE = dict(chains=2, chain_length=1000, burn=500, thin=5,
             py_chains=2, py_tune=100, py_draws=100)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_performance_summary(out):
    rows = []
    for path in out.glob("*/*/*/metadata.json"):
        m = read_json(path)
        if not m:
            continue
        rows.append({
            "unit": m.get("unit"), "round": path.parents[1].name,
            "engine": path.parent.name, "convergence_status": m.get("convergence_status"),
            "elapsed_seconds": m.get("elapsed_seconds"), "chains": m.get("chains"),
            "retained_draws_per_chain": m.get("retained_draws_per_chain"),
            "max_rhat": m.get("max_rhat"), "min_ess_bulk": m.get("min_ess_bulk"),
            "min_ess_tail": m.get("min_ess_tail"),
            "min_bulk_ess_per_second": m.get("min_bulk_ess_per_second"),
            "divergences": m.get("divergences"), "treedepth_hits": m.get("treedepth_hits"),
            "min_bfmi": m.get("min_bfmi"),
        })
    rows.sort(key=lambda x: (ORDER.index(x["unit"]), x["round"], x["engine"]))
    columns = list(rows[0]) if rows else ["unit", "round", "engine"]
    with (out / "performance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


parser = ArgumentParser()
parser.add_argument("--units", nargs="*", choices=ORDER)
parser.add_argument("--rounds", nargs="*", choices=list(ROUNDS), default=list(ROUNDS))
parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
units = args.units or ORDER
out_root = args.output_root.resolve()
out_root.mkdir(parents=True, exist_ok=True)

for unit in units:
    for engine in ("r", "python"):
        for round_name in (["smoke"] if args.smoke else args.rounds):
            sampling = SMOKE if args.smoke else ROUNDS[round_name]
            base = out_root / unit / round_name
            target = base / engine
            target.mkdir(parents=True, exist_ok=True)
            frozen = {"unit": unit, "round": round_name, "model": CONFIGS[unit],
                      "sampling": sampling}
            config = base / "benchmark_config.json"
            config.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
            done = target / "DONE"
            if done.exists() and not args.force:
                metadata = read_json(target / "metadata.json") or {}
            else:
                if engine == "r":
                    cmd = [str(R_EXE), str(ROOT / "validation" / "numerical_runner_r.R"),
                           "--unit", unit, "--config", str(config), "--output", str(target),
                           "--chains", str(sampling["chains"]),
                           "--chain-length", str(sampling["chain_length"]),
                           "--burn", str(sampling["burn"]), "--thin", str(sampling["thin"])]
                else:
                    preset = "normal" if args.smoke else round_name.replace("_", " ")
                    cmd = [sys.executable, str(ROOT / "validation" / "numerical_runner_python.py"),
                           unit, "--preset", preset, "--round-name", round_name,
                           "--output-root", str(out_root),
                           "--chains", str(sampling["py_chains"]),
                           "--tune", str(sampling["py_tune"]),
                           "--draws", str(sampling["py_draws"])]
                if args.force:
                    cmd.append("--force")
                with (target / "stdout.log").open("w", encoding="utf-8") as stdout, \
                     (target / "stderr.log").open("w", encoding="utf-8") as stderr:
                    returncode = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
                metadata = read_json(target / "metadata.json") or {}
                event = {"time": datetime.now().isoformat(), "unit": unit,
                         "round": round_name, "engine": engine,
                         "returncode": returncode,
                         "convergence_status": metadata.get("convergence_status")}
                with (out_root / "orchestrator.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(event) + "\n")
                print(f"{unit}/{engine}/{round_name}: "
                      f"{'FAILED' if returncode else metadata.get('convergence_status', 'DONE')}",
                      flush=True)
                write_performance_summary(out_root)
                if returncode:
                    break
            if args.smoke or metadata.get("convergence_status") != "NEEDS_LONGER_RUN":
                break

write_performance_summary(out_root)
