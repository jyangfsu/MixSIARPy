"""Command-line launcher and reproducibility helpers for the web GUI."""

from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    """Launch the installed Streamlit application in the default browser."""
    _configure_ca_bundle()
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The GUI dependencies are not installed. Run: "
            "python -m pip install 'mixsiarpy[gui,bayes]'"
        ) from exc
    bootstrap = Path(__file__).resolve().parent / "bootstrap.py"
    raise SystemExit(subprocess.call([sys.executable, str(bootstrap)]))


def _configure_ca_bundle():
    """Avoid malformed Windows certificate-store entries in some Conda builds."""
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


def write_reproducibility_bundle(directory, config):
    """Write JSON configuration and a standalone, editable Python analysis."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(config, default=str))
    (directory / "analysis_config.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "reproduce_analysis.py").write_text(
        _render_script(clean), encoding="utf-8"
    )
    (directory / "environment.txt").write_text(
        f"python={sys.version.split()[0]}\nplatform={sys.platform}\n", encoding="utf-8"
    )
    return directory / "reproduce_analysis.py"


def _render_script(c):
    formula = repr(c.get("composition_formula") or None)
    return f'''"""Automatically exported by the MixSIARPy GUI."""
from pathlib import Path
from mixsiarpy import load_mix_data, load_source_data, load_discr_data, run_model, save_results

HERE = Path(__file__).resolve().parent
mix = load_mix_data(
    HERE / {c['mixture_file']!r}, iso_names={c['iso_names']!r},
    factors={c.get('factors', [])!r}, fac_random={c.get('fac_random', [])!r},
    fac_nested={c.get('fac_nested', [])!r}, cont_effects={c.get('cont_effects', [])!r},
    composition_formula={formula},
)
source = load_source_data(
    HERE / {c['source_file']!r}, source_factors={c.get('source_factors', [])!r},
    conc_dep={c.get('conc_dep', False)!r}, data_type={c.get('source_data_type', 'raw')!r}, mix=mix,
)
discr = load_discr_data(HERE / {c['discrimination_file']!r}, mix)
fit = run_model(
    {c.get('run', 'test')!r}, mix, source, discr,
    alpha_prior={c.get('alpha_prior', 1)!r}, process_err={c.get('process_err', True)!r},
    resid_err={c.get('resid_err', True)!r}, random_seed={c.get('random_seed', 42)!r},
    backend={c.get('backend', 'pymc')!r}, device={c.get('device', 'auto')!r},
    target_accept={c.get('target_accept', 0.95)!r},
)
save_results(fit, HERE / "results", source["source_names"], mix)
'''
