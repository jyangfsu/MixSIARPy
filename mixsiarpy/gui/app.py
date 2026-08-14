"""Interactive, code-free MixSIARPy workflow."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
import os
import shutil
import tempfile
import time
import zipfile

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import scipy.signal as _signal
if not hasattr(_signal, "gaussian"):
    _signal.gaussian = _signal.windows.gaussian
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mixsiarpy import (
    backend_status, build_model, get_resource_path, load_discr_data,
    load_mix_data, load_source_data, run_model, save_results,
)
from mixsiarpy.backends import resolve_backend
from mixsiarpy.gui import write_reproducibility_bundle
from mixsiarpy.plotting import plot_data, plot_prior
from mixsiarpy.model import RUN_PRESETS


ASSET_DIR = Path(__file__).resolve().parent / "assets"
st.set_page_config(
    page_title="MixSIARPy · Bayesian Tracer Mixing",
    page_icon=str(ASSET_DIR / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --ink:#123B4A; --teal:#0B8C84; --mint:#EAF7F4; --sand:#F3B85B; --coral:#E46F61; }
html, body, [class*="css"] { font-family:'DM Sans','Segoe UI',sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(180deg,#F7FBFA 0,#FFFFFF 28rem); }
[data-testid="stHeader"] { background:rgba(247,251,250,.84); backdrop-filter:blur(12px); }
[data-testid="stSidebar"] { background:#113B49; border-right:0; }
[data-testid="stSidebar"] * { color:#F4FAF8; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input { background:#1A4C59 !important; border-color:#3A6973 !important; }
[data-testid="stSidebar"] hr { border-color:#315C66; }
.block-container { max-width:1440px; padding-top:1.7rem; padding-bottom:4rem; }
.hero { display:flex; align-items:center; gap:24px; padding:20px 27px; margin:0 0 22px;
 background:rgba(255,255,255,.82); border:1px solid #DDEBE8; border-radius:22px;
 box-shadow:0 15px 45px rgba(19,59,74,.08); }
.hero img { width:92px; height:92px; }
.hero h1 { margin:0; color:#123B4A; font-size:2.5rem; letter-spacing:-.045em; }
.hero h1 span { color:#0B8C84; }
.hero p { margin:7px 0 0; color:#60767C; font-size:1.02rem; }
.eyebrow { color:#0B8C84; font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:8px; background:#EEF6F4; padding:6px; border-radius:14px; }
div[data-testid="stTabs"] button {
  border-radius:11px;
  padding:.78rem 1.45rem;
  min-height:48px;
  color:#506A71;
  font-size:1.02rem;
  font-weight:750;
  letter-spacing:.025em;
}
div[data-testid="stTabs"] button p {
  font-size:1.02rem !important;
  font-weight:750 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] { background:#fff; color:#0B8C84; box-shadow:0 3px 12px rgba(18,59,74,.09); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #DDEBE8; border-radius:15px; padding:14px 17px; box-shadow:0 7px 20px rgba(18,59,74,.05); }
div[data-testid="stMetricLabel"] { color:#71868B; }
div[data-testid="stMetricValue"] { color:#123B4A; }
.stButton > button, .stDownloadButton > button { border-radius:11px; font-weight:700; min-height:44px; border:1px solid #0B8C84; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:#0B8C84; color:white; box-shadow:0 8px 18px rgba(11,140,132,.22); }
.stButton > button[kind="primary"]:hover { background:#08766F; border-color:#08766F; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #9BCBC5; border-radius:15px; padding:7px; }
div[data-testid="stExpander"] { background:#fff; border-color:#DDEBE8; border-radius:13px; }
div[data-testid="stStatusWidget"] { border-radius:15px; border-color:#BFDCD7; background:#fff; }
.section-note { background:#EAF7F4; border-left:4px solid #0B8C84; border-radius:0 12px 12px 0; padding:13px 16px; color:#31575F; margin:.4rem 0 1.1rem; }
.sidebar-brand { display:flex; align-items:center; gap:10px; padding:4px 0 15px; }
.sidebar-brand img { width:43px; height:43px; }
.sidebar-brand b { font-size:1.15rem; } .sidebar-brand span { color:#8CCEC7; }
h2, h3 { color:#123B4A !important; letter-spacing:-.02em; }
</style>
""", unsafe_allow_html=True)

import base64
_logo64 = base64.b64encode((ASSET_DIR / "mixsiarpy-logo.png").read_bytes()).decode()
st.markdown(f"""
<div class="hero">
  <img src="data:image/png;base64,{_logo64}" alt="MixSIARPy logo"/>
  <div><div class="eyebrow">Open Bayesian mixing platform</div>
  <h1>MixSIAR<span>Py</span></h1>
  <p>From tracer data to reproducible source estimates — no coding required.</p></div>
</div>
""", unsafe_allow_html=True)


def read_upload(upload, fallback=None):
    if upload is None:
        return (pd.read_csv(fallback), Path(fallback).name) if fallback else (None, None)
    return pd.read_csv(upload), upload.name


def save_frame(frame, directory, name):
    path = Path(directory) / name
    frame.to_csv(path, index=False)
    return path


def zip_directory(directory):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in Path(directory).rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return data.getvalue()


def options_for(frame, exclude=()):
    return [x for x in frame.columns if x not in set(exclude)] if frame is not None else []


CSV_DESCRIPTIONS = {
    "diagnostics.csv": "Parameter-level R-hat, ESS, and convergence checks.",
    "summary_all_parameters.csv": "Posterior summary and diagnostics for all model parameters.",
    "summary_source_proportions.csv": "Overall source proportions and credible intervals.",
    "summary_compositional_regression.csv": "ILR coefficients for the generalized compositional regression.",
    "compositional_regression_predictions.csv": "Predicted source proportions across covariate values.",
    "normalized_polygon_area.csv": "Geometric source separation in tracer space; not a convergence measure.",
}


def describe_csv(filename):
    if filename in CSV_DESCRIPTIONS:
        return CSV_DESCRIPTIONS[filename]
    if filename.startswith("summary_p_fac"):
        return "Source proportions for each fixed- or random-effect level."
    if filename == "summary_p_both.csv":
        return "Source proportions for combinations of two categorical effects."
    if filename.startswith("summary_"):
        return "Posterior summary and diagnostics for this model variable."
    return "A result table generated by this analysis."


EXAMPLES = {
    "Crop water uptake (generalized regression)": {
        "mix": "crop_water_consumer.csv", "source": "crop_water_sources.csv",
        "discr": "crop_water_discrimination.csv", "iso": ["delta2H", "delta18O"],
        "source_type": "raw", "formula": "species * biomass + diversity",
        "continuous": ["biomass"],
        "factors": [], "random": [], "nested": [], "source_factor": None,
        "conc": False, "error": "Process × Residual (default)",
    },
    "Wolves": {
        "mix": "wolves_consumer.csv", "source": "wolves_sources.csv",
        "discr": "wolves_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["Region", "Pack"], "random": [True, True],
        "nested": [False, True], "source_factor": "Region", "conc": False,
        "error": "Process × Residual (default)",
    },
    "Wolves (normal preset)": {
        "mix": "wolves_consumer.csv", "source": "wolves_sources.csv",
        "discr": "wolves_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["Region", "Pack"], "random": [True, True],
        "nested": [False, True], "source_factor": "Region", "conc": False,
        "error": "Process × Residual (default)", "run": "normal",
    },
    "Lake": {
        "mix": "lake_consumer.csv", "source": "lake_sources.csv",
        "discr": "lake_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "raw", "formula": "", "continuous": ["Secchi:Mixed"],
        "factors": [], "random": [], "nested": [], "source_factor": None,
        "conc": False, "error": "Residual only",
    },
    "Geese": {
        "mix": "geese_consumer.csv", "source": "geese_sources.csv",
        "discr": "geese_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["Group"], "random": [False], "nested": [False],
        "source_factor": None, "conc": True, "error": "Residual only",
    },
    "Cladocera": {
        "mix": "cladocera_consumer.csv", "source": "cladocera_sources.csv",
        "discr": "cladocera_discrimination.csv", "iso": ["c14.0", "c16.0", "c16.1w9", "c16.1w7", "c16.2w4", "c16.3w3", "c16.4w3", "c17.0", "c18.0", "c18.1w9", "c18.1w7", "c18.2w6", "c18.3w6", "c18.3w3", "c18.4w3", "c18.5w3", "c20.0", "c22.0", "c20.4w6", "c20.5w3", "c22.6w3", "BrFA"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["id"], "random": [False], "nested": [False],
        "source_factor": None, "conc": False, "error": "Process only",
    },
    "Isopod": {
        "mix": "isopod_consumer.csv", "source": "isopod_sources.csv",
        "discr": "isopod_discrimination.csv", "iso": ["c16.4w3", "c18.2w6", "c18.3w3", "c18.4w3", "c20.4w6", "c20.5w3", "c22.5w3", "c22.6w3"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["Site"], "random": [True], "nested": [False],
        "source_factor": None, "conc": False, "error": "Residual only",
    },
    "Palmyra": {
        "mix": "palmyra_consumer.csv", "source": "palmyra_sources.csv",
        "discr": "palmyra_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "raw", "formula": "", "continuous": [],
        "factors": ["Taxa"], "random": [False], "nested": [False],
        "source_factor": None, "conc": False, "error": "Process × Residual (default)",
    },
    "Snail": {
        "mix": "snail_consumer.csv", "source": "snail_sources.csv",
        "discr": "snail_discrimination.csv", "iso": ["d13C"],
        "source_type": "raw", "formula": "", "continuous": [],
        "factors": [], "random": [], "nested": [], "source_factor": None,
        "conc": False, "error": "Process × Residual (default)",
    },
    "Storm petrel": {
        "mix": "stormpetrel_consumer.csv", "source": "stormpetrel_sources.csv",
        "discr": "stormpetrel_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "raw", "formula": "", "continuous": [],
        "factors": ["Region"], "random": [False], "nested": [False],
        "source_factor": None, "conc": False, "error": "Process × Residual (default)",
    },
    "Killer whale": {
        "mix": "killerwhale_consumer.csv", "source": "killerwhale_sources.csv",
        "discr": "killerwhale_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": [], "random": [], "nested": [], "source_factor": None,
        "conc": False, "error": "Process × Residual (default)",
    },
    "Mantis shrimp": {
        "mix": "mantis_consumer.csv", "source": "mantis_source.csv",
        "discr": "mantis_discrimination.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": ["Habitat"], "random": [False], "nested": [False],
        "source_factor": "Habitat", "conc": True, "error": "Process × Residual (default)",
    },
    "Alligator (model comparison baseline)": {
        "mix": "alligator_consumer.csv", "source": "alligator_sources_simplemean.csv",
        "discr": "alligator_TEF.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": [],
        "factors": [], "random": [], "nested": [], "source_factor": None,
        "conc": False, "error": "Process × Residual (default)",
    },
    "Alligator length + individual": {
        "mix": "alligator_consumer.csv", "source": "alligator_sources_simplemean.csv",
        "discr": "alligator_TEF.csv", "iso": ["d13C", "d15N"],
        "source_type": "means", "formula": "", "continuous": ["Length"],
        "factors": ["ID"], "random": [True], "nested": [False],
        "source_factor": None, "conc": False, "error": "Process only",
    },
}

with st.sidebar:
    st.markdown(f'<div class="sidebar-brand"><img src="data:image/png;base64,{_logo64}"/><div><b>MixSIAR<span>Py</span></b><br><small>Analysis workspace</small></div></div>', unsafe_allow_html=True)
    st.markdown("#### Project")
    example_name = st.selectbox(
        "Start from an installed example",
        ["Custom CSV files", *sorted(EXAMPLES, key=str.casefold)],
        key="selected_example",
    )
    example = EXAMPLES.get(example_name)
    if st.session_state.get("configured_example") != example_name:
        # Remove widget state so every model control is repopulated from the
        # newly selected example instead of retaining the previous analysis.
        for widget_key in (
            "gui_isotopes", "gui_factors", "gui_continuous", "gui_formula",
            "gui_source_type", "gui_concentration", "gui_source_factor",
            "gui_error_structure",
        ):
            st.session_state.pop(widget_key, None)
        for key in list(st.session_state):
            if key.startswith(("random_", "nested_")):
                st.session_state.pop(key, None)
        if example and example.get("run"):
            st.session_state["gui_run_preset"] = example["run"]
        st.session_state["configured_example"] = example_name
    st.divider()
    st.markdown("#### Inference")
    statuses = backend_status()
    backend = st.selectbox("Backend", ["auto", *statuses], index=1)
    device = st.selectbox("Device", ["auto", "cpu", "gpu"])
    run = st.selectbox(
        "Run preset", ["test", "very short", "short", "normal", "long", "very long", "extreme"],
        key="gui_run_preset",
    )
    target_accept = st.slider("Target acceptance", 0.80, 0.99, 0.95, 0.01)
    random_seed = st.number_input("Random seed", value=42, step=1)
    if backend != "auto":
        state = statuses[backend]
        (st.success if state["available"] else st.warning)(state["note"])

data_root = get_resource_path("data")
fallback_mix = data_root / example["mix"] if example else None
fallback_source = data_root / example["source"] if example else None
fallback_discr = data_root / example["discr"] if example else None

tab_data, tab_model, tab_run, tab_results = st.tabs(
    ["01  DATA", "02  MODEL", "03  RUN", "04  RESULTS"]
)

with tab_data:
    st.subheader("Input data")
    st.markdown('<div class="section-note">Upload three compatible CSV tables or begin with an installed example. Tables are validated before inference.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    mix_upload = c1.file_uploader("Mixture CSV", type="csv")
    source_upload = c2.file_uploader("Source CSV", type="csv")
    discr_upload = c3.file_uploader("Discrimination CSV", type="csv")
    try:
        mix_frame, mix_name = read_upload(mix_upload, fallback_mix)
        source_frame, source_name = read_upload(source_upload, fallback_source)
        discr_frame, discr_name = read_upload(discr_upload, fallback_discr)
    except Exception as exc:
        st.error(f"Could not read a CSV file: {exc}")
        st.stop()
    if mix_frame is None or source_frame is None or discr_frame is None:
        st.info("Upload all three CSV files, or select an installed example.")
        st.stop()
    for title, frame in (("Mixtures", mix_frame), ("Sources", source_frame), ("Discrimination", discr_frame)):
        with st.expander(f"{title}: {len(frame)} rows"):
            st.dataframe(frame, use_container_width=True)
    default_iso = [x for x in (example or {}).get("iso", []) if x in mix_frame]
    iso_names = st.multiselect(
        "Tracer/isotope columns", mix_frame.columns, default=default_iso,
        key="gui_isotopes",
    )

with tab_model:
    st.subheader("Model specification")
    st.markdown('<div class="section-note">Define source structure, covariates and uncertainty. The same configuration can be exported as executable Python.</div>', unsafe_allow_html=True)
    remaining = options_for(mix_frame, iso_names)
    default_factors = [x for x in (example or {}).get("factors", []) if x in remaining]
    factors = st.multiselect(
        "Categorical effects (maximum 2 for MixSIAR-compatible mode)",
        remaining, default=default_factors, key="gui_factors",
    )
    fac_random, fac_nested = [], []
    example_factors = (example or {}).get("factors", [])
    for factor in factors:
        position = example_factors.index(factor) if factor in example_factors else -1
        default_random = (example or {}).get("random", [])[position] if position >= 0 else False
        default_nested = (example or {}).get("nested", [])[position] if position >= 0 else False
        a, b = st.columns(2)
        fac_random.append(a.checkbox(f"Random effect: {factor}", value=default_random, key=f"random_{factor}"))
        fac_nested.append(b.checkbox(f"Nested effect: {factor}", value=default_nested, key=f"nested_{factor}"))
    numeric = [x for x in remaining if pd.api.types.is_numeric_dtype(mix_frame[x])]
    continuous = st.multiselect(
        "Continuous variables", numeric,
        default=[x for x in (example or {}).get("continuous", []) if x in numeric],
        key="gui_continuous",
    )
    formula = st.text_input(
        "Generalized compositional-regression formula (optional)",
        value=(example or {}).get("formula", ""),
        key="gui_formula",
        help="Examples: depth + temperature; species * biomass + diversity. '*' includes main effects and interaction.",
    ).strip()
    if formula:
        st.code(f"ILR(source proportions) ~ {formula}", language=None)

    c1, c2, c3 = st.columns(3)
    source_type = c1.selectbox(
        "Source data", ["raw", "means"],
        index=0 if (example or {}).get("source_type") == "raw" else 1,
        key="gui_source_type",
    )
    conc_dep = c2.checkbox(
        "Concentration dependence", value=(example or {}).get("conc", False),
        key="gui_concentration",
    )
    source_factor_options = [x for x in factors if x in source_frame.columns]
    default_source_factor = (example or {}).get("source_factor")
    source_factor_choices = ["None", *source_factor_options]
    source_factor = c3.selectbox(
        "Source factor", source_factor_choices,
        index=source_factor_choices.index(default_source_factor) if default_source_factor in source_factor_choices else 0,
        key="gui_source_factor",
    )

    error_label = st.radio(
        "Error structure",
        ["Process × Residual (default)", "Residual only", "Process only"], horizontal=True,
        index=["Process × Residual (default)", "Residual only", "Process only"].index(
            (example or {}).get("error", "Process × Residual (default)")
        ),
        key="gui_error_structure",
    )
    process_err = error_label != "Residual only"
    resid_err = error_label != "Process only"
    alpha_text = st.text_input("Dirichlet prior α", "1", help="Use one value or comma-separated values per source")

    st.subheader("Run-time validation")
    validation_error = None
    work = Path(tempfile.mkdtemp(prefix="mixsiarpy_gui_validate_"))
    try:
        mix_path = save_frame(mix_frame, work, "mixture.csv")
        source_path = save_frame(source_frame, work, "source.csv")
        discr_path = save_frame(discr_frame, work, "discrimination.csv")
        alpha_values = [float(x.strip()) for x in alpha_text.split(",")]
        alpha_prior = alpha_values[0] if len(alpha_values) == 1 else np.asarray(alpha_values)
        mix = load_mix_data(
            mix_path, iso_names, factors, fac_random, fac_nested, continuous,
            composition_formula=formula or None,
        )
        source = load_source_data(
            source_path, None if source_factor == "None" else [source_factor],
            conc_dep, source_type, mix,
        )
        discr = load_discr_data(discr_path, mix)
        model = build_model(mix, source, discr, alpha_prior, process_err, resid_err)
        st.success(
            f"Valid model: {mix['N']} mixtures, {source['n_sources']} sources, "
            f"{mix['n_iso']} tracers."
        )
        if formula:
            st.dataframe(pd.DataFrame({"regression term": mix["regression"]["columns"]}))
    except Exception as exc:
        validation_error = exc
        st.error(f"Model validation failed: {exc}")

with tab_run:
    st.subheader("Run Bayesian inference")
    st.markdown('<div class="section-note">Review computational workload, then monitor live iterations, elapsed time and estimated time remaining.</div>', unsafe_allow_html=True)
    st.warning("The test preset checks the workflow only. Use normal/long and inspect diagnostics for scientific results.")
    run_configuration = RUN_PRESETS[run]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chains", run_configuration["chains"])
    c2.metric("Tune / chain", f"{run_configuration['tune']:,}")
    c3.metric("Draws / chain", f"{run_configuration['draws']:,}")
    c4.metric(
        "Total iterations",
        f"{run_configuration['chains'] * (run_configuration['tune'] + run_configuration['draws']):,}",
    )
    run_clicked = st.button("▶ Run model", type="primary", disabled=validation_error is not None)
    if run_clicked:
        result_dir = Path(tempfile.mkdtemp(prefix="mixsiarpy_gui_result_"))
        for frame, name in ((mix_frame, "mixture.csv"), (source_frame, "source.csv"), (discr_frame, "discrimination.csv")):
            save_frame(frame, result_dir, name)
        config = {
            "mixture_file": "mixture.csv", "source_file": "source.csv",
            "discrimination_file": "discrimination.csv", "iso_names": iso_names,
            "factors": factors, "fac_random": fac_random, "fac_nested": fac_nested,
            "cont_effects": continuous, "composition_formula": formula or None,
            "source_data_type": source_type, "source_factors": [] if source_factor == "None" else [source_factor],
            "conc_dep": conc_dep, "process_err": process_err, "resid_err": resid_err,
            "alpha_prior": alpha_values if len(alpha_values) > 1 else alpha_values[0],
            "run": run, "backend": backend, "device": device,
            "target_accept": target_accept, "random_seed": int(random_seed),
        }
        write_reproducibility_bundle(result_dir, config)
        try:
            started = time.monotonic()
            started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = st.status("Preparing Bayesian analysis…", expanded=True)
            status.write("✓ Input tables validated")
            status.write("✓ PyMC model graph built successfully")
            status.write(
                f"Sampler request: **{backend}** · device: **{device}** · "
                f"target_accept: **{target_accept:.2f}**"
            )
            status.write(
                f"Work: **{run_configuration['chains']} chains × "
                f"({run_configuration['tune']:,} tune + "
                f"{run_configuration['draws']:,} retained draws)**"
            )
            status.write(f"Sampling started: **{started_at}**")
            timer = st.empty()
            progress_bar = st.progress(0.0, text="Waiting for the sampler to start…")
            resolved_backend = resolve_backend(backend, device=device)
            total_iterations = run_configuration["chains"] * (
                run_configuration["tune"] + run_configuration["draws"]
            )
            sampling_progress = {"completed": 0, "last_update": started}

            def pymc_progress_callback(trace, draw):
                # Native PyMC invokes this once after every tuning or retained
                # draw. Assignment is atomic enough for this display-only use.
                sampling_progress["completed"] += 1
                sampling_progress["last_update"] = time.monotonic()

            # Sampling runs in a worker so the Streamlit page can keep its
            # elapsed-time display alive. PyMC's native terminal progress bar
            # is disabled because it cannot be rendered reliably in a browser.
            def sample_analysis():
                extra = {}
                if resolved_backend == "pymc":
                    extra["callback"] = pymc_progress_callback
                return run_model(
                    run, mix, source, discr, alpha_prior=alpha_prior,
                    process_err=process_err, resid_err=resid_err,
                    random_seed=int(random_seed), backend=backend, device=device,
                    target_accept=target_accept, progressbar=False, **extra,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(sample_analysis)
                while not future.done():
                    elapsed = int(time.monotonic() - started)
                    minutes, seconds = divmod(elapsed, 60)
                    completed = min(sampling_progress["completed"], total_iterations)
                    if resolved_backend == "pymc" and completed > 0:
                        fraction = completed / total_iterations
                        # Early iterations include compilation and adaptation;
                        # suppress unstable ETA until at least 2% is complete.
                        if completed >= max(5, int(total_iterations * 0.02)):
                            remaining_seconds = max(
                                0, int(elapsed / fraction - elapsed)
                            )
                            eta_minutes, eta_seconds = divmod(remaining_seconds, 60)
                            eta_text = f"estimated remaining {eta_minutes:02d}:{eta_seconds:02d}"
                        else:
                            eta_text = "estimating remaining time…"
                        progress_bar.progress(
                            fraction,
                            text=(
                                f"{completed:,}/{total_iterations:,} iterations "
                                f"({fraction:.1%}) · {eta_text}"
                            ),
                        )
                        timer.info(
                            f"Sampling is running · elapsed {minutes:02d}:{seconds:02d} · "
                            f"{eta_text}"
                        )
                    else:
                        progress_bar.progress(
                            0.0,
                            text=(
                                "Sampler is running · reliable iteration progress/ETA "
                                f"is not exposed by the {resolved_backend} backend"
                            ),
                        )
                        timer.info(
                            f"Sampling is running · elapsed {minutes:02d}:{seconds:02d} · "
                            f"{resolved_backend} does not expose a reliable live ETA"
                        )
                    time.sleep(1)
                fit = future.result()

            elapsed = time.monotonic() - started
            progress_bar.progress(1.0, text="Sampling complete (100%)")
            timer.success(f"Sampling finished in {elapsed / 60:.2f} minutes")
            status.write(
                f"✓ Sampling completed with **{fit.attrs.get('inference_backend', backend)}** "
                f"on **{fit.attrs.get('compute_device', device)}**"
            )
            status.update(label="Saving summaries, diagnostics and figures…", state="running")
            save_results(fit, result_dir / "results", source["source_names"], mix)
            plot_data(mix, source, discr, result_dir / "results" / "isospace.pdf")
            plt.close("all")
            plot_prior(alpha_prior, source, result_dir / "results" / "prior_distribution.png")
            plt.close("all")
            status.write("✓ Posterior, summaries, diagnostics and figures saved")
            status.update(label=f"Analysis complete · {elapsed / 60:.2f} minutes", state="complete")
            st.session_state["gui_fit"] = fit
            st.session_state["gui_result_dir"] = str(result_dir)
            st.session_state["gui_sources"] = source["source_names"]
            st.success("Complete. Open the Results tab.")
        except Exception as exc:
            st.exception(exc)

with tab_results:
    if "gui_fit" not in st.session_state:
        st.info("Run a model to display results.")
    else:
        fit = st.session_state["gui_fit"]
        result_dir = Path(st.session_state["gui_result_dir"])
        diagnostics_file = result_dir / "results" / "diagnostics.json"
        if diagnostics_file.exists():
            st.subheader("Convergence")
            report = json.loads(diagnostics_file.read_text(encoding="utf-8"))
            converged = bool(report.get("converged", False))
            if converged:
                st.success("The model passed the current automatic convergence checks. Continue by reviewing trace plots and posterior predictive behavior before interpretation.")
            else:
                st.error("The model did not pass all automatic convergence checks. Do not report the posterior results without addressing the issues below.")

            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Parameters checked", report.get("parameters", 0))
            d2.metric("R-hat failures", report.get("rhat_failures", 0))
            d3.metric("Bulk ESS failures", report.get("ess_bulk_failures", 0))
            d4.metric("Tail ESS failures", report.get("ess_tail_failures", 0))
            d5.metric("Divergences", report.get("divergences", 0))

            with st.expander("Diagnostic guide", expanded=False):
                st.markdown(f"""
**Parameters checked:** number of scalar parameters assessed.  
**R-hat failures:** chains disagree; target **R-hat ≤ 1.05**. Increase tuning/draws and inspect traces.  
**Bulk ESS failures:** central posterior estimates are inefficient; target **ESS ≥ 400**. Increase draws.  
**Tail ESS failures:** credible-interval endpoints are unstable; target **ESS ≥ 400**. Increase draws.  
**Divergences:** problematic NUTS transitions; target **0**. Increase `target_accept` and tuning.  
""")
        results_directory = result_dir / "results"
        csv_files = sorted(
            results_directory.glob("*.csv"),
            key=lambda path: path.name.casefold(),
        )
        if csv_files:
            st.subheader("CSV data and result tables")
            st.caption(
                "Every CSV result generated by the analysis is available "
                "for preview and individual download."
            )
            for csv_file in csv_files:
                with st.expander(csv_file.name):
                    st.markdown(describe_csv(csv_file.name))
                    try:
                        table = pd.read_csv(csv_file)
                        st.dataframe(table, use_container_width=True, height=320)
                        st.download_button(
                            f"Download {csv_file.name}",
                            csv_file.read_bytes(),
                            file_name=csv_file.name,
                            mime="text/csv",
                            key=f"download_csv_{csv_file.name}",
                        )
                    except Exception as exc:
                        st.warning(f"Could not preview this CSV: {exc}")
        images = sorted((result_dir / "results").glob("*.png"))
        if images:
            st.subheader("Figures")
            for image in images:
                with st.expander(image.name, expanded=image.name == "posterior_source_proportions.png"):
                    st.image(str(image), use_container_width=True)
        st.download_button(
            "Download complete reproducibility bundle (.zip)",
            zip_directory(result_dir), "mixsiarpy_analysis.zip", "application/zip",
            type="primary",
        )

st.divider()
st.caption("MixSIARPy research preview · GUI and Python API use the same model implementation")
