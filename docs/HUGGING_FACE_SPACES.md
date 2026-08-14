# Deploying MixSIARPy on Hugging Face Spaces

MixSIARPy must use a **Docker Space**, not a Static Space. The GUI is a
Streamlit server and model fitting executes PyMC on the server.

## 1. Create the Space

On Hugging Face, select **New Space** and choose:

- SDK: Docker
- Visibility: Public or Private
- Hardware: CPU Basic for interface testing, CPU Upgrade for practical fits

The root `README.md` already declares `sdk: docker` and `app_port: 7860`.
The root `Dockerfile` installs `mixsiarpy[bayes,gui]` and starts Streamlit on
that port.

## 2. Push the repository

Replace `HF_USERNAME` and `SPACE_NAME` below. Use a Hugging Face user access
token as the Git password; do not put the token in a file or commit it.

```powershell
git remote add space https://huggingface.co/spaces/HF_USERNAME/SPACE_NAME
git push space main
```

If the GitHub default branch is not `main`, push the current branch explicitly:

```powershell
git push space HEAD:main
```

Hugging Face rebuilds and restarts the container after each push.

## 3. Runtime expectations

- CPU Basic is suitable for opening the GUI and running the `test` preset.
- Normal and hierarchical models can be slow on two vCPUs. CPU Upgrade is the
  recommended public configuration.
- The container filesystem is ephemeral. Users should download result bundles
  from the Results tab; files are not guaranteed to survive a restart.
- No JAGS or R installation is required by the public application.
- The default Docker image installs the PyMC CPU backend. GPU/JAX deployment
  requires a separate CUDA image and paid GPU hardware.
- A public Space needs workload limits before broad release because concurrent
  Bayesian fits can exhaust CPU and memory.

## 4. Local container test

```powershell
docker build -t mixsiarpy-space .
docker run --rm -p 7860:7860 mixsiarpy-space
```

Open `http://localhost:7860` and run an installed example with the `test`
preset before pushing.

## 5. Updating the Space

After committing changes locally:

```powershell
git push origin main
git push space main
```

Keep benchmark outputs, posterior NetCDF files, RDS files, manuscript sources,
tokens, and local datasets outside the Space repository/build context.

## 6. GitHub Actions deployment

The repository contains `.github/workflows/sync-to-huggingface.yml`. Add a
GitHub repository secret named `HF_TOKEN` containing a fine-grained Hugging
Face token with write access to `JingHuggingface/MixSIARPy`. Every push to the
GitHub `main` branch will then mirror the repository to the Docker Space. The
workflow can also be started manually from GitHub's Actions tab.
