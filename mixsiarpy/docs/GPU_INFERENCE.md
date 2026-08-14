# GPU inference

MixSIARPy can run the same PyMC probability model with NumPyro or BlackJAX
NUTS on a JAX-visible GPU.

## Recommended platform

Use Linux or WSL2 with an NVIDIA GPU, a working NVIDIA driver, and a
CUDA-enabled JAX installation. Native Windows is suitable for PyMC and Nutpie
CPU sampling; JAX GPU support is normally easier through WSL2.

Installing `mixsiarpy[jax]` installs the Python-side sampler dependencies, but
does not guarantee that the installed JAX wheel supports your CUDA version.
Install CUDA-enabled JAX by following the current JAX installation guide, then
verify the device before starting a long run:

```python
import jax
print(jax.devices())
```

At least one device must report `GpuDevice` or platform `gpu`.

## Run the example

```text
python -m mixsiarpy.examples.wolves_gpu --run test
```

The example explicitly sets:

```python
backend="numpyro"
device="gpu"
```

If no GPU is detected, execution stops rather than silently falling back to
CPU. The result records `inference_backend`, `compute_device`, build time,
sampling time, Python version and platform in `InferenceData.attrs`.

GPU acceleration is workload dependent. Compilation may dominate small
models; report ESS/second as well as wall-clock time when benchmarking.
