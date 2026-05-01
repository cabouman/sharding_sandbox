import os
import subprocess


def _gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return len([line for line in out.decode().splitlines() if line.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


# If fewer than 2 GPUs are available, force JAX to expose 2 CPU devices.
# This must run BEFORE jax is imported, since XLA reads XLA_FLAGS at backend init.
USE_CPU_FALLBACK = _gpu_count() < 2
if USE_CPU_FALLBACK:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()
