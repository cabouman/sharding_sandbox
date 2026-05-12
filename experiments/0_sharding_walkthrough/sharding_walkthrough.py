"""Walkthrough of the four core JAX sharding primitives:
  Mesh           — names a logical grid over physical devices
  PartitionSpec  — declares which mesh axis shards which array axis
  NamedSharding  — pairs a Mesh with a PartitionSpec
  jax.device_put — actually places the array on the devices

We make an array on the host, shard it across two devices, then run a simple
multiplication and visualize each step."""

from utils import USE_CPU_FALLBACK, visualize_with_values

import numpy as np
import jax
import jax.numpy as jnp
from jax import sharding
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from rich.console import Console


console = Console()
platform = "cpu" if USE_CPU_FALLBACK else "gpu"


# Step 1 — pick the first 2 devices on the host.
devices = jax.devices(platform)[:2]
console.print(f"\n[bold cyan]Step 1[/]  {len(devices)} {platform} devices: {devices}")


# Step 2 — create a (4, 2) array.
A = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
console.print("\n[bold cyan]Step 2[/]  Create A on the host (uncommitted — JAX hasn't pinned it).")
visualize_with_values(A, title="A on host")


# Step 3 — build a 1D Mesh.
# The mesh can be multi-dimensional, but here we only use a 1D mesh, and we name it "x".
axis_name = "x"
mesh = Mesh(np.array(devices), (axis_name,))
console.print(f"\n[bold cyan]Step 3[/]  Mesh: {mesh}")
console.print(f"        ({axis_name},) is the axis name; the mesh has 2 devices along it.")


# Step 4 — declare a PartitionSpec.
# This determines which axis will be used for sharding.
spec = PartitionSpec(axis_name, None)
console.print(f"\n[bold cyan]Step 4[/]  PartitionSpec: {spec}")
console.print("        'x' on axis 0 → shard A's rows along the x mesh axis.")
console.print("        None on axis 1 → replicate A's columns.")


# Step 5 — combine into a NamedSharding.
# Pairs together a Mesh with a PartitionSpec.
sharding = NamedSharding(mesh, spec)
console.print(f"\n[bold cyan]Step 5[/]  NamedSharding: {sharding}")


# Step 6 — actually move the bytes onto the devices.
# The new array is sharded across the devices along the x axis.
A_sharded = jax.device_put(A, sharding)
console.print("\n[bold cyan]Step 6[/]  jax.device_put moves the data from host to devices.")
visualize_with_values(A_sharded, title="A sharded across devices")


# Step 7 — multiply. The op runs locally on each shard, no data movement needed.
# Note that the result is still sharded the same way.
result = A_sharded * 2
console.print("\n[bold cyan]Step 7[/]  result = A_sharded * 2 — element-wise, runs locally on each shard.")
visualize_with_values(result, title="A * 2 (still sharded the same way)")
