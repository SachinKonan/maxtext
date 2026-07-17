# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
from typing import Any
import jax
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis

try:
  from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as OrbaxSnapshotter
except (ImportError, ModuleNotFoundError):
  try:
    from orbax.checkpoint.experimental.v1.training.pathways.snapshotter import Snapshotter as OrbaxSnapshotter
  except (ImportError, ModuleNotFoundError):
    OrbaxSnapshotter = object

_logger = logging.getLogger(__name__)


def _unwrap_prng_keys(state: Any) -> Any:
  """Unwraps PRNGKeyArray objects to raw key data for safe host pinning/resharding."""
  def unwrap_leaf(x):
    if type(x).__name__ == "PRNGKeyArray" or hasattr(x, "_base_array"):
      try:
        return jax.random.key_data(x)
      except Exception:
        return x
    return x

  return jax.tree.map(unwrap_leaf, state)


def _wrap_prng_keys(restored_state: Any, abstract_state: Any) -> Any:
  """Wraps raw key data back into PRNGKeyArray matching abstract_state dtypes."""
  def wrap_leaf(restored, abstract):
    if (
        isinstance(restored, jax.Array)
        and (type(abstract).__name__ == "PRNGKeyArray" or hasattr(abstract, "_base_array"))
    ):
      try:
        return jax.random.wrap_key_data(restored, dtype=getattr(abstract, "dtype", None))
      except Exception:
        return restored
    return restored

  return jax.tree.map(wrap_leaf, restored_state, abstract_state)


def is_mesh_shardable_array(x: Any) -> bool:
  """Returns True if x is a concrete JAX Array with a NamedSharding mesh."""
  return isinstance(x, jax.Array) and hasattr(x, "sharding") and hasattr(x.sharding, "mesh")


class Snapshotter(OrbaxSnapshotter):
  """Manages asynchronous backups of JAX array states to pinned host memory, inheriting from Orbax."""

  def __init__(self, *, replica_axis_index: int = 0):
    super().__init__(replica_axis_index=replica_axis_index)

  def save(self, step: int, state: Any) -> None:
    unwrapped_state = _unwrap_prng_keys(state)

    # Find primary mesh to replicate non-mesh arrays/scalars across the full mesh
    primary_mesh = None
    for leaf in jax.tree.leaves(unwrapped_state):
      if is_mesh_shardable_array(leaf):
        primary_mesh = leaf.sharding.mesh
        break

    def replicate_non_mesh_leaf(x):
      if not isinstance(x, jax.Array):
        return x
      if is_mesh_shardable_array(x):
        return x
      elif primary_mesh is not None:
        replicated_sharding = jax.sharding.NamedSharding(
            primary_mesh, jax.sharding.PartitionSpec()
        )
        return jax.device_put(x, replicated_sharding)
      return x

    sharded_state = jax.tree.map(replicate_non_mesh_leaf, unwrapped_state)
    super().save(step, sharded_state)

  def load(
      self,
      abstract_state: Any,
      *,
      reset_snapshot_state: bool = True,
  ) -> Any:
    unwrapped_abstract = _unwrap_prng_keys(abstract_state)

    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def is_replica_active(arr):
      try:
        jax.block_until_ready(arr)
        return True
      except Exception:
        return False

    def get_active_pytree(x):
      if not is_mesh_shardable_array(x):
        return x
      mesh_axis_name = x.sharding.mesh.axis_names[self.replica_axis_index]
      all_replicas = split_by_mesh_axis.split_by_mesh_axis(x, mesh_axis_name)
      active_replicas = [r for r in all_replicas if is_replica_active(r)]
      if not active_replicas:
        raise RuntimeError("No active replicas found.")
      return concatenate_by_mesh_axis.concatenate_by_mesh_axis(active_replicas, mesh_axis_name)

    active_pinned_state = jax.tree.map(get_active_pytree, pinned_state)

    def _device_put_pinned(x, abs_x):
      if isinstance(x, jax.Array) and hasattr(abs_x, "sharding"):
        try:
          return jax.device_put(x, abs_x.sharding.with_memory_kind("pinned_host"))
        except Exception:
          return x
      return x

    host_target_state = jax.tree.map(_device_put_pinned, active_pinned_state, unwrapped_abstract)

    def _device_put_to_device(x, abs_x):
      if isinstance(x, jax.Array) and hasattr(abs_x, "sharding"):
        try:
          return jax.device_put(x, abs_x.sharding.with_memory_kind(None))
        except Exception:
          return x
      return x

    restored_unwrapped = jax.tree.map(_device_put_to_device, host_target_state, unwrapped_abstract)
    jax.block_until_ready(restored_unwrapped)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return _wrap_prng_keys(restored_unwrapped, abstract_state)

  def save_pytree(self, step: int, state: Any) -> None:
    self.save(step, state)

  def load_pytree(self, abstract_state: Any, *, reset_snapshot_state: bool = True) -> Any:
    return self.load(abstract_state, reset_snapshot_state=reset_snapshot_state)

  def join(self) -> None:
    if hasattr(self, "_queue"):
      self._queue.join()
