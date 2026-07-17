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

"""Multihost dataloading utilities."""

from collections.abc import Iterable, Sequence
from functools import partial
import json
from etils import epath
import numpy as np

import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax.tree_util as jtu

from maxtext.utils import max_logging


class MultiHostDataLoadIterator:
  """Wrapper for MultiHostDataLoadIterator that handles device placement."""

  def __init__(self, dataloader, global_mesh, generate_padding_batch_eval=False):
    self.dataloader = dataloader
    self.global_mesh = global_mesh
    self.generate_padding_batch_eval = generate_padding_batch_eval

    if hasattr(dataloader, "as_numpy_iterator"):
      self.iterator = dataloader.as_numpy_iterator()
    elif isinstance(dataloader, Iterable):
      self.iterator = iter(dataloader)
    else:
      raise ValueError("Type error: dataloader should be Iterable.")

  def reset(self):
    self.iterator.reset()  # pyrefly: ignore[missing-attribute]

  def __iter__(self):
    return self

  def __next__(self):
    try:
      local_data = next(self.iterator)
    except StopIteration:
      if self.generate_padding_batch_eval:
        local_data = self.dataloader.get_padding_batch()
      else:
        raise StopIteration  # pylint: disable=raise-missing-from

    def form_global_array(path, array, mesh):
      # We need to construct local device arrays for jax.make_array_from_single_device_arrays
      # by spliting the host array along the first axis.
      # When local_device_count is 1, local_data is directly the single device array.
      # When local_device_count > 1, local_data is a host array that needs to be split.
      if len(mesh.local_devices) == 1:
        device_arrays = [array]
      else:
        try:
          device_arrays = np.split(array, len(mesh.local_devices), axis=0)
        except ValueError as array_split_error:
          raise ValueError(
              f"Unable to put to devices shape {array.shape} with "
              f"local device count {len(mesh.local_devices)} "
              f"at {jtu.keystr(path)}"
          ) from array_split_error
      device_arrays = jax.device_put(device_arrays, mesh.local_devices)
      global_shape = (array.shape[0] * jax.process_count(), *array.shape[1:])
      return jax.make_array_from_single_device_arrays(
          shape=global_shape,
          sharding=NamedSharding(mesh, PartitionSpec(mesh.axis_names)),
          arrays=device_arrays,
      )

    return jtu.tree_map_with_path(
        partial(form_global_array, mesh=self.global_mesh),
        local_data,
    )

  def save_state(self, step):
    self.iterator.save(step)  # pyrefly: ignore[missing-attribute]

  def restore_state(self, step):
    self.iterator.restore(step)  # pyrefly: ignore[missing-attribute]


def _colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  return colocated_python.colocated_cpu_devices(devices)


def _colocated_cpu_mesh(mesh: Mesh) -> Mesh:
  """Returns a CPU mesh that has colocated CPU devices."""
  return colocated_python.colocated_cpu_devices(mesh)


class RemoteIterator:
  "iterator class for using colocated python class"

  def __init__(self, get_ds_fn, preprocessing_fn, global_shape, checkpoint_path, elastic=False):
    self.get_ds_fn = get_ds_fn
    self.preprocessing_fn = preprocessing_fn
    self.global_shape = global_shape
    self.checkpoint_path = checkpoint_path
    self.elastic = elastic
    self.reset()
    max_logging.log("RemoteIterator initiated")

  def reset(self):
    ds = self.get_ds_fn(dataloading_host_index=jax.process_index(), dataloading_host_count=jax.process_count())
    dataloader = self.preprocessing_fn(dataset=ds)
    if hasattr(dataloader, "as_numpy_iterator"):
      self.iterator = dataloader.as_numpy_iterator()
    elif isinstance(dataloader, Iterable):
      self.iterator = iter(dataloader)
    else:
      raise ValueError("Type error: dataloader should be Iterable.")

  def get_next(self, dummy_array):
    """Gets the next batch of data and forms a global array."""
    local_data = next(self.iterator)

    def form_global_array_colocated_python(path, array, devices, global_shape, sharding):
      try:
        device_arrays = np.split(array, len(devices), axis=0)
      except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error
      device_arrays = jax.device_put(device_arrays, devices)
      return jax.make_array_from_single_device_arrays(shape=global_shape, sharding=sharding, arrays=device_arrays)

    return jtu.tree_map_with_path(
        partial(
            form_global_array_colocated_python,
            devices=list(dummy_array.sharding.addressable_devices),
            global_shape=self.global_shape,
            sharding=dummy_array.sharding,
        ),
        local_data,
    )

  def save_state(self, step_array):
    """Saves the iterator state to a file."""
    step = step_array.addressable_data(0).item()
    directory = epath.Path(self.checkpoint_path) / str(step) / "iter"
    if self.elastic:
      if jax.process_index() == 0:
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "process_0.json"
        filename.write_text(json.dumps(self.iterator.get_state(), indent=4))  # pyrefly: ignore[missing-attribute]
      return step_array
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    state = json.dumps(self.iterator.get_state(), indent=4)  # pyrefly: ignore[missing-attribute]
    filename.write_text(state)
    return step_array

  def restore_state(self, step_array):
    step = step_array.addressable_data(0).item()
    directory = epath.Path(self.checkpoint_path) / str(step) / "iter"
    if self.elastic:
      filename = directory / "process_0.json"
    else:
      filename = directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    if not filename.exists():
      raise FileNotFoundError(f"State file not found: {filename}")
    state = json.loads(filename.read_text())
    self.iterator.set_state(state)  # pyrefly: ignore[missing-attribute]
    return step_array


class RemoteIteratorWrapper:
  """Wrapper for RemoteIterator that handles device placement."""

  def __init__(
      self,
      get_ds_fn,
      preprocessing_fn,
      global_mesh,
      global_shape,
      sharding_spec=None,
      checkpoint_path="",
      elastic=False,
  ):
    self.cpu_devices = _colocated_cpu_devices(tuple(global_mesh.devices.flat))
    self.cpu_mesh = _colocated_cpu_mesh(global_mesh)
    if sharding_spec is None:
      sharding_spec = PartitionSpec(global_mesh.axis_names)
    array_shape = global_shape if global_shape is not None else (len(self.cpu_devices),)

    self.tpu_sharding = jax.sharding.NamedSharding(global_mesh, sharding_spec)
    self.cpu_sharding = jax.sharding.NamedSharding(self.cpu_mesh, sharding_spec)
    self.dummy_array = jnp.zeros(array_shape, dtype=jnp.int32)
    self.dummy_array = jax.device_put(self.dummy_array, self.cpu_sharding)

    # This is a proxy to a RemoteIterator running in a colocated process,
    # named "local_iterator" to match MultiHostDataLoadIterator's interface.
    remote_iterator_cls = colocated_python.colocated_python_class(RemoteIterator)
    self.local_iterator = remote_iterator_cls(
        get_ds_fn,  # pyrefly: ignore[bad-argument-count]
        preprocessing_fn,
        global_shape,
        checkpoint_path,
        elastic=elastic,  # pyrefly: ignore[unexpected-keyword]
    )
    max_logging.log("RemoteIteratorWrapper initiated")

  def __iter__(self):
    return self

  def reset(self):
    self.local_iterator.reset()  # pyrefly: ignore[missing-attribute]

  def __next__(self):
    out = self.local_iterator.get_next(self.dummy_array)  # pyrefly: ignore[missing-attribute]
    return jax.device_put(out, self.tpu_sharding)

  def save_state(self, step):
    replicated_cpu_sharding = NamedSharding(self.cpu_mesh, PartitionSpec())
    step_array = jnp.array(step, dtype=jnp.int32)
    step_array = jax.device_put(step_array, replicated_cpu_sharding)
    self.local_iterator.save_state(step_array)  # pyrefly: ignore[missing-attribute]

  def restore_state(self, step):
    replicated_cpu_sharding = NamedSharding(self.cpu_mesh, PartitionSpec())
    step_array = jnp.array(step, dtype=jnp.int32)
    step_array = jax.device_put(step_array, replicated_cpu_sharding)
    self.local_iterator.restore_state(step_array)  # pyrefly: ignore[missing-attribute]
