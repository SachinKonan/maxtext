# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-regression tests for lazy scanned checkpoint conversion."""

from types import SimpleNamespace
from unittest import mock

import numpy as np

from maxtext.checkpoint_conversion import to_maxtext


def test_single_axis_scan_fills_preallocated_result_without_np_stack():
  source = {
      "layer.0": np.arange(6, dtype=np.float32).reshape(2, 3),
      "layer.1": np.arange(6, 12, dtype=np.float32).reshape(2, 3),
  }
  config = SimpleNamespace(scan_layers=True, param_scan_axis=1)

  with mock.patch.object(np, "stack", side_effect=AssertionError("full-size copy")):
    result = to_maxtext._build_single_axis_stacked_tensor(list(source), source.__getitem__, None, (2, 2, 3), config)

  expected = np.empty((2, 2, 3), dtype=np.float32)
  expected[:, 0, :] = source["layer.0"]
  expected[:, 1, :] = source["layer.1"]
  np.testing.assert_array_equal(result, expected)


def test_multi_axis_scan_fills_preallocated_result_without_np_stack():
  source = {
      f"expert.{expert}.layer.{layer}": np.full((2, 3), expert * 10 + layer, dtype=np.float32)
      for expert in range(2)
      for layer in range(3)
  }
  keys = [[f"expert.{expert}.layer.{layer}" for layer in range(3)] for expert in range(2)]
  config = SimpleNamespace(scan_layers=True, param_scan_axis=1)

  with mock.patch.object(np, "stack", side_effect=AssertionError("full-size copy")):
    result = to_maxtext._build_multi_axis_stacked_tensor(keys, source.__getitem__, None, (2, 3, 2, 3), config)

  expected = np.empty((2, 3, 2, 3), dtype=np.float32)
  for expert in range(2):
    for layer in range(3):
      expected[expert, layer] = source[f"expert.{expert}.layer.{layer}"]
  np.testing.assert_array_equal(result, expected)


def test_nested_gemma_scan_fills_nonleading_axes_without_np_stack_or_moveaxis():
  source = {
      f"block.{block}.layer.{layer}": np.full((2, 3), block * 10 + layer, dtype=np.float32)
      for block in range(2)
      for layer in range(3)
  }
  keys = [[f"block.{block}.layer.{layer}" for layer in range(3)] for block in range(2)]
  config = SimpleNamespace(scan_layers=True, param_scan_axis=1)

  with (
      mock.patch.object(np, "stack", side_effect=AssertionError("full-size copy")),
      mock.patch.object(np, "moveaxis", side_effect=AssertionError("full-size view")),
  ):
    result = to_maxtext._build_multi_axis_stacked_tensor(
        keys,
        source.__getitem__,
        None,
        (2, 2, 3, 3),
        config,
        "params-decoder-scanned_blocks-local_layers-test",
    )

  expected = np.empty((2, 2, 3, 3), dtype=np.float32)
  for block in range(2):
    for layer in range(3):
      expected[:, block, layer, :] = source[f"block.{block}.layer.{layer}"]
  np.testing.assert_array_equal(result, expected)


def test_lazy_composite_slice_does_not_call_copying_np_array():
  source = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
  final_weights = [None, None]

  to_maxtext._get_maxtext_weight(
      lambda: source,
      [0, 1],
      [(3, 4), (3, 4)],
      ("left", "right"),
      final_weights,
      "float32",
      True,
  )

  with mock.patch.object(np, "array", side_effect=AssertionError("copied base")):
    left = np.asarray(final_weights[0])
    right = np.asarray(final_weights[1])

  np.testing.assert_array_equal(left, source[..., 0])
  np.testing.assert_array_equal(right, source[..., 1])


def test_lazy_tensor_honors_numpy_copy_protocol():
  source = np.arange(6, dtype=np.float32)
  lazy = to_maxtext.LazyTensor(lambda: source, source.shape, source.dtype)

  shared = np.asarray(lazy)
  copied = np.array(lazy, copy=True)

  assert np.shares_memory(shared, source)
  assert not np.shares_memory(copied, source)
  np.testing.assert_array_equal(copied, source)
