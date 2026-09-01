# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for sparse expert LoRA factor creation and routed-token math."""

from types import SimpleNamespace

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
import qwix

from maxtext.configs import pyconfig
from maxtext.layers.moe import _sparse_expert_lora_down
from maxtext.layers.moe import _sparse_expert_lora_up
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils.lora_utils import install_sparse_expert_lora
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


class RoutedMoE(nnx.Module):
  """Minimal module with the state contract used by the installer."""

  def __init__(self, *, scanned: bool = False, prefuse: bool = False):
    e, layers, d, f = 3, 2, 5, 7
    devices = np.asarray(jax.devices()[:1])
    self.mesh = Mesh(devices, ("data",))
    self.config = SimpleNamespace(
        sparse_matmul=True,
        prefuse_moe_weights=prefuse,
        logical_axis_rules=[
            ("exp", None),
            ("embed_moe", None),
            ("mlp_moe", None),
            ("layers", None),
        ],
    )
    if scanned:
      metadata = {nnx.PARTITION_NAME: "layers", "param_scan_axis": 1}
      self.wi_0 = nnx.Param(
          jnp.zeros((e, layers, d, f)), out_sharding=("exp", "layers", "embed_moe", "mlp_moe"), **metadata
      )
      self.wi_1 = nnx.Param(
          jnp.zeros((e, layers, d, f)), out_sharding=("exp", "layers", "embed_moe", "mlp_moe"), **metadata
      )
      self.wo = nnx.Param(
          jnp.zeros((e, layers, f, d)), out_sharding=("exp", "layers", "mlp_moe", "embed_moe"), **metadata
      )
    else:
      self.wi_0 = nnx.Param(jnp.zeros((e, d, f)), out_sharding=("exp", "embed_moe", "mlp_moe"))
      self.wi_1 = nnx.Param(jnp.zeros((e, d, f)), out_sharding=("exp", "embed_moe", "mlp_moe"))
      self.wo = nnx.Param(jnp.zeros((e, f, d)), out_sharding=("exp", "mlp_moe", "embed_moe"))


class Model(nnx.Module):

  def __init__(self, **kwargs):
    self.moe = RoutedMoE(**kwargs)


class ModelWithLinear(nnx.Module):

  def __init__(self):
    self.moe = RoutedMoE()
    self.linear = nnx.Linear(5, 6, rngs=nnx.Rngs(0))

  def __call__(self, inputs):
    return self.linear(inputs)


def _grouped_gmm(group_sizes):
  expert_ids = jnp.repeat(
      jnp.arange(group_sizes.shape[0], dtype=jnp.int32),
      group_sizes,
      total_repeat_length=int(group_sizes.sum()),
  )

  def gmm(lhs, rhs, **_kwargs):
    return jnp.einsum("tk,tkn->tn", lhs, rhs[expert_ids])

  return gmm, expert_ids


@pytest.mark.parametrize("scanned", [False, True])
def test_install_sparse_expert_lora_shapes_metadata_and_initialization(scanned):
  model = Model(scanned=scanned)

  assert install_sparse_expert_lora(model, rank=4, alpha=8.0, rngs=nnx.Rngs(7)) == 1

  if scanned:
    assert model.moe.wi_0_lora_a.shape == (5, 2, 4)
    assert model.moe.wi_0_lora_b.shape == (4, 2, 3, 7)
    assert model.moe.wo_lora_a.shape == (3, 2, 7, 4)
    assert model.moe.wo_lora_b.shape == (4, 2, 5)
    for name in ("wi_0_lora_a", "wi_0_lora_b", "wi_1_lora_a", "wi_1_lora_b", "wo_lora_a", "wo_lora_b"):
      metadata = getattr(model.moe, name).get_metadata()
      assert metadata[nnx.PARTITION_NAME] == "layers"
      assert metadata["param_scan_axis"] == 1
  else:
    assert model.moe.wi_0_lora_a.shape == (5, 4)
    assert model.moe.wi_0_lora_b.shape == (4, 3, 7)
    assert model.moe.wo_lora_a.shape == (3, 7, 4)
    assert model.moe.wo_lora_b.shape == (4, 5)

  assert model.moe.sparse_expert_lora_scale == 2.0
  assert isinstance(model.moe.wi_0_lora_a, nnx.LoRAParam)
  assert model.moe.wi_0_lora_a.get_metadata()["sparse_expert_lora"] is True
  assert np.any(np.asarray(model.moe.wi_0_lora_a) != 0)
  assert np.any(np.asarray(model.moe.wo_lora_a) != 0)
  np.testing.assert_array_equal(np.asarray(model.moe.wi_0_lora_b), 0)
  np.testing.assert_array_equal(np.asarray(model.moe.wo_lora_b), 0)


def test_install_sparse_expert_lora_rejects_duplicate_and_prefused_weights():
  model = Model()
  install_sparse_expert_lora(model, rank=2, alpha=2.0, rngs=nnx.Rngs(0))
  with pytest.raises(ValueError, match="already installed"):
    install_sparse_expert_lora(model, rank=2, alpha=2.0, rngs=nnx.Rngs(0))

  with pytest.raises(ValueError, match="prefuse_moe_weights"):
    install_sparse_expert_lora(Model(prefuse=True), rank=2, alpha=2.0, rngs=nnx.Rngs(0))


def test_qwix_wrapping_preserves_preinstalled_sparse_factors():
  model = ModelWithLinear()
  install_sparse_expert_lora(model, rank=2, alpha=4.0, rngs=nnx.Rngs(1))

  adapted = qwix.apply_lora_to_model(
      model,
      qwix.LoraProvider(module_path="linear", rank=2, alpha=4.0),
      jnp.ones((1, 5), dtype=jnp.float32),
      rngs=nnx.Rngs(2),
  )

  for name in ("wi_0_lora_a", "wi_0_lora_b", "wi_1_lora_a", "wi_1_lora_b", "wo_lora_a", "wo_lora_b"):
    assert isinstance(getattr(adapted.moe, name), nnx.LoRAParam)
  lora_paths = {"/".join(map(str, path)) for path, value in nnx.iter_graph(adapted) if isinstance(value, nnx.LoRAParam)}
  assert any("linear" in path for path in lora_paths)
  assert sum("moe" in path for path in lora_paths) == 6


def test_sparse_expert_lora_matches_materialized_per_expert_deltas():
  key = jax.random.key(13)
  x_key, act_key, *factor_keys = jax.random.split(key, 8)
  x = jax.random.normal(x_key, (6, 5))
  activated = jax.random.normal(act_key, (6, 7))
  gate_a = jax.random.normal(factor_keys[0], (5, 3))
  gate_b = jax.random.normal(factor_keys[1], (3, 3, 7))
  down_a = jax.random.normal(factor_keys[2], (3, 7, 3))
  down_b = jax.random.normal(factor_keys[3], (3, 5))
  scale = 0.25
  gmm, expert_ids = _grouped_gmm(jnp.asarray([2, 1, 3], dtype=jnp.int32))

  actual_up = _sparse_expert_lora_up(x, gate_a, gate_b, gmm_fn=gmm, tiling=(1,) * 9, weight_gather_axes=[], scale=scale)
  actual_down = _sparse_expert_lora_down(
      activated, down_a, down_b, gmm_fn=gmm, tiling=(1,) * 9, weight_gather_axes=[], scale=scale
  )

  dense_up = jnp.einsum("dr,erf->edf", gate_a, gate_b)
  dense_down = jnp.einsum("efr,rd->efd", down_a, down_b)
  expected_up = jnp.einsum("td,tdf->tf", x, dense_up[expert_ids]) * scale
  expected_down = jnp.einsum("tf,tfd->td", activated, dense_down[expert_ids]) * scale
  np.testing.assert_allclose(actual_up, expected_up, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(actual_down, expected_down, rtol=1e-5, atol=1e-5)


def test_sparse_expert_lora_has_gradients_for_every_factor():
  keys = jax.random.split(jax.random.key(21), 6)
  x = jax.random.normal(keys[0], (6, 5))
  activated = jax.random.normal(keys[1], (6, 7))
  factors = (
      jax.random.normal(keys[2], (5, 3)),
      jax.random.normal(keys[3], (3, 3, 7)),
      jax.random.normal(keys[4], (3, 7, 3)),
      jax.random.normal(keys[5], (3, 5)),
  )
  gmm, _ = _grouped_gmm(jnp.asarray([2, 2, 2], dtype=jnp.int32))

  def loss(gate_a, gate_b, down_a, down_b):
    up = _sparse_expert_lora_up(x, gate_a, gate_b, gmm_fn=gmm, tiling=(1,) * 9, weight_gather_axes=[], scale=0.5)
    down = _sparse_expert_lora_down(
        activated, down_a, down_b, gmm_fn=gmm, tiling=(1,) * 9, weight_gather_axes=[], scale=0.5
    )
    return jnp.sum(jnp.square(up)) + jnp.sum(jnp.square(down))

  grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*factors)
  for grad, factor in zip(grads, factors):
    assert grad.shape == factor.shape
    assert np.isfinite(np.asarray(grad)).all()
    assert np.linalg.norm(np.asarray(grad)) > 0


def test_routed_moe_sparse_forward_consumes_installed_factors():
  """Exercises routing, shard_map, ragged-dot base GMMs, and all LoRA branches."""
  cfg = pyconfig.initialize(
      [None, f"{MAXTEXT_CONFIGS_DIR}/base.yml"],
      run_name="sparse_expert_lora_integration_test",
      enable_checkpointing=False,
      decoder_block="mixtral",
      num_experts=4,
      num_experts_per_tok=2,
      base_emb_dim=16,
      base_mlp_dim=24,
      base_moe_mlp_dim=24,
      dtype="float32",
      weight_dtype="float32",
      sparse_matmul=True,
      megablox=False,
      capacity_factor=-1,
      max_target_length=4,
      per_device_batch_size=1,
      log_config=False,
  )
  devices_array = maxtext_utils.create_device_mesh(cfg)
  model = moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=Mesh(devices_array, cfg.mesh_axes),
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      rngs=nnx.Rngs(0),
      intermediate_dim=cfg.base_moe_mlp_dim,
  )
  inputs = jax.random.normal(jax.random.key(4), (1, 4, cfg.base_emb_dim))
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    expected, _, _ = model(inputs)

  install_sparse_expert_lora(model, rank=4, alpha=8.0, rngs=nnx.Rngs(5))
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    zero_delta_output, _, _ = model(inputs)
  np.testing.assert_allclose(zero_delta_output, expected, rtol=1e-6, atol=1e-6)

  # B starts at zero. Making every expand factor nonzero forces all three
  # sparse adapter branches to contribute while keeping base weights fixed.
  model.wi_0_lora_b[...] = jnp.full_like(model.wi_0_lora_b[...], 0.02)
  model.wi_1_lora_b[...] = jnp.full_like(model.wi_1_lora_b[...], -0.015)
  model.wo_lora_b[...] = jnp.full_like(model.wo_lora_b[...], 0.01)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    adapted, _, _ = model(inputs)

  assert adapted.shape == expected.shape
  assert np.isfinite(np.asarray(adapted)).all()
  assert not np.allclose(adapted, expected)
