# Copyright 2023-2026 Google LLC
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
"""Regression test for Qwix quantization of a pure-NNX model with MTP enabled."""

import unittest

import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.common.common_types import Config, MODEL_MODE_TRAIN
from maxtext.layers import embeddings
from maxtext.layers import multi_token_prediction
from maxtext.layers import quantizations
from maxtext.layers.nnx_decoders import NNXDecoderLayer
from maxtext.utils import maxtext_utils

from tests.utils.test_helpers import get_test_config_path


class _MockDecoderForMTP:
  """Minimal decoder that supplies the embedding and output head the MTP block calls."""

  def __init__(self, config: Config):
    self.config = config
    self.model_mode = MODEL_MODE_TRAIN

  def _apply_embedding(self, _shared_embedding, input_ids, _position_ids, _deterministic, model_mode):
    batch_size, seq_len = input_ids.shape
    return jnp.zeros((batch_size, seq_len, self.config.base_emb_dim), dtype=self.config.dtype)

  def apply_output_head(self, _shared_embedding, hidden_state, _deterministic, model_mode):
    batch_size, seq_len, _ = hidden_state.shape
    return jnp.zeros((batch_size, seq_len, self.config.vocab_size), dtype=self.config.dtype)


class _TransformerWithMTP(nnx.Module):
  """Smallest NNX model that reads decoder targets through a real MTP block.

  The call signature matches what quantizations.maybe_quantize_model passes into
  qwix.quantize_model. decoder_target_tokens and decoder_target_mask default to
  None, mirroring the pure-NNX forward where nothing supplies them unless
  maybe_quantize_model does.
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self._shared_embedding = embeddings.Embed(
        num_embeddings=config.vocab_size,
        num_features=config.base_emb_dim,
        config=config,
        mesh=mesh,
        rngs=rngs,
    )
    self.decoder = _MockDecoderForMTP(config)
    self.mtp_block = multi_token_prediction.MultiTokenPredictionBlock(
        config=config,
        mesh=mesh,
        transformer_layer_module=NNXDecoderLayer,
        decoder=self.decoder,
        rngs=rngs,
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del enable_dropout
    main_hidden_state = self._shared_embedding(decoder_input_tokens)
    self.mtp_block(
        self._shared_embedding,
        main_hidden_state,
        decoder_input_tokens,
        decoder_target_tokens,
        decoder_target_mask,
        position_ids=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=MODEL_MODE_TRAIN,
        deterministic=True,
    )
    return self.decoder.apply_output_head(self._shared_embedding, main_hidden_state, True, MODEL_MODE_TRAIN)


class MaybeQuantizeModelMTPTest(unittest.TestCase):
  """Qwix must forward dummy decoder targets so the MTP block's roll does not see None."""

  def _build(self, mtp_num_layers):
    """Builds a tiny pure-NNX model with qwix int8 quantization configured."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="maybe_quantize_model_mtp_test",
        skip_jax_distributed_system=True,
        per_device_batch_size=1,
        mtp_num_layers=mtp_num_layers,
        base_emb_dim=16,
        base_mlp_dim=32,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        head_dim=8,
        max_target_length=16,
        vocab_size=32,
        pure_nnx=True,
        pure_nnx_decoder=True,
        use_qwix_quantization=True,
        quantization="int8",
        enable_dropout=False,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = _TransformerWithMTP(cfg, mesh, rngs=nnx.Rngs(params=0, dropout=0, aqt=0))
    return cfg, mesh, model

  def test_quantize_with_mtp_supplies_decoder_targets(self):
    """With MTP enabled, quantizing the model runs the qwix forward pass without error.

    Before the fix, maybe_quantize_model called qwix.quantize_model without the
    decoder targets, so the MTP block rolled a None target and jnp.roll raised a
    TypeError during tracing.
    """
    cfg, mesh, model = self._build(mtp_num_layers=1)
    with mesh:
      quantized = quantizations.maybe_quantize_model(model, cfg)
    self.assertIsNotNone(quantized)

  def test_quantize_without_mtp_still_works(self):
    """The mtp_num_layers=0 path (no dummy targets needed) must remain unaffected."""
    cfg, mesh, model = self._build(mtp_num_layers=0)
    with mesh:
      quantized = quantizations.maybe_quantize_model(model, cfg)
    self.assertIsNotNone(quantized)


if __name__ == "__main__":
  unittest.main()
