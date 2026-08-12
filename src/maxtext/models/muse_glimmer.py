"""
Copyright 2023-2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Decoder layer definition for Muse-Glimmer (text tower only).

Reference: ``transformers`` main,
``src/transformers/models/muse_glimmer/modeling_muse_glimmer.py``
(``MuseGlimmerTextModel``).  Structure, in HF terms::

    h = RMSNorm_noscale(embed_tokens[ids])              # NormedEmbedding, NO sqrt(d)
    for each layer:
        r = h
        x = CenteredRMSNorm(h, input_layernorm.w, rms_norm_eps)     # n * (1 + w)
        q,k,v = q_proj(x), k_proj(x), v_proj(x)
        q = RMSNorm_noscale(q, rms_norm_eps) * qk_scale_factor      # q ONLY
        k = RMSNorm_noscale(k, rms_norm_eps)
        if layer_rope_theta != 0: q,k = rope(q,k)                   # NoPE on full layers
        o = attn(q,k,v, scaling=head_dim**-0.5, causal,
                 window=sliding_window if sliding else None)
        o = o * sigmoid(gate_proj(x))                               # gate from x, pre-o_proj
        o = o_proj(o)
        o = CenteredRMSNorm(o, post_attention_layernorm.w, post_norm_eps)
        h = r + o
        r = h
        y = CenteredRMSNorm(h, pre_feedforward_layernorm.w, rms_norm_eps)
        y = down_proj(silu(gate_proj(y)) * up_proj(y))
        y = CenteredRMSNorm(y, post_feedforward_layernorm.w, post_norm_eps)
        h = r + y
    h = RMSNorm(h, model.norm.w, rms_norm_eps)          # plain `n * w`, NOT (1 + w)
    logits = softcap(lm_head(h) * output_multiplier)

MaxText mapping of the pieces that are not plain layer code:
  * NormedEmbedding      -> ``normed_embedding`` below, called from the decoders'
                            ``_apply_embedding`` when ``decoder_block == muse_glimmer``.
  * CenteredRMSNorm      -> ``RMSNorm(scale_init=zeros, scale_offset=1.0)``.
  * parameter-free qk    -> ``use_qk_norm=True`` + ``qk_norm_with_scale: false``.
  * qk_scale_factor      -> folded into ``query_pre_attn_scalar`` together with the
                            ``head_dim ** -0.5`` softmax scaling (MaxText's attention op
                            runs with ``sm_scale = 1.0``).  Both are pure scalars applied
                            to q, and RoPE is linear, so folding them is exact.
  * sigmoid attn gate    -> ``Attention(use_attn_output_gate=True)``.
  * NoPE on full layers  -> ``Attention(is_nope_layer=True)``.
  * output_multiplier    -> ``logits_output_multiplier`` in the output head.
"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Optional

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import AttentionType, Config
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


# -----------------------------------------------------------------------------
# Layer pattern
# -----------------------------------------------------------------------------
# config.json ships layer_types = [sliding, sliding, sliding, full] * 13 and
# layer_rope_theta = [500000, 500000, 500000, 0] * 13 -- i.e. the *full* layers
# are exactly the NoPE layers.  Both facts are derived from this one pattern.
MUSE_GLIMMER_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id: int) -> AttentionType:
  """Sliding vs full attention for a given (absolute or in-block) layer index."""
  return MUSE_GLIMMER_ATTENTION_PATTERN[layer_id % len(MUSE_GLIMMER_ATTENTION_PATTERN)]


def determine_is_nope_layer(layer_id: int) -> bool:
  """True when ``layer_rope_theta[layer_id] == 0`` (the full-attention layers)."""
  return get_attention_type(layer_id) == AttentionType.GLOBAL


def get_post_norm_epsilon(config: Config) -> float:
  """`post_norm_eps` (1e-8) for the two POST norms, falling back to the usual eps."""
  eps = getattr(config, "post_norm_layer_epsilon", 0.0)
  return eps if eps and eps > 0.0 else config.normalization_layer_epsilon


def get_query_pre_attn_scalar(config: Config) -> float:
  """`qk_scale_factor` * the softmax `head_dim ** -0.5`.

  MaxText's attention op always runs with ``sm_scale = 1.0``; the customary
  ``1/sqrt(head_dim)`` is expected to be folded into the query.  Muse-Glimmer
  additionally multiplies q (only q) by ``qk_scale_factor`` right after the
  parameter-free qk-norm.  Both are scalars applied to q and RoPE is linear in q,
  so applying their product after RoPE is numerically the same computation.
  """
  return float(getattr(config, "qk_scale_factor", 1.0)) * (config.head_dim**-0.5)


def normed_embedding(y: jnp.ndarray, epsilon: float, dtype) -> jnp.ndarray:
  """`MuseGlimmerTextNormedEmbedding`: parameter-free RMSNorm over the embedding.

  NOTE (spec trap): there is **no** ``sqrt(hidden_size)`` multiplier here, unlike
  Gemma.  Do not "optimise" this by folding it into the embedding table -- it is a
  per-token nonlinearity and the raw table is still needed by the (untied) head.
  """
  x = y.astype(jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  return (x * jax.lax.rsqrt(mean2 + epsilon)).astype(dtype)


def centered_rms_norm(config: Config, num_features: int, epsilon: float, rngs: nnx.Rngs) -> RMSNorm:
  """`MuseGlimmerTextCenteredRMSNorm`: normed * (1.0 + w), weights stored zero-centred."""
  return RMSNorm(
      num_features=num_features,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      kernel_axes=("norm",),
      epsilon=epsilon,
      scale_init=nn.initializers.zeros,
      scale_offset=1.0,
      shard_mode=config.shard_mode,
      rngs=rngs,
  )


# -----------------------------------------------------------------------------
# Decoder layer
# -----------------------------------------------------------------------------
class MuseGlimmerDecoderLayer(nnx.Module):
  """One Muse-Glimmer text decoder layer (sandwich norms + gated GQA)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
      is_nope_layer: bool = False,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.attention_type = attention_type
    self.is_nope_layer = is_nope_layer
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    post_eps = get_post_norm_epsilon(config)

    self.pre_self_attention_norm = centered_rms_norm(
        config, dummy_inputs_shape[-1], config.normalization_layer_epsilon, rngs
    )
    self.post_self_attention_norm = centered_rms_norm(config, dummy_inputs_shape[-1], post_eps, rngs)
    self.pre_ffw_norm = centered_rms_norm(config, dummy_inputs_shape[-1], config.normalization_layer_epsilon, rngs)
    self.post_ffw_norm = centered_rms_norm(config, dummy_inputs_shape[-1], post_eps, rngs)

    # NOTE: the attribute MUST be named `self_attention`. SkyRL's tunix backend
    # targets LoRA with the qwix regex `layers_[0-9]+/self_attention/(query|key|value|out)`
    # and `layers_[0-9]+/mlp/(wi_0|wi_1|wo)` (see skyrl/backends/tunix_backend.py
    # _MAXTEXT_ATTN_REGEX / _MAXTEXT_MLP_REGEX). Renaming this breaks LoRA silently
    # (zero adapter params, no error).
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_bias_in_projections=config.attention_bias,
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        is_nope_layer=self.is_nope_layer,
        query_pre_attn_scalar=get_query_pre_attn_scalar(config),
        model_mode=model_mode,
        use_qk_norm=True,
        use_attn_output_gate=True,
        rngs=rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.dropout = linears.Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # --- attention block -----------------------------------------------------
    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    attention_lnx = self.post_self_attention_norm(attention_lnx)
    intermediate_inputs = inputs + attention_lnx

    # --- feed-forward block --------------------------------------------------
    mlp_in = self.pre_ffw_norm(intermediate_inputs)
    mlp_in = nn.with_logical_constraint(mlp_in, ("activation_batch", "activation_norm_length", "activation_embed"))
    mlp_lnx = self.mlp(mlp_in)
    mlp_lnx = self.post_ffw_norm(mlp_lnx)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if is_scan_carry:

      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    elif cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


MuseGlimmerDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MuseGlimmerDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class MuseGlimmerScannableBlock(nnx.Module):
  """`inhomogeneous_layer_cycle_interval` (=4) consecutive layers, one [S,S,S,F] cycle.

  52 layers = 13 scanned repeats of this block.  Sub-layers are named
  ``layers_0..layers_3`` so the checkpoint mapping can address them.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    for layer_id in range(config.inhomogeneous_layer_cycle_interval):
      layer = MuseGlimmerDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          attention_type=get_attention_type(layer_id),
          is_nope_layer=determine_is_nope_layer(layer_id),
          quant=self.quant,
          rngs=rngs,
      )
      setattr(self, f"layers_{layer_id}", layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs
    for layer_id in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layers_{layer_id}")
      y = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
      if cfg.scan_layers:
        y = y[0]
    if cfg.scan_layers:
      return y, None
    else:
      return y


MuseGlimmerScannableBlockToLinen = nnx_wrappers.to_linen_class(
    MuseGlimmerScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
