# MaxText Checkpoint Architecture: Flax Linen vs. NNX

This document outlines the architectural differences, serialization structures, and interoperability designs of checkpoints in MaxText, spanning traditional Flax Linen layouts, native NNX shapes, and the unified packed auxiliary layout.

---

## 1. Flax Linen Checkpoint Structure

In traditional Flax Linen, models are stateless blueprints, and training variables are tracked inside a standard `TrainState` container (see [flax/training/train_state.py](https://github.com/google/flax/blob/main/flax/training/train_state.py)). 

### Standard Mode (No Emergency)
Standard Flax Linen checkpointing organizes variables under independent, modular subdirectories inside the training step directory.

#### On-Disk Directory Structure
* `step_1000/`
  * `items/` — Core model weights and optimizer parameters.
    * `params/` — Learnable model parameters (nested under "params" collection).
      * `params/` — Nested collection layer representing Linen weights.
    * `opt_state/` — Optimizer states (list of states with `None` placeholders).
    * `step` — Training step counter (serialized as `int32`).
  * `iter/` — Dataset progress state (optional, if using Grain).

#### Serialized PyTree Schema
The `items` directory contains a single serialized PyTree of arrays. Note that the weights are nested under the double `"params" -> "params"` collection layer to conform to Linen's collection structure:

```python
{
    "params": {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "query_proj": {"kernel": jax.Array(...)},
                        "key_proj": {"kernel": jax.Array(...)}
                    }
                }
            }
        }
    },
    "opt_state": [
        {
            "count": jax.Array(...),
            "mu": {"params": { ... }},
            "nu": {"params": { ... }}
        },
        None # Placeholders for EmptyState elements in the Optax chain
    ],
    "step": jax.Array(1000, dtype=jnp.int32)
}
```

### Emergency Mode (Emergency / Replicator Checkpoints)
Emergency managers (`EmergencyCheckpointManager` and `EmergencyReplicatorCheckpointManager`) are legacy v0 checkpointers designed for high-performance synchronous writes (e.g., during imminent cluster preemption). Because they only support writing a single, unified PyTree of states, they cannot create parallel composite directories.

#### On-Disk Directory Structure
* `step_1000/`
  * Contains a single consolidated folder holding the unified PyTree state payload.

#### Serialized PyTree Schema
The schema is a single, flattened `state` payload containing parameters, steps, and optimizers:

```python
{
    "params": {
        "params": { ... }
    },
    "opt_state": [ ... ],
    "step": jax.Array(...)
}
```

---

## 2. Native NNX Checkpoint Structure

Flax NNX shifts from a functional, stateless paradigm to an object-oriented, stateful paradigm. In MaxText, the native training state is managed inside the stateful container **`TrainStateNNX`** (defined in [maxtext/common/train_state_nnx.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py)), which wraps both the model (`nnx.Module`) and its optimizer (`nnx.Optimizer`) as stateful, mutable sub-objects.

The model module contains parameters (`nnx.Param`), batch statistics (`nnx.BatchStat`), attention caches (`nnx.Cache`), and random number generator states (`nnx.RngState`) directly as mutable attributes (defined in [flax/nnx/variablelib.py](https://github.com/google/flax/blob/main/flax/nnx/variablelib.py) and [flax/nnx/rnglib.py](https://github.com/google/flax/blob/main/flax/nnx/rnglib.py)).

### Default Serialization Layout
A native, unmodified `nnx.State` (see [flax/nnx/statelib.py](https://github.com/google/flax/blob/main/flax/nnx/statelib.py)) serializes everything (including learnable parameters, active random generators, and dynamic caches) together as a single flat PyTree representation of attributes.

#### On-Disk Directory Structure
* `step_1000/`
  * Contains a flat directory holding the full NNX State tree.

#### Serialized PyTree Schema
```python
{
    "model": {
        "decoder": {
            "layers_0": {
                "self_attention": {
                    "query_proj": {"kernel": jax.Array(...)},
                    "key_proj": {"kernel": jax.Array(...)}
                }
            }
        },
        "dropout": {
            "rngs": {
                "default": {"key": jax.Array(...)}
            }
        }
    },
    "optimizer": {
        "opt_state": {
            "0": {
                "count": jax.Array(...),
                "mu": { ... },
                "nu": { ... }
            }
        },
        "step": jax.Array(1000, dtype=jnp.uint32)
    }
}
```
* **Optimizer States:** Represent Optax chains as integer-keyed dictionaries (skipping empty states) rather than lists with `None` placeholders (see [flax/nnx/training/optimizer.py](https://github.com/google/flax/blob/main/flax/nnx/training/optimizer.py)).
* **Step Counters:** Track iterations as 32-bit unsigned integers (`uint32`) instead of `int32`.
* **In-flight variables:** RNG keys, dropout counters, and activation caches are packed directly inside the model tree, polluting the clean weight parameters.

---

## 3. The Imperative for Flax Linen & NNX Interoperability

As MaxText transitions from the functional, stateless Flax Linen framework to the object-oriented, stateful Flax NNX framework, maintaining strict bi-directional interoperability between their serialized checkpoint files is a critical engineering requirement. This interoperability serves three primary purposes:

### A. Zero-Downtime Training Resumption
Large-scale model training runs represent massive investments in time and compute. To ensure that ongoing pre-training or fine-tuning runs are not disrupted during framework upgrades, MaxText must support resuming an active Linen training run using a newer NNX trainer, or resuming an NNX-started run in Linen:
* **The Interoperability Goal:** This requires the on-disk parameter values and optimizer states to remain structurally identical, ensuring that the model's trajectory continues seamlessly without loss of execution history.

### B. Shared Downstream Ecosystem (Serving and Decoding)
MaxText maintains a highly optimized downstream ecosystem of serving, decoding (such as `decode.py`), quantization, and parameter conversion scripts. 
* **The Interoperability Goal:** By enforcing that NNX saves checkpoints in a 100% Linen-compatible layout on disk, NNX-trained models can be served, evaluated, or quantized using the pre-existing, production-tested Linen serving infrastructure. This prevents the need to duplicate complex decoding and serving scripts for separate frameworks.

### C. Unified Weight Conversion and Quantization Tooling
Tooling such as weight conversions (from Hugging Face checkpoints to MaxText formats) and quantization calibrations are heavily tied to the structural conventions of the Linen layout.
* **The Interoperability Goal:** Maintaining format interchangeability ensures that these shared conversion and quantization scripts can operate on a single standardized layout, significantly reducing code duplication and maintenance overhead for the engineering teams.

The bidirectional conversion functions implementing this parity layer are located in [maxtext/common/train_state_nnx.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/train_state_nnx.py).

---

## 4. MaxText Unified Checkpointing: The Linen-Interoperable NNX Layout with Packed `nnx_aux`

To achieve perfect cross-framework compatibility while completely unifying standard and emergency checkpointer paths, MaxText uses the **Packed Auxiliary Structure** as the single, universal layout for both standard and emergency managers. 

In this unified design, any dynamic, NNX-only auxiliary variables (RNG states, dropout counters) are **always packed directly inside the `"items"` dictionary** before being written to disk. This eliminates the need for separate composite directories on disk for standard runs, maintaining a single, consistent structure across all modes (implemented in [maxtext/common/checkpointing.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/common/checkpointing.py)).

### Unified On-Disk Directory Structure
Because the auxiliary state is embedded inside the `"items"` PyTree, the `"nnx_aux"` directory is written **under (inside)** the `"items"` folder on disk.

* `step_1000/`
  * `items/` — The items checkpointable (containing packed aux).
    * `params/` — Base model parameters (stripped of RNG keys).
      * `params/` — Nested collection layer representing Linen weights.
    * `opt_state/` — Optimizer states (list with `None` placeholders).
    * `step` — Step counter file (cast to `int32`).
    * `nnx_aux/` — RNG and dropout state saved directly inside `items/`.
      * `dropout/`
        * `count` — Maintained RNG continuity.
  * `iter/` — Dataset iterator progress state (optional, if using Grain).

* **For Standard Checkpointers (`ocp.training.Checkpointer`):** Orbax creates the parallel `"iter"` directory for dataset progress and writes `"items"` as a single composite checkpointable.
* **For Emergency Checkpointers (`EmergencyCheckpointManager`):** The manager saves only the unified `"items"` payload containing `"nnx_aux"` nested inside it, bypassing the single-PyTree limitations.

### Unified Serialized PyTree Schema
The complete PyTree representation of the `"items"` payload saved to disk is:

```python
{
    "params": {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "query_proj": {"kernel": jax.Array(...)},
                        "key_proj": {"kernel": jax.Array(...)}
                    }
                }
            }
        }
    },
    "step": jax.Array(1000, dtype=jnp.int32),           # Cast step counter
    "opt_state": [                                      # Optimizer states (list with None placeholders)
        {
            "count": jax.Array(...),
            "mu": {"params": { ... }},
            "nu": {"params": { ... }}
        },
        None
    ],
    "nnx_aux": {                                        # Packed auxiliary state containing RNGs
        "dropout": {
            "count": jax.Array(42)                      # Maintained RNG continuity
        }
    }
}
```

---

## 5. Cross-Framework Compatibility Mechanics

This unified packed solution achieves perfect interoperability during restoration across both frameworks:

| Target Framework | Restoration Behavior |
| :--- | :--- |
| **Flax Linen Runs** | Linen trainers initialize a standard `TrainState` containing only `"step"`, `"params"`, and `"opt_state"` keys. Because `"nnx_aux"` is absent from the Linen template, Orbax target-guided loading simply ignores the `step_1000/items/nnx_aux/` folder on disk during load, loading parameters and optimizer states normally. |
| **Flax NNX Runs** | The NNX checkpointer expects `"nnx_aux"` to be present in its `linen_abstract` target template. It restores the folder from `step_1000/items/nnx_aux/`, pops the branch, and merges the RNG stream back into the model state, guaranteeing RNG and dropout continuity across resumes. |

