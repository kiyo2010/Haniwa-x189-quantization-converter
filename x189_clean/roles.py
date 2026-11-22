# SPDX-License-Identifier: Apache-2.0
"""X189-style physical type assignments; contains no calibrated statistics."""

from collections import Counter

FORMAT_COUNTS = {"BF16": 113, "XE82": 121, "IQ4_NL": 43, "IQ3_XXS": 26, "IQ2_XXS": 5, "IQ1_S": 2}
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
EMBEDDING = "model.embed_tokens.weight"

IQ = {
    'IQ1_S': (
        'model.layers.23.self_attn.o_proj.weight',
        'model.layers.27.self_attn.q_proj.weight',
    ),
    'IQ2_XXS': (
        'model.layers.17.mlp.down_proj.weight',
        'model.layers.24.mlp.down_proj.weight',
        'model.layers.25.mlp.down_proj.weight',
        'model.layers.26.mlp.down_proj.weight',
        'model.layers.26.self_attn.q_proj.weight',
    ),
    'IQ3_XXS': (
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.1.mlp.down_proj.weight',
        'model.layers.1.mlp.up_proj.weight',
        'model.layers.1.self_attn.v_proj.weight',
        'model.layers.10.self_attn.k_proj.weight',
        'model.layers.14.self_attn.k_proj.weight',
        'model.layers.16.self_attn.k_proj.weight',
        'model.layers.17.self_attn.k_proj.weight',
        'model.layers.18.self_attn.k_proj.weight',
        'model.layers.2.mlp.down_proj.weight',
        'model.layers.2.mlp.gate_proj.weight',
        'model.layers.2.mlp.up_proj.weight',
        'model.layers.21.self_attn.k_proj.weight',
        'model.layers.26.self_attn.k_proj.weight',
        'model.layers.27.self_attn.k_proj.weight',
        'model.layers.3.mlp.down_proj.weight',
        'model.layers.3.mlp.gate_proj.weight',
        'model.layers.3.mlp.up_proj.weight',
        'model.layers.4.mlp.down_proj.weight',
        'model.layers.4.mlp.gate_proj.weight',
        'model.layers.4.mlp.up_proj.weight',
        'model.layers.5.mlp.down_proj.weight',
        'model.layers.5.mlp.up_proj.weight',
        'model.layers.6.mlp.down_proj.weight',
        'model.layers.6.mlp.gate_proj.weight',
        'model.layers.6.mlp.up_proj.weight',
    ),
    'IQ4_NL': (
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.self_attn.o_proj.weight',
        'model.layers.0.self_attn.v_proj.weight',
        'model.layers.10.self_attn.v_proj.weight',
        'model.layers.11.self_attn.k_proj.weight',
        'model.layers.11.self_attn.v_proj.weight',
        'model.layers.12.self_attn.k_proj.weight',
        'model.layers.12.self_attn.v_proj.weight',
        'model.layers.13.self_attn.k_proj.weight',
        'model.layers.13.self_attn.v_proj.weight',
        'model.layers.14.self_attn.v_proj.weight',
        'model.layers.15.self_attn.k_proj.weight',
        'model.layers.15.self_attn.v_proj.weight',
        'model.layers.16.self_attn.v_proj.weight',
        'model.layers.17.self_attn.v_proj.weight',
        'model.layers.18.self_attn.v_proj.weight',
        'model.layers.19.self_attn.k_proj.weight',
        'model.layers.19.self_attn.v_proj.weight',
        'model.layers.2.self_attn.v_proj.weight',
        'model.layers.20.self_attn.k_proj.weight',
        'model.layers.20.self_attn.v_proj.weight',
        'model.layers.21.self_attn.v_proj.weight',
        'model.layers.22.self_attn.k_proj.weight',
        'model.layers.22.self_attn.v_proj.weight',
        'model.layers.23.self_attn.v_proj.weight',
        'model.layers.24.self_attn.v_proj.weight',
        'model.layers.25.self_attn.v_proj.weight',
        'model.layers.26.self_attn.v_proj.weight',
        'model.layers.27.self_attn.v_proj.weight',
        'model.layers.3.self_attn.k_proj.weight',
        'model.layers.3.self_attn.v_proj.weight',
        'model.layers.4.self_attn.k_proj.weight',
        'model.layers.4.self_attn.v_proj.weight',
        'model.layers.5.self_attn.k_proj.weight',
        'model.layers.5.self_attn.v_proj.weight',
        'model.layers.6.self_attn.k_proj.weight',
        'model.layers.6.self_attn.v_proj.weight',
        'model.layers.7.self_attn.k_proj.weight',
        'model.layers.7.self_attn.v_proj.weight',
        'model.layers.8.self_attn.k_proj.weight',
        'model.layers.8.self_attn.v_proj.weight',
        'model.layers.9.self_attn.k_proj.weight',
        'model.layers.9.self_attn.v_proj.weight',
    ),
}


def roles():
    result = {EMBEDDING: "XE82", "model.norm.weight": "BF16"}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        result.update((prefix + name + ".weight", "XE82") for name in PROJECTIONS)
        result.update((prefix + name + ".weight", "BF16") for name in NORMS)
    for kind, names in IQ.items():
        for name in names:
            if result.get(name) != "XE82":
                raise ValueError(f"Unexpected IQ format assignment for {name}")
            result[name] = kind
    if len(result) != 310 or dict(Counter(result.values())) != FORMAT_COUNTS:
        raise ValueError("Expected fixed 310-tensor role map")
    return result
