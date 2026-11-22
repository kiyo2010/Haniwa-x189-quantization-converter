# SPDX-License-Identifier: Apache-2.0
"""An eight-dimensional codebook from the half-integer coset of the E8 lattice."""
from itertools import product
import hashlib

import numpy as np


def prototypes():
    """Choose 256 positive half-integer vectors by energy and a deterministic tie order."""
    candidates = list(product((1, 3, 5, 7), repeat=8))

    def order(values):
        energy = sum(v * v for v in values)
        tie = hashlib.sha256(b"x189-kftt-v2:half-integer-magnitude:" + bytes(values)).digest()
        return energy, tie, values

    chosen = sorted(candidates, key=order)[:256]
    return np.asarray(chosen, dtype=np.float32) * np.float32(.5)


def table_for(words, magnitudes):
    """Decode 8 magnitude bits, 1 translation bit, and 7 free sign bits."""
    words = np.asarray(words, dtype=np.uint16)
    magnitude = magnitudes[words >> 8]
    free = ((words[..., None] >> np.arange(7)) & 1).astype(np.int8)
    # For a half-integer vector, each sign change flips the parity of its integer sum.
    required = magnitude.sum(axis=-1).astype(np.int32) & 1
    inferred = (free.sum(axis=-1) & 1) ^ required
    bits = np.concatenate((free, inferred[..., None]), axis=-1)
    shift = np.where((words & 128) == 0, np.float32(.25), np.float32(-.25))
    return magnitude * (1 - 2 * bits) + shift[..., None]


def nearest(points, magnitudes, torch):
    """Minimize square error over all valid sign patterns without enumerating them."""
    magnitude_energy = magnitudes.square().sum(dim=1)
    required_parity = magnitudes.sum(dim=1).to(torch.int64) & 1
    row = torch.arange(len(points), device=points.device)
    bit_weights = 1 << torch.arange(7, device=points.device)
    best_cost = torch.full((len(points),), torch.inf, dtype=points.dtype, device=points.device)
    best_word = torch.zeros(len(points), dtype=torch.int64, device=points.device)
    for phase, translation in enumerate((.25, -.25)):
        z = points - translation
        signs = z < 0
        parity = signs.sum(dim=1, dtype=torch.int64) & 1
        needs_change = parity[:, None] != required_parity[None, :]
        weighted = z.abs().unsqueeze(1) * magnitudes.unsqueeze(0)
        # Flipping one sign increases the square distance by four times the product.
        smallest, axis = weighted.min(dim=2)
        penalties = 4 * smallest * needs_change
        norms = z.square().sum(dim=1)
        errors = norms[:, None] + magnitude_energy - 2 * weighted.sum(dim=2) + penalties
        distance, shape = errors.min(dim=1)
        chosen_axis = axis[row, shape]
        signs[row, chosen_axis] = signs[row, chosen_axis] ^ needs_change[row, shape]
        low = (signs[:, :7] * bit_weights).sum(dim=1)
        word = (shape << 8) | (phase << 7) | low
        replaces = distance < best_cost
        best_word = torch.where(replaces, word, best_word)
        best_cost = torch.minimum(best_cost, distance)
    return best_word
