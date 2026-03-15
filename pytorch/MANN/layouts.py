from __future__ import annotations

import numpy as np


def build_gating_indices(input_dim: int) -> np.ndarray:
    if input_dim == 348:
        return np.array([
            96, 97, 98,
            108, 109, 110,
            129, 130, 131,
            141, 142, 143,
            294, 295, 296,
            306, 307, 308,
            327, 328, 329,
            339, 340, 341,
        ])

    if input_dim == 382:
        return np.concatenate([
            np.array([
                117, 118, 119,
                129, 130, 131,
                141, 142, 143,
                153, 154, 155,
                333, 334, 335,
                345, 346, 347,
                357, 358, 359,
                369, 370, 371,
            ]),
            np.arange(372, 382),
        ])

    raise ValueError(f"Unsupported MANN input dimension: {input_dim}")

