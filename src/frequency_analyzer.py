# src/frequency_analyzer.py
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn

BLOCK_SIZE: int = 8

def compute_block_dct_variance(image_gray: np.ndarray) -> np.ndarray:
    h, w = image_gray.shape
    h_c = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_c = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = image_gray[:h_c, :w_c].astype(np.float32)

    n_rows = h_c // BLOCK_SIZE
    n_cols = w_c // BLOCK_SIZE

    s_r, s_c = img.strides
    blocks: np.ndarray = as_strided(
        img,
        shape=(n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE),
        strides=(s_r * BLOCK_SIZE, s_c * BLOCK_SIZE, s_r, s_c),
    )
    blocks = np.ascontiguousarray(blocks)

    dct_blocks: np.ndarray = np.asarray(
        dctn(blocks, norm="ortho", axes=(-2, -1))
    )

    flat = dct_blocks.reshape(n_rows, n_cols, BLOCK_SIZE * BLOCK_SIZE)
    
    # --- QIM-Invariant Variance ---
    # Exclude DC (0) and QIM target indices (1, 2) to prevent the 
    # embedding process from shifting the block's texture classification.
    mask = np.ones(BLOCK_SIZE * BLOCK_SIZE, dtype=bool)
    mask[0] = False
    mask[1] = False
    mask[2] = False
    
    ac = flat[:, :, mask]
    variance_map = np.var(ac, axis=-1).astype(np.float32)

    return variance_map


def build_ecc_rate_map(
    variance_map: np.ndarray,
    tau_low: float,
    tau_high: float,
    r_high: float = 0.75,
    r_mid: float  = 0.50,
    r_low: float  = 0.25,
) -> np.ndarray:
    rate_map = np.full(variance_map.shape, r_mid, dtype=np.float32)
    rate_map[variance_map < tau_low]  = r_high
    rate_map[variance_map > tau_high] = r_low
    return rate_map


def calibrate_thresholds(
    calibration_variances: np.ndarray,
    percentile_low: float  = 25.0,
    percentile_high: float = 75.0,
) -> tuple[float, float]:
    if calibration_variances.size == 0:
        raise ValueError("calibration_variances is empty — no blocks to calibrate from.")
    tau_low  = float(np.percentile(calibration_variances, percentile_low))
    tau_high = float(np.percentile(calibration_variances, percentile_high))
    return tau_low, tau_high