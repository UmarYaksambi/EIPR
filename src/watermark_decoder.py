from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme
from .watermark_embedder import (
    ALPHA,
    BLOCK_SIZE,
    BITS_PER_BLOCK,
    EMBED_COEFF_INDICES,
    _decode_coeff,
)
from .geometric_sync import correct_attacked_image

def extract_watermark(
    image_bgr: np.ndarray,
    n_bits: int,
    ecc_engine: AdaptiveECCEngine,
    rate_map: np.ndarray,
    scheme: ECCScheme = "reed_solomon",
    alpha: float = ALPHA,
) -> np.ndarray:
    
    # 1. Blind Geometric Correction
    image_bgr = correct_attacked_image(image_bgr)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    h, w = Y.shape
    n_rows = h // BLOCK_SIZE
    n_cols = w // BLOCK_SIZE

    s_r, s_c = Y.strides
    blocks = as_strided(
        Y,
        shape=(n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE),
        strides=(s_r * BLOCK_SIZE, s_c * BLOCK_SIZE, s_r, s_c),
    )
    dct_flat: np.ndarray = np.asarray(
        dctn(np.ascontiguousarray(blocks), norm="ortho", axes=(-2, -1))
    ).reshape(n_rows, n_cols, BLOCK_SIZE * BLOCK_SIZE)

    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(
        set(float(r) for r in np.unique(rounded_map)), reverse=True
    )

    tier_results: list[tuple[np.ndarray, float]] = []

    for tier_rate in unique_rates:
        if tier_rate <= 0.0:
            continue
            
        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        br_arr, bc_arr = np.where(tier_mask)
        tier_coords = list(zip(br_arr.tolist(), bc_arr.tolist()))
        if not tier_coords:
            continue

        dummy = np.zeros(n_bits, dtype=np.uint8)
        codeword_len = len(ecc_engine.encode_block(dummy, tier_rate, scheme))

        votes = np.zeros((codeword_len, 2), dtype=np.int32)
        bit_idx = 0

        for (br, bc) in tier_coords:
            # Bulletproof sync: advance bit_idx even if block is skipped!
            if br >= n_rows or bc >= n_cols:
                bit_idx += len(EMBED_COEFF_INDICES)
                continue

            coeffs = dct_flat[br, bc]
            for coeff_idx in EMBED_COEFF_INDICES:
                p = bit_idx % codeword_len
                bit = _decode_coeff(float(coeffs[coeff_idx]), alpha)
                votes[p, bit] += 1
                bit_idx += 1

        raw_codeword = (votes[:, 1] >= votes[:, 0]).astype(np.uint8)

        decoded, success = ecc_engine.decode_block(
            raw_codeword, tier_rate, scheme, n_payload=n_bits
        )

        weight = tier_rate * (100.0 if success else 1.0)
        tier_results.append((decoded, weight))

    if not tier_results:
        return np.zeros(n_bits, dtype=np.uint8)

    total_w = sum(w for _, w in tier_results)
    soft = np.zeros(n_bits, dtype=np.float64)
    for decoded, weight in tier_results:
        arr = np.zeros(n_bits, dtype=np.uint8)
        copy_len = min(n_bits, len(decoded))
        arr[:copy_len] = decoded[:copy_len]
        soft += weight * arr.astype(np.float64)

    return (soft / total_w >= 0.5).astype(np.uint8)