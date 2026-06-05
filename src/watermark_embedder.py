from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme
from .geometric_sync import embed_sync_chroma

# ALPHA is the base QIM step size.
# 8.0 targets mean PSNR >= 40 dB across a DiffusionDB dataset.
# With rate-coupled JND: smooth=4.0, mid=10.0, textured=20.0.
# This constant is imported by baseline_comparison.py as the default alpha
# for all baselines — it MUST stay in sync with experiment.yaml embedding.alpha.
ALPHA: float = 8.0
BLOCK_SIZE: int = 8
EMBED_COEFF_INDICES: list[int] = [1, 2, 3]
BITS_PER_BLOCK: int = len(EMBED_COEFF_INDICES)


def _embed_coeff(val: float, bit: int, alpha: float) -> float:
    q = int(np.floor(val / alpha))
    if (q % 2) != bit:
        q += 1
    return (q + 0.5) * alpha


def _decode_coeff(val: float, alpha: float) -> int:
    return int(np.floor(val / alpha)) % 2


def get_tier_alpha(tier_rate: float, base_alpha: float) -> float:
    """
    Rate-Coupled JND Mask: scale QIM step size by texture tier.

    With base_alpha=8.0 (default, targets PSNR >= 40 dB):
      Smooth   (rate >= 0.70): alpha = 4.0  — eye-sensitive smooth regions,
                                               minimal distortion
      Mid      (rate >= 0.40): alpha = 10.0 — moderate robustness
      Textured (rate <  0.40): alpha = 20.0 — strong embedding in high-variance
                                               blocks where noise is masked

    The multiplier ratios (0.50 / 1.25 / 2.50) are fixed. Only base_alpha
    is tuned to trade off PSNR vs robustness.
    """
    if tier_rate >= 0.70:
        return base_alpha * 0.50    # smooth   → 4.0  (invisible, PSNR-critical)
    elif tier_rate >= 0.40:
        return base_alpha * 1.50    # mid      → 12.0 (survives JPEG q50)
    else:
        return base_alpha * 3.50    # textured → 28.0 (survives gaussian_20)


def _image_to_dct_blocks(Y: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = Y.shape
    n_rows = h // BLOCK_SIZE
    n_cols = w // BLOCK_SIZE
    s_r, s_c = Y.strides
    blocks = as_strided(
        Y,
        shape=(n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE),
        strides=(s_r * BLOCK_SIZE, s_c * BLOCK_SIZE, s_r, s_c),
    )
    blocks = np.ascontiguousarray(blocks)
    dct_blocks: np.ndarray = np.asarray(dctn(blocks, norm="ortho", axes=(-2, -1)))
    return dct_blocks, n_rows, n_cols


def _embed_tier(
    Y_emb: np.ndarray,
    dct_blocks: np.ndarray,
    tier_coords: list[tuple[int, int]],
    codeword: np.ndarray,
    alpha: float,
) -> None:
    codeword_len = len(codeword)
    bit_idx = 0

    for (br, bc) in tier_coords:
        dct_b = dct_blocks[br, bc].copy()
        for coeff_idx in EMBED_COEFF_INDICES:
            p = bit_idx % codeword_len
            dct_b.flat[coeff_idx] = _embed_coeff(
                float(dct_b.flat[coeff_idx]), int(codeword[p]), alpha
            )
            bit_idx += 1

        r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
        Y_emb[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE] = np.asarray(
            idctn(dct_b, norm="ortho")
        )


def embed_watermark(
    image_bgr: np.ndarray,
    watermark_bits: np.ndarray,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    scheme: ECCScheme = "reed_solomon",
    alpha: float = ALPHA,
) -> np.ndarray:
    # Chroma sync tones embedded in Cr channel — does not collide with Y-channel QIM.
    image_bgr = embed_sync_chroma(image_bgr, channel=1)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    dct_blocks, n_rows, n_cols = _image_to_dct_blocks(Y)
    Y_emb = Y.copy()

    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(set(float(r) for r in np.unique(rounded_map)), reverse=True)

    for tier_rate in unique_rates:
        if tier_rate <= 0.0:
            continue

        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        br_arr, bc_arr = np.where(tier_mask)
        tier_coords = list(zip(br_arr.tolist(), bc_arr.tolist()))
        if not tier_coords:
            continue

        tier_alpha = get_tier_alpha(tier_rate, alpha)

        codeword = ecc_engine.encode_block(
            watermark_bits.astype(np.uint8), tier_rate, scheme
        )
        _embed_tier(Y_emb, dct_blocks, tier_coords, codeword, tier_alpha)

    ycrcb_out = ycrcb.copy()
    ycrcb_out[:, :, 0] = np.clip(Y_emb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def embedding_capacity(
    image_shape: tuple[int, ...],
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    n_bits: int,
    scheme: ECCScheme = "reed_solomon",
) -> int:
    h, w = image_shape[0], image_shape[1]
    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(set(float(r) for r in np.unique(rounded_map)))

    min_capacity = n_bits
    for tier_rate in unique_rates:
        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        tier_slots = int(np.sum(tier_mask)) * BITS_PER_BLOCK
        lo, hi = 1, n_bits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            dummy = np.zeros(mid, dtype=np.uint8)
            cw_len = len(ecc_engine.encode_block(dummy, tier_rate, scheme))
            if tier_slots >= cw_len:
                lo = mid
            else:
                hi = mid - 1
        min_capacity = min(min_capacity, lo)

    return min_capacity