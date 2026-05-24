"""
watermark_decoder.py — Adaptive-ECC QIM watermark extractor (per-tier + geometric sync).

Mirrors the embedder's per-tier block-DCT transform chain exactly.

Per-tier decoding  [FIXED vs first revision]
--------------------------------------------
The original decoder used a single global-mean-rate codeword, which discarded
the per-tier ECC information.  The new decoder:

  1. For each tier in rate_map:
     a. Collects raw bits from tier blocks in the same cycling order as the
        embedder.
     b. Folds the bit stream back to codeword length via majority vote at each
        codeword position (across multiple copies embedded by cycling).
     c. RS-decodes the majority-voted codeword to recover payload bits.
  2. Combines per-tier decoded payloads by weighted vote:
        weight = tier_rate  (0.75 / 0.50 / 0.25)
     Higher-rate tiers are more protected → receive more weight.

Geometric correction  [NEW]
---------------------------
When ``original_bgr`` is supplied (the sync-embedded original stored at the
provider's detector), the decoder calls geometric_sync.correct_attacked_image
to reverse crop, rotation, and scale before running QIM extraction.  This
fixes the BER ≈ 0.5 failures on all geometric attacks.
"""
from __future__ import annotations

import warnings
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


def extract_watermark(
    image_bgr: np.ndarray,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    n_bits: int,
    scheme: ECCScheme = "reed_solomon",
    alpha: float = ALPHA,
    original_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract and ECC-decode the watermark from a (possibly attacked) image.

    Args:
        image_bgr:    H×W×3 uint8 BGR image (possibly attacked).
        rate_map:     (n_rows, n_cols) float32 per-block ECC rate map —
                      must *exactly* match the map used during embedding.
        ecc_engine:   ``AdaptiveECCEngine`` instance.
        n_bits:       Number of payload bits to recover.
        scheme:       ECC scheme — must match the one used during embedding.
        alpha:        QIM step size — must match the one used during embedding.
        original_bgr: If provided, the sync-embedded original image is used
                      as reference for Fourier-Mellin geometric correction
                      before extraction.  Pass the watermarked (pre-attack)
                      image stored alongside the rate_map.

    Returns:
        Decoded payload as 1-D uint8 bit array of length ``n_bits``.
    """
    # --- Optional geometric correction ------------------------------------
    if original_bgr is not None:
        from .geometric_sync import correct_attacked_image
        image_bgr = correct_attacked_image(original_bgr, image_bgr)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    h, w = Y.shape
    n_rows = h // BLOCK_SIZE
    n_cols = w // BLOCK_SIZE

    # --- Vectorised block DCT -------------------------------------------
    s_r, s_c = Y.strides
    blocks = as_strided(
        Y,
        shape=(n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE),
        strides=(s_r * BLOCK_SIZE, s_c * BLOCK_SIZE, s_r, s_c),
    )
    # (n_rows, n_cols, 64) flat coefficient tensor
    dct_flat: np.ndarray = np.asarray(
        dctn(np.ascontiguousarray(blocks), norm="ortho", axes=(-2, -1))
    ).reshape(n_rows, n_cols, BLOCK_SIZE * BLOCK_SIZE)

    # --- Per-tier decoding -----------------------------------------------
    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(
        set(float(r) for r in np.unique(rounded_map)), reverse=True
    )

    tier_results: list[tuple[np.ndarray, float]] = []  # (decoded_bits, weight)

    for tier_rate in unique_rates:
        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        br_arr, bc_arr = np.where(tier_mask)
        tier_coords = list(zip(br_arr.tolist(), bc_arr.tolist()))
        if not tier_coords:
            continue

        # Determine codeword length (must match embedder exactly)
        dummy = np.zeros(n_bits, dtype=np.uint8)
        codeword_len = len(ecc_engine.encode_block(dummy, tier_rate, scheme))

        # Accumulate per-position vote counts: votes[p, 0]=count(bit=0), [p, 1]=count(bit=1)
        votes = np.zeros((codeword_len, 2), dtype=np.int32)
        bit_idx = 0

        for (br, bc) in tier_coords:
            coeffs = dct_flat[br, bc]
            for coeff_idx in EMBED_COEFF_INDICES:
                p = bit_idx % codeword_len
                bit = _decode_coeff(float(coeffs[coeff_idx]), alpha)
                votes[p, bit] += 1
                bit_idx += 1

        # Majority vote at each codeword position → hard decisions for RS decoder
        raw_codeword = (votes[:, 1] >= votes[:, 0]).astype(np.uint8)

        # ECC decode
        decoded = ecc_engine.decode_block(
            raw_codeword, tier_rate, scheme, n_payload=n_bits
        )

        # Weight = ECC rate: higher rate → more parity → more reliable
        tier_results.append((decoded, tier_rate))

    if not tier_results:
        return np.zeros(n_bits, dtype=np.uint8)

    # --- Weighted cross-tier vote ----------------------------------------
    total_w = sum(w for _, w in tier_results)
    soft = np.zeros(n_bits, dtype=np.float64)
    for decoded, weight in tier_results:
        arr = np.zeros(n_bits, dtype=np.uint8)
        copy_len = min(n_bits, len(decoded))
        arr[:copy_len] = decoded[:copy_len]
        soft += weight * arr.astype(np.float64)

    return (soft / total_w >= 0.5).astype(np.uint8)