"""
watermark_decoder.py — Adaptive-ECC QIM watermark extractor.

Mirrors the embedder's block-DCT transform chain exactly.
The ``rate_map`` is supplied as side information (stored at embed time
alongside the watermark key).

Vectorisation
-------------
DCT of all blocks is computed in one batched call (same strategy as the
embedder) so the per-block coefficient read loop only touches the flat
coefficient array rather than re-calling ``dctn`` per block.

Zero-padding guard
------------------
If the image is too small to fill the entire codeword (shouldn't happen
in normal use but can occur in edge-case tests), the raw codeword array
is zero-padded before RS decoding to keep array lengths consistent with
the encoder's expectation.
"""
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


def extract_watermark(
    image_bgr: np.ndarray,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    n_bits: int,
    scheme: ECCScheme = "reed_solomon",
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Extract and ECC-decode the watermark from a (possibly attacked) image.

    Args:
        image_bgr:  H × W × 3 uint8 BGR image (possibly attacked).
        rate_map:   (n_rows, n_cols) float32 per-block ECC rate map —
                    must *exactly* match the map used during embedding.
        ecc_engine: ``AdaptiveECCEngine`` instance.
        n_bits:     Number of payload bits to recover.
        scheme:     ECC scheme — must match the one used during embedding.
        alpha:      QIM step size — must match the one used during embedding.

    Returns:
        Decoded payload as 1-D uint8 bit array of length ``n_bits``.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # --- Derive expected codeword length (must be identical to embedder) --
    global_ecc_rate = float(np.mean(rate_map))
    dummy_payload = np.zeros(n_bits, dtype=np.uint8)
    codeword_len = len(ecc_engine.encode_block(dummy_payload, global_ecc_rate, scheme))

    # --- Vectorised block DCT (one call for all blocks) -------------------
    h, w = Y.shape
    n_rows = h // BLOCK_SIZE
    n_cols = w // BLOCK_SIZE

    s_r, s_c = Y.strides
    blocks = as_strided(
        Y,
        shape=(n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE),
        strides=(s_r * BLOCK_SIZE, s_c * BLOCK_SIZE, s_r, s_c),
    )
    # (n_rows, n_cols, 64) — flat coefficients for each block
    dct_flat: np.ndarray = np.asarray(
        dctn(np.ascontiguousarray(blocks), norm="ortho", axes=(-2, -1))
    ).reshape(n_rows, n_cols, BLOCK_SIZE * BLOCK_SIZE)

    # --- Read raw codeword bits in raster order ---------------------------
    raw_codeword = np.zeros(codeword_len, dtype=np.uint8)
    bit_ptr = 0

    for br in range(n_rows):
        if bit_ptr >= codeword_len:
            break
        for bc in range(n_cols):
            if bit_ptr >= codeword_len:
                break
            coeffs = dct_flat[br, bc]           # (64,) view — no copy
            for coeff_idx in EMBED_COEFF_INDICES:
                if bit_ptr >= codeword_len:
                    break
                raw_codeword[bit_ptr] = _decode_coeff(float(coeffs[coeff_idx]), alpha)
                bit_ptr += 1

    # Zero-padding guard: if image is smaller than expected, bit_ptr < codeword_len
    # raw_codeword is pre-initialised to zero so padding is implicit.
    # (A warning is appropriate for debugging; suppress in batch runs.)
    if bit_ptr < codeword_len:
        import warnings
        warnings.warn(
            f"extract_watermark: image has only {bit_ptr} embeddable coefficient "
            f"slots but codeword_len={codeword_len}. "
            f"Trailing {codeword_len - bit_ptr} bits zero-padded — BER will be elevated.",
            stacklevel=2,
        )

    # --- ECC decode -------------------------------------------------------
    decoded = ecc_engine.decode_block(
        raw_codeword, global_ecc_rate, scheme, n_payload=n_bits
    )
    # Guarantee output length even if decoder returns fewer bits
    out = np.zeros(n_bits, dtype=np.uint8)
    copy_len = min(n_bits, len(decoded))
    out[:copy_len] = decoded[:copy_len]
    return out