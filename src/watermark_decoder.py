"""
Watermark decoder — Adaptive ECC Watermarking.

Mirrors the embedder's block-DCT transform chain exactly.
The rate_map is passed as side information (stored at embed time).
"""
from __future__ import annotations

import numpy as np
import cv2
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
) -> np.ndarray:
    """
    Extract and ECC-decode the watermark from a (possibly attacked) image.

    Args:
        image_bgr:  H x W x 3 uint8 BGR image (possibly attacked).
        rate_map:   (n_rows, n_cols) float32 per-block ECC rate map —
                    must exactly match the map used during embedding.
        ecc_engine: ``AdaptiveECCEngine`` instance.
        n_bits:     Number of payload bits to recover.
        scheme:     ECC scheme — must match the one used during embedding.

    Returns:
        Decoded payload as 1-D uint8 bit array of length ``n_bits``.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # Derive codeword length by dummy-encoding (keeps encoder/decoder in sync)
    global_ecc_rate = float(np.mean(rate_map))
    dummy_payload = np.zeros(n_bits, dtype=np.uint8)
    codeword_len = len(ecc_engine.encode_block(dummy_payload, global_ecc_rate, scheme))

    # Read raw codeword bits from the same positions as the embedder
    h, w = Y.shape
    n_br = h // BLOCK_SIZE
    n_bc = w // BLOCK_SIZE

    raw_codeword: list = []
    bit_ptr = 0

    for br in range(n_br):
        for bc in range(n_bc):
            for coeff_idx in EMBED_COEFF_INDICES:
                if bit_ptr >= codeword_len:
                    break
                r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
                block = Y[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
                dct_b: np.ndarray = np.asarray(dctn(block, norm="ortho"))
                raw_codeword.append(
                    _decode_coeff(float(dct_b.flat[coeff_idx]), ALPHA)
                )
                bit_ptr += 1
            if bit_ptr >= codeword_len:
                break

    raw_cw_arr = np.array(raw_codeword, dtype=np.uint8)

    # ECC decode
    decoded = ecc_engine.decode_block(
        raw_cw_arr, global_ecc_rate, scheme, n_payload=n_bits
    )
    return decoded[:n_bits]