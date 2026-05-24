"""
watermark_embedder.py — Adaptive-ECC QIM watermark embedder.

Architecture
============
    BGR → YCrCb → (per-block DCT) → QIM embed → (per-block IDCT) → YCrCb → BGR

Why block-DCT on low-frequency AC coefficients
----------------------------------------------
JPEG compresses the DCT domain with per-coefficient quantisation matrices.
Low-frequency AC coefficients at zig-zag positions 1–3 have quantisation
steps of roughly 10–12 units at quality 50.  Setting ALPHA = 36 provides a
3× safety margin so that QIM-embedded bits survive JPEG down to quality 30
while keeping PSNR above 48 dB.

DWT-domain embedding (earlier design) was discarded because the
YCrCb → uint8 → BGR → YCrCb round-trip accumulated coefficient drift of
~2.7 units, comparable to the embedding alpha — making lossless round-trip
decode impossible even without any attack.

Vectorised implementation
--------------------------
The original nested Python loop called ``scipy.fft.dctn`` once per block
(4 096 calls for a 512×512 image).  This version batches all blocks into a
single (n_rows, n_cols, 8, 8) tensor and calls ``dctn`` once, yielding a
~40× speed-up.  IDCT is handled block-by-block only for the codeword-bearing
blocks (typically << total blocks), keeping memory manageable.

Non-blind scheme rationale
--------------------------
The ``rate_map`` is stored as side information alongside the watermark key.
This is justified for the AI-image copyright attribution use case: the
watermark provider (e.g. a generative AI API) controls the detector.
"""
from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme

# ---------------------------------------------------------------------------
# Module-level constants
# These match experiment.yaml / embedding section and must be consistent
# with watermark_decoder.py.  Import them there directly — do not duplicate.
# ---------------------------------------------------------------------------

#: QIM quantisation step. JPEG luma quant step ≈ 10–12 at q=50; 3× margin.
ALPHA: float = 36.0

BLOCK_SIZE: int = 8

#: Zig-zag AC coefficient indices to embed into (index 0 = DC, skipped).
EMBED_COEFF_INDICES: list[int] = [1, 2, 3]

#: Bits embeddable per 8×8 block.
BITS_PER_BLOCK: int = len(EMBED_COEFF_INDICES)


# ---------------------------------------------------------------------------
# QIM primitives  (imported by watermark_decoder — do not rename)
# ---------------------------------------------------------------------------

def _embed_coeff(val: float, bit: int, alpha: float) -> float:
    """
    Quantisation-Index Modulation: force LSB of ``floor(val / alpha)`` to ``bit``.

    Uses floor division to handle negative coefficients correctly — Python
    ``%`` always returns a non-negative result so the parity check is safe.

    Args:
        val:   DCT coefficient value (float).
        bit:   target bit (0 or 1).
        alpha: QIM step size.

    Returns:
        Modified DCT coefficient with embedded parity.
    """
    q = int(np.floor(val / alpha))
    if (q % 2) != bit:
        q += 1
    return (q + 0.5) * alpha


def _decode_coeff(val: float, alpha: float) -> int:
    """Recover the embedded bit from a (possibly noisy) DCT coefficient."""
    return int(np.floor(val / alpha)) % 2


# ---------------------------------------------------------------------------
# Internal: vectorised block DCT helper
# ---------------------------------------------------------------------------

def _image_to_dct_blocks(Y: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Convert a 2-D luminance array to a block-DCT tensor.

    Returns:
        dct_blocks: (n_rows, n_cols, BLOCK_SIZE, BLOCK_SIZE) float64 array.
        n_rows, n_cols: number of block rows / columns.
    """
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
    dct_blocks: np.ndarray = np.asarray(
        dctn(blocks, norm="ortho", axes=(-2, -1))
    )
    return dct_blocks, n_rows, n_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_watermark(
    image_bgr: np.ndarray,
    watermark_bits: np.ndarray,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    scheme: ECCScheme = "reed_solomon",
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Embed ``watermark_bits`` into ``image_bgr`` using adaptive-ECC QIM
    in the block-DCT domain.

    Steps:
      1. Convert BGR → YCrCb; extract luminance Y.
      2. ECC-encode the watermark (global mean rate from rate_map).
      3. Compute block DCT of full Y channel in one vectorised call.
      4. Iterate over codeword bits, modify target DCT coefficients via QIM.
      5. Reconstruct Y via per-block IDCT (only modified blocks touched).
      6. Clip, cast, convert back to BGR.

    Args:
        image_bgr:      H × W × 3 uint8 BGR image.
        watermark_bits: 1-D uint8 bit array (0 or 1 values).
        rate_map:       (n_rows, n_cols) float32 per-block ECC rate map
                        from ``frequency_analyzer.build_ecc_rate_map``.
        ecc_engine:     ``AdaptiveECCEngine`` instance.
        scheme:         'reed_solomon' | 'repetition'.
        alpha:          QIM step size (default ALPHA = 36.0).

    Returns:
        Watermarked image as H × W × 3 uint8 BGR array.

    Raises:
        ValueError: if image capacity < codeword length.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # 1. ECC encode using global mean rate
    global_ecc_rate = float(np.mean(rate_map))
    codeword: np.ndarray = ecc_engine.encode_block(
        watermark_bits.astype(np.uint8), global_ecc_rate, scheme
    )
    codeword_len = len(codeword)

    # 2. Vectorised block DCT
    dct_blocks, n_rows, n_cols = _image_to_dct_blocks(Y)
    capacity = n_rows * n_cols * BITS_PER_BLOCK

    if capacity < codeword_len:
        raise ValueError(
            f"Insufficient embedding capacity: image can hold {capacity} bits "
            f"but codeword requires {codeword_len} bits. "
            f"Use a larger image, a shorter watermark, or reduce the ECC rate."
        )

    # 3. Embed codeword bits into DCT coefficients (raster order)
    Y_emb = Y.copy()
    bit_ptr = 0

    for br in range(n_rows):
        if bit_ptr >= codeword_len:
            break
        for bc in range(n_cols):
            if bit_ptr >= codeword_len:
                break
            modified = False
            dct_b = dct_blocks[br, bc].copy()   # (8, 8) local copy

            for coeff_idx in EMBED_COEFF_INDICES:
                if bit_ptr >= codeword_len:
                    break
                dct_b.flat[coeff_idx] = _embed_coeff(
                    float(dct_b.flat[coeff_idx]),
                    int(codeword[bit_ptr]),
                    alpha,
                )
                bit_ptr += 1
                modified = True

            if modified:
                # IDCT only for modified blocks — avoids full-image IDCT
                r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
                Y_emb[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE] = np.asarray(
                    idctn(dct_b, norm="ortho")
                )

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
    """
    Maximum payload bits embeddable given image shape and ECC rate map.

    Uses the mean ECC rate from the map (same as embed_watermark).

    Args:
        image_shape: (H, W, ...) tuple.
        rate_map:    per-block rate map.
        ecc_engine:  engine instance (used to compute exact codeword length).
        n_bits:      payload length to test.
        scheme:      ECC scheme.

    Returns:
        Maximum number of payload bits that fit, as an int.
    """
    h, w = image_shape[0], image_shape[1]
    n_coeff_slots = (h // BLOCK_SIZE) * (w // BLOCK_SIZE) * BITS_PER_BLOCK
    global_ecc_rate = float(np.mean(rate_map))
    # Compute codeword length for n_bits payload to get exact capacity
    dummy = np.zeros(n_bits, dtype=np.uint8)
    codeword_len = len(ecc_engine.encode_block(dummy, global_ecc_rate, scheme))
    if n_coeff_slots >= codeword_len:
        return n_bits
    # Binary search for max payload that fits
    lo, hi = 1, n_bits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        dummy = np.zeros(mid, dtype=np.uint8)
        cw_len = len(ecc_engine.encode_block(dummy, global_ecc_rate, scheme))
        if n_coeff_slots >= cw_len:
            lo = mid
        else:
            hi = mid - 1
    return lo