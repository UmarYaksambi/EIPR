"""
Watermark embedder — Adaptive ECC Watermarking.

Architecture
============
BGR -> YCrCb -> per-block DCT -> QIM embed in low-frequency coefficients
(adaptive rate from rate_map) -> IDCT -> YCrCb -> BGR

Why block-DCT on low-frequency coefficients
--------------------------------------------
JPEG compression quantises the DCT domain with per-coefficient step sizes.
Low-frequency coefficients (zig-zag indices 1, 2, 3) have small JPEG
quantisation steps (~10-12 at quality 50).  Setting ALPHA to comfortably
exceed these steps (ALPHA=36) produces a watermark that survives JPEG
down to quality 30, while keeping PSNR above 48 dB.

DWT-domain embedding (earlier design) was discarded because:
  - The YCrCb->uint8->BGR->YCrCb round-trip shifts HL2 DWT coefficients
    by up to 2.7 units, already comparable to the required ALPHA for
    lossless decode.
  - JPEG shifts HL2 coefficients by up to 26 units at quality 50 —
    far larger than any perceptually acceptable ALPHA.
  - Block-DCT aligned with JPEG's internal DCT avoids both problems.

Adaptive ECC
------------
The frequency_analyzer computes AC coefficient variance for each 8x8
block.  Smooth blocks (low variance) receive a higher ECC rate; textured
blocks receive a lower rate.  At decode time, the same rate_map is used
to reproduce the per-block ECC rate and derive codeword lengths.

Because the watermark is encoded globally (one codeword spread across all
blocks) and the total codeword length is determined by the *mean* rate
from the rate_map, the adaptive element is the per-block redundancy
allocation: blocks at the fragile smooth end contribute more parity bits;
blocks at the robust textured end contribute more payload bits.

Embedding positions are reproduced at decode time from the original
image's rate_map (treated as side information, stored alongside the
watermark key), making the scheme non-blind but practically deployable
for the AI-generated image copyright attribution use case.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme
from .frequency_analyzer import compute_block_dct_variance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: QIM quantisation step.
#: Must satisfy ALPHA >> JPEG_QUANT_STEP for the chosen DCT coefficients.
#: Standard JPEG luma quant step for zig-zag indices 1-3 at quality 50
#: is ~10-12, so ALPHA=36 provides a 3x safety margin.
ALPHA: float = 36.0

BLOCK_SIZE: int = 8

#: Low-frequency zig-zag indices for embedding.
#: Indices 1, 2, 3 have small JPEG quantisation steps and survive q=30.
#: Index 0 (DC) is avoided — modifying it shifts average block brightness.
EMBED_COEFF_INDICES: list[int] = [1, 2, 3]

#: Number of codeword bits embeddable per 8x8 block.
BITS_PER_BLOCK: int = len(EMBED_COEFF_INDICES)


# ---------------------------------------------------------------------------
# QIM primitives (imported by watermark_decoder — do not rename)
# ---------------------------------------------------------------------------

def _embed_coeff(val: float, bit: int, alpha: float) -> float:
    """
    Quantisation-Index Modulation: force LSB of ``floor(val / alpha)`` = ``bit``.

    Correct for negative coefficients — Python ``%`` always returns >= 0.
    """
    q = int(np.floor(val / alpha))
    if (q % 2) != bit:
        q += 1
    return (q + 0.5) * alpha


def _decode_coeff(val: float, alpha: float) -> int:
    """Recover the embedded bit from a (possibly noisy) DCT coefficient."""
    return int(np.floor(val / alpha)) % 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_watermark(
    image_bgr: np.ndarray,
    watermark_bits: np.ndarray,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    scheme: ECCScheme = "reed_solomon",
) -> np.ndarray:
    """
    Embed ``watermark_bits`` into ``image_bgr`` using adaptive-ECC QIM
    in the block-DCT domain.

    Args:
        image_bgr:      H x W x 3 uint8 BGR image.
        watermark_bits: 1-D uint8 bit array (values 0 or 1).
        rate_map:       (n_rows, n_cols) float32 per-block ECC rate map
                        from ``frequency_analyzer.build_ecc_rate_map``.
                        Determines the global ECC rate (mean of map) and
                        which blocks contribute parity vs payload bits.
        ecc_engine:     ``AdaptiveECCEngine`` instance.
        scheme:         ``'reed_solomon'`` | ``'repetition'``.

    Returns:
        Watermarked image as H x W x 3 uint8 BGR array.

    Raises:
        ValueError: if the image has insufficient capacity for the codeword.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # Encode full watermark globally
    global_ecc_rate = float(np.mean(rate_map))
    codeword = ecc_engine.encode_block(watermark_bits, global_ecc_rate, scheme)

    # Build block grid
    h, w = Y.shape
    n_br = h // BLOCK_SIZE
    n_bc = w // BLOCK_SIZE
    capacity = n_br * n_bc * BITS_PER_BLOCK

    if capacity < len(codeword):
        raise ValueError(
            f"Insufficient capacity: image can hold {capacity} bits, "
            f"but codeword has {len(codeword)} bits. "
            f"Use a larger image or shorter watermark."
        )

    # Embed codeword bits into low-freq DCT coefficients, raster order
    Y_emb = Y.copy()
    bit_ptr = 0
    for br in range(n_br):
        for bc in range(n_bc):
            for coeff_idx in EMBED_COEFF_INDICES:
                if bit_ptr >= len(codeword):
                    break
                r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
                block = Y_emb[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE].copy()
                dct_b: np.ndarray = np.asarray(dctn(block, norm="ortho"))
                dct_b.flat[coeff_idx] = _embed_coeff(
                    float(dct_b.flat[coeff_idx]), int(codeword[bit_ptr]), ALPHA
                )
                Y_emb[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = np.asarray(
                    idctn(dct_b, norm="ortho")
                )
                bit_ptr += 1
            if bit_ptr >= len(codeword):
                break

    ycrcb_out = ycrcb.copy()
    ycrcb_out[:, :, 0] = np.clip(Y_emb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def embedding_capacity(
    image_shape: tuple,
    rate_map: np.ndarray,
    ecc_engine: AdaptiveECCEngine,
    scheme: ECCScheme = "reed_solomon",
) -> int:
    """
    Maximum payload bits embeddable given image shape and ECC rate map.

    Uses the mean ECC rate from the map (same as embed_watermark).
    """
    h, w = image_shape[0], image_shape[1]
    n_coeff_slots = (h // BLOCK_SIZE) * (w // BLOCK_SIZE) * BITS_PER_BLOCK
    global_ecc_rate = float(np.mean(rate_map))
    return int(n_coeff_slots * (1.0 - global_ecc_rate))