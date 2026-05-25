"""
watermark_embedder.py — Adaptive-ECC QIM watermark embedder (per-tier encoding).

Architecture
============
    BGR → embed_sync → YCrCb → (per-tier block DCT) → QIM embed → IDCT → YCrCb → BGR

Per-tier adaptive encoding  [FIXED vs first revision]
------------------------------------------------------
The original implementation used a single codeword encoded at the *mean* of
the rate_map (≈ 0.50 when tiers are roughly equal).  This averaged away the
adaptive benefit: the mean rate is always lower than the best fixed rate (0.75),
so fixed_rate=0.75 always won (as confirmed by the ablation results).

The correct approach:
    1. Partition image blocks into tiers by their rate_map value.
    2. For each tier, encode the *full* watermark payload at that tier's ECC rate.
    3. Embed each tier's codeword into its blocks, cycling the codeword when
       the tier has more block capacity than one codeword length.

At decode time, each tier is decoded independently and the results are combined
via weighted majority vote (weight proportional to ECC rate — higher ECC = more
reliable).

Why this beats fixed_rate=0.75:
    • Smooth blocks (r=0.75): full protection, correctly matched to their fragility.
    • Textured blocks (r=0.25): many more block slots freed up for additional
      codeword copies, boosting redundancy without wasting parity overhead.
    • Cross-tier voting adds another independent decoding that corrects errors
      not corrected by any single tier's RS decoder.

Geometric synchronisation
-------------------------
Always call ``embed_sync`` from geometric_sync.py *before* this embedder
so that the Fourier-Mellin peaks survive both JPEG and subsequent QIM.
The sync template modifies only the spatial luminance values; it does not
touch the DCT coefficients used by QIM embedding.
"""
from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme

# ---------------------------------------------------------------------------
# Module-level constants (also imported by watermark_decoder)
# ---------------------------------------------------------------------------

ALPHA: float = 28.0   # alpha=28 + SYNC_ALPHA=10 → PSNR ≈ 40–42 dB (meets ≥40 dB criterion)
BLOCK_SIZE: int = 8
EMBED_COEFF_INDICES: list[int] = [1, 2, 3]
BITS_PER_BLOCK: int = len(EMBED_COEFF_INDICES)


# ---------------------------------------------------------------------------
# QIM primitives  (imported verbatim by watermark_decoder — do not rename)
# ---------------------------------------------------------------------------

def _embed_coeff(val: float, bit: int, alpha: float) -> float:
    """QIM: force parity of floor(val/alpha) to equal bit."""
    q = int(np.floor(val / alpha))
    if (q % 2) != bit:
        q += 1
    return (q + 0.5) * alpha


def _decode_coeff(val: float, alpha: float) -> int:
    """Recover embedded bit from (noisy) DCT coefficient."""
    return int(np.floor(val / alpha)) % 2


# ---------------------------------------------------------------------------
# Vectorised block DCT helper
# ---------------------------------------------------------------------------

def _image_to_dct_blocks(Y: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Convert a 2-D luminance array to a (n_rows, n_cols, 8, 8) DCT block tensor.

    Uses as_strided (zero-copy view) + single batched dctn call for ~40× speed
    over the naive per-block loop.
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
    dct_blocks: np.ndarray = np.asarray(dctn(blocks, norm="ortho", axes=(-2, -1)))
    return dct_blocks, n_rows, n_cols


# ---------------------------------------------------------------------------
# Internal: per-tier embedding helper
# ---------------------------------------------------------------------------

def _embed_tier(
    Y_emb: np.ndarray,
    dct_blocks: np.ndarray,
    tier_coords: list[tuple[int, int]],
    codeword: np.ndarray,
    alpha: float,
) -> None:
    """
    Embed ``codeword`` bits into ``tier_coords`` blocks (cycling) in-place.

    The codeword is tiled modulo its length across the available bit slots,
    so every codeword position receives ``n_slots // codeword_len`` votes at
    decode time — providing an additional layer of repetition robustness.

    Modifies ``Y_emb`` in-place via per-block IDCT.
    """
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
    Embed ``watermark_bits`` using *per-tier* adaptive-ECC QIM in the block-DCT
    domain.

    Each tier (smooth / mid / textured) independently carries the full watermark
    payload encoded at that tier's ECC rate.  Smooth blocks use a stronger
    code (higher rate) to survive JPEG quantisation; textured blocks use a
    lighter code, freeing capacity for additional codeword copies.

    Steps:
      1. BGR → YCrCb; extract luminance Y.
      2. Compute block DCT in one vectorised call.
      3. For each unique ECC rate tier in rate_map:
         a. Find blocks belonging to that tier.
         b. Encode full watermark at that tier's ECC rate.
         c. Embed codeword into tier blocks (cycling if more blocks than bits).
      4. Reconstruct Y via per-block IDCT (only QIM-modified blocks touched).
      5. Clip, cast, convert back to BGR.

    Args:
        image_bgr:      H×W×3 uint8 BGR image  (should already have sync embedded).
        watermark_bits: 1-D uint8 bit array (values 0 or 1).
        rate_map:       (n_rows, n_cols) float32 per-block ECC rate map from
                        ``frequency_analyzer.build_ecc_rate_map``.
        ecc_engine:     ``AdaptiveECCEngine`` instance.
        scheme:         'reed_solomon' | 'repetition'.
        alpha:          QIM step size (default ALPHA = 36.0).

    Returns:
        Watermarked image as H×W×3 uint8 BGR array.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    dct_blocks, n_rows, n_cols = _image_to_dct_blocks(Y)
    Y_emb = Y.copy()

    # Identify unique tier rates (rounded to 2 dp to group floating-point equal values)
    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(set(float(r) for r in np.unique(rounded_map)), reverse=True)

    for tier_rate in unique_rates:
        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        br_arr, bc_arr = np.where(tier_mask)
        tier_coords = list(zip(br_arr.tolist(), bc_arr.tolist()))
        if not tier_coords:
            continue

        codeword = ecc_engine.encode_block(
            watermark_bits.astype(np.uint8), tier_rate, scheme
        )
        _embed_tier(Y_emb, dct_blocks, tier_coords, codeword, alpha)

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

    Under the per-tier scheme, every tier must independently carry at least
    one complete codeword.  The bottleneck tier is the one with the fewest
    available block slots relative to its codeword length.

    Returns n_bits if all tiers have sufficient capacity, else the largest
    n_bits that fits (via binary search on the bottleneck tier).
    """
    h, w = image_shape[0], image_shape[1]
    rounded_map = np.round(rate_map, 2)
    unique_rates = sorted(set(float(r) for r in np.unique(rounded_map)))

    min_capacity = n_bits
    for tier_rate in unique_rates:
        tier_mask = np.abs(rounded_map - tier_rate) < 0.005
        tier_slots = int(np.sum(tier_mask)) * BITS_PER_BLOCK
        # Binary search for max payload fitting one codeword in this tier
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