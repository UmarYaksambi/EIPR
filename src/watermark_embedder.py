from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine, ECCScheme
from .geometric_sync import embed_sync_chroma

ALPHA: float = 28.0
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

        codeword = ecc_engine.encode_block(
            watermark_bits.astype(np.uint8), tier_rate, scheme
        )
        _embed_tier(Y_emb, dct_blocks, tier_coords, codeword, alpha)

    ycrcb_out = ycrcb.copy()
    ycrcb_out[:, :, 0] = np.clip(Y_emb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)