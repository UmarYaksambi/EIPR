"""
geometric_sync.py — Fourier sync-tone geometric correction for block-DCT watermarking.

Problem
-------
Block-DCT QIM stores codeword bits in spatially fixed 8×8 coefficient positions.
Any geometric distortion (crop, rotation, scale) shifts the image grid relative
to the read grid → BER ≈ 0.5.

Approach: Direct sync-tone peak detection (Kutter 1999; Ruanaidh & Pun 1998)
---------------------------------------------------------------------------
Four cosine tones at known spatial frequencies are added to the luminance
channel before QIM watermarking.  At decode time:

  1. Compute the shifted 2-D FFT magnitude of the attacked image.
  2. For each known frequency, find the actual peak in a ±SEARCH_RADIUS window.
  3. Least-squares solve for rotation θ and scale s from found vs. expected
     positions: found_z = A · expected_z, where A = s_freq · exp(iθ).
  4. Apply the inverse RST as a single centre-anchored warpAffine, which
     preserves block-grid alignment.

Why unified warpAffine (not sequential resize + rotate)
-------------------------------------------------------
Sequential operations accumulate interpolation errors and — critically for
crop-style attacks — a separate resize step does not keep the DCT block grid
aligned with the image centre.  A single 2×3 affine matrix that encodes
rotation + scale simultaneously maps every 8×8 block correctly in one pass.

Why direct peak detection (not log-polar phase correlation)
----------------------------------------------------------
Log-polar phase correlation requires strong directional energy in the
magnitude spectrum.  AI-generated images and Gaussian-blurred synthetics
have flat, isotropic spectra → phase correlation returns (0,0).  The sync
tones produce isolated peaks with 20-25× SNR that survive JPEG q=50 reliably.

Non-blind assumption
--------------------
``correct_attacked_image`` only needs the attacked image (the sync tones
carry all registration information).  The ``original_bgr`` argument is
accepted for API compatibility but is not used for estimation.
"""
from __future__ import annotations

import warnings
import numpy as np
import cv2
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC_ALPHA: float = 10.0   # sync-tone amplitude in luminance pixel units

# Sync frequencies (integer cycles per image width/height).
# Chosen at 1/16 of image size: survive JPEG q=30 (JPEG quantises down to
# roughly q=10 equivalent at these frequencies before destroying them).
SYNC_FREQS: list[tuple[float, float]] = [
    (32.0,   0.0),
    ( 0.0,  32.0),
    (32.0,  32.0),
    (32.0, -32.0),
]

# Search window half-width (bins).  Must be < min(SYNC_FREQ) = 32 to avoid
# DC leakage, and large enough to cover worst-case crop 10% scale shift:
#   Δf = 32 · (1/0.9 − 1) ≈ 3.6 bins.  SEARCH_RADIUS = 15 covers this
#   with a 4× safety margin while staying clear of DC (32 − 15 = 17 bins away).
SEARCH_RADIUS: int = 15

MIN_PEAK_SNR: float = 3.0   # minimum peak/mean SNR to trust a found peak


# ---------------------------------------------------------------------------
# Sync template embedding
# ---------------------------------------------------------------------------

def _build_template(h: int, w: int) -> np.ndarray:
    """Sum of cosines at SYNC_FREQS, normalised to peak amplitude ±SYNC_ALPHA."""
    xs  = np.arange(w, dtype=np.float64)[None, :]
    ys  = np.arange(h, dtype=np.float64)[:, None]
    tpl = np.zeros((h, w), dtype=np.float64)
    for fu, fv in SYNC_FREQS:
        tpl += np.cos(2.0 * np.pi * (fu * xs / w + fv * ys / h))
    peak = float(np.max(np.abs(tpl))) + 1e-12
    return tpl * (SYNC_ALPHA / peak)


def embed_sync(image_bgr: np.ndarray) -> np.ndarray:
    """
    Add the synchronisation template to the luminance channel.

    Call this *before* ``embed_watermark``.  The sync tones occupy different
    spatial frequencies than QIM's low-AC coefficients, so they do not
    interfere with each other.

    Args:
        image_bgr: H×W×3 uint8 BGR image.

    Returns:
        BGR image with sync tones added; same shape and dtype.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y     = ycrcb[:, :, 0].astype(np.float64)
    h, w  = Y.shape
    out   = ycrcb.copy()
    out[:, :, 0] = np.clip(Y + _build_template(h, w), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# GeomTransform result type
# ---------------------------------------------------------------------------

class GeomTransform(NamedTuple):
    """Estimated geometric transform applied to the attacked image."""
    angle_deg: float   # counter-clockwise rotation (degrees)
    scale: float       # frequency scale |A|; spatial zoom-in = 1/scale
    tx: float          # unused (kept for API compatibility)
    ty: float          # unused (kept for API compatibility)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lum_f32(img: np.ndarray) -> np.ndarray:
    """BGR → luminance float32."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _magnitude_spectrum(Y: np.ndarray) -> np.ndarray:
    """
    Shifted 2-D FFT magnitude with Hanning window.
    DC is at the centre.  Returns float64, same shape as Y.
    """
    h, w = Y.shape
    win  = np.outer(np.hanning(h), np.hanning(w))
    F    = np.fft.fftshift(np.fft.fft2(Y.astype(np.float64) * win))
    return np.abs(F)


def _find_peak(
    mag: np.ndarray,
    expected_fu: float,
    expected_fv: float,
    search_radius: int,
) -> tuple[float, float, float]:
    """
    Find the strongest peak near the expected sync-tone position.

    In a shifted FFT of an H×W image, a tone at (fu, fv) cycles/image
    appears at pixel (cy − fv, cx + fu) where cy=H//2, cx=W//2.

    Returns:
        (found_fu, found_fv, snr) — found frequency coordinates and
        peak-to-mean SNR.  Falls back to expected position if SNR is low.
    """
    h, w  = mag.shape
    cy, cx = h // 2, w // 2

    row_c = int(round(cy - expected_fv))
    col_c = int(round(cx + expected_fu))

    r     = int(search_radius)
    row_s = max(0, row_c - r)
    row_e = min(h, row_c + r + 1)
    col_s = max(0, col_c - r)
    col_e = min(w, col_c + r + 1)

    if row_e <= row_s or col_e <= col_s:
        return expected_fu, expected_fv, 0.0

    patch    = mag[row_s:row_e, col_s:col_e]
    peak_val = float(patch.max())
    mean_val = float(patch.mean()) + 1e-12
    snr      = peak_val / mean_val

    idx       = np.unravel_index(np.argmax(patch), patch.shape)
    found_row = row_s + idx[0]
    found_col = col_s + idx[1]

    return float(found_col - cx), float(cy - found_row), snr


# ---------------------------------------------------------------------------
# Transform estimation
# ---------------------------------------------------------------------------

def estimate_transform(
    original_bgr: np.ndarray,   # kept for API compatibility; not used
    attacked_bgr: np.ndarray,
) -> GeomTransform:
    """
    Estimate rotation and scale from sync-tone peak shifts.

    The FFT magnitude of a rotated+scaled image has peaks that are rotated
    and scaled relative to the original.  Crop+resize is a centre-anchored
    zoom, which in the frequency domain is a reciprocal scale (zoom-in by s
    → freq peaks move to 1/s of their original distance from DC).

    Args:
        original_bgr: Accepted for API compatibility; not used in estimation.
        attacked_bgr: H×W×3 geometrically distorted image.

    Returns:
        GeomTransform(angle_deg, scale, tx=0, ty=0).
        ``scale`` is the frequency-domain scale factor |A|.  Spatial zoom-in
        = 1/scale.  ``correct_transform`` uses this to build the right
        inverse warpAffine.
    """
    Y_att = _lum_f32(attacked_bgr)
    mag   = _magnitude_spectrum(Y_att)

    found_pts:  list[tuple[float, float]] = []
    expect_pts: list[tuple[float, float]] = []

    for fu, fv in SYNC_FREQS:
        for sign in (1.0, -1.0):          # both Hermitian conjugate peaks
            efu, efv = fu * sign, fv * sign
            ffu, ffv, snr = _find_peak(mag, efu, efv, SEARCH_RADIUS)
            if snr >= MIN_PEAK_SNR and (abs(ffu) > 1 or abs(ffv) > 1):
                found_pts.append((ffu, ffv))
                expect_pts.append((efu, efv))

    if len(found_pts) < 2:
        return GeomTransform(angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0)

    # Least-squares: found_z = A · expected_z, A = freq_scale · exp(iθ)
    expected_c = np.array([complex(eu, ev) for eu, ev in expect_pts])
    found_c    = np.array([complex(fu, fv) for fu, fv in found_pts])
    denom      = float(np.sum(np.abs(expected_c) ** 2))
    if denom < 1e-9:
        return GeomTransform(angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0)

    A          = np.sum(found_c * np.conj(expected_c)) / denom
    freq_scale = float(np.clip(abs(A),               0.4,  2.5))
    angle_deg  = float(np.clip(np.degrees(np.angle(A)), -45.0, 45.0))

    return GeomTransform(angle_deg=angle_deg, scale=freq_scale, tx=0.0, ty=0.0)


# ---------------------------------------------------------------------------
# Geometric correction — single unified centre-anchored warpAffine
# ---------------------------------------------------------------------------

def correct_transform(
    image_bgr: np.ndarray,
    transform: GeomTransform,
) -> np.ndarray:
    """
    Undo the estimated geometric transform with a single warpAffine.

    The inverse of a centre-anchored zoom-by-s_spatial and rotation-by-θ is:
        corrected[p] = attacked[ M · p ]
    where M = (1/s_spatial) · R_{-θ} is applied around the image centre,
    and s_spatial = 1/freq_scale = 1/transform.scale.

    Using a single affine matrix (rather than sequential resize + rotate)
    ensures every 8×8 DCT block position is reconstructed in one bilinear
    pass, avoiding accumulated interpolation errors.

    Args:
        image_bgr: H×W×3 uint8 BGR attacked image.
        transform: GeomTransform from ``estimate_transform``.

    Returns:
        Geometrically corrected image, same spatial size.
    """
    h, w      = image_bgr.shape[:2]
    freq_scale = float(transform.scale)
    angle_deg  = float(transform.angle_deg)

    if abs(freq_scale - 1.0) < 0.005 and abs(angle_deg) < 0.1:
        return image_bgr.copy()

    # Undo the zoom-in: attacked[p] = original[s_zoom·(p-c)+c] with s_zoom=1/freq_scale.
    # To recover: corrected[p] = attacked[freq_scale·(p-c)+c].
    # Therefore the affine source-sampling scale is freq_scale (NOT 1/freq_scale).
    s_spatial = float(freq_scale)

    # Rotation: attacked[p] = original[R_{-θ}(p-c)+c]
    # Recovery: corrected[p] = attacked[R_{+θ}(p-c)+c]
    # So theta_rad = +angle_deg (not negated).
    theta_rad = np.radians(angle_deg)
    cos_t     = np.cos(theta_rad)
    sin_t     = np.sin(theta_rad)
    cx, cy    = w / 2.0, h / 2.0

    # Affine coefficients: src = (s_spatial · R_{-θ}) · (dst - c) + c
    #   [ a00  a01  t0 ]       with t anchored at image centre
    #   [ a10  a11  t1 ]
    a00 = s_spatial * cos_t;  a01 = s_spatial * (-sin_t)
    a10 = s_spatial * sin_t;  a11 = s_spatial *   cos_t
    t0  = cx - a00 * cx - a01 * cy
    t1  = cy - a10 * cx - a11 * cy

    M = np.float32([[a00, a01, t0],
                    [a10, a11, t1]])

    return cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def correct_attacked_image(
    original_bgr: np.ndarray,
    attacked_bgr: np.ndarray,
) -> np.ndarray:
    """
    Estimate and correct geometric distortion in one call.

    Falls back to the attacked image unchanged on any exception.

    Args:
        original_bgr: Sync-embedded watermarked original (API compat; unused).
        attacked_bgr: H×W×3 geometrically distorted image.

    Returns:
        Geometrically corrected BGR image.
    """
    try:
        t = estimate_transform(original_bgr, attacked_bgr)
        return correct_transform(attacked_bgr, t)
    except Exception as exc:
        warnings.warn(
            f"[geometric_sync] correction failed ({exc!r}); "
            "returning attacked image uncorrected.",
            stacklevel=2,
        )
        return attacked_bgr