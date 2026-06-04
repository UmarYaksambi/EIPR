from __future__ import annotations

import warnings
import numpy as np
import cv2
from typing import NamedTuple

SYNC_ALPHA: float = 10.0
SYNC_FREQS: list[tuple[float, float]] = [
    (32.0,   0.0), ( 0.0,  32.0),
    (32.0,  32.0), (32.0, -32.0),
]
SEARCH_RADIUS: int = 15
MIN_PEAK_SNR: float = 5.0

def _build_template(h: int, w: int) -> np.ndarray:
    xs  = np.arange(w, dtype=np.float64)[None, :]
    ys  = np.arange(h, dtype=np.float64)[:, None]
    tpl = np.zeros((h, w), dtype=np.float64)
    for fu, fv in SYNC_FREQS:
        tpl += np.cos(2.0 * np.pi * (fu * xs / w + fv * ys / h))
    peak = float(np.max(np.abs(tpl))) + 1e-12
    return tpl * (SYNC_ALPHA / peak)

def embed_sync_chroma(image_bgr: np.ndarray, channel: int = 1) -> np.ndarray:
    if channel not in (1, 2):
        raise ValueError(f"channel must be 1 (Cr) or 2 (Cb), got {channel}")
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    C     = ycrcb[:, :, channel].astype(np.float64)
    h, w  = C.shape
    out   = ycrcb.copy()
    out[:, :, channel] = np.clip(C + _build_template(h, w), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

class GeomTransform(NamedTuple):
    angle_deg: float
    scale: float
    tx: float
    ty: float

def _magnitude_spectrum(Y: np.ndarray) -> np.ndarray:
    h, w = Y.shape
    win  = np.outer(np.hanning(h), np.hanning(w))
    F    = np.fft.fftshift(np.fft.fft2(Y.astype(np.float64) * win))
    return np.abs(F)

def _find_peak(mag: np.ndarray, expected_fu: float, expected_fv: float, search_radius: int) -> tuple[float, float, float]:
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

    idx = np.unravel_index(np.argmax(patch), patch.shape)
    
    # Sub-pixel peak estimation using Center of Mass
    r_idx, c_idx = idx[0], idx[1]
    r_s_sub = max(0, r_idx - 1); r_e_sub = min(patch.shape[0], r_idx + 2)
    c_s_sub = max(0, c_idx - 1); c_e_sub = min(patch.shape[1], c_idx + 2)
    neighborhood = patch[r_s_sub:r_e_sub, c_s_sub:c_e_sub]
    mass = neighborhood.sum() + 1e-12
    rr, cc = np.indices(neighborhood.shape)
    r_offset = (rr * neighborhood).sum() / mass - (r_idx - r_s_sub)
    c_offset = (cc * neighborhood).sum() / mass - (c_idx - c_s_sub)

    found_row = row_s + r_idx + r_offset
    found_col = col_s + c_idx + c_offset
    return float(found_col - cx), float(cy - found_row), snr

def estimate_transform(attacked_bgr: np.ndarray) -> GeomTransform:
    ycrcb = cv2.cvtColor(attacked_bgr, cv2.COLOR_BGR2YCrCb)
    C_att = ycrcb[:, :, 1].astype(np.float32)
    mag   = _magnitude_spectrum(C_att)

    found_pts, expect_pts = [], []
    for fu, fv in SYNC_FREQS:
        for sign in (1.0, -1.0):
            efu, efv = fu * sign, fv * sign
            ffu, ffv, snr = _find_peak(mag, efu, efv, SEARCH_RADIUS)
            if snr >= MIN_PEAK_SNR and (abs(ffu) > 1 or abs(ffv) > 1):
                found_pts.append((ffu, ffv))
                expect_pts.append((efu, efv))

    if len(found_pts) < 2:
        return GeomTransform(angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0)

    expected_c = np.array([complex(eu, ev) for eu, ev in expect_pts])
    found_c    = np.array([complex(fu, fv) for fu, fv in found_pts])
    denom      = float(np.sum(np.abs(expected_c) ** 2))
    if denom < 1e-9:
        return GeomTransform(angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0)

    A          = np.sum(found_c * np.conj(expected_c)) / denom
    freq_scale = float(np.clip(abs(A), 0.4, 2.5))
    angle_deg  = float(np.clip(np.degrees(np.angle(A)), -45.0, 45.0))
    return GeomTransform(angle_deg=angle_deg, scale=freq_scale, tx=0.0, ty=0.0)

def correct_transform(image_bgr: np.ndarray, transform: GeomTransform) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    freq_scale = float(transform.scale)
    angle_deg  = float(transform.angle_deg)

    # Increased thresholds to prevent false positives on pure valumetric attacks
    if abs(freq_scale - 1.0) < 0.04 and abs(angle_deg) < 1.5:
        return image_bgr.copy()

    s_spatial = float(freq_scale)
    theta_rad = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cx, cy = w / 2.0, h / 2.0

    a00 = s_spatial * cos_t;  a01 = s_spatial * (-sin_t)
    a10 = s_spatial * sin_t;  a11 = s_spatial * cos_t
    t0  = cx - a00 * cx - a01 * cy
    t1  = cy - a10 * cx - a11 * cy

    M = np.float32([[a00, a01, t0], [a10, a11, t1]])
    return cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def correct_attacked_image(attacked_bgr: np.ndarray) -> np.ndarray:
    try:
        t = estimate_transform(attacked_bgr)
        return correct_transform(attacked_bgr, t)
    except Exception as exc:
        return attacked_bgr