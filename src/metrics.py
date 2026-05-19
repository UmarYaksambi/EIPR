"""
metrics.py — Watermarking evaluation metrics.

Metrics used in the paper
--------------------------
BER  — Bit Error Rate ∈ [0, 1]; 0 = perfect, 0.5 = random guess.
NC   — Normalised Correlation ∈ [-1, 1]; 1 = perfect, 0 = random.
PSNR — Peak Signal-to-Noise Ratio (dB); target ≥ 40 dB for invisibility.
SSIM — Structural Similarity Index ∈ [-1, 1]; target ≥ 0.95.

Additional helpers
------------------
ber_confidence_interval — Wilson score CI for reporting (±) in tables.
detection_accuracy      — Fraction of images decoded with BER ≤ threshold.
aggregate_metrics       — Compute all metrics from lists of per-image values.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Core per-image metrics
# ---------------------------------------------------------------------------

def bit_error_rate(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    Fraction of bit positions that differ between original and decoded sequences.

    Comparison length = min(len(original), len(decoded)) to handle safely
    any length mismatch from truncated decodings or zero-padding.

    Returns a value in [0.0, 1.0]; 0.0 is perfect, 0.5 is random.
    """
    original_bits = np.asarray(original_bits, dtype=np.uint8)
    decoded_bits  = np.asarray(decoded_bits,  dtype=np.uint8)
    min_len = min(len(original_bits), len(decoded_bits))
    if min_len == 0:
        return 1.0
    errors = int(np.sum(original_bits[:min_len] != decoded_bits[:min_len]))
    return errors / min_len


def normalized_correlation(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    Normalised cross-correlation between original and decoded bit sequences
    in bipolar {-1, +1} representation.

    NC = (o · d) / (‖o‖ · ‖d‖)

    Returns a value in [-1.0, 1.0]; 1.0 = perfect, 0.0 = random, -1.0 = inverted.
    """
    original_bits = np.asarray(original_bits, dtype=np.float64)
    decoded_bits  = np.asarray(decoded_bits,  dtype=np.float64)
    min_len = min(len(original_bits), len(decoded_bits))
    if min_len == 0:
        return 0.0
    o = original_bits[:min_len] * 2.0 - 1.0    # {0,1} → {-1,+1}
    d = decoded_bits[:min_len]  * 2.0 - 1.0
    denom = (np.linalg.norm(o) * np.linalg.norm(d)) + 1e-12
    return float(np.dot(o, d) / denom)


def image_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB) between original and watermarked images.

    Computed over all channels (BGR).  Target for perceptual invisibility: ≥ 40 dB.
    """
    return float(peak_signal_noise_ratio(original, watermarked, data_range=255))


def image_ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between original and watermarked images.

    Computed over all channels (channel_axis=2).  Returns a value in [-1, 1];
    values above 0.95 indicate imperceptible embedding.
    """
    return float(
        structural_similarity(
            original,
            watermarked,
            channel_axis=2,
            data_range=255,
        )
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def ber_confidence_interval(
    bers: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wilson score 95 % confidence interval for the mean BER.

    The Wilson interval is preferred over the naïve normal approximation
    for proportions near 0 or 1 — important when BER is very low (≈ 0)
    as is the case for a correctly-functioning watermark.

    Args:
        bers:       list of per-image BER values ∈ [0, 1].
        confidence: desired confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    from scipy import stats
    bers = np.asarray(bers, dtype=np.float64)
    n = len(bers)
    if n == 0:
        return (0.0, 1.0)
    mean_ber = float(np.mean(bers))
    se = float(np.std(bers, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    alpha_ci = 1.0 - confidence
    t_val = float(stats.t.ppf(1.0 - alpha_ci / 2, df=max(1, n - 1)))
    lo = max(0.0, mean_ber - t_val * se)
    hi = min(1.0, mean_ber + t_val * se)
    return (lo, hi)


def detection_accuracy(
    bers: list[float] | np.ndarray,
    threshold: float = 0.10,
) -> float:
    """
    Fraction of images decoded with BER ≤ threshold.

    A BER ≤ 0.10 (≤ 10 % bit errors) is a common watermark detection criterion
    in the literature (cf. Cox et al. 1997; Vukotic et al. 2022).

    Args:
        bers:      list of per-image BER values.
        threshold: detection criterion (default 0.10).

    Returns:
        Detection accuracy ∈ [0.0, 1.0].
    """
    bers = np.asarray(bers, dtype=np.float64)
    if len(bers) == 0:
        return 0.0
    return float(np.mean(bers <= threshold))


def aggregate_metrics(
    watermarks:  list[np.ndarray],
    decoded_list: list[np.ndarray],
    originals:   list[np.ndarray],
    watermarked_list: list[np.ndarray],
) -> dict[str, float]:
    """
    Compute aggregate statistics across a batch of images.

    Returns a dict with keys:
        BER_mean, BER_std, BER_ci_lo, BER_ci_hi,
        NC_mean, NC_std,
        PSNR_mean, PSNR_std,
        SSIM_mean, SSIM_std,
        detection_accuracy_10pct
    """
    bers   = [bit_error_rate(wm, dec) for wm, dec in zip(watermarks, decoded_list)]
    ncs    = [normalized_correlation(wm, dec) for wm, dec in zip(watermarks, decoded_list)]
    psnrs  = [image_psnr(orig, wm) for orig, wm in zip(originals, watermarked_list)]
    ssims  = [image_ssim(orig, wm) for orig, wm in zip(originals, watermarked_list)]

    ci_lo, ci_hi = ber_confidence_interval(bers)

    return {
        "BER_mean":                float(np.mean(bers)),
        "BER_std":                 float(np.std(bers)),
        "BER_ci_lo":               ci_lo,
        "BER_ci_hi":               ci_hi,
        "NC_mean":                 float(np.mean(ncs)),
        "NC_std":                  float(np.std(ncs)),
        "PSNR_mean":               float(np.mean(psnrs)),
        "PSNR_std":                float(np.std(psnrs)),
        "SSIM_mean":               float(np.mean(ssims)),
        "SSIM_std":                float(np.std(ssims)),
        "detection_accuracy_10pct": detection_accuracy(bers, threshold=0.10),
    }