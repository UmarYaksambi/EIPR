from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def bit_error_rate(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    Fraction of bit positions that differ between original and decoded.

    Shorter array sets the comparison length (safe for truncated decodings).
    Returns a value in [0.0, 1.0]; 0.0 is perfect, 0.5 is random.
    """
    min_len = min(len(original_bits), len(decoded_bits))
    if min_len == 0:
        return 1.0
    errors = int(np.sum(original_bits[:min_len] != decoded_bits[:min_len]))
    return errors / min_len


def normalized_correlation(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    Normalised cross-correlation between original and decoded bit sequences,
    mapped to {-1, +1} bipolar representation.

    Returns a value in [-1.0, 1.0]; 1.0 is perfect, 0.0 is random,
    -1.0 is fully inverted.
    """
    min_len = min(len(original_bits), len(decoded_bits))
    if min_len == 0:
        return 0.0
    o = original_bits[:min_len].astype(np.float64) * 2.0 - 1.0
    d = decoded_bits[:min_len].astype(np.float64) * 2.0 - 1.0
    denom = np.linalg.norm(o) * np.linalg.norm(d) + 1e-8
    return float(np.dot(o, d) / denom)


def image_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB) between original and watermarked images.

    Target for perceptual invisibility: PSNR >= 40 dB.
    """
    return float(peak_signal_noise_ratio(original, watermarked, data_range=255))


def image_ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between original and watermarked images.

    Returns a value in [-1.0, 1.0]; values above 0.95 are imperceptible.
    """
    return float(
        structural_similarity(
            original,
            watermarked,
            channel_axis=2,   # channel_axis replaces multichannel kwarg in skimage >= 0.19
            data_range=255,
        )
    )