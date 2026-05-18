"""
Baseline comparison — Adaptive ECC Watermarking.

Implements three baselines compared against the proposed adaptive-ECC scheme
in the paper:

  1. **Fixed-rate ECC** (``FixedRateWatermarker``)
     Same QIM block-DCT embedder, but with a uniform ECC rate across every
     block (no frequency-adaptive allocation).  Used in Table 2 to isolate
     the contribution of the adaptive rate map.

  2. **Blind LSB** (``LSBWatermarker``)
     Classic spatial-domain least-significant-bit substitution in the
     luminance channel.  Extremely fragile under JPEG / noise — sets a
     lower-bound reference BER.

  3. **Spread-Spectrum DCT** (``SSWatermarker``)
     Additive spread-spectrum embedding in the DCT domain (Cox et al., 1997).
     Robust under mild attacks but lacks error-correction.

All baselines expose ``embed`` / ``extract`` that match the signature of
``watermark_embedder.embed_watermark`` / ``watermark_decoder.extract_watermark``
so they can be plugged into ``experiment_runner`` without modification.

Usage (from experiment_runner or a notebook):
    from src.baseline_comparison import run_baseline_comparison
    results = run_baseline_comparison(cfg, images)
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine
from .frequency_analyzer import compute_block_dct_variance, build_ecc_rate_map, calibrate_thresholds
from .watermark_embedder import (
    embed_watermark,
    ALPHA,
    BLOCK_SIZE,
    BITS_PER_BLOCK,
    EMBED_COEFF_INDICES,
    _embed_coeff,
    _decode_coeff,
)
from .watermark_decoder import extract_watermark
from .metrics import bit_error_rate, normalized_correlation, image_psnr, image_ssim
from .attack_suite import ATTACK_SUITE


# ===========================================================================
# 1. Fixed-rate ECC baseline
# ===========================================================================

class FixedRateWatermarker:
    """
    Identical pipeline to the adaptive embedder but with a constant ECC rate.

    Matches bit budget of adaptive scheme by using the *mean* adaptive rate
    as the fixed rate, making comparisons fair.
    """

    def __init__(self, fixed_rate: float = 0.50) -> None:
        self.fixed_rate = fixed_rate
        self._engine = AdaptiveECCEngine()

    def _make_rate_map(self, image_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        return np.full(var_map.shape, self.fixed_rate, dtype=np.float32)

    def embed(
        self,
        image_bgr: np.ndarray,
        watermark_bits: np.ndarray,
        scheme: str = "reed_solomon",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (watermarked_image, rate_map)."""
        rate_map = self._make_rate_map(image_bgr)
        watermarked = embed_watermark(
            image_bgr, watermark_bits, rate_map, self._engine, scheme  # type: ignore[arg-type]
        )
        return watermarked, rate_map

    def extract(
        self,
        image_bgr: np.ndarray,
        rate_map: np.ndarray,
        n_bits: int,
        scheme: str = "reed_solomon",
    ) -> np.ndarray:
        return extract_watermark(
            image_bgr, rate_map, self._engine, n_bits, scheme  # type: ignore[arg-type]
        )


# ===========================================================================
# 2. LSB spatial baseline
# ===========================================================================

class LSBWatermarker:
    """
    Least-Significant-Bit substitution in the luminance channel (Y of YCrCb).

    Embedding capacity  = H * W  bits (one bit per pixel).
    No error-correction — every bit flip due to compression is unrecoverable.

    Expected BER under JPEG q=50: ~0.40 (near random), confirming fragility.
    """

    def embed(
        self,
        image_bgr: np.ndarray,
        watermark_bits: np.ndarray,
    ) -> np.ndarray:
        """Embed watermark_bits into LSBs of the Y channel. Returns watermarked image."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).copy()
        Y = ycrcb[:, :, 0]
        h, w = Y.shape
        capacity = h * w
        n = len(watermark_bits)
        if n > capacity:
            raise ValueError(
                f"LSB capacity={capacity} bits < watermark length={n} bits."
            )
        flat = Y.flatten().copy()
        flat[:n] = (flat[:n] & 0xFE) | watermark_bits.astype(np.uint8)
        ycrcb[:, :, 0] = flat.reshape(h, w)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def extract(
        self,
        image_bgr: np.ndarray,
        n_bits: int,
    ) -> np.ndarray:
        """Read LSBs of the Y channel to recover watermark bits."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0]
        flat = Y.flatten()
        return (flat[:n_bits] & 0x01).astype(np.uint8)


# ===========================================================================
# 3. Spread-Spectrum DCT baseline  (Cox et al., 1997)
# ===========================================================================

class SSWatermarker:
    """
    Additive spread-spectrum watermarking in the global DCT domain.

    Embeds the watermark as a pseudo-random ±alpha perturbation added to
    the highest-energy mid-frequency DCT coefficients of the full image.
    Detected by correlation (soft decision), decoded by thresholding.

    No error-correction — robust to mild geometric/compression attacks but
    breaks under strong JPEG or regeneration.
    """

    def __init__(self, alpha: float = 8.0, seed: int = 99) -> None:
        self.alpha = alpha
        self.seed = seed

    def _carrier(self, n_bits: int, n_coeffs: int) -> np.ndarray:
        """Pseudo-random {-1, +1} carrier matrix [n_bits x n_coeffs]."""
        rng = np.random.default_rng(self.seed)
        return rng.choice([-1.0, 1.0], size=(n_bits, n_coeffs))

    def embed(
        self,
        image_bgr: np.ndarray,
        watermark_bits: np.ndarray,
    ) -> np.ndarray:
        """Add spread-spectrum watermark to mid-frequency DCT coefficients."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float64)
        dct_full: np.ndarray = np.asarray(dctn(Y, norm="ortho"))

        n_bits = len(watermark_bits)
        flat = dct_full.flatten()
        # Select the n_bits * 8 highest-energy mid-frequency coefficients
        n_coeffs = n_bits * 8
        mid_start = flat.size // 8          # skip DC and very-low-freq
        mid_end = mid_start + n_coeffs * 4  # window wide enough
        energy_idx = np.argsort(np.abs(flat[mid_start:mid_end]))[::-1][:n_coeffs]
        abs_idx = energy_idx + mid_start

        bipolar = (watermark_bits.astype(np.float64) * 2.0 - 1.0)  # {-1, +1}
        carrier = self._carrier(n_bits, n_coeffs)  # (n_bits, n_coeffs)
        delta = (bipolar[:, None] * carrier).sum(axis=0)             # (n_coeffs,)
        flat[abs_idx] += self.alpha * delta

        Y_wm = np.asarray(idctn(flat.reshape(dct_full.shape), norm="ortho"))
        ycrcb_out = ycrcb.copy()
        ycrcb_out[:, :, 0] = np.clip(Y_wm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)

    def extract(
        self,
        image_bgr: np.ndarray,
        original_bgr: np.ndarray,
        n_bits: int,
    ) -> np.ndarray:
        """
        Correlation detector (non-blind — requires original for subtraction).

        In a real deployment the detector would store the carrier and the
        original DCT; here we use the original image as side information.
        """
        def _dct_flat(img: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[:, :, 0].astype(np.float64)
            return np.asarray(dctn(Y, norm="ortho")).flatten()

        flat_orig = _dct_flat(original_bgr)
        flat_recv = _dct_flat(image_bgr)
        diff = flat_recv - flat_orig

        n_coeffs = n_bits * 8
        mid_start = flat_orig.size // 8
        mid_end = mid_start + n_coeffs * 4
        energy_idx = np.argsort(np.abs(flat_orig[mid_start:mid_end]))[::-1][:n_coeffs]
        abs_idx = energy_idx + mid_start

        carrier = self._carrier(n_bits, n_coeffs)
        correlations = carrier @ diff[abs_idx]  # (n_bits,)
        return (correlations >= 0).astype(np.uint8)


# ===========================================================================
# Comparison runner
# ===========================================================================

def run_baseline_comparison(
    cfg: dict,
    images: list[np.ndarray],
    n_bits: int = 64,
    seed: int = 42,
    attacks: dict | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Run all three baselines + the proposed adaptive-ECC scheme over *images*
    under *attacks*, returning nested results suitable for table generation.

    Returns:
        {
          "adaptive_ecc": { attack_name: { "BER_mean": ..., ... }, ... },
          "fixed_rate_25": { ... },
          "fixed_rate_50": { ... },
          "fixed_rate_75": { ... },
          "lsb":           { ... },
          "spread_spectrum": { ... },
        }
    """
    if attacks is None:
        attacks = ATTACK_SUITE

    rng = np.random.default_rng(seed)
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)

    engine = AdaptiveECCEngine()
    scheme = cfg.get("ecc", {}).get("scheme", "reed_solomon")
    tau_low = float((cfg.get("ecc") or {}).get("tau_low") or 50.0)
    tau_high = float((cfg.get("ecc") or {}).get("tau_high") or 200.0)

    lsb_baseline = LSBWatermarker()
    ss_baseline = SSWatermarker()
    fixed_baselines = {
        "fixed_rate_25": FixedRateWatermarker(0.25),
        "fixed_rate_50": FixedRateWatermarker(0.50),
        "fixed_rate_75": FixedRateWatermarker(0.75),
    }

    # Pre-compute per-image rate maps for the adaptive scheme
    def _adaptive_rate_map(img: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        return build_ecc_rate_map(var_map, tau_low, tau_high)

    all_results: dict[str, dict[str, dict]] = {
        "adaptive_ecc": {},
        **{k: {} for k in fixed_baselines},
        "lsb": {},
        "spread_spectrum": {},
    }

    for attack_name, attack_fn in attacks.items():
        print(f"  [baseline] Attack: {attack_name}")

        adap_bers, adap_psnrs = [], []
        fixed_bers: dict[str, list] = {k: [] for k in fixed_baselines}
        lsb_bers, ss_bers = [], []

        for img in images:
            # ---- Adaptive ECC (proposed) ----
            rate_map = _adaptive_rate_map(img)
            wm = embed_watermark(img, watermark, rate_map, engine, scheme)  # type: ignore[arg-type]
            attacked = attack_fn(wm)  # type: ignore[operator]
            dec = extract_watermark(attacked, rate_map, engine, n_bits, scheme)  # type: ignore[arg-type]
            adap_bers.append(bit_error_rate(watermark, dec))
            adap_psnrs.append(image_psnr(img, wm))

            # ---- Fixed-rate ECC baselines ----
            for key, fb in fixed_baselines.items():
                wm_f, rm_f = fb.embed(img, watermark, scheme)
                att_f = attack_fn(wm_f)  # type: ignore[operator]
                dec_f = fb.extract(att_f, rm_f, n_bits, scheme)
                fixed_bers[key].append(bit_error_rate(watermark, dec_f))

            # ---- LSB ----
            wm_lsb = lsb_baseline.embed(img, watermark)
            att_lsb = attack_fn(wm_lsb)  # type: ignore[operator]
            dec_lsb = lsb_baseline.extract(att_lsb, n_bits)
            lsb_bers.append(bit_error_rate(watermark, dec_lsb))

            # ---- Spread-Spectrum ----
            wm_ss = ss_baseline.embed(img, watermark)
            att_ss = attack_fn(wm_ss)  # type: ignore[operator]
            dec_ss = ss_baseline.extract(att_ss, img, n_bits)
            ss_bers.append(bit_error_rate(watermark, dec_ss))

        all_results["adaptive_ecc"][attack_name] = {
            "BER_mean": float(np.mean(adap_bers)),
            "BER_std":  float(np.std(adap_bers)),
            "PSNR_mean": float(np.mean(adap_psnrs)),
        }
        for key in fixed_baselines:
            all_results[key][attack_name] = {
                "BER_mean": float(np.mean(fixed_bers[key])),
                "BER_std":  float(np.std(fixed_bers[key])),
            }
        all_results["lsb"][attack_name] = {
            "BER_mean": float(np.mean(lsb_bers)),
            "BER_std":  float(np.std(lsb_bers)),
        }
        all_results["spread_spectrum"][attack_name] = {
            "BER_mean": float(np.mean(ss_bers)),
            "BER_std":  float(np.std(ss_bers)),
        }

    return all_results