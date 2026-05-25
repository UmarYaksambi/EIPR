"""
baseline_comparison.py — Adaptive ECC vs baseline watermarking methods.

Baselines
---------
1. **FixedRateWatermarker** — same QIM block-DCT embedder, constant ECC rate.
   Isolates the contribution of the adaptive rate map (Table 2 / Table 3).

2. **LSBWatermarker** — spatial LSB substitution in the Y channel.
   Fragile under JPEG (expected BER ≈ 0.5); sets lower-bound reference.

3. **SSWatermarker** — additive spread-spectrum in global DCT (Cox et al. 1997).
   No error correction; robust to mild attacks but not JPEG q=30 or regeneration.

All methods expose ``embed`` / ``extract`` with compatible signatures so they
slot into the experiment runner without modification.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine
from .frequency_analyzer import (
    compute_block_dct_variance,
    build_ecc_rate_map,
    calibrate_thresholds,
)
from .watermark_embedder import (
    embed_watermark,
    ALPHA,
    BLOCK_SIZE,
    BITS_PER_BLOCK,
    EMBED_COEFF_INDICES,
)
from .watermark_decoder import extract_watermark
from .metrics import bit_error_rate, normalized_correlation, image_psnr, image_ssim
from .attack_suite import ATTACK_SUITE, BASELINE_ATTACKS


# ===========================================================================
# 1. Fixed-rate ECC baseline
# ===========================================================================

class FixedRateWatermarker:
    """
    Same QIM block-DCT pipeline as the proposed method, but with a uniform
    (non-adaptive) ECC rate across every block.

    The fixed rate is supplied at construction time so the caller can sweep
    over {0.25, 0.50, 0.75} to reproduce the ablation study.
    """

    def __init__(self, fixed_rate: float = 0.50) -> None:
        self.fixed_rate = float(np.clip(fixed_rate, 0.0, 0.99))
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
        """Embed watermark; return (watermarked_image, rate_map)."""
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
        original_bgr: np.ndarray | None = None,
    ) -> np.ndarray:
        return extract_watermark(
            image_bgr, rate_map, self._engine, n_bits, scheme,  # type: ignore[arg-type]
            original_bgr=original_bgr,
        )


# ===========================================================================
# 2. LSB spatial baseline
# ===========================================================================

class LSBWatermarker:
    """
    Least-Significant-Bit substitution in the luminance (Y) channel.

    Capacity  = H × W bits (one bit per pixel).
    No error-correction — any compression quantisation immediately flips bits.
    Expected BER under JPEG q=50: ≈ 0.40–0.50 (near random).
    """

    def embed(
        self,
        image_bgr: np.ndarray,
        watermark_bits: np.ndarray,
    ) -> np.ndarray:
        """Replace LSBs of Y-channel pixels. Returns watermarked BGR image."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).copy()
        Y = ycrcb[:, :, 0].flatten().copy()
        n = len(watermark_bits)
        if n > len(Y):
            raise ValueError(
                f"LSB capacity = {len(Y)} bits but watermark has {n} bits."
            )
        Y[:n] = (Y[:n] & 0xFE) | watermark_bits[:n].astype(np.uint8)
        ycrcb[:, :, 0] = Y.reshape(ycrcb.shape[:2])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def extract(self, image_bgr: np.ndarray, n_bits: int) -> np.ndarray:
        """Read LSBs of Y-channel. Returns decoded bit array."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        flat = ycrcb[:, :, 0].flatten()
        return (flat[:n_bits] & 0x01).astype(np.uint8)


# ===========================================================================
# 3. Spread-Spectrum DCT baseline  (Cox et al., 1997)
# ===========================================================================

class SSWatermarker:
    """
    Additive spread-spectrum watermarking in the global 2-D DCT domain.

    Embedding: w[k] += α · b_i · c[i, k]
    where b_i ∈ {-1, +1} is the bipolar watermark symbol,
    c[i, k] is the pseudo-random PN carrier for bit i,
    and k indexes the selected mid-frequency DCT coefficients.

    Detection uses a correlation decoder (non-blind — requires original image
    as side information, consistent with the non-blind adaptive-ECC scheme).

    Reference: I. J. Cox, J. Kilian, F. T. Leighton, T. Shamoon.
               "Secure Spread Spectrum Watermarking for Multimedia."
               IEEE Trans. Image Process., 6(12):1673–1687, 1997.
    """

    def __init__(self, alpha: float = 8.0, seed: int = 99) -> None:
        self.alpha = float(alpha)
        self.seed = int(seed)

    def _carrier(self, n_bits: int, n_coeffs: int) -> np.ndarray:
        """Pseudo-random {-1, +1} carrier matrix [n_bits × n_coeffs]."""
        rng = np.random.default_rng(self.seed)
        return rng.choice([-1.0, 1.0], size=(n_bits, n_coeffs))

    def _mid_freq_indices(self, flat_size: int, n_coeffs: int) -> np.ndarray:
        """
        Select indices of the ``n_coeffs`` highest-energy mid-frequency DCT
        coefficients from a flat DCT array of ``flat_size`` elements.

        'Mid-frequency' is defined as the range [flat_size//8, flat_size//8 + 4*n_coeffs].
        This avoids DC (too perceptible) and very high frequencies (too noisy).
        """
        mid_start = flat_size // 8
        mid_end   = mid_start + n_coeffs * 4
        mid_end   = min(mid_end, flat_size)
        # Return indices relative to the full flat array
        return np.arange(mid_start, mid_end)

    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        """Add spread-spectrum delta to selected global DCT coefficients."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float64)
        dct_full: np.ndarray = np.asarray(dctn(Y, norm="ortho"))
        flat = dct_full.flatten().copy()

        n_bits = len(watermark_bits)
        n_coeffs = n_bits * 8
        candidate_idx = self._mid_freq_indices(len(flat), n_coeffs)
        # Sort by energy descending to pick the most robust coefficients
        energy_order = np.argsort(np.abs(flat[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        bipolar = watermark_bits.astype(np.float64) * 2.0 - 1.0   # {-1, +1}
        carrier = self._carrier(n_bits, n_coeffs)                  # (n_bits, n_coeffs)
        delta   = (bipolar[:, None] * carrier).sum(axis=0)         # (n_coeffs,)
        flat[abs_idx] += self.alpha * delta

        Y_wm: np.ndarray = np.asarray(idctn(flat.reshape(dct_full.shape), norm="ortho"))
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
        Correlation decoder (non-blind — original image used for subtraction).

        The detector stores the PN carrier and original DCT; in deployment
        the original is available at the provider's detector.
        """
        def _dct_flat(img: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[:, :, 0].astype(np.float64)
            return np.asarray(dctn(Y, norm="ortho")).flatten()

        flat_orig = _dct_flat(original_bgr)
        flat_recv = _dct_flat(image_bgr)
        diff = flat_recv - flat_orig

        n_coeffs = n_bits * 8
        candidate_idx = self._mid_freq_indices(len(flat_orig), n_coeffs)
        energy_order  = np.argsort(np.abs(flat_orig[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        carrier = self._carrier(n_bits, n_coeffs)
        correlations = carrier @ diff[abs_idx]
        return (correlations >= 0).astype(np.uint8)

    def extract_blind(self, image_bgr: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Blind correlation decoder — does NOT use the original image.

        Correlates the PN carrier directly against the received DCT coefficients
        without subtracting the host image.  Expected BER ≈ 0.45–0.50.

        Included as ``spread_spectrum_blind`` in Table 3 to show that SS's low
        BER (non-blind row) is contingent on storing the original at the detector,
        a much stronger requirement than our method's key-only detection.
        """
        def _dct_flat(img: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[:, :, 0].astype(np.float64)
            return np.asarray(dctn(Y, norm="ortho")).flatten()

        flat_recv = _dct_flat(image_bgr)
        n_coeffs  = n_bits * 8
        candidate_idx = self._mid_freq_indices(len(flat_recv), n_coeffs)
        energy_order  = np.argsort(np.abs(flat_recv[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        carrier      = self._carrier(n_bits, n_coeffs)
        correlations = carrier @ flat_recv[abs_idx]
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
    Evaluate all baselines + proposed adaptive-ECC over *images* under *attacks*.

    Pipeline (identical to run_full_experiment — no geometric sync):
      embed_watermark → attack → extract_watermark

    embed_sync() is deliberately excluded: SYNC_FREQS at 32 cycles/image collides
    with QIM flat index 1 (also 32 cycles for 512×512/8×8), corrupting QIM decisions.
    See geometric_sync.py SYNC_FREQS comment for the full analysis.

    Returns:
        Nested dict::

            {
              "adaptive_ecc":    {attack_name: {"BER_mean": ..., "BER_std": ..., ...}},
              "fixed_rate_0.25": {...},
              "fixed_rate_0.50": {...},
              "fixed_rate_0.75": {...},
              "lsb":             {...},
              "spread_spectrum": {...},
              "spread_spectrum_blind": {...},
            }
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    # NOTE: embed_sync is deliberately NOT called here.
    # SYNC_FREQS at 32 cycles/image collides with QIM flat index 1 (also 32 cycles/image
    # for 512×512 / 8×8 blocks), corrupting QIM decisions and producing BER≈0.38.
    # See geometric_sync.py for the full collision analysis.

    if attacks is None:
        attacks = BASELINE_ATTACKS

    rng = np.random.default_rng(seed)
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)

    engine = AdaptiveECCEngine()
    scheme: str = cfg.get("ecc", {}).get("scheme", "reed_solomon")
    tau_low  = float((cfg.get("ecc") or {}).get("tau_low")  or 50.0)
    tau_high = float((cfg.get("ecc") or {}).get("tau_high") or 200.0)

    lsb_bl  = LSBWatermarker()
    ss_bl   = SSWatermarker()
    fixed_bls = {
        "fixed_rate_0.25": FixedRateWatermarker(0.25),
        "fixed_rate_0.50": FixedRateWatermarker(0.50),
        "fixed_rate_0.75": FixedRateWatermarker(0.75),
    }

    method_keys = ["adaptive_ecc", *fixed_bls.keys(), "lsb", "spread_spectrum", "spread_spectrum_blind"]
    all_results: dict[str, dict[str, dict]] = {k: {} for k in method_keys}

    def _adaptive_rate_map(img: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        return build_ecc_rate_map(var_map, tau_low, tau_high)

    attack_iter = tqdm(attacks.items(), desc="Attacks") if _tqdm_available else attacks.items()

    for attack_name, attack_fn in attack_iter:
        print(f"  [baseline] Attack: {attack_name} | n_images={len(images)}")

        adap_bers:  list[float] = []
        adap_psnrs: list[float] = []
        fixed_bers: dict[str, list[float]] = {k: [] for k in fixed_bls}
        lsb_bers:      list[float] = []
        ss_bers:       list[float] = []
        ss_blind_bers: list[float] = []

        img_iter = (
            tqdm(images, desc=f"  {attack_name}", leave=False)
            if _tqdm_available else images
        )

        for img in img_iter:
            # ---- Proposed adaptive-ECC ----------------------------------
            rate_map = _adaptive_rate_map(img)
            wm = embed_watermark(img, watermark, rate_map, engine, scheme)  # type: ignore[arg-type]
            attacked = attack_fn(wm)                                         # type: ignore[operator]
            dec = extract_watermark(                                          # type: ignore[arg-type]
                attacked, rate_map, engine, n_bits, scheme,
            )
            adap_bers.append(bit_error_rate(watermark, dec))
            adap_psnrs.append(image_psnr(img, wm))

            # ---- Fixed-rate ECC baselines --------------------------------
            for key, fb in fixed_bls.items():
                wm_f, rm_f = fb.embed(img, watermark, scheme)
                att_f = attack_fn(wm_f)                                      # type: ignore[operator]
                dec_f = fb.extract(att_f, rm_f, n_bits, scheme)
                fixed_bers[key].append(bit_error_rate(watermark, dec_f))

            # ---- LSB (no geometric correction — LSB has no sync) ---------
            wm_lsb = lsb_bl.embed(img, watermark)
            att_lsb = attack_fn(wm_lsb)                                          # type: ignore[operator]
            dec_lsb = lsb_bl.extract(att_lsb, n_bits)
            lsb_bers.append(bit_error_rate(watermark, dec_lsb))

            # ---- Spread-Spectrum (non-blind; subtracts original) ---------
            wm_ss = ss_bl.embed(img, watermark)
            att_ss = attack_fn(wm_ss)                                        # type: ignore[operator]
            dec_ss = ss_bl.extract(att_ss, img, n_bits)
            ss_bers.append(bit_error_rate(watermark, dec_ss))

            # ---- Spread-Spectrum blind (no original) ---------------------
            dec_ss_blind = ss_bl.extract_blind(att_ss, n_bits)
            ss_blind_bers.append(bit_error_rate(watermark, dec_ss_blind))

        # Aggregate
        all_results["adaptive_ecc"][attack_name] = {
            "BER_mean":  float(np.mean(adap_bers)),
            "BER_std":   float(np.std(adap_bers)),
            "PSNR_mean": float(np.mean(adap_psnrs)),
        }
        for key in fixed_bls:
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
        all_results["spread_spectrum_blind"][attack_name] = {
            "BER_mean": float(np.mean(ss_blind_bers)),
            "BER_std":  float(np.std(ss_blind_bers)),
        }

    return all_results