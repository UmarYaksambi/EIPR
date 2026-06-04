from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import dctn, idctn

from .ecc_engine import AdaptiveECCEngine
from .frequency_analyzer import compute_block_dct_variance, build_ecc_rate_map, calibrate_thresholds
from .watermark_embedder import embed_watermark, ALPHA, BLOCK_SIZE, BITS_PER_BLOCK, EMBED_COEFF_INDICES
from .watermark_decoder import extract_watermark
from .metrics import bit_error_rate, normalized_correlation, image_psnr, image_ssim
from .attack_suite import ATTACK_SUITE, BASELINE_ATTACKS

class FixedRateWatermarker:
    def __init__(self, fixed_rate: float = 0.50) -> None:
        self.fixed_rate = float(np.clip(fixed_rate, 0.0, 0.99))
        self._engine = AdaptiveECCEngine()

    def _make_rate_map(self, image_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        return np.full(var_map.shape, self.fixed_rate, dtype=np.float32)

    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray, scheme: str = "reed_solomon", alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
        rate_map = self._make_rate_map(image_bgr)
        return embed_watermark(image_bgr, watermark_bits, rate_map, self._engine, scheme, alpha=alpha), rate_map

    def extract(self, image_bgr: np.ndarray, rate_map: np.ndarray, n_bits: int, scheme: str = "reed_solomon", alpha: float = ALPHA, original_bgr: np.ndarray | None = None) -> np.ndarray:
        return extract_watermark(image_bgr, rate_map, self._engine, n_bits, scheme=scheme, alpha=alpha, original_bgr=original_bgr)

class SOTA_AdaptiveDCT_2023:
    def __init__(self, alpha: float = 36.0):
        self.alpha = alpha
        self._engine = AdaptiveECCEngine()
        self._last_rate_map = None

    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        tau = np.percentile(var_map, 50)
        rate_map = np.zeros_like(var_map, dtype=np.float32)
        rate_map[var_map >= tau] = 0.50
        self._last_rate_map = rate_map
        return embed_watermark(image_bgr, watermark_bits, rate_map, self._engine, alpha=self.alpha)

    def extract(self, image_bgr: np.ndarray, n_bits: int, original_bgr: np.ndarray | None = None) -> np.ndarray:
        return extract_watermark(image_bgr, self._last_rate_map, self._engine, n_bits, alpha=self.alpha, original_bgr=original_bgr)

class LSBWatermarker:
    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).copy()
        Y = ycrcb[:, :, 0].flatten().copy()
        n = len(watermark_bits)
        if n > len(Y): raise ValueError(f"LSB capacity = {len(Y)} bits but watermark has {n} bits.")
        Y[:n] = (Y[:n] & 0xFE) | watermark_bits[:n].astype(np.uint8)
        ycrcb[:, :, 0] = Y.reshape(ycrcb.shape[:2])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def extract(self, image_bgr: np.ndarray, n_bits: int) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        flat = ycrcb[:, :, 0].flatten()
        return (flat[:n_bits] & 0x01).astype(np.uint8)

class SSWatermarker:
    def __init__(self, alpha: float = 8.0, seed: int = 99) -> None:
        self.alpha, self.seed = float(alpha), int(seed)

    def _carrier(self, n_bits: int, n_coeffs: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.choice([-1.0, 1.0], size=(n_bits, n_coeffs))

    def _mid_freq_indices(self, flat_size: int, n_coeffs: int) -> np.ndarray:
        mid_start, mid_end = flat_size // 8, flat_size // 8 + n_coeffs * 4
        return np.arange(mid_start, min(mid_end, flat_size))

    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float64)
        dct_full: np.ndarray = np.asarray(dctn(Y, norm="ortho"))
        flat = dct_full.flatten().copy()

        n_bits, n_coeffs = len(watermark_bits), len(watermark_bits) * 8
        candidate_idx = self._mid_freq_indices(len(flat), n_coeffs)
        energy_order = np.argsort(np.abs(flat[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        bipolar = watermark_bits.astype(np.float64) * 2.0 - 1.0  
        carrier = self._carrier(n_bits, n_coeffs)                  
        delta   = (bipolar[:, None] * carrier).sum(axis=0)         
        flat[abs_idx] += self.alpha * delta

        Y_wm: np.ndarray = np.asarray(idctn(flat.reshape(dct_full.shape), norm="ortho"))
        ycrcb_out = ycrcb.copy()
        ycrcb_out[:, :, 0] = np.clip(Y_wm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)

    def extract(self, image_bgr: np.ndarray, original_bgr: np.ndarray, n_bits: int) -> np.ndarray:
        def _dct_flat(img: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            return np.asarray(dctn(ycrcb[:, :, 0].astype(np.float64), norm="ortho")).flatten()

        flat_orig, flat_recv = _dct_flat(original_bgr), _dct_flat(image_bgr)
        diff = flat_recv - flat_orig

        n_coeffs = n_bits * 8
        candidate_idx = self._mid_freq_indices(len(flat_orig), n_coeffs)
        energy_order  = np.argsort(np.abs(flat_orig[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        carrier = self._carrier(n_bits, n_coeffs)
        correlations = carrier @ diff[abs_idx]
        return (correlations >= 0).astype(np.uint8)

    def extract_blind(self, image_bgr: np.ndarray, n_bits: int) -> np.ndarray:
        def _dct_flat(img: np.ndarray) -> np.ndarray:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            return np.asarray(dctn(ycrcb[:, :, 0].astype(np.float64), norm="ortho")).flatten()

        flat_recv = _dct_flat(image_bgr)
        n_coeffs  = n_bits * 8
        candidate_idx = self._mid_freq_indices(len(flat_recv), n_coeffs)
        energy_order  = np.argsort(np.abs(flat_recv[candidate_idx]))[::-1]
        abs_idx = candidate_idx[energy_order[:n_coeffs]]

        carrier      = self._carrier(n_bits, n_coeffs)
        correlations = carrier @ flat_recv[abs_idx]
        return (correlations >= 0).astype(np.uint8)

def run_baseline_comparison(cfg: dict, images: list[np.ndarray], n_bits: int = 64, seed: int = 42, attacks: dict | None = None) -> dict[str, dict[str, dict]]:
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError: _tqdm_available = False

    if attacks is None: attacks = BASELINE_ATTACKS
    rng = np.random.default_rng(seed)
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)

    engine = AdaptiveECCEngine()
    scheme: str = cfg.get("ecc", {}).get("scheme", "reed_solomon")
    tau_low  = float((cfg.get("ecc") or {}).get("tau_low")  or 50.0)
    tau_high = float((cfg.get("ecc") or {}).get("tau_high") or 200.0)
    alpha = float(cfg.get("embedding", {}).get("alpha", ALPHA))

    lsb_bl  = LSBWatermarker()
    ss_bl   = SSWatermarker()
    sota_bl = SOTA_AdaptiveDCT_2023(alpha=alpha)
    fixed_bls = {
        "fixed_rate_0.25": FixedRateWatermarker(0.25),
        "fixed_rate_0.50": FixedRateWatermarker(0.50),
        "fixed_rate_0.75": FixedRateWatermarker(0.75),
    }

    method_keys = ["adaptive_ecc", "sota_adaptive_dct_2023", *fixed_bls.keys(), "lsb", "spread_spectrum", "spread_spectrum_blind"]
    all_results: dict[str, dict[str, dict]] = {k: {} for k in method_keys}

    def _adaptive_rate_map(img: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        return build_ecc_rate_map(
            var_map, tau_low, tau_high,
            r_high=float(cfg.get("ecc", {}).get("r_high", 0.75)),
            r_mid=float(cfg.get("ecc", {}).get("r_mid", 0.50)),
            r_low=float(cfg.get("ecc", {}).get("r_low", 0.25)),
        )

    attack_iter = tqdm(attacks.items(), desc="Attacks") if _tqdm_available else attacks.items()

    for attack_name, attack_fn in attack_iter:
        adap_bers, adap_psnrs, sota_bers, lsb_bers, ss_bers, ss_blind_bers = [], [], [], [], [], []
        fixed_bers: dict[str, list[float]] = {k: [] for k in fixed_bls}
        img_iter = tqdm(images, desc=f"  {attack_name}", leave=False) if _tqdm_available else images

        for img in img_iter:
            rate_map = _adaptive_rate_map(img)
            wm = embed_watermark(img, watermark, rate_map, engine, scheme, alpha=alpha) 
            attacked = attack_fn(wm)                                         
            
            # 1. Proposed Method (Triggering Geometric sync via original_bgr)
            dec = extract_watermark(attacked, rate_map, engine, n_bits, scheme=scheme, alpha=alpha, original_bgr=wm)
            adap_bers.append(bit_error_rate(watermark, dec))
            adap_psnrs.append(image_psnr(img, wm))
            
            # 2. SOTA
            wm_sota = sota_bl.embed(img, watermark)
            dec_sota = sota_bl.extract(attack_fn(wm_sota), n_bits, original_bgr=wm_sota)
            sota_bers.append(bit_error_rate(watermark, dec_sota))

            # 3. Fixed Rate
            for key, fb in fixed_bls.items():
                wm_f, rm_f = fb.embed(img, watermark, scheme, alpha=alpha)
                dec_f = fb.extract(attack_fn(wm_f), rm_f, n_bits, scheme, alpha=alpha, original_bgr=wm_f)
                fixed_bers[key].append(bit_error_rate(watermark, dec_f))

            # 4. LSB
            wm_lsb = lsb_bl.embed(img, watermark)
            dec_lsb = lsb_bl.extract(attack_fn(wm_lsb), n_bits)
            lsb_bers.append(bit_error_rate(watermark, dec_lsb))

            # 5. Spread Spectrum
            wm_ss = ss_bl.embed(img, watermark)
            att_ss = attack_fn(wm_ss)                                        
            dec_ss = ss_bl.extract(att_ss, img, n_bits)
            ss_bers.append(bit_error_rate(watermark, dec_ss))
            ss_blind_bers.append(bit_error_rate(watermark, ss_bl.extract_blind(att_ss, n_bits)))

        all_results["adaptive_ecc"][attack_name] = {"BER_mean": float(np.mean(adap_bers)), "BER_std": float(np.std(adap_bers)), "PSNR_mean": float(np.mean(adap_psnrs))}
        all_results["sota_adaptive_dct_2023"][attack_name] = {"BER_mean": float(np.mean(sota_bers)), "BER_std": float(np.std(sota_bers))}
        for key in fixed_bls: all_results[key][attack_name] = {"BER_mean": float(np.mean(fixed_bers[key])), "BER_std": float(np.std(fixed_bers[key]))}
        all_results["lsb"][attack_name] = {"BER_mean": float(np.mean(lsb_bers)), "BER_std": float(np.std(lsb_bers))}
        all_results["spread_spectrum"][attack_name] = {"BER_mean": float(np.mean(ss_bers)), "BER_std": float(np.std(ss_bers))}
        all_results["spread_spectrum_blind"][attack_name] = {"BER_mean": float(np.mean(ss_blind_bers)), "BER_std": float(np.std(ss_blind_bers))}
    return all_results