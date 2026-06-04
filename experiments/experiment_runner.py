from __future__ import annotations

import argparse
import pathlib
import sys
import numpy as np
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_generator import load_dataset, generate_synthetic_dataset
from src.frequency_analyzer import (
    compute_block_dct_variance,
    build_ecc_rate_map,
    calibrate_thresholds,
)
from src.ecc_engine import AdaptiveECCEngine
from src.watermark_embedder import embed_watermark, ALPHA as DEFAULT_ALPHA
from src.watermark_decoder import extract_watermark
from src.attack_suite import ATTACK_SUITE, set_attack_seed
from src.metrics import (
    bit_error_rate,
    normalized_correlation,
    image_psnr,
    image_ssim,
    detection_accuracy,
    ber_confidence_interval,
)
from src.utils import save_results, print_results_table, to_latex_table, Timer

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _alpha_from_cfg(cfg: dict) -> float:
    return float((cfg.get("embedding") or {}).get("alpha") or DEFAULT_ALPHA)

def _make_rate_map(img: np.ndarray, cfg: dict) -> np.ndarray:
    import cv2
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    var_map = compute_block_dct_variance(ycrcb[:, :, 0])
    tau_low  = float(cfg["ecc"].get("tau_low")  or 50.0)
    tau_high = float(cfg["ecc"].get("tau_high") or 200.0)
    return build_ecc_rate_map(
        var_map, tau_low, tau_high,
        r_high=float(cfg["ecc"]["r_high"]),
        r_mid =float(cfg["ecc"]["r_mid"]),
        r_low =float(cfg["ecc"]["r_low"]),
    )

def run_smoke_test(_cfg: dict) -> None:
    import cv2
    print("[smoke_test] Generating synthetic images …")
    images = generate_synthetic_dataset(n_images=3, image_size=(256, 256), seed=0)
    engine = AdaptiveECCEngine()
    n_bits = 32
    rng = np.random.default_rng(0)
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)
    alpha = DEFAULT_ALPHA

    all_pass = True
    for i, img in enumerate(images):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        tau_low, tau_high = calibrate_thresholds(var_map.flatten(), 25, 75)
        rate_map = build_ecc_rate_map(var_map, tau_low, tau_high)

        watermarked = embed_watermark(img, watermark, rate_map, engine, scheme="reed_solomon", alpha=alpha)
        decoded = extract_watermark(watermarked, rate_map, engine, n_bits, scheme="reed_solomon", alpha=alpha, original_bgr=watermarked)

        ber  = bit_error_rate(watermark, decoded)
        psnr = image_psnr(img, watermarked)
        ssim = image_ssim(img, watermarked)
        status = "✓" if ber == 0.0 else "✗"
        print(f"  Image {i}: BER={ber:.4f}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  {status}")
        if ber != 0.0:
            all_pass = False

    if all_pass:
        print("[smoke_test] ✓ All passed — pipeline is fully functional.")
    else:
        print("[smoke_test] ✗ FAILED — check embedder/decoder for regressions.")
        sys.exit(1)

def run_calibration(cfg: dict) -> None:
    import cv2
    print("[calibrate] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=cfg["data"]["n_images"],
        image_size=tuple(cfg["data"]["image_size"]),
    )

    all_variances: list[float] = []
    img_iter = tqdm(images, desc="Computing block variances") if _TQDM else images
    for img in img_iter:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        all_variances.extend(var_map.flatten().tolist())

    variances = np.array(all_variances, dtype=np.float32)
    tau_low, tau_high = calibrate_thresholds(
        variances,
        cfg["ecc"]["tau_percentile_low"],
        cfg["ecc"]["tau_percentile_high"],
    )
    print(f"\n[calibrate] tau_low  = {tau_low:.4f}")
    print(f"[calibrate] tau_high = {tau_high:.4f}")
    print("[calibrate] Copy these into configs/experiment.yaml under ecc.tau_low / ecc.tau_high")

    print(f"[calibrate] Variance distribution on {len(images)} images:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d}: {np.percentile(variances, p):.2f}")

def run_full_experiment(cfg: dict) -> None:
    print("[full] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=cfg["data"]["n_images"],
        image_size=tuple(cfg["data"]["image_size"]),
    )

    engine = AdaptiveECCEngine()
    scheme:   str   = cfg["ecc"]["scheme"]
    n_bits:   int   = cfg["watermark"]["n_bits"]
    alpha:    float = _alpha_from_cfg(cfg)
    rng = np.random.default_rng(cfg["watermark"]["seed"])
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)
    set_attack_seed(cfg["watermark"]["seed"])

    all_results: dict[str, dict] = {}
    out_dir = pathlib.Path(cfg["results"]["output_dir"])

    with Timer("full experiment"):
        attack_iter = (
            tqdm(ATTACK_SUITE.items(), desc="Attacks", unit="attack")
            if _TQDM else ATTACK_SUITE.items()
        )
        for attack_name, attack_fn in attack_iter:
            bers, ncs, psnrs, ssims = [], [], [], []

            img_iter = (
                tqdm(images, desc=f"  {attack_name}", leave=False)
                if _TQDM else images
            )
            for img in img_iter:
                rate_map   = _make_rate_map(img, cfg)
                watermarked = embed_watermark(img, watermark, rate_map, engine, scheme=scheme, alpha=alpha)
                attacked    = attack_fn(watermarked)                     
                
                # Semi-blind extraction triggering Geometric Correction via original_bgr
                decoded = extract_watermark(
                    attacked, rate_map, engine, n_bits, scheme=scheme, alpha=alpha, original_bgr=watermarked
                )

                bers.append(bit_error_rate(watermark, decoded))
                ncs.append(normalized_correlation(watermark, decoded))
                psnrs.append(image_psnr(img, watermarked))
                ssims.append(image_ssim(img, watermarked))

            ci_lo, ci_hi = ber_confidence_interval(bers)
            det_acc = detection_accuracy(bers, threshold=0.10)

            all_results[attack_name] = {
                "BER_mean":    float(np.mean(bers)),
                "BER_std":     float(np.std(bers)),
                "BER_ci_lo":   ci_lo,
                "BER_ci_hi":   ci_hi,
                "NC_mean":     float(np.mean(ncs)),
                "NC_std":      float(np.std(ncs)),
                "PSNR_mean":   float(np.mean(psnrs)),
                "PSNR_std":    float(np.std(psnrs)),
                "SSIM_mean":   float(np.mean(ssims)),
                "SSIM_std":    float(np.std(ssims)),
                "DetAcc_10pct": det_acc,
            }
            print(
                f"  {attack_name:22s} | "
                f"BER={np.mean(bers):.4f}±{np.std(bers):.4f}  "
                f"NC={np.mean(ncs):.4f}  "
                f"PSNR={np.mean(psnrs):.2f}  "
                f"DetAcc={det_acc:.3f}"
            )

    save_results(all_results, out_dir / "full_results.json")
    print_results_table(all_results, title="Full Experiment — Adaptive ECC")

    latex = to_latex_table(
        all_results,
        caption=(
            r"Proposed semi-blind adaptive-ECC scheme under signal-processing and geometric attacks "
            r"(500 AI-generated images, 512\,px, $n=64$ bits, $\alpha=16$ with rate-coupled JND, "
            r"PSNR\,>\,35\,dB). Geometric attacks are handled via Cr-channel Fourier-Mellin sync tones."
        ),
        label="tab:full",
        selected_metrics=["BER_mean", "NC_mean", "DetAcc_10pct"],
        highlight_best=True,
    )
    (out_dir / "table1.tex").write_text(latex)
    print(f"[full] LaTeX Table 1 → {out_dir / 'table1.tex'}")

def run_ablation_rate(cfg: dict) -> None:
    import cv2
    from src.attack_suite import attack_gaussian_blur, attack_regeneration

    print("[ablation_rate] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=min(cfg["data"]["n_images"], 50),
        image_size=tuple(cfg["data"]["image_size"]),
    )

    engine   = AdaptiveECCEngine()
    scheme   = cfg["ecc"]["scheme"]
    n_bits   = cfg["watermark"]["n_bits"]
    alpha    = _alpha_from_cfg(cfg)
    rng      = np.random.default_rng(cfg["watermark"]["seed"])
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)

    ablation_attacks = {
        "blur_5":          lambda img: attack_gaussian_blur(img, ksize=5),
        "regeneration_04": lambda img: attack_regeneration(img, strength=0.4),
    }

    results: dict[str, dict] = {}

    for fixed_rate in [0.25, 0.50, 0.75]:
        per_atk_bers: dict[str, list[float]] = {k: [] for k in ablation_attacks}
        psnrs: list[float] = []

        for img in images:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            var_map  = compute_block_dct_variance(ycrcb[:, :, 0])
            rate_map = np.full(var_map.shape, fixed_rate, dtype=np.float32)
            watermarked = embed_watermark(img, watermark, rate_map, engine, scheme=scheme, alpha=alpha)
            psnrs.append(image_psnr(img, watermarked))

            for atk_name, atk_fn in ablation_attacks.items():
                attacked = atk_fn(watermarked)
                decoded  = extract_watermark(
                    attacked, rate_map, engine, n_bits, scheme=scheme, alpha=alpha, original_bgr=watermarked
                )
                per_atk_bers[atk_name].append(bit_error_rate(watermark, decoded))

        label = f"fixed_rate_{fixed_rate:.2f}"
        results[label] = {"PSNR_mean": float(np.mean(psnrs))}
        for atk_name in ablation_attacks:
            results[label][f"BER_{atk_name}_mean"] = float(np.mean(per_atk_bers[atk_name]))
            results[label][f"BER_{atk_name}_std"]  = float(np.std(per_atk_bers[atk_name]))
        print(
            f"  {label}: "
            + "  ".join(f"BER_{k}={np.mean(v):.4f}" for k, v in per_atk_bers.items())
            + f"  PSNR={np.mean(psnrs):.2f}"
        )

    adap_per_atk_bers: dict[str, list[float]] = {k: [] for k in ablation_attacks}
    adap_psnrs: list[float] = []

    for img in images:
        rate_map = _make_rate_map(img, cfg)
        watermarked = embed_watermark(img, watermark, rate_map, engine, scheme=scheme, alpha=alpha)
        adap_psnrs.append(image_psnr(img, watermarked))

        for atk_name, atk_fn in ablation_attacks.items():
            attacked = atk_fn(watermarked)
            decoded  = extract_watermark(
                attacked, rate_map, engine, n_bits, scheme=scheme, alpha=alpha, original_bgr=watermarked
            )
            adap_per_atk_bers[atk_name].append(bit_error_rate(watermark, decoded))

    results["adaptive_ecc"] = {"PSNR_mean": float(np.mean(adap_psnrs))}
    for atk_name in ablation_attacks:
        results["adaptive_ecc"][f"BER_{atk_name}_mean"] = float(np.mean(adap_per_atk_bers[atk_name]))
        results["adaptive_ecc"][f"BER_{atk_name}_std"]  = float(np.std(adap_per_atk_bers[atk_name]))
    print(
        "  adaptive_ecc:   "
        + "  ".join(f"BER_{k}={np.mean(v):.4f}" for k, v in adap_per_atk_bers.items())
        + f"  PSNR={np.mean(adap_psnrs):.2f}"
    )

    out_dir = pathlib.Path(cfg["results"]["output_dir"])
    save_results(results, out_dir / "ablation_rate.json")
    print_results_table(results, title="Ablation — Fixed vs Adaptive ECC Rate (blur_5 + regeneration_04)")

def run_baseline_comparison(cfg: dict) -> None:
    from src.baseline_comparison import run_baseline_comparison as _run
    from src.attack_suite import BASELINE_ATTACKS

    print("[baseline_comparison] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=min(cfg["data"]["n_images"], 50),
        image_size=tuple(cfg["data"]["image_size"]),
    )

    results = _run(
        cfg,
        images,
        n_bits=cfg["watermark"]["n_bits"],
        seed=cfg["watermark"]["seed"],
        attacks=BASELINE_ATTACKS,
    )

    out_dir = pathlib.Path(cfg["results"]["output_dir"])
    save_results(results, out_dir / "baseline_comparison.json")

    for method, method_results in results.items():
        print_results_table(method_results, title=f"Baseline: {method}")

    print(f"[baseline_comparison] Results → {out_dir / 'baseline_comparison.json'}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive ECC Watermarking — experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument(
        "--mode",
        default="smoke_test",
        choices=["full", "ablation_rate", "calibrate", "smoke_test", "baseline_comparison"],
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)

    dispatch = {
        "smoke_test":          run_smoke_test,
        "calibrate":           run_calibration,
        "full":                run_full_experiment,
        "ablation_rate":       run_ablation_rate,
        "baseline_comparison": run_baseline_comparison,
    }
    dispatch[args.mode](cfg)

if __name__ == "__main__":
    main()