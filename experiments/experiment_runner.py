"""
experiment_runner.py — Entry point for all Adaptive ECC Watermarking experiments.

Usage
-----
    python experiments/experiment_runner.py --config configs/experiment.yaml --mode smoke_test
    python experiments/experiment_runner.py --config configs/experiment.yaml --mode calibrate
    python experiments/experiment_runner.py --config configs/experiment.yaml --mode full
    python experiments/experiment_runner.py --config configs/experiment.yaml --mode ablation_rate
    python experiments/experiment_runner.py --config configs/experiment.yaml --mode baseline_comparison

Modes
-----
smoke_test          Fast end-to-end check on 3 synthetic 256×256 images (no data needed).
calibrate           Compute dataset-specific tau_low / tau_high and print them.
full                Run all attacks on 500 images → Table 1.
ablation_rate       Sweep fixed ECC rates under JPEG q=50 → Table 2.
baseline_comparison Compare proposed vs LSB / SS / fixed-rate ECC → Table 3.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Add project root to sys.path so `src` is importable when running directly
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _alpha_from_cfg(cfg: dict) -> float:
    """Read QIM step alpha from config; fall back to module default."""
    return float((cfg.get("embedding") or {}).get("alpha") or DEFAULT_ALPHA)


def _make_rate_map(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert image → luminance → block-DCT variance → ECC rate map."""
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


# ---------------------------------------------------------------------------
# Mode: smoke_test
# ---------------------------------------------------------------------------

def run_smoke_test(_cfg: dict) -> None:
    """
    Fast end-to-end sanity check on three synthetic 256×256 images.
    No real data required.  Expected result: BER=0.0 on all images.
    """
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

        watermarked = embed_watermark(img, watermark, rate_map, engine, alpha=alpha)
        decoded = extract_watermark(watermarked, rate_map, engine, n_bits, alpha=alpha)

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


# ---------------------------------------------------------------------------
# Mode: calibrate
# ---------------------------------------------------------------------------

def run_calibration(cfg: dict) -> None:
    """
    Compute dataset-specific tau_low / tau_high from the AC variance
    distribution of the AI-generated image set.

    Copy the printed values into experiment.yaml before running full
    experiments.
    """
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

    # Additional statistics useful for the paper
    print(f"[calibrate] Variance distribution on {len(images)} images:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d}: {np.percentile(variances, p):.2f}")


# ---------------------------------------------------------------------------
# Mode: full
# ---------------------------------------------------------------------------

def run_full_experiment(cfg: dict) -> None:
    """
    Embed watermarks into all images, apply all attacks, record metrics.
    Produces Table 1 of the paper.
    """
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
                watermarked = embed_watermark(img, watermark, rate_map, engine, scheme, alpha=alpha)
                attacked    = attack_fn(watermarked)                     # type: ignore[operator]
                decoded     = extract_watermark(attacked, rate_map, engine, n_bits, scheme, alpha=alpha)

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

    # LaTeX: Table 1 — select key metrics for the paper
    latex = to_latex_table(
        all_results,
        caption=(
            r"Proposed adaptive-ECC scheme under various attacks "
            r"(500 AI-generated images, 512\,px, $n=64$ bits, $\alpha=36$)."
        ),
        label="tab:full",
        selected_metrics=["BER_mean", "NC_mean", "PSNR_mean", "SSIM_mean"],
        highlight_best=True,
    )
    (out_dir / "table1.tex").write_text(latex)
    print(f"[full] LaTeX Table 1 → {out_dir / 'table1.tex'}")


# ---------------------------------------------------------------------------
# Mode: ablation_rate
# ---------------------------------------------------------------------------

def run_ablation_rate(cfg: dict) -> None:
    """
    Sweep fixed ECC rates (0.25, 0.50, 0.75) against the proposed adaptive
    scheme on a 50-image subset under JPEG q=50.  Produces Table 2.
    """
    import cv2
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

    results: dict[str, dict] = {}

    # Fixed-rate sweep
    for fixed_rate in [0.25, 0.50, 0.75]:
        bers, psnrs = [], []
        for img in images:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            var_map  = compute_block_dct_variance(ycrcb[:, :, 0])
            rate_map = np.full(var_map.shape, fixed_rate, dtype=np.float32)

            watermarked = embed_watermark(img, watermark, rate_map, engine, scheme, alpha=alpha)
            _, buf = cv2.imencode(".jpg", watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            attacked = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            decoded = extract_watermark(attacked, rate_map, engine, n_bits, scheme, alpha=alpha)

            bers.append(bit_error_rate(watermark, decoded))
            psnrs.append(image_psnr(img, watermarked))

        label = f"fixed_rate_{fixed_rate:.2f}"
        results[label] = {
            "BER_mean":  float(np.mean(bers)),
            "BER_std":   float(np.std(bers)),
            "PSNR_mean": float(np.mean(psnrs)),
        }
        print(f"  {label}: BER={np.mean(bers):.4f}±{np.std(bers):.4f}  PSNR={np.mean(psnrs):.2f}")

    # Proposed adaptive scheme (for comparison row)
    adap_bers, adap_psnrs = [], []
    tau_low  = float(cfg["ecc"].get("tau_low")  or 50.0)
    tau_high = float(cfg["ecc"].get("tau_high") or 200.0)
    for img in images:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map  = compute_block_dct_variance(ycrcb[:, :, 0])
        rate_map = build_ecc_rate_map(
            var_map, tau_low, tau_high,
            r_high=cfg["ecc"]["r_high"],
            r_mid =cfg["ecc"]["r_mid"],
            r_low =cfg["ecc"]["r_low"],
        )
        watermarked = embed_watermark(img, watermark, rate_map, engine, scheme, alpha=alpha)
        _, buf = cv2.imencode(".jpg", watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        attacked = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        decoded = extract_watermark(attacked, rate_map, engine, n_bits, scheme, alpha=alpha)
        adap_bers.append(bit_error_rate(watermark, decoded))
        adap_psnrs.append(image_psnr(img, watermarked))

    results["adaptive_ecc"] = {
        "BER_mean":  float(np.mean(adap_bers)),
        "BER_std":   float(np.std(adap_bers)),
        "PSNR_mean": float(np.mean(adap_psnrs)),
    }
    print(
        f"  adaptive_ecc:    BER={np.mean(adap_bers):.4f}±{np.std(adap_bers):.4f}  "
        f"PSNR={np.mean(adap_psnrs):.2f}"
    )

    out_dir = pathlib.Path(cfg["results"]["output_dir"])
    save_results(results, out_dir / "ablation_rate.json")
    print_results_table(results, title="Ablation — Fixed vs Adaptive ECC Rate (JPEG q=50)")


# ---------------------------------------------------------------------------
# Mode: baseline_comparison
# ---------------------------------------------------------------------------

def run_baseline_comparison(cfg: dict) -> None:
    """Compare proposed adaptive-ECC against LSB, SS, and fixed-rate ECC baselines."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        help=(
            "smoke_test          — quick end-to-end check with synthetic images (default)\n"
            "calibrate           — compute tau thresholds from the dataset\n"
            "full                — run all attacks and save results (Table 1)\n"
            "ablation_rate       — fixed vs adaptive ECC rate sweep (Table 2)\n"
            "baseline_comparison — compare vs LSB / SS / fixed-rate ECC (Table 3)\n"
        ),
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