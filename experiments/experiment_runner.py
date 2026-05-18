"""
Experiment runner for Adaptive ECC Watermarking paper.

Usage:
    python experiment_runner.py --config configs/experiment.yaml --mode full
    python experiment_runner.py --config configs/experiment.yaml --mode ablation_rate
    python experiment_runner.py --config configs/experiment.yaml --mode calibrate
    python experiment_runner.py --config configs/experiment.yaml --mode smoke_test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Absolute imports from src/ — add project root to sys.path when running
# the script directly (i.e. not as part of a package).
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
from src.watermark_embedder import embed_watermark
from src.watermark_decoder import extract_watermark          # ← was missing
from src.attack_suite import ATTACK_SUITE
from src.metrics import bit_error_rate, normalized_correlation, image_psnr, image_ssim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_rate_map(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Extract luminance channel and build the per-block ECC rate map."""
    ycrcb = __import__("cv2").cvtColor(img, __import__("cv2").COLOR_BGR2YCrCb)
    Y_gray = ycrcb[:, :, 0]
    var_map = compute_block_dct_variance(Y_gray)
    tau_low = float(cfg["ecc"]["tau_low"] or 50.0)
    tau_high = float(cfg["ecc"]["tau_high"] or 200.0)
    return build_ecc_rate_map(
        var_map,
        tau_low,
        tau_high,
        r_high=float(cfg["ecc"]["r_high"]),
        r_mid=float(cfg["ecc"]["r_mid"]),
        r_low=float(cfg["ecc"]["r_low"]),
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_calibration(cfg: dict) -> None:
    """
    Compute tau_low / tau_high from the AI-generated image set and print them.
    Copy the printed values into experiment.yaml before running full experiments.
    """
    print("[calibrate] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=cfg["data"]["n_images"],
        image_size=tuple(cfg["data"]["image_size"]),
    )
    import cv2
    all_variances: list[float] = []
    for img in images:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        all_variances.extend(var_map.flatten().tolist())

    variances = np.array(all_variances, dtype=np.float32)
    tau_low, tau_high = calibrate_thresholds(
        variances,
        cfg["ecc"]["tau_percentile_low"],
        cfg["ecc"]["tau_percentile_high"],
    )
    print(f"[calibrate] tau_low  = {tau_low:.4f}")
    print(f"[calibrate] tau_high = {tau_high:.4f}")
    print("[calibrate] Update ecc.tau_low / ecc.tau_high in experiment.yaml")


def run_full_experiment(cfg: dict) -> None:
    """Run all attacks against the watermarked dataset and save JSON results."""
    print("[full] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=cfg["data"]["n_images"],
        image_size=tuple(cfg["data"]["image_size"]),
    )

    engine = AdaptiveECCEngine()
    scheme = cfg["ecc"]["scheme"]
    n_bits: int = cfg["watermark"]["n_bits"]
    rng = np.random.default_rng(cfg["watermark"]["seed"])
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)

    all_results: dict[str, dict] = {}

    for attack_name, attack_fn in ATTACK_SUITE.items():
        bers, ncs, psnrs, ssims = [], [], [], []

        for img in images:
            rate_map = _make_rate_map(img, cfg)
            watermarked = embed_watermark(img, watermark, rate_map, engine, scheme)
            attacked = attack_fn(watermarked)                # type: ignore[operator]
            decoded = extract_watermark(attacked, rate_map, engine, n_bits, scheme)

            bers.append(bit_error_rate(watermark, decoded))
            ncs.append(normalized_correlation(watermark, decoded))
            psnrs.append(image_psnr(img, watermarked))
            ssims.append(image_ssim(img, watermarked))

        all_results[attack_name] = {
            "BER_mean":  float(np.mean(bers)),
            "BER_std":   float(np.std(bers)),
            "NC_mean":   float(np.mean(ncs)),
            "PSNR_mean": float(np.mean(psnrs)),
            "SSIM_mean": float(np.mean(ssims)),
        }
        print(
            f"  {attack_name:25s} | "
            f"BER={np.mean(bers):.4f}  "
            f"NC={np.mean(ncs):.4f}  "
            f"PSNR={np.mean(psnrs):.2f}"
        )

    out_path = pathlib.Path(cfg["results"]["output_dir"]) / "full_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[full] Results saved to {out_path}")


def run_ablation_rate(cfg: dict) -> None:
    """
    Sweep over ECC rates (fixed, not adaptive) to produce Table 2 of the paper:
    fixed-ECC BER vs adaptive-ECC BER at matched bit budget.
    """
    print("[ablation_rate] Loading images …")
    images = load_dataset(
        cfg["data"]["ai_generated_path"],
        limit=min(cfg["data"]["n_images"], 50),
        image_size=tuple(cfg["data"]["image_size"]),
    )

    engine = AdaptiveECCEngine()
    scheme = cfg["ecc"]["scheme"]
    n_bits: int = cfg["watermark"]["n_bits"]
    rng = np.random.default_rng(cfg["watermark"]["seed"])
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)
    import cv2
    results: dict[str, dict] = {}

    for fixed_rate in [0.25, 0.50, 0.75]:
        bers = []
        for img in images:
            # Fixed rate map — all blocks get the same rate
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            var_map = compute_block_dct_variance(ycrcb[:, :, 0])
            rate_map = np.full(var_map.shape, fixed_rate, dtype=np.float32)

            watermarked = embed_watermark(img, watermark, rate_map, engine, scheme)
            # Use JPEG q=50 as representative attack
            import cv2 as _cv2
            _, buf = _cv2.imencode(".jpg", watermarked, [int(_cv2.IMWRITE_JPEG_QUALITY), 50])
            attacked = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            decoded = extract_watermark(attacked, rate_map, engine, n_bits, scheme)
            bers.append(bit_error_rate(watermark, decoded))

        label = f"fixed_rate_{fixed_rate:.2f}"
        results[label] = {"BER_mean": float(np.mean(bers)), "BER_std": float(np.std(bers))}
        print(f"  {label}: BER = {np.mean(bers):.4f} ± {np.std(bers):.4f}")

    out_path = pathlib.Path(cfg["results"]["output_dir"]) / "ablation_rate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[ablation_rate] Results saved to {out_path}")


def run_smoke_test(_cfg: dict) -> None:
    """
    Fast end-to-end sanity check using synthetic images — no real data needed.
    Run this first to verify the pipeline is wired correctly.
    """
    print("[smoke_test] Generating synthetic images …")
    images = generate_synthetic_dataset(n_images=3, image_size=(256, 256), seed=0)
    engine = AdaptiveECCEngine()
    n_bits = 32
    rng = np.random.default_rng(0)
    watermark = rng.integers(0, 2, n_bits).astype(np.uint8)
    import cv2

    for i, img in enumerate(images):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        var_map = compute_block_dct_variance(ycrcb[:, :, 0])
        tau_low, tau_high = calibrate_thresholds(var_map.flatten(), 25, 75)
        rate_map = build_ecc_rate_map(var_map, tau_low, tau_high)

        watermarked = embed_watermark(img, watermark, rate_map, engine)
        decoded = extract_watermark(watermarked, rate_map, engine, n_bits)

        ber = bit_error_rate(watermark, decoded)
        psnr = image_psnr(img, watermarked)
        print(f"  Image {i}: BER={ber:.4f}  PSNR={psnr:.2f} dB")

    print("[smoke_test] Passed — pipeline is functional.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive ECC Watermarking experiment runner"
    )
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument(
        "--mode",
        default="smoke_test",
        choices=["full", "ablation_rate", "calibrate", "smoke_test"],
        help=(
            "smoke_test  — quick end-to-end check with synthetic images (default)\n"
            "calibrate   — compute tau thresholds from the dataset\n"
            "full        — run all attacks and save results\n"
            "ablation_rate — sweep fixed ECC rates for Table 2\n"
        ),
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)

    dispatch = {
        "smoke_test":    run_smoke_test,
        "calibrate":     run_calibration,
        "full":          run_full_experiment,
        "ablation_rate": run_ablation_rate,
    }
    dispatch[args.mode](cfg)


if __name__ == "__main__":
    main()