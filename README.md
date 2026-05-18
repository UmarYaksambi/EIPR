# Adaptive ECC for Watermark Robustness in AI-Generated Images

**Paper target:** IEEE Transactions on Information Forensics and Security (TIFS)

This repository implements the full experimental pipeline for the paper
*"Adaptive ECC for Watermark Robustness in AI-Generated Images"*, including
the proposed adaptive-ECC QIM watermarking scheme, all attack evaluations,
ablation studies, and baseline comparisons.

---

## Table of Contents

1. [Method Overview](#method-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Step-by-Step Experiment Execution](#step-by-step-experiment-execution)
   - [Step 0 — Smoke Test](#step-0--smoke-test-no-data-needed)
   - [Step 1 — Calibrate Thresholds](#step-1--calibrate-thresholds)
   - [Step 2 — Full Robustness Experiment (Table 1)](#step-2--full-robustness-experiment-table-1)
   - [Step 3 — Ablation: ECC Rate (Table 2)](#step-3--ablation-ecc-rate-table-2)
   - [Step 4 — Baseline Comparison (Table 3)](#step-4--baseline-comparison-table-3)
6. [Regeneration Attack with a Real Diffusion Model](#regeneration-attack-with-a-real-diffusion-model)
7. [Reading Results](#reading-results)
8. [Reproducing Paper Tables](#reproducing-paper-tables)
9. [Key Hyperparameters](#key-hyperparameters)
10. [Bug Summary & Design Decisions](#bug-summary--design-decisions)
11. [Citation](#citation)

---

## Method Overview

```
Input Image (BGR)
      │
      ▼
  YCrCb conversion
      │
      ▼
  8×8 block DCT (luminance Y channel)
      │
      ├─── DCT AC variance per block ──► Adaptive ECC rate map
      │         (frequency_analyzer)         (3-tier: smooth→high rate,
      │                                        textured→low rate)
      ▼
  Reed–Solomon encode watermark
  at global mean ECC rate
      │
      ▼
  QIM embed codeword bits into
  low-frequency DCT coefficients
  (zig-zag indices 1, 2, 3; α=36)
      │
      ▼
  IDCT → clip → YCrCb → BGR
      │
      ▼
  Watermarked Image
```

**Key insight:** AI-generated images have characteristically *smooth* frequency
spectra. Smooth blocks are fragile under JPEG/noise and receive more ECC
redundancy (rate 0.75); textured blocks are robust and get less (rate 0.25).
The global codeword is spread across all blocks, keeping encoder/decoder
synchronized using only the stored rate map as side information.

---

## Repository Structure

```
.
├── configs/
│   └── experiment.yaml          # All hyperparameters (edit tau_low/tau_high after calibration)
├── data/
│   ├── ai_generated/            # AI-generated images (populate before running)
│   └── natural/                 # Real photographs (used in baseline_comparison)
├── experiments/
│   └── experiment_runner.py     # Main entry point for all experiment modes
├── notebooks/
│   ├── 01_frequency_analysis.ipynb
│   ├── 02_ecc_rate_sweep.ipynb
│   └── 03_results_tables.ipynb
├── results/                     # Auto-created; all JSON + LaTeX outputs land here
├── src/
│   ├── __init__.py
│   ├── attack_suite.py          # All 11 attacks (JPEG, noise, crop, rotation, regeneration …)
│   ├── baseline_comparison.py   # LSB, Spread-Spectrum, Fixed-rate ECC baselines
│   ├── dataset_generator.py     # load_dataset + generate_synthetic_dataset
│   ├── ecc_engine.py            # AdaptiveECCEngine (Reed–Solomon + repetition)
│   ├── frequency_analyzer.py    # Block-DCT variance, rate map, threshold calibration
│   ├── metrics.py               # BER, NC, PSNR, SSIM
│   ├── utils.py                 # JSON I/O, LaTeX table, visualizations, Timer
│   ├── watermark_decoder.py     # extract_watermark
│   └── watermark_embedder.py    # embed_watermark (QIM in DCT domain)
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU / diffusion model (optional):** The regeneration attack uses a
> surrogate (JPEG + noise + blur) by default.  To use a real Stable Diffusion
> pipeline, also install:
> ```bash
> pip install torch diffusers transformers accelerate
> ```
> See [Regeneration Attack](#regeneration-attack-with-a-real-diffusion-model).

---

## Data Preparation

The pipeline expects two image folders:

| Folder | Contents | Suggested source |
|--------|----------|------------------|
| `data/ai_generated/` | AI-generated images (PNG/JPG) | LAION-Aesthetics, COCO-SD, Stable Diffusion outputs |
| `data/natural/` | Natural photographs | COCO val2017, DIV2K, ImageNet val |

```bash
mkdir -p data/ai_generated data/natural
# Copy your images into these folders.
# The pipeline will resize everything to image_size in experiment.yaml.
```

**Minimum recommended:** 500 AI-generated images for the full experiment
(`data.n_images: 500` in `experiment.yaml`).
The smoke test and ablation runs use 3–50 images and work with any data.

---

## Step-by-Step Experiment Execution

All modes share one entry point:

```bash
python experiments/experiment_runner.py --config configs/experiment.yaml --mode <MODE>
```

### Step 0 — Smoke Test (no data needed)

Verifies the full pipeline (embed → attack-free → decode) on three synthetic
images.  **Run this first** after installation.  Expected output: BER=0.0 on
all three images, PSNR ≈ 46–48 dB.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode smoke_test
```

Expected output:
```
[smoke_test] Generating synthetic images …
  Image 0: BER=0.0000  PSNR=46.50 dB  SSIM=0.9890
  Image 1: BER=0.0000  PSNR=46.80 dB  SSIM=0.9900
  Image 2: BER=0.0000  PSNR=46.05 dB  SSIM=0.9889
[smoke_test] ✓ Passed — pipeline is fully functional.
```

---

### Step 1 — Calibrate Thresholds

Computes dataset-specific `tau_low` and `tau_high` from the AC variance
distribution of your AI-generated image set.  These two values control which
blocks are classified as "smooth" vs "textured".

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode calibrate
```

Expected output (values are dataset-dependent):
```
[calibrate] tau_low  = 47.2831
[calibrate] tau_high = 312.4417
[calibrate] Update ecc.tau_low / ecc.tau_high in experiment.yaml
```

**After calibration:** open `configs/experiment.yaml` and set:
```yaml
ecc:
  tau_low: 47.2831    # ← paste your value
  tau_high: 312.4417  # ← paste your value
```

---

### Step 2 — Full Robustness Experiment (Table 1)

Embeds a 64-bit watermark into all 500 images, applies each of the 11 attacks,
and reports BER, NC, PSNR, and SSIM.  Produces **Table 1** of the paper.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode full
```

Outputs written to `results/`:
- `results/full_results.json` — raw metrics
- `results/table1.tex` — copy-paste LaTeX `booktabs` table

Runtime: ~30 min on CPU for 500 × 512² images; ~3 min on GPU.

**Attacks evaluated:**

| Attack | Parameters |
|--------|-----------|
| JPEG compression | q=50, q=30 |
| Gaussian noise | σ=10, σ=20 |
| Crop + resize | 10% border |
| Rotation | 5° |
| Median filter | 3×3 |
| Gaussian blur | 5×5 |
| Brightness shift | +20 |
| Regeneration (surrogate) | strength=0.4, 0.6 |

---

### Step 3 — Ablation: ECC Rate (Table 2)

Sweeps three fixed ECC rates (0.25, 0.50, 0.75) on a 50-image subset under
JPEG q=50, isolating the contribution of the adaptive rate map.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode ablation_rate
```

Output: `results/ablation_rate.json`

Use this to fill **Table 2**: "Fixed vs Adaptive ECC rate ablation."

---

### Step 4 — Baseline Comparison (Table 3)

Runs the proposed adaptive-ECC scheme against three baselines on a 50-image
subset under 5 representative attacks:

| Baseline | Description |
|----------|-------------|
| `fixed_rate_25/50/75` | Same QIM embedder, constant ECC rate |
| `lsb` | Spatial LSB substitution in Y channel |
| `spread_spectrum` | Additive DCT spread-spectrum (Cox et al., 1997) |

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode baseline_comparison
```

Output: `results/baseline_comparison.json`

---

## Regeneration Attack with a Real Diffusion Model

By default the regeneration attack uses a surrogate (JPEG q=75 + noise +
blur).  For the paper, replace it with Stable Diffusion img2img:

```python
# In a notebook or custom script:
from diffusers import StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

from src.attack_suite import attack_regeneration
attacked = attack_regeneration(image_bgr, strength=0.4, pipe=pipe)
```

Then pass `pipe` into the attack suite in your custom runner.  Report results
separately from the surrogate results in the paper.

---

## Reading Results

All JSON result files share this schema:

```json
{
  "jpeg_q50": {
    "BER_mean": 0.0312,
    "BER_std":  0.0081,
    "NC_mean":  0.9375,
    "PSNR_mean": 46.82,
    "SSIM_mean": 0.9891
  },
  ...
}
```

Load and render programmatically:

```python
from src.utils import load_results, print_results_table, to_latex_table

results = load_results("results/full_results.json")
print_results_table(results, title="Proposed Method — All Attacks")
print(to_latex_table(results, caption="...", label="tab:full"))
```

---

## Reproducing Paper Tables

| Table | Mode | Output file |
|-------|------|-------------|
| Table 1 — Robustness vs all attacks | `full` | `results/table1.tex` |
| Table 2 — ECC rate ablation | `ablation_rate` | `results/ablation_rate.json` |
| Table 3 — Baseline comparison | `baseline_comparison` | `results/baseline_comparison.json` |

For Tables 2 and 3, use `src/utils.to_latex_table()` to convert JSON → LaTeX
(see notebooks/03_results_tables.ipynb for a complete rendering workflow).

---

## Key Hyperparameters

All hyperparameters live in `configs/experiment.yaml`.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `watermark.n_bits` | 64 | Payload length. Reduce for higher robustness at lower capacity. |
| `ecc.scheme` | `reed_solomon` | Switch to `repetition` for ablation §5.3. |
| `ecc.r_high / r_mid / r_low` | 0.75 / 0.50 / 0.25 | ECC rates per texture tier. |
| `ecc.tau_low / tau_high` | `null` → must calibrate | Block classification thresholds. |
| `embedding.alpha` (`ALPHA` in code) | 36.0 | QIM step. Higher → more robust, lower PSNR. |
| `data.n_images` | 500 | Images used. Set to 50 for quick runs. |

---

## Bug Summary & Design Decisions

The following bugs were fixed from the original codebase:

| File | Bug | Fix |
|------|-----|-----|
| `requirements.txt` | `torch6` (invalid package name) | Removed; torch is optional, documented separately |
| `src/__init__.py` | Missing — broke all relative imports (`from .ecc_engine import …`) | Created empty `__init__.py` |
| `src/watermark_embedder.py` | Dead import of `compute_block_dct_variance` (unused in this file) | Removed import |
| `src/watermark_decoder.py` | No guard when `raw_cw_arr` shorter than `codeword_len` (small images) | Added zero-padding fallback |
| `src/ecc_engine.py` | `_rs_decode` could call `np.packbits([])` on zero-length input | Added `trim_len == 0` early return |
| `src/baseline_comparison.py` | File was empty | Fully implemented |
| `src/utils.py` | File was empty | Fully implemented |
| `experiments/experiment_runner.py` | `extract_watermark` import was noted as missing; duplicate `import cv2 as _cv2` in ablation; no `baseline_comparison` mode | Fixed imports, deduplicated, added mode |

**Design decisions preserved from original:**
- Global (not per-block) ECC encoding: one codeword spread across all blocks,
  rate determined by `mean(rate_map)`. Simpler to synchronize; the adaptive
  element is in the block *selection*, not per-block coding.
- Non-blind scheme: `rate_map` is stored as side information alongside the
  watermark key. Justified for AI-image copyright attribution where the
  watermark provider controls the detector.

---

## Citation

```bibtex
@article{yourname2025adaptiveecc,
  title   = {Adaptive {ECC} for Watermark Robustness in {AI}-Generated Images},
  author  = {Your Name},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
}
```