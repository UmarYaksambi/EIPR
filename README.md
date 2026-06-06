<div align="center">

# Adaptive ECC Watermarking for AI-Generated Images

**Texture-Aware Reed–Solomon Coding in the DCT Domain with Geometric Synchronisation**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target Journal](https://img.shields.io/badge/Target-MTAP%20%7C%20Springer-orange)](https://link.springer.com/journal/11042)
[![Framework](https://img.shields.io/badge/Framework-OpenCV%20%7C%20SciPy%20%7C%20reedsolo-red)](requirements.txt)

*Accepted / Under Review — Multimedia Tools and Applications (MTAP), Springer*

</div>

---

## Overview

This repository contains the complete, reproducible experimental pipeline for the paper:

> **"Adaptive Error-Correcting Code Allocation for Robust Watermarking of AI-Generated Images"**
>
> *Multimedia Tools and Applications (MTAP), Springer*

AI-generated images (Stable Diffusion, Midjourney, DALL·E) have a characteristically **smooth frequency spectrum** — most of their DCT energy is concentrated in a narrow band of low-frequency coefficients. Standard fixed-rate ECC watermarking schemes waste redundancy in textured regions and under-protect smooth ones, making them fragile to the very attacks AI-generated content is exposed to: JPEG re-saves, social media recompression, and diffusion-model regeneration.

This work introduces a **three-tier adaptive ECC rate map** driven by per-block DCT AC variance. Smooth blocks (fragile, perceptually sensitive) receive a high ECC rate; textured blocks (robust, noise-masking) receive a low rate. A QIM embedder places codeword bits into low-frequency DCT coefficients of the luminance channel, while Fourier sync tones in the Cr chroma channel correct rotation and scaling attacks blindly at decode time.

### Key Results (500 AI-generated images, DiffusionDB, 512×512 px)

| Attack | BER (Proposed) | BER (Fixed-rate 0.50) | BER (LSB) |
|--------|---------------|----------------------|-----------|
| JPEG q=50 | **0.031 ± 0.008** | 0.089 ± 0.021 | 0.481 ± 0.012 |
| JPEG q=30 | **0.067 ± 0.014** | 0.154 ± 0.033 | 0.499 ± 0.008 |
| Gaussian σ=20 | **0.044 ± 0.011** | 0.098 ± 0.024 | 0.492 ± 0.014 |
| Regeneration (SD, s=0.4) | **0.058 ± 0.013** | 0.141 ± 0.029 | 0.500 ± 0.011 |
| Rotation 5° | **0.039 ± 0.009** | 0.112 ± 0.027 | 0.498 ± 0.013 |
| Mean PSNR | **41.3 dB** | 41.1 dB | 51.2 dB |

> Full results, ablation, and baseline tables are in `results/`. LaTeX-ready tables are auto-generated.

---

## Method

```
Input Image (BGR)
      │
      ▼
  YCrCb conversion
      │
      ├──► Cr channel: embed Fourier sync tones (geometric correction at decode)
      │
      ▼
  8×8 block DCT  (luminance Y channel)
      │
      ├─── QIM-invariant AC variance per block ──► 3-tier ECC rate map
      │                                              • var < τ_low  →  r = 0.75  (smooth, fragile)
      │                                              • τ_low ≤ var ≤ τ_high → r = 0.50  (mid)
      │                                              • var > τ_high →  r = 0.25  (textured, robust)
      ▼
  Reed–Solomon encode watermark at per-tier rate
  + rate-coupled JND alpha scaling
      │
      ▼
  QIM embed codeword bits into DCT coefficients
  (zig-zag indices 1, 2, 3; base α = 8.0)
      │
      ▼
  IDCT → clip → YCrCb → BGR
      │
      ▼
  Watermarked Image
```

**Decoder** reads sync tones from the Cr channel → estimates and corrects geometric distortion → extracts votes from each tier independently → weighted ECC decode → majority-vote fusion.

---

## Repository Structure

```
adaptive-ecc-watermark-ai-images/
├── configs/
│   └── experiment.yaml          # All hyperparameters — edit tau_low/tau_high after calibration
├── data/
│   ├── ai_generated/            # AI-generated images (populate via scripts/download_datasets.py)
│   └── natural/                 # Natural photographs (for baseline_comparison)
├── experiments/
│   └── experiment_runner.py     # Single entry point for all experiment modes
├── notebooks/
│   ├── 01_frequency_analysis.ipynb
│   ├── 02_ecc_rate_sweep.ipynb
│   └── 03_results_tables.ipynb
├── results/
│   ├── full_results.json        # Table 1: full robustness evaluation
│   ├── ablation_rate.json       # Table 2: fixed vs adaptive ECC
│   ├── baseline_comparison.json # Table 3: proposed vs baselines
│   ├── table1.tex               # Auto-generated LaTeX (booktabs)
│   ├── table02_ablation.tex
│   └── table03_full_comparison.tex
├── scripts/
│   ├── download_datasets.py     # Downloads DiffusionDB AI images + Picsum natural images
│   └── generate_paper_visuals.py # Generates PDF figures for the paper
└── src/
    ├── __init__.py
    ├── attack_suite.py          # 22 attacks (JPEG, noise, crop, rotation, regeneration, …)
    ├── baseline_comparison.py   # LSB, SS, fixed-rate, variance-adaptive baselines
    ├── dataset_generator.py     # load_dataset + generate_synthetic_dataset
    ├── ecc_engine.py            # AdaptiveECCEngine (Reed–Solomon + repetition)
    ├── frequency_analyzer.py    # Block-DCT variance, rate map, threshold calibration
    ├── geometric_sync.py        # Cr-channel Fourier sync tone embed/detect/correct
    ├── metrics.py               # BER, NC, PSNR, SSIM, detection accuracy, CI
    ├── utils.py                 # JSON I/O, LaTeX table, visualizations, Timer
    ├── watermark_decoder.py     # extract_watermark (per-tier weighted fusion)
    └── watermark_embedder.py    # embed_watermark (QIM + JND scaling)
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-username/adaptive-ecc-watermark-ai-images.git
cd adaptive-ecc-watermark-ai-images

# 2. Create virtual environment (Python 3.10+ required)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt
```

**For the real Stable Diffusion regeneration attack** (optional, requires CUDA):

```bash
pip install torch diffusers transformers accelerate
```

Without a CUDA machine, set `require_real_regeneration: false` in `configs/experiment.yaml` — the pipeline will use a JPEG+Gaussian surrogate for development runs and clearly warns you not to report those numbers as SD results.

---

## Data Preparation

| Folder | Contents | Default source |
|--------|----------|----------------|
| `data/ai_generated/` | AI-generated images (PNG/JPG) | DiffusionDB via HuggingFace |
| `data/natural/` | Natural photographs | Picsum Photos API |

```bash
# Automated download (handles deduplication and retries)
python scripts/download_datasets.py
```

Or populate the folders manually. The pipeline resizes everything to `image_size` in `experiment.yaml`. Minimum recommended: **500 AI-generated images** for the full experiment; **3 synthetic images** suffice for the smoke test.

---

## Running Experiments

All modes share one entry point:

```bash
python experiments/experiment_runner.py --config configs/experiment.yaml --mode <MODE>
```

### Step 0 — Smoke Test *(no data needed)*

Verifies the full embed → decode round-trip on 3 synthetic images. **Run this first.** Expected: BER = 0.000, PSNR ≥ 38 dB.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode smoke_test
```

Expected output:

```
[smoke_test] Generating synthetic images …
  Image 0: BER=0.0000  PSNR=46.50 dB  SSIM=0.9890  ✓
  Image 1: BER=0.0000  PSNR=46.80 dB  SSIM=0.9900  ✓
  Image 2: BER=0.0000  PSNR=46.05 dB  SSIM=0.9889  ✓
[smoke_test] ✓ All passed — pipeline is fully functional.
```

---

### Step 1 — Calibrate Thresholds

Computes dataset-specific `τ_low` and `τ_high` from the AC variance distribution of your images. **Must be run once on your real dataset** — the placeholder values in `experiment.yaml` were computed on synthetic images and are wrong for real SD output.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode calibrate
```

Then paste the printed `tau_low` / `tau_high` values into `configs/experiment.yaml`:

```yaml
ecc:
  tau_low: 47.2831    # ← your value
  tau_high: 312.4417  # ← your value
```

---

### Step 2 — Full Robustness Evaluation (Table 1)

Embeds a 64-bit watermark into 500 images and evaluates all 22 attacks. Generates `results/full_results.json` and `results/table1.tex`.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode full
```

Runtime: ~30 min on CPU for 500 × 512² images; ~3 min on GPU.

**Attacks evaluated:**

| Category | Attacks |
|----------|---------|
| Lossy compression | JPEG q=70, 50, 30 |
| Additive noise | Gaussian σ=5, 10, 20 |
| Geometric | Crop 5%/10%, Rotation 2°/5°, Scale 50% |
| Filtering | Median 3×3/5×5, Gaussian blur 3×3/5×5 |
| Colour | Brightness ±10/20, Sharpening, Colour jitter |
| AI regeneration | SD img2img strength=0.3, 0.4, 0.6 |

---

### Step 3 — Ablation: ECC Rate (Table 2)

Sweeps fixed ECC rates (0.25, 0.50, 0.75) on a 50-image subset, isolating the contribution of the adaptive rate map.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode ablation_rate
```

---

### Step 4 — Baseline Comparison (Table 3)

Runs the proposed method against five baselines (LSB, spread-spectrum, fixed-rate 0.25/0.50/0.75, variance-adaptive DCT) over 5 representative attacks.

```bash
python experiments/experiment_runner.py \
    --config configs/experiment.yaml \
    --mode baseline_comparison
```

---

## Regeneration Attack (Real Stable Diffusion)

The default configuration uses a JPEG+Gaussian surrogate when SD is unavailable. For paper results, use the real SD img2img pipeline:

```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# Protocol: empty prompt, guidance_scale=1.0 (Zhao et al. 2023; Wen et al. 2023)
from src.attack_suite import attack_regeneration
attacked = attack_regeneration(image_bgr, strength=0.4, pipe=pipe)
```

Set `require_real_regeneration: true` in `experiment.yaml` to hard-crash instead of silently using the surrogate — use this for all final paper runs.

---

## Paper Figures

```bash
# Generates three PDF figures ready for LaTeX inclusion
python scripts/generate_paper_visuals.py
```

| Figure | File | Content |
|--------|------|---------|
| Fig. 1 | `results/Fig_Adaptive_RateMap.pdf` | Original → log-variance map → colour-coded ECC tier map |
| Fig. 2 | `results/Fig_Imperceptibility_Residual.pdf` | Original → watermarked → ×30 amplified residual |
| Fig. 3 | `results/Fig_Geometric_Sync_Peaks.pdf` | Cr-channel FFT spectrum with annotated sync peaks |

Requires one AI-generated image at `data/ai_generated/ai_gen_101.jpg` (or edit `img_path` in the script).

---

## Reading Results

All JSON files share this schema:

```json
{
  "jpeg_q50": {
    "BER_mean": 0.0312,
    "BER_std":  0.0081,
    "BER_ci_lo": 0.0274,
    "BER_ci_hi": 0.0351,
    "NC_mean":  0.9375,
    "PSNR_mean": 46.82,
    "SSIM_mean": 0.9891,
    "DetAcc_10pct": 0.974
  }
}
```

Load and render programmatically:

```python
from src.utils import load_results, print_results_table, to_latex_table

results = load_results("results/full_results.json")
print_results_table(results, title="Proposed Method — All Attacks")
latex = to_latex_table(results, caption="...", label="tab:full")
print(latex)
```

---

## Reproducing Paper Tables

| Paper table | Mode | Output |
|-------------|------|--------|
| Table 1 — Full robustness evaluation | `full` | `results/full_results.json`, `results/table1.tex` |
| Table 2 — ECC rate ablation | `ablation_rate` | `results/ablation_rate.json`, `results/table02_ablation.tex` |
| Table 3 — Baseline comparison | `baseline_comparison` | `results/baseline_comparison.json`, `results/table03_full_comparison.tex` |

For Tables 2 and 3, call `src/utils.to_latex_table()` on the JSON output (see `notebooks/03_results_tables.ipynb` for the full rendering workflow).

---

## Key Hyperparameters

All hyperparameters are in `configs/experiment.yaml`. The most consequential ones:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `watermark.n_bits` | 64 | Payload length. Reduce for higher robustness at lower capacity. |
| `ecc.scheme` | `reed_solomon` | Switch to `repetition` for ablation §5.3. |
| `ecc.r_high / r_mid / r_low` | 0.75 / 0.50 / 0.25 | ECC rates per texture tier. |
| `ecc.tau_low / tau_high` | *calibrate first* | Block classification thresholds. Run `--mode calibrate`. |
| `embedding.alpha` | 8.0 | QIM base step. Higher → more robust, lower PSNR. Rate-coupled JND multipliers: ×0.5 (smooth), ×1.5 (mid), ×3.5 (textured). |
| `data.n_images` | 500 | Images used in full/ablation/baseline runs. Set to 50 for quick iteration. |
| `attacks.require_real_regeneration` | `true` | When `true`, hard-crashes if SD unavailable. Set `false` for CPU dev runs. |

---

## Design Decisions

**Global ECC with adaptive allocation.** One codeword per tier is spread across all same-tier blocks. This is simpler to synchronise than per-block coding — the rate map itself (stored as side information) is the only synchronisation signal needed.

**Non-blind scheme.** The `rate_map` is stored alongside the watermark key. This is appropriate for AI-image copyright attribution, where the watermark provider controls the detector. The spread-spectrum baseline is also evaluated in both informed and blind modes for completeness.

**QIM-invariant variance.** The block variance computation deliberately excludes DCT coefficients that are modified by QIM embedding (indices 0, 1, 2) to prevent the embedding process from shifting a block's tier classification after the rate map is built.

**Rate-coupled JND scaling.** The QIM step `α` is scaled by tier: smooth blocks get a smaller step (less visible distortion in perceptually sensitive regions); textured blocks get a larger step (stronger embedding where noise is masked by texture). Ratio is fixed at 0.5 / 1.5 / 3.5 × base `α`.

---

## Requirements

```
numpy>=1.24
scipy>=1.10
scikit-image>=0.21
opencv-python>=4.8
PyWavelets>=1.4
reedsolo>=1.7
Pillow>=10.0
pyyaml>=6.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
tqdm>=4.65

# Optional — real SD regeneration attack (CUDA required)
torch>=2.0
diffusers>=0.27
transformers>=4.40
accelerate>=0.27
```

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{yaksambi2025adaptiveecc,
  title   = {Adaptive {ECC} Allocation for Robust Watermarking of {AI}-Generated Images},
  author  = {Yaksambi, Umar},
  journal = {Multimedia Tools and Applications},
  year    = {2025},
  publisher = {Springer},
  doi     = {}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built as part of undergraduate research at RV College of Engineering, Bangalore.</sub>
</div>
