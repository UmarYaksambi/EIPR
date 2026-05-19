"""
attack_suite.py — Complete attack suite for watermark robustness evaluation.

Each attack function:
  • Accepts an H × W × 3 uint8 BGR image.
  • Returns an H × W × 3 uint8 BGR image of the same spatial size.
  • Is deterministic given a fixed random seed (set via ``set_attack_seed``).

Attack taxonomy (cf. Stirmark benchmark):
  Valumetric  — JPEG, noise, brightness, blur, median
  Geometric   — crop+resize, rotation, scaling
  Generative  — regeneration (diffusion img2img surrogate)

ATTACK_SUITE dict maps string keys to zero-argument lambdas over a single
image argument.  Keys must remain stable — they appear verbatim in the
paper's Table 1 and in result JSON files.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level RNG for reproducible noise attacks
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(seed=1234)


def set_attack_seed(seed: int = 1234) -> None:
    """
    Re-seed the module-level RNG.  Call this at the start of each experiment
    run to guarantee reproducible Gaussian noise across images.
    """
    global _RNG
    _RNG = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Valumetric attacks
# ---------------------------------------------------------------------------

def attack_jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    JPEG compression at the given quality level (1–100; lower = more loss).

    This is the primary attack for DCT-domain watermarks.  Quantisation
    steps at q=50 are ~10–12 for low-frequency coefficients; ALPHA=36
    provides a 3× safety margin.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buf = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian_noise(
    image: np.ndarray, sigma: float = 10.0, seed: int | None = None
) -> np.ndarray:
    """
    Additive i.i.d. Gaussian noise N(0, σ²).

    Args:
        sigma: standard deviation in [0, 255] pixel units.
        seed:  if given, overrides module RNG for this call (for unit tests).
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def attack_crop(image: np.ndarray, crop_fraction: float = 0.10) -> np.ndarray:
    """
    Crop ``crop_fraction`` of the border on each side, then up-scale back.

    Simulates unintentional cropping or thumbnail generation.  Geometric
    misalignment is the primary challenge for block-based schemes.
    """
    h, w = image.shape[:2]
    dy = max(1, int(h * crop_fraction / 2))
    dx = max(1, int(w * crop_fraction / 2))
    cropped = image[dy:h - dy, dx:w - dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_rotation(image: np.ndarray, angle: float = 5.0) -> np.ndarray:
    """
    Rotate the image by ``angle`` degrees (counter-clockwise), filling
    borders with reflected content to avoid black-corner artefacts.
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def attack_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median filter of kernel size ``ksize`` (must be positive and odd).

    Effective against high-frequency watermarks; DCT-domain embedding at
    low frequencies is more resistant.
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def attack_gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Gaussian blur with kernel size ``ksize``.

    Approximates mild low-pass post-processing applied by social media
    platforms to reduce file size before hosting.
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)


def attack_brightness(image: np.ndarray, delta: float = 20.0) -> np.ndarray:
    """
    Uniform additive brightness shift by ``delta`` intensity units.

    Simulates gamma correction or monitor calibration changes.
    """
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def attack_scale(
    image: np.ndarray,
    scale_factor: float = 0.5,
) -> np.ndarray:
    """
    Down-scale by ``scale_factor`` then up-scale back to original size.

    Simulates image resizing performed by image-hosting platforms.
    Bilinear interpolation introduces half-pixel misalignment stress.
    """
    h, w = image.shape[:2]
    small_h = max(1, int(h * scale_factor))
    small_w = max(1, int(w * scale_factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Unsharp masking sharpening (high-frequency boost).

    Kernel: identity - strength * Laplacian.  Common photo post-processing step.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        image, 1.0 + strength, blurred, -strength, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def attack_color_jitter(
    image: np.ndarray,
    hue_shift: int = 10,
    sat_scale: float = 1.2,
) -> np.ndarray:
    """
    Random hue shift + saturation scaling in HSV space.

    Simulates social-media colour filters.  Hue shift is in [0, 180] OpenCV units.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Generative / regeneration attack
# ---------------------------------------------------------------------------

def attack_regeneration(
    image: np.ndarray,
    strength: float = 0.4,
    pipe=None,
) -> np.ndarray:
    """
    Regeneration attack — the primary adversarial threat for AI-image watermarks.

    A diffusion model img2img pipeline re-renders the image at ``strength``,
    destroying watermarks that rely on subtle coefficient perturbations.

    If ``pipe`` is None (no GPU / HuggingFace not installed), a cheap
    surrogate is used:
        JPEG q=75 → Gaussian noise (σ = 5 × strength) → Gaussian blur (3×3)
    The surrogate approximates the spectral smoothing that diffusion denoising
    performs but is *less* destructive.  Label results accordingly in the paper.

    Args:
        image:    H × W × 3 uint8 BGR image.
        strength: img2img strength ∈ [0, 1]; higher = more regeneration.
        pipe:     HuggingFace ``StableDiffusionImg2ImgPipeline`` or None.

    Returns:
        Attacked H × W × 3 uint8 BGR image (same spatial size).
    """
    if pipe is not None:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = pipe(
            prompt="",
            image=pil_img,
            strength=float(strength),
            guidance_scale=0.0,
            num_inference_steps=20,
        ).images[0]
        out = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        # Resize back to original in case the pipeline changed resolution
        h, w = image.shape[:2]
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    # --- Deterministic surrogate (no GPU) ---------------------------------
    attacked = attack_jpeg(image, quality=75)
    attacked = attack_gaussian_noise(attacked, sigma=5.0 * float(strength))
    attacked = cv2.GaussianBlur(attacked, (3, 3), sigmaX=0)
    return attacked


# ---------------------------------------------------------------------------
# ATTACK_SUITE — canonical attack registry used by experiment_runner
# ---------------------------------------------------------------------------
# Keys appear verbatim in JSON result files and paper tables — do not rename.

ATTACK_SUITE: dict[str, object] = {
    # --- JPEG ---
    "jpeg_q70":         lambda img: attack_jpeg(img, 70),
    "jpeg_q50":         lambda img: attack_jpeg(img, 50),
    "jpeg_q30":         lambda img: attack_jpeg(img, 30),
    # --- Gaussian noise ---
    "gaussian_05":      lambda img: attack_gaussian_noise(img, 5.0),
    "gaussian_10":      lambda img: attack_gaussian_noise(img, 10.0),
    "gaussian_20":      lambda img: attack_gaussian_noise(img, 20.0),
    # --- Geometric ---
    "crop_05pct":       lambda img: attack_crop(img, 0.05),
    "crop_10pct":       lambda img: attack_crop(img, 0.10),
    "rotation_2":       lambda img: attack_rotation(img, 2.0),
    "rotation_5":       lambda img: attack_rotation(img, 5.0),
    "scale_50pct":      lambda img: attack_scale(img, 0.5),
    # --- Filtering ---
    "median_3":         lambda img: attack_median_filter(img, 3),
    "median_5":         lambda img: attack_median_filter(img, 5),
    "blur_3":           lambda img: attack_gaussian_blur(img, 3),
    "blur_5":           lambda img: attack_gaussian_blur(img, 5),
    # --- Valumetric ---
    "brightness_10":    lambda img: attack_brightness(img, 10.0),
    "brightness_20":    lambda img: attack_brightness(img, 20.0),
    "sharpening":       lambda img: attack_sharpening(img, 1.0),
    "color_jitter":     lambda img: attack_color_jitter(img),
    # --- Regeneration ---
    "regeneration_03":  lambda img: attack_regeneration(img, 0.3),
    "regeneration_04":  lambda img: attack_regeneration(img, 0.4),
    "regeneration_06":  lambda img: attack_regeneration(img, 0.6),
}

# Compact subset used for baseline comparison (to limit runtime)
BASELINE_ATTACKS: dict[str, object] = {
    k: ATTACK_SUITE[k]
    for k in ("jpeg_q50", "jpeg_q30", "gaussian_10", "crop_10pct", "regeneration_04")
}