# src/attack_suite.py
"""
attack_suite.py — Complete attack suite for watermark robustness evaluation.

Regeneration attack notes
--------------------------
The regeneration attack uses Stable Diffusion img2img with an empty prompt and
guidance_scale=1.0, following the protocol of Zhao et al. (2023) and Wen et al.
(2023, Tree-Ring).  guidance_scale=0.0 is non-standard and must not be used for
published results.

When SD is unavailable (no CUDA / diffusers not installed), a JPEG+Gaussian
surrogate is used instead.  The surrogate path prints a visible warning on every
call and is gated by the `require_real_regeneration` flag in the experiment
config.  When that flag is True, the surrogate raises RuntimeError instead of
running silently — use this for all final paper experiments.

Runtime detection
-----------------
_SD_AVAILABLE : bool — True only when the real SD pipeline loaded successfully.
                Check this before deciding whether to set require_real=True.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Global RNG
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=1234)


def set_attack_seed(seed: int = 1234) -> None:
    global _RNG
    _RNG = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# SD pipeline — loaded once at import time
# ---------------------------------------------------------------------------

_PIPE = None
_SD_AVAILABLE: bool = False

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    _PIPE = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    _PIPE.enable_attention_slicing()
    _SD_AVAILABLE = True
    print("[attack_suite] Stable Diffusion loaded — real regeneration attack active.")
except ImportError:
    print(
        "[attack_suite] WARNING: 'diffusers' not installed. "
        "Regeneration attacks will use a JPEG+noise surrogate."
    )
except Exception as _exc:
    print(
        f"[attack_suite] WARNING: SD load failed ({_exc}). "
        "Regeneration attacks will use a JPEG+noise surrogate."
    )


# ---------------------------------------------------------------------------
# Standard signal-processing attacks
# ---------------------------------------------------------------------------

def attack_jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buf = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian_noise(
    image: np.ndarray, sigma: float = 10.0, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def attack_crop(image: np.ndarray, crop_fraction: float = 0.10) -> np.ndarray:
    h, w = image.shape[:2]
    dy = max(1, int(h * crop_fraction / 2))
    dx = max(1, int(w * crop_fraction / 2))
    cropped = image[dy:h - dy, dx:w - dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_rotation(image: np.ndarray, angle: float = 5.0) -> np.ndarray:
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def attack_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def attack_gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)


def attack_brightness(image: np.ndarray, delta: float = 20.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def attack_scale(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    small_h = max(1, int(h * scale_factor))
    small_w = max(1, int(w * scale_factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def attack_color_jitter(
    image: np.ndarray, hue_shift: int = 10, sat_scale: float = 1.2
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Regeneration attack (Stable Diffusion img2img)
# ---------------------------------------------------------------------------

# One-time warning flag so the surrogate doesn't spam the terminal.
_SURROGATE_WARNED: bool = False


def attack_regeneration(
    image: np.ndarray,
    strength: float = 0.4,
    pipe=None,
    require_real: bool = False,
) -> np.ndarray:
    """
    Regeneration attack via Stable Diffusion img2img.

    Protocol (Zhao et al. 2023; Wen et al. 2023):
      - prompt = "" (empty — no semantic steering)
      - guidance_scale = 1.0 (standard for pure reconstruction)
      - num_inference_steps = 20
      - strength controls how many denoising steps run (0.3–0.6 typical)

    Args:
        image:        Input BGR uint8 image.
        strength:     Noise strength passed to SD img2img (0.0–1.0).
        pipe:         Optional externally-supplied pipeline (overrides global).
        require_real: If True, raise RuntimeError when SD is unavailable
                      instead of falling back to the surrogate.  Set this
                      to True for all final paper experiments.

    Returns:
        BGR uint8 attacked image, same spatial dimensions as input.
    """
    global _SURROGATE_WARNED

    active_pipe = pipe if pipe is not None else _PIPE

    if active_pipe is not None:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = active_pipe(
            prompt="",               # empty — no semantic steering
            image=pil_img,
            strength=float(strength),
            guidance_scale=1.0,      # standard reconstruction protocol
            num_inference_steps=20,
        ).images[0]
        out = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    # --- Surrogate path ---
    if require_real:
        raise RuntimeError(
            "attack_regeneration: require_real=True but Stable Diffusion is not "
            "available (no CUDA or diffusers not installed). "
            "Either install diffusers on a CUDA machine or set "
            "require_real_regeneration: false in experiment.yaml for dev runs."
        )

    if not _SURROGATE_WARNED:
        print(
            "[attack_suite] SURROGATE regeneration active (JPEG+noise). "
            "Do NOT report these results as 'Stable Diffusion regeneration' "
            "in any publication. Set require_real_regeneration: true in "
            "experiment.yaml and run on a CUDA machine for paper results."
        )
        _SURROGATE_WARNED = True

    attacked = attack_jpeg(image, quality=75)
    attacked = attack_gaussian_noise(attacked, sigma=5.0 * float(strength))
    attacked = cv2.GaussianBlur(attacked, (3, 3), sigmaX=0)
    return attacked


# ---------------------------------------------------------------------------
# Attack factory — builds callables with require_real wired in
# ---------------------------------------------------------------------------

def _make_regeneration_fn(strength: float, require_real: bool = False):
    """
    Return a zero-argument-image callable for use in ATTACK_SUITE dicts.
    require_real is captured at construction time from the experiment config.
    """
    def _fn(img: np.ndarray) -> np.ndarray:
        return attack_regeneration(img, strength=strength, require_real=require_real)
    return _fn


def build_attack_suite(require_real_regeneration: bool = False) -> dict[str, object]:
    """
    Build the full attack suite dict.

    Call this from experiment_runner with the value from experiment.yaml rather
    than importing ATTACK_SUITE directly, so the require_real flag is respected.

    Args:
        require_real_regeneration: mirrors experiment.yaml attacks.require_real_regeneration
    """
    return {
        "jpeg_q70":        lambda img: attack_jpeg(img, 70),
        "jpeg_q50":        lambda img: attack_jpeg(img, 50),
        "jpeg_q30":        lambda img: attack_jpeg(img, 30),
        "gaussian_05":     lambda img: attack_gaussian_noise(img, 5.0),
        "gaussian_10":     lambda img: attack_gaussian_noise(img, 10.0),
        "gaussian_20":     lambda img: attack_gaussian_noise(img, 20.0),
        "crop_05pct":      lambda img: attack_crop(img, 0.05),
        "crop_10pct":      lambda img: attack_crop(img, 0.10),
        "rotation_2":      lambda img: attack_rotation(img, 2.0),
        "rotation_5":      lambda img: attack_rotation(img, 5.0),
        "scale_50pct":     lambda img: attack_scale(img, 0.5),
        "median_3":        lambda img: attack_median_filter(img, 3),
        "median_5":        lambda img: attack_median_filter(img, 5),
        "blur_3":          lambda img: attack_gaussian_blur(img, 3),
        "blur_5":          lambda img: attack_gaussian_blur(img, 5),
        "brightness_10":   lambda img: attack_brightness(img, 10.0),
        "brightness_20":   lambda img: attack_brightness(img, 20.0),
        "sharpening":      lambda img: attack_sharpening(img, 1.0),
        "color_jitter":    lambda img: attack_color_jitter(img),
        "regeneration_03": _make_regeneration_fn(0.3, require_real_regeneration),
        "regeneration_04": _make_regeneration_fn(0.4, require_real_regeneration),
        "regeneration_06": _make_regeneration_fn(0.6, require_real_regeneration),
    }


def build_baseline_attack_suite(require_real_regeneration: bool = False) -> dict[str, object]:
    suite = build_attack_suite(require_real_regeneration)
    return {k: suite[k] for k in ("jpeg_q50", "jpeg_q30", "gaussian_10", "crop_10pct", "regeneration_04")}


# ---------------------------------------------------------------------------
# Legacy module-level dicts — kept for import compatibility.
# These use require_real=False (surrogate allowed).
# Use build_attack_suite() in experiment code instead.
# ---------------------------------------------------------------------------

ATTACK_SUITE: dict[str, object] = build_attack_suite(require_real_regeneration=False)
BASELINE_ATTACKS: dict[str, object] = build_baseline_attack_suite(require_real_regeneration=False)