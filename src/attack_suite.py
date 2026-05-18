from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Individual attacks
# ---------------------------------------------------------------------------

def attack_jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """JPEG compression at the given quality (1–100, lower = more loss)."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", image, encode_params)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Additive Gaussian noise with standard deviation sigma."""
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def attack_crop(image: np.ndarray, crop_fraction: float = 0.10) -> np.ndarray:
    """Crop `crop_fraction` of the border on each side, then resize back."""
    h, w = image.shape[:2]
    top = int(h * crop_fraction / 2)
    left = int(w * crop_fraction / 2)
    cropped = image[top : h - top, left : w - left]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_rotation(image: np.ndarray, angle: float = 5.0) -> np.ndarray:
    """Rotate by `angle` degrees (with black fill at borders)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def attack_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filtering with kernel size ksize (must be odd)."""
    return cv2.medianBlur(image, ksize)


def attack_gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur (approximates mild low-pass post-processing)."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def attack_brightness(image: np.ndarray, delta: float = 20.0) -> np.ndarray:
    """Uniform brightness shift by delta intensity units."""
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def attack_regeneration(
    image: np.ndarray,
    strength: float = 0.4,
    pipe=None,          # HuggingFace StableDiffusionImg2ImgPipeline or None
) -> np.ndarray:
    """
    Regeneration attack: the key adversarial threat for AI-generated
    image watermarks.

    A diffusion model img2img pipeline re-renders the image at the given
    `strength`, destroying watermarks that depend on subtle coefficient
    changes without altering perceptual content.

    If `pipe` is None (no GPU / HuggingFace not installed), a cheap
    surrogate (JPEG + Gaussian noise + blur) is used.  The surrogate
    approximates the spectral smoothing that diffusion denoising performs
    but is *not* as destructive; label results accordingly in the paper.

    Args:
        image:    H x W x 3 uint8 BGR image
        strength: img2img strength in [0, 1]; higher = more regeneration
        pipe:     a HuggingFace img2img pipeline (optional)

    Returns:
        attacked H x W x 3 uint8 BGR image
    """
    if pipe is not None:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = pipe(
            prompt="",
            image=pil_img,
            strength=strength,
            guidance_scale=0.0,
            num_inference_steps=20,
        ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # ---- Surrogate (no GPU) ----
    attacked = attack_jpeg(image, quality=75)
    attacked = attack_gaussian_noise(attacked, sigma=5.0 * strength)
    attacked = cv2.GaussianBlur(attacked, (3, 3), 0)
    return attacked


# ---------------------------------------------------------------------------
# Attack suite dictionary — used by experiment_runner
# ---------------------------------------------------------------------------

ATTACK_SUITE: dict[str, object] = {
    "jpeg_q50":         lambda img: attack_jpeg(img, 50),
    "jpeg_q30":         lambda img: attack_jpeg(img, 30),
    "gaussian_10":      lambda img: attack_gaussian_noise(img, 10.0),
    "gaussian_20":      lambda img: attack_gaussian_noise(img, 20.0),
    "crop_10pct":       lambda img: attack_crop(img, 0.10),
    "rotation_5":       lambda img: attack_rotation(img, 5.0),
    "median_3":         lambda img: attack_median_filter(img, 3),
    "blur_5":           lambda img: attack_gaussian_blur(img, 5),
    "brightness_20":    lambda img: attack_brightness(img, 20.0),
    "regeneration_04":  lambda img: attack_regeneration(img, 0.4),
    "regeneration_06":  lambda img: attack_regeneration(img, 0.6),
}