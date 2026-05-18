from __future__ import annotations

import os
import glob
import numpy as np
import cv2


def load_dataset(
    directory: str,
    limit: int = 500,
    image_size: tuple[int, int] = (512, 512),
    extensions: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.webp"),
) -> list[np.ndarray]:
    """
    Load up to `limit` images from `directory` into a list of BGR uint8 arrays.

    Images are resized to `image_size` (width, height).  Unreadable files are
    skipped with a warning rather than raising an exception.

    Args:
        directory:  path to folder containing images
        limit:      maximum number of images to load
        image_size: (width, height) to resize each image to
        extensions: glob patterns to match

    Returns:
        list of H x W x 3 uint8 BGR numpy arrays
    """
    paths: list[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(paths)[:limit]

    if not paths:
        raise FileNotFoundError(
            f"No images found in {directory!r} matching {extensions}"
        )

    images: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[dataset_generator] Warning: could not read {p!r}, skipping.")
            continue
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
        images.append(img)

    print(f"[dataset_generator] Loaded {len(images)} images from {directory!r}")
    return images


def generate_synthetic_dataset(
    n_images: int = 10,
    image_size: tuple[int, int] = (512, 512),
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Generate random smooth images for unit testing without real data.

    Images are created as Gaussian-blurred random noise to crudely mimic
    the smooth frequency characteristics of AI-generated images.

    Args:
        n_images:   number of images to generate
        image_size: (width, height)
        seed:       random seed for reproducibility

    Returns:
        list of H x W x 3 uint8 BGR numpy arrays
    """
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    h, w = image_size[1], image_size[0]
    for _ in range(n_images):
        noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        smooth = cv2.GaussianBlur(noise, (31, 31), 8)
        images.append(smooth)
    return images