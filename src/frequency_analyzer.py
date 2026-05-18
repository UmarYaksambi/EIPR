import numpy as np
from scipy.fft import dctn
from skimage.util import view_as_blocks

BLOCK_SIZE = 8


def compute_block_dct_variance(image_gray: np.ndarray) -> np.ndarray:
    """
    Compute AC coefficient variance for each 8x8 DCT block.
    This is the texture score T that drives ECC rate assignment.

    AI-generated images produce characteristically lower variance in smooth
    regions (fewer high-frequency surprises) compared to natural photographs,
    making adaptive rate assignment especially impactful.

    Returns:
        variance_map: (n_rows, n_cols) float32 array of per-block AC variances
    """
    h, w = image_gray.shape
    h_c = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_c = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = image_gray[:h_c, :w_c].astype(np.float32)

    blocks = view_as_blocks(img, (BLOCK_SIZE, BLOCK_SIZE))
    n_rows, n_cols = blocks.shape[:2]
    variance_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            block = blocks[i, j]
            # np.asarray() makes the ndarray type explicit to static analysers
            dct_block: np.ndarray = np.asarray(dctn(block, norm="ortho"))
            # AC coefficients only — skip DC at index [0, 0]
            ac = dct_block.flatten()[1:]
            variance_map[i, j] = float(np.var(ac))

    return variance_map


def build_ecc_rate_map(
    variance_map: np.ndarray,
    tau_low: float,
    tau_high: float,
    r_high: float = 0.75,
    r_mid: float = 0.50,
    r_low: float = 0.25,
) -> np.ndarray:
    """
    Three-tier adaptive ECC rate map.

    Smooth blocks (variance < tau_low)   -> r_high: more redundancy,
        embedding in these blocks is fragile.
    Textured blocks (variance > tau_high) -> r_low: less redundancy,
        embedding is safer in high-energy regions.
    Intermediate blocks                   -> r_mid.

    Returns:
        rate_map: same shape as variance_map, dtype float32
    """
    rate_map = np.full(variance_map.shape, r_mid, dtype=np.float32)
    rate_map[variance_map < tau_low] = r_high
    rate_map[variance_map > tau_high] = r_low
    return rate_map


def calibrate_thresholds(
    calibration_variances: np.ndarray,
    percentile_low: float = 25.0,
    percentile_high: float = 75.0,
) -> tuple:
    """
    Learn tau_low and tau_high from a held-out calibration set.

    Call once on ~1 000 AI-generated images; persist the returned
    thresholds in experiment.yaml.

    Args:
        calibration_variances: 1-D array of all per-block AC variances
                               collected from the calibration images.
        percentile_low:  lower percentile for tau_low  (default 25th)
        percentile_high: upper percentile for tau_high (default 75th)

    Returns:
        (tau_low, tau_high) as floats
    """
    tau_low = float(np.percentile(calibration_variances, percentile_low))
    tau_high = float(np.percentile(calibration_variances, percentile_high))
    return tau_low, tau_high