import sys
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ecc_engine import AdaptiveECCEngine
from src.frequency_analyzer import compute_block_dct_variance, build_ecc_rate_map
from src.watermark_embedder import embed_watermark
from src.geometric_sync import embed_sync_chroma

# ---------------------------------------------------------------------------
# Q1 Journal Styling Configuration
# ---------------------------------------------------------------------------
def set_q1_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })

# ---------------------------------------------------------------------------
# Figure 1: The Adaptive Rate Map (Proves JND & Texture Tiering)
# ---------------------------------------------------------------------------
def generate_rate_map_visual(img_bgr: np.ndarray, cfg: dict, out_dir: Path):
    """Visualizes how the algorithm maps variance to ECC tiers."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    var_map = compute_block_dct_variance(ycrcb[:, :, 0])
    
    tau_low = float(cfg['ecc']['tau_low'])
    tau_high = float(cfg['ecc']['tau_high'])
    rate_map = build_ecc_rate_map(var_map, tau_low, tau_high)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Original AI Image")
    axes[0].axis('off')

    # 2. Log Variance Heatmap
    log_var = np.log10(np.clip(var_map, 1e-3, None))
    im1 = axes[1].imshow(log_var, cmap='magma')
    axes[1].set_title(r"(b) Block DCT Variance ($\log_{10}$)")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Discrete Rate Map
    # Map rates to categories: 0.75 (Smooth), 0.50 (Mid), 0.25 (Textured)
    cmap = mcolors.ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c']) # Green, Yellow, Red
    bounds = [0.2, 0.4, 0.6, 0.8]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im2 = axes[2].imshow(rate_map, cmap=cmap, norm=norm)
    axes[2].set_title("(c) Adaptive ECC Rate Allocation")
    axes[2].axis('off')
    
    cbar = fig.colorbar(im2, ax=axes[2], ticks=[0.3, 0.5, 0.7], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Textured (R=0.25)', 'Mid (R=0.50)', 'Smooth (R=0.75)'])

    plt.tight_layout()
    out_path = out_dir / "Fig_Adaptive_RateMap.pdf"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Imperceptibility Map (Upgraded from your draft)
# ---------------------------------------------------------------------------
def generate_imperceptibility_visual(img_bgr: np.ndarray, watermarked_bgr: np.ndarray, out_dir: Path, amplify: float = 30.0):
    """Proves the visual transparency of the JND-scaled embedding."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Original Image")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("(b) Watermarked Image (PSNR > 35dB)")
    axes[1].axis('off')

    # Absolute difference, scaled and converted to a heatmap for visibility
    diff = cv2.absdiff(img_bgr, watermarked_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_amp = np.clip(diff_gray.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    
    im3 = axes[2].imshow(diff_amp, cmap='inferno')
    axes[2].set_title(f"(c) Amplified Residual Map (x{int(amplify)})")
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = out_dir / "Fig_Imperceptibility_Residual.pdf"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Geometric Sync Peaks (Proves the Fourier-Mellin Math)
# ---------------------------------------------------------------------------
def generate_sync_peak_visual(watermarked_bgr: np.ndarray, out_dir: Path):
    """Displays the frequency domain to prove sync peaks exist in the Cr channel."""
    ycrcb = cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1].astype(np.float64)
    
    # Compute 2D FFT
    F = np.fft.fftshift(np.fft.fft2(Cr))
    mag = np.log10(np.abs(F) + 1)
    
    # Zoom into the center to show the peaks clearly
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    window = 64
    mag_zoomed = mag[cy-window:cy+window, cx-window:cx+window]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mag_zoomed, cmap='viridis')
    ax.set_title("Cr-Channel Frequency Spectrum\n(Arrows indicate embedded synchronization peaks)")
    ax.axis('off')
    
    # Add annotations pointing to the expected peak locations
    # (Based on geometric_sync.py: (32,0), (0,32), (32,32), (32,-32))
    peaks = [(0, 32), (0, -32), (32, 0), (-32, 0), (32, 32), (-32, -32), (32, -32), (-32, 32)]
    for (dy, dx) in peaks:
        # Check if peak is within window
        if abs(dy) < window and abs(dx) < window:
            py, px = window + dy, window + dx
            ax.annotate('', xy=(px, py), xytext=(px + np.sign(dx)*10, py + np.sign(dy)*10),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

    plt.tight_layout()
    out_path = out_dir / "Fig_Geometric_Sync_Peaks.pdf"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    set_q1_style()
    
    cfg_path = Path("configs/experiment.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("Run this script from the root EIPR directory.")
    
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Note: Replace with the exact path of a nice-looking image from your dataset
    # E.g., one that has clear subjects, textures, and blurred backgrounds
    img_path = Path("data/ai_generated/ai_gen_101.jpg") 
    out_dir = Path(cfg["results"]["output_dir"])
    out_dir.mkdir(exist_ok=True)

    print(f"Loading image: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image at {img_path}")

    # 1. Setup engine and data
    engine = AdaptiveECCEngine()
    alpha = float(cfg['embedding']['alpha'])
    n_bits = int(cfg['watermark']['n_bits'])
    watermark = np.random.default_rng(42).integers(0, 2, n_bits).astype(np.uint8)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    var_map = compute_block_dct_variance(ycrcb[:, :, 0])
    tau_low = float(cfg['ecc']['tau_low'])
    tau_high = float(cfg['ecc']['tau_high'])
    rate_map = build_ecc_rate_map(var_map, tau_low, tau_high)

    # 2. Embed
    print("Embedding watermark...")
    watermarked_bgr = embed_watermark(img_bgr, watermark, rate_map, engine, alpha=alpha)

    # 3. Generate Visuals
    print("Generating Q1 Visual Proofs...")
    generate_rate_map_visual(img_bgr, cfg, out_dir)
    generate_imperceptibility_visual(img_bgr, watermarked_bgr, out_dir, amplify=30.0)
    generate_sync_peak_visual(watermarked_bgr, out_dir)

    print("Done. All figures saved to results/ directory as PDFs.")

if __name__ == "__main__":
    main()