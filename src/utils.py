"""
Utility functions — Adaptive ECC Watermarking.

Covers:
  - JSON results I/O
  - LaTeX / terminal table formatting (for paper Table 1 / Table 2)
  - Rate-map and watermark-diff visualization helpers
  - Reproducibility seed helpers
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: dict[str, Any], path: str | pathlib.Path) -> None:
    """Serialize *results* dict to a JSON file, creating parent dirs if needed."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    print(f"[utils] Saved results → {p}")


def load_results(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a JSON results file produced by save_results / experiment_runner."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Terminal / LaTeX table formatting
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict], title: str = "Results") -> None:
    """
    Pretty-print a results dict to stdout in a human-readable table.

    Expected structure:
        {attack_name: {"BER_mean": float, "NC_mean": float, ...}, ...}
    """
    # Discover all metric keys from the first entry
    if not results:
        print("(empty results)")
        return

    sample = next(iter(results.values()))
    keys = list(sample.keys())

    col_w = 25
    met_w = 12

    header = f"{'Attack':<{col_w}}" + "".join(f"{k:>{met_w}}" for k in keys)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    print(sep)
    print(header)
    print(sep)
    for attack, metrics in results.items():
        row = f"{attack:<{col_w}}"
        for k in keys:
            v = metrics.get(k, float("nan"))
            row += f"{v:>{met_w}.4f}"
        print(row)
    print("=" * len(header))


def to_latex_table(
    results: dict[str, dict],
    caption: str = "Watermark robustness under various attacks.",
    label: str = "tab:results",
) -> str:
    """
    Render *results* as a LaTeX ``booktabs`` table string.

    Paste the output directly into the paper .tex source.
    """
    if not results:
        return "% empty results"

    sample = next(iter(results.values()))
    metric_keys = list(sample.keys())

    col_spec = "l" + "r" * len(metric_keys)
    header_row = "Attack & " + " & ".join(
        k.replace("_", r"\_") for k in metric_keys
    ) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header_row}",
        r"    \midrule",
    ]
    for attack, metrics in results.items():
        vals = " & ".join(f"{metrics.get(k, float('nan')):.4f}" for k in metric_keys)
        lines.append(rf"    {attack.replace('_', r' ')} & {vals} \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_rate_map(
    rate_map: np.ndarray,
    save_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> None:
    """
    Display or save a colour-coded heat-map of the per-block ECC rate_map.

    Colour legend: blue = high rate (smooth blocks), red = low rate (textured).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[utils] matplotlib not installed — skipping rate-map visualization.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rate_map, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="ECC rate")
    ax.set_title("Per-block adaptive ECC rate map")
    ax.set_xlabel("Block column")
    ax.set_ylabel("Block row")
    plt.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[utils] Rate map saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_watermark_diff(
    original: np.ndarray,
    watermarked: np.ndarray,
    amplify: float = 10.0,
    save_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> None:
    """
    Side-by-side view: original | watermarked | amplified difference.

    Useful to confirm perceptual invisibility (PSNR target ≥ 40 dB).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[utils] matplotlib not installed — skipping diff visualization.")
        return

    import cv2
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    wm_rgb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)
    diff = np.clip(
        np.abs(orig_rgb.astype(np.float32) - wm_rgb.astype(np.float32)) * amplify,
        0,
        255,
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig_rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(wm_rgb);   axes[1].set_title("Watermarked"); axes[1].axis("off")
    axes[2].imshow(diff);     axes[2].set_title(f"Diff ×{amplify:.0f}"); axes[2].axis("off")
    plt.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[utils] Diff plot saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set NumPy random seed for reproducible experiment runs."""
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for profiling experiment stages."""

    def __init__(self, label: str = ""):
        self.label = label
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = time.perf_counter() - self._start
        tag = f"[{self.label}] " if self.label else ""
        print(f"{tag}Elapsed: {elapsed:.2f}s")