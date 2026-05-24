"""
utils.py — I/O, table formatting, visualization, and timing utilities.

Covers:
  • JSON result serialization / deserialization
  • Terminal and LaTeX ``booktabs`` table generation (paper-ready)
  • Rate-map and watermark-diff visualization helpers
  • Global reproducibility seed helper
  • Wall-clock Timer context manager
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
    """Serialize *results* to JSON, creating parent directories if needed."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, sort_keys=False))
    print(f"[utils] Saved results → {p}")


def load_results(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a JSON results file produced by ``save_results``."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict], title: str = "Results") -> None:
    """
    Pretty-print a nested results dict to stdout.

    Expected structure:
        {attack_name: {metric_key: float, ...}, ...}

    Includes mean ± std display when both ``X_mean`` and ``X_std`` keys exist
    for the same base metric X.
    """
    if not results:
        print("(empty results)")
        return

    sample = next(iter(results.values()))
    all_keys = list(sample.keys())

    # Pair up mean / std keys for compact display
    display_keys: list[str] = []
    skip: set[str] = set()
    for k in all_keys:
        if k in skip:
            continue
        std_key = k.replace("_mean", "_std")
        if k.endswith("_mean") and std_key in sample:
            display_keys.append(k)  # will render as "mean ± std"
            skip.add(std_key)
        else:
            display_keys.append(k)

    # Column widths
    col_w   = 26
    val_w   = 20
    header  = f"{'Attack':<{col_w}}" + "".join(f"{k:^{val_w}}" for k in display_keys)
    sep     = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)

    for attack, metrics in results.items():
        row = f"{attack:<{col_w}}"
        for k in display_keys:
            std_key = k.replace("_mean", "_std")
            if k.endswith("_mean") and std_key in metrics:
                cell = f"{metrics[k]:.4f}±{metrics[std_key]:.4f}"
            else:
                v = metrics.get(k, float("nan"))
                cell = f"{v:.4f}"
            row += f"{cell:^{val_w}}"
        print(row)

    print("═" * len(header))


# ---------------------------------------------------------------------------
# LaTeX table (paper-ready, booktabs)
# ---------------------------------------------------------------------------

def to_latex_table(
    results: dict[str, dict],
    caption: str = "Watermark robustness under various attacks.",
    label: str = "tab:results",
    selected_metrics: list[str] | None = None,
    highlight_best: bool = True,
) -> str:
    """
    Render *results* as a LaTeX ``booktabs`` table string.

    Features:
      • Mean ± std rendered as ``$x \\pm y$`` (requires siunitx or plain math).
      • Best BER value per column highlighted with ``\\textbf`` when
        ``highlight_best=True``.
      • ``selected_metrics`` filters to a subset of metric keys.

    Args:
        results:          nested dict {attack: {metric: value}}.
        caption:          LaTeX table caption.
        label:            LaTeX \label{} value.
        selected_metrics: if given, only these metric keys are included.
        highlight_best:   bold the best (min BER / max NC / max PSNR / max SSIM)
                          value in each metric column.

    Returns:
        Multi-line LaTeX string ready to paste into the paper.
    """
    if not results:
        return "% empty results — nothing to render"

    sample = next(iter(results.values()))
    all_keys = list(sample.keys())

    # Select and pair mean/std keys
    if selected_metrics is not None:
        mean_keys = [k for k in all_keys if k in selected_metrics]
    else:
        mean_keys = [k for k in all_keys if k.endswith("_mean")]
        if not mean_keys:
            mean_keys = all_keys  # fallback: use all keys as-is

    def _header_name(k: str) -> str:
        base = k.replace("_mean", "").replace("_", "\\_")
        return base.upper() if len(base) <= 4 else base.title()

    # Determine "best" direction per metric (lower BER/std = better; higher NC/PSNR/SSIM = better)
    def _is_lower_better(k: str) -> bool:
        return "BER" in k or "ber" in k

    # Collect all values per metric key for highlighting
    metric_vals: dict[str, list[float]] = {k: [] for k in mean_keys}
    for m in results.values():
        for k in mean_keys:
            metric_vals[k].append(float(m.get(k, float("nan"))))

    def _best_val(k: str) -> float:
        vals = [v for v in metric_vals[k] if not np.isnan(v)]
        if not vals:
            return float("nan")
        return min(vals) if _is_lower_better(k) else max(vals)

    best_vals = {k: _best_val(k) for k in mean_keys}

    col_spec = "l" + "r" * len(mean_keys)
    header_cells = " & ".join(_header_name(k) for k in mean_keys)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Attack & {header_cells} \\\\",
        r"    \midrule",
    ]

    for attack, metrics in results.items():
        cells = []
        for k in mean_keys:
            std_key = k.replace("_mean", "_std")
            v = float(metrics.get(k, float("nan")))
            has_std = std_key in metrics
            std_v = float(metrics.get(std_key, 0.0)) if has_std else None

            if has_std and std_v is not None:
                cell = f"${v:.4f} \\pm {std_v:.4f}$"
            else:
                cell = f"{v:.4f}"

            if highlight_best and not np.isnan(v) and not np.isnan(best_vals[k]):
                if abs(v - best_vals[k]) < 1e-9:
                    cell = rf"\textbf{{{cell}}}"
            cells.append(cell)

        attack_tex = attack.replace("_", r"\_")
        lines.append(f"    {attack_tex} & {' & '.join(cells)} \\\\")

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
    title: str = "Per-block adaptive ECC rate map",
) -> None:
    """
    Display or save a colour-coded heat-map of the per-block ECC rate_map.

    Colour legend: blue = high rate (smooth / fragile blocks),
                   red  = low rate  (textured / robust blocks).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[utils] matplotlib not installed — skipping rate-map visualisation.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rate_map, cmap="coolwarm_r", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ECC rate", fontsize=11)
    cbar.set_ticks([0.25, 0.50, 0.75])
    cbar.set_ticklabels(["0.25 (textured)", "0.50 (mid)", "0.75 (smooth)"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Block column", fontsize=10)
    ax.set_ylabel("Block row", fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    psnr: float | None = None,
    ssim: float | None = None,
) -> None:
    """
    Side-by-side view: original | watermarked | amplified difference.

    Confirms perceptual invisibility (target PSNR ≥ 40 dB, SSIM ≥ 0.95).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[utils] matplotlib not installed — skipping diff visualisation.")
        return

    orig_rgb = original[..., ::-1]           # BGR → RGB
    wm_rgb   = watermarked[..., ::-1]
    diff = np.clip(
        np.abs(orig_rgb.astype(np.float32) - wm_rgb.astype(np.float32)) * amplify,
        0, 255,
    ).astype(np.uint8)

    subtitle = ""
    if psnr is not None:
        subtitle += f"PSNR = {psnr:.2f} dB   "
    if ssim is not None:
        subtitle += f"SSIM = {ssim:.4f}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, img, ttl in zip(
        axes,
        [orig_rgb, wm_rgb, diff],
        ["Original", "Watermarked", f"Diff ×{amplify:.0f}"],
    ):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    if subtitle:
        fig.suptitle(subtitle, fontsize=10, y=0.02)
    plt.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Diff plot saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_ber_distribution(
    bers: list[float],
    attack_name: str = "",
    save_path: str | pathlib.Path | None = None,
    show: bool = False,
) -> None:
    """
    Histogram of per-image BER values with mean and ±1σ lines.

    Useful for diagnosing whether failures are systematic (shifted mean)
    or image-specific (high variance).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bers_arr = np.array(bers, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bers_arr, bins=30, range=(0, 0.5), color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(bers_arr.mean(), color="red",    lw=1.8, ls="--", label=f"Mean {bers_arr.mean():.4f}")
    ax.axvline(bers_arr.mean() + bers_arr.std(), color="orange", lw=1.2, ls=":", label=f"±1σ")
    ax.axvline(bers_arr.mean() - bers_arr.std(), color="orange", lw=1.2, ls=":")
    ax.set_xlabel("BER", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"BER Distribution — {attack_name}", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set NumPy and Python random seeds for reproducible runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """
    Wall-clock timer context manager.

    Usage::

        with Timer("full experiment") as t:
            run_experiment()
        # prints: [full experiment] Elapsed: 42.17 s
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        tag = f"[{self.label}] " if self.label else ""
        mins, secs = divmod(self.elapsed, 60)
        if mins >= 1:
            print(f"{tag}Elapsed: {int(mins)}m {secs:.1f}s")
        else:
            print(f"{tag}Elapsed: {secs:.2f}s")