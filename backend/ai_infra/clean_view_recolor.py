"""Re-colorize the gpt-image-2 grayscale clean view back to RdYlBu_r depth.

The cleaned ortho PNG produced by ``scripts/synthesize_clean_views.py`` is
intentionally grayscale: gpt-image-2 was repainting any rainbow palette
back into "natural" colours and destroying the depth encoding. Once
gpt-image-2 has finished its work we can safely re-apply the colormap so
downstream Claude calls see the same red=near / blue=far convention used
by the rest of the pipeline (``backend/pipeline/stl_renderer.py`` uses
RdYlBu_r for its depth channel).

Layout assumed (matches ``synthesize_clean_views.render_clean_input_grid``):
    1536 x 1024 canvas, 3 cols x 2 rows of 512x512 cells. Each cell is
    split LEFT/RIGHT into a 256x512 grayscale depth panel and a 256x512
    silhouette panel. Only the LEFT half is recoloured; the silhouette
    half is preserved verbatim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps as _mpl_colormaps
from PIL import Image

# Mirror the constants in synthesize_clean_views so the recoloriser stays
# correct if the layout ever changes (a future merge can centralise these).
GRID_COLS = 3
GRID_ROWS = 2
CANVAS_W = 1536
CANVAS_H = 1024
CELL_W = CANVAS_W // GRID_COLS  # 512
CELL_H = CANVAS_H // GRID_ROWS  # 512
HALF_W = CELL_W // 2            # 256

DEPTH_COLORMAP = "RdYlBu_r"

# Background detection. The synthesise step paints both depth and
# silhouette backgrounds at (245, 245, 245) / pure white. After
# gpt-image-2 the background is rarely an exact integer match, so the
# threshold is generous: any pixel whose channels are all >= 235 AND
# whose RGB span is small (i.e. clearly desaturated near-white) counts.
BG_BRIGHTNESS_THRESHOLD = 235
BG_CHANNEL_SPREAD_MAX = 12


def _is_background(panel: np.ndarray) -> np.ndarray:
    """Boolean mask where True = treat as background (no depth)."""
    bright = (panel >= BG_BRIGHTNESS_THRESHOLD).all(axis=-1)
    spread = panel.max(axis=-1).astype(np.int16) - panel.min(axis=-1).astype(np.int16)
    desaturated = spread <= BG_CHANNEL_SPREAD_MAX
    return bright & desaturated


def _recolor_depth_panel(panel: np.ndarray) -> np.ndarray:
    """Convert the grayscale depth panel back to RdYlBu_r.

    The synthesise step encoded depth as gray = 60 + 170 * normalised, so
    we invert that by collapsing channels to a single luminance and then
    re-applying the matplotlib colormap. Any pixel that looks like
    background stays white.
    """
    bg = _is_background(panel)
    # Use the green channel as the luminance signal (matches the way the
    # original encoder built the gray image with equal R=G=B). gpt-image-2
    # sometimes shifts the channels slightly so an average across the
    # three is a tiny bit more robust.
    luminance = panel.astype(np.float32).mean(axis=-1)
    normalised = np.clip((luminance - 60.0) / 170.0, 0.0, 1.0)
    # Invert so the RdYlBu_r convention (red=near=darker_input,
    # blue=far=lighter_input) holds. The encoder set near=darker, so
    # the smaller the luminance the closer to camera. RdYlBu_r maps
    # 0->red and 1->blue, so we want normalised=0 for near-camera
    # pixels — i.e. invert the (luminance - 60)/170 rescale's direction.
    cmap = _mpl_colormaps[DEPTH_COLORMAP]
    coloured = cmap(1.0 - normalised)[:, :, :3]  # drop alpha
    out = (coloured * 255).astype(np.uint8)
    out[bg] = (255, 255, 255)
    return out


def recolorize_clean_view(in_path: str | Path, out_path: str | Path) -> Path:
    """Read a grayscale clean view PNG and write a recolourised copy.

    Returns the output path. Silhouette halves and overall layout are
    preserved verbatim; only the depth halves are re-coloured.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    img = np.asarray(Image.open(in_path).convert("RGB"))
    if img.shape[:2] != (CANVAS_H, CANVAS_W):
        raise ValueError(
            f"unexpected canvas shape {img.shape[:2]}, "
            f"expected ({CANVAS_H}, {CANVAS_W}). "
            "Did the clean view layout change?"
        )

    out = img.copy()
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cy = row * CELL_H
            cx = col * CELL_W
            depth = img[cy:cy + CELL_H, cx:cx + HALF_W]
            out[cy:cy + CELL_H, cx:cx + HALF_W] = _recolor_depth_panel(depth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path, format="PNG")
    return out_path
