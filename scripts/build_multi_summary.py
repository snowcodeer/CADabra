"""Compose a single image summarising several sketch_roundtrip runs.

Reads pairs of (original 6-view grid, reconstruction 6-view grid) for
the part IDs listed below and stacks them into one tall PNG with a
per-row header.

Usage:
    python scripts/build_multi_summary.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
ORIG_DIR = REPO / "backend" / "outputs" / "deepcad_selected_grids"
RECON_DIR = REPO / "backend" / "outputs" / "multi"

ROWS = [
    ("061490", "U-channel (base + 2 pillars + 2 holes)"),
    ("089073", "L-bracket extruded along Z (+ 2 side holes)"),
    ("108596", "Octagonal plate (+ central hole + slot)"),
    ("014528", "Cross/plus profile extruded along Y (+ holes)"),
]

ROW_HEIGHT = 520
PANEL_WIDTH = 1200
PANEL_HEIGHT = 500
HEADER_HEIGHT = 36


def _load_pair(pid: str) -> tuple[Image.Image, Image.Image]:
    orig = Image.open(ORIG_DIR / f"deepcadimg_{pid}_grid.png").convert("RGB")
    recon = Image.open(RECON_DIR / f"{pid}_sketch_grid.png").convert("RGB")
    orig.thumbnail((PANEL_WIDTH, PANEL_HEIGHT))
    recon.thumbnail((PANEL_WIDTH, PANEL_HEIGHT))
    return orig, recon


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(out: Path) -> None:
    rows = [(pid, label, _load_pair(pid)) for pid, label in ROWS]

    canvas_w = PANEL_WIDTH * 2 + 30
    row_h = HEADER_HEIGHT + max(p[2][0].height for p in rows) + 20
    canvas_h = row_h * len(rows) + 60

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    label_font = _font(18)
    sub_font = _font(16)

    draw.text(
        (20, 16),
        "Sketch-plane pipeline — original (left) vs reconstruction (right)",
        fill="black",
        font=title_font,
    )

    y = 60
    for pid, label, (orig, recon) in rows:
        draw.text((20, y), f"deepcadimg_{pid}: {label}", fill="black", font=label_font)
        draw.text(
            (PANEL_WIDTH + 30, y),
            "reconstruction",
            fill="black",
            font=sub_font,
        )
        canvas.paste(orig, (10, y + HEADER_HEIGHT))
        canvas.paste(recon, (PANEL_WIDTH + 20, y + HEADER_HEIGHT))
        y += row_h

    canvas.save(out)
    print(f"Wrote {out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "backend" / "outputs" / "multi" / "summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
