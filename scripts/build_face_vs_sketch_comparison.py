"""Compose a single PNG showing FACE-GEOMETRY vs SKETCH-PLANE side-by-side.

Reads the two existing comparison artifacts and stacks them with
labels so the user can see the head-to-head verdict in one image.

Usage:
    python scripts/build_face_vs_sketch_comparison.py <stem> [output_path]

    e.g.  python scripts/build_face_vs_sketch_comparison.py deepcadimg_061490
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "backend" / "outputs"

ROW_HEADER_H = 40
TITLE_H = 60
PADDING = 10
TARGET_WIDTH = 2400  # rescale both rows to this width


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scaled(img: Image.Image, width: int) -> Image.Image:
    h = int(round(img.height * (width / img.width)))
    return img.resize((width, h), Image.LANCZOS)


def main(stem: str, out: Path) -> None:
    face_path = OUT_DIR / f"face_{stem}_comparison.png"
    sketch_path = OUT_DIR / f"sketch_{stem}_comparison.png"
    if not face_path.exists():
        raise SystemExit(f"missing {face_path}")
    if not sketch_path.exists():
        raise SystemExit(f"missing {sketch_path}")

    face = _scaled(Image.open(face_path).convert("RGB"), TARGET_WIDTH)
    sketch = _scaled(Image.open(sketch_path).convert("RGB"), TARGET_WIDTH)

    canvas_h = TITLE_H + ROW_HEADER_H + face.height + ROW_HEADER_H + sketch.height + PADDING * 4
    canvas = Image.new("RGB", (TARGET_WIDTH + PADDING * 2, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(26)
    row_font = _font(18)

    draw.text(
        (PADDING, 12),
        f"{stem} — face-geometry approach vs sketch-plane approach",
        fill="black", font=title_font,
    )

    y = TITLE_H
    draw.rectangle([0, y, canvas.width, y + ROW_HEADER_H], fill="#1a1a1a")
    draw.text(
        (PADDING, y + 8),
        "FACE-GEOMETRY APPROACH  —  left: clean face diagram from mesh   |   right: reconstructed 6-view render",
        fill="white", font=row_font,
    )
    y += ROW_HEADER_H
    canvas.paste(face, (PADDING, y))
    y += face.height + PADDING

    draw.rectangle([0, y, canvas.width, y + ROW_HEADER_H], fill="#1a1a1a")
    draw.text(
        (PADDING, y + 8),
        "SKETCH-PLANE APPROACH  —  left: original 6-view render   |   right: reconstructed 6-view render",
        fill="white", font=row_font,
    )
    y += ROW_HEADER_H
    canvas.paste(sketch, (PADDING, y))

    canvas.save(out)
    print(f"wrote {out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_face_vs_sketch_comparison.py <stem> [output_path]")
    stem = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR / f"face_vs_sketch_{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    main(stem, out)
