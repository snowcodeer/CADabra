"""Compose a single PNG summarising face_roundtrip across multiple shapes.

Stacks the per-shape comparison images (original + reconstruction)
into one verdict sheet so the user can scan all results at once.

Usage:
    python scripts/build_face_multi_summary.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "backend" / "outputs"

TARGET_WIDTH = 2400
ROW_HEADER_H = 46
TITLE_H = 70
PADDING = 10


SHAPES = [
    {
        "id": "deepcadimg_000017",
        "label": "Washer (cylinder + central through-hole)",
        "verdict": "HIGH confidence — single circular extrude + concentric Ø60.4 mm hole. Topology fully recovered.",
    },
    {
        "id": "deepcadimg_108596",
        "label": "Octagonal plate (with central hole + slot)",
        "verdict": "HIGH confidence on what was extracted — but cylinder detector missed the central hole walls, "
                   "so reconstruction is the bare polygonal prism without the hole/slot. Failure is in face_extractor, not Claude.",
    },
    {
        "id": "deepcadimg_089073",
        "label": "L-bracket with 8 side-face holes (cylinder axes on +Y)",
        "verdict": "HIGH confidence — L-shaped sketch on the +Y face, eight Ø5.93 mm holes correctly placed. "
                   "Sketch-plane pipeline failed this case because it forced +Z as the base plane.",
    },
    {
        "id": "deepcadimg_014528",
        "label": "Slab pierced by vertical cylinder (boss above + boss below + through-hole + counterbore)",
        "verdict": "MEDIUM confidence — Claude correctly identified the slab+boss-up+boss-down topology with "
                   "through-hole and counterbore. Reconstruction matches.",
    },
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scaled(img: Image.Image, width: int) -> Image.Image:
    h = int(round(img.height * (width / img.width)))
    return img.resize((width, h), Image.LANCZOS)


def main(out: Path) -> None:
    rows = []
    for shape in SHAPES:
        path = OUT_DIR / f"face_{shape['id']}_comparison.png"
        if not path.exists():
            print(f"warn: missing {path}")
            continue
        rows.append((shape, _scaled(Image.open(path).convert("RGB"), TARGET_WIDTH)))

    if not rows:
        raise SystemExit("no comparison images found")

    canvas_h = TITLE_H + sum(ROW_HEADER_H + img.height + PADDING for _, img in rows) + PADDING
    canvas = Image.new("RGB", (TARGET_WIDTH + PADDING * 2, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(28)
    row_font = _font(18)

    draw.text(
        (PADDING, 12),
        "face-geometry approach — multi-shape summary  (left: face diagram from mesh   |   right: reconstructed 6-view render)",
        fill="black", font=title_font,
    )
    draw.text(
        (PADDING, 40),
        "Each row shows the actual mesh translated into a clean engineering drawing, then the part Claude built from that drawing.",
        fill="#444", font=_font(16),
    )

    y = TITLE_H
    for shape, img in rows:
        draw.rectangle([0, y, canvas.width, y + ROW_HEADER_H], fill="#1a1a1a")
        draw.text(
            (PADDING, y + 4),
            f"deepcadimg_{shape['id']}  —  {shape['label']}",
            fill="white", font=_font(20),
        )
        draw.text(
            (PADDING, y + 26),
            shape["verdict"],
            fill="#cccccc", font=row_font,
        )
        y += ROW_HEADER_H
        canvas.paste(img, (PADDING, y))
        y += img.height + PADDING

    canvas.save(out)
    print(f"wrote {out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "face_multi_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
