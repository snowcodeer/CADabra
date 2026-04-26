#!/usr/bin/env python3
"""Batch-run ``face_roundtrip`` over several STLs and produce a single
triptych summary so you can scan input vs output for many shapes at once.

Inputs may be individual ``.stl`` files OR directories — each directory
expands to every ``.stl`` it contains (sorted, top level only).

For each input STL ``<stem>.stl`` the script:

    1. Runs ``scripts/face_roundtrip.py <stem>.stl`` (unless --no-run).
    2. Locates the two artifacts that matter for visual comparison:
         - INPUT  → ``backend/outputs/deepcad_selected_grids/<stem>_grid.png``
                    (6-view render of the input STL; rendered on-the-fly
                    if no pre-baked grid exists)
         - OUTPUT → ``backend/outputs/face_<stem>_recon_grid.png``
                    (6-view render of the rebuilt STL)
    3. Stacks them horizontally into one row per shape.

All rows are then stacked vertically into a single summary PNG with
labels and headers, and (with ``--open``) opened in Preview.

Use this when you want to look at "input STL vs what we generated"
across multiple parts in one glance.

Usage:
    # Run on every STL in a folder:
    python scripts/face_batch.py backend/outputs/deepcad_selected_stl --open

    # First 4 only (cheap smoke test, won't burn all the API credit):
    python scripts/face_batch.py backend/outputs/deepcad_selected_stl --limit 4 --open

    # Specific STLs:
    python scripts/face_batch.py path/to/a.stl path/to/b.stl --open

    # Mix files and directories:
    python scripts/face_batch.py backend/outputs/deepcad_selected_stl extra.stl --open

    # Skip running (rebuild summary from existing artifacts on disk):
    python scripts/face_batch.py backend/outputs/deepcad_selected_stl --no-run --open

    # Custom output path:
    python scripts/face_batch.py *.stl --output backend/outputs/my_batch.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
DEEPCAD_GRIDS_DIR = OUTPUT_DIR / "deepcad_selected_grids"

# Canvas geometry. Each panel inside a row is rescaled to PANEL_WIDTH
# while preserving aspect ratio, so the two panels always line up
# regardless of the source PNG dimensions.
PANEL_WIDTH = 1800
ROW_HEADER_H = 48
TITLE_H = 80
PADDING = 12
GUTTER = 8

LABEL_BG = (0x1A, 0x1A, 0x1A)
LABEL_FG = (0xFF, 0xFF, 0xFF)
SUBLABEL_FG = (0xCC, 0xCC, 0xCC)
PANEL_BG = (0xFF, 0xFF, 0xFF)
PANEL_LABEL_BG = (0x33, 0x33, 0x33)
GUTTER_COLOR = (0x44, 0x44, 0x44)


# ---------------------------------------------------------------------------
# Per-shape artifact bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeArtifacts:
    """All paths needed to render one row of the batch summary."""

    stl_path: Path
    stem: str
    original_grid: Path  # 6-view of the input STL
    recon_grid: Path  # face_<stem>_recon_grid.png — 6-view of the rebuilt STL

    @property
    def all_present(self) -> bool:
        return self.original_grid.is_file() and self.recon_grid.is_file()


def _resolve_original_grid(stem: str, stl_path: Path) -> Path:
    """Return a path to a 6-view PNG of the input STL.

    Uses the pre-rendered DeepCAD grid if one exists for this stem;
    otherwise renders one into ``backend/outputs/`` (cached for future
    runs) so brand-new STLs work out of the box.
    """
    pre = DEEPCAD_GRIDS_DIR / f"{stem}_grid.png"
    if pre.is_file():
        return pre

    cached = OUTPUT_DIR / f"input_{stem}_grid.png"
    if cached.is_file():
        return cached
    print(f"  [render] no pre-rendered grid for {stem}; rendering -> {cached.name}")
    render_stl_to_grid(stl_path, cached, part_id=stem)
    return cached


def _artifacts_for(stl_path: Path) -> ShapeArtifacts:
    stem = stl_path.stem
    return ShapeArtifacts(
        stl_path=stl_path,
        stem=stem,
        original_grid=_resolve_original_grid(stem, stl_path),
        recon_grid=OUTPUT_DIR / f"face_{stem}_recon_grid.png",
    )


# ---------------------------------------------------------------------------
# Pipeline runner (subprocess wrapper around face_roundtrip.py)
# ---------------------------------------------------------------------------
def _run_face_roundtrip(stl_path: Path) -> bool:
    """Run ``scripts/face_roundtrip.py`` for one STL with live-streamed
    output. Returns True if the subprocess exited 0."""
    print()
    print("#" * 78)
    print(f"# Running face_roundtrip on {stl_path.name}")
    print("#" * 78)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "face_roundtrip.py"), str(stl_path)],
        cwd=str(REPO_ROOT),
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------
def _font(size: int) -> ImageFont.ImageFont:
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


def _panel(img: Image.Image, label: str) -> Image.Image:
    """Wrap a panel image with a small label bar above it so each
    column is self-describing (INPUT / OUTPUT)."""
    label_h = 32
    out = Image.new("RGB", (img.width, img.height + label_h), PANEL_LABEL_BG)
    draw = ImageDraw.Draw(out)
    font = _font(18)
    draw.text((10, 6), label, fill=LABEL_FG, font=font)
    out.paste(img, (0, label_h))
    return out


def _build_row(art: ShapeArtifacts) -> Image.Image:
    """Compose one row: [input STL render | output STL render].

    Both panels are 6-view grids rendered by the same renderer, so the
    comparison is genuinely apples-to-apples."""
    panels = [
        _panel(
            _scaled(Image.open(art.original_grid).convert("RGB"), PANEL_WIDTH),
            f"INPUT STL — {art.stem}.stl",
        ),
        _panel(
            _scaled(Image.open(art.recon_grid).convert("RGB"), PANEL_WIDTH),
            f"OUTPUT STL — face_{art.stem}.stl  (rebuilt by face-geometry pipeline)",
        ),
    ]
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + GUTTER * (len(panels) - 1)
    row = Image.new("RGB", (w, h), GUTTER_COLOR)
    x = 0
    for i, p in enumerate(panels):
        row.paste(p, (x, 0))
        x += p.width
        if i < len(panels) - 1:
            x += GUTTER
    return row


def _build_summary(arts: list[ShapeArtifacts], out_path: Path) -> None:
    rows = []
    for art in arts:
        if not art.all_present:
            missing = [
                p.name
                for p in (art.original_grid, art.recon_grid)
                if not p.is_file()
            ]
            print(f"  [skip] {art.stem}: missing {', '.join(missing)}")
            continue
        rows.append((art, _build_row(art)))

    if not rows:
        raise SystemExit(
            "no rows to compose — none of the inputs have a complete artifact set"
        )

    canvas_w = max(row.width for _, row in rows) + PADDING * 2
    canvas_h = (
        TITLE_H
        + sum(ROW_HEADER_H + row.height + PADDING for _, row in rows)
        + PADDING
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(30)
    sub_font = _font(17)
    row_font = _font(20)

    draw.text(
        (PADDING, 14),
        f"face-geometry batch summary  ·  {len(rows)} shape{'s' if len(rows) != 1 else ''}",
        fill="black", font=title_font,
    )
    draw.text(
        (PADDING, 50),
        "Each row: 6-view render of the input STL  vs.  6-view render of the rebuilt STL  (same renderer on both sides)",
        fill="#444", font=sub_font,
    )

    y = TITLE_H
    for art, row in rows:
        draw.rectangle([0, y, canvas_w, y + ROW_HEADER_H], fill=LABEL_BG)
        draw.text(
            (PADDING, y + 4),
            f"{art.stem}",
            fill=LABEL_FG, font=row_font,
        )
        draw.text(
            (PADDING, y + 26),
            f"input: {art.stl_path.relative_to(REPO_ROOT) if str(art.stl_path).startswith(str(REPO_ROOT)) else art.stl_path}"
            f"   |   comparison PNG: backend/outputs/face_{art.stem}_comparison.png",
            fill=SUBLABEL_FG, font=_font(14),
        )
        y += ROW_HEADER_H
        canvas.paste(row, (PADDING, y))
        y += row.height + PADDING

    canvas.save(out_path)
    print()
    out_abs = out_path.resolve()
    try:
        shown = out_abs.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_abs
    print(f"wrote {shown} ({canvas.width}x{canvas.height})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run face_roundtrip on multiple STLs and stack input-vs-output "
            "6-view comparisons (one row per shape) into one summary PNG."
        ),
    )
    parser.add_argument(
        "stls",
        nargs="+",
        help=(
            "One or more input STL files, OR directories — each directory "
            "expands to every .stl inside it (top level, sorted)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of STLs processed (handy for cheap smoke tests).",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running face_roundtrip; just rebuild the summary "
             "from existing face_<stem>_* artifacts on disk.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(OUTPUT_DIR / "face_batch_summary.png"),
        help="Where to write the summary PNG (default: backend/outputs/face_batch_summary.png).",
    )
    parser.add_argument(
        "--open",
        dest="open_after",
        action="store_true",
        help="Open the summary PNG in Preview when done (macOS only).",
    )
    args = parser.parse_args(argv[1:])

    stl_paths: list[Path] = []
    seen: set[Path] = set()
    for raw in args.stls:
        p = Path(raw).resolve()
        if p.is_dir():
            children = sorted(p.glob("*.stl"))
            if not children:
                print(f"[warn] no .stl files in directory: {p}", file=sys.stderr)
            for child in children:
                if child not in seen:
                    stl_paths.append(child)
                    seen.add(child)
        elif p.is_file():
            if p.suffix.lower() != ".stl":
                print(f"[warn] not an .stl file (skipping): {p}", file=sys.stderr)
                continue
            if p not in seen:
                stl_paths.append(p)
                seen.add(p)
        else:
            print(f"[error] path does not exist: {p}", file=sys.stderr)
            return 2

    if not stl_paths:
        print("[error] no STL files to process.", file=sys.stderr)
        return 2

    if args.limit is not None and args.limit > 0:
        stl_paths = stl_paths[: args.limit]

    print(f"[batch] processing {len(stl_paths)} STL(s):")
    for p in stl_paths:
        print(f"   - {p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the pipeline (or skip and assume artifacts already exist).
    if not args.no_run:
        for stl in stl_paths:
            ok = _run_face_roundtrip(stl)
            if not ok:
                print(
                    f"[warn] face_roundtrip failed for {stl.name}; "
                    "continuing — that row will be skipped if artifacts are missing.",
                    file=sys.stderr,
                )

    # Resolve the per-shape artifact bundles. Original grids fall
    # back to on-the-fly rendering (cached as input_<stem>_grid.png)
    # if no pre-baked grid exists.
    arts = [_artifacts_for(p) for p in stl_paths]
    _build_summary(arts, Path(args.output))

    out_path = Path(args.output)
    if args.open_after and sys.platform == "darwin" and out_path.is_file():
        subprocess.Popen(
            ["open", str(out_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
