#!/usr/bin/env python3
"""End-to-end round-trip for the FACE-GEOMETRY architecture.

Sibling of ``scripts/sketch_roundtrip.py``. Where the sketch pipeline
treats the 6-view PNG as ground truth and uses OpenCV+Claude to
recover 2D profiles, this pipeline reads the STL mesh directly and
hands Claude clean per-face data.

Step-by-step:

    1. face_extractor.extract_faces  (no API call) — ExtractedGeometry
    2. face_diagram_renderer.render_face_diagrams — clean 2D PNG + JSON
    3. face_llm_client.call_claude_faces — SketchPartDescription
    4. sketch_builder.build_from_sketches — CadQuery code string
    5. exec the code in a subprocess (15 s timeout) — STL + STEP
    6. stl_renderer.render_stl_to_grid — reconstructed 6-view grid
    7. PIL side-by-side — face_comparison.png

All output filenames are prefixed with ``face_`` so the script can
run alongside ``sketch_roundtrip.py`` without trampling its
artifacts. Designed for a head-to-head A/B comparison of the two
architectures.

Usage:
    python scripts/face_roundtrip.py path/to/part.stl [--strict] [--open]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.face_diagram_renderer import render_face_diagrams  # noqa: E402
from backend.ai_infra.face_extractor import (  # noqa: E402
    ExtractedGeometry,
    extract_faces,
    summarise_geometry,
)
from backend.ai_infra.face_llm_client import call_claude_faces  # noqa: E402
from backend.ai_infra.sketch_builder import build_from_sketches  # noqa: E402
from backend.ai_infra.sketch_llm_client import warn_if_uncertain  # noqa: E402
from backend.ai_infra.sketch_models import SketchPartDescription  # noqa: E402
from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
STL_PATH = OUTPUT_DIR / "face_roundtrip.stl"
STEP_PATH = OUTPUT_DIR / "face_roundtrip.step"
GRID_PATH = OUTPUT_DIR / "face_roundtrip_grid.png"
COMPARISON_PATH = OUTPUT_DIR / "face_comparison.png"
GENERATED_CODE_PATH = OUTPUT_DIR / "face_roundtrip_generated.py"
GEOMETRY_SUMMARY_PATH = OUTPUT_DIR / "face_roundtrip_geometry.txt"


EXEC_TIMEOUT_SECONDS = 15

LABEL_BAR_HEIGHT = 30
DIVIDER_WIDTH = 4
DIVIDER_COLOR = (0x33, 0x33, 0x33)
LABEL_BG = (0x1A, 0x1A, 0x1A)
LABEL_FG = (0xFF, 0xFF, 0xFF)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
def _load_label_font(size: int = 18) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _print_header(label: str) -> None:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


def _print_sketch_summary(part: SketchPartDescription) -> None:
    print(part.model_dump_json(indent=2))
    print()
    print(f"Sketch operations ({len(part.sketches)}):")
    for op in part.sketches:
        prof = op.profile
        if prof.shape == "circle":
            dims = f"diameter={prof.diameter_mm}mm"
        elif prof.shape in ("rectangle",):
            dims = f"{prof.width_mm}x{prof.depth_mm}mm"
        else:
            dims = f"polyline({len(prof.vertices or [])} verts)"
        print(
            f"  [{op.order}] {op.operation:<7} on plane={op.plane:<3} "
            f"profile={prof.shape}({dims}) distance={op.distance_mm}mm "
            f"direction={op.direction} pos=({op.position_x}, {op.position_y})"
        )
    bb = part.bounding_box_mm
    print(f"Bounding box: {bb[0]} x {bb[1]} x {bb[2]} mm")
    print(
        f"Confidence: {part.confidence}"
        + (f"  notes={part.notes!r}" if part.notes else "")
    )


# ---------------------------------------------------------------------------
# Build / execute / re-render
# ---------------------------------------------------------------------------
def build_and_render(part: SketchPartDescription) -> tuple[str, bool]:
    _print_header("Step 4 — Build CadQuery code")
    code = build_from_sketches(part)
    print(code)
    GENERATED_CODE_PATH.write_text(code)
    print(f"(saved to {GENERATED_CODE_PATH.relative_to(REPO_ROOT)})")

    _print_header("Step 5 — Execute CadQuery in subprocess")
    wrapper = (
        code
        + "\n"
        + f"result.val().exportStep({str(STEP_PATH)!r})\n"
        + f"cq.exporters.export(result, {str(STL_PATH)!r})\n"
    )

    if STL_PATH.exists():
        STL_PATH.unlink()
    if STEP_PATH.exists():
        STEP_PATH.unlink()

    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        timeout=EXEC_TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if proc.returncode != 0 or not STL_PATH.exists():
        print("CadQuery execution FAILED.", file=sys.stderr)
        if proc.stdout:
            print("--- stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)

    print(
        f"STL written: {STL_PATH.relative_to(REPO_ROOT)} "
        f"({STL_PATH.stat().st_size} bytes)"
    )
    print(
        f"STEP written: {STEP_PATH.relative_to(REPO_ROOT)} "
        f"({STEP_PATH.stat().st_size} bytes)"
    )

    _print_header("Step 6 — Render reconstructed STL")
    try:
        render_stl_to_grid(STL_PATH, GRID_PATH, part_id=STL_PATH.stem)
        print(f"Rendered grid: {GRID_PATH.relative_to(REPO_ROOT)}")
        return code, True
    except Exception as exc:
        print(f"Rendering FAILED (continuing): {exc}", file=sys.stderr)
        traceback.print_exc()
        return code, False


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
def _resize_to_h(img: Image.Image, h: int) -> Image.Image:
    new_w = int(round(img.width * (h / img.height)))
    return img.resize((new_w, h), Image.LANCZOS)


def _draw_label_bar(
    canvas: Image.Image, draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont, x: int, width: int, text: str,
) -> None:
    bar = Image.new("RGB", (width, LABEL_BAR_HEIGHT), LABEL_BG)
    canvas.paste(bar, (x, 0))
    try:
        ascent, descent = font.getmetrics()
        ty = (LABEL_BAR_HEIGHT - (ascent + descent)) // 2
    except Exception:
        ty = 6
    draw.text((x + 10, ty), text, fill=LABEL_FG, font=font)


def step_compare(diagram_path: Path, render_ok: bool, part_id: str) -> bool:
    _print_header("Step 7 — Side-by-side comparison")
    if not render_ok or not GRID_PATH.exists():
        print("Skipping comparison: reconstructed grid is missing.")
        return False

    left = Image.open(diagram_path).convert("RGB")
    right = Image.open(GRID_PATH).convert("RGB")
    target_h = min(left.height, right.height)
    left = _resize_to_h(left, target_h)
    right = _resize_to_h(right, target_h)

    label_left = f"FACE GEOMETRY DIAGRAM — {part_id}"
    label_right = "RECONSTRUCTION — face_roundtrip.stl"

    total_w = left.width + DIVIDER_WIDTH + right.width
    total_h = LABEL_BAR_HEIGHT + target_h
    canvas = Image.new("RGB", (total_w, total_h), LABEL_BG)
    canvas.paste(left, (0, LABEL_BAR_HEIGHT))
    canvas.paste(right, (left.width + DIVIDER_WIDTH, LABEL_BAR_HEIGHT))
    divider = Image.new("RGB", (DIVIDER_WIDTH, total_h), DIVIDER_COLOR)
    canvas.paste(divider, (left.width, 0))

    draw = ImageDraw.Draw(canvas)
    font = _load_label_font(18)
    _draw_label_bar(canvas, draw, font, 0, left.width, label_left)
    _draw_label_bar(
        canvas, draw, font,
        left.width + DIVIDER_WIDTH, right.width, label_right,
    )

    canvas.save(COMPARISON_PATH)
    print(
        f"Wrote: {COMPARISON_PATH.relative_to(REPO_ROOT)} "
        f"({canvas.width}x{canvas.height})"
    )
    return True


# ---------------------------------------------------------------------------
# Optional --open behaviour: comparison + 3D viewers
# ---------------------------------------------------------------------------
def open_artifacts(original_stl: Path) -> None:
    _print_header("Open artifacts")

    if COMPARISON_PATH.exists() and sys.platform == "darwin":
        print(
            f"Opening face comparison PNG: "
            f"{COMPARISON_PATH.relative_to(REPO_ROOT)}"
        )
        subprocess.Popen(
            ["open", str(COMPARISON_PATH)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    elif COMPARISON_PATH.exists():
        print(
            f"Face comparison PNG ready: "
            f"{COMPARISON_PATH.relative_to(REPO_ROOT)}"
        )

    view3d_script = REPO_ROOT / "scripts" / "view3d.py"
    popen_kwargs = dict(
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    env = os.environ.copy()

    if original_stl.exists():
        print(f"Launching view3d on ORIGINAL: {original_stl.relative_to(REPO_ROOT)}")
        subprocess.Popen(
            [sys.executable, str(view3d_script), str(original_stl)],
            env=env, **popen_kwargs,
        )

    if STL_PATH.exists():
        print(
            f"Launching view3d on FACE RECONSTRUCTION: "
            f"{STL_PATH.relative_to(REPO_ROOT)}"
        )
        subprocess.Popen(
            [sys.executable, str(view3d_script), str(STL_PATH)],
            env=env, **popen_kwargs,
        )


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def step_summary(
    results: dict[str, bool], final_part: SketchPartDescription | None,
) -> None:
    _print_header("Final summary")
    for name, ok in results.items():
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {name}")
    if final_part is not None:
        print(f"  Final confidence: {final_part.confidence}")
        if final_part.notes:
            print(f"  Final notes: {final_part.notes!r}")
    print()
    print("Artifacts:")
    for path in (
        STL_PATH, STEP_PATH, GRID_PATH, COMPARISON_PATH,
        GENERATED_CODE_PATH, GEOMETRY_SUMMARY_PATH,
    ):
        if path.exists():
            print(f"  {path.relative_to(REPO_ROOT)}")
    print()
    print(
        "Compare backend/outputs/face_comparison.png with "
        "backend/outputs/sketch_comparison.png to see the new approach "
        "head-to-head with the sketch-plane pipeline."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip an STL through the face-geometry pipeline.",
    )
    parser.add_argument("stl", help="Path to the input .stl file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if final confidence is not 'high'.",
    )
    parser.add_argument(
        "--open",
        dest="open_artifacts",
        action="store_true",
        help="After the run, open the face comparison PNG and launch "
             "view3d for both the original and reconstructed STL.",
    )
    args = parser.parse_args(argv[1:])

    stl_path = Path(args.stl).resolve()
    if not stl_path.exists():
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    part_id = stl_path.stem
    diagram_path = OUTPUT_DIR / f"face_diagram_{part_id}.png"
    results: dict[str, bool] = {}
    final_part: SketchPartDescription | None = None

    # ------- Step 1: extract faces from the mesh -----------------------
    _print_header(f"Step 1 — Extract faces from {stl_path.name}")
    try:
        geometry: ExtractedGeometry = extract_faces(stl_path)
        summary = summarise_geometry(geometry)
        print(summary)
        GEOMETRY_SUMMARY_PATH.write_text(summary + "\n")
        print(
            f"\n{geometry.face_count} faces "
            f"({len(geometry.planar_faces)} planar, "
            f"{len(geometry.cylindrical_faces)} cylindrical), "
            f"bbox {geometry.bounding_box_mm[0]:.1f} x "
            f"{geometry.bounding_box_mm[1]:.1f} x "
            f"{geometry.bounding_box_mm[2]:.1f} mm."
        )
        print(f"(saved to {GEOMETRY_SUMMARY_PATH.relative_to(REPO_ROOT)})")
        results["1. extract faces"] = True
    except Exception as exc:
        print(f"Step 1 (extract faces) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Step 2: render the face diagram + sibling JSON ------------
    _print_header("Step 2 — Render face diagram")
    try:
        render_face_diagrams(geometry, diagram_path, part_id=part_id)
        print(f"Diagram: {diagram_path.relative_to(REPO_ROOT)}")
        print(
            f"Geometry JSON: "
            f"{diagram_path.with_suffix('.json').relative_to(REPO_ROOT)}"
        )
        results["2. render diagram"] = True
    except Exception as exc:
        print(f"Step 2 (render diagram) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Step 3: Claude — construction-sequence reasoning ---------
    _print_header("Step 3 — Claude (face-geometry construction reasoning)")
    try:
        part = call_claude_faces(diagram_path, geometry)
        _print_sketch_summary(part)
        warn_if_uncertain(part, "face-vision")
        results["3. claude faces"] = True
        final_part = part
    except Exception as exc:
        print(f"Step 3 (claude) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Steps 4-6: build, execute, re-render ----------------------
    try:
        build_and_render(part)
        results["4. build code"] = True
        results["5. execute cadquery"] = True
        results["6. render reconstruction"] = True
        render_ok = True
    except SystemExit:
        results["4. build code"] = True
        results["5. execute cadquery"] = False
        results["6. render reconstruction"] = False
        step_summary(results, final_part)
        return 1

    # ------- Step 7: comparison ----------------------------------------
    compare_ok = step_compare(diagram_path, render_ok, part_id)
    results["7. compare"] = compare_ok

    # ------- Final summary ---------------------------------------------
    step_summary(results, final_part)

    if args.open_artifacts:
        open_artifacts(stl_path)

    if args.strict and final_part is not None and final_part.confidence != "high":
        print(
            f"--strict: final confidence is {final_part.confidence!r}, "
            "exiting non-zero.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
