#!/usr/bin/env python3
"""End-to-end round-trip for the SKETCH-PLANE architecture.

Sibling of ``scripts/roundtrip_demo.py`` (which uses the original
``BaseBody + features`` schema). This driver uses the OpenCV-assisted
sketch pipeline:

    1. OpenCV reads the 6-view grid PNG and prints the contour summary
       it would feed to Claude.
    2. Claude returns a SketchPartDescription (one Anthropic call).
    3. ``build_from_sketches`` -> CadQuery code string.
    4. exec the code in a subprocess (15 s timeout) -> STL + STEP.
    5. Re-render the produced STL -> reconstructed 6-view grid.
    6. Side-by-side comparison PNG (original vs reconstructed).
    7. Summary.

All output filenames are prefixed with ``sketch_`` so this script can
be run alongside the original ``roundtrip_demo.py`` without trampling
its artifacts. Designed for an A/B comparison of the two architectures.

Usage:
    python scripts/sketch_roundtrip.py path/to/grid.png [--strict] [--open]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.contour_extractor import (  # noqa: E402
    extract_all_views,
    summarise_contours,
)
from backend.ai_infra.sketch_builder import build_from_sketches  # noqa: E402
from backend.ai_infra.sketch_llm_client import (  # noqa: E402
    call_claude_sketch,
    warn_if_uncertain,
)
from backend.ai_infra.sketch_models import SketchPartDescription  # noqa: E402
from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402


# The renderer normalises every input mesh to longest-edge=100mm before
# rendering. Mirror that here so the bounds we hand to the contour
# extractor match the bounds OpenCV is actually measuring against.
NORMALISE_LONGEST_MM = 100.0


def _normalised_bounds_mm(stl_path: Path) -> tuple[float, float, float, float, float, float]:
    """Return the original mesh's bounds AS RENDERED (normalised to
    longest-edge = NORMALISE_LONGEST_MM, mesh re-centred at origin XY).

    This must match ``backend.pipeline.stl_renderer._normalise_mesh``.
    """
    mesh = pv.read(str(stl_path))
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    x_ext, y_ext, z_ext = xmax - xmin, ymax - ymin, zmax - zmin
    longest = max(x_ext, y_ext, z_ext, 1e-9)
    s = NORMALISE_LONGEST_MM / longest
    return (
        xmin * s, xmax * s,
        ymin * s, ymax * s,
        zmin * s, zmax * s,
    )


OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
STL_PATH = OUTPUT_DIR / "sketch_roundtrip.stl"
STEP_PATH = OUTPUT_DIR / "sketch_roundtrip.step"
GRID_PATH = OUTPUT_DIR / "sketch_roundtrip_grid.png"
COMPARISON_PATH = OUTPUT_DIR / "sketch_comparison.png"
GENERATED_CODE_PATH = OUTPUT_DIR / "sketch_roundtrip_generated.py"
CONTOUR_SUMMARY_PATH = OUTPUT_DIR / "sketch_roundtrip_contours.txt"

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
            dims = f"diameter={prof.diameter_mm or min(prof.width_mm, prof.depth_mm)}mm"
        else:
            dims = f"{prof.width_mm}x{prof.depth_mm}mm"
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
# Build / execute / render
# ---------------------------------------------------------------------------
def build_and_render(part: SketchPartDescription) -> tuple[str, bool]:
    _print_header("Steps 3-5 — Build CadQuery code")
    code = build_from_sketches(part)
    print(code)
    GENERATED_CODE_PATH.write_text(code)
    print(f"(saved to {GENERATED_CODE_PATH.relative_to(REPO_ROOT)})")

    _print_header("Steps 3-5 — Execute CadQuery in subprocess")
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

    _print_header("Steps 3-5 — Render reconstructed STL")
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


def step_compare(original_path: Path, render_ok: bool) -> bool:
    _print_header("Step 6 — Side-by-side comparison")
    if not render_ok or not GRID_PATH.exists():
        print("Skipping comparison: reconstructed grid is missing.")
        return False

    left = Image.open(original_path).convert("RGB")
    right = Image.open(GRID_PATH).convert("RGB")
    target_h = min(left.height, right.height)
    left = _resize_to_h(left, target_h)
    right = _resize_to_h(right, target_h)

    label_left = f"ORIGINAL — {original_path.name}"
    label_right = "SKETCH RECONSTRUCTION — sketch_roundtrip.stl"

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
# Optional: open the comparison + 3D viewers (mirrors roundtrip_demo)
# ---------------------------------------------------------------------------
def _find_original_stl(image_path: Path) -> Path | None:
    stem = image_path.stem
    if stem.endswith("_grid"):
        stem = stem[: -len("_grid")]
    candidate_dirs = [
        REPO_ROOT / "backend" / "outputs" / "deepcad_selected_stl",
        REPO_ROOT / "backend" / "outputs",
        REPO_ROOT / "backend" / "sample_data",
    ]
    for d in candidate_dirs:
        candidate = d / f"{stem}.stl"
        if candidate.is_file():
            return candidate
    return None


def open_artifacts(original_image_path: Path) -> None:
    _print_header("Open artifacts")

    if COMPARISON_PATH.exists() and sys.platform == "darwin":
        print(
            f"Opening sketch comparison PNG: "
            f"{COMPARISON_PATH.relative_to(REPO_ROOT)}"
        )
        subprocess.Popen(
            ["open", str(COMPARISON_PATH)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    elif COMPARISON_PATH.exists():
        print(
            f"Sketch comparison PNG ready: "
            f"{COMPARISON_PATH.relative_to(REPO_ROOT)}"
        )

    view3d_script = REPO_ROOT / "scripts" / "view3d.py"
    popen_kwargs = dict(
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    env = os.environ.copy()

    original_stl = _find_original_stl(original_image_path)
    if original_stl is not None:
        print(f"Launching view3d on ORIGINAL: {original_stl.relative_to(REPO_ROOT)}")
        subprocess.Popen(
            [sys.executable, str(view3d_script), str(original_stl)],
            env=env, **popen_kwargs,
        )

    if STL_PATH.exists():
        print(f"Launching view3d on SKETCH RECONSTRUCTION: {STL_PATH.relative_to(REPO_ROOT)}")
        subprocess.Popen(
            [sys.executable, str(view3d_script), str(STL_PATH)],
            env=env, **popen_kwargs,
        )


def step_summary(
    results: dict[str, bool], final_part: SketchPartDescription | None,
) -> None:
    _print_header("Step 7 — Summary")
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
        GENERATED_CODE_PATH, CONTOUR_SUMMARY_PATH,
    ):
        if path.exists():
            print(f"  {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip a 6-view grid PNG through the sketch-plane pipeline.",
    )
    parser.add_argument("image", help="Path to the input 6-view grid PNG.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if final confidence is not 'high'.",
    )
    parser.add_argument(
        "--open",
        dest="open_artifacts",
        action="store_true",
        help="After the run, open the sketch comparison PNG and launch "
             "view3d for both the original and reconstructed STL.",
    )
    args = parser.parse_args(argv[1:])

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    final_part: SketchPartDescription | None = None

    # ------- Step 0: Find original STL so we can compute world-mm bounds.
    # Without these, OpenCV vertices come back normalised 0-1 and Claude
    # has to re-derive scale from the labels — slower and lossier.
    original_stl = _find_original_stl(image_path)
    bounds_mm: tuple[float, float, float, float, float, float] | None = None
    if original_stl is not None:
        try:
            bounds_mm = _normalised_bounds_mm(original_stl)
            print(
                f"[info] using world-mm contours from "
                f"{original_stl.relative_to(REPO_ROOT)}: "
                f"X[{bounds_mm[0]:.1f},{bounds_mm[1]:.1f}] "
                f"Y[{bounds_mm[2]:.1f},{bounds_mm[3]:.1f}] "
                f"Z[{bounds_mm[4]:.1f},{bounds_mm[5]:.1f}]"
            )
        except Exception as exc:
            print(f"[warn] could not read original STL bounds: {exc}", file=sys.stderr)
    else:
        print(
            "[warn] no original STL found next to the input grid — "
            "OpenCV will report normalised 0-1 coords only.",
            file=sys.stderr,
        )

    # ------- Step 1: OpenCV contour extraction (no API call) -------
    _print_header("Step 1 — OpenCV contour extraction")
    try:
        contours = extract_all_views(str(image_path), mesh_bounds_mm=bounds_mm)
        summary = summarise_contours(contours)
        print(summary)
        CONTOUR_SUMMARY_PATH.write_text(summary + "\n")
        print(f"\n(saved to {CONTOUR_SUMMARY_PATH.relative_to(REPO_ROOT)})")
        results["1. opencv contours"] = True
    except Exception as exc:
        print(f"Step 1 (opencv) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Step 2: Claude reasoning -> SketchPartDescription -------
    _print_header("Step 2 — Claude (sketch architecture)")
    try:
        part = call_claude_sketch(str(image_path), mesh_bounds_mm=bounds_mm)
        _print_sketch_summary(part)
        warn_if_uncertain(part, "step2")
        results["2. claude sketch"] = True
        final_part = part
    except Exception as exc:
        print(f"Step 2 (claude) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Steps 3-5: build, execute, render -------
    try:
        build_and_render(part)
        results["3. build code"] = True
        results["4. execute cadquery"] = True
        results["5. render reconstruction"] = True
        render_ok = True
    except SystemExit:
        results["3. build code"] = True
        results["4. execute cadquery"] = False
        results["5. render reconstruction"] = False
        step_summary(results, final_part)
        return 1

    # ------- Step 6: comparison -------
    compare_ok = step_compare(image_path, render_ok)
    results["6. compare"] = compare_ok

    # ------- Step 7: summary -------
    step_summary(results, final_part)

    if args.open_artifacts:
        open_artifacts(image_path)

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
