#!/usr/bin/env python3
"""End-to-end round-trip: 6-view grid PNG -> Claude -> CadQuery -> STL -> grid -> diff.

Steps, in order:

    1. Claude Vision -> PartDescription
    2. PartDescription -> CadQuery code string
    3. exec the code in a subprocess (15 s timeout) -> STL + STEP
    4. Re-render the produced STL with the existing renderer -> 6-view grid
    4b. (--refine) If confidence is not "high", send the original AND the
        reconstruction back to Claude for a correction pass, then redo
        steps 2-4 on the corrected description.
    5. Side-by-side comparison PNG (original vs reconstructed)
    6. Summary

Outputs are written to ``backend/outputs/`` (the project's existing
artifact dir, which is gitignored) rather than the spec's plain
``outputs/`` so they live alongside the rest of the pipeline output.

Flags:
    --refine   After the first pass, if Claude reports anything other
               than "high" confidence, do one refinement round-trip
               using the rendered reconstruction as visual feedback.
    --strict   Exit non-zero if the FINAL confidence is not "high".
    --open     After the run, ``open`` the comparison PNG and launch
               the interactive ``view3d.py`` viewer for both the
               reconstructed STL and (if found) the original STL.
               The original STL is auto-discovered by stripping
               ``_grid`` from the input image stem and searching
               under ``backend/outputs/`` (deepcad samples) and
               ``backend/sample_data/``.

Usage:
    python scripts/roundtrip_demo.py path/to/grid.png [--refine] [--strict] [--open]
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.cadquery_builder import build_cadquery  # noqa: E402
from backend.ai_infra.llm_client import (  # noqa: E402
    call_claude,
    call_claude_refine,
    warn_if_uncertain,
)
from backend.ai_infra.models import PartDescription  # noqa: E402
from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
STL_PATH = OUTPUT_DIR / "roundtrip.stl"
STEP_PATH = OUTPUT_DIR / "roundtrip.step"
GRID_PATH = OUTPUT_DIR / "roundtrip_grid.png"
COMPARISON_PATH = OUTPUT_DIR / "comparison.png"
GENERATED_CODE_PATH = OUTPUT_DIR / "roundtrip_generated.py"

# First-pass artifacts kept around when refinement runs, so you can see
# what the model corrected. Overwritten each --refine run.
PASS1_STL_PATH = OUTPUT_DIR / "roundtrip_pass1.stl"
PASS1_GRID_PATH = OUTPUT_DIR / "roundtrip_pass1_grid.png"
PASS1_CODE_PATH = OUTPUT_DIR / "roundtrip_pass1_generated.py"

EXEC_TIMEOUT_SECONDS = 15

LABEL_BAR_HEIGHT = 30
DIVIDER_WIDTH = 4
DIVIDER_COLOR = (0x33, 0x33, 0x33)
LABEL_BG = (0x1A, 0x1A, 0x1A)
LABEL_FG = (0xFF, 0xFF, 0xFF)


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


def _print_part_summary(part: PartDescription) -> None:
    print(part.model_dump_json(indent=2))
    print()
    print(f"Features ({len(part.features)}):")
    for i, feat in enumerate(part.features, start=1):
        if feat.shape == "circle":
            dims = f"diameter={feat.diameter_mm}mm"
        else:
            dims = f"width={feat.width_mm}mm depth={feat.depth_mm}mm"
        print(
            f"  [{i}] {feat.type} on {feat.face} — {feat.shape}, {dims}, "
            f"height={feat.height_mm}mm, depth_type={feat.depth_type}, "
            f"position=({feat.position_x}, {feat.position_y})"
        )
    print(f"Confidence: {part.confidence}"
          + (f"  notes={part.notes!r}" if part.notes else ""))


# ---------------------------------------------------------------------------
# Reusable build / execute / render helper (steps 2-4)
# ---------------------------------------------------------------------------


def build_and_render(
    part: PartDescription,
    *,
    stl_path: Path,
    step_path: Path,
    grid_path: Path,
    code_path: Path,
    label: str,
) -> tuple[str, bool]:
    """Run code-gen + subprocess execution + re-render for ``part``.

    Returns ``(generated_code, render_ok)``. Raises ``SystemExit`` if
    code execution fails (which the caller can let propagate or trap).
    """
    _print_header(f"{label} — Build CadQuery code")
    code = build_cadquery(part)
    print(code)
    code_path.write_text(code)
    print(f"(saved to {code_path.relative_to(REPO_ROOT)})")

    _print_header(f"{label} — Execute CadQuery in subprocess")
    wrapper = (
        code
        + "\n"
        + f"result.val().exportStep({str(step_path)!r})\n"
        + f"cq.exporters.export(result, {str(stl_path)!r})\n"
    )

    if stl_path.exists():
        stl_path.unlink()
    if step_path.exists():
        step_path.unlink()

    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        timeout=EXEC_TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if proc.returncode != 0 or not stl_path.exists():
        print(f"{label} — CadQuery execution FAILED.", file=sys.stderr)
        if proc.stdout:
            print("--- stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)

    print(
        f"STL written: {stl_path.relative_to(REPO_ROOT)} "
        f"({stl_path.stat().st_size} bytes)"
    )
    print(
        f"STEP written: {step_path.relative_to(REPO_ROOT)} "
        f"({step_path.stat().st_size} bytes)"
    )

    _print_header(f"{label} — Render reconstructed STL")
    try:
        render_stl_to_grid(stl_path, grid_path, part_id=stl_path.stem)
        print(f"Rendered grid: {grid_path.relative_to(REPO_ROOT)}")
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


def step5_compare(original_path: Path, render_ok: bool) -> bool:
    _print_header("Step 5 — Side-by-side comparison")
    if not render_ok or not GRID_PATH.exists():
        print("Skipping comparison: reconstructed grid is missing.")
        return False

    left = Image.open(original_path).convert("RGB")
    right = Image.open(GRID_PATH).convert("RGB")
    target_h = min(left.height, right.height)
    left = _resize_to_h(left, target_h)
    right = _resize_to_h(right, target_h)

    label_left = f"ORIGINAL — {original_path.name}"
    label_right = "RECONSTRUCTED — roundtrip.stl"

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


def _find_original_stl(image_path: Path) -> Path | None:
    """Try to locate the source STL that produced ``image_path``.

    The renderer names grid PNGs ``<stem>_grid.png``. We strip the
    ``_grid`` suffix and look for ``<stem>.stl`` under the usual
    artifact directories. Returns ``None`` if no candidate exists,
    which is fine — the caller just skips opening it.
    """
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
    """Open the comparison PNG and launch view3d for the two STLs.

    macOS ``open`` is used for the PNG so it lands in Preview. The
    interactive viewers are launched as detached background processes
    via ``subprocess.Popen`` so they pop up simultaneously without
    blocking the script.
    """
    _print_header("Open artifacts")

    if COMPARISON_PATH.exists() and sys.platform == "darwin":
        print(f"Opening comparison PNG: {COMPARISON_PATH.relative_to(REPO_ROOT)}")
        subprocess.Popen(
            ["open", str(COMPARISON_PATH)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    elif COMPARISON_PATH.exists():
        print(
            f"Comparison PNG ready: {COMPARISON_PATH.relative_to(REPO_ROOT)} "
            "(auto-open is macOS-only; open it manually)"
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
    else:
        print(
            "Original STL not auto-found "
            "(searched backend/outputs/deepcad_selected_stl, backend/outputs, backend/sample_data)."
        )

    if STL_PATH.exists():
        print(f"Launching view3d on RECONSTRUCTION: {STL_PATH.relative_to(REPO_ROOT)}")
        subprocess.Popen(
            [sys.executable, str(view3d_script), str(STL_PATH)],
            env=env, **popen_kwargs,
        )
    else:
        print("Reconstruction STL is missing; skipping its viewer.")


def step6_summary(
    results: dict[str, bool], final_part: PartDescription | None,
) -> None:
    _print_header("Step 6 — Summary")
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
        GENERATED_CODE_PATH,
        PASS1_STL_PATH, PASS1_GRID_PATH, PASS1_CODE_PATH,
    ):
        if path.exists():
            print(f"  {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip a 6-view grid PNG through Claude and CadQuery.",
    )
    parser.add_argument("image", help="Path to the input 6-view grid PNG.")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="If first-pass confidence is not 'high', do a second pass "
             "feeding both the original and the reconstruction back to "
             "Claude for correction.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the FINAL confidence is not 'high'.",
    )
    parser.add_argument(
        "--open",
        dest="open_artifacts",
        action="store_true",
        help="After the run, open comparison.png and launch view3d for "
             "the original STL (auto-discovered) and the reconstruction.",
    )
    args = parser.parse_args(argv[1:])

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    final_part: PartDescription | None = None

    # ------- Step 1: vision (initial pass) -------
    _print_header("Step 1 — Claude Vision (initial pass)")
    try:
        part = call_claude(str(image_path))
        _print_part_summary(part)
        warn_if_uncertain(part, "step1")
        results["1. vision"] = True
    except Exception as exc:
        print(f"Step 1 (vision) FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------- Steps 2-4: build / execute / render -------
    try:
        _, render_ok = build_and_render(
            part,
            stl_path=STL_PATH, step_path=STEP_PATH,
            grid_path=GRID_PATH, code_path=GENERATED_CODE_PATH,
            label="Steps 2-4",
        )
        results["2. build code"] = True
        results["3. execute cadquery"] = True
        results["4. render reconstruction"] = render_ok
    except SystemExit:
        results["2. build code"] = True
        results["3. execute cadquery"] = False
        results["4. render reconstruction"] = False
        step6_summary(results, part)
        return 1

    final_part = part

    # ------- Step 4b: optional refinement -------
    if args.refine:
        if part.confidence == "high":
            _print_header("Step 4b — Refinement skipped (confidence already high)")
        elif not render_ok:
            _print_header("Step 4b — Refinement skipped (no reconstruction grid to compare)")
        else:
            # Stash the first-pass artifacts before they get overwritten.
            STL_PATH.replace(PASS1_STL_PATH)
            GRID_PATH.replace(PASS1_GRID_PATH)
            GENERATED_CODE_PATH.replace(PASS1_CODE_PATH)

            _print_header(
                f"Step 4b — Refinement pass (initial confidence={part.confidence})"
            )
            try:
                refined = call_claude_refine(
                    original_image_path=str(image_path),
                    reconstruction_image_path=str(PASS1_GRID_PATH),
                    current=part,
                )
            except Exception as exc:
                print(f"Refinement FAILED (keeping first pass): {exc}", file=sys.stderr)
                traceback.print_exc()
                # Restore first-pass outputs so the comparison still has data.
                PASS1_STL_PATH.replace(STL_PATH)
                PASS1_GRID_PATH.replace(GRID_PATH)
                PASS1_CODE_PATH.replace(GENERATED_CODE_PATH)
                results["4b. refine"] = False
            else:
                _print_header("Step 4b — Refined PartDescription")
                _print_part_summary(refined)
                warn_if_uncertain(refined, "step4b")
                results["4b. refine"] = True

                # Rebuild + re-execute + re-render with the refined part.
                try:
                    _, render_ok = build_and_render(
                        refined,
                        stl_path=STL_PATH, step_path=STEP_PATH,
                        grid_path=GRID_PATH, code_path=GENERATED_CODE_PATH,
                        label="Steps 2-4 (refined)",
                    )
                    results["4c. rebuild+render (refined)"] = render_ok
                    final_part = refined
                except SystemExit:
                    print(
                        "Refined code failed to build; restoring first-pass artifacts.",
                        file=sys.stderr,
                    )
                    PASS1_STL_PATH.replace(STL_PATH)
                    PASS1_GRID_PATH.replace(GRID_PATH)
                    PASS1_CODE_PATH.replace(GENERATED_CODE_PATH)
                    results["4c. rebuild+render (refined)"] = False
                    final_part = part

    # ------- Step 5: comparison -------
    compare_ok = step5_compare(image_path, render_ok)
    results["5. compare"] = compare_ok

    # ------- Step 6: summary -------
    step6_summary(results, final_part)

    # ------- Optional: open comparison + viewers -------
    if args.open_artifacts:
        open_artifacts(image_path)

    if args.strict and final_part is not None and final_part.confidence != "high":
        print(
            f"--strict: final confidence is {final_part.confidence!r}, exiting non-zero.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
