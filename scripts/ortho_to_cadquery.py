#!/usr/bin/env python3
"""Deterministic ortho-view → CadQuery roundtrip (no LLM in the loop).

Pipeline:
    1. Resolve the cleaned 6-view PNG for a sample id.
    2. Run ``ortho_view_segmenter.segment_ortho_png`` (cv2 contours +
       depth peak finding) on the PNG.
    3. Run ``ortho_feature_inferencer.infer_sketches`` to turn the
       per-view CV features into a SketchPartDescription deterministically.
    4. Build the CadQuery code via the existing
       ``sketch_builder.build_from_sketches`` helper.
    5. Execute the code in a 15-second subprocess to produce STL + STEP.
    6. Re-render the rebuild as a 6-view grid for visual comparison.

All artifacts use the prefix ``backend/outputs/ortho_<sample>_*`` so the
deterministic-CV path's outputs never collide with the older
``face_<sample>_*`` LLM path's outputs.

Usage:
    python scripts/ortho_to_cadquery.py --sample-id deepcadimg_000035
    python scripts/ortho_to_cadquery.py --png path/to/clean.png --part-id mything
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.ortho_feature_inferencer import infer_sketches  # noqa: E402
from backend.ai_infra.ortho_view_segmenter import (  # noqa: E402
    render_debug_overlay,
    segment_ortho_png,
)
from backend.ai_infra.sketch_builder import build_from_sketches  # noqa: E402
from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
RECON_OUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_pc_recon_stl"
RECON_MANIFEST = RECON_OUT_DIR / "manifest.json"
EXEC_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class RunPaths:
    part_id: str
    features_json: Path
    overlay_png: Path
    code_py: Path
    stl: Path
    step: Path
    recon_grid: Path
    sketch_json: Path

    @classmethod
    def for_part_id(cls, part_id: str) -> "RunPaths":
        prefix = OUTPUT_DIR / f"ortho_{part_id}"
        return cls(
            part_id=part_id,
            features_json=Path(f"{prefix}_features.json"),
            overlay_png=Path(f"{prefix}_overlay.png"),
            code_py=Path(f"{prefix}_generated.py"),
            stl=prefix.with_suffix(".stl"),
            step=prefix.with_suffix(".step"),
            recon_grid=Path(f"{prefix}_recon_grid.png"),
            sketch_json=Path(f"{prefix}_sketches.json"),
        )

    def all(self) -> tuple[Path, ...]:
        return (
            self.features_json, self.overlay_png, self.code_py,
            self.stl, self.step, self.recon_grid, self.sketch_json,
        )


def _resolve_clean_png(sample_id: str, *, skip_cleanup: bool = False) -> Path:
    if skip_cleanup:
        # Read the 1536x1024 segmenter-compatible canvas BEFORE gpt-image-2 ran.
        # Same layout/format as the cleaned output; just unedited. Use this when
        # the cleanup step is doing more harm than good (e.g. over-smoothing
        # octagons into circles at low scan-noise levels).
        p = RECON_OUT_DIR / "clean_view_inputs" / f"{sample_id}_clean_input.png"
        if not p.exists():
            raise SystemExit(
                f"pre-cleanup canvas missing: {p}. Run "
                "scripts/synthesize_clean_views.py (it writes the input even "
                "before calling gpt-image-2)."
            )
        return p
    if not RECON_MANIFEST.exists():
        raise SystemExit(f"manifest missing: {RECON_MANIFEST.relative_to(REPO_ROOT)}")
    manifest = json.loads(RECON_MANIFEST.read_text())
    entry = next((e for e in manifest if e.get("sample_id") == sample_id), None)
    if entry is None:
        raise SystemExit(f"sample id not in manifest: {sample_id!r}")
    rel = (entry.get("cohesion") or {}).get("geometry_view_clean")
    if not rel:
        raise SystemExit(
            f"sample {sample_id!r} has no geometry_view_clean — run "
            "scripts/synthesize_clean_views.py first."
        )
    p = RECON_OUT_DIR / rel
    if not p.exists():
        raise SystemExit(f"clean PNG missing on disk: {p}")
    return p


def _print_header(label: str) -> None:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-id", help="resolve clean PNG via the recon manifest")
    g.add_argument("--png", help="direct path to a cleaned 6-view PNG")
    p.add_argument(
        "--part-id",
        help="override the part_id used for output filenames. "
             "Defaults to --sample-id, or the PNG stem.",
    )
    p.add_argument(
        "--open", action="store_true",
        help="open the overlay + recon grid in Preview (macOS only).",
    )
    p.add_argument(
        "--no-cleanup", action="store_true",
        help="skip the gpt-image-2 cleanup step; CV reads the pre-cleanup "
             "1536x1024 canvas directly. Useful when cleanup over-smooths "
             "(e.g. octagons -> circles) at low scan-noise levels.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_id:
        clean_png = _resolve_clean_png(args.sample_id, skip_cleanup=args.no_cleanup)
        part_id = args.part_id or args.sample_id
    else:
        clean_png = Path(args.png).resolve()
        if not clean_png.exists():
            raise SystemExit(f"PNG not found: {clean_png}")
        part_id = args.part_id or clean_png.stem

    paths = RunPaths.for_part_id(part_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] artifacts -> backend/outputs/ortho_{part_id}_*")

    # 1 — segment
    _print_header("Step 1 — OpenCV segmentation")
    feats = segment_ortho_png(clean_png)
    paths.features_json.write_text(feats.to_json())
    render_debug_overlay(feats, paths.overlay_png)
    print(f"  wrote {paths.features_json.relative_to(REPO_ROOT)}")
    print(f"  wrote {paths.overlay_png.relative_to(REPO_ROOT)}")
    for name, v in feats.views.items():
        circ = ""
        if v.is_circle and v.circle_diameter_px is not None:
            circ = f" CIRCLE(d={v.circle_diameter_px:.0f}px)"
        print(
            f"  {name:6}  verts={len(v.polygon_px):2}  "
            f"flats={len(v.straight_edges):2}  "
            f"parallel_pairs={len(v.parallel_pairs)}  "
            f"holes={len(v.interior_holes)}  "
            f"tiers={v.depth_tier_count}"
            + circ
        )

    # 2 — infer sketches
    _print_header("Step 2 — Cross-view feature inference")
    try:
        part = infer_sketches(feats)
    except Exception as exc:
        print(f"inference FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    paths.sketch_json.write_text(part.model_dump_json(indent=2))
    print(part.model_dump_json(indent=2))
    print(f"  wrote {paths.sketch_json.relative_to(REPO_ROOT)}")
    print()
    print(f"Sketch operations ({len(part.sketches)}):")
    for op in part.sketches:
        prof = op.profile
        if prof.shape == "circle":
            dims = f"d={prof.diameter_mm}mm"
        elif prof.shape == "rectangle":
            dims = f"{prof.width_mm}x{prof.depth_mm}mm"
        else:
            dims = f"polyline({len(prof.vertices or [])} verts)"
        print(
            f"  [{op.order}] {op.operation:<7} on plane={op.plane:<3} "
            f"profile={prof.shape}({dims}) distance={op.distance_mm}mm "
            f"direction={op.direction} pos=({op.position_x}, {op.position_y})"
        )
    print(f"Confidence: {part.confidence}")
    if part.notes:
        print(f"Notes: {part.notes}")

    # 3 — build CadQuery code
    _print_header("Step 3 — Build CadQuery code")
    code = build_from_sketches(part)
    paths.code_py.write_text(code)
    print(code)
    print(f"  wrote {paths.code_py.relative_to(REPO_ROOT)}")

    # 4 — execute
    _print_header("Step 4 — Execute CadQuery in subprocess")
    wrapper = (
        code
        + "\n"
        + f"result.val().exportStep({str(paths.step)!r})\n"
        # Tight tolerances so circular features tessellate smoothly
        # instead of showing the default ~10-segment polygon facets.
        # tolerance = chord-height (mm), angularTolerance = chord-arc (rad).
        + f"cq.exporters.export(result, {str(paths.stl)!r}, "
          "tolerance=0.02, angularTolerance=0.05)\n"
    )
    if paths.stl.exists():
        paths.stl.unlink()
    if paths.step.exists():
        paths.step.unlink()
    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        timeout=EXEC_TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0 or not paths.stl.exists():
        print("CadQuery execution FAILED.", file=sys.stderr)
        if proc.stdout:
            print("--- stdout ---", file=sys.stderr); print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("--- stderr ---", file=sys.stderr); print(proc.stderr, file=sys.stderr)
        return 1
    print(f"  STL: {paths.stl.relative_to(REPO_ROOT)} ({paths.stl.stat().st_size} bytes)")
    print(f"  STEP: {paths.step.relative_to(REPO_ROOT)} ({paths.step.stat().st_size} bytes)")

    # 5 — re-render the rebuild as a 6-view grid for visual compare
    _print_header("Step 5 — Re-render rebuild")
    try:
        render_stl_to_grid(paths.stl, paths.recon_grid, part_id=part_id)
        print(f"  wrote {paths.recon_grid.relative_to(REPO_ROOT)}")
    except Exception as exc:
        print(f"  render failed (continuing): {exc}", file=sys.stderr)

    print()
    print("artifacts:")
    for p in paths.all():
        if p.exists():
            print(f"  {p.relative_to(REPO_ROOT)}")

    if args.open and sys.platform == "darwin":
        for p in (paths.overlay_png, paths.recon_grid):
            if p.exists():
                subprocess.Popen(
                    ["open", str(p)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
