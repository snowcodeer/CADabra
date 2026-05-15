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
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.ortho_feature_inferencer import (  # noqa: E402
    _drop_unpaired_silhouette_holes,
    _world_frame,
    infer_sketches,
)
from backend.ai_infra.ortho_view_segmenter import (  # noqa: E402
    render_debug_overlay,
    segment_ortho_png,
)
from backend.ai_infra.sketch_builder import build_from_sketches  # noqa: E402
from backend.ai_infra.step_off_classifier import (  # noqa: E402
    pick_axis_from_step_offs,
)
from backend.ai_infra.step_offs import (  # noqa: E402
    extract_step_offs,
    step_off_to_dict,
)
from backend.pipeline.stl_renderer import render_stl_to_grid  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
RECON_OUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_pc_recon_stl"
RECON_MANIFEST = RECON_OUT_DIR / "manifest.json"
EXEC_TIMEOUT_SECONDS = 15
# Below this average silhouette IoU we treat the deterministic CV rebuild
# as untrustworthy and fall back to the Claude vision path. 0.85 picks up
# clear shape mismatches (off-centre boss carved-out features, ~10mm
# bbox misalignment) while leaving small triangulation jitter alone.
CONFIDENCE_THRESHOLD = 0.85


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


def _resolve_input_noisy_stl(sample_id: str) -> Path | None:
    """Locate the noisy reconstruction STL that feeds this sample.

    Returned for two purposes downstream: (a) computing silhouette IoU
    against the rebuild and (b) extracting world-mm bounds for the
    vision-fallback contour-coordinate scale.
    """
    candidates = [
        RECON_OUT_DIR / f"{sample_id}_recon_noisy.stl",
        REPO_ROOT / "frontend" / "deepcad_pcrecon_stl" / f"{sample_id}_recon_noisy.stl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _silhouette_mask_from_view(mesh: "pv.PolyData", view: str,  # type: ignore[name-defined] # noqa: F821
                                common_center: tuple[float, float, float],
                                common_span: float,
                                resolution: int = 400) -> np.ndarray:
    """Render ``mesh`` from one orthographic axis with a SHARED camera
    framing (so two meshes can be compared pixel-for-pixel) and return
    the binary silhouette mask."""
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=[resolution, resolution])
    pl.set_background("white")
    pl.add_mesh(mesh, color="black", lighting=False, ambient=1.0)
    pl.enable_parallel_projection()
    cx, cy, cz = common_center
    d = common_span * 2.5
    if view == "Z":
        cam_pos = (cx, cy, cz + d); view_up = (0, 1, 0)
    elif view == "Y":
        cam_pos = (cx, cy + d, cz); view_up = (0, 0, 1)
    else:  # "X"
        cam_pos = (cx + d, cy, cz); view_up = (0, 0, 1)
    pl.camera_position = [cam_pos, (cx, cy, cz), view_up]
    pl.camera.parallel_scale = common_span / 2.0
    img = np.asarray(pl.screenshot(return_img=True))
    pl.close()
    return img[:, :, 0] < 100


def _normalize_mesh_to_unit_cube(mesh: "pv.PolyData") -> "pv.PolyData":  # type: ignore[name-defined] # noqa: F821
    """Translate the mesh's bbox centre to origin and scale so the
    longest edge fits in a unit cube. Used to bring the input noisy
    STL (~unit cube as rendered) and the rebuild STL (real mm) into
    a common frame before silhouette IoU.
    """
    b = mesh.bounds
    cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) or 1.0
    out = mesh.copy()
    out.translate((-cx, -cy, -cz), inplace=True)
    out.scale((1.0 / span, 1.0 / span, 1.0 / span), inplace=True)
    return out


def _silhouette_iou(stl_a: Path, stl_b: Path, resolution: int = 400) -> tuple[float, dict[str, float]]:
    """Average silhouette IoU across +Z, +X, +Y orthographic views.

    Both meshes are first normalised to a unit cube (centre at origin,
    longest edge = 1) so the comparison ignores absolute scale and
    translation differences between the noisy input (rendered at unit
    scale) and the rebuild (real mm). What remains in the IoU signal:
    aspect ratio, shape outline, and feature placement. Returns
    (avg_iou, per_view).
    """
    import pyvista as pv
    ma = _normalize_mesh_to_unit_cube(pv.read(str(stl_a)))
    mb = _normalize_mesh_to_unit_cube(pv.read(str(stl_b)))
    center = (0.0, 0.0, 0.0)
    # Both meshes now fit in a [-0.5, 0.5] cube; pick a span that gives
    # a bit of padding so silhouettes don't clip at the frame edge.
    span = 1.1
    per_view: dict[str, float] = {}
    for view in ("Z", "X", "Y"):
        sa = _silhouette_mask_from_view(ma, view, center, span, resolution)
        sb = _silhouette_mask_from_view(mb, view, center, span, resolution)
        inter = int(np.logical_and(sa, sb).sum())
        union = int(np.logical_or(sa, sb).sum())
        per_view[view] = float(inter / union) if union > 0 else 0.0
    avg = float(np.mean(list(per_view.values())))
    return avg, per_view


def _build_execute_render(part, paths: "RunPaths", part_id: str, *, label: str) -> bool:
    """Build CadQuery code from a SketchPartDescription, exec it in a
    subprocess, and re-render the rebuild as a 6-view grid. Returns
    True on success. Used twice in the pipeline: once for the CV
    inferencer's part, optionally again for the vision fallback's part.
    """
    _print_header(f"Build / Execute / Render — {label}")
    code = build_from_sketches(part)
    paths.code_py.write_text(code)
    print(code)
    print(f"  wrote {paths.code_py.relative_to(REPO_ROOT)}")

    wrapper = (
        code + "\n"
        + f"result.val().exportStep({str(paths.step)!r})\n"
        # Tight tolerances so circular features tessellate smoothly
        # instead of the default ~10-segment polygon facets.
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
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0 or not paths.stl.exists():
        print(f"  CadQuery execution FAILED ({label}).", file=sys.stderr)
        if proc.stdout:
            print("--- stdout ---", file=sys.stderr); print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("--- stderr ---", file=sys.stderr); print(proc.stderr, file=sys.stderr)
        return False
    print(f"  STL: {paths.stl.relative_to(REPO_ROOT)} ({paths.stl.stat().st_size} bytes)")
    print(f"  STEP: {paths.step.relative_to(REPO_ROOT)} ({paths.step.stat().st_size} bytes)")
    try:
        render_stl_to_grid(paths.stl, paths.recon_grid, part_id=part_id)
        print(f"  wrote {paths.recon_grid.relative_to(REPO_ROOT)}")
    except Exception as exc:
        print(f"  render failed (continuing): {exc}", file=sys.stderr)
    return True


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
    p.add_argument(
        "--dump-step-offs", action="store_true",
        help="after segmentation, extract step-off features (tier pairs + "
             "through-holes) and dump them to "
             "backend/outputs/ortho_<part>_step_offs.json. Audit-only; the "
             "existing infer_sketches path runs unchanged.",
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

    # 1b — optional step-off audit dump (additive; doesn't gate the pipeline).
    if args.dump_step_offs:
        try:
            views_clean = _drop_unpaired_silhouette_holes(feats.views)
            frame = _world_frame(views_clean)
            audit_features = type(feats)(source_png=feats.source_png, views=views_clean)
            step_offs = extract_step_offs(audit_features, frame)
            axis_picked = pick_axis_from_step_offs(step_offs, audit_features)
            external_tallies = {"Z": 0, "Y": 0, "X": 0}
            for s in step_offs:
                if s.kind == "external" and s.axis in external_tallies:
                    external_tallies[s.axis] += 1
            perp_tiers = {
                "Z": getattr(views_clean.get("Top"), "depth_tier_count", 0)
                if views_clean.get("Top") is not None else 0,
                "Y": getattr(views_clean.get("Front"), "depth_tier_count", 0)
                if views_clean.get("Front") is not None else 0,
                "X": getattr(views_clean.get("Right"), "depth_tier_count", 0)
                if views_clean.get("Right") is not None else 0,
            }
            dump_path = OUTPUT_DIR / f"ortho_{part_id}_step_offs.json"
            dump_path.write_text(
                json.dumps(
                    {
                        "sample_id": part_id,
                        "axis_picked": axis_picked,
                        "external_tallies": external_tallies,
                        "step_offs": [step_off_to_dict(s) for s in step_offs],
                    },
                    indent=2,
                )
            )
            print(
                f"[step-offs] {part_id} axis={axis_picked} "
                f"tallies={external_tallies} "
                f"perp_tiers(Top/Front/Right)="
                f"{perp_tiers['Z']}/{perp_tiers['Y']}/{perp_tiers['X']} "
                f"count={len(step_offs)}"
            )
            print(f"  wrote {dump_path.relative_to(REPO_ROOT)}")
        except Exception as exc:
            print(f"step-off dump FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

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

    # 3-5 — build CadQuery code, execute, render rebuild grid.
    if not _build_execute_render(part, paths, part_id, label="CV"):
        return 1

    # 6 — Silhouette IoU between input noisy mesh and the rebuild. This
    # is the confidence signal that decides whether the deterministic CV
    # path produced something faithful to the input geometry.
    _print_header("Step 6 — Silhouette IoU vs input")
    input_stl = _resolve_input_noisy_stl(part_id)
    confidence_cv = 0.0
    iou_breakdown_cv: dict[str, float] = {}
    if input_stl is None:
        print(f"  no input noisy STL for {part_id!r} — skipping confidence check")
    else:
        try:
            confidence_cv, iou_breakdown_cv = _silhouette_iou(input_stl, paths.stl)
            print(
                f"  deterministic CV  avg IoU = {confidence_cv:.3f}  "
                f"per-view: Z={iou_breakdown_cv['Z']:.3f} "
                f"X={iou_breakdown_cv['X']:.3f} Y={iou_breakdown_cv['Y']:.3f}"
            )
        except Exception as exc:
            print(f"  silhouette IoU failed: {exc}", file=sys.stderr)

    # 7 — Vision fallback when the CV rebuild's silhouette diverges
    # materially from the input. The fallback re-classifies the shape
    # using Claude vision (sketch_llm_client.call_claude_sketch), then
    # rebuilds via the same _build_execute_render path. Only adopt the
    # vision result if it ACTUALLY scores higher; otherwise keep CV.
    used_vision = False
    if (
        input_stl is not None
        and 0.0 < confidence_cv < CONFIDENCE_THRESHOLD
    ):
        _print_header(
            f"Step 7 — Confidence {confidence_cv:.3f} < "
            f"{CONFIDENCE_THRESHOLD} → Claude vision fallback"
        )
        # Backup CV attempt so we can restore it if vision is worse.
        cv_backup_stl = OUTPUT_DIR / f"ortho_{part_id}_cv_attempt.stl"
        cv_backup_step = OUTPUT_DIR / f"ortho_{part_id}_cv_attempt.step"
        cv_backup_grid = OUTPUT_DIR / f"ortho_{part_id}_cv_attempt_recon_grid.png"
        cv_backup_code = OUTPUT_DIR / f"ortho_{part_id}_cv_attempt_generated.py"
        cv_backup_sketch = OUTPUT_DIR / f"ortho_{part_id}_cv_attempt_sketches.json"
        shutil.copy(paths.stl, cv_backup_stl)
        shutil.copy(paths.step, cv_backup_step)
        if paths.recon_grid.exists():
            shutil.copy(paths.recon_grid, cv_backup_grid)
        shutil.copy(paths.code_py, cv_backup_code)
        shutil.copy(paths.sketch_json, cv_backup_sketch)

        try:
            from backend.ai_infra.sketch_llm_client import call_claude_sketch
            import pyvista as pv
            input_bounds = tuple(pv.read(str(input_stl)).bounds)
            # The vision pipeline's contour_extractor.extract_all_views
            # only accepts the 2400x1000 grid layout from
            # backend.pipeline.stl_renderer, NOT the 1536x1024 cleaned
            # PNG that the deterministic CV path reads. Render the
            # input noisy STL through that renderer first so the vision
            # client sees the format it expects.
            input_grid = OUTPUT_DIR / f"ortho_{part_id}_input_grid.png"
            if not input_grid.exists():
                print(f"  rendering input grid {input_grid.relative_to(REPO_ROOT)}...")
                render_stl_to_grid(input_stl, input_grid, part_id=part_id)
            print(f"  calling Claude vision on {input_grid.relative_to(REPO_ROOT)}...")
            sys.stdout.flush()
            vision_part = call_claude_sketch(
                input_grid, mesh_bounds_mm=input_bounds  # type: ignore[arg-type]
            )
            paths.sketch_json.write_text(vision_part.model_dump_json(indent=2))
            ok = _build_execute_render(
                vision_part, paths, part_id, label="Vision",
            )
            if ok:
                # Save the vision attempt to a stable audit path BEFORE
                # scoring + maybe-restoring, so the discarded attempt is
                # still inspectable from disk afterwards.
                vision_attempt_stl = OUTPUT_DIR / f"ortho_{part_id}_vision_attempt.stl"
                vision_attempt_step = OUTPUT_DIR / f"ortho_{part_id}_vision_attempt.step"
                vision_attempt_grid = OUTPUT_DIR / f"ortho_{part_id}_vision_attempt_recon_grid.png"
                vision_attempt_code = OUTPUT_DIR / f"ortho_{part_id}_vision_attempt_generated.py"
                vision_attempt_sketch = OUTPUT_DIR / f"ortho_{part_id}_vision_attempt_sketches.json"
                shutil.copy(paths.stl, vision_attempt_stl)
                shutil.copy(paths.step, vision_attempt_step)
                if paths.recon_grid.exists():
                    shutil.copy(paths.recon_grid, vision_attempt_grid)
                shutil.copy(paths.code_py, vision_attempt_code)
                shutil.copy(paths.sketch_json, vision_attempt_sketch)
                confidence_vision, iou_breakdown_vision = _silhouette_iou(
                    input_stl, paths.stl,
                )
                print(
                    f"  vision           avg IoU = {confidence_vision:.3f}  "
                    f"per-view: Z={iou_breakdown_vision['Z']:.3f} "
                    f"X={iou_breakdown_vision['X']:.3f} "
                    f"Y={iou_breakdown_vision['Y']:.3f}"
                )
                if confidence_vision > confidence_cv:
                    used_vision = True
                    print(
                        f"  → using vision result (better by "
                        f"{confidence_vision - confidence_cv:.3f})"
                    )
                else:
                    print(
                        f"  → vision NOT better — restoring CV attempt"
                    )
                    shutil.copy(cv_backup_stl, paths.stl)
                    shutil.copy(cv_backup_step, paths.step)
                    if cv_backup_grid.exists():
                        shutil.copy(cv_backup_grid, paths.recon_grid)
                    shutil.copy(cv_backup_code, paths.code_py)
                    shutil.copy(cv_backup_sketch, paths.sketch_json)
            else:
                print(f"  vision rebuild FAILED — restoring CV attempt")
                shutil.copy(cv_backup_stl, paths.stl)
                shutil.copy(cv_backup_step, paths.step)
                if cv_backup_grid.exists():
                    shutil.copy(cv_backup_grid, paths.recon_grid)
                shutil.copy(cv_backup_code, paths.code_py)
                shutil.copy(cv_backup_sketch, paths.sketch_json)
        except Exception as exc:
            sys.stdout.flush()
            print(f"  vision fallback raised: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.stderr.flush()
            print(f"  restoring CV attempt")
            shutil.copy(cv_backup_stl, paths.stl)
            shutil.copy(cv_backup_step, paths.step)
            if cv_backup_grid.exists():
                shutil.copy(cv_backup_grid, paths.recon_grid)
            shutil.copy(cv_backup_code, paths.code_py)
            shutil.copy(cv_backup_sketch, paths.sketch_json)

    print()
    print("artifacts:")
    for p in paths.all():
        if p.exists():
            print(f"  {p.relative_to(REPO_ROOT)}")
    if used_vision:
        print("  [final result from VISION fallback]")
    elif confidence_cv > 0:
        print(f"  [final result from CV (IoU {confidence_cv:.3f})]")

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
