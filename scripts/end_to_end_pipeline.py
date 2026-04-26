#!/usr/bin/env python3
"""End-to-end CADabra pipeline: synthetic point cloud -> CadQuery script.

Glues the two halves of the project together for a single sample id:

    HALF 1 (point cloud -> clean orthographic views)
      A. reconstruct_meshes_from_pointclouds.py  — PC .ply -> recon STL
      B. synthesize_clean_views.py               — recon STL -> 6-view ortho PNG
                                                    + gpt-image-2 cleanup pass

    -- HUMAN-IN-THE-LOOP HEAL CHECKPOINT (future feature, see below) --

    HALF 2 (STL -> face geometry -> CadQuery)
      C. face_roundtrip.py                        — recon STL -> face diagram
                                                    -> Claude -> SketchPartDescription
                                                    -> CadQuery code -> exec -> STL/STEP

Why this script exists
======================
Half 1 was authored on the point-cloud branch and produces:
    backend/outputs/deepcad_pc_recon_stl/<sample_id>_recon_noisy.stl
    backend/outputs/deepcad_pc_recon_stl/geometry_views/<sample_id>_geometry_clean.png

Half 2 was authored on the face-geometry-approach branch and consumes:
    a single .stl path (positional CLI arg of face_roundtrip.py)

The two halves were merged on `main` but never wired to run as one flow.
The recon STL drops straight into face_roundtrip's input slot, so the
glue here is mostly orchestration + clearly marking the seams that still
need design work.

RESOLVED LOGIC GAPS — the seams that matter
===========================================
1. Cleaned 6-view PNG is currently consumed by NO downstream step.
   face_roundtrip reads the STL directly via Open3D (face_extractor) and
   never looks at synthesize_clean_views' output. The cleaned PNG is an
   eyeball-only artifact today. See HUMAN_HEAL section below — that's the
   intended future home for the cleaned views as a 3D-heal driver.

2. Two scale conventions live in the codebase.
   - synthesize_clean_views renders at the recon mesh's raw mm extents.
   - face_extractor normalises to NORMALISE_LONGEST_MM = 100mm before
     emitting ExtractedGeometry.
   Single-direction flow (recon -> faces) is fine because each side
   normalises internally, but any code that wants to overlay the cleaned
   ortho views ON TOP of the face diagram must reconcile this.

3. No shared manifest across halves.
   Half 1 writes backend/outputs/deepcad_pc_recon_stl/manifest.json (per
   sample: recon STL, raw geometry view, clean geometry view, cohesion
   metrics). Half 2 writes loose backend/outputs/face_<stem>_*.{png,stl,
   step,py,txt,json} files with no central registry. This driver reports
   both sets at the end, but a future cleanup should give half 2 its own
   manifest entries keyed by source sample_id (right now part_id ends up
   as e.g. `deepcadimg_000035_recon_noisy` because it inherits the STL
   stem, which is noisy but functional).

Usage
=====
    # Default: run all four canonical demo ids
    python scripts/end_to_end_pipeline.py

    # One sample (recon + clean PNG must already exist on disk)
    python scripts/end_to_end_pipeline.py --sample-id deepcadimg_000035

    # Skip the gpt-image-2 cleanup (cheap dry-run / no API spend)
    python scripts/end_to_end_pipeline.py --sample-id deepcadimg_000035 --no-clean-views

    # Skip the back half (just produce recon + clean views, no Claude/CadQuery)
    python scripts/end_to_end_pipeline.py --sample-id deepcadimg_000035 --no-face

Required artifacts
==================
This driver does NOT regenerate recon STLs from scratch. The user-half
batch is expensive (multi-strategy reconstruction + optional vision tuner),
so we expect it to have been run already via:
    python scripts/vision_tune_recon.py --max-iters 2 --synth-quality high
which leaves the recon STLs and the cohesion manifest in
backend/outputs/deepcad_pc_recon_stl/. If a recon STL is missing this
script prints how to regenerate it and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RECON_OUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_pc_recon_stl"
RECON_MANIFEST = RECON_OUT_DIR / "manifest.json"
GEOMETRY_VIEWS_DIR = RECON_OUT_DIR / "geometry_views"
FACE_OUT_DIR = REPO_ROOT / "backend" / "outputs"

SYNTH_CLEAN_SCRIPT = REPO_ROOT / "scripts" / "synthesize_clean_views.py"
FACE_ROUNDTRIP_SCRIPT = REPO_ROOT / "scripts" / "face_roundtrip.py"

# Mirrors synthesize_clean_views.DEFAULT_TEST_IDS so the merged flow stays
# bounded to the curated demo set unless the caller explicitly overrides.
DEFAULT_SAMPLE_IDS = [
    "deepcadimg_000035",
    "deepcadimg_002354",
    "deepcadimg_117514",
    "deepcadimg_128105",
]


@dataclass
class SampleArtifacts:
    sample_id: str
    recon_stl: Path
    raw_geometry_view: Path | None
    clean_geometry_view: Path | None
    face_stem: str

    @property
    def face_diagram(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}_diagram.png"

    @property
    def face_recon_grid(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}_recon_grid.png"

    @property
    def face_comparison(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}_comparison.png"

    @property
    def face_generated_code(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}_generated.py"

    @property
    def face_step(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}.step"

    @property
    def face_stl(self) -> Path:
        return FACE_OUT_DIR / f"face_{self.face_stem}.stl"


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------
def _print_banner(text: str) -> None:
    print()
    print("#" * 78)
    print(f"# {text}")
    print("#" * 78)


def locate(sample_id: str) -> SampleArtifacts | None:
    """Resolve the recon STL + view PNGs for a sample id from the user-half
    manifest. Returns None (and prints why) if the user half hasn't been run.
    """
    if not RECON_MANIFEST.exists():
        print(
            f"ERROR: {RECON_MANIFEST.relative_to(REPO_ROOT)} is missing.\n"
            f"Run the user-half pipeline first:\n"
            f"  python scripts/vision_tune_recon.py --only {sample_id} "
            f"--max-iters 0 --no-synth-clean",
            file=sys.stderr,
        )
        return None

    manifest = json.loads(RECON_MANIFEST.read_text())
    entry = next((e for e in manifest if e.get("sample_id") == sample_id), None)
    if entry is None:
        print(
            f"ERROR: sample id {sample_id!r} is not in the recon manifest "
            f"({len(manifest)} entries).",
            file=sys.stderr,
        )
        return None
    if not entry.get("success"):
        print(
            f"ERROR: sample id {sample_id!r} did not reconstruct successfully "
            f"in the user-half batch.",
            file=sys.stderr,
        )
        return None

    recon_stl = RECON_OUT_DIR / entry["recon_noisy_stl"]
    if not recon_stl.exists():
        print(
            f"ERROR: manifest references {recon_stl.relative_to(REPO_ROOT)} "
            "but the file is missing on disk.",
            file=sys.stderr,
        )
        return None

    cohesion = entry.get("cohesion") or {}
    raw_view_rel = cohesion.get("geometry_view")
    clean_view_rel = cohesion.get("geometry_view_clean")

    raw_view = (RECON_OUT_DIR / raw_view_rel) if raw_view_rel else None
    clean_view = (RECON_OUT_DIR / clean_view_rel) if clean_view_rel else None

    return SampleArtifacts(
        sample_id=sample_id,
        recon_stl=recon_stl,
        raw_geometry_view=raw_view if raw_view and raw_view.exists() else None,
        clean_geometry_view=clean_view if clean_view and clean_view.exists() else None,
        # face_roundtrip derives part_id = STL stem, so we mirror that here
        # so the post-run paths line up.
        face_stem=recon_stl.stem,
    )


# ---------------------------------------------------------------------------
# Stage B — clean orthographic views
# ---------------------------------------------------------------------------
def stage_b_clean_views(art: SampleArtifacts, force: bool, quality: str) -> bool:
    """Ensure the cleaned 6-view ortho PNG exists for this sample.

    synthesize_clean_views handles the gpt-image-2 call + manifest update;
    we just shell out so the merged driver doesn't duplicate that logic.
    """
    _print_banner(
        f"Stage B — clean orthographic views ({art.sample_id})"
    )
    if art.clean_geometry_view and not force:
        print(
            f"  clean view already on disk: "
            f"{art.clean_geometry_view.relative_to(REPO_ROOT)}"
        )
        print("  (re-run with --force-clean to regenerate via gpt-image-2)")
        return True

    cmd = [
        sys.executable,
        str(SYNTH_CLEAN_SCRIPT),
        "--only", art.sample_id,
        "--quality", quality,
    ]
    if force:
        cmd.append("--force")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"  WARN: synthesize_clean_views exited with code {proc.returncode}")
        return False

    # Re-resolve so we pick up the manifest update written by the subprocess.
    refreshed = locate(art.sample_id)
    if refreshed and refreshed.clean_geometry_view:
        art.clean_geometry_view = refreshed.clean_geometry_view
        return True
    print("  WARN: synth-clean reported success but no clean PNG was registered")
    return False


# ---------------------------------------------------------------------------
# HUMAN-IN-THE-LOOP HEAL CHECKPOINT
# ---------------------------------------------------------------------------
# TODO(human-in-the-loop heal — future feature):
# Once the cleaned 6-view PNG is in hand we have a vision-only opinion
# about what the part SHOULD look like, but the underlying recon STL may
# still carry topology damage that pure 2D cleanup cannot recover. Two
# example failure modes the cleaned PNG cannot fix on its own:
#
#   - A through-bore that the scan fragmented into 6+ tiny boundary loops
#     instead of one clean ring. Vision says "circle here", but the mesh
#     still has the broken topology that breaks face_extractor's
#     cylinder classifier.
#   - A thin internal pocket whose floor was never sampled by the scan.
#     The cleaned silhouette will still show the pocket because the side
#     walls were sampled, but the recon mesh has no floor triangles, so
#     extract_faces can't find the bottom face.
#
# The intended UX:
#   1. Pop the cleaned PNG + raw recon STL into a small reviewer.
#   2. Operator marks loops/pockets/holes that the 2D cleanup got right
#      but that the 3D mesh still mis-represents.
#   3. A heal tool (Open3D fill_holes with operator-supplied hints, or a
#      manual brush in MeshLab/Blender) repairs the mesh in-place.
#   4. The healed STL replaces recon_noisy.stl for the rest of the flow.
#
# Until that ships, this stage is a passthrough and the recon STL flows
# straight into Stage C unmodified. Logging is deliberately verbose so it
# is obvious in the run log that a manual step was skipped.
def stage_human_heal(art: SampleArtifacts, interactive: bool) -> Path:
    _print_banner(
        f"Stage HUMAN-HEAL — manual mesh repair checkpoint ({art.sample_id})"
    )
    print("  [PASSTHROUGH] No 3D heal applied; recon STL flows straight to face stage.")
    print("  RESOLVED LOGIC GAP: the cleaned 6-view PNG is currently NOT consumed")
    print("  downstream. A future feature will let an operator review it here and")
    print("  apply targeted mesh repairs (broken bores, missing pocket floors, etc.)")
    print("  before face_extractor runs. See module docstring for the spec.")
    if art.clean_geometry_view:
        print(f"  clean view ready for review: "
              f"{art.clean_geometry_view.relative_to(REPO_ROOT)}")
    if interactive and sys.platform == "darwin" and art.clean_geometry_view:
        # Best-effort visual handoff. Non-fatal if it fails — the run continues.
        try:
            subprocess.Popen(
                ["open", str(art.clean_geometry_view)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("  (opened cleaned PNG in Preview for eyeball review)")
        except Exception:
            pass
    return art.recon_stl


# ---------------------------------------------------------------------------
# Stage C — face-geometry round-trip
# ---------------------------------------------------------------------------
def stage_c_face_roundtrip(art: SampleArtifacts, healed_stl: Path) -> bool:
    """Hand the (possibly healed) recon STL to face_roundtrip.py, which
    runs the full face-extract -> Claude -> CadQuery -> re-render flow.

    When the cleaned 6-view PNG exists for this sample we forward it via
    --clean-view; face_roundtrip then recolours it (RdYlBu_r depth) and
    attaches it as a SECOND image to the Claude call. That closes the
    "cleaned PNG goes nowhere" gap noted in the module docstring.

    RESOLVED LOGIC GAP: face_roundtrip's part_id = STL stem, so artifacts
    end up at backend/outputs/face_<sample_id>_recon_noisy_*. That's noisy
    but unambiguous; a future tweak can add a --part-id override to
    face_roundtrip so the back-half artifacts can be keyed by the cleaner
    sample_id directly.
    """
    _print_banner(
        f"Stage C — face geometry -> CadQuery ({art.sample_id})"
    )
    cmd = [sys.executable, str(FACE_ROUNDTRIP_SCRIPT), str(healed_stl)]
    if art.clean_geometry_view:
        cmd.extend(["--clean-view", str(art.clean_geometry_view)])
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"  WARN: face_roundtrip exited with code {proc.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_final_summary(samples: list[tuple[SampleArtifacts, dict[str, bool]]]) -> None:
    _print_banner("End-to-end summary")
    for art, results in samples:
        ok = "OK  " if all(results.values()) else "PART"
        print(f"  [{ok}] {art.sample_id}")
        for stage, passed in results.items():
            marker = "OK " if passed else "FAIL"
            print(f"        [{marker}] {stage}")
        print(f"        artifacts:")
        # Half 1
        print(f"          recon STL          {_rel(art.recon_stl)}")
        if art.raw_geometry_view:
            print(f"          raw 6-view PNG     {_rel(art.raw_geometry_view)}")
        if art.clean_geometry_view:
            print(f"          clean 6-view PNG   {_rel(art.clean_geometry_view)}")
        # Half 2
        for label, path in (
            ("face diagram      ", art.face_diagram),
            ("face recon grid   ", art.face_recon_grid),
            ("face comparison   ", art.face_comparison),
            ("CadQuery code     ", art.face_generated_code),
            ("CadQuery STL      ", art.face_stl),
            ("CadQuery STEP     ", art.face_step),
        ):
            if path.exists():
                print(f"          {label} {_rel(path)}")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sample-id",
        action="append",
        help="sample id (repeatable). Defaults to the four curated demo ids.",
    )
    p.add_argument(
        "--no-clean-views",
        action="store_true",
        help="skip Stage B (gpt-image-2 cleanup). Use for cheap dry-runs.",
    )
    p.add_argument(
        "--force-clean",
        action="store_true",
        help="re-call gpt-image-2 even if a clean PNG already exists.",
    )
    p.add_argument(
        "--clean-quality", default="high",
        choices=["low", "medium", "high", "auto", "standard"],
        help="quality tier passed to synthesize_clean_views (default: high).",
    )
    p.add_argument(
        "--no-face",
        action="store_true",
        help="skip Stage C (face-geometry / Claude / CadQuery). Half-1 only.",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="do not pop the cleaned PNG in Preview during the heal checkpoint.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sample_ids = args.sample_id or list(DEFAULT_SAMPLE_IDS)

    runs: list[tuple[SampleArtifacts, dict[str, bool]]] = []
    overall_ok = True

    for sid in sample_ids:
        _print_banner(f"=== {sid} ===")
        art = locate(sid)
        if art is None:
            overall_ok = False
            continue

        results: dict[str, bool] = {}

        # Stage B — cleaned ortho views
        if args.no_clean_views:
            print("  Stage B skipped (--no-clean-views)")
            results["B. clean views"] = bool(art.clean_geometry_view)
        else:
            results["B. clean views"] = stage_b_clean_views(
                art, force=args.force_clean, quality=args.clean_quality,
            )

        # HUMAN HEAL — passthrough today
        healed_stl = stage_human_heal(art, interactive=not args.no_open)
        results["heal (passthrough)"] = True

        # Stage C — face geometry -> CadQuery
        if args.no_face:
            print("  Stage C skipped (--no-face)")
        else:
            results["C. face -> cadquery"] = stage_c_face_roundtrip(art, healed_stl)
            if not results["C. face -> cadquery"]:
                overall_ok = False

        runs.append((art, results))

    print_final_summary(runs)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
