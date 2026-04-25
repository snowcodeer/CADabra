"""Synthesize clean orthographic views with OpenAI gpt-image-2.

Pipeline:
  1. Render a 1536x1024 6-view depth + silhouette grid for each target sample
     (recon-only, no labels, no point cloud overlay). Each view occupies one
     cell of a 3x2 layout. This is what gets sent to gpt-image-2.
  2. Call gpt-image-2 image-edits with a prompt that asks the model to
     polish the views as a CAD engineering blueprint:
       - sharpen silhouette edges
       - fill small surface noise gaps
       - PRESERVE every dark hole / opening (those are real through-holes)
       - keep the same six views in the same 3x2 layout
  3. Save the cleaned PNG as `<sample>_geometry_clean.png` in the geometry
     views directory and copy it into the frontend mirror.

Usage:
  source .venv/bin/activate
  python scripts/synthesize_clean_views.py                                 # all test ids
  python scripts/synthesize_clean_views.py --only deepcadimg_002354        # one sample
  python scripts/synthesize_clean_views.py --no-sync --quality medium      # local only

Environment: OPENAI_API_KEY (loaded from .env).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "backend" / "outputs" / "deepcad_pc_recon_stl"
GEOMETRY_VIEWS_DIR = OUT_DIR / "geometry_views"
CLEAN_VIEWS_SOURCE_DIR = OUT_DIR / "clean_view_inputs"
FRONTEND_GEOMETRY_DIR = REPO / "frontend" / "deepcad_geometry_views"
MANIFEST_PATH = OUT_DIR / "manifest.json"

# Mirrors frontend/deepcad-selector.html TUNER_TEST_IDS. Kept in sync by hand
# so we never accidentally synthesise images for the full 35-sample gallery
# (each call costs and we only want clean views for the highlighted demos).
DEFAULT_TEST_IDS = [
    "deepcadimg_000035",
    "deepcadimg_002354",
    "deepcadimg_117514",
    "deepcadimg_128105",
]

VIEW_DIRS: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
    ("Top",    np.array([0, 0, -1.0]),  np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
    ("Bottom", np.array([0, 0,  1.0]),  np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    ("Front",  np.array([0, -1.0, 0]),  np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    ("Back",   np.array([0,  1.0, 0]),  np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    ("Right",  np.array([-1.0, 0, 0]),  np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    ("Left",   np.array([ 1.0, 0, 0]),  np.array([0,  1.0, 0]), np.array([0, 0, 1.0])),
]

# Output canvas: 1536x1024 = supported gpt-image-2 size, 3:2 aspect.
# 3 cols * 2 rows of view cells, each cell 512x512 (depth left half + silhouette
# right half stacked).
GRID_COLS = 3
GRID_ROWS = 2
CANVAS_W = 1536
CANVAS_H = 1024
CELL_W = CANVAS_W // GRID_COLS  # 512
CELL_H = CANVAS_H // GRID_ROWS  # 512


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _depth_to_rgb_array(values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    H, W = mask.shape
    img = np.full((H, W, 3), 245, dtype=np.uint8)  # white background for image-gen friendliness
    if not mask.any():
        return img
    span = max(vmax - vmin, 1e-9)
    d = values[mask]
    norm = np.clip((d - vmin) / span, 0.0, 1.0)
    # Friendly grayscale-ish ramp so the model doesn't try to "correct" the
    # rainbow palette into something else. Near = darker, far = lighter.
    g = (60 + 170 * norm).astype(np.uint8)
    img[mask] = np.stack([g, g, g], axis=-1)
    return img


def _raycast_mesh_view(scene, view_dir, right, up, center, size, panel) -> np.ndarray:
    u = np.linspace(-size / 2, size / 2, panel)
    v = np.linspace(size / 2, -size / 2, panel)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    plane_origin = center + (-view_dir) * size
    origins = (
        plane_origin
        + uu[..., None] * right
        + vv[..., None] * up
    ).reshape(-1, 3)
    dirs = np.tile(view_dir, (origins.shape[0], 1))
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    return ans["t_hit"].numpy().reshape(panel, panel)


def render_clean_input_grid(stl_path: Path, out_path: Path) -> dict:
    """Render the recon as a 1536x1024 6-view grid at gpt-image-2 input size.

    Each cell is 512x512: left half = grayscale depth, right half = silhouette
    mask. No labels, no overlay, no rainbow palette. Background is white. This
    is what the image-edit endpoint receives.
    """
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"empty mesh: {stl_path}")
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    bb = mesh.get_axis_aligned_bounding_box()
    bmin = np.asarray(bb.get_min_bound())
    bmax = np.asarray(bb.get_max_bound())
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    size = float(extent.max()) * 1.18

    half_w = CELL_W // 2  # 256
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color="white")

    silhouette_areas: dict[str, float] = {}
    for idx, (name, view_dir, right, up) in enumerate(VIEW_DIRS):
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        cx = col * CELL_W
        cy = row * CELL_H

        hit_t = _raycast_mesh_view(scene, view_dir, right, up, center, size, half_w)
        # Resample vertically to fill the 512px-tall cell so the cell is a
        # square depth-then-silhouette pair scaled to (256+256) x 512.
        hit_t_resized = np.repeat(hit_t, CELL_H // half_w, axis=0)
        mesh_mask = np.isfinite(hit_t_resized)

        if mesh_mask.any():
            vmin = float(hit_t_resized[mesh_mask].min())
            vmax = float(hit_t_resized[mesh_mask].max())
        else:
            vmin = 0.0
            vmax = 1.0
        depth_img = _depth_to_rgb_array(hit_t_resized, mesh_mask, vmin, vmax)

        sil = np.full((CELL_H, half_w, 3), 255, dtype=np.uint8)  # white background
        sil[mesh_mask] = (40, 40, 48)  # near-black ink for the part body
        # Holes will appear as white islands inside the dark silhouette - the
        # opposite contrast convention from the dark-on-white CAD drawing
        # convention but easier for gpt-image-2 to "preserve dark holes"...
        # Actually, flip: light part on dark would confuse "preserve dark holes
        # = preserve through-holes". So we keep the part DARK and the holes
        # WHITE, and our prompt says "preserve every white island inside the
        # dark silhouette - those are real through-holes".

        silhouette_areas[name] = float(mesh_mask.mean())

        canvas.paste(Image.fromarray(depth_img), (cx, cy))
        canvas.paste(Image.fromarray(sil), (cx + half_w, cy))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return {"silhouette_area": silhouette_areas, "bbox_size": size}


# ---------------------------------------------------------------------------
# OpenAI gpt-image-2 cleanup
# ---------------------------------------------------------------------------

CLEANUP_PROMPT = """This is a 6-view orthographic engineering blueprint of one mechanical CAD part,
reverse-engineered from a noisy 3D point-cloud scan. Your job is to redraw the
views as if they came from the ORIGINAL clean CAD model: keep the same shape
and proportions, clean up scan noise, and keep only the openings that are
unambiguously real designed through-features.

LAYOUT (do NOT change): a 3 columns x 2 rows grid of six views in this order
- top row, left to right:     Top, Bottom, Front
- bottom row, left to right:  Back, Right, Left
Each cell is split into two halves:
- LEFT half:  grayscale depth render (darker = nearer to camera)
- RIGHT half: black-on-white silhouette mask of the same view

DEPTH HALF (LEFT) - PRESERVE HEIGHT STEPS AND BANDS. DO NOT FLATTEN.
The left half encodes surface height: each clearly different brightness level is
a different coplanar face, step, shoulder, or depth tier (e.g. L-shaped step on
a top face, counterbore rings, abutting regions at different heights).
- Keep the SAME NUMBER of distinct gray levels as the input, in the SAME
  relative order (which tone is darker / lighter must match the input).
- You may smooth speckle and ragged pixels WITHIN a single band; you may
  sharpen the boundary BETWEEN two bands; you may straighten step edges that
  should be axis-aligned. You may NOT merge two or more bands into one
  uniform gray — that erases engineering depth context the downstream CAD
  inference needs.
- If the input shows side-by-side or concentric regions at different grays,
  the output must still show all of them as separate regions (cleaner edges,
  same tier count).
- Only the RIGHT half (silhouette) uses the hole/noise rules below; the LEFT
  half is for depth fidelity, not for "simplifying away" real geometry.

ABSOLUTE RULE - PRESERVE OVERALL SIZE, POSITION AND PROPORTIONS.
The size, position and aspect ratio of each silhouette in the input is correct.
The cleaned output must occupy the same cell, at the same overall scale, with
the same overall aspect ratio. You may NOT:
  - shrink or squash the silhouette into a different aspect ratio (e.g. do
    NOT redraw a long oblong as a small square)
  - reposition the part within its cell
  - change which cell of the grid is occupied by which view
  - introduce new global geometry that wasn't suggested by the input

CLEAN UP NOISE BOTH INSIDE AND ON THE BOUNDARY.
Within the constraint above, you SHOULD aggressively smooth out scan noise
that appears either inside the silhouette (white pinholes/blobs) or on the
silhouette boundary (small notches, peninsulas, ragged edges). A clean CAD
silhouette has straight edges where it should be straight, smooth curves
where it should be curved, and clean axis-aligned corners.

A small notch or pocket on the OUTER BOUNDARY of one view (e.g. a tiny
"hanging slot" cut into the top of a column in the Front view) is REAL only
if perpendicular views confirm it. Specifically:
  - A slot cut down through the top of a column in Front view must show a
    matching slot-shaped opening in the Top silhouette.
  - A notch cut sideways into the column in Front view must show a matching
    notch in the Right or Left silhouette at the same height.
If the perpendicular view does NOT confirm the boundary notch, the notch is
scan noise. SMOOTH IT OUT - redraw the boundary as the clean continuous
outline that the noise was nibbling away at.

INTERIOR FEATURES - distinguish real openings from noise.

A white opening INSIDE a dark silhouette is REAL only if it passes the matching
opposite-view test:

  - A vertical (Z-axis) through-bore must show a clean opening in BOTH Top
    AND Bottom silhouettes at the SAME relative location.
  - A horizontal (Y-axis, front-back) through-bore must show a clean opening
    in BOTH Front AND Back silhouettes at the SAME relative location.
  - A horizontal (X-axis, left-right) through-bore must show a clean opening
    in BOTH Right AND Left silhouettes at the SAME relative location.
  - A through-bore visible end-on (a small clean circle inside a silhouette
    that is the END FACE of the bore) is real if the perpendicular views
    show the bore as a clear opening. Example: dumbbell with bores along
    the long axis - Front + Back show large openings, Right + Left show
    small end-on circles. Preserve all four.

If a feature passes the matching test, KEEP it (preserve size, shape and
position). If a feature fails the matching test, FILL IT with the silhouette
color (it is reconstruction noise, not a real hole).

EXPLICITLY NOISE - always close these:
  - Small notches near the top or bottom of a column that have NO matching
    opening in the perpendicular axial view (e.g. a Front-view "hanging
    slot" with no corresponding opening in Top). FILL.
  - Concentric rings in a depth view paired with a SOLID opposite silhouette
    -> that's a blind recess / counterbore, NOT a through-hole. The recess
    is captured by depth shading; keep the silhouette solid.
  - Pinholes, irregular speckle, ragged peninsulas of white inside the
    silhouette. FILL.
  - Any opening visible in fewer than the required number of matched views.

NEVER invent symmetric features. If Front + Back show a real bore, keep it
in Front + Back; do NOT add a matching opening to Right or Left unless those
views already show one.

When you redraw each cell:
  - LEFT (depth): preserve all height bands; denoise inside bands only; do not
    homogenize distinct grays into one flat surface.
  - RIGHT (silhouette): reproduce the outer outline at the SAME size and
    position as the input; fill noise gaps inside; keep real openings; make
    edges crisp (straight where straight, smooth curves where curved).

KEEP the same 3x2 LAYOUT and the same 6 views in the same positions. KEEP the
depth-then-silhouette internal split of each cell. Use grayscale only. Do NOT
add labels, captions, dimensions, borders, arrows, perspective, shading or
artistic flourishes. No extra geometry, no decoration. The result should look
like a clean engineering orthographic set with the same overall layout and
proportions as the input."""


def _png_b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


def synthesize_clean(client: OpenAI, input_png: Path, out_png: Path,
                     model: str, quality: str, max_attempts: int = 4) -> dict:
    """Call gpt-image-2 image-edit and write the result PNG."""
    last_err = ""
    for attempt in range(max_attempts):
        try:
            with input_png.open("rb") as f:
                resp = client.images.edit(
                    model=model,
                    image=f,
                    prompt=CLEANUP_PROMPT,
                    size=f"{CANVAS_W}x{CANVAS_H}",
                    quality=quality,
                    output_format="png",
                )
            data = resp.data[0].b64_json
            if not data:
                raise RuntimeError("response had no b64_json payload")
            out_png.write_bytes(_png_b64_to_bytes(data))
            usage = getattr(resp, "usage", None)
            return {
                "success": True,
                "out_path": str(out_png),
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                "model": model,
                "quality": quality,
            }
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return {"success": False, "error": f"all {max_attempts} attempts failed: {last_err}"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated sample ids; defaults to the test set")
    p.add_argument("--model", default="gpt-image-2",
                   help="OpenAI image model id (default gpt-image-2). Falls back automatically "
                        "if the id isn't available on your account.")
    p.add_argument("--quality", default="high",
                   choices=["low", "medium", "high", "auto", "standard"],
                   help="gpt-image-2 quality tier")
    p.add_argument("--no-sync", action="store_true", help="do not copy clean PNGs to frontend")
    p.add_argument("--force", action="store_true",
                   help="re-render and re-call OpenAI even if the clean PNG already exists")
    return p.parse_args()


def resolve_targets(only_arg: str | None, manifest: list[dict]) -> list[str]:
    if only_arg:
        wanted = [s.strip() for s in only_arg.split(",") if s.strip()]
    else:
        wanted = list(DEFAULT_TEST_IDS)
    have = {e.get("sample_id") for e in manifest if e.get("success")}
    final = [sid for sid in wanted if sid in have]
    missing = [sid for sid in wanted if sid not in have]
    if missing:
        print(f"WARNING: skipping ids not in manifest / not successfully reconstructed: {missing}")
    return final


def stl_path_for(manifest: list[dict], sample_id: str) -> Path | None:
    for e in manifest:
        if e.get("sample_id") == sample_id and e.get("recon_noisy_stl"):
            p = OUT_DIR / e["recon_noisy_stl"]
            return p if p.exists() else None
    return None


def update_manifest_clean_path(manifest: list[dict], sample_id: str, rel_path: str) -> None:
    for e in manifest:
        if e.get("sample_id") == sample_id:
            cohesion = e.get("cohesion") or {}
            cohesion["geometry_view_clean"] = rel_path
            e["cohesion"] = cohesion
            return


def sync_to_frontend() -> None:
    if not GEOMETRY_VIEWS_DIR.exists():
        return
    FRONTEND_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    for png in GEOMETRY_VIEWS_DIR.glob("*_geometry_clean.png"):
        shutil.copy2(png, FRONTEND_GEOMETRY_DIR / png.name)
    if MANIFEST_PATH.exists():
        FRONTEND_MANIFEST = REPO / "frontend" / "deepcad_pcrecon_stl" / "manifest.json"
        if FRONTEND_MANIFEST.parent.exists():
            shutil.copy2(MANIFEST_PATH, FRONTEND_MANIFEST)


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (add it to .env)", file=sys.stderr)
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest missing at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    targets = resolve_targets(args.only, manifest)
    if not targets:
        print("nothing to do (no targets resolved)")
        return

    GEOMETRY_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_VIEWS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    print(f"=== synth-clean: {len(targets)} samples ({args.model}, quality={args.quality}) ===")
    succ = 0
    for sid in targets:
        out_png = GEOMETRY_VIEWS_DIR / f"{sid}_geometry_clean.png"
        if out_png.exists() and not args.force:
            print(f"  {sid}: clean PNG already exists, skipping (use --force to redo)")
            update_manifest_clean_path(manifest, sid, f"geometry_views/{sid}_geometry_clean.png")
            succ += 1
            continue

        stl = stl_path_for(manifest, sid)
        if stl is None:
            print(f"  {sid}: no recon_noisy_stl on disk, skipping")
            continue

        input_png = CLEAN_VIEWS_SOURCE_DIR / f"{sid}_clean_input.png"
        try:
            render_clean_input_grid(stl, input_png)
        except Exception as exc:
            print(f"  {sid}: render failed: {exc}")
            continue

        print(f"  {sid}: calling {args.model}...")
        result = synthesize_clean(client, input_png, out_png, args.model, args.quality)
        if not result.get("success"):
            print(f"    -> FAILED: {result.get('error')}")
            continue
        succ += 1
        print(f"    -> wrote {out_png.name}  ({result.get('input_tokens')} in / {result.get('output_tokens')} out tokens)")
        update_manifest_clean_path(manifest, sid, f"geometry_views/{sid}_geometry_clean.png")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    if not args.no_sync:
        sync_to_frontend()

    print(f"\n=== done: {succ}/{len(targets)} samples cleaned ===")


if __name__ == "__main__":
    main()
