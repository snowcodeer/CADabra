"""Vision-guided per-sample parameter tuning for noisy point cloud reconstruction.

Workflow (hybrid mode, default):
    1. Global pass: pick the worst N non-watertight samples, render each as a
       6-view orthographic depth grid, send all images + their metrics +
       current default RECON_PARAMS / SCORE_PARAMS to Claude Opus 4.7,
       receive new shared defaults that should improve the batch as a whole.
       Re-run all 35 samples with the new shared defaults.
    2. Per-sample pass: for any sample that is still not watertight, do up to
       --max-iters vision-guided regen passes. The model only sees the
       reconstruction (never the ground truth STL) to avoid data leakage.

Outputs land in backend/outputs/deepcad_pc_recon_stl/:
    - tuner_renders/*.png       per-iteration multi-view renders
    - param_overrides.json      cumulative overrides keyed by sample id, plus
                                "_global" for the shared defaults
    - tuner_log.json            full per-iteration log (diagnoses, metrics)

The reconstruction script gained a `--params-override <json>` flag in the
same change, which this tuner uses to inject overrides per subprocess call.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

import anthropic
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env")
except ImportError:
    pass


REPO = Path(__file__).resolve().parent.parent
RECON_SCRIPT = REPO / "scripts" / "reconstruct_meshes_from_pointclouds.py"
SRC_PLY_DIR = REPO / "backend" / "sample_data" / "deepcad_selected_ply"
OUT_DIR = REPO / "backend" / "outputs" / "deepcad_pc_recon_stl"
FRONTEND_OUT_DIR = REPO / "frontend-preview" / "deepcad_pcrecon_stl"
RENDER_DIR = OUT_DIR / "tuner_renders"
COHESION_RENDER_DIR = OUT_DIR / "cohesion_renders"
GEOMETRY_VIEWS_DIR = OUT_DIR / "geometry_views"
FRONTEND_GEOMETRY_DIR = REPO / "frontend-preview" / "deepcad_geometry_views"
OVERRIDES_PATH = OUT_DIR / "param_overrides.json"
LOG_PATH = OUT_DIR / "tuner_log.json"
MANIFEST_PATH = OUT_DIR / "manifest.json"

MODEL_ID = "claude-opus-4-7"
PYTHON_BIN = sys.executable

# Acceptance gate. Watertight is the primary requirement.
def is_acceptable(metrics: dict) -> bool:
    return bool(metrics.get("watertight"))


def quality_score(metrics: dict) -> tuple:
    """Sortable key. Larger = better. Watertight wins, then fewer boundary
    edges, then lower long-edge density, then lower chamfer.

    Use explicit None checks rather than `or` so that legitimately good values
    of 0 (e.g. boundary_edges == 0 with a non-orientable mesh) aren't treated
    as missing and replaced with worst-case sentinels.
    """
    bnd = metrics.get("boundary_edges")
    led = metrics.get("long_edge_density")
    cham = metrics.get("chamfer")
    return (
        1 if metrics.get("watertight") else 0,
        -int(bnd if bnd is not None else 10**9),
        -float(led if led is not None else 1.0),
        -float(cham if cham is not None else 10.0),
    )


# ---------------------------------------------------------------------------
# Renderer (Open3D RaycastingScene, 6-view orthographic depth grid)
# ---------------------------------------------------------------------------

VIEW_DIRS: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
    # name, view direction (camera -> origin), right axis, up axis
    ("Top",    np.array([0, 0, -1.0]),  np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
    ("Bottom", np.array([0, 0,  1.0]),  np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    ("Front",  np.array([0, -1.0, 0]),  np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    ("Back",   np.array([0,  1.0, 0]),  np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    ("Right",  np.array([-1.0, 0, 0]),  np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    ("Left",   np.array([ 1.0, 0, 0]),  np.array([0,  1.0, 0]), np.array([0, 0, 1.0])),
]

PANEL = 280  # px per view panel
LABEL_BAND = 24
SPLAT_RADIUS = 2  # pixel radius for point cloud splats


def _depth_to_rgb_array(values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map depth values into the same red-green-blue ramp used for both panels."""
    H, W = mask.shape
    img = np.full((H, W, 3), 18, dtype=np.uint8)
    if not mask.any():
        return img
    span = max(vmax - vmin, 1e-9)
    d = values[mask]
    norm = np.clip((d - vmin) / span, 0.0, 1.0)
    r = (255 * (1.0 - norm)).astype(np.uint8)
    g = (60 + 140 * (1.0 - np.abs(norm - 0.5) * 2.0)).clip(0, 255).astype(np.uint8)
    b = (255 * norm).astype(np.uint8)
    img[mask] = np.stack([r, g, b], axis=-1)
    return img


def _project_points(points: np.ndarray, view_dir: np.ndarray, right: np.ndarray, up: np.ndarray,
                    center: np.ndarray, size: float, panel: int):
    """Orthographic projection of points onto a (panel, panel) image plane.
    Returns pixel x, pixel y, and depth measured from the camera plane along
    the view direction (smaller = closer to camera). Matches the raycast
    convention so PC and mesh panels can share a depth ramp.
    """
    plane_origin = center + (-view_dir) * size
    R_uv = np.stack([right, up], axis=1)
    uv = (points - center) @ R_uv
    u = uv[:, 0]
    v = uv[:, 1]
    depth = (points - plane_origin) @ view_dir
    mask = (np.abs(u) <= size / 2) & (np.abs(v) <= size / 2)
    u, v, depth = u[mask], v[mask], depth[mask]
    px = ((u + size / 2) / size * (panel - 1)).astype(np.int32)
    py = ((size / 2 - v) / size * (panel - 1)).astype(np.int32)
    return px, py, depth


def _splat_pointcloud(panel: int, px: np.ndarray, py: np.ndarray, depth: np.ndarray,
                      vmin: float, vmax: float, radius: int = SPLAT_RADIUS) -> np.ndarray:
    """Z-buffered splat of points into an RGB panel using the same depth ramp as the mesh view."""
    img = np.full((panel, panel, 3), 18, dtype=np.uint8)
    if px.size == 0:
        return img
    zbuf = np.full((panel, panel), np.inf, dtype=np.float32)
    span = max(vmax - vmin, 1e-9)
    norm = np.clip((depth - vmin) / span, 0.0, 1.0)
    r = (255 * (1.0 - norm)).astype(np.uint8)
    g = (60 + 140 * (1.0 - np.abs(norm - 0.5) * 2.0)).clip(0, 255).astype(np.uint8)
    b = (255 * norm).astype(np.uint8)
    colors = np.stack([r, g, b], axis=-1)
    # draw far points first so near points overwrite
    order = np.argsort(-depth)
    px, py, depth, colors = px[order], py[order], depth[order], colors[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xx = np.clip(px + dx, 0, panel - 1)
            yy = np.clip(py + dy, 0, panel - 1)
            current = zbuf[yy, xx]
            update = depth < current
            yy_u, xx_u = yy[update], xx[update]
            if yy_u.size:
                zbuf[yy_u, xx_u] = depth[update]
                img[yy_u, xx_u] = colors[update]
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


def render_pair_six_view_grid(ply_path: Path, stl_path: Path, out_path: Path) -> None:
    """Side-by-side orthographic depth views of source point cloud and reconstructed mesh.

    Layout (panel = PANEL px):
        Row 1 label band, then 3 cells of [PC | Recon] per view (Top, Front, Left)
        Row 2 label band, then 3 cells of [PC | Recon] per view (Bottom, Back, Right)
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"empty point cloud: {ply_path}")

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"empty mesh: {stl_path}")
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    # Use joint bbox so PC and mesh share the same scale, even if the recon is
    # slightly bigger or smaller than the source cloud.
    pc_min = points.min(0); pc_max = points.max(0)
    mesh_bb = mesh.get_axis_aligned_bounding_box()
    m_min = np.asarray(mesh_bb.get_min_bound()); m_max = np.asarray(mesh_bb.get_max_bound())
    joint_min = np.minimum(pc_min, m_min)
    joint_max = np.maximum(pc_max, m_max)
    center = (joint_min + joint_max) / 2.0
    extent = joint_max - joint_min
    size = float(extent.max()) * 1.18

    cell_w = PANEL * 2  # PC + Recon
    cell_h = PANEL + LABEL_BAND
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 2), color="white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, (name, view_dir, right, up) in enumerate(VIEW_DIRS):
        col = idx % 3
        row = idx // 3
        cx = col * cell_w
        cy = row * cell_h

        hit_t = _raycast_mesh_view(scene, view_dir, right, up, center, size, PANEL)
        mesh_mask = np.isfinite(hit_t)

        px_pc, py_pc, z_pc = _project_points(points, view_dir, right, up, center, size, PANEL)

        # shared depth normalisation across PC + mesh for honest visual comparison
        depths_combined: list[np.ndarray] = []
        if mesh_mask.any():
            depths_combined.append(hit_t[mesh_mask])
        if z_pc.size:
            depths_combined.append(z_pc)
        if depths_combined:
            all_d = np.concatenate(depths_combined)
            vmin = float(all_d.min())
            vmax = float(all_d.max())
        else:
            vmin = 0.0
            vmax = 1.0

        pc_img = _splat_pointcloud(PANEL, px_pc, py_pc, z_pc, vmin, vmax)
        mesh_img = _depth_to_rgb_array(hit_t, mesh_mask, vmin, vmax)

        # label band
        draw.rectangle([cx, cy, cx + cell_w, cy + LABEL_BAND], fill=(245, 245, 248))
        draw.text((cx + 6, cy + 4), f"{name} - point cloud", fill=(40, 40, 40), font=font)
        draw.text((cx + PANEL + 6, cy + 4), f"{name} - reconstruction", fill=(40, 40, 40), font=font)

        canvas.paste(Image.fromarray(pc_img), (cx, cy + LABEL_BAND))
        canvas.paste(Image.fromarray(mesh_img), (cx + PANEL, cy + LABEL_BAND))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_cohesion_grid(stl_path: Path, out_path: Path,
                          panel: int = 360) -> dict:
    """Render the reconstruction alone as a clean orthographic geometry view.

    This is the canonical "captured geometry representation" — a 6-view depth
    grid with a silhouette band per view. Produced entirely from the recon
    mesh, with no point cloud overlay; downstream consumers (e.g. parametric
    CAD inference) can take this PNG as their geometry input regardless of
    whether the underlying 3D mesh is fully watertight.

    Returns a small dict of "what's in the picture" stats (silhouette area
    per view, etc.) that the caller can stash alongside the image.
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

    cell_w = panel * 2  # depth | silhouette
    cell_h = panel + LABEL_BAND
    cols, rows = 3, 2
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), color="white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    silhouette_areas: dict[str, float] = {}
    for idx, (name, view_dir, right, up) in enumerate(VIEW_DIRS):
        col = idx % cols
        row = idx // cols
        cx = col * cell_w
        cy = row * cell_h

        hit_t = _raycast_mesh_view(scene, view_dir, right, up, center, size, panel)
        mesh_mask = np.isfinite(hit_t)

        if mesh_mask.any():
            vmin = float(hit_t[mesh_mask].min())
            vmax = float(hit_t[mesh_mask].max())
        else:
            vmin = 0.0
            vmax = 1.0
        depth_img = _depth_to_rgb_array(hit_t, mesh_mask, vmin, vmax)

        # silhouette = solid white where mesh hit, black otherwise. Holes show
        # as black islands inside the part outline so a downstream LLM can
        # read topology directly off the mask.
        sil = np.zeros((panel, panel, 3), dtype=np.uint8)
        sil[mesh_mask] = (235, 235, 240)

        silhouette_areas[name] = float(mesh_mask.mean())

        draw.rectangle([cx, cy, cx + cell_w, cy + LABEL_BAND], fill=(245, 245, 248))
        draw.text((cx + 6, cy + 4), f"{name} - depth", fill=(40, 40, 40), font=font)
        draw.text((cx + panel + 6, cy + 4), f"{name} - silhouette", fill=(40, 40, 40), font=font)
        canvas.paste(Image.fromarray(depth_img), (cx, cy + LABEL_BAND))
        canvas.paste(Image.fromarray(sil), (cx + panel, cy + LABEL_BAND))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {"silhouette_area": silhouette_areas, "bbox_size": size}


def ply_path_for(sample_id: str) -> Path:
    return SRC_PLY_DIR / f"{sample_id}.ply"


# ---------------------------------------------------------------------------
# Recon subprocess wrapper
# ---------------------------------------------------------------------------

def run_recon(sample_id: str, ply_file: str, overrides: dict, max_attempts: int = 6) -> dict:
    payload = json.dumps(overrides) if overrides else ""
    cmd = [PYTHON_BIN, str(RECON_SCRIPT), "--sample-id", sample_id, "--ply-file", ply_file]
    if payload:
        cmd.extend(["--params-override", payload])
    last_err = ""
    for attempt in range(max_attempts):
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        out = proc.stdout.strip().splitlines()
        if proc.returncode == 0 and out:
            try:
                result = json.loads(out[-1])
                if result.get("success"):
                    return result
                last_err = result.get("error") or "no error message"
            except json.JSONDecodeError as exc:
                last_err = f"json decode: {exc}; raw={out[-1][:200]}"
        else:
            last_err = (proc.stderr or "")[-300:].strip() or f"exit={proc.returncode}"
        time.sleep(0.4 * (attempt + 1))
    return {"success": False, "error": f"all {max_attempts} attempts failed: {last_err}"}


# ---------------------------------------------------------------------------
# Default RECON_PARAMS / SCORE_PARAMS reader (parsed from recon script)
# ---------------------------------------------------------------------------

def read_default_params() -> dict:
    spec_globals: dict = {}
    text = RECON_SCRIPT.read_text()
    # extract just the constant blocks safely
    for var in ("CLEAN_PARAMS", "RECON_PARAMS", "SCORE_PARAMS", "HOLE_FILL_PARAMS",
                "PLANE_SNAP_PARAMS", "DECIMATION_PARAMS"):
        marker = f"{var} = "
        i = text.find(marker)
        if i < 0:
            continue
        # naive but works: literal_eval the dict by grabbing balanced braces
        start = text.find("{", i)
        depth = 0
        end = -1
        for j in range(start, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end > 0:
            try:
                spec_globals[var] = eval(text[start:end], {"__builtins__": {}}, {})
            except Exception:
                pass
    return spec_globals


def serialize_for_prompt(params: dict) -> dict:
    """Tuples -> lists so they round-trip through JSON cleanly."""
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, dict):
            out[k] = serialize_for_prompt(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def merge_overrides(base: dict, new: dict) -> dict:
    """Recursive shallow merge of the supported override dicts."""
    merged = deepcopy(base) if base else {"RECON_PARAMS": {}, "SCORE_PARAMS": {}, "HOLE_FILL_PARAMS": {}}
    for top in ("RECON_PARAMS", "SCORE_PARAMS", "HOLE_FILL_PARAMS"):
        for k, v in (new.get(top) or {}).items():
            merged.setdefault(top, {})[k] = v
    return merged


# ---------------------------------------------------------------------------
# Anthropic vision call
# ---------------------------------------------------------------------------

GLOBAL_SYSTEM_PROMPT = """You are an expert in 3D point-cloud surface reconstruction.

You will be shown several composite renderings. Each composite shows six
orthographic views of one CAD part. For every view there are TWO panels side
by side:
  - LEFT panel  = the source point cloud (noisy virtual scan, the input data).
  - RIGHT panel = the current reconstructed mesh (the output we are tuning).
Both panels share the same camera and the same depth color ramp (red = near,
blue = far) so they are directly comparable.

How to read the panels:
  - HOLES IN THE MESH: the LEFT panel shows points covering an area but the
    RIGHT panel has dark / background pixels there.
  - OVER-EXTRUSION / BRIDGING: the RIGHT panel shows mesh in an area where
    the LEFT panel has no points (or a clear gap).
  - DIAGONAL BRIDGES across shallow steps: smooth color ramp on the right
    where the point cloud on the left shows a clear depth discontinuity.

Your job: propose ONE SHARED set of reconstruction defaults that, applied
globally, should produce watertight meshes across all of the parts shown.

Your job: propose ONE SHARED set of reconstruction defaults that, applied
globally, should produce watertight meshes across all of the parts shown.

Adjustable parameters (only return keys you want to change):
  RECON_PARAMS:
    - poisson_depths:           list[int] in [7, 12]. Higher = finer detail. Default [8,9,10,11].
    - poisson_density_quantile: float in [0.005, 0.05]. Lower = keeps more wall coverage.
    - envelope_radius_factor:   float in [1.5, 4.0]. How tightly to crop Poisson fringe (smaller = looser crop, more coverage).
    - alpha_factors:            list[float] in [0.01, 0.15]. Smaller alpha = sharper, larger = fills more.
    - ball_pivot_radius_factors: list[float] in [0.5, 4.0], or [] to disable ball pivot.
  SCORE_PARAMS:
    - poisson_bonus:            float in [0.0, 0.05]. Reward Poisson candidates.
    - alpha_bonus:              float in [0.0, 0.05]. Reward alpha-shape candidates.
    - boundary_penalty:         float in [0.0, 5.0]. Penalize high boundary edge density.
    - long_edge_penalty:        float in [0.0, 5.0]. Penalize diagonal bridges.
    - real_hole_bonus:          float in [0.0, 0.10]. Reward each detected real-hole loop in
                                a candidate. Increase this when alpha-shape candidates are
                                winning by closing CAD bores during construction. Default 0.025.
  HOLE_FILL_PARAMS (post-recon hole-stitching, OFF by default):
    - enable_in_noisy:    bool. Set true to run a 2-pass aggressive hole fill on the chosen candidate.
    - pass1_frac:         float in [0.05, 0.6]. Pass 1 closes loops up to (frac x bbox diagonal).
    - pass2_frac:         float in [0.3, 1.5]. Pass 2 closes loops up to (frac x bbox diagonal).
    - min_loop_length_abs: float in [0.05, 1.0]. Absolute minimum loop length closed.
    - protect_real_holes: bool. Default true. Auto-detects circular/large boundary loops and refuses
                          to ever close them. When true, pass1/pass2 above are AUTOMATICALLY CAPPED
                          below the smallest detected real-hole perimeter. NEVER turn this off.
    - circular_protect_perim_frac: float in [0.04, 0.20]. Min loop perim (frac of bbox diag) for circular protection.
    - circularity_threshold:       float in [0.40, 0.85]. 4*pi*A/P^2 above this counts as circular.
    - planarity_threshold:         float in [0.01, 0.10]. Max planar residual (frac of bbox) for circular classification.
    - absolute_protect_perim_frac: float in [0.10, 0.35]. Loops bigger than this are protected regardless of shape.

Return ONLY valid JSON, no markdown fences:
{
  "diagnosis": "<what you see across the samples in 1-2 sentences>",
  "RECON_PARAMS": {<subset>},
  "SCORE_PARAMS": {<subset>},
  "HOLE_FILL_PARAMS": {<subset>},
  "expected_improvement": "<one line>"
}
"""

PER_SAMPLE_SYSTEM_PROMPT = """You are an expert in 3D point-cloud surface reconstruction.

You will be shown ONE composite rendering of a single CAD part: six
orthographic views, each split into TWO side-by-side panels.
  - LEFT  = source point cloud (noisy virtual scan, the input data).
  - RIGHT = current reconstructed mesh (the output we are tuning).
Both panels share the same camera and depth color ramp (red=near, blue=far).

Diagnose by COMPARING the two panels:
  - HOLE in mesh: LEFT has dense points where RIGHT has dark / background.
  - OVER-EXTRUSION / BRIDGE: RIGHT has mesh where LEFT has a clear gap.
  - DIAGONAL BRIDGE across a shallow step: smooth depth ramp on RIGHT where
    LEFT shows two flat depth bands at clearly different depths.

You may NOT see the ground-truth STL; the LEFT panel (point cloud) is the
closest proxy. Propose parameter overrides that should make this specific
sample watertight on the next attempt.

Adjustable parameters (only return keys you want to change):
  RECON_PARAMS:
    - poisson_depths:           list[int] in [7, 12].
    - poisson_density_quantile: float in [0.005, 0.05]. Lower = keeps more wall coverage.
    - envelope_radius_factor:   float in [1.5, 4.0]. Smaller = looser crop, more coverage.
    - alpha_factors:            list[float] in [0.01, 0.15].
    - ball_pivot_radius_factors: list[float] in [0.5, 4.0], or [] to disable.
  SCORE_PARAMS:
    - poisson_bonus, alpha_bonus, boundary_penalty, long_edge_penalty: float in [0.0, 5.0].
    - real_hole_bonus: float in [0.0, 0.10]. Reward per detected real-hole loop. Bump this when
      a candidate that closed real CAD bores is being chosen over one that preserved them.
  HOLE_FILL_PARAMS (post-recon hole-stitching, OFF by default. Use this when the recon
                     looks mostly correct but has small/medium holes you want closed.
                     This is the preferred LAST-RESORT lever before declaring the sample
                     'improved but not watertight'.):
    - enable_in_noisy:     bool. Set true to actually run hole filling.
    - pass1_frac:          float in [0.05, 0.6]. Pass 1 closes loops up to frac x bbox diagonal.
    - pass2_frac:          float in [0.3, 1.5]. Pass 2 is more aggressive.
    - min_loop_length_abs: float in [0.05, 1.0]. Absolute floor for closeable loop length.
    - protect_real_holes:  bool. Default true. Auto-detects circular/large boundary loops and
                           refuses to ever close them. When on, pass1/pass2 above are
                           AUTOMATICALLY CAPPED below the smallest detected real-hole perimeter
                           (so 'watertight via bridging a CAD bore' literally cannot happen).
                           NEVER turn this off.
    - circular_protect_perim_frac: float in [0.04, 0.20]. Min loop perim (frac of bbox diag) for circular protection.
    - circularity_threshold:       float in [0.40, 0.85]. 4*pi*A/P^2 above this counts as circular.
    - planarity_threshold:         float in [0.01, 0.10]. Max planar residual (frac of bbox) for circular classification.
    - absolute_protect_perim_frac: float in [0.10, 0.35]. Loops bigger than this are protected regardless of shape.

Hard rules:
  - Never reduce poisson_depths below the current effective set.
  - NEVER set HOLE_FILL_PARAMS.protect_real_holes = false. The system auto-detects and
    protects real CAD through-holes. Closing them to chase watertight is a regression,
    not progress. If you see a real hole getting filled in the reconstruction render,
    LOWER circular_protect_perim_frac (more aggressive protection) instead.
  - If you see HOLES, prefer DECREASING envelope_radius_factor and/or
    DECREASING poisson_density_quantile rather than increasing depth alone.
  - If you see DIAGONAL BRIDGES, ADD smaller values to alpha_factors and
    INCREASE long_edge_penalty.
  - If two iterations of recon-only tuning have already failed to close
    boundaries, ENABLE HOLE_FILL_PARAMS as the next step.
  - "Watertight" is only a win if the part still has all its real cavities. A part with
    visible bridging across a real hole is WORSE than a non-watertight recon that
    preserved the hole.

Return ONLY valid JSON, no markdown fences:
{
  "diagnosis": "<what you see in 1-2 sentences>",
  "RECON_PARAMS": {<subset>},
  "SCORE_PARAMS": {<subset>},
  "HOLE_FILL_PARAMS": {<subset>},
  "expected_improvement": "<one line>"
}
"""


COHESION_SYSTEM_PROMPT = """You are evaluating a CAD reconstruction's GEOMETRY REPRESENTATION:
six clean orthographic depth + silhouette views of just the reconstructed mesh.
This PNG is the input to TWO downstream steps:
  (a) an image-generation pass (gpt-image-2) that synthetically cleans up
      small surface gaps in the silhouette / sharpens edges WITHOUT modifying
      the underlying 3D mesh, and
  (b) a parametric CAD-inference step that reads the cleaned views.

Because the image-gen pass exists, the bar here is "does the underlying 3D
recon preserve the part's TOPOLOGICAL INTENT (real through-holes, real
mirror/rotational symmetry, real face boundaries)?". SMALL COSMETIC GAPS in
the silhouette are FINE and even preferred over aggressive geometric closure
that risks bridging a real hole. The image-gen pass will fill those.

For each view you have two side-by-side panels:
  - LEFT  = depth-shaded render (red=near, blue=far) showing surface slope.
  - RIGHT = silhouette mask: light pixels where mesh exists, dark where it
            does not. Holes show as dark islands inside the part outline.

Score COHESION (0-100). The axes that matter, in priority order:

1. HOLE INTENT (most important by far)
   GOOD: every through-hole shows as a clean dark island in BOTH the
         depth view and the silhouette view, with an outline that matches
         across opposing views (e.g. Top and Bottom). A bore visible from
         one face MUST also exit on the opposing face.
   BAD : a hole visible in one view but bridged in the opposite view; a
         hole reduced to a shallow bowl in the depth view; a silhouette
         that looks like a pillow with no holes when there should be some.
   This is a REGRESSION even if the rest of the mesh is perfect.

2. SYMMETRIC FEATURES
   GOOD: paired holes/bosses match in size and position across the natural
         symmetry axis (Z-rotational, X- or Y-mirror) of the part.
   BAD : one of a symmetric pair is missing, smaller, or shifted.

3. EDGE QUALITY (silhouette boundary)
   GOOD: silhouette outlines are crisp and axis-aligned; corners are sharp;
         steps between depth bands are clean cliffs in the depth view.
   BAD : ragged/wavy silhouette outlines, staircase edges, smooth depth
         ramps where there should be a hard step.

4. SMALL SURFACE GAPS (cosmetic only)
   Small dark patches inside the silhouette that aren't real holes are
   acceptable - the downstream image-gen pass will infill them. Mention
   them in `diagnosis` but DO NOT score them harshly.

Suggest parameter overrides that would IMPROVE the things you scored low on.
Available knobs (omit a key to leave it unchanged):

  RECON_PARAMS:
    - poisson_depths:           list[int] in [7, 12].
    - poisson_density_quantile: float in [0.005, 0.05]. Lower = more wall coverage.
    - envelope_radius_factor:   float in [1.5, 4.0]. Smaller = looser crop, more coverage.
    - alpha_factors:            list[float] in [0.01, 0.15].
    - ball_pivot_radius_factors: list[float] in [0.5, 4.0], or [] to disable.
  SCORE_PARAMS:
    - poisson_bonus, alpha_bonus: float in [0.0, 0.05].
    - boundary_penalty, long_edge_penalty: float in [0.0, 5.0].
    - real_hole_bonus: float in [0.0, 0.10]. Reward per detected real-hole loop.
                       BUMP THIS if a candidate that bridged a bore is winning.
  HOLE_FILL_PARAMS:
    - enable_in_noisy:     bool. Default ON in cohesion baseline. Conservative
                           2-pass fill that closes only loops smaller than the
                           smallest protected real-hole perimeter. Safe.
    - pass1_frac:          float in [0.05, 0.6]. Pass 1 cap (frac of bbox diag).
    - pass2_frac:          float in [0.3, 1.5].
    - min_loop_length_abs: float in [0.05, 1.0].
    - protect_real_holes:  bool. ALWAYS true. NEVER set false.
    - circular_protect_perim_frac: float in [0.04, 0.20]. LOWER this if a real
                                   hole is being filled in.
    - circularity_threshold:       float in [0.40, 0.85].
    - planarity_threshold:         float in [0.01, 0.10].
    - absolute_protect_perim_frac: float in [0.10, 0.35].
    - through_hole_detection: bool. Default true. Raycast-confirmed through-bores
                              are auto-protected; do not turn off.
    - through_hole_max_depth_frac: float in [0.2, 0.9]. How deep into the part
                                   a probe ray may travel and still classify the
                                   loop as a through-hole. Lower = stricter.
    - meshy_close:         bool. DEFAULT OFF and YOU SHOULD KEEP IT OFF unless
                           the part has provably ZERO real through-holes (genus 0)
                           and the recon shows large fringe gaps. The image-gen
                           pass handles cosmetic cleanup better than meshy_close.

Hard rules:
  - PRESERVING REAL THROUGH-HOLES IS NON-NEGOTIABLE. A "cohesive but bridged"
    mesh scores LOWER than a "gappy but topologically faithful" one.
  - The image-gen pass will clean small cosmetic gaps for you. Do NOT enable
    meshy_close to chase visual cleanliness; that lever has bridged real holes
    in past runs.
  - NEVER set HOLE_FILL_PARAMS.protect_real_holes = false. If a real hole is
    being filled in, LOWER circular_protect_perim_frac AND/OR LOWER
    through_hole_max_depth_frac so the protection triggers sooner.
  - Smallest, safest tweaks per round. Big jumps regress.

Return ONLY valid JSON, no markdown fences:
{
  "cohesion_score": <int 0-100>,
  "diagnosis": "<2-3 sentences naming the top defects, in priority order>",
  "hole_count": <int>,
  "symmetry": "<e.g. 'Z-rotational 4-fold', 'X-mirror', 'none'>",
  "RECON_PARAMS": {<subset>},
  "SCORE_PARAMS": {<subset>},
  "HOLE_FILL_PARAMS": {<subset>},
  "expected_improvement": "<one line>"
}
"""


def encode_image(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def call_opus(client: anthropic.Anthropic, system_prompt: str, user_text: str,
              image_paths: list[Path], max_tokens: int = 1500) -> dict:
    content: list[dict] = [encode_image(p) for p in image_paths]
    content.append({"type": "text", "text": user_text})
    msg = client.messages.create(
        model=MODEL_ID,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def metrics_of(entry: dict) -> dict:
    return {
        "watertight": entry.get("watertight"),
        "edge_manifold": entry.get("edge_manifold"),
        "boundary_edges": entry.get("boundary_edges"),
        "long_edge_density": entry.get("long_edge_density"),
        "chamfer": entry.get("chamfer"),
        "triangles": entry.get("triangles"),
        "strategy": entry.get("strategy"),
    }


def update_manifest_in_place(manifest: list[dict], sample_id: str, new_result: dict) -> None:
    for i, entry in enumerate(manifest):
        if entry.get("sample_id") == sample_id:
            preserved_overrides = entry.get("applied_overrides")
            entry.update(new_result)
            if "applied_overrides" not in new_result and preserved_overrides:
                entry["applied_overrides"] = preserved_overrides
            return
    manifest.append({"sample_id": sample_id, **new_result})


def sync_to_frontend() -> None:
    if FRONTEND_OUT_DIR.exists():
        for stl in OUT_DIR.glob("*_recon_noisy.stl"):
            shutil.copy2(stl, FRONTEND_OUT_DIR / stl.name)
        if MANIFEST_PATH.exists():
            shutil.copy2(MANIFEST_PATH, FRONTEND_OUT_DIR / "manifest.json")
    if GEOMETRY_VIEWS_DIR.exists():
        FRONTEND_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
        for png in GEOMETRY_VIEWS_DIR.glob("*.png"):
            shutil.copy2(png, FRONTEND_GEOMETRY_DIR / png.name)


def render_all_geometry_views(manifest: list[dict], target_ids: set[str] | None = None) -> None:
    """Render the canonical clean geometry view for every successful sample.

    This is independent of any LLM call: just the cohesion grid (depth +
    silhouette per view) emitted to GEOMETRY_VIEWS_DIR. Cheap, deterministic,
    and re-runnable any time.

    If `target_ids` is provided, only those samples are (re-)rendered. Use this
    to keep tuner runs scoped to the highlighted test set without re-rendering
    every sample on disk.
    """
    GEOMETRY_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if not entry.get("success"):
            continue
        sid = entry["sample_id"]
        if target_ids is not None and sid not in target_ids:
            continue
        stl_name = entry.get("recon_noisy_stl")
        if not stl_name:
            continue
        stl_path = OUT_DIR / stl_name
        if not stl_path.exists():
            continue
        out_path = GEOMETRY_VIEWS_DIR / f"{sid}_geometry.png"
        try:
            render_cohesion_grid(stl_path, out_path)
        except Exception as exc:
            print(f"  geometry render failed for {sid}: {exc}")


def pick_global_samples(manifest: list[dict], n: int) -> list[dict]:
    failing = [e for e in manifest if e.get("success") and not is_acceptable(metrics_of(e))]
    failing.sort(key=lambda e: e.get("boundary_edges", 0) / max(e.get("triangles", 1), 1), reverse=True)
    if len(failing) <= n:
        return failing
    # spread across worst, median, near-passing
    quartiles = [failing[0],
                 failing[len(failing) // 4],
                 failing[len(failing) // 2],
                 failing[3 * len(failing) // 4],
                 failing[-1]]
    seen = set()
    pick = []
    for e in quartiles:
        if e["sample_id"] not in seen:
            pick.append(e)
            seen.add(e["sample_id"])
        if len(pick) == n:
            break
    for e in failing:
        if len(pick) == n:
            break
        if e["sample_id"] not in seen:
            pick.append(e)
            seen.add(e["sample_id"])
    return pick


def global_pass(client: anthropic.Anthropic, manifest: list[dict], log: list[dict],
                overrides_log: dict, n_samples: int = 4) -> dict:
    print(f"\n=== global pass: picking {n_samples} representative non-watertight samples ===")
    picks = pick_global_samples(manifest, n_samples)
    if not picks:
        print("nothing to tune in global pass")
        return {}

    image_paths: list[Path] = []
    blurbs: list[str] = []
    for entry in picks:
        sid = entry["sample_id"]
        stl = OUT_DIR / entry["recon_noisy_stl"]
        ply = ply_path_for(sid)
        png = RENDER_DIR / f"global_{sid}.png"
        try:
            render_pair_six_view_grid(ply, stl, png)
        except Exception as exc:
            print(f"  render failed for {sid}: {exc}")
            continue
        image_paths.append(png)
        blurbs.append(f"- {sid}: strategy={entry.get('strategy')} watertight={entry.get('watertight')} "
                      f"boundary_edges={entry.get('boundary_edges')} long_edge_density={entry.get('long_edge_density')} "
                      f"chamfer={entry.get('chamfer')}")

    if not image_paths:
        print("no images rendered, aborting global pass")
        return {}

    defaults = read_default_params()
    user_text = (
        "Current shared defaults:\n"
        + json.dumps({"RECON_PARAMS": serialize_for_prompt(defaults.get("RECON_PARAMS", {})),
                      "SCORE_PARAMS": serialize_for_prompt(defaults.get("SCORE_PARAMS", {})),
                      "HOLE_FILL_PARAMS": serialize_for_prompt(defaults.get("HOLE_FILL_PARAMS", {}))}, indent=2)
        + "\n\nSamples shown:\n"
        + "\n".join(blurbs)
        + "\n\nPropose new shared defaults to apply across all samples."
    )
    print(f"  calling {MODEL_ID} with {len(image_paths)} images...")
    try:
        response = call_opus(client, GLOBAL_SYSTEM_PROMPT, user_text, image_paths, max_tokens=1500)
    except Exception as exc:
        print(f"  opus call failed: {exc}")
        log.append({"phase": "global", "error": str(exc)})
        return {}

    diag = response.get("diagnosis", "<no diagnosis>")
    print(f"  diagnosis: {diag}")
    new_overrides = {
        "RECON_PARAMS": response.get("RECON_PARAMS") or {},
        "SCORE_PARAMS": response.get("SCORE_PARAMS") or {},
        "HOLE_FILL_PARAMS": response.get("HOLE_FILL_PARAMS") or {},
    }
    print(f"  proposed defaults: {json.dumps(new_overrides, indent=2)}")
    log.append({"phase": "global", "samples_shown": [e["sample_id"] for e in picks],
                "diagnosis": diag, "overrides": new_overrides,
                "expected_improvement": response.get("expected_improvement")})

    if not any(new_overrides[k] for k in ("RECON_PARAMS", "SCORE_PARAMS", "HOLE_FILL_PARAMS")):
        print("  model proposed no changes")
        return {}

    print(f"\n  re-running all {len(manifest)} samples with new shared defaults...")
    samples_to_rerun = [e for e in manifest if e.get("success")]
    improved_count = 0
    regressed_count = 0
    for entry in samples_to_rerun:
        sid = entry["sample_id"]
        ply = f"{sid}.ply"
        baseline_metrics = metrics_of(entry)
        result = run_recon(sid, ply, new_overrides)
        if not result.get("success"):
            print(f"    {sid}: REGEN FAILED ({result.get('error')})")
            continue
        new_metrics = metrics_of(result)
        if quality_score(new_metrics) > quality_score(baseline_metrics):
            update_manifest_in_place(manifest, sid, result)
            improved_count += 1
            tag = "BEST"
        else:
            regressed_count += 1
            tag = "kept baseline"
        print(f"    {sid}: wt={new_metrics['watertight']} bnd={new_metrics['boundary_edges']} chamfer={new_metrics['chamfer']}  [{tag}]")

    print(f"  global pass net: {improved_count} improved, {regressed_count} kept baseline")
    overrides_log["_global"] = new_overrides
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    OVERRIDES_PATH.write_text(json.dumps(overrides_log, indent=2))
    LOG_PATH.write_text(json.dumps(log, indent=2))
    return new_overrides


def per_sample_pass(client: anthropic.Anthropic, manifest: list[dict], log: list[dict],
                    overrides_log: dict, max_iters: int) -> None:
    global_overrides = overrides_log.get("_global", {"RECON_PARAMS": {}, "SCORE_PARAMS": {}, "HOLE_FILL_PARAMS": {}})
    failing = [e for e in manifest if e.get("success") and not is_acceptable(metrics_of(e))]
    print(f"\n=== per-sample pass: {len(failing)} samples still failing ===")
    if not failing:
        return

    defaults = read_default_params()

    for entry in failing:
        sid = entry["sample_id"]
        ply_name = f"{sid}.ply"
        baseline_metrics = metrics_of(entry)
        print(f"\n  {sid}: starting wt={baseline_metrics['watertight']} bnd={baseline_metrics['boundary_edges']} chamfer={baseline_metrics['chamfer']}")

        # Track the best attempt across all iterations: the manifest entry as
        # produced by run_recon, the overrides used, and the metrics. Start
        # with the baseline (the entry already in the manifest from global pass).
        best_record = {
            "metrics": baseline_metrics,
            "overrides": deepcopy(global_overrides),
            "result": deepcopy(entry),
        }

        attempt_history: list[dict] = []  # what we tell the model on the next call
        applied = deepcopy(global_overrides)
        last_applied = deepcopy(global_overrides)  # tracks which overrides last wrote the STL on disk

        for it in range(1, max_iters + 1):
            stl = OUT_DIR / best_record["result"]["recon_noisy_stl"]
            ply = ply_path_for(sid)
            png = RENDER_DIR / f"{sid}_iter{it - 1}.png"
            try:
                render_pair_six_view_grid(ply, stl, png)
            except Exception as exc:
                print(f"    render failed: {exc}")
                break

            effective = merge_overrides({"RECON_PARAMS": serialize_for_prompt(defaults.get("RECON_PARAMS", {})),
                                         "SCORE_PARAMS": serialize_for_prompt(defaults.get("SCORE_PARAMS", {})),
                                         "HOLE_FILL_PARAMS": serialize_for_prompt(defaults.get("HOLE_FILL_PARAMS", {}))},
                                        applied)

            history_blurb = ""
            if attempt_history:
                history_blurb = (
                    "\nPrevious attempts in this loop (do not repeat ineffective changes):\n"
                    + json.dumps(attempt_history, indent=2)
                )

            user_text = (
                f"Sample id: {sid}\n"
                f"Current effective parameters (defaults + global overrides + previous per-sample overrides):\n"
                f"{json.dumps(effective, indent=2)}\n\n"
                f"Best-so-far metrics shown in image: {json.dumps(best_record['metrics'], indent=2)}"
                f"{history_blurb}\n\n"
                f"Propose overrides to make THIS sample watertight. If the previous attempt regressed "
                f"(more boundary edges than baseline), reverse course."
            )
            try:
                response = call_opus(client, PER_SAMPLE_SYSTEM_PROMPT, user_text, [png], max_tokens=1500)
            except Exception as exc:
                print(f"    opus call failed: {exc}")
                log.append({"phase": "per_sample", "sample_id": sid, "iter": it, "error": str(exc)})
                break

            diag = response.get("diagnosis", "<no diagnosis>")
            new = {
                "RECON_PARAMS": response.get("RECON_PARAMS") or {},
                "SCORE_PARAMS": response.get("SCORE_PARAMS") or {},
                "HOLE_FILL_PARAMS": response.get("HOLE_FILL_PARAMS") or {},
            }
            print(f"    iter {it}: {diag}")
            print(f"      proposed: {json.dumps(new)}")
            attempt_overrides = merge_overrides(applied, new)

            result = run_recon(sid, ply_name, attempt_overrides)
            if not result.get("success"):
                print(f"      regen failed: {result.get('error')}")
                log.append({"phase": "per_sample", "sample_id": sid, "iter": it,
                            "diagnosis": diag, "overrides": new, "error": result.get("error")})
                attempt_history.append({"iter": it, "diagnosis": diag, "overrides_applied": new,
                                        "outcome": "regen_failed"})
                continue
            last_applied = attempt_overrides  # disk now holds STL for these overrides
            new_metrics = metrics_of(result)
            improved = quality_score(new_metrics) > quality_score(best_record["metrics"])
            tag = "BEST" if improved else "WORSE"
            print(f"      -> wt={new_metrics['watertight']} bnd={new_metrics['boundary_edges']} chamfer={new_metrics['chamfer']}  [{tag}]")

            attempt_history.append({"iter": it, "diagnosis": diag, "overrides_applied": new,
                                    "outcome_metrics": new_metrics, "improved": improved})
            log.append({"phase": "per_sample", "sample_id": sid, "iter": it,
                        "diagnosis": diag, "overrides": new, "metrics": new_metrics, "improved": improved})

            if improved:
                best_record = {"metrics": new_metrics, "overrides": attempt_overrides, "result": result}
                applied = attempt_overrides

            if is_acceptable(best_record["metrics"]):
                print(f"      -> acceptable best so far, stopping for this sample")
                break

        # If the last iter we ran wasn't the BEST one, the STL on disk is stale.
        # Re-run with the best overrides so the canonical _recon_noisy.stl matches
        # the metrics we're about to commit to the manifest.
        if best_record["overrides"] != last_applied:
            print(f"      regenerating canonical STL with best overrides (last on disk was a regression)")
            regen = run_recon(sid, ply_name, best_record["overrides"])
            if regen.get("success"):
                best_record["result"] = regen
                best_record["metrics"] = metrics_of(regen)
            else:
                print(f"      regen-best failed: {regen.get('error')} -- STL on disk may be stale")

        # Commit the BEST result for this sample (rollback if last iter regressed).
        update_manifest_in_place(manifest, sid, best_record["result"])
        if best_record["overrides"] != global_overrides and best_record["overrides"]:
            overrides_log[sid] = best_record["overrides"]
        elif sid in overrides_log:
            overrides_log.pop(sid, None)

        OVERRIDES_PATH.write_text(json.dumps(overrides_log, indent=2))
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        LOG_PATH.write_text(json.dumps(log, indent=2))


def cohesion_pass(client: anthropic.Anthropic, manifest: list[dict], log: list[dict],
                  overrides_log: dict, max_iters: int, target_score: int = 88) -> None:
    """Vision-graded geometry-representation tuning.

    Operates on every successful manifest entry (filtered upstream by --only).
    For each sample we render the cohesion grid (recon-only depth + silhouette,
    six views), ask Claude Opus 4.7 to grade COHESION 0-100, then iterate by
    applying the model's suggested overrides. Best-of-N is kept by cohesion
    score; the canonical STL on disk and the saved geometry-view PNG always
    correspond to the highest-scoring iteration.
    """
    global_overrides = overrides_log.get("_global",
                                         {"RECON_PARAMS": {}, "SCORE_PARAMS": {}, "HOLE_FILL_PARAMS": {}})
    successful = [e for e in manifest if e.get("success")]
    print(f"\n=== cohesion pass: vision-grading {len(successful)} samples ===")
    if not successful:
        return

    defaults = read_default_params()
    COHESION_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_VIEWS_DIR.mkdir(parents=True, exist_ok=True)

    # New cohesion contract: keep through-holes IN, accept small surface gaps,
    # let the downstream gpt-image-2 cleanup pass make the orthographic views
    # look polished. We bake the conservative capped 2-pass fill on (closes
    # noise loops only, never bridges far across) but DO NOT enable meshy_close
    # because it still occasionally bridges through-bores even with the
    # edge-guard, and the image-gen pass handles cosmetic gap cleanup better
    # than mesh repair does.
    cohesion_baseline_overrides = {
        "RECON_PARAMS": {},
        "SCORE_PARAMS": {},
        "HOLE_FILL_PARAMS": {
            "meshy_close": False,
            "enable_in_noisy": True,
        },
    }

    for entry in successful:
        sid = entry["sample_id"]
        ply_name = f"{sid}.ply"

        # Re-run reconstruction with the safer cohesion baseline (capped fill,
        # NO meshy_close) so the initial render matches the actual policy in
        # effect, not whatever leftover STL is on disk from a previous run.
        # Stack: global defaults <- previous per-sample (preserves learned tuning)
        # <- cohesion baseline (turns off meshy_close, turns on capped 2-pass fill).
        previous_per_sample = overrides_log.get(sid, {})
        starting_overrides = merge_overrides(global_overrides, previous_per_sample)
        starting_overrides = merge_overrides(starting_overrides, cohesion_baseline_overrides)
        baseline_result = run_recon(sid, ply_name, starting_overrides)
        if not baseline_result.get("success"):
            print(f"\n  {sid}: cohesion baseline regen failed: {baseline_result.get('error')}")
            continue
        update_manifest_in_place(manifest, sid, baseline_result)
        entry = baseline_result

        stl = OUT_DIR / entry["recon_noisy_stl"]
        baseline_png = COHESION_RENDER_DIR / f"{sid}_iter0.png"
        try:
            render_cohesion_grid(stl, baseline_png)
        except Exception as exc:
            print(f"\n  {sid}: baseline render failed: {exc}")
            continue

        try:
            baseline = call_opus(
                client,
                COHESION_SYSTEM_PROMPT,
                f"Sample id: {sid}\n"
                f"Iteration 0 (baseline). Grade this geometry representation and propose adjustments.",
                [baseline_png],
                max_tokens=1500,
            )
        except Exception as exc:
            print(f"\n  {sid}: baseline opus call failed: {exc}")
            log.append({"phase": "cohesion", "sample_id": sid, "iter": 0, "error": str(exc)})
            continue

        baseline_score = int(baseline.get("cohesion_score", 0) or 0)
        print(f"\n  {sid}: baseline cohesion={baseline_score}  symmetry={baseline.get('symmetry')}  holes={baseline.get('hole_count')}")
        print(f"    diag: {(baseline.get('diagnosis') or '')[:160]}")

        log.append({"phase": "cohesion", "sample_id": sid, "iter": 0,
                    "score": baseline_score, "diagnosis": baseline.get("diagnosis"),
                    "symmetry": baseline.get("symmetry"), "hole_count": baseline.get("hole_count"),
                    "metrics": metrics_of(entry)})

        applied = deepcopy(starting_overrides)
        last_applied = deepcopy(starting_overrides)

        best_record = {
            "score": baseline_score,
            "overrides": deepcopy(applied),
            "result": deepcopy(entry),
            "render_path": baseline_png,
            "diagnosis": baseline.get("diagnosis", ""),
            "symmetry": baseline.get("symmetry", "none"),
            "hole_count": int(baseline.get("hole_count") or 0),
            "expected_improvement": baseline.get("expected_improvement", ""),
        }

        next_response = baseline
        attempt_history: list[dict] = [{"iter": 0, "score": baseline_score,
                                        "diagnosis": baseline.get("diagnosis")}]

        if baseline_score >= target_score:
            print(f"    baseline already at target ({baseline_score} >= {target_score}), no iteration needed")
        else:
            for it in range(1, max_iters + 1):
                new = {
                    "RECON_PARAMS": next_response.get("RECON_PARAMS") or {},
                    "SCORE_PARAMS": next_response.get("SCORE_PARAMS") or {},
                    "HOLE_FILL_PARAMS": next_response.get("HOLE_FILL_PARAMS") or {},
                }
                if not any(new.values()):
                    print(f"    iter {it}: model suggested no parameter changes, stopping")
                    break
                attempt_overrides = merge_overrides(applied, new)
                print(f"    iter {it}: applying {json.dumps(new)}")

                result = run_recon(sid, ply_name, attempt_overrides)
                if not result.get("success"):
                    print(f"      regen failed: {result.get('error')}")
                    log.append({"phase": "cohesion", "sample_id": sid, "iter": it,
                                "overrides": new, "error": result.get("error")})
                    attempt_history.append({"iter": it, "outcome": "regen_failed",
                                            "overrides_applied": new})
                    break
                last_applied = attempt_overrides

                new_stl = OUT_DIR / result["recon_noisy_stl"]
                new_png = COHESION_RENDER_DIR / f"{sid}_iter{it}.png"
                try:
                    render_cohesion_grid(new_stl, new_png)
                except Exception as exc:
                    print(f"      render failed: {exc}")
                    break

                history_blurb = (
                    "\nIteration history (do not repeat ineffective changes):\n"
                    + json.dumps(attempt_history, indent=2)
                    + f"\nBest cohesion so far: {best_record['score']}"
                )
                user_text = (
                    f"Sample id: {sid}\nIteration {it}.{history_blurb}\n"
                    f"Reassess this new render and propose the next adjustments. "
                    f"If the score regressed, reverse course."
                )
                try:
                    response = call_opus(
                        client, COHESION_SYSTEM_PROMPT, user_text, [new_png], max_tokens=1500,
                    )
                except Exception as exc:
                    print(f"      opus call failed: {exc}")
                    log.append({"phase": "cohesion", "sample_id": sid, "iter": it, "error": str(exc)})
                    break

                new_score = int(response.get("cohesion_score", 0) or 0)
                improved = new_score > best_record["score"]
                tag = "BEST" if improved else "no improvement"
                print(f"      -> cohesion={new_score}  symmetry={response.get('symmetry')}  holes={response.get('hole_count')}  [{tag}]")
                print(f"      diag: {(response.get('diagnosis') or '')[:160]}")

                attempt_history.append({"iter": it, "score": new_score,
                                        "overrides_applied": new,
                                        "diagnosis": response.get("diagnosis"),
                                        "improved": improved})
                log.append({"phase": "cohesion", "sample_id": sid, "iter": it,
                            "score": new_score, "diagnosis": response.get("diagnosis"),
                            "symmetry": response.get("symmetry"),
                            "hole_count": response.get("hole_count"),
                            "overrides": new, "metrics": metrics_of(result),
                            "improved": improved})

                if improved:
                    best_record = {
                        "score": new_score,
                        "overrides": deepcopy(attempt_overrides),
                        "result": result,
                        "render_path": new_png,
                        "diagnosis": response.get("diagnosis", ""),
                        "symmetry": response.get("symmetry", "none"),
                        "hole_count": int(response.get("hole_count") or 0),
                        "expected_improvement": response.get("expected_improvement", ""),
                    }
                    applied = attempt_overrides

                next_response = response
                if best_record["score"] >= target_score:
                    print(f"      cohesion >= {target_score}, stopping")
                    break

        # Make sure the canonical STL on disk corresponds to the best iteration.
        if best_record["overrides"] != last_applied:
            print(f"    regenerating canonical STL with best overrides (last on disk regressed)")
            regen = run_recon(sid, ply_name, best_record["overrides"])
            if regen.get("success"):
                best_record["result"] = regen
                final_render = COHESION_RENDER_DIR / f"{sid}_final.png"
                try:
                    render_cohesion_grid(OUT_DIR / regen["recon_noisy_stl"], final_render)
                    best_record["render_path"] = final_render
                except Exception as exc:
                    print(f"      final render failed: {exc}")

        # Persist the canonical clean orthographic geometry view as the deliverable.
        final_geometry = GEOMETRY_VIEWS_DIR / f"{sid}_geometry.png"
        try:
            shutil.copy2(best_record["render_path"], final_geometry)
        except Exception as exc:
            print(f"    warn: failed to copy geometry view: {exc}")

        # Stash cohesion metadata on the manifest entry alongside topology metrics.
        result_with_meta = dict(best_record["result"])
        result_with_meta["cohesion"] = {
            "score": best_record["score"],
            "diagnosis": best_record["diagnosis"],
            "symmetry": best_record["symmetry"],
            "hole_count": best_record["hole_count"],
            "expected_improvement": best_record["expected_improvement"],
            "geometry_view": f"geometry_views/{sid}_geometry.png",
        }
        update_manifest_in_place(manifest, sid, result_with_meta)

        if best_record["overrides"] != global_overrides and any(best_record["overrides"].values()):
            overrides_log[sid] = best_record["overrides"]

        OVERRIDES_PATH.write_text(json.dumps(overrides_log, indent=2))
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        LOG_PATH.write_text(json.dumps(log, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["global", "per-sample", "cohesion", "hybrid"], default="hybrid",
                   help="hybrid = global -> cohesion (vision-graded geometry rep). "
                        "per-sample preserves the legacy watertight-driven loop.")
    p.add_argument("--max-iters", type=int, default=3,
                   help="max vision-guided regen iterations per sample in cohesion / per-sample passes")
    p.add_argument("--global-samples", type=int, default=4,
                   help="how many sample renders to show the model in the global pass")
    p.add_argument("--cohesion-target", type=int, default=88,
                   help="early-exit threshold for cohesion score (0-100)")
    p.add_argument("--no-sync", action="store_true", help="skip syncing STLs/manifest to frontend")
    p.add_argument("--only", help="comma-separated sample ids to restrict the run to")
    p.add_argument("--no-synth-clean", action="store_true",
                   help="skip the gpt-image-2 cleanup step (default is to run it on --only ids if "
                        "OPENAI_API_KEY is set)")
    p.add_argument("--synth-quality", default="high",
                   choices=["low", "medium", "high", "auto", "standard"],
                   help="gpt-image-2 quality tier for the cleanup step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    overrides_log: dict = {}
    if OVERRIDES_PATH.exists():
        overrides_log = json.loads(OVERRIDES_PATH.read_text())
    log: list[dict] = []
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text())
        except Exception:
            log = []

    client = anthropic.Anthropic()

    # If --only is set, restrict both passes to that subset of the manifest.
    working_manifest = manifest
    if args.only:
        wanted = set(s.strip() for s in args.only.split(",") if s.strip())
        working_manifest = [e for e in manifest if e.get("sample_id") in wanted]
        missing = wanted - {e["sample_id"] for e in working_manifest}
        if missing:
            print(f"WARNING: --only ids not found in manifest: {sorted(missing)}")
        print(f"--only restricting to {len(working_manifest)} samples: {[e['sample_id'] for e in working_manifest]}")

    if args.mode in ("global", "hybrid"):
        global_pass(client, working_manifest, log, overrides_log, n_samples=args.global_samples)

    if args.mode == "per-sample":
        per_sample_pass(client, working_manifest, log, overrides_log, args.max_iters)

    if args.mode in ("cohesion", "hybrid"):
        cohesion_pass(client, working_manifest, log, overrides_log,
                      args.max_iters, target_score=args.cohesion_target)

    # Push working_manifest changes back into the full manifest if filtered.
    if args.only:
        for upd in working_manifest:
            update_manifest_in_place(manifest, upd["sample_id"], upd)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    OVERRIDES_PATH.write_text(json.dumps(overrides_log, indent=2))
    LOG_PATH.write_text(json.dumps(log, indent=2))

    # Render the canonical clean ortho geometry view. When --only restricts the
    # run to a test set, ONLY re-render those samples; existing PNGs for samples
    # we didn't tune stay as-is (the user explicitly asked for tight scoping).
    target_geometry_ids: set[str] | None = None
    if args.only:
        target_geometry_ids = {e["sample_id"] for e in working_manifest if e.get("success")}
    render_all_geometry_views(manifest, target_ids=target_geometry_ids)

    # Synthesise the gpt-image-2 cleaned ortho views for any sample we tuned.
    # Cosmetic gaps in the recon get filled by the image model so the final
    # captured representation looks clean even when the underlying mesh is
    # (intentionally) not watertight.
    if not args.no_synth_clean and target_geometry_ids and os.environ.get("OPENAI_API_KEY"):
        synth_script = REPO / "scripts" / "synthesize_clean_views.py"
        synth_cmd = [PYTHON_BIN, str(synth_script),
                     "--only", ",".join(sorted(target_geometry_ids)),
                     "--quality", args.synth_quality]
        if args.no_sync:
            synth_cmd.append("--no-sync")
        print(f"\n=== running gpt-image-2 cleanup on {len(target_geometry_ids)} samples ===")
        rc = subprocess.call(synth_cmd, cwd=str(REPO))
        if rc != 0:
            print(f"  WARN: synthesize_clean_views exited with code {rc}")
    elif not args.no_synth_clean and not target_geometry_ids:
        print("  (skipping gpt-image-2 cleanup: no --only target ids)")
    elif not args.no_synth_clean and not os.environ.get("OPENAI_API_KEY"):
        print("  (skipping gpt-image-2 cleanup: OPENAI_API_KEY not set)")

    if not args.no_sync:
        sync_to_frontend()

    # Report against the working subset when --only is used, otherwise the full manifest.
    report_pool = working_manifest if args.only else manifest
    final_total = sum(1 for e in report_pool if e.get("success"))
    final_wt = sum(1 for e in report_pool if is_acceptable(metrics_of(e)))
    final_floor = final_total - final_wt
    scored = [e for e in report_pool if isinstance(e.get("cohesion"), dict)]
    avg_cohesion = (sum(e["cohesion"].get("score", 0) for e in scored) / len(scored)) if scored else None
    scope = f"--only subset ({len(report_pool)})" if args.only else "full manifest"
    print(f"\n=== done ({scope}) ===")
    print(f"    watertight:                   {final_wt}/{final_total}")
    print(f"    improved-but-not-watertight:  {final_floor}/{final_total} (best-of-N kept)")
    if avg_cohesion is not None:
        print(f"    avg cohesion (vision):        {avg_cohesion:.1f}/100  ({len(scored)} graded)")
    print()
    for e in sorted(report_pool, key=lambda x: x.get("sample_id", "")):
        if not e.get("success"):
            continue
        sid = e["sample_id"]
        wt = "WATERTIGHT " if e.get("watertight") else "floor      "
        bnd = e.get("boundary_edges")
        chamfer = e.get("chamfer")
        strat = e.get("strategy")
        coh = e.get("cohesion") or {}
        coh_str = f"  coh={coh.get('score'):>3}  sym={coh.get('symmetry')}  holes={coh.get('hole_count')}" if coh else ""
        print(f"    {wt} {sid:24s}  bnd={bnd:>5}  chamfer={chamfer:.5f}  strat={strat}{coh_str}")


if __name__ == "__main__":
    main()
