#!/usr/bin/env python3
"""Generate a realistically-noisy STL by perturbing clean-mesh vertices along
their surface normals.

Why this exists
===============
The original pipeline made noisy STLs the hard way: sample clean mesh -> noisy
point cloud (SCAN_PROFILE) -> SOR cleanup -> Poisson/BPA reconstruction. That
chain introduced *structural* artifacts (random holes, missing chunks at
grazing angles, jagged through-bore edges) that aren't real scan noise, just
recon failures. The downstream CV stage spent most of its complexity fighting
those artifacts.

This script skips the detour. Each vertex of the clean STL is displaced along
its outward surface normal by an independent Gaussian sample. Topology is
preserved exactly (watertight in -> watertight out), so what reaches the CV
stage looks like a real metrology scan: clean silhouettes with gentle surface
ripple, not a Swiss-cheese mesh.

Usage:
    python scripts/noisify_stl.py                              # batch (4 demo ids)
    python scripts/noisify_stl.py --sample-id deepcadimg_000035
    python scripts/noisify_stl.py --sigma-frac 0.003           # louder noise

Output:
    backend/outputs/deepcad_pc_recon_stl/<sample_id>_recon_noisy.stl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pyacvd
import pyvista as pv
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_STL_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_selected_stl"
OUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_pc_recon_stl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

DEFAULT_SAMPLE_IDS = [
    "deepcadimg_000035",
    "deepcadimg_002354",
    "deepcadimg_117514",
    "deepcadimg_128105",
]

DEFAULT_SIGMA_FRAC = 0.0015
DEFAULT_TARGET_FACES = 3500  # bunny-style: visibly triangulated, ~few thousand tris


def _remesh_uniform(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Remesh a CAD mesh into a Stanford-bunny-style mesh: ~target_faces
    triangles, roughly equilateral, uniform edge length across the entire
    surface (NOT feature-preserving).

    Uses ACVD (Approximated Centroidal Voronoi Diagrams) via pyacvd. ACVD
    spreads `target_clusters` seeds uniformly across the surface area, then
    each output triangle is one Voronoi cell. The result is the canonical
    "uniform remesh" look you see on Stanford bunny renders — every triangle
    has roughly the same size whether it sits on a flat or a curve.

    Why NOT quadric decimation: quadric-decimation is feature-PRESERVING
    (keeps small triangles near corners, merges flats into huge ones). That
    produces a non-uniform remesh and doesn't have the bunny aesthetic.

    pyacvd needs enough source vertices to cluster, so we subdivide first.
    """
    # Step 1: dense subdivision so ACVD has enough verts to redistribute.
    upsample_target = max(target_faces * 6, 20_000)
    while len(mesh.faces) < upsample_target:
        prev = len(mesh.faces)
        mesh = mesh.subdivide()
        mesh.merge_vertices()
        if len(mesh.faces) == prev:  # safety: no progress
            break

    # Step 2: ACVD remesh via pyvista. The cluster count maps roughly to the
    # output vertex count; for a closed surface, triangles ≈ 2·vertices, so
    # request target_faces // 2 clusters.
    pv_mesh = pv.PolyData(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            np.asarray(mesh.faces, dtype=np.int64),
        ]).ravel(),
    )
    clus = pyacvd.Clustering(pv_mesh)
    # subdivide() inside pyacvd if vert density is uneven; harmless if it isn't
    clus.subdivide(2)
    clus.cluster(max(int(target_faces // 2), 256))
    remesh = clus.create_mesh()
    faces = remesh.faces.reshape(-1, 4)[:, 1:]  # strip the leading "3" cell-count

    out = trimesh.Trimesh(
        vertices=np.asarray(remesh.points),
        faces=np.asarray(faces),
        process=True,
    )
    out.merge_vertices()
    return out


def noisify(stl_path: Path, out_path: Path, sigma_frac: float, seed: int, target_faces: int) -> dict:
    # process=True merges coincident vertices. STL is a "triangle soup" format —
    # every face stores its own 3 independent vertex positions even when they
    # geometrically coincide. Without the merge, perturbing each vertex
    # independently splits the seams between adjacent faces and the mesh
    # explodes into a million tiny triangles.
    mesh = trimesh.load(stl_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"not a trimesh: {stl_path}")
    mesh.merge_vertices()  # belt-and-braces: ensure shared verts even if process didn't
    if len(mesh.vertices) == 0:
        raise ValueError(f"empty mesh: {stl_path}")

    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    # Stanford-bunny-style remesh: take the CAD mesh (a few huge flat faces)
    # and turn it into a uniformly-triangulated surface of ~target_faces
    # roughly equilateral triangles. Without this, vertex noise would only
    # deform the handful of CAD corners (star distortion) — and even with
    # uniform subdivision into millions of micro-triangles it just looks like
    # surface static. Bunny-class density (~3–5k tris) keeps the faceting
    # visually legible: you can see each triangle.
    mesh = _remesh_uniform(mesh, target_faces=target_faces)

    sigma = sigma_frac * bbox_diag

    rng = np.random.default_rng(seed)
    normals = mesh.vertex_normals  # smooth-shaded per-vertex normals (now well-defined)
    along = rng.normal(scale=sigma, size=len(mesh.vertices))[:, None] * normals

    out_mesh = trimesh.Trimesh(
        vertices=mesh.vertices + along,
        faces=mesh.faces,
        process=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(out_path)

    return {
        "source_stl": str(stl_path.relative_to(REPO_ROOT)),
        "noisy_stl": str(out_path.relative_to(REPO_ROOT)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_diag_mm": round(bbox_diag, 6),
        "sigma_frac": sigma_frac,
        "sigma_mm": round(sigma, 6),
        "seed": seed,
        "watertight": bool(out_mesh.is_watertight),
        "noise_model": "vertex_along_normal_gaussian",
    }


def update_manifest(updates: dict[str, dict]) -> None:
    if not MANIFEST_PATH.exists():
        print(f"  (manifest missing at {MANIFEST_PATH}; skipping manifest update)")
        return
    manifest = json.loads(MANIFEST_PATH.read_text())
    for row in manifest:
        sid = row.get("sample_id")
        if sid in updates:
            row["recon_noisy_stl"] = Path(updates[sid]["noisy_stl"]).name
            row["success"] = True
            row["noise_source"] = "vertex_perturbation"
            row["noise_stats"] = {
                "sigma_frac": updates[sid]["sigma_frac"],
                "sigma_mm": updates[sid]["sigma_mm"],
                "bbox_diag_mm": updates[sid]["bbox_diag_mm"],
                "noise_model": updates[sid]["noise_model"],
            }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sample-id", help="single sample; otherwise process the 4 demo ids")
    p.add_argument(
        "--sigma-frac",
        type=float,
        default=DEFAULT_SIGMA_FRAC,
        help="Gaussian sigma as a fraction of bbox-diag (default %(default)s "
             "= ~0.09mm on a 60mm part, metrology-scanner class)",
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="add to per-sample seed (default 0). Use to draw different "
             "noise realisations of the same parts.",
    )
    p.add_argument(
        "--target-faces",
        type=int,
        default=DEFAULT_TARGET_FACES,
        help="post-remesh triangle budget (default %(default)s, Stanford-bunny "
             "class). Lower = chunkier facets, higher = finer triangulation.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sample_ids = [args.sample_id] if args.sample_id else DEFAULT_SAMPLE_IDS

    updates: dict[str, dict] = {}
    for sid in sample_ids:
        stl_in = SRC_STL_DIR / f"{sid}.stl"
        stl_out = OUT_DIR / f"{sid}_recon_noisy.stl"
        if not stl_in.exists():
            print(f"  {sid}: source STL missing ({stl_in.relative_to(REPO_ROOT)})", file=sys.stderr)
            continue
        # deterministic per-sample seed
        seed = int(sid.split("_")[-1]) + args.seed_offset
        stats = noisify(stl_in, stl_out, args.sigma_frac, seed, args.target_faces)
        updates[sid] = stats
        print(
            f"  {sid}: sigma={stats['sigma_mm']:.3f}mm "
            f"(bbox_diag={stats['bbox_diag_mm']:.1f}mm) "
            f"V={stats['vertices']} F={stats['faces']} "
            f"watertight={stats['watertight']}"
        )

    if updates:
        update_manifest(updates)
        print(f"\n  noisified {len(updates)}/{len(sample_ids)} samples -> {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0 if updates else 1


if __name__ == "__main__":
    sys.exit(main())
