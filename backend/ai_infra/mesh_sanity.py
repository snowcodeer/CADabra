"""Lightweight STL diagnostics for comparing an original mesh to a rebuild.

Used after CadQuery export in ``face_roundtrip`` to surface obvious failures
(collapsed solids, tiny triangle count) before the user opens a 3D viewer.

All metrics are designed to be **scale-invariant** where possible so a
30 mm DeepCAD STL can be compared to a 100 mm normalised rebuild.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def load_mesh_clean(path: Path | str) -> o3d.geometry.TriangleMesh:
    """Read an STL and apply the same cleanup chain as ``preprocess``."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise ValueError(f"{path}: no triangles")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def stl_diagnostics(path: Path | str) -> dict[str, Any]:
    """Triangle/vertex counts, bbox, and two scale-invariant shape hints."""
    mesh = load_mesh_clean(path)
    t = int(len(mesh.triangles))
    v = int(len(mesh.vertices))
    bbox = mesh.get_axis_aligned_bounding_box()
    ext = np.asarray(bbox.get_extent(), dtype=float)
    longest = float(np.max(ext))
    shortest = float(np.min(ext))
    vol = float(np.prod(ext))
    # How "chunky" the AABB is vs a line (1 = cube, near 0 = very flat / needle)
    chunkiness = shortest / longest if longest > 0 else 0.0
    # Volume of AABB normalised by longest edge cubed — scale cancels.
    fill_ratio = vol / (longest**3) if longest > 0 else 0.0
    return {
        "triangles": t,
        "vertices": v,
        "extents_mm": (float(ext[0]), float(ext[1]), float(ext[2])),
        "longest_mm": longest,
        "chunkiness": chunkiness,
        "fill_ratio": fill_ratio,
    }


def compare_rebuild_sanity(input_stl: Path, output_stl: Path) -> list[str]:
    """Return warning lines if the rebuilt mesh looks nothing like the input.

    Heuristics only — false positives are possible on legitimately thin parts.
    """
    warnings: list[str] = []
    d_in = stl_diagnostics(input_stl)
    d_out = stl_diagnostics(output_stl)

    if d_out["longest_mm"] < 1e-5:
        warnings.append("Rebuilt mesh has essentially zero extent — export failed or empty solid.")
        return warnings

    if d_in["triangles"] > 200 and d_out["triangles"] < max(8, d_in["triangles"] * 0.03):
        warnings.append(
            f"Rebuilt mesh has far fewer triangles than the input "
            f"({d_out['triangles']} vs {d_in['triangles']}) — model may be over-simplified or wrong."
        )

    # Input is a recognisable 3D solid; output is a needle or ribbon
    if d_in["chunkiness"] > 0.03 and d_out["chunkiness"] < 0.002:
        warnings.append(
            f"Rebuilt part is much thinner than the input in relative terms "
            f"(chunkiness {d_out['chunkiness']:.4f} vs {d_in['chunkiness']:.4f}) — "
            "check generated CadQuery for a collapsed or wrong extrude."
        )

    # Same idea using normalised bbox "fullness"
    if d_in["fill_ratio"] > 0.005 and d_out["fill_ratio"] < d_in["fill_ratio"] * 0.02:
        warnings.append(
            "Rebuilt bounding box is much emptier than the input (normalised volume) — "
            "reconstruction may be a small fragment or blob."
        )

    return warnings


def print_rebuild_sanity(input_stl: Path, output_stl: Path) -> None:
    """Print a one-line summary and any warnings to stdout."""
    d_in = stl_diagnostics(input_stl)
    d_out = stl_diagnostics(output_stl)
    print(
        f"[sanity] input:  {d_in['triangles']} tris, "
        f"bbox {d_in['extents_mm'][0]:.2f}×{d_in['extents_mm'][1]:.2f}×{d_in['extents_mm'][2]:.2f} mm"
    )
    print(
        f"[sanity] output: {d_out['triangles']} tris, "
        f"bbox {d_out['extents_mm'][0]:.2f}×{d_out['extents_mm'][1]:.2f}×{d_out['extents_mm'][2]:.2f} mm"
    )
    for w in compare_rebuild_sanity(input_stl, output_stl):
        print(f"[sanity] WARNING: {w}")
