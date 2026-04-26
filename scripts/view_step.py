#!/usr/bin/env python3
"""Open a STEP file in an interactive PyVista 3D viewer.

PyVista can't read STEP directly (no OCC backend in VTK), so we route
through CadQuery: ``cq.importers.importStep`` → tessellate the
B-Rep solid → wrap as a ``pv.PolyData`` → render with the same lighting
rig used by ``view3d.py`` (matching the offscreen ortho renderer).

Tessellation tolerance defaults to 0.1 mm — fine enough that arcs and
circles look smooth at typical zoom levels but not so fine that loading
a complex part takes seconds.

Usage:
    python scripts/view_step.py path/to/file.step [--tol 0.05]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cadquery as cq
import numpy as np
import pyvista as pv


MESH_COLOR = "#AAAAAA"
BG_COLOR = "white"


def _setup_lights(plotter: pv.Plotter) -> None:
    plotter.remove_all_lights()
    key = pv.Light(position=(1.0, 1.0, 2.0), focal_point=(0.0, 0.0, 0.0))
    key.intensity = 0.8
    plotter.add_light(key)
    fill = pv.Light(position=(-1.0, -0.5, 0.5), focal_point=(0.0, 0.0, 0.0))
    fill.intensity = 0.3
    plotter.add_light(fill)
    ambient = pv.Light(light_type="headlight")
    ambient.intensity = 0.2
    plotter.add_light(ambient)


def step_to_polydata(step_path: Path, tol: float) -> pv.PolyData:
    """Load STEP via CadQuery, tessellate, and return a PyVista PolyData.

    CadQuery 2.5+ exposes ``Shape.tessellate(tolerance)`` which returns
    (vertices, triangles). We pack those into a VTK-friendly ``faces``
    array: each face is [n, i0, i1, ..., i(n-1)]; for triangles n=3.
    """
    shape = cq.importers.importStep(str(step_path))
    # importStep returns a Workplane wrapping the imported solid(s).
    solid = shape.val()
    verts, tris = solid.tessellate(tol)
    points = np.array([(p.x, p.y, p.z) for p in verts], dtype=np.float64)
    faces = np.empty((len(tris), 4), dtype=np.int64)
    faces[:, 0] = 3
    for i, t in enumerate(tris):
        faces[i, 1:] = t
    return pv.PolyData(points, faces.flatten())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", help="Path to the .step file")
    parser.add_argument(
        "--tol", type=float, default=0.1,
        help="Tessellation tolerance in mm (default 0.1; smaller = smoother curves)",
    )
    args = parser.parse_args(argv[1:])

    step_path = Path(args.step).resolve()
    if not step_path.exists():
        print(f"STEP not found: {step_path}", file=sys.stderr)
        return 2

    print(f"loading {step_path.name}...")
    mesh = step_to_polydata(step_path, args.tol)
    print(f"  {mesh.n_points} points, {mesh.n_cells} triangles")

    plotter = pv.Plotter(window_size=(1024, 768),
                         title=f"STEP — {step_path.name}")
    plotter.set_background(BG_COLOR)
    _setup_lights(plotter)
    plotter.add_mesh(
        mesh, color=MESH_COLOR, smooth_shading=True,
        show_edges=False, ambient=0.2, diffuse=0.7, specular=0.3,
    )
    plotter.add_axes()
    plotter.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
