#!/usr/bin/env python3
"""Open an STL file in an interactive PyVista 3D viewer.

Native desktop window; no web server, no browser. Mirrors the lighting
rig used by ``backend.pipeline.stl_renderer`` so what you see here
matches what the 6-view grid renderer captures.

Controls (PyVista trackball default):
    Left-click + drag   -> rotate
    Right-click + drag  -> pan
    Scroll wheel        -> zoom

Usage:
    python scripts/view3d.py path/to/file.stl
    python scripts/view3d.py path/to/file.stl --smooth
"""

import argparse
import sys
from pathlib import Path

import pyvista as pv


MESH_COLOR = "#AAAAAA"
BG_COLOR = "white"


def _setup_lights(plotter: pv.Plotter) -> None:
    """Three-point rig identical to the offscreen renderer."""
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


def view_stl(stl_path: Path, *, smooth_shading: bool = False) -> None:
    mesh = pv.read(str(stl_path))

    plotter = pv.Plotter(window_size=(1100, 800))
    plotter.set_background(BG_COLOR)
    _setup_lights(plotter)

    plotter.add_mesh(
        mesh,
        color=MESH_COLOR,
        smooth_shading=smooth_shading,
        ambient=0.15,
        diffuse=0.85,
        specular=0.0,
    )

    plotter.add_axes()
    plotter.enable_trackball_style()

    plotter.add_text(
        f"STL Viewer — {stl_path.name}",
        position="upper_edge",
        font_size=12,
        color="black",
    )

    plotter.reset_camera()
    plotter.show()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Interactive PyVista STL viewer.")
    p.add_argument("stl", type=Path, help="Path to an .stl file.")
    p.add_argument(
        "--smooth",
        action="store_true",
        help="Interpolated vertex normals (can show banding on faceted CAD).",
    )
    ns = p.parse_args(argv[1:])

    stl_path = ns.stl.resolve()
    if not stl_path.exists():
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 2

    view_stl(stl_path, smooth_shading=ns.smooth)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
