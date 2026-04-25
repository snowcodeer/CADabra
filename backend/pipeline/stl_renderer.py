"""STL to 6-view PNG grid renderer (Person 1 deliverable).

Takes an .stl mesh file and produces a single 3072x1024 PNG containing 6
orthographic renders (one per cube-face direction) paired with their depth
maps. The PNG is the handoff artifact consumed by the downstream Claude
Vision stage.

See RENDER_CONTEXT.md for the full specification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

PANEL_SIZE = 512
MESH_COLOR = "#AAAAAA"
BG_DEPTH = (40, 40, 40)

CAMERA_VIEWS: dict[str, dict[str, tuple[float, float, float]]] = {
    "+Z": {"position": (0.0, 0.0, 1.0), "up": (0.0, 1.0, 0.0)},
    "-Z": {"position": (0.0, 0.0, -1.0), "up": (0.0, 1.0, 0.0)},
    "+X": {"position": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
    "-X": {"position": (-1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
    "+Y": {"position": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)},
    "-Y": {"position": (0.0, -1.0, 0.0), "up": (0.0, 0.0, 1.0)},
}

GRID_ORDER: list[list[str]] = [
    ["+Z", "+X", "+Y"],
    ["-Z", "-X", "-Y"],
]

if sys.platform.startswith("linux"):
    pv.start_xvfb()


def load_mesh(stl_path: str | Path) -> pv.PolyData:
    raise NotImplementedError


def render_view(mesh: pv.PolyData, direction: str) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def build_grid(
    renders: dict[str, np.ndarray],
    depth_maps: dict[str, np.ndarray],
) -> Image.Image:
    raise NotImplementedError


def render_stl_to_grid(stl_path: str | Path, output_path: str | Path) -> Path:
    raise NotImplementedError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stl_renderer",
        description="Render an STL mesh into a 3072x1024 6-view PNG grid.",
    )
    parser.add_argument("stl_path", type=Path, help="Input .stl mesh file")
    parser.add_argument("output_path", type=Path, help="Output .png file path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    render_stl_to_grid(args.stl_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
