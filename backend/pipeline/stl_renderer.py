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

import matplotlib
import numpy as np
import pyvista as pv
from PIL import Image

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

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
    """Load an STL mesh with PyVista and report its bounding box."""
    stl_path = Path(stl_path)
    if not stl_path.is_file():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    mesh = pv.read(str(stl_path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh is empty: {stl_path}")

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    print(
        f"[stl_renderer] Loaded {stl_path.name}: "
        f"bounds X[{xmin:.3f}, {xmax:.3f}] "
        f"Y[{ymin:.3f}, {ymax:.3f}] "
        f"Z[{zmin:.3f}, {zmax:.3f}]"
    )
    return mesh


def render_view(mesh: pv.PolyData, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Render one orthographic view; return (rgb_uint8, raw_depth_float)."""
    if direction not in CAMERA_VIEWS:
        raise KeyError(f"Unknown camera direction: {direction}")

    view = CAMERA_VIEWS[direction]
    center = np.asarray(mesh.center, dtype=float)
    diag = float(np.linalg.norm(np.asarray(mesh.bounds).reshape(3, 2).ptp(axis=1)))
    distance = max(diag, 1.0) * 2.0
    cam_pos = tuple(center + np.asarray(view["position"], dtype=float) * distance)

    plotter = pv.Plotter(off_screen=True, window_size=[PANEL_SIZE, PANEL_SIZE])
    try:
        plotter.set_background("white")
        plotter.enable_lightkit()
        plotter.add_mesh(
            mesh,
            color=MESH_COLOR,
            smooth_shading=False,
            ambient=0.3,
            diffuse=0.8,
            specular=0.0,
        )
        plotter.enable_parallel_projection()
        plotter.camera_position = [cam_pos, tuple(center), view["up"]]
        plotter.reset_camera()

        rgb = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)
        depth = plotter.get_image_depth(fill_value=np.nan)
    finally:
        plotter.close()

    return rgb, np.asarray(depth, dtype=np.float32)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert a raw depth buffer into a false-coloured RGB depth map.

    Background pixels (no surface hit) are painted with `BG_DEPTH`. Surface
    depths are normalised to 0-1 using only valid pixels, then inverted so
    near surfaces map to warm colours via matplotlib's ``plasma`` colormap.
    """
    depth = np.asarray(depth, dtype=np.float32)
    background = ~np.isfinite(depth) | (depth >= 1.0)
    surface = ~background

    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    out[background] = BG_DEPTH

    if not np.any(surface):
        return out

    surface_vals = depth[surface]
    d_min = float(surface_vals.min())
    d_max = float(surface_vals.max())

    if d_max <= d_min:
        normalised = np.ones_like(surface_vals)
    else:
        normalised = (surface_vals - d_min) / (d_max - d_min)
        normalised = 1.0 - normalised

    cmap = cm.get_cmap("plasma")
    colored = cmap(normalised)[:, :3]
    out[surface] = (colored * 255.0).astype(np.uint8)
    return out


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
