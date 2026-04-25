from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

PANEL_SIZE = 512

VIEW_SPECS = [
    ("Top", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("Front", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Left", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("Bottom", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("Back", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("Right", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
]


def _mesh_to_polydata(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int64)
    return pv.PolyData(vertices, faces)


def _render_rgb(poly: pv.PolyData, direction: tuple[float, float, float], view_up: tuple[float, float, float]) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=[PANEL_SIZE, PANEL_SIZE])
    plotter.set_background("white")
    plotter.add_mesh(poly, color="#B0B0B0", smooth_shading=False, ambient=0.6, diffuse=0.8)
    plotter.enable_parallel_projection()
    center = np.array(poly.center)
    length = float(np.linalg.norm(np.array(poly.bounds)[1::2] - np.array(poly.bounds)[::2]))
    cam_pos = center + np.array(direction) * max(length, 1.0)
    plotter.camera_position = [tuple(cam_pos), tuple(center), view_up]
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _render_depth(poly: pv.PolyData, direction: tuple[float, float, float], view_up: tuple[float, float, float]) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=[PANEL_SIZE, PANEL_SIZE])
    plotter.set_background("black")
    plotter.add_mesh(poly, color="white")
    plotter.enable_parallel_projection()
    center = np.array(poly.center)
    length = float(np.linalg.norm(np.array(poly.bounds)[1::2] - np.array(poly.bounds)[::2]))
    cam_pos = center + np.array(direction) * max(length, 1.0)
    plotter.camera_position = [tuple(cam_pos), tuple(center), view_up]
    plotter.show(auto_close=False)
    depth = plotter.get_image_depth(fill_value=np.nan)
    plotter.close()

    depth = np.nan_to_num(depth, nan=np.nanmax(depth[np.isfinite(depth)]) if np.isfinite(depth).any() else 0.0)
    d_min, d_max = float(np.min(depth)), float(np.max(depth))
    norm = np.zeros_like(depth) if d_max <= d_min else (depth - d_min) / (d_max - d_min)
    cmap = plt_colormap(norm)
    return (cmap[:, :, :3] * 255).astype(np.uint8)


def plt_colormap(norm_depth: np.ndarray) -> np.ndarray:
    # simple cool-warm mapping: blue far -> red near
    red = 1.0 - norm_depth
    blue = norm_depth
    green = 0.2 + 0.5 * (1.0 - np.abs(norm_depth - 0.5) * 2)
    return np.stack([red, green, blue, np.ones_like(red)], axis=-1)


def render_views_grid(mesh: o3d.geometry.TriangleMesh, output_dir: str | Path, input_id: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poly = _mesh_to_polydata(mesh)

    canvas = Image.new("RGB", (PANEL_SIZE * 4, PANEL_SIZE * 3), color="white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (name, direction, view_up) in enumerate(VIEW_SPECS):
        row, col_pair = divmod(idx, 3)
        x_rgb = col_pair * 2 * PANEL_SIZE
        y = row * PANEL_SIZE

        rgb = Image.fromarray(_render_rgb(poly, direction, view_up))
        depth = Image.fromarray(_render_depth(poly, direction, view_up))
        canvas.paste(rgb, (x_rgb, y))
        canvas.paste(depth, (x_rgb + PANEL_SIZE, y))

        draw.text((x_rgb + 10, y + 10), f"{name} RGB", fill="black", font=font)
        draw.text((x_rgb + PANEL_SIZE + 10, y + 10), f"{name} Depth", fill="black", font=font)

    output_path = output_dir / f"{input_id}_views.png"
    canvas.save(output_path)
    return output_path
