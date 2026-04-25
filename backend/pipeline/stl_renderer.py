"""STL to 6-view PNG grid renderer (Person 1 deliverable).

Takes an .stl mesh file and produces a single 3072x1024 PNG containing 6
orthographic renders (one per cube-face direction) paired with their depth
maps. The PNG is the handoff artifact consumed by the downstream Claude
Vision stage.

See RENDER_CONTEXT.md for the full specification.

Output sanity checks (per spec, "What good output looks like"):
    - Final image is exactly 3072x1024 pixels.
    - Top row is +Z, +X, +Y; bottom row is -Z, -X, -Y.
    - Each cell shows an RGB shaded render on the left and a colourised
      depth map on the right, both labelled with their direction.
    - Depth panels use the matplotlib `plasma` colormap with warm = near
      and cool = far. Background pixels are dark gray (40, 40, 40).
    - The mesh surface is rendered as neutral mid-gray (#AAAAAA) on a
      white background, with no perspective distortion.

CLI usage:
    python -m backend.pipeline.stl_renderer <input.stl> <output.png>
    python -m backend.pipeline.stl_renderer --verify <output.png>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import colormaps as _mpl_colormaps  # noqa: E402

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


def depth_to_colormap(
    depth: np.ndarray,
    background_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a raw depth buffer into a false-coloured RGB depth map.

    Background pixels (no surface hit) are painted with `BG_DEPTH`. Surface
    depths are normalised to 0-1 using only valid pixels, then inverted so
    near surfaces map to warm colours via matplotlib's ``plasma`` colormap.

    `background_mask` (preferred) is a boolean array, True where there is no
    surface. When omitted, the function falls back to detecting NaN/Inf and
    pixels at >= 1.0; in practice that fallback is unreliable because PyVista
    returns world-space depth, so callers should supply the mask explicitly.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if background_mask is None:
        background = ~np.isfinite(depth) | (depth >= 1.0)
    else:
        background = np.asarray(background_mask, dtype=bool)
        if background.shape != depth.shape:
            raise ValueError(
                f"background_mask shape {background.shape} does not match depth {depth.shape}"
            )
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

    cmap = _mpl_colormaps["plasma"]
    colored = cmap(normalised)[:, :3]
    out[surface] = (colored * 255.0).astype(np.uint8)
    return out


CELL_WIDTH = PANEL_SIZE * 2
CELL_HEIGHT = PANEL_SIZE
GRID_WIDTH = CELL_WIDTH * 3
GRID_HEIGHT = CELL_HEIGHT * 2


def _load_label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font) -> None:
    pad = 4
    try:
        bbox = draw.textbbox((x + pad, y + pad), text, font=font)
    except AttributeError:
        w, h = draw.textsize(text, font=font)
        bbox = (x + pad, y + pad, x + pad + w, y + pad + h)
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 200),
    )
    draw.text((x + pad, y + pad), text, fill="white", font=font)


def build_grid(
    renders: dict[str, np.ndarray],
    depth_maps: dict[str, np.ndarray],
) -> Image.Image:
    """Compose 6 RGB+Depth pairs into a single 3072x1024 PIL image."""
    missing = [d for row in GRID_ORDER for d in row if d not in renders or d not in depth_maps]
    if missing:
        raise ValueError(f"Missing renders/depths for directions: {missing}")

    canvas = Image.new("RGB", (GRID_WIDTH, GRID_HEIGHT), color="white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _load_label_font()

    for row_idx, row in enumerate(GRID_ORDER):
        for col_idx, direction in enumerate(row):
            cell_x = col_idx * CELL_WIDTH
            cell_y = row_idx * CELL_HEIGHT

            rgb_img = Image.fromarray(renders[direction]).resize(
                (PANEL_SIZE, PANEL_SIZE), Image.BILINEAR
            )
            depth_img = Image.fromarray(depth_maps[direction]).resize(
                (PANEL_SIZE, PANEL_SIZE), Image.NEAREST
            )

            canvas.paste(rgb_img, (cell_x, cell_y))
            canvas.paste(depth_img, (cell_x + PANEL_SIZE, cell_y))
            _draw_label(draw, cell_x, cell_y, f"{direction} RGB", font)
            _draw_label(draw, cell_x + PANEL_SIZE, cell_y, f"{direction} Depth", font)

    return canvas


def render_stl_to_grid(stl_path: str | Path, output_path: str | Path) -> Path:
    """End-to-end: STL path in, 3072x1024 PNG written to output_path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(stl_path)

    renders: dict[str, np.ndarray] = {}
    depth_maps: dict[str, np.ndarray] = {}
    for direction in CAMERA_VIEWS:
        rgb, raw_depth = render_view(mesh, direction)
        renders[direction] = rgb
        bg_mask = (rgb == 255).all(axis=-1)
        depth_maps[direction] = depth_to_colormap(raw_depth, background_mask=bg_mask)

    grid = build_grid(renders, depth_maps)
    if grid.size != (GRID_WIDTH, GRID_HEIGHT):
        raise RuntimeError(
            f"Unexpected grid size {grid.size}, want ({GRID_WIDTH}, {GRID_HEIGHT})"
        )

    grid.save(output_path, format="PNG")
    print(f"[stl_renderer] Wrote {output_path} ({grid.size[0]}x{grid.size[1]})")
    return output_path


def verify_grid(png_path: str | Path) -> bool:
    """Assert that a generated PNG matches the expected grid dimensions.

    Returns True on success and prints a short diagnostic line. Raises
    AssertionError with a clear message if the file is missing or the
    pixel size does not match (GRID_WIDTH, GRID_HEIGHT).
    """
    png_path = Path(png_path)
    if not png_path.is_file():
        raise AssertionError(f"Grid PNG not found: {png_path}")

    with Image.open(png_path) as img:
        size = img.size

    expected = (GRID_WIDTH, GRID_HEIGHT)
    if size != expected:
        raise AssertionError(
            f"Grid size mismatch for {png_path}: got {size}, expected {expected}"
        )

    print(f"[stl_renderer] OK {png_path} matches {expected[0]}x{expected[1]}")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stl_renderer",
        description="Render an STL mesh into a 3072x1024 6-view PNG grid.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing grid PNG matches the expected dimensions.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input .stl mesh file (or grid PNG when --verify is set)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        help="Output .png file path (omit when --verify is set)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        if args.output_path is not None:
            raise SystemExit("--verify takes a single PNG path; output_path not allowed")
        verify_grid(args.input_path)
        return 0

    if args.output_path is None:
        raise SystemExit("output_path is required when not using --verify")
    render_stl_to_grid(args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
