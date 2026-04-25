"""STL to 6-view PNG grid renderer (Person 1 deliverable).

Takes an .stl mesh file and produces a single 2400x1000 PNG containing
6 orthographic renders (one per cube-face direction) paired with their
depth maps. The PNG is the deterministic handoff artifact consumed by
Claude Vision; its layout is the binding contract between renderer and
prompt.

The canonical specification is GRID_FORMAT_SPEC.md. If anything in this
module diverges from that document the document wins; both must be
updated together because the vision prompt makes hard pixel-level
assumptions about the layout.

Output structure (top to bottom):
    Header bar  : 2400x40   (#1a1a1a, 14px bold)
                  Left:  "SCAN-TO-CAD  |  6-View Orthographic Grid"
                  Right: "Part ID: {part_id}"
    Grid area   : 2400x930  (3 cols x 2 rows of 800x465 cells)
                  Row 1: +Z top,    +X right,  +Y front
                  Row 2: -Z bottom, -X left,   -Y back
                  Each cell: 800x40 label bar + 400x400 RGB + 400x400
                  Depth + 800x25 sub-label bar, separated by 2px
                  #333333 borders.
    Legend bar  : 2400x30   (#111111, 12px) — depth colormap key

Render rules:
    - Orthographic projection (no perspective).
    - Mesh colour #AAAAAA, smooth shading, three-light rig
      (key 0.8 + fill 0.3 + headlight 0.2).
    - Per-view depth normalisation against object pixels only;
      flat surfaces map to the mid plasma value (0.5).
    - Depth background is #1e1e1e, RGB background is white.
    - Both panels are cropped to the object bbox + 10% padding and
      LANCZOS-resized to exactly 400x400.

CLI usage:
    python -m backend.pipeline.stl_renderer <input.stl> <output.png>
    python -m backend.pipeline.stl_renderer --part-id <id> <input.stl> <output.png>
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
RENDER_PANEL_SIZE = 400
CROP_PADDING_FRAC = 0.1
MESH_COLOR = "#AAAAAA"
BG_DEPTH = (30, 30, 30)
FLAT_SURFACE_RADIAL_HIGH = 1.0
FLAT_SURFACE_RADIAL_LOW = 0.92
FLAT_DEPTH_EPSILON = 1e-6

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

VIEW_LABELS: dict[str, str] = {
    "+Z": "+Z \u2014 top view      (base footprint + hole detection)",
    "+X": "+X \u2014 right side    (height + depth profile)",
    "+Y": "+Y \u2014 front view    (height + width profile)",
    "-Z": "-Z \u2014 bottom view   (through-hole confirmation)",
    "-X": "-X \u2014 left side     (asymmetry check)",
    "-Y": "-Y \u2014 back view     (rear feature check)",
}

LABEL_BAR_HEIGHT = 40
SUB_LABEL_HEIGHT = 25
CELL_WIDTH = RENDER_PANEL_SIZE * 2
CELL_RENDER_HEIGHT = RENDER_PANEL_SIZE
CELL_HEIGHT = LABEL_BAR_HEIGHT + CELL_RENDER_HEIGHT + SUB_LABEL_HEIGHT
GRID_WIDTH = CELL_WIDTH * 3
GRID_HEIGHT = CELL_HEIGHT * 2
BORDER_WIDTH = 2
BORDER_COLOR = (0x33, 0x33, 0x33)
BAR_BG_COLOR = (0x1A, 0x1A, 0x1A)
LEGEND_BG_COLOR = (0x11, 0x11, 0x11)
LABEL_FONT_SIZE = 18
SUB_LABEL_FONT_SIZE = 13
HEADER_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12

HEADER_HEIGHT = 40
LEGEND_HEIGHT = 30
HEADER_PAD_X = 16
FULL_WIDTH = GRID_WIDTH
FULL_HEIGHT = HEADER_HEIGHT + GRID_HEIGHT + LEGEND_HEIGHT

HEADER_TITLE = "SCAN-TO-CAD  |  6-View Orthographic Grid"
LEGEND_TEXT = (
    "DEPTH MAP KEY:  \u25A0 BRIGHT YELLOW = closest to camera  \u2192  "
    "\u25A0 DARK PURPLE = furthest from camera  |  "
    "Background = #1e1e1e (not part of object)"
)

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


def _crop_to_object(
    rgb: np.ndarray,
    depth_color: np.ndarray,
    padding: float = CROP_PADDING_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop RGB + colourised depth to the object's bounding box with padding.

    The object is defined as the set of pixels in `rgb` that are not
    near-white (< 250 in any channel). Both arrays are cropped to the same
    box so RGB and depth stay pixel-aligned.
    """
    if rgb.shape[:2] != depth_color.shape[:2]:
        raise ValueError(
            f"rgb shape {rgb.shape[:2]} does not match depth shape {depth_color.shape[:2]}"
        )

    is_object = ~(rgb >= 250).all(axis=-1)
    if not np.any(is_object):
        return rgb, depth_color

    rows = np.any(is_object, axis=1)
    cols = np.any(is_object, axis=0)
    r_min, r_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c_min, c_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    h, w = rgb.shape[:2]
    pad_r = int(round((r_max - r_min) * padding))
    pad_c = int(round((c_max - c_min) * padding))
    r_min = max(0, r_min - pad_r)
    r_max = min(h - 1, r_max + pad_r)
    c_min = max(0, c_min - pad_c)
    c_max = min(w - 1, c_max + pad_c)

    return (
        rgb[r_min : r_max + 1, c_min : c_max + 1],
        depth_color[r_min : r_max + 1, c_min : c_max + 1],
    )


def _resize_panel(arr: np.ndarray, size: int = RENDER_PANEL_SIZE) -> np.ndarray:
    """Resize an HxWx3 uint8 panel to size x size using PIL LANCZOS."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 array, got shape {arr.shape} dtype {arr.dtype}")
    img = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _background_mask(rgb: np.ndarray, raw_depth: np.ndarray) -> np.ndarray:
    """Return True where the pixel is background, by RGB whiteness OR far clip.

    Pure-RGB detection alone leaks anti-aliased edge pixels (which appear
    almost-white) into the surface set, but those same pixels carry the
    far-clip depth and would otherwise blow up per-view normalisation. We
    union both signals so flat surfaces correctly collapse to d_min == d_max.
    """
    rgb_bg = (rgb >= 250).all(axis=-1)
    far_clip = float(raw_depth.max())
    spread = float(raw_depth.max() - raw_depth.min())
    tol = max(spread * 0.001, 1e-5)
    depth_bg = raw_depth >= (far_clip - tol)
    return rgb_bg | depth_bg


def _setup_lights(plotter: pv.Plotter) -> None:
    """Install the spec's three-light rig: key + fill + soft headlight ambient."""
    plotter.remove_all_lights()

    key_light = pv.Light(position=(1.0, 1.0, 2.0), focal_point=(0.0, 0.0, 0.0))
    key_light.intensity = 0.8
    plotter.add_light(key_light)

    fill_light = pv.Light(position=(-1.0, -0.5, 0.5), focal_point=(0.0, 0.0, 0.0))
    fill_light.intensity = 0.3
    plotter.add_light(fill_light)

    ambient = pv.Light(light_type="headlight")
    ambient.intensity = 0.2
    plotter.add_light(ambient)


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
        _setup_lights(plotter)
        if direction == "-Z":
            below_light = pv.Light(
                position=(0.0, 0.0, -2.0),
                focal_point=(0.0, 0.0, 0.0),
            )
            below_light.intensity = 0.7
            plotter.add_light(below_light)
        plotter.add_mesh(
            mesh,
            color=MESH_COLOR,
            smooth_shading=True,
            ambient=0.15,
            diffuse=0.85,
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
    """Convert a raw depth buffer into a per-view plasma-coloured depth map.

    Per GRID_FORMAT_SPEC.md:
      - Normalise using only object pixels (not background).
      - Invert so high normalised value = closest = bright yellow.
      - Completely flat surfaces are mapped to mid colormap (orange) so they
        are still visually distinct from the background.
      - Background pixels are painted ``BG_DEPTH`` (very dark gray, distinct
        from the dark-purple end of plasma).

    `background_mask` (strongly preferred) is a boolean array, True where
    there is no surface. When omitted, falls back to detecting NaN/Inf or
    values within 0.1% of the array's far-clip max — this fallback is only
    reliable when the renderer guarantees a well-defined far clip.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if background_mask is None:
        if np.isfinite(depth).any():
            far_clip = float(np.nanmax(depth))
            background = ~np.isfinite(depth) | (depth >= far_clip * 0.999)
        else:
            background = np.ones_like(depth, dtype=bool)
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

    if d_max <= d_min or (d_max - d_min) < FLAT_DEPTH_EPSILON:
        h, w = depth.shape
        cy, cx = h / 2.0, w / 2.0
        y_idx, x_idx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        max_dist = float(np.sqrt(cy * cy + cx * cx))
        dist_norm = dist / max_dist
        radial_span = FLAT_SURFACE_RADIAL_HIGH - FLAT_SURFACE_RADIAL_LOW
        radial = FLAT_SURFACE_RADIAL_HIGH - dist_norm * radial_span
        normalised = radial[surface].astype(np.float32)
    else:
        normalised = (surface_vals - d_min) / (d_max - d_min)
        normalised = 1.0 - normalised

    cmap = _mpl_colormaps["plasma"]
    colored = cmap(normalised)[:, :3]
    out[surface] = (colored * 255.0).astype(np.uint8)
    return out


_FONT_CANDIDATES_BOLD = (
    "DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "Arial Bold.ttf",
)
_FONT_CANDIDATES_REGULAR = (
    "DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "Arial.ttf",
)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except AttributeError:
        return draw.textsize(text, font=font)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill=(255, 255, 255),
) -> None:
    x0, y0, x1, y1 = box
    tw, th = _text_size(draw, text, font)
    cx = x0 + (x1 - x0 - tw) // 2
    cy = y0 + (y1 - y0 - th) // 2
    draw.text((cx, cy), text, fill=fill, font=font)


def _paste_array(canvas: Image.Image, arr: np.ndarray, xy: tuple[int, int]) -> None:
    canvas.paste(Image.fromarray(arr), xy)


def build_grid(
    renders: dict[str, np.ndarray],
    depth_maps: dict[str, np.ndarray],
) -> Image.Image:
    """Compose 6 cells per GRID_FORMAT_SPEC.md into a 2400x930 PIL image.

    Each 800x465 cell stacks: top label bar (800x40), render area
    (RGB 400x400 + Depth 400x400), and a sub-label bar (800x25). Cells
    are separated by 2px #333333 borders; the outer edge has no border.
    """
    missing = [d for row in GRID_ORDER for d in row if d not in renders or d not in depth_maps]
    if missing:
        raise ValueError(f"Missing renders/depths for directions: {missing}")

    canvas = Image.new("RGB", (GRID_WIDTH, GRID_HEIGHT), color=BAR_BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(LABEL_FONT_SIZE, bold=True)
    sub_font = _load_font(SUB_LABEL_FONT_SIZE, bold=False)

    for row_idx, row in enumerate(GRID_ORDER):
        for col_idx, direction in enumerate(row):
            cell_x = col_idx * CELL_WIDTH
            cell_y = row_idx * CELL_HEIGHT

            label_box = (cell_x, cell_y, cell_x + CELL_WIDTH, cell_y + LABEL_BAR_HEIGHT)
            draw.rectangle(label_box, fill=BAR_BG_COLOR)
            _draw_centered_text(draw, label_box, VIEW_LABELS[direction], label_font)

            render_y = cell_y + LABEL_BAR_HEIGHT
            _paste_array(canvas, renders[direction], (cell_x, render_y))
            _paste_array(
                canvas,
                depth_maps[direction],
                (cell_x + RENDER_PANEL_SIZE, render_y),
            )

            sub_y = render_y + RENDER_PANEL_SIZE
            sub_box_rgb = (cell_x, sub_y, cell_x + RENDER_PANEL_SIZE, sub_y + SUB_LABEL_HEIGHT)
            sub_box_depth = (
                cell_x + RENDER_PANEL_SIZE,
                sub_y,
                cell_x + CELL_WIDTH,
                sub_y + SUB_LABEL_HEIGHT,
            )
            draw.rectangle(sub_box_rgb, fill=BAR_BG_COLOR)
            draw.rectangle(sub_box_depth, fill=BAR_BG_COLOR)
            _draw_centered_text(draw, sub_box_rgb, "RGB", sub_font)
            _draw_centered_text(draw, sub_box_depth, "Depth", sub_font)

    _draw_cell_borders(draw)
    return canvas


def _draw_cell_borders(draw: ImageDraw.ImageDraw) -> None:
    """Draw 2px #333333 borders between cells (no outer border)."""
    for col_idx in range(1, 3):
        x = col_idx * CELL_WIDTH - BORDER_WIDTH // 2
        draw.rectangle((x, 0, x + BORDER_WIDTH - 1, GRID_HEIGHT - 1), fill=BORDER_COLOR)
    for row_idx in range(1, 2):
        y = row_idx * CELL_HEIGHT - BORDER_WIDTH // 2
        draw.rectangle((0, y, GRID_WIDTH - 1, y + BORDER_WIDTH - 1), fill=BORDER_COLOR)


def _build_header_bar(part_id: str) -> Image.Image:
    bar = Image.new("RGB", (FULL_WIDTH, HEADER_HEIGHT), color=BAR_BG_COLOR)
    draw = ImageDraw.Draw(bar)
    font = _load_font(HEADER_FONT_SIZE, bold=True)

    _, th = _text_size(draw, HEADER_TITLE, font)
    cy = (HEADER_HEIGHT - th) // 2
    draw.text((HEADER_PAD_X, cy), HEADER_TITLE, fill=(255, 255, 255), font=font)

    right_text = f"Part ID: {part_id}"
    rw, rh = _text_size(draw, right_text, font)
    draw.text(
        (FULL_WIDTH - HEADER_PAD_X - rw, (HEADER_HEIGHT - rh) // 2),
        right_text,
        fill=(255, 255, 255),
        font=font,
    )
    return bar


def _build_legend_bar() -> Image.Image:
    bar = Image.new("RGB", (FULL_WIDTH, LEGEND_HEIGHT), color=LEGEND_BG_COLOR)
    draw = ImageDraw.Draw(bar)
    font = _load_font(LEGEND_FONT_SIZE, bold=False)
    _draw_centered_text(
        draw,
        (0, 0, FULL_WIDTH, LEGEND_HEIGHT),
        LEGEND_TEXT,
        font,
    )
    return bar


def _compose_final_image(grid: Image.Image, part_id: str) -> Image.Image:
    if grid.size != (GRID_WIDTH, GRID_HEIGHT):
        raise RuntimeError(f"Grid size {grid.size}, expected ({GRID_WIDTH}, {GRID_HEIGHT})")

    canvas = Image.new("RGB", (FULL_WIDTH, FULL_HEIGHT), color=BAR_BG_COLOR)
    canvas.paste(_build_header_bar(part_id), (0, 0))
    canvas.paste(grid, (0, HEADER_HEIGHT))
    canvas.paste(_build_legend_bar(), (0, HEADER_HEIGHT + GRID_HEIGHT))
    return canvas


def render_stl_to_grid(
    stl_path: str | Path,
    output_path: str | Path,
    part_id: str | None = None,
) -> Path:
    """End-to-end: STL path in, full 2400x1000 PNG written to output_path.

    `part_id` is embedded in the header bar; when omitted it defaults to the
    STL filename stem so the CLI works without extra arguments.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stl_path = Path(stl_path)
    if part_id is None:
        part_id = stl_path.stem

    mesh = load_mesh(stl_path)

    renders: dict[str, np.ndarray] = {}
    depth_maps: dict[str, np.ndarray] = {}
    for direction in CAMERA_VIEWS:
        rgb, raw_depth = render_view(mesh, direction)
        bg_mask = _background_mask(rgb, raw_depth)
        depth_color = depth_to_colormap(raw_depth, background_mask=bg_mask)
        rgb_cropped, depth_cropped = _crop_to_object(rgb, depth_color)
        renders[direction] = _resize_panel(rgb_cropped, RENDER_PANEL_SIZE)
        depth_maps[direction] = _resize_panel(depth_cropped, RENDER_PANEL_SIZE)

    grid = build_grid(renders, depth_maps)
    final = _compose_final_image(grid, part_id)
    if final.size != (FULL_WIDTH, FULL_HEIGHT):
        raise RuntimeError(
            f"Unexpected final image size {final.size}, want ({FULL_WIDTH}, {FULL_HEIGHT})"
        )

    final.save(output_path, format="PNG")
    print(
        f"[stl_renderer] Wrote {output_path} "
        f"({final.size[0]}x{final.size[1]}, part_id={part_id})"
    )
    return output_path


def verify_grid(png_path: str | Path) -> bool:
    """Assert that a generated PNG matches the GRID_FORMAT_SPEC dimensions.

    Verifies the full image is exactly (FULL_WIDTH, FULL_HEIGHT) and that
    the dark header/legend bars sit at the expected pixel rows. Returns
    True on success and prints a short diagnostic line. Raises
    AssertionError with a clear message on any mismatch.
    """
    png_path = Path(png_path)
    if not png_path.is_file():
        raise AssertionError(f"Grid PNG not found: {png_path}")

    with Image.open(png_path) as img:
        rgb_img = img.convert("RGB")
        size = rgb_img.size
        pixels = np.asarray(rgb_img)

    expected = (FULL_WIDTH, FULL_HEIGHT)
    if size != expected:
        raise AssertionError(
            f"Image size mismatch for {png_path}: got {size}, expected {expected}"
        )

    header_row = pixels[HEADER_HEIGHT // 2]
    if int(np.median(header_row.max(axis=-1))) > 0x30:
        raise AssertionError(f"Header bar at y={HEADER_HEIGHT // 2} is not predominantly dark")

    legend_y = HEADER_HEIGHT + GRID_HEIGHT + LEGEND_HEIGHT // 2
    legend_row = pixels[legend_y]
    if int(np.median(legend_row.max(axis=-1))) > 0x30:
        raise AssertionError(f"Legend bar at y={legend_y} is not predominantly dark")

    print(
        f"[stl_renderer] OK {png_path} matches {expected[0]}x{expected[1]} "
        f"(header + grid + legend bars present)"
    )
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stl_renderer",
        description=f"Render an STL mesh into a {FULL_WIDTH}x{FULL_HEIGHT} 6-view PNG grid.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing grid PNG matches the expected dimensions.",
    )
    parser.add_argument(
        "--part-id",
        dest="part_id",
        default=None,
        help="Identifier embedded in the header bar (defaults to STL filename stem).",
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
    render_stl_to_grid(args.input_path, args.output_path, part_id=args.part_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
