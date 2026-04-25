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
    - Per-view depth normalisation against object pixels only,
      coloured with matplotlib's RdYlBu_r colormap (red=near,
      blue=far); flat surfaces fall back to a subtle centre-to-edge
      radial gradient in the warm 0.92-1.0 range.
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

# Mapping of view direction -> (horizontal world axis, vertical world axis).
# Used to annotate each cell with the actual mm span that's visible. This
# turns "guess the proportions" into "read the number" for the vision model.
VIEW_AXES: dict[str, tuple[str, str]] = {
    "+Z": ("X", "Y"),
    "-Z": ("X", "Y"),
    "+X": ("Y", "Z"),
    "-X": ("Y", "Z"),
    "+Y": ("X", "Z"),
    "-Y": ("X", "Z"),
}

# "Round" lengths the per-panel scale bar will pick from, in mm. The
# renderer chooses whichever value lands closest to a 60-pixel bar at
# the current view's pixels-per-mm ratio.
SCALE_BAR_CANDIDATES_MM: list[float] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
SCALE_BAR_TARGET_PX = 60

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
    "DEPTH MAP KEY:  \u25A0 RED/ORANGE = closest to camera  \u2192  "
    "\u25A0 DARK BLUE = furthest from camera  |  "
    "YELLOW = mid-distance  |  "
    "SCALE BAR (bottom-left of each RGB panel) gives the horizontal mm reference  |  "
    "Cell label states the visible region's world span"
)
DEPTH_COLORMAP = "RdYlBu_r"

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
    """Return True where the pixel is background.

    Two VTK behaviours need handling. Modern builds honour the
    ``fill_value=np.nan`` we pass to ``get_image_depth`` and return NaN at
    background and anti-aliased edge pixels; in that case RGB-whiteness
    plus NaN is enough. Older builds ignore ``fill_value`` and return the
    far-clip depth value for both background and edges; there we additionally
    mask depths within tolerance of the far-clip max so a constant-depth
    surface still survives as ``surface``.
    """
    rgb_bg = (rgb >= 250).all(axis=-1)
    nan_bg = ~np.isfinite(raw_depth)
    if nan_bg.any():
        return rgb_bg | nan_bg

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
    diag = float(np.linalg.norm(np.ptp(np.asarray(mesh.bounds).reshape(3, 2), axis=1)))
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
    """Convert a raw depth buffer into a per-view RdYlBu_r-coloured depth map.

    Per GRID_FORMAT_SPEC.md:
      - Normalise using only object pixels (not background).
      - Invert so high normalised value = closest = red/orange.
      - Completely flat surfaces fall back to a centre-to-edge radial
        gradient in the warm 0.92-1.0 range so they read as intentional
        rather than as a broken mid-colormap solid block.
      - Background pixels are painted ``BG_DEPTH`` (#1e1e1e), distinct
        from the dark-blue far end of the colormap.

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

    cmap = _mpl_colormaps[DEPTH_COLORMAP]
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


def _view_spans_mm(
    mesh_bounds: tuple[float, float, float, float, float, float],
    direction: str,
) -> tuple[float, float]:
    """Return (horizontal_mm, vertical_mm) world span visible in this view.

    Each panel is cropped to the object's projected bounding box plus a
    fixed padding fraction on every side, so the world span actually
    visible is the bbox extent along that axis multiplied by
    ``(1 + 2 * CROP_PADDING_FRAC)``. The two axes are decided by
    ``VIEW_AXES``.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds
    extents = {"X": xmax - xmin, "Y": ymax - ymin, "Z": zmax - zmin}
    h_axis, v_axis = VIEW_AXES[direction]
    pad_factor = 1.0 + 2.0 * CROP_PADDING_FRAC
    return extents[h_axis] * pad_factor, extents[v_axis] * pad_factor


def _format_mm(value: float) -> str:
    """Format a millimetre value as an integer when round, else 1 decimal."""
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _annotated_label(direction: str, h_mm: float, v_mm: float) -> str:
    """Compose the cell label with the visible world dimensions appended."""
    h_axis, v_axis = VIEW_AXES[direction]
    return (
        f"{VIEW_LABELS[direction]}    "
        f"|  visible: {h_axis} {_format_mm(h_mm)}mm \u00d7 "
        f"{v_axis} {_format_mm(v_mm)}mm"
    )


def _pick_scale_bar_length_mm(world_span_mm: float) -> float:
    """Pick a 'round' scale-bar length that targets ~60 px on screen."""
    pixels_per_mm = RENDER_PANEL_SIZE / max(world_span_mm, 1e-6)
    target_mm = SCALE_BAR_TARGET_PX / pixels_per_mm
    return min(SCALE_BAR_CANDIDATES_MM, key=lambda c: abs(c - target_mm))


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_y: int,
    horizontal_mm: float,
    font: ImageFont.ImageFont,
) -> None:
    """Draw a small horizontal scale bar in the bottom-left of an RGB panel.

    Bar sits on a white background pad so it stays legible regardless of
    the underlying render. End ticks mark the bar length precisely; the
    label below states the length in millimetres.
    """
    bar_mm = _pick_scale_bar_length_mm(horizontal_mm)
    pixels_per_mm = RENDER_PANEL_SIZE / max(horizontal_mm, 1e-6)
    bar_px = max(8, int(round(bar_mm * pixels_per_mm)))

    margin = 12
    bar_y = panel_y + RENDER_PANEL_SIZE - margin - 6
    label = f"{_format_mm(bar_mm)}mm"

    label_w = int(draw.textlength(label, font=font))
    pad = 4
    bg_x0 = panel_x + margin - pad
    bg_y0 = bar_y - pad - 2
    bg_x1 = panel_x + margin + bar_px + 6 + label_w + pad
    bg_y1 = bar_y + 8 + pad
    draw.rectangle((bg_x0, bg_y0, bg_x1, bg_y1), fill=(255, 255, 255))

    x0 = panel_x + margin
    x1 = x0 + bar_px
    cy = bar_y + 4
    draw.line((x0, cy, x1, cy), fill=(0, 0, 0), width=2)
    draw.line((x0, cy - 4, x0, cy + 4), fill=(0, 0, 0), width=2)
    draw.line((x1, cy - 4, x1, cy + 4), fill=(0, 0, 0), width=2)
    draw.text((x1 + 6, bar_y - 2), label, fill=(0, 0, 0), font=font)


def build_grid(
    renders: dict[str, np.ndarray],
    depth_maps: dict[str, np.ndarray],
    mesh_bounds: tuple[float, float, float, float, float, float] | None = None,
) -> Image.Image:
    """Compose 6 cells per GRID_FORMAT_SPEC.md into a 2400x930 PIL image.

    Each 800x465 cell stacks: top label bar (800x40), render area
    (RGB 400x400 + Depth 400x400), and a sub-label bar (800x25). Cells
    are separated by 2px #333333 borders; the outer edge has no border.

    When ``mesh_bounds`` is supplied, every cell label is annotated with
    the visible world-coordinate span (e.g. ``X 100mm × Y 80mm``) and
    the RGB panel gets a small scale-bar overlay. This is the only way
    the vision model can recover absolute dimensions from the renders;
    without it the prompt has to fall back to "longest dim = 100mm".
    """
    missing = [d for row in GRID_ORDER for d in row if d not in renders or d not in depth_maps]
    if missing:
        raise ValueError(f"Missing renders/depths for directions: {missing}")

    canvas = Image.new("RGB", (GRID_WIDTH, GRID_HEIGHT), color=BAR_BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(LABEL_FONT_SIZE, bold=True)
    sub_font = _load_font(SUB_LABEL_FONT_SIZE, bold=False)
    scale_font = _load_font(SUB_LABEL_FONT_SIZE, bold=True)

    for row_idx, row in enumerate(GRID_ORDER):
        for col_idx, direction in enumerate(row):
            cell_x = col_idx * CELL_WIDTH
            cell_y = row_idx * CELL_HEIGHT

            label_box = (cell_x, cell_y, cell_x + CELL_WIDTH, cell_y + LABEL_BAR_HEIGHT)
            draw.rectangle(label_box, fill=BAR_BG_COLOR)
            if mesh_bounds is not None:
                h_mm, v_mm = _view_spans_mm(mesh_bounds, direction)
                label_text = _annotated_label(direction, h_mm, v_mm)
            else:
                label_text = VIEW_LABELS[direction]
            _draw_centered_text(draw, label_box, label_text, label_font)

            render_y = cell_y + LABEL_BAR_HEIGHT
            _paste_array(canvas, renders[direction], (cell_x, render_y))
            _paste_array(
                canvas,
                depth_maps[direction],
                (cell_x + RENDER_PANEL_SIZE, render_y),
            )

            if mesh_bounds is not None:
                _draw_scale_bar(draw, cell_x, render_y, h_mm, scale_font)

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


def _normalise_mesh(mesh: pv.PolyData, target_longest_mm: float) -> pv.PolyData:
    """Uniformly scale ``mesh`` so its longest bbox extent equals ``target_longest_mm``.

    DeepCAD-style sources arrive in normalised cube coordinates (extents
    ~1.5), so the dimension annotations would read sub-millimetre and
    contradict the prompt's mm-scale heuristics. This rescales the mesh
    in place so the longest dimension is a sane working size; the other
    two scale proportionally.
    """
    bounds = np.asarray(mesh.bounds).reshape(3, 2)
    extents = np.ptp(bounds, axis=1)
    longest = float(np.max(extents))
    if longest <= 0:
        return mesh
    factor = target_longest_mm / longest
    if abs(factor - 1.0) < 1e-6:
        return mesh
    print(
        f"[stl_renderer] Scaling mesh by {factor:.4f}x "
        f"(longest extent {longest:.4f} -> {target_longest_mm} mm)"
    )
    return mesh.scale(factor, inplace=False)


def render_stl_to_grid(
    stl_path: str | Path,
    output_path: str | Path,
    part_id: str | None = None,
    normalize_longest_to_mm: float | None = None,
) -> Path:
    """End-to-end: STL path in, full 2400x1000 PNG written to output_path.

    `part_id` is embedded in the header bar; when omitted it defaults to the
    STL filename stem so the CLI works without extra arguments.

    `normalize_longest_to_mm` rescales the mesh so its longest bounding-
    box extent equals the given value before rendering. Use it for
    sources that are not in real-world millimetres (e.g. DeepCAD samples
    sit in a [-0.75, 0.75] cube). Leave it ``None`` for STLs already in
    mm so their dimension annotations remain truthful.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stl_path = Path(stl_path)
    if part_id is None:
        part_id = stl_path.stem

    mesh = load_mesh(stl_path)
    if normalize_longest_to_mm is not None:
        mesh = _normalise_mesh(mesh, normalize_longest_to_mm)
    mesh_bounds = tuple(float(v) for v in mesh.bounds)  # type: ignore[arg-type]

    renders: dict[str, np.ndarray] = {}
    depth_maps: dict[str, np.ndarray] = {}
    for direction in CAMERA_VIEWS:
        rgb, raw_depth = render_view(mesh, direction)
        bg_mask = _background_mask(rgb, raw_depth)
        depth_color = depth_to_colormap(raw_depth, background_mask=bg_mask)
        rgb_cropped, depth_cropped = _crop_to_object(rgb, depth_color)
        renders[direction] = _resize_panel(rgb_cropped, RENDER_PANEL_SIZE)
        depth_maps[direction] = _resize_panel(depth_cropped, RENDER_PANEL_SIZE)

    grid = build_grid(renders, depth_maps, mesh_bounds=mesh_bounds)
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
        "--normalize-longest-mm",
        dest="normalize_longest_mm",
        type=float,
        default=None,
        help=(
            "Scale the mesh so its longest bbox extent equals this many mm "
            "before rendering. Use for non-mm sources like DeepCAD samples; "
            "leave unset for STLs already in real-world millimetres."
        ),
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
    render_stl_to_grid(
        args.input_path, args.output_path,
        part_id=args.part_id,
        normalize_longest_to_mm=args.normalize_longest_mm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
