"""OpenCV-based 2D contour extraction from a 6-view orthographic grid PNG.

Belongs to the experimental ``sketch-plane-architecture`` branch. The
goal is to give Claude *measured* 2D geometry (vertex lists, aspect
ratios, depth-warm-region bounding boxes) so it does not have to
re-extract that information from raw pixels. Claude can then focus on
construction order and depth reasoning.

Usage::

    from backend.ai_infra.contour_extractor import (
        extract_all_views, summarise_contours,
    )
    contours = extract_all_views("backend/outputs/.../foo_grid.png")
    text = summarise_contours(contours)

The grid layout matches what ``backend.pipeline.stl_renderer`` produces:

    full image: 2400 x 1000 px
        header bar:  y =   0 ..  40
        grid:        y =  40 .. 970   (two rows of 465 px each)
        legend:      y = 970 .. 1000

    each cell within the grid (800 x 465 px):
        label bar:   first 40 px of the cell (inside the cell)
        render row:  next 400 px (LEFT 400 = RGB panel, RIGHT 400 = depth)
        sub-labels:  bottom 25 px

So the absolute pixel rectangle of a cell's RGB panel is::

    rgb = (cell_x,      40 + cell_y + 40, cell_x + 400,      40 + cell_y + 40 + 400)
    dep = (cell_x + 400, 40 + cell_y + 40, cell_x + 800,      40 + cell_y + 40 + 400)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Type alias for ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in world mm,
# matching ``pyvista.PolyData.bounds``. Used to convert normalised
# panel coordinates to world mm in the view's local 2D frame.
MeshBounds = Tuple[float, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Layout constants — must mirror backend/pipeline/stl_renderer.py.
# Hard-coded here on purpose: we want the contour extractor to remain
# usable even if someone runs it on a grid PNG saved out separately.
# ---------------------------------------------------------------------------
HEADER_HEIGHT = 40
LABEL_BAR_HEIGHT = 40
PANEL_PX = 400
CELL_W = 800
CELL_H = 465

# (cell_x, cell_y) of the top-left of each cell INSIDE the grid area
# (i.e. before the header offset is applied).
CELL_ORIGINS: dict[str, Tuple[int, int]] = {
    "+Z": (0,    0),
    "+X": (800,  0),
    "+Y": (1600, 0),
    "-Z": (0,    465),
    "-X": (800,  465),
    "-Y": (1600, 465),
}
VIEW_ORDER: List[str] = ["+Z", "+X", "+Y", "-Z", "-X", "-Y"]

# Padding the renderer applies on every side when cropping each panel
# to the projected bbox. MUST match ``CROP_PADDING_FRAC`` in
# ``backend/pipeline/stl_renderer.py``: each panel covers
# ``bbox_extent * (1 + 2 * CROP_PADDING_FRAC)`` of world mm.
CROP_PAD_FACTOR = 1.0 + 2.0 * 0.1

# Per-view (horizontal, vertical) world axis the panel maps to. MUST
# match ``VIEW_AXES`` in stl_renderer. ``+y`` of the panel image
# (which goes DOWN in pixel coordinates) corresponds to ``-`` along
# the world vertical axis listed here.
VIEW_AXES: dict[str, Tuple[str, str]] = {
    "+Z": ("X", "Y"),
    "-Z": ("X", "Y"),
    "+X": ("Y", "Z"),
    "-X": ("Y", "Z"),
    "+Y": ("X", "Z"),
    "-Y": ("X", "Z"),
}


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
class ViewContour(BaseModel):
    """Per-view OpenCV analysis of one cell of the 6-view grid.

    Two coordinate systems live on this object:

    * **Normalised panel coords** (``vertices``, ``warm_regions``) —
      0..1 inside the RGB panel, origin top-left, +x right, +y down.
      These are computed unconditionally and are useful even without
      a mesh bbox.
    * **World mm coords** (``vertices_mm``, ``bbox_mm``,
      ``warm_regions_mm``) — only populated when the caller supplied
      ``mesh_bounds_mm`` to ``extract_all_views``. These are in the
      view's local 2D frame: +x along the view's horizontal world
      axis, +y along the view's vertical world axis (so y is FLIPPED
      relative to the image), origin at the panel centre. This is
      directly usable as input to a CadQuery ``.polyline(...)`` call.
    """

    view_name: str = Field(..., description='View label, e.g. "+Z" or "-X".')
    shape_type: str = Field(
        ...,
        description='Coarse classification of the 2D outline: '
        '"rectangle" | "circle" | "polygon".',
    )
    vertices: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="Simplified polygon vertices of the outline, normalised "
        "to 0-1 inside the RGB panel (origin top-left, +x right, +y down).",
    )
    vertex_count: int = Field(..., description="len(vertices) for convenience.")
    aspect_ratio: float = Field(
        ...,
        description="Bounding rect width / height. >1 means wider than tall.",
    )
    warm_regions: List[Tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Bounding boxes of red/orange (= near-camera) regions "
        "in the depth panel, normalised (x, y, w, h) inside the panel.",
    )
    area_ratio: float = Field(
        ...,
        description="contour_area / bounding_rect_area. Close to 1.0 = solid "
        "rectangle; close to pi/4 ~= 0.785 = circle; lower = irregular.",
    )
    vertices_mm: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="Same vertices as ``vertices`` but in world mm in the "
        "view's local 2D frame (origin at panel centre, y-up). Empty when "
        "no mesh bbox was provided.",
    )
    bbox_mm: Optional[Tuple[float, float]] = Field(
        None,
        description="(width_mm, height_mm) of the contour's bounding box in "
        "the view's local 2D frame. None when no mesh bbox was provided.",
    )
    warm_regions_mm: List[Tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Same as ``warm_regions`` but in world mm: "
        "(centre_x_mm, centre_y_mm, width_mm, height_mm) in the view's "
        "local 2D frame. Empty when no mesh bbox was provided.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _crop_panel(
    image_bgr: np.ndarray, cell_origin: Tuple[int, int], depth: bool
) -> np.ndarray:
    """Return the RGB or depth sub-panel for a given cell.

    ``cell_origin`` is the (x, y) of the cell within the grid area
    (header excluded). The full-image y coordinate is
    ``HEADER_HEIGHT + cell_y + LABEL_BAR_HEIGHT``.
    """
    cx, cy = cell_origin
    x0 = cx + (PANEL_PX if depth else 0)
    y0 = HEADER_HEIGHT + cy + LABEL_BAR_HEIGHT
    return image_bgr[y0 : y0 + PANEL_PX, x0 : x0 + PANEL_PX]


def _outline_contour(rgb_panel: np.ndarray) -> np.ndarray | None:
    """Threshold the RGB panel and return the largest external contour.

    The renderer paints a white (255, 255, 255) background behind RGB
    panels, so anything below 240 in grayscale is part of the object.
    Returns ``None`` when no object is found (degenerate cell).
    """
    gray = cv2.cvtColor(rgb_panel, cv2.COLOR_BGR2GRAY)
    # background -> 0, object -> 255
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Tiny morph close to bridge anti-aliased gaps in thin features.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _classify_shape(
    vertex_count: int, area_ratio: float, aspect_ratio: float
) -> str:
    """Coarse shape label from vertex count + how rectangular / circular."""
    if vertex_count <= 4 and area_ratio > 0.85:
        return "rectangle"
    if vertex_count >= 9:
        # Many small edges => approximation of a curve. If the bounding
        # rect is roughly square AND area_ratio is near pi/4 (~0.785),
        # call it a circle; otherwise it's a curvy polygon.
        if 0.9 <= aspect_ratio <= 1.1 and 0.7 <= area_ratio <= 0.9:
            return "circle"
        return "circle"  # default for many-vertex shapes
    if 5 <= vertex_count <= 8:
        return "polygon"
    return "rectangle"


def _warm_regions(depth_panel: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Bounding boxes of red/orange regions in the depth panel.

    A "warm" region in the RdYlBu_r colormap means the part is closest
    to the camera there, which corresponds to a raised feature on the
    facing side. We look for hue near 0 (red) or near 180 (also red) in
    HSV space, with a sat/val floor that excludes the dark grey
    background (#1e1e1e ~ HSV value 30).
    """
    hsv = cv2.cvtColor(depth_panel, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,   100, 100], dtype=np.uint8)
    upper1 = np.array([20,  255, 255], dtype=np.uint8)
    lower2 = np.array([160, 100, 100], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # Suppress speckle so a single noisy red pixel does not become a region.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[float, float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        # Drop blobs smaller than 0.25% of the panel area.
        if area < 0.0025 * PANEL_PX * PANEL_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(
            (x / PANEL_PX, y / PANEL_PX, w / PANEL_PX, h / PANEL_PX)
        )
    # Sort largest-first so the prompt sees the dominant region first.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


# ---------------------------------------------------------------------------
# Normalised panel <-> world mm conversion
# ---------------------------------------------------------------------------
def _view_panel_extents_mm(
    bounds: MeshBounds, view_name: str
) -> Tuple[float, float]:
    """Return (h_mm, v_mm) the world span the panel covers for this view.

    Mirrors ``backend.pipeline.stl_renderer._view_spans_mm``: each panel
    is the projected bbox padded by ``CROP_PADDING_FRAC`` on every side.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extents = {"X": xmax - xmin, "Y": ymax - ymin, "Z": zmax - zmin}
    h_axis, v_axis = VIEW_AXES[view_name]
    return extents[h_axis] * CROP_PAD_FACTOR, extents[v_axis] * CROP_PAD_FACTOR


def _norm_to_mm(
    nx: float, ny: float, h_mm: float, v_mm: float
) -> Tuple[float, float]:
    """Map a normalised (0..1, 0..1, y-down) panel point to world mm.

    Origin is the panel centre. +x along the view's horizontal world
    axis, +y along the view's vertical world axis (so we flip y because
    image y points down).
    """
    return (nx - 0.5) * h_mm, -(ny - 0.5) * v_mm


def _norm_box_to_mm(
    box: Tuple[float, float, float, float], h_mm: float, v_mm: float
) -> Tuple[float, float, float, float]:
    """Convert a normalised (x, y, w, h) box (top-left origin, y-down) to
    a (centre_x_mm, centre_y_mm, width_mm, height_mm) tuple in world mm
    centred on the panel."""
    x, y, w, h = box
    cx_norm = x + w / 2
    cy_norm = y + h / 2
    cx_mm, cy_mm = _norm_to_mm(cx_norm, cy_norm, h_mm, v_mm)
    return cx_mm, cy_mm, w * h_mm, h * v_mm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_view(
    image_bgr: np.ndarray,
    view_name: str,
    cell_origin: Tuple[int, int],
    panel_extents_mm: Optional[Tuple[float, float]] = None,
) -> ViewContour:
    """Run the OpenCV pipeline for a single view cell.

    When ``panel_extents_mm`` is provided (``(h_mm, v_mm)``), the result
    will additionally carry world-mm vertex / bbox / warm-region data.
    """
    rgb = _crop_panel(image_bgr, cell_origin, depth=False)
    depth = _crop_panel(image_bgr, cell_origin, depth=True)

    contour = _outline_contour(rgb)
    warm_norm = _warm_regions(depth)

    if contour is None or cv2.contourArea(contour) < 1.0:
        # Empty / degenerate cell: emit a 'nothing here' record so the
        # prompt summary still mentions the view.
        warm_mm: List[Tuple[float, float, float, float]] = []
        if panel_extents_mm is not None:
            h_mm, v_mm = panel_extents_mm
            warm_mm = [_norm_box_to_mm(b, h_mm, v_mm) for b in warm_norm]
        return ViewContour(
            view_name=view_name,
            shape_type="rectangle",
            vertices=[],
            vertex_count=0,
            aspect_ratio=1.0,
            warm_regions=warm_norm,
            area_ratio=0.0,
            vertices_mm=[],
            bbox_mm=None,
            warm_regions_mm=warm_mm,
        )

    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = 0.02 * perimeter
    poly = cv2.approxPolyDP(contour, epsilon, closed=True)
    verts = [(float(p[0][0]) / PANEL_PX, float(p[0][1]) / PANEL_PX) for p in poly]

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h if h > 0 else 1.0
    bbox_area = max(w * h, 1)
    area_ratio = float(cv2.contourArea(contour)) / bbox_area

    shape = _classify_shape(len(verts), area_ratio, aspect)

    verts_mm: List[Tuple[float, float]] = []
    bbox_mm: Optional[Tuple[float, float]] = None
    warm_mm = []
    if panel_extents_mm is not None:
        h_mm, v_mm = panel_extents_mm
        verts_mm = [_norm_to_mm(nx, ny, h_mm, v_mm) for nx, ny in verts]
        bbox_mm = ((w / PANEL_PX) * h_mm, (h / PANEL_PX) * v_mm)
        warm_mm = [_norm_box_to_mm(b, h_mm, v_mm) for b in warm_norm]

    return ViewContour(
        view_name=view_name,
        shape_type=shape,
        vertices=verts,
        vertex_count=len(verts),
        aspect_ratio=float(aspect),
        warm_regions=warm_norm,
        area_ratio=float(area_ratio),
        vertices_mm=verts_mm,
        bbox_mm=bbox_mm,
        warm_regions_mm=warm_mm,
    )


def extract_all_views(
    image_path: str | Path,
    mesh_bounds_mm: Optional[MeshBounds] = None,
) -> dict[str, ViewContour]:
    """Run :func:`extract_view` for all 6 cells. Order matches VIEW_ORDER.

    Pass ``mesh_bounds_mm`` (the original mesh's
    ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in mm) to populate the
    world-mm fields on each :class:`ViewContour`. Without it the result
    only carries normalised coordinates — useful for quick inspection
    but not enough to build a CadQuery polyline.
    """
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not load image: {path}")
    h, w = img.shape[:2]
    if (w, h) != (2400, 1000):
        raise ValueError(
            f"unexpected grid size {(w, h)}, expected (2400, 1000) — is this "
            "actually a 6-view grid PNG produced by backend.pipeline.stl_renderer?"
        )
    return {
        name: extract_view(
            img,
            name,
            CELL_ORIGINS[name],
            panel_extents_mm=(
                _view_panel_extents_mm(mesh_bounds_mm, name)
                if mesh_bounds_mm is not None
                else None
            ),
        )
        for name in VIEW_ORDER
    }


# ---------------------------------------------------------------------------
# Plain-English summary
# ---------------------------------------------------------------------------
def _aspect_phrase(aspect: float) -> str:
    if 0.95 <= aspect <= 1.05:
        return "aspect ~1:1"
    if aspect > 1.0:
        return f"aspect {aspect:.2f}:1 (wider than tall)"
    return f"aspect 1:{1.0/aspect:.2f} (taller than wide)"


def _warm_phrase_norm(boxes: list) -> str:
    if not boxes:
        return "no near-camera (warm) regions in the depth panel"
    if len(boxes) == 1:
        b = boxes[0]
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        return (
            f"one near-camera region centred at ({cx:.2f}, {cy:.2f}) in the "
            f"depth panel covering {b[2]*100:.0f}% x {b[3]*100:.0f}% of the panel"
        )
    parts = []
    for b in boxes:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        parts.append(f"({cx:.2f}, {cy:.2f}) {b[2]*100:.0f}%x{b[3]*100:.0f}%")
    return f"{len(boxes)} distinct near-camera regions: " + ", ".join(parts)


def _warm_phrase_mm(boxes_mm: list) -> str:
    """Like ``_warm_phrase_norm`` but in world mm (centre x,y, w x h)."""
    if not boxes_mm:
        return "no near-camera (warm) regions in the depth panel"
    if len(boxes_mm) == 1:
        cx, cy, w, h = boxes_mm[0]
        return (
            f"one near-camera region centred at ({cx:+.1f}, {cy:+.1f}) mm "
            f"covering {w:.1f} x {h:.1f} mm"
        )
    parts = [
        f"({cx:+.1f}, {cy:+.1f})mm {w:.1f}x{h:.1f}mm"
        for cx, cy, w, h in boxes_mm
    ]
    return f"{len(boxes_mm)} distinct near-camera regions: " + ", ".join(parts)


def _fmt_verts_mm(verts: list) -> str:
    """Render a vertex list in mm as a compact ``[(x, y), ...]`` string.

    Truncates to the first/last few when there are many vertices, since
    LLM token budgets are real and the hierarchy of a profile is
    usually clear from the corner sequence.
    """
    if not verts:
        return "[]"
    body = ", ".join(f"({x:+.1f}, {y:+.1f})" for x, y in verts)
    return f"[{body}]"


def summarise_contours(contours: dict[str, ViewContour]) -> str:
    """One block per view, suitable for inlining in a Claude prompt.

    When world-mm fields are populated on the ``ViewContour`` objects
    (i.e. the caller passed ``mesh_bounds_mm`` to
    :func:`extract_all_views`) the summary includes the bbox and
    vertex list IN MM so Claude can copy them straight into a
    ``polyline`` profile without re-measuring pixels.
    """
    lines: list[str] = []
    for name in VIEW_ORDER:
        c = contours[name]
        if c.vertex_count == 0:
            lines.append(f"{name}: no object detected in this view.")
            continue

        head = (
            f"{name}: {c.shape_type} outline with {c.vertex_count} vertices "
            f"({_aspect_phrase(c.aspect_ratio)}, area_ratio={c.area_ratio:.2f})."
        )

        if c.bbox_mm is not None:
            w_mm, h_mm = c.bbox_mm
            head += f" Bounding box: {w_mm:.1f} x {h_mm:.1f} mm."
            head += (
                "\n    vertices_mm (panel-local, +x right, +y up, origin at "
                f"panel centre): {_fmt_verts_mm(c.vertices_mm)}"
            )
            head += f"\n    depth-warm: {_warm_phrase_mm(c.warm_regions_mm)}."
        else:
            head += f" {_warm_phrase_norm(c.warm_regions)}."

        lines.append(head)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI for quick inspection: ``python -m backend.ai_infra.contour_extractor IMG``
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Extract OpenCV contours from a 6-view grid PNG.")
    p.add_argument("image", type=Path, help="Path to a *_grid.png file (2400x1000).")
    p.add_argument("--summary-only", action="store_true",
                   help="Print only the plain-English summary.")
    args = p.parse_args()

    contours = extract_all_views(args.image)
    if args.summary_only:
        print(summarise_contours(contours))
        return
    print(summarise_contours(contours))
    print()
    print(json.dumps(
        {k: v.model_dump() for k, v in contours.items()},
        indent=2,
    ))


if __name__ == "__main__":
    _main()
