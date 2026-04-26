"""OpenCV validation layer for the face-geometry pipeline.

Reads the 6-view render PNG produced by ``backend.pipeline.stl_renderer``
and looks for features that ``face_extractor`` did NOT recover:

  * Through-holes / blind pockets   — dark-blue patches in the top/bottom
                                       depth panels (RdYlBu_r colormap
                                       puts "far" at deep blue).
  * Depth steps / ledges            — distinct hue bands in a side-view
                                       depth panel (one band per depth
                                       level on that axis).
  * Small bosses / mounting holes   — small contours in the +Z RGB panel
                                       not accounted for by any planar
                                       face projection.

Findings are added to ``ExtractedGeometry.missed_features``. Existing
``planar_faces`` and ``cylindrical_faces`` are NEVER mutated — face
extractor stays the source of truth, OpenCV is the safety net.

DESIGN NOTES (corrections vs the spec prompt):

1) Pixel -> mm uses the OBJECT's pixel bbox per panel, not the panel
   size. The renderer crops to object bbox + 10% padding before resizing
   to 400x400, so the part fills only ~83% of each panel. Naively
   mapping ``x / PANEL_W`` would put hole centres ~10% off — bigger than
   the 5 mm dedup tolerance — and every existing hole would be
   re-reported as "missed."

2) The -Z view mirrors world X. To match a +Z hole against a -Z patch
   for through/blind classification, the lookup x-coordinate must be
   negated.

3) ``_load_normalised_mesh`` only scales — it does not translate to
   origin. The validator derives the actual world bbox from the face
   centres rather than assuming ``bounding_box_mm/2`` is the half-width
   around 0.

The grid layout constants here are duplicated rather than imported so
this module stays read-only on ``stl_renderer``. Both must move
together if the layout ever changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.ai_infra.face_extractor import (
    CylindricalFace,
    ExtractedGeometry,
    MissedFeature,
    PlanarFace,
)


# ---------------------------------------------------------------------------
# Grid layout constants — MUST match backend/pipeline/stl_renderer.py
# ---------------------------------------------------------------------------
HEADER_H = 40
LEGEND_H = 30
CELL_W, CELL_H = 800, 465
PANEL_W = PANEL_H = 400
LABEL_H = 40
SUBLABEL_H = 25

CELL_ORIGINS: dict[str, tuple[int, int]] = {
    "+Z": (0,    HEADER_H),
    "+X": (800,  HEADER_H),
    "+Y": (1600, HEADER_H),
    "-Z": (0,    HEADER_H + CELL_H),
    "-X": (800,  HEADER_H + CELL_H),
    "-Y": (1600, HEADER_H + CELL_H),
}

# (horizontal_axis_index, vertical_axis_index) into world (X=0, Y=1, Z=2).
VIEW_AXES: dict[str, tuple[int, int]] = {
    "+Z": (0, 1), "-Z": (0, 1),
    "+X": (1, 2), "-X": (1, 2),
    "+Y": (0, 2), "-Y": (0, 2),
}

# Whether the panel's image-X is flipped relative to its world horizontal
# axis. Determined by the camera "up" vectors and view directions in
# stl_renderer.CAMERA_VIEWS:
#   +Z (cam at +Z, up=+Y): img-X = +world_X
#   -Z (cam at -Z, up=+Y): img-X = -world_X (mirrored: looking from below)
#   +X (cam at +X, up=+Z): img-X = -world_Y (right-hand rule)
#   -X (cam at -X, up=+Z): img-X = +world_Y
#   +Y (cam at +Y, up=+Z): img-X = +world_X
#   -Y (cam at -Y, up=+Z): img-X = -world_X
H_AXIS_FLIPPED: dict[str, bool] = {
    "+Z": False, "-Z": True,
    "+X": True,  "-X": False,
    "+Y": False, "-Y": True,
}

# Background colours from stl_renderer:
#   RGB panels: white (255,255,255)
#   Depth panels: #1e1e1e (30,30,30)
RGB_BG_THRESHOLD = 250
DEPTH_BG_VALUE = 30
DEPTH_BG_TOLERANCE = 4

# Tolerances
DEDUP_CENTRE_TOL_MM = 5.0
DEDUP_RADIUS_TOL_MM = 3.0
# Through-match tolerance is wider than DEDUP because we're matching a
# +Z hole against its -Z mirror image and the renderer's silhouette
# bbox can drift up to ~7mm due to aspect-ratio letterboxing inside the
# 400x400 panel. Tight enough that adjacent holes (>20mm) still resolve
# but slack enough that the same hole always matches itself.
THROUGH_MATCH_TOL_MM = 10.0


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def crop_panel(image: np.ndarray, view: str, panel_type: str) -> np.ndarray:
    """Slice the named panel out of the full 2400x1000 grid PNG.

    ``panel_type`` is ``"rgb"`` or ``"depth"``. Returned array is a
    ``PANEL_H x PANEL_W x 3`` BGR view (``cv2.imread`` convention).
    """
    if panel_type not in ("rgb", "depth"):
        raise ValueError(f"panel_type must be 'rgb' or 'depth', got {panel_type!r}")
    if view not in CELL_ORIGINS:
        raise ValueError(f"view must be one of {list(CELL_ORIGINS)}, got {view!r}")

    cx, cy = CELL_ORIGINS[view]
    x0 = cx + (0 if panel_type == "rgb" else PANEL_W)
    y0 = cy + LABEL_H
    return image[y0 : y0 + PANEL_H, x0 : x0 + PANEL_W]


# ---------------------------------------------------------------------------
# Per-panel object bbox (corrects renderer's 10% padding)
# ---------------------------------------------------------------------------
def _object_pixel_bbox_rgb(panel: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(x_min, y_min, x_max, y_max)`` of non-white pixels.
    ``None`` if the panel is entirely background."""
    is_object = ~(panel >= RGB_BG_THRESHOLD).all(axis=-1)
    return _bbox_from_mask(is_object)


def _object_pixel_bbox_depth(panel: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the depth panel's object bbox by masking out the
    ``#1e1e1e`` background (with a small tolerance for JPEG-style
    blending at the edges)."""
    diff = np.abs(panel.astype(np.int32) - DEPTH_BG_VALUE).max(axis=-1)
    is_object = diff > DEPTH_BG_TOLERANCE
    return _bbox_from_mask(is_object)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min, x_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# World bbox (recover where the part actually lives in world space)
# ---------------------------------------------------------------------------
def _world_bbox_from_geometry(
    geometry: ExtractedGeometry,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(world_min, world_max)`` in mm.

    The face extractor's ``bounding_box_mm`` is just the (X, Y, Z) SIZE.
    To know where the part LIVES in world space (since the mesh is only
    scaled, not centred), we walk every face centre and extend each axis
    by that face's own extent. Falls back to a zero-centred bbox of the
    correct size when no face data is available.
    """
    centres: list[tuple[float, float, float]] = []
    for f in geometry.planar_faces:
        centres.append(tuple(f.centre_3d_mm))
    for c in geometry.cylindrical_faces:
        centres.append(tuple(c.centre_3d_mm))

    bbox = geometry.bounding_box_mm
    if not centres:
        half = (bbox[0] / 2, bbox[1] / 2, bbox[2] / 2)
        return ((-half[0], -half[1], -half[2]), (half[0], half[1], half[2]))

    arr = np.asarray(centres, dtype=float)
    cmin = arr.min(axis=0)
    cmax = arr.max(axis=0)
    centre = (cmin + cmax) / 2.0
    half = np.asarray(bbox, dtype=float) / 2.0
    return (
        (float(centre[0] - half[0]), float(centre[1] - half[1]), float(centre[2] - half[2])),
        (float(centre[0] + half[0]), float(centre[1] + half[1]), float(centre[2] + half[2])),
    )


# ---------------------------------------------------------------------------
# Pixel -> mm conversion (corrected for 10% render padding + axis flips)
# ---------------------------------------------------------------------------
def _panel_to_world(
    view: str,
    panel_xy: tuple[float, float],
    obj_pixel_bbox: tuple[int, int, int, int],
    world_min: tuple[float, float, float],
    world_max: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert (panel_x, panel_y) in pixels to (world_h, world_v) in mm.

    Returns the world-mm coordinate along the view's HORIZONTAL and
    VERTICAL world axes (see VIEW_AXES). Image-Y is treated as growing
    downward; world vertical grows upward, so the V mapping is flipped.
    Image-X is flipped per H_AXIS_FLIPPED so the +Z and -Z views agree
    on the same world X for matching through-hole detection.
    """
    px, py = panel_xy
    x_min, y_min, x_max, y_max = obj_pixel_bbox
    w_px = max(1, x_max - x_min)
    h_px = max(1, y_max - y_min)
    u = (px - x_min) / w_px
    v = (py - y_min) / h_px
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))

    h_axis, v_axis = VIEW_AXES[view]
    h_lo, h_hi = world_min[h_axis], world_max[h_axis]
    v_lo, v_hi = world_min[v_axis], world_max[v_axis]

    if H_AXIS_FLIPPED[view]:
        world_h = h_hi - u * (h_hi - h_lo)
    else:
        world_h = h_lo + u * (h_hi - h_lo)

    # Image-Y down -> world vertical up
    world_v = v_hi - v * (v_hi - v_lo)
    return float(world_h), float(world_v)


def _panel_extent_to_world(
    view: str,
    panel_w_px: float,
    panel_h_px: float,
    obj_pixel_bbox: tuple[int, int, int, int],
    world_min: tuple[float, float, float],
    world_max: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert (width, height) in panel pixels to mm extents along the
    view's horizontal/vertical world axes (sign-free; just sizes)."""
    x_min, y_min, x_max, y_max = obj_pixel_bbox
    w_px = max(1, x_max - x_min)
    h_px = max(1, y_max - y_min)
    h_axis, v_axis = VIEW_AXES[view]
    h_world = world_max[h_axis] - world_min[h_axis]
    v_world = world_max[v_axis] - world_min[v_axis]
    return (
        float(panel_w_px / w_px * h_world),
        float(panel_h_px / h_px * v_world),
    )


# ---------------------------------------------------------------------------
# Detector 1 — missed holes / pockets from depth maps (+Z, -Z)
# ---------------------------------------------------------------------------
# RdYlBu_r at the "far" end is deep blue. In OpenCV's HSV space (H 0-179),
# blue sits around H=110. We want SATURATED + DARK blue (background is
# very dark grey, but its saturation is ~0 so it's filtered out by S>80).
HOLE_HUE_MIN = 100
HOLE_HUE_MAX = 140
HOLE_SAT_MIN = 80
HOLE_VAL_MAX = 130
HOLE_MIN_AREA_FRAC = 0.01


def _dark_blue_contours(panel: np.ndarray) -> list[np.ndarray]:
    """Return contours of dark-blue regions in a depth panel (BGR)."""
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h >= HOLE_HUE_MIN) & (h <= HOLE_HUE_MAX)
            & (s >= HOLE_SAT_MIN)
            & (v <= HOLE_VAL_MAX))
    mask_u8 = mask.astype(np.uint8) * 255
    # Close small gaps so a tessellated hole doesn't fragment.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_area = PANEL_W * PANEL_H
    return [c for c in contours if cv2.contourArea(c) >= HOLE_MIN_AREA_FRAC * panel_area]


def _already_a_known_hole(
    centre_h_mm: float,
    centre_v_mm: float,
    z_mm: float,
    diameter_mm: float,
    cylindricals: Iterable[CylindricalFace],
) -> bool:
    """Return True if a CylindricalFace at the same (X, Y) and matching
    radius already exists. Z tolerance is generous because the
    cylinder's centre_3d_mm is its axis MIDPOINT, not its top face."""
    for cyl in cylindricals:
        cx, cy, _cz = cyl.centre_3d_mm
        d_centre = float(np.hypot(cx - centre_h_mm, cy - centre_v_mm))
        d_radius = abs(cyl.radius_mm - diameter_mm / 2.0)
        if d_centre <= DEDUP_CENTRE_TOL_MM and d_radius <= DEDUP_RADIUS_TOL_MM:
            return True
        _ = z_mm  # currently unused; reserved for axis-aware dedup
    return False


def _through_hole_contours(
    rgb_panel: np.ndarray, depth_panel: np.ndarray
) -> list[np.ndarray]:
    """Through-hole footprints in a top/bottom view.

    A through-hole is a region where the camera sees through the part to
    the rendered background — in the depth panel that's ``#1e1e1e``.
    Algorithm:
      1. Mask depth-background pixels.
      2. Find connected components.
      3. Drop the component that touches the panel border (= the
         surrounding background, not a hole).
      4. Each remaining component is a hole's footprint.

    This is robust to holes of any size, including large washer-style
    cores that no fixed morphological closing can handle. The RGB panel
    is currently unused but kept in the signature in case we want to
    dual-confirm later.
    """
    _ = rgb_panel
    depth_diff = np.abs(depth_panel.astype(np.int32) - DEPTH_BG_VALUE).max(axis=-1)
    depth_is_bg = (depth_diff <= DEPTH_BG_TOLERANCE).astype(np.uint8) * 255

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        depth_is_bg, connectivity=8,
    )

    h, w = depth_is_bg.shape
    panel_area = w * h
    keep_min_px = max(20, int(0.001 * panel_area))  # 0.1% of panel, floor 20px

    contours_out: list[np.ndarray] = []
    for label in range(1, n_labels):
        x, y, cw, ch, area = stats[label]
        if area < keep_min_px:
            continue
        touches_border = (x == 0 or y == 0 or x + cw >= w or y + ch >= h)
        if touches_border:
            continue  # this is the outside background, not a hole
        component_mask = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours_out.extend(cnts)
    return contours_out


def detect_missed_holes(
    image: np.ndarray, geometry: ExtractedGeometry
) -> list[MissedFeature]:
    """Find through-holes and blind pockets in the +Z / -Z views that
    don't correspond to a known cylindrical hole.

    Through-holes show as the depth-panel BACKGROUND colour (#1e1e1e)
    inside the object silhouette — the camera sees straight through.
    Blind pockets show as anomalously dark BLUE patches (per-view depth
    normalisation puts the deepest visible point at the bluest end of
    RdYlBu_r).

    Through vs blind classification: a through-hole's footprint must
    appear in BOTH the +Z and -Z views at the same world (X, Y); the
    -Z view's image-X is flipped (see H_AXIS_FLIPPED) but world coords
    are directly comparable after our pixel->world conversion.
    """
    world_min, world_max = _world_bbox_from_geometry(geometry)
    z_top = world_max[2]

    plus_z_rgb = crop_panel(image, "+Z", "rgb")
    plus_z_depth = crop_panel(image, "+Z", "depth")
    minus_z_rgb = crop_panel(image, "-Z", "rgb")
    minus_z_depth = crop_panel(image, "-Z", "depth")

    # Use RGB silhouette for the pixel->world bbox: depth panels suffer
    # colormap quantization that can shift the apparent bbox by several
    # pixels and break the +Z/-Z world-coord match for symmetric parts.
    plus_obj = _object_pixel_bbox_rgb(plus_z_rgb) or _object_pixel_bbox_depth(plus_z_depth)
    minus_obj = _object_pixel_bbox_rgb(minus_z_rgb) or _object_pixel_bbox_depth(minus_z_depth)
    if plus_obj is None:
        return []

    # Two signals stacked: through-hole footprints + blind pocket patches.
    plus_through = _through_hole_contours(plus_z_rgb, plus_z_depth)
    plus_blind = _dark_blue_contours(plus_z_depth)

    minus_through_world: list[tuple[float, float]] = []
    if minus_obj is not None:
        for c in _through_hole_contours(minus_z_rgb, minus_z_depth):
            x, y, w, h = cv2.boundingRect(c)
            wx, wy = _panel_to_world(
                "-Z", (x + w / 2.0, y + h / 2.0), minus_obj, world_min, world_max,
            )
            minus_through_world.append((wx, wy))

    found: list[MissedFeature] = []

    def _emit(c: np.ndarray, source_kind: str) -> None:
        x, y, w, h = cv2.boundingRect(c)
        wx, wy = _panel_to_world(
            "+Z", (x + w / 2.0, y + h / 2.0), plus_obj, world_min, world_max,
        )
        w_mm, h_mm = _panel_extent_to_world(
            "+Z", float(w), float(h), plus_obj, world_min, world_max,
        )
        ratio = w / h if h > 0 else 0.0
        if 0.8 <= ratio <= 1.2:
            shape = "circle"
            diameter = (w_mm + h_mm) / 2.0
            size_mm = (float(diameter), float(diameter))
        else:
            shape = "rectangle"
            diameter = max(w_mm, h_mm)
            size_mm = (float(w_mm), float(h_mm))

        if _already_a_known_hole(wx, wy, z_top, diameter, geometry.cylindrical_faces):
            return

        if source_kind == "through_candidate":
            depth_type = "blind"
            for mwx, mwy in minus_through_world:
                if (abs(mwx - wx) <= THROUGH_MATCH_TOL_MM
                        and abs(mwy - wy) <= THROUGH_MATCH_TOL_MM):
                    depth_type = "through"
                    break
            feature_type = "hole" if depth_type == "through" else "pocket"
            note = f"+Z depth-bg patch inside silhouette; matched in -Z: {depth_type=='through'}"
        else:
            depth_type = "blind"
            feature_type = "pocket"
            note = "+Z dark-blue depth patch (per-view RdYlBu_r far end)"

        # Skip duplicates between the two signals (a through-hole can also
        # appear as dark blue at the rim where it transitions).
        for existing in found:
            ex, ey, _ez = existing.approximate_centre_mm
            if abs(ex - wx) <= DEDUP_CENTRE_TOL_MM and abs(ey - wy) <= DEDUP_CENTRE_TOL_MM:
                return

        found.append(
            MissedFeature(
                feature_type=feature_type,  # type: ignore[arg-type]
                detection_source="depth_map_top",
                approximate_centre_mm=(wx, wy, z_top),
                approximate_size_mm=size_mm,
                approximate_shape=shape,  # type: ignore[arg-type]
                depth_type=depth_type,  # type: ignore[arg-type]
                confidence="medium",
                notes=note,
            )
        )

    for c in plus_through:
        _emit(c, "through_candidate")
    for c in plus_blind:
        _emit(c, "blind_candidate")

    return found


# ---------------------------------------------------------------------------
# Detector 2 — depth-level steps from a side-view depth panel (+X)
# ---------------------------------------------------------------------------
DEPTH_STRIP_PX = 10
HUE_CLUSTER_TOL = 15
DEPTH_BG_SAT_MAX = 30
DEPTH_BG_VAL_MAX = 60
# Require a sharp hue jump between adjacent strips to declare a step
# boundary — smooth gradients (e.g. on cylinders/cones) must not be
# read as multiple discrete depth levels.
SHARP_STEP_HUE_JUMP = 35


def _is_horizontal_face(face: PlanarFace) -> bool:
    """True iff this planar face's normal is along the world Z axis
    (top or bottom face)."""
    return face.normal.direction in ("+Z", "-Z")


def _z_levels_in_planar_faces(geometry: ExtractedGeometry, tol_mm: float = 0.5) -> int:
    """Distinct Z heights among horizontal planar faces (normal +/- Z).
    Used as the baseline against which OpenCV's depth-band count is
    compared to detect missed steps."""
    zs = [f.centre_3d_mm[2] for f in geometry.planar_faces if _is_horizontal_face(f)]
    if not zs:
        return 0
    zs.sort()
    distinct = [zs[0]]
    for z in zs[1:]:
        if abs(z - distinct[-1]) > tol_mm:
            distinct.append(z)
    return len(distinct)


def detect_depth_levels(
    image: np.ndarray, geometry: ExtractedGeometry
) -> list[MissedFeature]:
    """Look at the +X side depth panel; count distinct horizontal hue
    bands; if there are more bands than known Z-levels, each unaccounted
    band becomes a ``step`` MissedFeature."""
    panel = crop_panel(image, "+X", "depth")
    obj = _object_pixel_bbox_depth(panel)
    if obj is None:
        return []
    world_min, world_max = _world_bbox_from_geometry(geometry)

    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Per-strip median hue + "is this strip mostly background?" flag.
    strips: list[tuple[int, float, bool]] = []  # (centre_y_px, median_hue, is_bg)
    for y0 in range(0, PANEL_H, DEPTH_STRIP_PX):
        y1 = min(PANEL_H, y0 + DEPTH_STRIP_PX)
        strip_h = h[y0:y1, :]
        strip_s = s[y0:y1, :]
        strip_v = v[y0:y1, :]
        is_bg = bool(
            np.median(strip_s) < DEPTH_BG_SAT_MAX
            and np.median(strip_v) < DEPTH_BG_VAL_MAX
        )
        strips.append(((y0 + y1) // 2, float(np.median(strip_h)), is_bg))

    # Split into clusters at SHARP hue jumps only — smooth gradients
    # (cylinders, cones) stay one cluster.
    clusters: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    last_hue: float | None = None
    for cy, hue, is_bg in strips:
        if is_bg:
            if current:
                clusters.append(current)
                current = []
                last_hue = None
            continue
        if last_hue is None or abs(hue - last_hue) < SHARP_STEP_HUE_JUMP:
            current.append((cy, hue))
            last_hue = hue
        else:
            clusters.append(current)
            current = [(cy, hue)]
            last_hue = hue
    if current:
        clusters.append(current)

    if len(clusters) < 2:
        return []

    known_z_levels = _z_levels_in_planar_faces(geometry)
    if len(clusters) <= known_z_levels:
        return []

    found: list[MissedFeature] = []
    for cluster in clusters:
        cy_centre = float(np.mean([p[0] for p in cluster]))
        # Convert panel-Y centre to world-Z. (We only need the V axis
        # of the +X view, which is Z; horizontal world value is meaningless.)
        _hx, world_z = _panel_to_world("+X", (PANEL_W / 2.0, cy_centre), obj, world_min, world_max)
        found.append(
            MissedFeature(
                feature_type="step",
                detection_source="side_depth_bands",
                approximate_centre_mm=(0.0, 0.0, world_z),
                approximate_size_mm=(0.0, 0.0),
                approximate_shape="unknown",
                depth_type="unknown",
                confidence="medium",
                notes=(
                    f"Side-view depth band at world Z={world_z:.1f} mm not "
                    f"accounted for by face extraction "
                    f"(face Z-levels={known_z_levels}, bands={len(clusters)})"
                ),
            )
        )

    return found


# ---------------------------------------------------------------------------
# Detector 3 — small features from the +Z RGB panel
# ---------------------------------------------------------------------------
SMALL_FEATURE_MIN_AREA_FRAC = 0.02
SMALL_FEATURE_MAX_AREA_FRAC = 0.80
PROJECTION_OVERLAP_FRAC = 0.50


def _projected_face_rect_panel(
    face: PlanarFace,
    obj_pixel_bbox: tuple[int, int, int, int],
    world_min: tuple[float, float, float],
    world_max: tuple[float, float, float],
) -> tuple[int, int, int, int] | None:
    """Project a top-facing planar face's bounding rectangle into the
    +Z RGB panel's pixel coordinates. Returns ``(x, y, w, h)`` or
    ``None`` if the face isn't horizontal."""
    if not _is_horizontal_face(face):
        return None

    cx_world, cy_world, _ = face.centre_3d_mm
    u_extent, v_extent = face.bounding_box_mm  # face-local; for +/-Z faces (X, Y)
    h_axis, v_axis = VIEW_AXES["+Z"]
    h_lo, h_hi = world_min[h_axis], world_max[h_axis]
    v_lo, v_hi = world_min[v_axis], world_max[v_axis]
    h_world_span = max(1e-9, h_hi - h_lo)
    v_world_span = max(1e-9, v_hi - v_lo)

    x_min_px, y_min_px, x_max_px, y_max_px = obj_pixel_bbox
    w_px = max(1, x_max_px - x_min_px)
    h_px = max(1, y_max_px - y_min_px)

    # World rect of this face on the +Z plane.
    x_lo_w = cx_world - u_extent / 2.0
    x_hi_w = cx_world + u_extent / 2.0
    y_lo_w = cy_world - v_extent / 2.0
    y_hi_w = cy_world + v_extent / 2.0

    def world_x_to_px(wx: float) -> int:
        u = (wx - h_lo) / h_world_span
        if H_AXIS_FLIPPED["+Z"]:
            u = 1.0 - u
        return int(round(x_min_px + u * w_px))

    def world_y_to_px(wy: float) -> int:
        # world Y increases up, image Y increases down
        v = 1.0 - (wy - v_lo) / v_world_span
        return int(round(y_min_px + v * h_px))

    px_x0 = min(world_x_to_px(x_lo_w), world_x_to_px(x_hi_w))
    px_x1 = max(world_x_to_px(x_lo_w), world_x_to_px(x_hi_w))
    px_y0 = min(world_y_to_px(y_lo_w), world_y_to_px(y_hi_w))
    px_y1 = max(world_y_to_px(y_lo_w), world_y_to_px(y_hi_w))
    return (px_x0, px_y0, max(1, px_x1 - px_x0), max(1, px_y1 - px_y0))


def _rect_overlap_frac(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """``area(a ∩ b) / area(a)`` — i.e., how much of A is inside B."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    a_area = max(1, aw * ah)
    return (iw * ih) / a_area


def detect_small_features(
    image: np.ndarray, geometry: ExtractedGeometry
) -> list[MissedFeature]:
    """Find contours in the +Z RGB panel that aren't accounted for by
    any horizontal planar face's projection."""
    panel = crop_panel(image, "+Z", "rgb")
    obj = _object_pixel_bbox_rgb(panel)
    if obj is None:
        return []
    world_min, world_max = _world_bbox_from_geometry(geometry)
    z_top = world_max[2]

    grey = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_area = PANEL_W * PANEL_H
    keep_min = SMALL_FEATURE_MIN_AREA_FRAC * panel_area
    keep_max = SMALL_FEATURE_MAX_AREA_FRAC * panel_area

    projected_rects = []
    for face in geometry.planar_faces:
        rect = _projected_face_rect_panel(face, obj, world_min, world_max)
        if rect is not None:
            projected_rects.append(rect)

    found: list[MissedFeature] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < keep_min or area > keep_max:
            continue
        rect = cv2.boundingRect(c)
        # Skip if the contour overlaps significantly with any known face.
        if any(_rect_overlap_frac(rect, pr) >= PROJECTION_OVERLAP_FRAC
               for pr in projected_rects):
            continue
        x, y, w, h = rect
        cx_panel = x + w / 2.0
        cy_panel = y + h / 2.0
        wx, wy = _panel_to_world("+Z", (cx_panel, cy_panel), obj, world_min, world_max)
        w_mm, h_mm = _panel_extent_to_world(
            "+Z", float(w), float(h), obj, world_min, world_max,
        )
        ratio = w / h if h > 0 else 0.0
        if 0.8 <= ratio <= 1.2:
            shape = "circle"
        else:
            shape = "rectangle"
        found.append(
            MissedFeature(
                feature_type="small_feature",
                detection_source="rgb_contour",
                approximate_centre_mm=(wx, wy, z_top),
                approximate_size_mm=(float(w_mm), float(h_mm)),
                approximate_shape=shape,  # type: ignore[arg-type]
                depth_type="unknown",
                confidence="low",
                notes="RGB contour on +Z panel; not overlapping any known top-face projection",
            )
        )
    return found


# ---------------------------------------------------------------------------
# Dedup + main entry point
# ---------------------------------------------------------------------------
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _deduplicate_missed(
    items: list[MissedFeature], threshold_mm: float = DEDUP_CENTRE_TOL_MM
) -> list[MissedFeature]:
    """Greedy O(n^2) dedup: if two findings are within ``threshold_mm``
    of each other, keep the one with the higher confidence (ties: keep
    first). Order-stable for the survivors."""
    survivors: list[MissedFeature] = []
    for item in items:
        replaced = False
        for i, existing in enumerate(survivors):
            d = float(np.linalg.norm(
                np.asarray(item.approximate_centre_mm)
                - np.asarray(existing.approximate_centre_mm)
            ))
            if d <= threshold_mm:
                if _CONFIDENCE_RANK[item.confidence] > _CONFIDENCE_RANK[existing.confidence]:
                    survivors[i] = item
                replaced = True
                break
        if not replaced:
            survivors.append(item)
    return survivors


def validate_with_opencv(
    render_path: str | Path, geometry: ExtractedGeometry
) -> ExtractedGeometry:
    """Run all OpenCV detectors against the 6-view render and write the
    deduplicated findings into ``geometry.missed_features``.

    The input ``geometry`` is mutated in place AND returned for chaining.
    Existing ``planar_faces`` and ``cylindrical_faces`` are NOT touched.
    """
    image = cv2.imread(str(render_path))
    if image is None:
        raise FileNotFoundError(f"Render PNG could not be read: {render_path}")

    expected = (HEADER_H + 2 * CELL_H + LEGEND_H, 3 * CELL_W)
    if image.shape[:2] != expected:
        raise ValueError(
            f"Render shape {image.shape[:2]} does not match the expected "
            f"6-view grid {expected} (HxW). Layout constants in "
            "opencv_validator must stay aligned with stl_renderer."
        )

    found: list[MissedFeature] = []
    found.extend(detect_missed_holes(image, geometry))
    found.extend(detect_depth_levels(image, geometry))
    found.extend(detect_small_features(image, geometry))

    geometry.missed_features = _deduplicate_missed(found, threshold_mm=DEDUP_CENTRE_TOL_MM)
    return geometry
