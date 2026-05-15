"""Deterministic OpenCV segmentation of the cleaned 6-view orthographic blueprint.

Replaces (or runs alongside) the LLM-driven ``face_llm_client`` pipeline.
The cleaned PNG produced by ``scripts/synthesize_clean_views.py`` has a
known layout that we can carve up and analyse with cv2 instead of asking
Claude to "interpret" pixels.

Input layout (matches ``synthesize_clean_views.render_clean_input_grid``):
    1536 x 1024 canvas. 3 columns x 2 rows of 512 x 512 cells. Each cell
    is split LEFT/RIGHT into a 256x512 grayscale depth panel and a
    256x512 silhouette panel.

Per-view extraction:
    silhouette_panel -> outer contour polygon, interior holes,
                        straight-edge segments (chord cuts), inscribed
                        circles
    depth_panel       -> ordered list of height tiers (via histogram
                        peak finding on the gray ramp)

Cross-view inference (cylinder vs box vs D-cut etc.) is the next module's
job. This one's contract: turn raw pixels into a typed, numerically
comparable per-view feature dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


GRID_COLS = 3
GRID_ROWS = 2
CANVAS_W = 1536
CANVAS_H = 1024
CELL_W = CANVAS_W // GRID_COLS  # 512
CELL_H = CANVAS_H // GRID_ROWS  # 512
HALF_W = CELL_W // 2            # 256

# View order matches synthesize_clean_views.VIEW_DIRS row-major:
#   row 0: Top, Bottom, Front
#   row 1: Back, Right, Left
VIEW_NAMES: list[str] = [
    "Top", "Bottom", "Front",
    "Back", "Right", "Left",
]

# Background detection — mirrors clean_view_recolor's threshold so a
# pixel is considered background under the same convention.
BG_THRESHOLD = 235

# Silhouette threshold — anything darker than this counts as "part body"
# in the silhouette panel. The silhouette half uses (40, 40, 48) for body
# fill on a white background, so a hard threshold at 200 is comfortable.
SILHOUETTE_BODY_MAX_LUMA = 200

# Polygon approximation tolerance as a fraction of the contour perimeter.
# Tuned so a 3-pixel-thick straight edge collapses into a single segment
# while a 100-pixel arc stays as ~10 segments.
POLY_APPROX_EPSILON_FRAC = 0.005

# An edge in the polygon counts as "straight" (i.e. a flat / chord cut /
# axis-aligned face) when its length divided by the polygon's bbox
# diagonal is at least this. Stops one-pixel jitter being labelled as a
# real flat.
STRAIGHT_EDGE_MIN_LEN_FRAC = 0.08

# Two candidate straight edges count as a parallel pair if their
# direction vectors are within this many degrees of antiparallel. 7°
# is loose enough to absorb cleanup noise, tight enough to reject
# accidental near-parallels.
PARALLEL_TOL_DEG = 7.0

# Depth histogram peak finding. After luminance histogramming we smooth
# the histogram and pick local maxima above this fraction of the highest
# bin. Each peak corresponds to one height tier in the depth panel. Two
# peaks closer than DEPTH_PEAK_MIN_LUMA_GAP are merged into one (catches
# over-segmentation on near-flat parts where small luma jitter shows up
# as multiple "tiers" that would yield 0-thickness extrudes).
# Tuned against 128105 (rectangular bar with surface features): the
# smaller-than-bar tiers were sitting ~10 luma above the bar face, so a
# gap of 8 keeps them visible while still merging neighbouring jitter.
DEPTH_PEAK_PROMINENCE_FRAC = 0.04
DEPTH_HIST_BINS = 32
DEPTH_PEAK_MIN_LUMA_GAP = 8

# Arc detection. We walk the dense outer contour (not the polygon
# approximation) and fit circles to consecutive runs of points; a run
# whose RMS residual is below this fraction of the bbox diagonal is
# emitted as an Arc. Tight enough to reject noisy stretches; loose enough
# to absorb gpt-image-2's mild boundary jitter.
ARC_FIT_RMS_FRAC = 0.012

# An arc must span at least this many degrees (computed from the chord
# subtended at the fitted centre) AND this many contour samples to be
# considered real. Filters out the dozens of tiny "arcs" you'd otherwise
# pick up at every corner of a rectangle.
ARC_MIN_SPAN_DEG = 25.0
ARC_MIN_POINT_COUNT = 12

# A full-circle detection (cv2.HoughCircles) only counts when the
# silhouette body covers at least this fraction of the inscribed circle's
# area — otherwise Hough sometimes hallucinates circles inside concave
# silhouettes.
CIRCLE_AREA_COVERAGE_MIN = 0.85

# Per-edge line vs arc classification (replaces vertex-angle-only logic).
# For each polygon edge we fit BOTH a line and a circle to the dense
# contour points spanning that edge, then choose the better fit. This
# correctly classifies both:
#   - Octagon edges (low residual to a line, high to any circle that
#     would have to absorb the corner) → all lines.
#   - Obround rounded ends (low residual to a circle, high to a line
#     because the contour bows out from the chord) → arcs even when
#     approxPolyDP only gave one polygon edge for the entire end.
LINE_FIT_RMS_FRAC = 0.004       # max RMS for a line fit, fraction of bbox diag
ARC_FIT_EDGE_RMS_FRAC = 0.012   # max RMS for an arc fit (looser than line)
LINE_VS_ARC_MARGIN = 1.4        # arc wins only if its RMS < line_rms / margin
ARC_MIN_RADIUS_FRAC = 0.05      # arcs smaller than this fraction of bbox diag are noise
ARC_EDGE_MIN_POINTS = 6         # need this many contour points to fit an arc to an edge

# Two consecutive arc edges merge into one arc when their fitted
# centres are within MERGE_CENTRE_TOL_FRAC of the bbox diagonal AND
# their fitted radii agree within MERGE_RADIUS_TOL_FRAC.
MERGE_CENTRE_TOL_FRAC = 0.05
MERGE_RADIUS_TOL_FRAC = 0.10

# When the inferencer falls back to a polyline profile, sample the dense
# contour at this stride (in pixels) so the rebuild's curve looks smooth
# instead of being a 13-vertex jagged approximation.
POLYLINE_FALLBACK_SAMPLE_STRIDE_PX = 6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StraightEdge:
    """One straight segment of the silhouette outer polygon."""
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_px: float
    angle_deg: float  # direction in [0, 180), measured from +x axis


@dataclass
class InteriorHole:
    """A white island inside the dark silhouette body."""
    centre_xy: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int]
    area_px: float
    circularity: float       # 4*pi*A / P^2, in [0, 1]
    equivalent_diameter_px: float


@dataclass
class DepthTier:
    """One band in the depth panel histogram."""
    luma_value: int          # gray level at the peak
    pixel_count: int
    relative_depth: float    # 0.0 = nearest (darkest), 1.0 = farthest


@dataclass
class CrossSectionHole:
    """An interior hole inside a CrossSection (pipe lumen, bore, slot)."""
    polygon_px: list[tuple[int, int]]
    bbox_px: tuple[int, int, int, int]
    centre_xy: tuple[float, float]
    area_px: float
    circularity: float
    equivalent_diameter_px: float
    outline: list["OutlineSegment"]


@dataclass
class CrossSection:
    """Bounded 2D cross-section of the part at a specific position along
    one axis. Produced by the slicing pass — distinct from TierRegion
    (depth-tier based) because slicing computes the actual material at
    a slice plane via per-pixel min/max from opposite views, then
    enforces a single closed OUTER contour PLUS records any interior
    holes (annular cross-sections like pipe lumens, blind bores).
    """
    axis: str                         # "X" | "Y" | "Z"
    position_norm: float              # [0, 1] in the part's bbox along axis
    polygon_px: list[tuple[int, int]]
    bbox_px: tuple[int, int, int, int]
    area_px: float
    outline: list["OutlineSegment"]
    smooth_polyline_px: list[tuple[int, int]]
    holes: list[CrossSectionHole]     # interior holes (CCOMP hierarchy)


@dataclass
class AxisZone:
    """A plateau region between two transitions in the area-vs-position
    curve. Each zone has a constant cross-section that becomes one
    extrude operation.
    """
    axis: str
    start_norm: float        # zone start position [0, 1] (inclusive)
    end_norm: float          # zone end position [0, 1] (exclusive)
    cross_section: CrossSection


@dataclass
class AxisSlices:
    """All slices along one axis, plus the per-zone cross-sections that
    survived the bounded-shape validation.
    """
    axis: str
    n_slices: int
    positions_norm: list[float]
    areas_px: list[int]
    transitions_idx: list[int]
    zones: list[AxisZone]


@dataclass
class TierRegion:
    """A closed silhouette region living at a specific depth tier.

    Used by the inferencer to build stacked extrudes: each tier gets
    one or more regions, and each region becomes one sketch on a plane
    perpendicular to the view axis at the tier's relative depth.
    """
    luma_value: int                     # depth-panel luma at this tier
    relative_depth: float               # 0.0 = nearest, 1.0 = farthest
    polygon_px: list[tuple[int, int]]   # closed loop in panel pixels
    bbox_px: tuple[int, int, int, int]
    area_px: float
    outline: list["OutlineSegment"]     # line/arc-segmented version
    smooth_polyline_px: list[tuple[int, int]]
    enclosed_holes: list[InteriorHole]  # holes wholly inside this region


@dataclass
class Arc:
    """A circular arc fitted to a run of consecutive contour points.

    ``span_deg`` close to 360 means a full circle; smaller spans are
    partial arcs (rounded corners, the curved sides of a D-cut, etc.).
    """
    centre_xy: tuple[float, float]
    radius_px: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    span_deg: float
    point_count: int
    rms_residual_px: float


@dataclass
class DetectedCircle:
    """A full circle detected directly via Hough transform on the body
    mask. Distinct from ``Arc`` (which comes from contour-fitting) because
    Hough also yields high-confidence circles that aren't part of the
    outer outline (e.g. concentric guides). Currently only outer-outline
    circles are emitted."""
    centre_xy: tuple[float, float]
    radius_px: float
    area_coverage: float     # silhouette body area / inscribed circle area


@dataclass
class OutlineSegment:
    """One segment in the line/arc-segmented outline.

    ``kind == "line"`` → ``start_xy`` and ``end_xy`` define the segment.
    ``kind == "arc"``  → ``start_xy``, ``end_xy``, ``arc_centre_xy`` and
    ``arc_radius_px`` together define the arc; ``arc_span_deg`` is its
    angular extent.

    The outline is a closed loop: the last segment's end_xy equals the
    first segment's start_xy (within a pixel).
    """
    kind: str                # "line" | "arc"
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    arc_centre_xy: tuple[float, float] | None = None
    arc_radius_px: float | None = None
    arc_span_deg: float | None = None
    arc_point_count: int | None = None
    arc_rms_residual_px: float | None = None


@dataclass
class ViewFeatures:
    name: str                # "Top" | "Bottom" | ...
    polygon_px: list[tuple[int, int]]
    bbox_px: tuple[int, int, int, int]   # x, y, w, h in panel coords
    silhouette_area_frac: float          # body area / panel area
    is_circle: bool                      # outer outline is a single ring
    circle_diameter_px: float | None     # populated when is_circle
    straight_edges: list[StraightEdge]
    parallel_pairs: list[tuple[int, int]]  # indices into straight_edges
    interior_holes: list[InteriorHole]
    depth_tier_count: int
    depth_tiers: list[DepthTier]
    arcs: list[Arc]                      # partial arcs fitted to the contour
    detected_circles: list[DetectedCircle]  # full circles via Hough
    outline: list[OutlineSegment]        # second-pass line/arc segmented loop
    smooth_polyline_px: list[tuple[int, int]]  # dense subsampled contour (fallback)
    tier_regions: list[TierRegion]       # closed regions per depth tier (far→near)


@dataclass
class OrthoFeatures:
    """All six views' features. JSON-serialisable via ``to_json``."""
    source_png: str
    views: dict[str, ViewFeatures] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "source_png": self.source_png,
            "views": {
                name: asdict(v) for name, v in self.views.items()
            },
        }
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Panel splitting
# ---------------------------------------------------------------------------
def _split_cells(rgb: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Carve the 1536x1024 canvas into (view_name, depth_panel, silhouette_panel)."""
    if rgb.shape[:2] != (CANVAS_H, CANVAS_W):
        raise ValueError(
            f"unexpected canvas {rgb.shape[:2]}, expected ({CANVAS_H}, {CANVAS_W})"
        )
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for idx, name in enumerate(VIEW_NAMES):
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        cy = row * CELL_H
        cx = col * CELL_W
        depth = rgb[cy:cy + CELL_H, cx:cx + HALF_W].copy()
        sil = rgb[cy:cy + CELL_H, cx + HALF_W:cx + CELL_W].copy()
        out.append((name, depth, sil))
    return out


# ---------------------------------------------------------------------------
# Silhouette analysis
# ---------------------------------------------------------------------------
def _silhouette_mask(panel: np.ndarray) -> np.ndarray:
    """Return a boolean mask: True where the part body lives."""
    luma = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
    body = luma < SILHOUETTE_BODY_MAX_LUMA
    # Close pinholes left by gpt-image-2 cleanup so contour detection
    # doesn't return one master contour with hundreds of internal holes
    # for what is really a near-clean silhouette. 5x5 single-iteration
    # close is a small bump over the original 3x3 to handle modest
    # render gaps from triangle-soup noisy meshes without warping the
    # silhouette boundary itself (anything more aggressive merges real
    # features or shifts edges by several mm).
    body_u8 = body.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_u8 = cv2.morphologyEx(body_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return body_u8 > 0


def _outer_contour(body_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        body_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _interior_holes(body_mask: np.ndarray) -> list[InteriorHole]:
    """White islands inside the dark silhouette = through-holes / pockets."""
    inverted = (~body_mask).astype(np.uint8) * 255
    h, w = body_mask.shape

    # Flood-fill the background from the four corners so the OUTSIDE of
    # the part doesn't get reported as a "hole". Anything left in the
    # inverted mask after that is genuinely interior.
    flood = inverted.copy()
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed[1], seed[0]] == 255:
            cv2.floodFill(flood, None, seed, 0)

    contours, _ = cv2.findContours(flood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    holes: list[InteriorHole] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 30:  # ignore single-pixel speckle
            continue
        perim = float(cv2.arcLength(c, closed=True))
        if perim <= 0:
            continue
        circularity = float(4 * np.pi * area / (perim ** 2))
        x, y, ww, hh = cv2.boundingRect(c)
        equiv_d = float(2.0 * np.sqrt(area / np.pi))
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        holes.append(InteriorHole(
            centre_xy=(cx, cy),
            bbox_xywh=(int(x), int(y), int(ww), int(hh)),
            area_px=area,
            circularity=min(circularity, 1.0),
            equivalent_diameter_px=equiv_d,
        ))
    return holes


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> bool:
    """Proper-intersection test for two segments AB and CD in 2D.

    Returns True only when the segments cross each other's interior
    (collinear-overlap and endpoint-touching cases return False so the
    polygon simplicity check doesn't false-positive on shared vertices
    between adjacent edges).
    """
    def ccw(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def _find_first_self_intersection(
    poly: np.ndarray,
) -> tuple[int, int] | None:
    """Return ``(i, j)`` for the first non-adjacent edge pair that
    crosses, or ``None`` if the polygon is simple.

    Edge ``i`` is the segment ``poly[i] -> poly[i+1]``; edge ``j`` is
    ``poly[j] -> poly[j+1]``. ``i < j`` always.
    """
    pts = np.asarray(poly, dtype=float)
    n = len(pts)
    if n < 4:
        return None
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return (i, j)
    return None


def _polygon_is_simple(poly: np.ndarray) -> bool:
    """Return True if no two non-adjacent edges of ``poly`` cross."""
    return _find_first_self_intersection(poly) is None


def _line_line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
) -> tuple[float, float] | None:
    """Intersection point of the infinite lines through ``p1-p2`` and
    ``p3-p4``. Returns ``None`` if the lines are parallel."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _repair_self_intersections(
    poly: np.ndarray, max_iters: int = 12,
) -> np.ndarray:
    """Iteratively excise polyline spikes caused by self-intersection.

    When edges ``i`` and ``j`` cross, the vertices ``i+1 .. j`` form
    a sub-loop attached to the main polygon via the two crossing
    edges. Replace those vertices with the single crossing point —
    that excises the spike while leaving the rest of the outline
    intact (so arcs further around the polygon survive untouched).

    Repeats until simple or ``max_iters`` reached. Falls back to
    the convex hull on the rare case where the repair cannot make
    progress (e.g. nested intersections that the simple linear
    excision can't resolve).
    """
    pts = [tuple(float(c) for c in p) for p in np.asarray(poly)]
    for _ in range(max_iters):
        n = len(pts)
        if n < 4:
            return np.asarray(pts, dtype=float)
        cross = _find_first_self_intersection(np.asarray(pts, dtype=float))
        if cross is None:
            return np.asarray(pts, dtype=float)
        i, j = cross
        a, b = np.asarray(pts[i]), np.asarray(pts[(i + 1) % n])
        c, d = np.asarray(pts[j]), np.asarray(pts[(j + 1) % n])
        ip = _line_line_intersection(a, b, c, d)
        # Replace vertices i+1..j with the intersection point. Walk
        # forward from index 0 through i, append the intersection,
        # then resume at j+1 through n-1.
        kept_prefix = pts[: i + 1]
        kept_suffix = pts[(j + 1) % n: n] if (j + 1) < n else []
        if ip is None:
            pts = kept_prefix + kept_suffix
        else:
            pts = kept_prefix + [ip] + kept_suffix
        if len(pts) < 3:
            break
    return np.asarray(pts, dtype=float)


def _polygon_and_edges(contour: np.ndarray) -> tuple[
    list[tuple[int, int]], tuple[int, int, int, int], list[StraightEdge],
]:
    """Approximate contour to polygon, then identify straight edges."""
    perim = cv2.arcLength(contour, closed=True)
    eps = perim * POLY_APPROX_EPSILON_FRAC
    poly = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    if not _polygon_is_simple(poly):
        repaired = _repair_self_intersections(poly)
        if len(repaired) >= 3 and _polygon_is_simple(repaired):
            poly = repaired.astype(poly.dtype)
        else:
            poly = cv2.convexHull(contour).reshape(-1, 2)
    bx, by, bw, bh = cv2.boundingRect(contour)
    bbox_diag = float(np.hypot(bw, bh))

    edges: list[StraightEdge] = []
    n = len(poly)
    for i in range(n):
        p0 = poly[i].astype(float)
        p1 = poly[(i + 1) % n].astype(float)
        seg = p1 - p0
        length = float(np.hypot(seg[0], seg[1]))
        if bbox_diag <= 0:
            continue
        if length / bbox_diag < STRAIGHT_EDGE_MIN_LEN_FRAC:
            continue
        # Direction in [0, 180) — collapse 180° flips.
        ang = float(np.degrees(np.arctan2(seg[1], seg[0])))
        if ang < 0:
            ang += 180.0
        if ang >= 180.0:
            ang -= 180.0
        edges.append(StraightEdge(
            start_xy=(float(p0[0]), float(p0[1])),
            end_xy=(float(p1[0]), float(p1[1])),
            length_px=length,
            angle_deg=ang,
        ))
    poly_list = [(int(p[0]), int(p[1])) for p in poly]
    return poly_list, (int(bx), int(by), int(bw), int(bh)), edges


def _parallel_pairs(edges: list[StraightEdge]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            diff = abs(edges[i].angle_deg - edges[j].angle_deg)
            diff = min(diff, 180.0 - diff)
            if diff <= PARALLEL_TOL_DEG:
                pairs.append((i, j))
    return pairs


def _is_circle_outline(
    contour: np.ndarray, bbox: tuple[int, int, int, int],
) -> tuple[bool, float | None]:
    """Detect a single-circle outline. Returns (is_circle, diameter_px)."""
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, closed=True))
    if perim <= 0:
        return False, None
    circularity = 4.0 * np.pi * area / (perim ** 2)
    bx, by, bw, bh = bbox
    aspect = bw / bh if bh else 0
    # Circle = high circularity AND near-1.0 aspect ratio.
    if circularity > 0.92 and 0.92 < aspect < 1.08:
        diameter = float((bw + bh) / 2.0)
        return True, diameter
    return False, None


# ---------------------------------------------------------------------------
# Depth analysis
# ---------------------------------------------------------------------------
def _depth_tiers(depth_panel: np.ndarray) -> list[DepthTier]:
    """Find distinct height tiers via k-means clustering on luma.

    Histogram-peak finding fails when two real tiers sit next to each
    other on the rising slope of a dominant peak (e.g. 128105's boss
    tier at luma~70 right next to the bar face at luma~85 — neither is
    a local max so neither shows up as a peak). K-means clusters by
    similarity in luma space, not by being a histogram extremum, so it
    can recover both centroids.

    Algorithm:
      1. Try K = MAX_K, MAX_K-1, ..., 1 and pick the largest K where
         every cluster has at least MIN_CLUSTER_FRAC of foreground
         pixels AND adjacent cluster centres differ by at least
         DEPTH_PEAK_MIN_LUMA_GAP.
      2. Each surviving centroid becomes one DepthTier.
    """
    luma = cv2.cvtColor(depth_panel, cv2.COLOR_RGB2GRAY)
    fg = luma < BG_THRESHOLD
    if not fg.any():
        return []
    values = luma[fg].astype(np.float32).reshape(-1, 1)
    span = max(float(values.max() - values.min()), 1.0)
    fg_count = len(values)

    MAX_K = 5
    MIN_CLUSTER_FRAC = 0.015
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)

    chosen_k = 1
    chosen_centres: list[float] = []
    chosen_counts: list[int] = []
    for k in range(min(MAX_K, fg_count), 0, -1):
        if k == 1:
            chosen_k = 1
            chosen_centres = [float(values.mean())]
            chosen_counts = [fg_count]
            break
        try:
            _, labels, centres = cv2.kmeans(
                values, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error:
            continue
        centres_flat = centres.flatten()
        order = np.argsort(centres_flat)
        sorted_centres = centres_flat[order].tolist()
        # Cluster sizes in the sorted order.
        labels_flat = labels.flatten()
        sorted_counts = [
            int((labels_flat == int(order[i])).sum()) for i in range(k)
        ]
        # Validate: every cluster has enough pixels AND is well-separated.
        ok = all(c / fg_count >= MIN_CLUSTER_FRAC for c in sorted_counts)
        if ok:
            for a, b in zip(sorted_centres, sorted_centres[1:]):
                if b - a < DEPTH_PEAK_MIN_LUMA_GAP:
                    ok = False
                    break
        if ok:
            chosen_k = k
            chosen_centres = sorted_centres
            chosen_counts = sorted_counts
            break

    tiers: list[DepthTier] = []
    for centre, count in zip(chosen_centres, chosen_counts):
        rel = float((centre - values.min()) / span)
        tiers.append(DepthTier(
            luma_value=int(round(centre)),
            pixel_count=count,
            relative_depth=rel,
        ))
    return tiers


# ---------------------------------------------------------------------------
# Arc + circle detection
# ---------------------------------------------------------------------------
def _fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic least-squares circle fit. Returns (cx, cy, r, rms_residual)."""
    if len(points) < 3:
        return 0.0, 0.0, 0.0, float("inf")
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return cx, cy, 0.0, float("inf")
    r = float(np.sqrt(r2))
    residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return cx, cy, r, rms


def _arc_span_deg(centre: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    """Angular span (degrees) subtended at ``centre`` from ``start`` to ``end``."""
    a0 = np.degrees(np.arctan2(start[1] - centre[1], start[0] - centre[0]))
    a1 = np.degrees(np.arctan2(end[1] - centre[1], end[0] - centre[0]))
    diff = abs(a1 - a0) % 360.0
    return min(diff, 360.0 - diff)


def _detect_arcs(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> list[Arc]:
    """Walk the dense contour and greedily fit circles to consecutive runs.

    Algorithm:
      1. Slide a small starting window along the contour.
      2. For each starting position, fit a circle and grow the window
         while the RMS residual stays below ARC_FIT_RMS_FRAC * bbox_diag.
      3. When growth stalls, emit the run as an Arc if it passes the
         span and point-count gates, then jump past it and repeat.

    Greedy is good enough for clean ortho silhouettes — there's no
    overlap of meaningful arcs in a single outer contour.
    """
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < ARC_MIN_POINT_COUNT:
        return []
    bx, by, bw, bh = bbox
    bbox_diag = float(np.hypot(bw, bh)) or 1.0
    rms_tol = ARC_FIT_RMS_FRAC * bbox_diag

    arcs: list[Arc] = []
    # Walk with wraparound. We use a contiguous-index window on a doubled
    # array so we can grow past the end of the contour.
    pts2 = np.vstack([pts, pts])
    i = 0
    while i < n:
        # Initial window
        window = ARC_MIN_POINT_COUNT
        if i + window > 2 * n:
            break
        cx, cy, r, rms = _fit_circle(pts2[i:i + window])
        if rms > rms_tol or r > 4 * bbox_diag:
            i += 1
            continue

        # Grow while the fit stays within tolerance.
        best = (cx, cy, r, rms, window)
        while i + window + 4 <= 2 * n:
            window += 4
            cx, cy, r, rms = _fit_circle(pts2[i:i + window])
            if rms > rms_tol or r > 4 * bbox_diag:
                window -= 4  # undo last growth
                break
            best = (cx, cy, r, rms, window)

        cx, cy, r, rms, win = best
        if win < ARC_MIN_POINT_COUNT:
            i += 1
            continue

        start = (float(pts2[i, 0]), float(pts2[i, 1]))
        end = (float(pts2[i + win - 1, 0]), float(pts2[i + win - 1, 1]))
        span = _arc_span_deg((cx, cy), start, end)
        if span < ARC_MIN_SPAN_DEG:
            i += 1
            continue

        arcs.append(Arc(
            centre_xy=(cx, cy),
            radius_px=r,
            start_xy=start,
            end_xy=end,
            span_deg=span,
            point_count=int(win),
            rms_residual_px=rms,
        ))
        i += win  # jump past the consumed run
    return arcs


def _detect_full_circles(body_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> list[DetectedCircle]:
    """Hough-detect full circles inside the body mask."""
    h, w = body_mask.shape
    bx, by, bw, bh = bbox
    bbox_diag = float(np.hypot(bw, bh)) or 1.0
    bm = body_mask.astype(np.uint8) * 255
    # Param tuning: minDist large so we don't get duplicate detections;
    # min/max radius bounded by bbox so we don't return absurd circles.
    min_r = max(8, int(bbox_diag * 0.10))
    max_r = max(min_r + 1, int(bbox_diag * 0.55))
    blurred = cv2.GaussianBlur(bm, (5, 5), 1.0)
    hits = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(bbox_diag * 0.5),
        param1=120,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if hits is None:
        return []
    out: list[DetectedCircle] = []
    body_area = float(body_mask.sum())
    for c in hits[0, :]:
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        if r <= 0:
            continue
        circle_area = float(np.pi * r * r)
        coverage = body_area / circle_area if circle_area > 0 else 0.0
        if coverage < CIRCLE_AREA_COVERAGE_MIN:
            continue
        out.append(DetectedCircle(
            centre_xy=(cx, cy),
            radius_px=r,
            area_coverage=coverage,
        ))
    return out


# ---------------------------------------------------------------------------
# Line/arc segmentation — a SECOND PASS on top of approxPolyDP.
# ---------------------------------------------------------------------------
# Why a second pass: approxPolyDP forces every curve into straight
# segments. A clean rebuild of a circle-with-flats wants two flats AND
# two arcs, not 13 short straight pieces. The pass below reconciles the
# sparse polygon with the dense contour by looking at the angle change
# at each polygon vertex:
#
#   - Sharp tangent change (>= CORNER_ANGLE_DEG_MIN) → it's a real
#     corner. The two adjacent polygon edges become two LINE segments.
#   - Smooth tangent change (< CORNER_ANGLE_DEG_MIN) → the vertex is
#     riding a curve. Group consecutive smooth vertices into an arc run
#     bounded by the surrounding corners; fit a circle to the dense
#     contour points spanned by the run; emit one ARC for the whole run.
def _nearest_contour_index(contour_pts: np.ndarray, target_xy: tuple[int, int]) -> int:
    diffs = contour_pts - np.array(target_xy, dtype=float)
    d2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    return int(np.argmin(d2))


def _fit_line(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Total-least-squares line fit. Returns (start_proj, end_proj, rms_perp_residual).

    Projects the first and last points onto the fitted line so the caller
    can emit a clean straight segment. RMS is the perpendicular distance
    from each input point to the fitted line.
    """
    if len(points) < 2:
        return (0.0, 0.0), (0.0, 0.0), float("inf")
    centroid = points.mean(axis=0)
    centred = points - centroid
    # Principal direction = first eigenvector of the covariance matrix.
    cov = centred.T @ centred / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    # Perpendicular distances of each point.
    normal = np.array([-direction[1], direction[0]])
    perp = centred @ normal
    rms = float(np.sqrt((perp ** 2).mean()))
    # Project first and last input points onto the line.
    t_first = (points[0] - centroid) @ direction
    t_last = (points[-1] - centroid) @ direction
    start_proj = (float(centroid[0] + t_first * direction[0]),
                  float(centroid[1] + t_first * direction[1]))
    end_proj = (float(centroid[0] + t_last * direction[0]),
                float(centroid[1] + t_last * direction[1]))
    return start_proj, end_proj, rms


def _classify_edge(
    contour_pts: np.ndarray,
    bbox_diag: float,
    i0: int,
    i1: int,
) -> OutlineSegment:
    """Decide whether the contour run from ``i0`` to ``i1`` is best
    described as a line or a circular arc, and emit the corresponding
    OutlineSegment with endpoints snapped onto the fitted primitive.

    The contour is treated as a closed loop, so wrap-around runs (i1 < i0)
    are concatenated.
    """
    if i1 <= i0:
        run = np.vstack([contour_pts[i0:], contour_pts[:i1 + 1]])
    else:
        run = contour_pts[i0:i1 + 1]
    raw_start = (float(contour_pts[i0, 0]), float(contour_pts[i0, 1]))
    raw_end = (float(contour_pts[i1, 0]), float(contour_pts[i1, 1]))

    # Always try a line fit; it's the fallback.
    line_start, line_end, line_rms = _fit_line(run)

    # Arc fit only when there's enough material AND a sensible result.
    arc_ok = False
    arc_centre = (0.0, 0.0)
    arc_radius = 0.0
    arc_rms = float("inf")
    if len(run) >= ARC_EDGE_MIN_POINTS:
        cx, cy, r, rms = _fit_circle(run)
        # Reject tiny arcs (artefacts of cleaning) and absurd radii.
        if r >= ARC_MIN_RADIUS_FRAC * bbox_diag and r < 8.0 * bbox_diag:
            arc_ok = True
            arc_centre = (cx, cy)
            arc_radius = r
            arc_rms = rms

    line_acceptable = line_rms <= LINE_FIT_RMS_FRAC * bbox_diag
    arc_acceptable = arc_ok and arc_rms <= ARC_FIT_EDGE_RMS_FRAC * bbox_diag

    # Decide. Prefer line (simpler primitive) unless the arc fit is
    # MARKEDLY better. This is the key bias that lets octagons stay as
    # 8 lines instead of being averaged into one circle.
    use_arc = (
        arc_acceptable
        and (not line_acceptable or arc_rms * LINE_VS_ARC_MARGIN < line_rms)
    )

    if use_arc:
        span = _arc_span_deg(arc_centre, raw_start, raw_end)
        # Reject "noise arcs" — short angular spans (< ARC_MIN_SPAN_DEG)
        # are usually gpt-image-2 cleanup artifacts at polygon corners
        # (slight rounding) rather than real curved features. Demote them
        # back to lines so an octagon doesn't pick up phantom arcs at
        # every corner that would inflate its arc-perimeter fraction.
        if span < ARC_MIN_SPAN_DEG:
            use_arc = False

    if use_arc:
        span = _arc_span_deg(arc_centre, raw_start, raw_end)
        # Snap endpoints onto the fitted circle so adjacent arcs/lines
        # share an exact endpoint when they meet at a corner.
        def _snap(pt):
            dx = pt[0] - arc_centre[0]
            dy = pt[1] - arc_centre[1]
            mag = (dx * dx + dy * dy) ** 0.5 or 1.0
            return (arc_centre[0] + dx / mag * arc_radius,
                    arc_centre[1] + dy / mag * arc_radius)
        return OutlineSegment(
            kind="arc",
            start_xy=_snap(raw_start),
            end_xy=_snap(raw_end),
            arc_centre_xy=arc_centre,
            arc_radius_px=arc_radius,
            arc_span_deg=span,
            arc_point_count=int(len(run)),
            arc_rms_residual_px=arc_rms,
        )
    return OutlineSegment(
        kind="line",
        start_xy=line_start if line_acceptable else raw_start,
        end_xy=line_end if line_acceptable else raw_end,
    )


def _merge_consecutive_arcs(segments: list[OutlineSegment], bbox_diag: float) -> list[OutlineSegment]:
    """Merge adjacent arcs that share a fitted centre + radius.

    approxPolyDP tends to put 2-3 vertices on a single rounded end of an
    obround; if every edge becomes its own arc the rebuild stutters. This
    pass collapses runs of compatible arcs into one.
    """
    if not segments:
        return segments
    out: list[OutlineSegment] = []
    centre_tol = MERGE_CENTRE_TOL_FRAC * bbox_diag
    radius_tol = MERGE_RADIUS_TOL_FRAC * bbox_diag
    for seg in segments:
        if (out and seg.kind == "arc" and out[-1].kind == "arc"
                and out[-1].arc_centre_xy and seg.arc_centre_xy
                and out[-1].arc_radius_px and seg.arc_radius_px):
            cprev = out[-1].arc_centre_xy
            cnow = seg.arc_centre_xy
            dc = ((cprev[0] - cnow[0]) ** 2 + (cprev[1] - cnow[1]) ** 2) ** 0.5
            dr = abs(out[-1].arc_radius_px - seg.arc_radius_px)
            if dc <= centre_tol and dr <= radius_tol:
                # Merge: average the centres weighted by point count;
                # extend the angular span; new end_xy = seg.end_xy.
                w_a = out[-1].arc_point_count or 1
                w_b = seg.arc_point_count or 1
                merged_centre = (
                    (cprev[0] * w_a + cnow[0] * w_b) / (w_a + w_b),
                    (cprev[1] * w_a + cnow[1] * w_b) / (w_a + w_b),
                )
                merged_radius = (
                    (out[-1].arc_radius_px * w_a + seg.arc_radius_px * w_b)
                    / (w_a + w_b)
                )
                merged_span = _arc_span_deg(
                    merged_centre, out[-1].start_xy, seg.end_xy,
                )
                out[-1] = OutlineSegment(
                    kind="arc",
                    start_xy=out[-1].start_xy,
                    end_xy=seg.end_xy,
                    arc_centre_xy=merged_centre,
                    arc_radius_px=merged_radius,
                    arc_span_deg=merged_span,
                    arc_point_count=w_a + w_b,
                    arc_rms_residual_px=max(
                        out[-1].arc_rms_residual_px or 0.0,
                        seg.arc_rms_residual_px or 0.0,
                    ),
                )
                continue
        out.append(seg)
    return out


def _segment_line_arc(
    contour: np.ndarray, polygon: list[tuple[int, int]],
) -> list[OutlineSegment]:
    """Per-edge line vs arc segmentation.

    For each polygon edge, fit BOTH a line and a circle to the dense
    contour points spanning that edge and pick the better fit. Then
    merge adjacent arcs that share a centre + radius (so an obround's
    rounded end becomes one arc, not two). Endpoints are snapped onto
    the fitted primitive so consecutive segments connect cleanly.
    """
    n = len(polygon)
    if n < 3:
        return []
    contour_pts = contour.reshape(-1, 2).astype(float)
    bx, by, bw, bh = cv2.boundingRect(contour)
    bbox_diag = float(np.hypot(bw, bh)) or 1.0

    # Map every polygon vertex to its nearest dense-contour index. This
    # lets us slice the contour cleanly per polygon edge.
    poly_indices = [_nearest_contour_index(contour_pts, p) for p in polygon]

    raw_segments: list[OutlineSegment] = []
    for i in range(n):
        i0 = poly_indices[i]
        i1 = poly_indices[(i + 1) % n]
        seg = _classify_edge(contour_pts, bbox_diag, i0, i1)
        raw_segments.append(seg)

    # Merge adjacent arcs that came from the same underlying curve.
    return _merge_consecutive_arcs(raw_segments, bbox_diag)


def _smooth_polyline_from_contour(contour: np.ndarray, stride_px: int) -> list[tuple[int, int]]:
    """Subsample the dense contour at ``stride_px`` so a polyline-based
    profile renders as a smooth curve, not a 13-vertex chain."""
    pts = contour.reshape(-1, 2)
    if len(pts) <= 4:
        return [(int(p[0]), int(p[1])) for p in pts]
    # Cumulative arclength sampling so the spacing stays roughly constant
    # even on contours where opencv emits unevenly distributed points.
    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return [(int(p[0]), int(p[1])) for p in pts]
    n_samples = max(8, int(total // stride_px))
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    sampled = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            sampled.append(pts[-1])
            continue
        a = (t - cum[j]) / max(seg_len[j], 1e-9)
        x = pts[j][0] * (1 - a) + pts[j + 1][0] * a
        y = pts[j][1] * (1 - a) + pts[j + 1][1] * a
        sampled.append((int(round(x)), int(round(y))))
    return sampled


# ---------------------------------------------------------------------------
# Per-tier region extraction
# ---------------------------------------------------------------------------
# Each detected depth peak corresponds to a planar face at a specific
# distance from the camera. Thresholding the depth panel around that
# luma value isolates the pixels at THAT face's height; running contours
# on the resulting mask gives us the closed silhouette region(s) of that
# face. Stacking these from far→near reconstructs the engineering intent
# of a multi-step extrude.
TIER_LUMA_BAND_HALF = 6         # ± luma values around the tier centre (tightened from 12)
TIER_MIN_AREA_FRAC = 0.005      # keep smaller features (lowered from 0.01)
TIER_REGION_POLY_EPSILON = 0.005


def _tier_polygon(contour: np.ndarray) -> tuple[
    list[tuple[int, int]], tuple[int, int, int, int],
]:
    perim = float(cv2.arcLength(contour, closed=True))
    eps = max(1.0, perim * TIER_REGION_POLY_EPSILON)
    poly = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    # Same surgical spike-removal as _polygon_and_edges: tier-region
    # polygons feed _segment_line_arc and become the extrude profile,
    # so they must be simple. Surgical repair preserves arcs elsewhere
    # in the outline; convex hull is the last-ditch fallback.
    if not _polygon_is_simple(poly):
        repaired = _repair_self_intersections(poly)
        if len(repaired) >= 3 and _polygon_is_simple(repaired):
            poly = repaired.astype(poly.dtype)
        else:
            poly = cv2.convexHull(contour).reshape(-1, 2)
    bx, by, bw, bh = cv2.boundingRect(contour)
    return ([(int(p[0]), int(p[1])) for p in poly],
            (int(bx), int(by), int(bw), int(bh)))


def _hole_inside_polygon(hole: InteriorHole, polygon_px: list[tuple[int, int]]) -> bool:
    """Point-in-polygon test for the hole's centroid against a polygon."""
    if len(polygon_px) < 3:
        return False
    poly_np = np.array(polygon_px, dtype=np.int32)
    cx, cy = hole.centre_xy
    return cv2.pointPolygonTest(poly_np, (float(cx), float(cy)), False) > 0


def _extract_tier_regions(
    depth_panel: np.ndarray,
    silhouette_mask: np.ndarray,
    tiers: list[DepthTier],
    interior_holes: list[InteriorHole],
) -> list[TierRegion]:
    if not tiers:
        return []
    luma = cv2.cvtColor(depth_panel, cv2.COLOR_RGB2GRAY)
    silhouette_area = float(silhouette_mask.sum()) or 1.0
    out: list[TierRegion] = []
    # Sort tiers FAR→NEAR (largest luma_value first; large luma = lighter
    # gray = farther from camera by the synthesise step's encoding).
    for tier in sorted(tiers, key=lambda t: -t.luma_value):
        lo = max(0, tier.luma_value - TIER_LUMA_BAND_HALF)
        hi = min(255, tier.luma_value + TIER_LUMA_BAND_HALF)
        band_mask = ((luma >= lo) & (luma <= hi) & silhouette_mask).astype(np.uint8) * 255
        # Close pinholes from gpt-image-2 cleanup so each tier is one
        # connected region per geometric face, not 50 specks.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        for c in contours:
            area = float(cv2.contourArea(c))
            if area / silhouette_area < TIER_MIN_AREA_FRAC:
                continue
            poly_px, bbox = _tier_polygon(c)
            outline = _segment_line_arc(c, poly_px)
            smooth = _smooth_polyline_from_contour(
                c, POLYLINE_FALLBACK_SAMPLE_STRIDE_PX,
            )
            enclosed = [h for h in interior_holes if _hole_inside_polygon(h, poly_px)]
            out.append(TierRegion(
                luma_value=tier.luma_value,
                relative_depth=tier.relative_depth,
                polygon_px=poly_px,
                bbox_px=bbox,
                area_px=area,
                outline=outline,
                smooth_polyline_px=smooth,
                enclosed_holes=enclosed,
            ))
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def segment_view(name: str, depth_panel: np.ndarray, silhouette_panel: np.ndarray) -> ViewFeatures:
    body = _silhouette_mask(silhouette_panel)
    contour = _outer_contour(body)
    panel_area = float(silhouette_panel.shape[0] * silhouette_panel.shape[1])
    if contour is None:
        return ViewFeatures(
            name=name,
            polygon_px=[],
            bbox_px=(0, 0, 0, 0),
            silhouette_area_frac=0.0,
            is_circle=False,
            circle_diameter_px=None,
            straight_edges=[],
            parallel_pairs=[],
            interior_holes=[],
            depth_tier_count=0,
            depth_tiers=[],
            arcs=[],
            detected_circles=[],
            outline=[],
            smooth_polyline_px=[],
            tier_regions=[],
        )

    poly, bbox, edges = _polygon_and_edges(contour)
    pairs = _parallel_pairs(edges)
    is_circ, diam = _is_circle_outline(contour, bbox)
    holes = _interior_holes(body)
    tiers = _depth_tiers(depth_panel)
    sil_frac = float(body.sum()) / panel_area
    arcs = _detect_arcs(contour, bbox)
    full_circles = _detect_full_circles(body, bbox)

    # Promote: if Hough finds a full circle that closely matches the
    # bbox extents, override the polygon-based circle classification.
    if not is_circ and full_circles:
        biggest = max(full_circles, key=lambda c: c.radius_px)
        if biggest.area_coverage >= CIRCLE_AREA_COVERAGE_MIN:
            is_circ = True
            diam = biggest.radius_px * 2.0

    outline = _segment_line_arc(contour, poly)
    smooth_poly = _smooth_polyline_from_contour(
        contour, POLYLINE_FALLBACK_SAMPLE_STRIDE_PX,
    )
    tier_regions = _extract_tier_regions(depth_panel, body, tiers, holes)

    return ViewFeatures(
        name=name,
        polygon_px=poly,
        bbox_px=bbox,
        silhouette_area_frac=sil_frac,
        is_circle=is_circ,
        circle_diameter_px=diam,
        straight_edges=edges,
        parallel_pairs=pairs,
        interior_holes=holes,
        depth_tier_count=len(tiers),
        depth_tiers=tiers,
        arcs=arcs,
        detected_circles=full_circles,
        outline=outline,
        smooth_polyline_px=smooth_poly,
        tier_regions=tier_regions,
    )


# ---------------------------------------------------------------------------
# Slice-based axis volume extraction
# ---------------------------------------------------------------------------
# Why slice the part: depth-tier histogram peak finding picks "where most
# pixels share a depth value" — but a sketch plane really lives wherever
# the cross-sectional AREA has a sudden change. A 5-step staircase has
# 5 area plateaus separated by 4 transitions; tier peaks on the depth
# histogram lump them together if their luma values are similar.
#
# To slice, we need per-pixel min/max along the slicing axis. Two
# opposite views give us that: the "low" view (e.g. Top of Z axis: high z
# = small luma = nearer +Z camera) tells us the part's TOP at each (x, y);
# the "high" view (Bottom) tells us its BOTTOM. The pixel column at (x, y)
# spans [bot, top]. Slicing at z_k gives the cross-section as the set of
# (x, y) where bot < z_k < top.
#
# Bounded-shape enforcement: each slice mask is morphologically closed
# to fill rendering gaps, then ``cv2.findContours`` with RETR_EXTERNAL
# extracts only the outer boundary. Slices below an area threshold are
# rejected so a single noisy pixel run never becomes an "extrude".

# Slicing tunables.
SLICE_LUMA_NEAR = 60     # synthesize_clean_views encodes nearest = luma 60
SLICE_LUMA_FAR = 230     # farthest = luma 230 (60 + 170*1.0)
SLICE_LUMA_RANGE = SLICE_LUMA_FAR - SLICE_LUMA_NEAR
SLICE_DEFAULT_N = 60     # ~60 slices gives a 1.6% resolution along an axis
SLICE_AREA_MIN_FRAC = 0.005      # cross-section area must be >= 0.5% of panel
# Transition threshold: |dA| must exceed this fraction of MAX_AREA (not
# max |dA|). Scaling with max(|dA|) trips on every plateau jitter; scaling
# with max(area) only fires on real cross-section changes (e.g. the
# 13000→7000 drop between two stepped octagonal tiers is 46% of max area
# = a clear transition; a 13238→13200 jitter on a flat tier is 0.3%).
# Threshold stays at 0.04 of max area, but combined with the wider
# 4-slice lag (below) it now picks up smooth/tapered transitions a
# pure 1-slice diff would miss. The 4-slice window accumulates the
# area change across the gradient so the threshold compares the
# total step, not the per-bin slope.
SLICE_TRANSITION_FRAC_OF_MAX_AREA = 0.04
# After detection, merge transitions that sit within this many slices of
# each other into ONE (their centre). A multi-slice "ramp" between two
# tiers shows up as several consecutive transitions; without merging we'd
# emit 4 zones for a 2-tier transition.
SLICE_TRANSITION_MERGE_RADIUS = 3
# Smoothing kernel = 3: enough to suppress single-bin noise, NOT enough
# to wash out a real one-step area drop. Larger kernels (7+) blur the
# transition between adjacent tiers into a slow ramp the threshold then
# misses entirely (113K's tier change at slice 12 disappeared with k=7).
SLICE_SMOOTH_KERNEL = 3
SLICE_MORPH_KERNEL = 5
# Regular polygon detection. A regular N-gon has equal edge lengths,
# equal interior angles, and vertices equidistant from the centroid.
# When a closed outline matches this pattern we emit a perfect regular
# polygon (computed from centroid + radius + N + rotation) instead of
# the noisy approxPolyDP vertex list. STEP exports preserve the
# parametric polygon constraint instead of discretizing it.
REGULAR_POLY_MIN_SIDES = 5            # 3/4-sided handled by triangle/rect classifiers
REGULAR_POLY_MAX_SIDES = 12
# Two strict gates: vertices must lie on a circle (radius stdev tight)
# AND must be spread roughly evenly around it (angular gap ratio tight).
# The angular check is what distinguishes a regular polygon from a
# D-cut whose vertices cluster on the curved arcs with big angular
# gaps where the flat cuts sit.
REGULAR_POLY_EDGE_LEN_TOL = 0.35      # loose: the snap rebuild fixes it
REGULAR_POLY_RADIUS_TOL = 0.13        # vertices must lie on a circle
REGULAR_POLY_ANGLE_TOL_DEG = 15.0     # loose: the snap rebuild fixes it
# Largest angular gap between consecutive vertices must be at most this
# multiple of the expected uniform gap (360/N). Loosened to 1.7 — the
# tighter 1.4 over-rejected genuine octagons that gpt-image-2 cleanup
# wobbled a bit. False D-cut positives are still caught by the
# epsilon cap (see inferencer: only eps_frac <= 0.020 considered).
REGULAR_POLY_ANG_GAP_RATIO_MAX = 1.7


# Cross-sections may contain INTERIOR HOLES (a slice through a pipe is
# an annulus, not a disc). CrossSection records hole contours alongside
# the outer outline; the inferencer turns each hole into a cut operation
# after the zone extrude.
SLICE_HOLE_MIN_AREA_FRAC = 0.001    # hole must be >= 0.1% of panel
SLICE_HOLE_MIN_AREA_VS_OUTER = 0.005  # hole must be >= 0.5% of its outer contour
# Drop zones whose cross-section area is within this fraction of the
# previous zone's area — they're really the same plateau split by a
# spurious transition that survived the merge.
# Lowered 0.12 → 0.05: was merging adjacent stepped tiers whose area
# differed by 5-12% (e.g. 000035's boss has 4 stepped tiers each ~10%
# smaller than the previous, all of which were collapsing into one).
SLICE_ZONE_DEDUP_AREA_FRAC = 0.05

# "Transition zone" filter: a zone is treated as a smooth-gradient artifact
# (not a real tier) when its height fraction is below this threshold AND
# its area sits between its neighbours' areas. 002354's L-shape produced a
# spurious thin (~7%-of-axis) middle zone whose cross-section was halfway
# between the wide base and the tall protrusion — this is the cleanup
# render's gradient ramp at a sharp step, not a physical tier. We merge
# such zones into the area-closer neighbour.
SLICE_TRANSITION_MAX_HEIGHT_FRAC = 0.12
# How much "between" the neighbour areas the candidate's area must be: it
# qualifies as transitional when min(prev, next) < area < max(prev, next).
# (No extra tolerance — strict monotonic-between only.)


def _luma_to_depth_norm(luma: np.ndarray) -> np.ndarray:
    """Convert depth panel luma to normalized depth in [0, 1]
    (0 = nearest to camera, 1 = farthest)."""
    return np.clip(
        (luma.astype(np.float32) - SLICE_LUMA_NEAR) / SLICE_LUMA_RANGE,
        0.0, 1.0,
    )


def _axis_volume(
    panels: dict[str, tuple[np.ndarray, np.ndarray]], axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-pixel (low_norm, high_norm, valid_mask) along ``axis``.

    The "low" output is the smaller-coordinate boundary along the axis
    (e.g. for Z it's the part's BOTTOM at each (x, y) column). "high" is
    the larger-coordinate boundary (Z TOP). Both are normalized to [0, 1]
    where 0 = the part's most-negative-axis extent and 1 = the most-
    positive-axis extent.

    The opposite-view alignment per axis (read from
    ``synthesize_clean_views.VIEW_DIRS``):

      Z: Top (camera +Z, up=+Y, right=+X)     ↔ Bottom (up=-Y, right=+X)
         → flip Bottom vertically (axis 0)
      Y: Front (camera -Y, up=+Z, right=+X)   ↔ Back (up=+Z, right=-X)
         → flip Back horizontally (axis 1)
      X: Right (camera -X, up=+Z, right=-Y)   ↔ Left (up=+Z, right=+Y)
         → flip Left horizontally (axis 1)
    """
    if axis == "Z":
        lo_v, hi_v = "Top", "Bottom"
        flip_axis = 0
        # Top: small luma = high z. Bottom: small luma = low z.
        invert_lo = True
        invert_hi = False
    elif axis == "Y":
        lo_v, hi_v = "Front", "Back"
        flip_axis = 1
        # Front: small luma = LOW y. Back: small luma = HIGH y.
        invert_lo = False
        invert_hi = True
    elif axis == "X":
        lo_v, hi_v = "Right", "Left"
        flip_axis = 1
        # Right: small luma = LOW x. Left: small luma = HIGH x.
        invert_lo = False
        invert_hi = True
    else:
        return None

    if lo_v not in panels or hi_v not in panels:
        return None
    lo_depth, lo_sil = panels[lo_v]
    hi_depth, hi_sil = panels[hi_v]
    hi_depth_aligned = np.flip(hi_depth, axis=flip_axis)
    hi_sil_aligned = np.flip(hi_sil, axis=flip_axis)

    lo_norm = _luma_to_depth_norm(lo_depth)
    hi_norm = _luma_to_depth_norm(hi_depth_aligned)

    # When invert_lo / invert_hi is True, the lower-luma value of that
    # view corresponds to the HIGH world coordinate — flip the normalized
    # value so both arrays map to the same "0=part_min, 1=part_max" frame.
    low_world = (1.0 - lo_norm) if invert_lo else lo_norm
    high_world = (1.0 - hi_norm) if invert_hi else hi_norm

    # The "low" view sees the closer-to-camera surface, which is the
    # MAX coordinate for an inverted view (Top for Z) or the MIN for a
    # non-inverted view (Front for Y). Reconcile:
    if invert_lo:
        # Top: low_world holds the part's TOP z at each pixel = HIGH z
        # so this is actually the "high" boundary, and the other view is
        # the "low" boundary. Swap labels.
        max_per_pixel = low_world
        min_per_pixel = high_world
    else:
        # Front: low_world holds the part's LOW y = the "low" boundary.
        min_per_pixel = low_world
        max_per_pixel = high_world

    valid = lo_sil & hi_sil_aligned
    return min_per_pixel, max_per_pixel, valid


def detect_regular_polygon(
    polygon_px: list[tuple[int, int]],
) -> tuple[int, float, float, tuple[float, float]] | None:
    """Detect whether a polygon is a REGULAR N-gon (equal edges, equal
    angles, vertices equidistant from centroid).

    Returns (n_sides, radius_px, rotation_rad, centre_xy) if regular,
    else None. Rotation is the angle (rad) from centroid to vertex 0,
    measured counter-clockwise from the +X axis. Caller can reconstruct
    every vertex as ``(centre_x + r*cos(rot + 2*pi*i/n),
                          centre_y + r*sin(rot + 2*pi*i/n))``.
    """
    n = len(polygon_px)
    if n < REGULAR_POLY_MIN_SIDES or n > REGULAR_POLY_MAX_SIDES:
        return None
    pts = np.array(polygon_px, dtype=np.float64)
    centre = pts.mean(axis=0)
    rs = np.linalg.norm(pts - centre, axis=1)
    if rs.mean() <= 0:
        return None
    if rs.std() / rs.mean() > REGULAR_POLY_RADIUS_TOL:
        return None

    # Edge lengths.
    edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    if edges.mean() <= 0:
        return None
    if edges.std() / edges.mean() > REGULAR_POLY_EDGE_LEN_TOL:
        return None

    # Interior angle at each vertex should equal 180*(n-2)/n.
    expected_interior = 180.0 * (n - 2) / n
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_this = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p_this
        v2 = p_next - p_this
        n1 = np.linalg.norm(v1) or 1.0
        n2 = np.linalg.norm(v2) or 1.0
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos)))
        if abs(angle - expected_interior) > REGULAR_POLY_ANGLE_TOL_DEG:
            return None

    # Angular uniformity: vertices should spread evenly around the
    # centroid. D-cuts have low radius stdev too (vertices on a circle)
    # but their angular gaps are far from uniform — tight clusters on
    # the curved arcs and large gaps across the flat cuts.
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    angles_sorted = np.sort(angles)
    gaps = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))
    expected_gap = 2 * np.pi / n
    if gaps.max() > REGULAR_POLY_ANG_GAP_RATIO_MAX * expected_gap:
        return None

    # Rotation = angle to vertex 0 (in image coords; caller converts to
    # workplane convention with the y-flip).
    v0 = pts[0] - centre
    rotation = float(np.arctan2(v0[1], v0[0]))
    return (
        int(n),
        float(rs.mean()),
        rotation,
        (float(centre[0]), float(centre[1])),
    )


def _polygon_from_contour(c: np.ndarray) -> tuple[
    list[tuple[int, int]], tuple[int, int, int, int],
]:
    perim = float(cv2.arcLength(c, closed=True))
    eps = max(1.0, perim * POLY_APPROX_EPSILON_FRAC)
    poly = cv2.approxPolyDP(c, eps, closed=True).reshape(-1, 2)
    bx, by, bw, bh = cv2.boundingRect(c)
    return ([(int(p[0]), int(p[1])) for p in poly],
            (int(bx), int(by), int(bw), int(bh)))


def _slice_axis(
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    axis: str, n_slices: int,
) -> AxisSlices | None:
    vol = _axis_volume(panels, axis)
    if vol is None:
        return None
    lo, hi, valid = vol
    panel_area = float(lo.size)
    area_min = panel_area * SLICE_AREA_MIN_FRAC

    positions = np.linspace(0.02, 0.98, n_slices)
    areas: list[int] = []
    masks: list[np.ndarray] = []
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (SLICE_MORPH_KERNEL, SLICE_MORPH_KERNEL),
    )
    for k in positions:
        in_part = (lo < k) & (k < hi) & valid
        m = (in_part.astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        bool_m = m > 0
        masks.append(bool_m)
        areas.append(int(bool_m.sum()))

    areas_arr = np.array(areas, dtype=np.float32)
    if SLICE_SMOOTH_KERNEL > 1:
        smoothed = np.convolve(
            areas_arr,
            np.ones(SLICE_SMOOTH_KERNEL) / SLICE_SMOOTH_KERNEL,
            mode="same",
        )
    else:
        smoothed = areas_arr
    # Use a 2-slice lag for the area difference. This catches transitions
    # that are spread over a 2-bin ramp (very common: gpt-image-2 cleanup
    # softens sharp tier boundaries into a 1-2 slice gradient). A pure
    # 1-slice diff misses those because the per-bin change is smaller.
    da_lag = np.abs(smoothed[2:] - smoothed[:-2])
    # Pad on both sides so transition indices align with slice midpoints.
    da = np.concatenate([[da_lag[0]], da_lag, [da_lag[-1]]])[:len(smoothed) - 1]
    if da.size == 0 or smoothed.max() == 0:
        transitions = []
    else:
        # Threshold against MAX AREA, not max |dA|. Plateaus with small
        # noise produce |dA| spikes too; only a real cross-section change
        # is "large" relative to the part's overall scale.
        threshold = max(
            SLICE_TRANSITION_FRAC_OF_MAX_AREA * float(smoothed.max()),
            area_min,
        )
        raw = [int(i) for i in np.where(da > threshold)[0].tolist()]
        # Merge transitions that cluster together (a single multi-slice
        # ramp between tiers shows up as 3-5 adjacent transitions; we
        # only want one boundary, placed at the cluster centre).
        transitions: list[int] = []
        for t in raw:
            if transitions and t - transitions[-1] <= SLICE_TRANSITION_MERGE_RADIUS:
                continue
            transitions.append(t)

    # Build zones: bracket of slices between consecutive transitions
    # (or between bbox start/end and the nearest transition). For each
    # zone, take the cross-section at its midpoint slice and pull both
    # the outer contour AND any interior holes (RETR_CCOMP hierarchy).
    boundaries = [0] + [t + 1 for t in transitions] + [len(positions)]
    boundaries = sorted(set(boundaries))
    panel_area_total = float(lo.size)
    hole_min_area_abs = panel_area_total * SLICE_HOLE_MIN_AREA_FRAC
    zones: list[AxisZone] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s < 1:
            continue
        midpoint = (s + e) // 2
        if midpoint >= len(masks):
            continue
        slice_mask = masks[midpoint]
        if slice_mask.sum() < area_min:
            continue
        # CCOMP gives 2-level hierarchy: outer contours at level 0,
        # their holes at level 1. The hierarchy's parent index lets
        # us pair each hole with its outer contour.
        contours, hierarchy = cv2.findContours(
            slice_mask.astype(np.uint8) * 255,
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
        )
        if not contours or hierarchy is None:
            continue
        # hierarchy shape: (1, N, 4) where each row is [next, prev, child, parent]
        hier = hierarchy[0]
        # Outer contours = parent == -1.
        outer_indices = [i for i in range(len(contours)) if hier[i][3] == -1]
        if not outer_indices:
            continue
        # Pick the largest outer contour as the "main" cross-section.
        main_idx = max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))
        main_c = contours[main_idx]
        c_area = float(cv2.contourArea(main_c))
        if c_area < area_min:
            continue
        poly_px, bbox = _polygon_from_contour(main_c)
        outline = _segment_line_arc(main_c, poly_px)
        smooth = _smooth_polyline_from_contour(
            main_c, POLYLINE_FALLBACK_SAMPLE_STRIDE_PX,
        )
        # Collect this outer contour's holes.
        cs_holes: list[CrossSectionHole] = []
        for j in range(len(contours)):
            if hier[j][3] != main_idx:
                continue
            hc = contours[j]
            ha = float(cv2.contourArea(hc))
            if ha < hole_min_area_abs or ha < c_area * SLICE_HOLE_MIN_AREA_VS_OUTER:
                continue
            hpoly, hbbox = _polygon_from_contour(hc)
            houtline = _segment_line_arc(hc, hpoly)
            perim = float(cv2.arcLength(hc, closed=True)) or 1.0
            circ = min(4 * np.pi * ha / (perim ** 2), 1.0)
            m = cv2.moments(hc)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            cs_holes.append(CrossSectionHole(
                polygon_px=hpoly,
                bbox_px=hbbox,
                centre_xy=(cx, cy),
                area_px=ha,
                circularity=circ,
                equivalent_diameter_px=float(2.0 * np.sqrt(ha / np.pi)),
                outline=houtline,
            ))
        cs = CrossSection(
            axis=axis,
            position_norm=float(positions[midpoint]),
            polygon_px=poly_px,
            bbox_px=bbox,
            area_px=c_area,
            outline=outline,
            smooth_polyline_px=smooth,
            holes=cs_holes,
        )
        zones.append(AxisZone(
            axis=axis,
            start_norm=float(positions[s]) if s > 0 else 0.0,
            end_norm=float(positions[e - 1]) if e <= len(positions) else 1.0,
            cross_section=cs,
        ))

    # Post-merge: dedup on OUTER contour area. Tiers with the same
    # outer diameter but different interior holes are treated as one
    # tier — the inner holes are emitted separately as cut operations.
    deduped: list[AxisZone] = []
    for z in zones:
        if deduped:
            prev = deduped[-1]
            prev_a = max(prev.cross_section.area_px, 1.0)
            if abs(z.cross_section.area_px - prev_a) / prev_a < SLICE_ZONE_DEDUP_AREA_FRAC:
                deduped[-1] = AxisZone(
                    axis=prev.axis,
                    start_norm=prev.start_norm,
                    end_norm=z.end_norm,
                    cross_section=prev.cross_section,
                )
                continue
        deduped.append(z)

    # Transition-zone filter: a THIN zone whose area sits between the
    # neighbours' is a render-gradient artifact at a sharp Z step. Merge
    # it into the area-closer neighbour by extending that neighbour's Z
    # range. We never drop the FIRST or LAST zone this way, and we only
    # touch one zone per pass so the iteration keeps the list well-formed.
    changed = True
    while changed and len(deduped) >= 3:
        changed = False
        for i in range(1, len(deduped) - 1):
            cur = deduped[i]
            h = cur.end_norm - cur.start_norm
            if h >= SLICE_TRANSITION_MAX_HEIGHT_FRAC:
                continue
            prev = deduped[i - 1]
            nxt = deduped[i + 1]
            prev_a = prev.cross_section.area_px
            cur_a = cur.cross_section.area_px
            nxt_a = nxt.cross_section.area_px
            lo, hi = min(prev_a, nxt_a), max(prev_a, nxt_a)
            if not (lo < cur_a < hi):
                continue
            # Merge into the area-closer neighbour, extending its Z range.
            if abs(cur_a - prev_a) <= abs(cur_a - nxt_a):
                deduped[i - 1] = AxisZone(
                    axis=prev.axis,
                    start_norm=prev.start_norm,
                    end_norm=cur.end_norm,
                    cross_section=prev.cross_section,
                )
            else:
                deduped[i + 1] = AxisZone(
                    axis=nxt.axis,
                    start_norm=cur.start_norm,
                    end_norm=nxt.end_norm,
                    cross_section=nxt.cross_section,
                )
            del deduped[i]
            changed = True
            break

    return AxisSlices(
        axis=axis, n_slices=n_slices,
        positions_norm=positions.tolist(),
        areas_px=areas, transitions_idx=transitions, zones=deduped,
    )


def compute_axis_slices(
    png_path: str | Path, n_slices: int = SLICE_DEFAULT_N,
) -> dict[str, AxisSlices]:
    """Public entry: load the PNG, build per-axis (low, high, valid)
    volumes, slice along each axis, and return AxisSlices per axis.
    Always returns Z, Y, X keys (axis missing -> empty AxisSlices).
    """
    png_path = Path(png_path)
    img = np.asarray(Image.open(png_path).convert("RGB"))
    cells = _split_cells(img)
    panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, depth, sil in cells:
        depth_luma = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        sil_luma = cv2.cvtColor(sil, cv2.COLOR_RGB2GRAY)
        sil_mask = sil_luma < SILHOUETTE_BODY_MAX_LUMA
        # Close pinholes in the silhouette before using it for slice masking
        sil_u8 = (sil_mask.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sil_u8 = cv2.morphologyEx(sil_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        panels[name] = (depth_luma, sil_u8 > 0)

    out: dict[str, AxisSlices] = {}
    for axis in ("Z", "Y", "X"):
        sl = _slice_axis(panels, axis, n_slices)
        if sl is not None:
            out[axis] = sl
        else:
            out[axis] = AxisSlices(
                axis=axis, n_slices=n_slices, positions_norm=[],
                areas_px=[], transitions_idx=[], zones=[],
            )
    return out


def segment_ortho_png(png_path: str | Path) -> OrthoFeatures:
    png_path = Path(png_path)
    img = np.asarray(Image.open(png_path).convert("RGB"))
    out = OrthoFeatures(source_png=str(png_path))
    for name, depth, sil in _split_cells(img):
        out.views[name] = segment_view(name, depth, sil)
    return out


# ---------------------------------------------------------------------------
# Debug overlay — draw the polygon + holes on top of each silhouette so a
# human can verify the segmentation in one glance.
# ---------------------------------------------------------------------------
_OUTLINE_COLOR = (255, 70, 70)
_HOLE_COLOR = (70, 200, 255)
_VERTEX_COLOR = (60, 220, 60)
_ARC_COLOR = (255, 200, 30)
_FULL_CIRCLE_COLOR = (200, 60, 200)
_LABEL_BG = (20, 20, 20)
_LABEL_FG = (255, 255, 255)


def _label_font(size: int = 14) -> ImageFont.ImageFont:
    for cand in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_debug_overlay(features: OrthoFeatures, out_path: str | Path) -> Path:
    """Annotate the source PNG with detected polygons + holes per view."""
    img = Image.open(features.source_png).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _label_font(14)

    for idx, name in enumerate(VIEW_NAMES):
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        cy = row * CELL_H
        cx_left = col * CELL_W            # depth panel origin
        cx_right = cx_left + HALF_W       # silhouette panel origin

        v = features.views.get(name)
        if v is None:
            continue

        # Render only the CANONICAL line/arc outline (the second-pass
        # segmentation). The raw approxPolyDP polygon, raw HoughCircles,
        # and greedy arc fits are kept in the JSON for debugging but
        # would clutter this overlay if drawn here.

        for hole in v.interior_holes:
            hx, hy, hw, hh = hole.bbox_xywh
            # Draw the hole as an ellipse inscribed in its bbox — a
            # round hole renders as a circle, an oval pocket as an oval.
            # (Was a rectangle outline; that misrepresented the shape.)
            draw.ellipse(
                [(cx_right + hx, cy + hy), (cx_right + hx + hw, cy + hy + hh)],
                outline=_HOLE_COLOR, width=2,
            )

        # Canonical outline: lines in red, arcs traced with their
        # fitted-centre marker so it's clear which arc each segment is on.
        for seg in v.outline:
            if seg.kind == "line":
                draw.line(
                    [(cx_right + seg.start_xy[0], cy + seg.start_xy[1]),
                     (cx_right + seg.end_xy[0], cy + seg.end_xy[1])],
                    fill=_OUTLINE_COLOR, width=3,
                )
            elif seg.kind == "arc" and seg.arc_centre_xy and seg.arc_radius_px:
                ccx = cx_right + seg.arc_centre_xy[0]
                ccy = cy + seg.arc_centre_xy[1]
                r = seg.arc_radius_px
                # Draw the actual ARC PORTION of the fitted circle by
                # sampling along the arc from start to end. Avoids the
                # visual clutter of drawing the whole circle outline.
                import math
                a0 = math.atan2(seg.start_xy[1] - seg.arc_centre_xy[1],
                                seg.start_xy[0] - seg.arc_centre_xy[0])
                a1 = math.atan2(seg.end_xy[1] - seg.arc_centre_xy[1],
                                seg.end_xy[0] - seg.arc_centre_xy[0])
                # Sweep the shorter way.
                diff = a1 - a0
                while diff > math.pi:
                    diff -= 2 * math.pi
                while diff < -math.pi:
                    diff += 2 * math.pi
                steps = max(8, int(abs(diff) * 24))
                pts = []
                for k in range(steps + 1):
                    t = a0 + diff * (k / steps)
                    pts.append((ccx + r * math.cos(t),
                                ccy + r * math.sin(t)))
                draw.line(pts, fill=_ARC_COLOR, width=3)
                # Centre marker as a small + so the user can see where
                # each arc's centre actually is (helpful for verifying
                # consistency across views).
                draw.line(
                    [(ccx - 4, ccy), (ccx + 4, ccy)],
                    fill=_ARC_COLOR, width=1,
                )
                draw.line(
                    [(ccx, ccy - 4), (ccx, ccy + 4)],
                    fill=_ARC_COLOR, width=1,
                )

        # Corner markers: filled green squares at every line endpoint
        # AND arc endpoint. Same green dot whether the corner is between
        # two lines, two arcs, or one of each — visually consistent.
        corner_pts: set[tuple[int, int]] = set()
        for seg in v.outline:
            for ex, ey in (seg.start_xy, seg.end_xy):
                corner_pts.add((int(round(ex)), int(round(ey))))
        for ex, ey in corner_pts:
            px, py = cx_right + ex, cy + ey
            draw.rectangle(
                [(px - 3, py - 3), (px + 3, py + 3)],
                fill=_VERTEX_COLOR,
            )

        # (Hough Circle detections intentionally NOT drawn — they fire
        # on octagons / regular polygons too and visually misrepresent
        # the geometry. Kept in JSON as ``detected_circles`` for debug.)

        # Per-cell label: counts that map to the visible overlay
        # (arcs/lines from the canonical outline, holes, depth tiers).
        n_arcs = sum(1 for s in v.outline if s.kind == "arc")
        n_lines = sum(1 for s in v.outline if s.kind == "line")
        label = (
            f"{name}: {n_arcs} arcs + {n_lines} lines, "
            f"{len(v.interior_holes)} holes, "
            f"{v.depth_tier_count} depth tiers"
        )
        label_w = int(draw.textlength(label, font=font)) + 8
        draw.rectangle(
            [(cx_right, cy), (cx_right + label_w, cy + 18)],
            fill=_LABEL_BG,
        )
        draw.text((cx_right + 4, cy + 2), label, fill=_LABEL_FG, font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path
