"""Cross-view inference: turn per-view CV features into a SketchPartDescription.

Consumes the JSON output of ``ortho_view_segmenter.segment_ortho_png`` and
produces a fully populated ``SketchPartDescription`` ready for
``sketch_builder.build_from_sketches``.

This is intentionally deterministic Python — no LLM. The 6 orthographic
views give us redundant projections of the same part, and CV gives us
exact pixel-level outlines, so the construction sequence falls out of a
small set of rules:

    base profile  ← Top view's outer outline, classified as one of
                    {circle, rectangle, D-cut, polyline}
    extrude depth ← height of the part in the side views (Front / Right)
    through-holes ← interior holes that appear in BOTH +/-axis views at
                    matching position
    side bosses   ← (TODO; MVP handles single-extrude prismatic parts)

World frame convention (matches synthesize_clean_views VIEW_DIRS):
    Top    looks down  (-Z)  → silhouette = projection on XY
    Bottom looks up    (+Z)  → silhouette = projection on XY
    Front  looks from  (-Y)  → silhouette = projection on XZ
    Back   looks from  (+Y)  → silhouette = projection on XZ
    Right  looks from  (-X)  → silhouette = projection on YZ
    Left   looks from  (+X)  → silhouette = projection on YZ

Pixel-to-mm: each panel is 256 px wide. After the aspect-fix in
synthesize_clean_views the pixel size is isotropic, so 1 px corresponds
to the same world distance in U and V. We anchor scale by forcing the
longest of (X_px, Y_px, Z_px) to equal NORMALISE_LONGEST_MM (100 mm),
matching the convention used by face_extractor and stl_renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from backend.ai_infra.ortho_view_segmenter import (
    AxisSlices,
    AxisZone,
    CrossSection,
    InteriorHole,
    OrthoFeatures,
    OutlineSegment,
    StraightEdge,
    TierRegion,
    ViewFeatures,
    compute_axis_slices,
    detect_regular_polygon,
)
from backend.ai_infra.sketch_models import (
    ArcLineSegment,
    Profile2D,
    SketchOperation,
    SketchPartDescription,
)


NORMALISE_LONGEST_MM = 100.0

# Geometric-mean target: scale the part so the geometric mean of its three
# bbox extents equals this value. Preserves aspect ratios but pulls
# extreme-aspect parts (e.g. 117514's 2:2:1 octagonal post) toward a
# "neutral" overall size instead of stretching the longest axis to 100mm
# while leaving the shortest at ~50mm. Picked 60mm so a roughly-cubic part
# still lands near 60mm on every side; high-aspect parts get their longest
# axis around 90-110mm and shortest around 30-40mm.
NORMALISE_GEOMETRIC_MEAN_MM = 60.0

# A polygon counts as a "D-cut" (circle with parallel chord cuts) when
# its strongest parallel pair has BOTH edges at least this long relative
# to the bbox diagonal. The polygon must also have more than 4 vertices
# (otherwise it is just a rectangle) and the bbox must not be near-square
# (otherwise it is a regular polygon dressed as a stadium).
D_CUT_PAIR_MIN_LEN_FRAC = 0.20      # min individual edge length / bbox diag
D_CUT_BBOX_SQUARENESS_TOL = 0.05    # if |aspect - 1| < this, treat as square
D_CUT_FLAT_AXIS_TOL_DEG = 30.0      # how close to axis-aligned the flats must be

# A through-hole appears in both ±axis views at the same relative position
# within this fraction of the panel size. Loose enough to absorb gpt-image-2
# pixel jitter; tight enough that two unrelated holes don't pair up.
HOLE_MATCH_TOL_FRAC = 0.10

# Two edges count as perpendicular within this many degrees. Used to
# decide whether a 4-vertex polygon is a rectangle (two perpendicular
# parallel pairs).
PERP_TOL_DEG = 12.0


@dataclass
class WorldFrame:
    """Pixel spans of the bounding box in each world axis, plus the
    derived mm-per-pixel scale."""
    x_px: float
    y_px: float
    z_px: float
    mm_per_px: float

    @property
    def x_mm(self) -> float: return self.x_px * self.mm_per_px
    @property
    def y_mm(self) -> float: return self.y_px * self.mm_per_px
    @property
    def z_mm(self) -> float: return self.z_px * self.mm_per_px


# ---------------------------------------------------------------------------
# Step 1 — world frame from cross-view bbox consistency
# ---------------------------------------------------------------------------
def _world_frame(views: dict[str, ViewFeatures]) -> WorldFrame:
    """X/Y/Z spans in pixels, averaged across the redundant view pairs."""
    def _bbox(name: str) -> tuple[int, int]:
        v = views.get(name)
        if v is None or v.bbox_px == (0, 0, 0, 0):
            return (0, 0)
        return (v.bbox_px[2], v.bbox_px[3])  # (w, h)

    top_w, top_h = _bbox("Top")
    bot_w, bot_h = _bbox("Bottom")
    fr_w, fr_h = _bbox("Front")
    bk_w, bk_h = _bbox("Back")
    rt_w, rt_h = _bbox("Right")
    lt_w, lt_h = _bbox("Left")

    # X span is the panel WIDTH in Top/Bottom and Front/Back.
    x_candidates = [v for v in (top_w, bot_w, fr_w, bk_w) if v > 0]
    # Y span is the panel HEIGHT in Top/Bottom and the WIDTH in Right/Left.
    y_candidates = [v for v in (top_h, bot_h, rt_w, lt_w) if v > 0]
    # Z span is the panel HEIGHT in Front/Back and Right/Left.
    z_candidates = [v for v in (fr_h, bk_h, rt_h, lt_h) if v > 0]

    x_px = sum(x_candidates) / len(x_candidates) if x_candidates else 1.0
    y_px = sum(y_candidates) / len(y_candidates) if y_candidates else 1.0
    z_px = sum(z_candidates) / len(z_candidates) if z_candidates else 1.0

    # Scale so the geometric mean of the three axes hits the target. The
    # geometric mean keeps aspect ratios untouched but is less sensitive
    # to a single dominant axis than the longest-axis rule (which made
    # tall thin parts feel disproportionately large).
    gm = (x_px * y_px * z_px) ** (1.0 / 3.0)
    if gm <= 0:
        gm = max(x_px, y_px, z_px, 1.0)
    mm_per_px = NORMALISE_GEOMETRIC_MEAN_MM / gm
    return WorldFrame(x_px=x_px, y_px=y_px, z_px=z_px, mm_per_px=mm_per_px)


# ---------------------------------------------------------------------------
# Step 2 — base profile classification (circle / rect / D-cut / polyline)
# ---------------------------------------------------------------------------
@dataclass
class BaseProfile:
    """The base sketch profile + any subtractive flat cuts that produced
    its silhouette (D-cut decomposition).

    The flat_cut_* lists are parallel: ``flat_cut_positions_mm[i]`` is
    the (X, Y) centre offset of ``flat_cuts[i]`` in the base plane, and
    each cut's profile already encodes its axis (a tall thin rectangle
    cuts a vertical strip; a wide thin rectangle cuts a horizontal strip).
    """
    profile: Profile2D
    flat_cuts: list[Profile2D]            # rect cuts to subtract from base
    flat_cut_positions_mm: list[tuple[float, float]]  # (px_x, py_y) per cut
    notes: str


def _is_perpendicular(a: StraightEdge, b: StraightEdge) -> bool:
    diff = abs(a.angle_deg - b.angle_deg)
    diff = min(diff, 180.0 - diff)
    return abs(diff - 90.0) <= PERP_TOL_DEG


def _classify_rectangle(view: ViewFeatures) -> tuple[bool, float, float]:
    """Detect a clean axis-aligned rectangle. Returns (yes, width_px, height_px)."""
    edges = view.straight_edges
    if len(edges) != 4 or len(view.parallel_pairs) < 2:
        return False, 0.0, 0.0
    # Check the two parallel pairs are perpendicular to each other.
    p0_i, p0_j = view.parallel_pairs[0]
    perp_pair_found = False
    for (q_i, q_j) in view.parallel_pairs[1:]:
        if _is_perpendicular(edges[p0_i], edges[q_i]):
            perp_pair_found = True
            break
    if not perp_pair_found:
        return False, 0.0, 0.0
    bx, by, bw, bh = view.bbox_px
    return True, float(bw), float(bh)


# NOTE: _detect_d_cut was removed in favour of the arc_line path. It used
# to emit "extrude full circle, then cut 2 rect chord-shaped strips off
# the sides", which is the wrong CAD abstraction for a D-cut silhouette.
# The arc_line path traces the silhouette boundary directly as one closed
# loop of arcs and lines — what a human CAD designer writes.


def _profile_from_polyline(view: ViewFeatures, mm_per_px: float) -> Profile2D:
    """Fallback: dump the smooth-sampled contour as a polyline profile.

    Uses ``smooth_polyline_px`` (dense subsampled contour) instead of
    ``polygon_px`` (approxPolyDP). The dense version traces the actual
    silhouette shape without the 13-vertex jagged look, so the CadQuery
    rebuild's curve sides come out smooth.
    """
    bx, by, bw, bh = view.bbox_px
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    src = view.smooth_polyline_px or view.polygon_px
    verts_mm = [
        ((p[0] - cx) * mm_per_px, -(p[1] - cy) * mm_per_px)
        for p in src
    ]
    return Profile2D(
        shape="polyline",
        width_mm=bw * mm_per_px,
        depth_mm=bh * mm_per_px,
        vertices=verts_mm,
        diameter_mm=None,
    )


def _classify_from_outline(view: ViewFeatures) -> str:
    """Look at the line/arc-segmented outline and pick a primitive label.

    Returns one of ``"circle"``, ``"rectangle"``, ``"arc_line"``,
    ``"polyline"``, or ``""`` (let the heuristics decide).

    Decision is driven by ARC PERIMETER FRACTION rather than raw segment
    counts so that an octagon (where the per-edge classifier may pick
    a few short "arcs" inside otherwise-straight cleanup noise) doesn't
    get promoted to arc_line — the four short arcs cover < 15% of the
    perimeter, so the outline is treated as a polygon.
    """
    if not view.outline:
        return ""
    import math
    arcs = [s for s in view.outline if s.kind == "arc"]
    lines = [s for s in view.outline if s.kind == "line"]
    n_seg = len(view.outline)

    def _line_len(s: OutlineSegment) -> float:
        return math.hypot(s.end_xy[0] - s.start_xy[0], s.end_xy[1] - s.start_xy[1])

    def _arc_len(s: OutlineSegment) -> float:
        if not s.arc_radius_px or s.arc_span_deg is None:
            return 0.0
        return s.arc_radius_px * math.radians(s.arc_span_deg)

    arc_perim = sum(_arc_len(s) for s in arcs)
    line_perim = sum(_line_len(s) for s in lines)
    total_perim = arc_perim + line_perim
    if total_perim <= 0:
        return ""
    arc_frac = arc_perim / total_perim

    # 1 arc spanning ~360° → pure circle.
    if n_seg == 1 and arcs and (arcs[0].arc_span_deg or 0) > 270.0:
        return "circle"

    # Mostly-arcs (≥ 80%) outline with multiple arcs → emit as arc_line
    # (handles ellipse-like shapes approximated by 4 arcs).
    if arc_frac >= 0.80 and len(arcs) >= 2:
        return "arc_line"

    # Significant arc perimeter (≥ 15%) AND at least one line → genuine
    # mixed shape (D-cut, obround, racetrack, rounded rectangle).
    if arc_frac >= 0.15 and lines:
        return "arc_line"

    # 4 lines, no arcs → rectangle.
    if len(lines) == 4 and len(arcs) == 0:
        return "rectangle"

    # Mostly-straight outline with negligible arc content → regular n-gon
    # / freeform polygon. Emit as a smooth polyline (uses the dense
    # subsampled contour, not approxPolyDP) so an octagon stays an
    # octagon instead of getting promoted to a circle.
    if arc_frac < 0.15:
        return "polyline"

    return ""


def _outline_to_arc_line_profile(
    view: ViewFeatures, mm_per_px: float,
) -> Profile2D | None:
    """Convert the segmenter's pixel-space outline into a millimetre
    arc_line profile centred on the silhouette bbox centre."""
    if not view.outline:
        return None
    bx, by, bw, bh = view.bbox_px
    cx_px = bx + bw / 2.0
    cy_px = by + bh / 2.0

    def _to_mm(p: tuple[float, float]) -> tuple[float, float]:
        # Pixel y points DOWN; CAD plane y points UP. Flip y.
        return ((p[0] - cx_px) * mm_per_px,
                -(p[1] - cy_px) * mm_per_px)

    seg_models: list[ArcLineSegment] = []
    for s in view.outline:
        start_mm = _to_mm(s.start_xy)
        end_mm = _to_mm(s.end_xy)
        if s.kind == "line":
            seg_models.append(ArcLineSegment(
                kind="line", start=start_mm, end=end_mm,
            ))
        else:
            if s.arc_centre_xy is None or not s.arc_radius_px:
                # Bad arc — fall back to a line segment.
                seg_models.append(ArcLineSegment(
                    kind="line", start=start_mm, end=end_mm,
                ))
                continue
            centre_mm = _to_mm(s.arc_centre_xy)
            r_mm = s.arc_radius_px * mm_per_px
            seg_models.append(ArcLineSegment(
                kind="arc",
                start=start_mm,
                end=end_mm,
                arc_centre=centre_mm,
                arc_radius_mm=r_mm,
                # Don't lock CCW; the builder picks the shorter arc.
                arc_ccw=None,
            ))

    # Close-the-loop guard: snap the final segment's end onto the first
    # segment's start so the closed loop is exact (within the float fmt).
    if seg_models:
        first_start = seg_models[0].start
        seg_models[-1] = ArcLineSegment(
            kind=seg_models[-1].kind,
            start=seg_models[-1].start,
            end=first_start,
            arc_centre=seg_models[-1].arc_centre,
            arc_radius_mm=seg_models[-1].arc_radius_mm,
            arc_ccw=seg_models[-1].arc_ccw,
        )

    return Profile2D(
        shape="arc_line",
        width_mm=bw * mm_per_px,
        depth_mm=bh * mm_per_px,
        arc_line_segments=seg_models,
    )


def _base_profile(view: ViewFeatures, frame: WorldFrame) -> BaseProfile:
    mm = frame.mm_per_px

    # 0. Outline-driven hint. The line/arc segmentation pass is far more
    #    reliable than counting polygon vertices because it ignores the
    #    short fragments approxPolyDP scatters along curves.
    outline_label = _classify_from_outline(view)
    if outline_label == "circle":
        # Use the fitted arc radius — it's measured directly from contour
        # points, not estimated from the bbox.
        arcs = [s for s in view.outline if s.kind == "arc"]
        r_px = arcs[0].arc_radius_px or 0.0
        if r_px > 0:
            d_mm = 2.0 * r_px * mm
            return BaseProfile(
                profile=Profile2D(
                    shape="circle",
                    width_mm=d_mm,
                    depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                flat_cuts=[],
                flat_cut_positions_mm=[],
                notes=f"circle (outline, fitted r={r_px:.1f}px)",
            )

    if outline_label == "arc_line":
        # Emit the silhouette outline as ONE closed arc+line loop. No
        # subtractive chord cuts — the loop traces the actual silhouette
        # boundary directly (arc → line → arc → line for a D-cut).
        prof = _outline_to_arc_line_profile(view, mm)
        if prof is not None:
            line_n = sum(1 for s in view.outline if s.kind == "line")
            arc_n = sum(1 for s in view.outline if s.kind == "arc")
            return BaseProfile(
                profile=prof,
                flat_cuts=[],
                flat_cut_positions_mm=[],
                notes=f"arc_line outline ({arc_n} arcs + {line_n} lines)",
            )

    if outline_label == "polyline":
        # Polyline routed via the smooth-sampled contour fallback below
        # (uses the dense contour, not jagged approxPolyDP). This is what
        # an octagon, hexagon, or arbitrary n-gon should land on.
        return BaseProfile(
            profile=_profile_from_polyline(view, mm),
            flat_cuts=[],
            flat_cut_positions_mm=[],
            notes=f"polyline (n-gon, {len(view.smooth_polyline_px)} verts)",
        )

    # Polygon-based ``is_circle`` shortcut intentionally NOT used here.
    # Hough Circle is happy to fire on a regular octagon (its area /
    # perimeter ratio is close to a circle), and the polygon-based
    # is_circle test isn't strict enough either. The outline-based path
    # above (1 arc spanning >=270°) is the only trustworthy circle signal.

    is_rect, w_px, h_px = _classify_rectangle(view)
    if is_rect:
        return BaseProfile(
            profile=Profile2D(
                shape="rectangle",
                width_mm=w_px * mm,
                depth_mm=h_px * mm,
            ),
            flat_cuts=[],
            flat_cut_positions_mm=[],
            notes="rectangle",
        )

    # Legacy D-cut branch (circle minus 2 rect cuts) intentionally REMOVED.
    # That decomposition was the wrong semantic abstraction (a CAD designer
    # would sketch the closed silhouette directly). The arc_line path above
    # handles D-cut, obround, racetrack, etc. as one closed loop. If the
    # arc_line classification didn't fire, the part is too irregular for a
    # primitive — fall through to the smooth polyline below.

    # Fallback: polyline
    return BaseProfile(
        profile=_profile_from_polyline(view, mm),
        flat_cuts=[],
        flat_cut_positions_mm=[],
        notes=f"polyline ({len(view.polygon_px)} vertices)",
    )


# ---------------------------------------------------------------------------
# Step 3 — through-hole detection (matched interior holes in opposite views)
# ---------------------------------------------------------------------------
@dataclass
class ThroughHole:
    axis: str                # "Z" | "Y" | "X"
    diameter_mm: float
    centre_mm: tuple[float, float]   # in the base view's local UV (mm)


# ---------------------------------------------------------------------------
# Reusable region → Profile2D classification.
# The ViewFeatures-specific entry point (_base_profile) below uses this
# for the OUTER outline; _build_stacked_extrudes uses it per tier region.
# ---------------------------------------------------------------------------
@dataclass
class ClassifiedProfile:
    profile: Profile2D
    label: str               # "circle" | "rectangle" | "arc_line" | "polyline"


def _collapse_collinear_segments(
    segs: list[ArcLineSegment], bbox_diag_mm: float,
    angle_tol_deg: float = 8.0,
    min_len_frac: float = 0.01,
) -> list[ArcLineSegment]:
    """Smooth a closed arc/line loop by merging tiny or near-collinear
    consecutive LINE segments. Arcs are never absorbed — they carry true
    curvature that the rebuild would lose if collapsed.

    A line is replaced when EITHER it's shorter than ``min_len_frac`` of
    the bbox diagonal OR the angle it forms with the previous line is
    within ``angle_tol_deg``. In both cases the merge extends the
    previous segment's end to this segment's end (i.e. the kink is
    erased and the geometry of the loop tightens by a fraction of a
    millimetre at most).
    """
    if not segs:
        return segs
    import math
    min_len_mm = max(bbox_diag_mm * min_len_frac, 0.0)

    def _dir(a: ArcLineSegment) -> tuple[float, float]:
        dx, dy = a.end[0] - a.start[0], a.end[1] - a.start[1]
        n = math.hypot(dx, dy) or 1.0
        return (dx / n, dy / n)

    def _line_len(a: ArcLineSegment) -> float:
        return math.hypot(a.end[0] - a.start[0], a.end[1] - a.start[1])

    out: list[ArcLineSegment] = []
    for seg in segs:
        if seg.kind != "line" or not out or out[-1].kind != "line":
            out.append(seg)
            continue
        prev = out[-1]
        merge = False
        if _line_len(seg) < min_len_mm:
            merge = True
        else:
            dx0, dy0 = _dir(prev)
            dx1, dy1 = _dir(seg)
            dot = max(-1.0, min(1.0, dx0 * dx1 + dy0 * dy1))
            angle_deg = math.degrees(math.acos(dot))
            if angle_deg < angle_tol_deg:
                merge = True
        if merge:
            out[-1] = ArcLineSegment(
                kind="line",
                start=prev.start,
                end=seg.end,
                arc_centre=None,
                arc_radius_mm=None,
                arc_ccw=None,
            )
        else:
            out.append(seg)
    # The closing wrap: the last segment's end is anchored to the first
    # segment's start, but after merges the closure may now have a near-
    # zero-length tail. Drop trailing micro-segments back-to-front.
    while len(out) > 3 and out[-1].kind == "line" and _line_len(out[-1]) < min_len_mm:
        # Extend the new last segment to close the loop instead.
        dropped = out.pop()
        out[-1] = ArcLineSegment(
            kind=out[-1].kind,
            start=out[-1].start,
            end=dropped.end,
            arc_centre=out[-1].arc_centre,
            arc_radius_mm=out[-1].arc_radius_mm,
            arc_ccw=out[-1].arc_ccw,
        )
    return out


def _classify_outline_perimeter_fraction(outline: list[OutlineSegment]) -> tuple[str, float]:
    """Return (label, arc_fraction) where label ∈ {circle, rectangle,
    arc_line, polyline, empty} based on how much of the perimeter is
    arcs vs lines."""
    import math
    if not outline:
        return "empty", 0.0
    arcs = [s for s in outline if s.kind == "arc"]
    lines = [s for s in outline if s.kind == "line"]
    arc_perim = sum(
        (s.arc_radius_px or 0.0) * math.radians(s.arc_span_deg or 0.0)
        for s in arcs
    )
    line_perim = sum(
        math.hypot(s.end_xy[0] - s.start_xy[0], s.end_xy[1] - s.start_xy[1])
        for s in lines
    )
    total = arc_perim + line_perim
    if total <= 0:
        return "empty", 0.0
    arc_frac = arc_perim / total
    if len(outline) == 1 and arcs and (arcs[0].arc_span_deg or 0) > 270.0:
        return "circle", 1.0
    if arc_frac >= 0.80 and len(arcs) >= 2:
        return "arc_line", arc_frac
    if arc_frac >= 0.15 and lines:
        return "arc_line", arc_frac
    if len(lines) == 4 and not arcs:
        return "rectangle", 0.0
    # Even with low arc_frac, if the segmenter found genuine arcs they
    # are meaningful primitives — emit as arc_line so the rebuild gets
    # clean piecewise segments instead of a noisy 100-vertex polyline.
    if len(arcs) >= 2 and lines:
        return "arc_line", arc_frac
    if arc_frac < 0.15:
        return "polyline", arc_frac
    return "polyline", arc_frac


def _profile_from_outline_polyline_bbox(
    outline: list[OutlineSegment],
    smooth_polyline_px: list[tuple[int, int]],
    bbox_px: tuple[int, int, int, int],
    mm_per_px: float,
) -> ClassifiedProfile:
    """Generic profile extractor for any closed region (outer outline OR
    a depth-tier region).

    Tries CLEAN PRIMITIVES first (in priority order):
      1. Regular polygon (octagon, hexagon, ...) → exact polyline of
         perfect vertex coords. Survives STEP export as a parametric
         constraint (vertices stay equidistant from centroid).
      2. Circle → ``shape="circle"``.
      3. Rectangle → ``shape="rectangle"``.
      4. arc_line hybrid (D-cuts, obrounds, dumbbells).
      5. polyline (smooth dense fallback).
    """
    label, _ = _classify_outline_perimeter_fraction(outline)
    bx, by, bw, bh = bbox_px
    cx_px = bx + bw / 2.0
    cy_px = by + bh / 2.0

    def _to_mm(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - cx_px) * mm_per_px, -(p[1] - cy_px) * mm_per_px)

    # 1a. EARLY EXIT for shapes with significant arc content. A regular
    #     polygon has ZERO real arcs in its outline; if the line/arc
    #     segmenter found arcs covering >= 15% of the perimeter, this
    #     is a curved or composite shape (D-cut, obround, etc.) and
    #     forcing it into a regular polygon would lose the geometric
    #     intent. Compute arc-perimeter fraction inline.
    if outline:
        import math as _m
        _arc_perim = sum(
            (s.arc_radius_px or 0.0) * _m.radians(s.arc_span_deg or 0.0)
            for s in outline if s.kind == "arc"
        )
        _line_perim = sum(
            ((s.end_xy[0] - s.start_xy[0]) ** 2 +
             (s.end_xy[1] - s.start_xy[1]) ** 2) ** 0.5
            for s in outline if s.kind == "line"
        )
        _total = _arc_perim + _line_perim
        _arc_frac = _arc_perim / _total if _total > 0 else 0.0
        _has_real_arcs = _arc_frac >= 0.15
    else:
        _has_real_arcs = False

    # 1. CIRCLE detection — runs FIRST so a near-perfectly circular
    #    smooth polyline doesn't get false-positived as an n-gon. A
    #    true circle has radial stdev/mean below ~2% (anti-aliasing
    #    floor at ~0.3%, gpt-image-2 cleanup typically lands 0.7-1.7%)
    #    AND a near-square bbox (aspect 0.95-1.05). Real regular
    #    polygons sit higher: 8-gon ~3-7%, 12-gon ~2-3%. We threshold
    #    at 2.0% which separates clean circles from polygons reliably
    #    on the cleaned-PNG corpus we've measured.
    bx, by, bw, bh = bbox_px
    aspect = bw / bh if bh else 0.0
    if (smooth_polyline_px and len(smooth_polyline_px) >= 16
            and 0.95 <= aspect <= 1.05):
        pts = np.array(smooth_polyline_px, dtype=np.float64)
        centre = pts.mean(axis=0)
        rs = np.linalg.norm(pts - centre, axis=1)
        if rs.mean() > 0 and (rs.std() / rs.mean()) < 0.020:
            d_mm = 2.0 * float(rs.mean()) * mm_per_px
            return ClassifiedProfile(
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                label=f"circle(r_stdev={rs.std()/rs.mean()*100:.1f}%)",
            )

    # 2. Regular polygon detection — try BEFORE the outline classifier.
    #    A regular octagon classifies as "polyline" in the outline pass
    #    (no arcs), but we want to emit it as a perfect n-gon. The
    #    cleaned PNG's gpt-image-2 cleanup tends to round polygon
    #    corners, so approxPolyDP at the segmenter's default epsilon
    #    leaves 11-15 verts on what should be an 8-gon. Retry at a
    #    sequence of larger epsilons until the vertex count drops into
    #    a regular-polygon range.
    #    GATE: regular polygons inscribe in roughly-SQUARE bboxes (a
    #    regular hex/oct/etc. has equal extent in X and Y). Reject if
    #    aspect is outside [0.90, 1.11] — that's an oval / elongated
    #    silhouette (e.g. a chamfered cylinder or arc_line outline) and
    #    forcing it to a regular n-gon overshoots the true bbox by
    #    20%+ when the polygon's circumscribed circle is bigger than
    #    the smaller bbox dimension.
    import cv2 as _cv2
    if (not _has_real_arcs
            and 0.90 <= aspect <= 1.11
            and smooth_polyline_px and len(smooth_polyline_px) >= 12):
        dense_np = np.array(
            smooth_polyline_px, dtype=np.int32,
        ).reshape(-1, 1, 2)
        perim = float(_cv2.arcLength(dense_np, closed=True))
        if perim > 0:
            # Cap epsilon at 0.020 (larger collapses D-cuts into false-
            # positive hexagons). Try all epsilons in range, collect
            # every valid regular polygon, then pick the one with the
            # FEWEST vertices — a 17-vert polygon and an 8-vert polygon
            # of the same shape are both "regular" but the 8-vert one
            # is the cleanest CAD representation.
            candidates: list[tuple[int, int, float, float, tuple[float, float], float]] = []
            for eps_frac in (0.005, 0.012, 0.020):
                eps = max(2.0, perim * eps_frac)
                approx = _cv2.approxPolyDP(
                    dense_np, eps, closed=True,
                ).reshape(-1, 2)
                if not (5 <= len(approx) <= 12):
                    continue
                poly_list = [(int(p[0]), int(p[1])) for p in approx]
                reg = detect_regular_polygon(poly_list)
                if reg is not None:
                    n, r_px, rot_rad, ctr = reg
                    candidates.append((n, len(approx), r_px, rot_rad, ctr, eps_frac))
            if candidates:
                # Sort: prefer fewer regular-polygon sides (cleaner),
                # then prefer the smallest epsilon at that N.
                candidates.sort(key=lambda c: (c[0], c[5]))
                n, _, r_px, rot_rad, ctr, chosen_eps = candidates[0]
                verts_mm: list[tuple[float, float]] = []
                for i in range(n):
                    ang = rot_rad + 2.0 * np.pi * i / n
                    vx_px = ctr[0] + r_px * np.cos(ang)
                    vy_px = ctr[1] + r_px * np.sin(ang)
                    verts_mm.append(_to_mm((vx_px, vy_px)))
                xs = [v[0] for v in verts_mm]
                ys = [v[1] for v in verts_mm]
                return ClassifiedProfile(
                    profile=Profile2D(
                        shape="polyline",
                        width_mm=max(xs) - min(xs),
                        depth_mm=max(ys) - min(ys),
                        vertices=verts_mm,
                    ),
                    label=f"regular-{n}gon(eps={chosen_eps:.3f})",
                )

    # 2. CIRCLE detection (post-polygon). A 12-gon has radial stdev ~1.7%,
    #    a 16-gon ~1.0%, a true circle ~0.3% (anti-aliasing only). The
    #    threshold has to sit BELOW a 12-gon's stdev so a clean polygon
    #    that escaped the regular-polygon detector above doesn't get
    #    promoted to a circle and lose its faceted geometry. Tightened
    #    0.025 → 0.010 after 117514's 12-gon top tier was being baked
    #    into a smooth cylinder instead of a 12-sided post.
    if smooth_polyline_px and len(smooth_polyline_px) >= 16:
        pts = np.array(smooth_polyline_px, dtype=np.float64)
        centre = pts.mean(axis=0)
        rs = np.linalg.norm(pts - centre, axis=1)
        if rs.mean() > 0 and (rs.std() / rs.mean()) < 0.010:
            d_mm = 2.0 * float(rs.mean()) * mm_per_px
            return ClassifiedProfile(
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                label=f"circle(r_stdev={rs.std()/rs.mean()*100:.1f}%)",
            )

    if label == "circle":
        arcs = [s for s in outline if s.kind == "arc"]
        r_px = arcs[0].arc_radius_px or max(bw, bh) / 2.0
        d_mm = 2.0 * r_px * mm_per_px
        return ClassifiedProfile(
            profile=Profile2D(
                shape="circle", width_mm=d_mm, depth_mm=d_mm, diameter_mm=d_mm,
            ),
            label="circle",
        )
    if label == "rectangle":
        return ClassifiedProfile(
            profile=Profile2D(
                shape="rectangle", width_mm=bw * mm_per_px, depth_mm=bh * mm_per_px,
            ),
            label="rectangle",
        )
    if label == "arc_line":
        seg_models: list[ArcLineSegment] = []
        for s in outline:
            start_mm = _to_mm(s.start_xy)
            end_mm = _to_mm(s.end_xy)
            if s.kind == "line":
                seg_models.append(ArcLineSegment(kind="line", start=start_mm, end=end_mm))
            else:
                if s.arc_centre_xy is None or not s.arc_radius_px:
                    seg_models.append(ArcLineSegment(kind="line", start=start_mm, end=end_mm))
                    continue
                seg_models.append(ArcLineSegment(
                    kind="arc",
                    start=start_mm,
                    end=end_mm,
                    arc_centre=_to_mm(s.arc_centre_xy),
                    arc_radius_mm=s.arc_radius_px * mm_per_px,
                    arc_ccw=None,
                ))
        if seg_models:
            seg_models[-1] = ArcLineSegment(
                kind=seg_models[-1].kind,
                start=seg_models[-1].start,
                end=seg_models[0].start,
                arc_centre=seg_models[-1].arc_centre,
                arc_radius_mm=seg_models[-1].arc_radius_mm,
                arc_ccw=seg_models[-1].arc_ccw,
            )
        # Denoise the closed loop: collapse consecutive line segments that
        # are nearly collinear and drop micro-segments. The segmenter often
        # splits a long flange edge into 3-4 tiny pieces because the cleaned
        # PNG has 1-2 px of edge jitter; the rebuild then renders that as
        # a visibly faceted side. Threshold is 8° between consecutive line
        # directions (~ a 12-sided regular polygon's interior angle), and
        # any segment shorter than 1% of the bbox diagonal is absorbed
        # into its predecessor.
        seg_models = _collapse_collinear_segments(
            seg_models, bbox_diag_mm=((bw ** 2 + bh ** 2) ** 0.5) * mm_per_px,
        )
        return ClassifiedProfile(
            profile=Profile2D(
                shape="arc_line",
                width_mm=bw * mm_per_px,
                depth_mm=bh * mm_per_px,
                arc_line_segments=seg_models,
            ),
            label="arc_line",
        )
    # polyline / empty fallback. Simplify the dense smooth contour with
    # approxPolyDP at a modest epsilon (~1.2% of the bbox diagonal) to
    # cut a 100-vertex jittery polyline down to a 8-15 vertex clean
    # outline. The visible noise on L-shapes and irregular flanges is
    # this dense sampling rendered as facets — Douglas-Peucker keeps
    # the corners and drops the 1-2 px wobbles along straight runs.
    src = smooth_polyline_px or []
    if len(src) >= 8:
        import cv2 as _cv2
        dense = np.array(src, dtype=np.int32).reshape(-1, 1, 2)
        perim = float(_cv2.arcLength(dense, closed=True))
        if perim > 0:
            approx = _cv2.approxPolyDP(
                dense, perim * 0.012, closed=True,
            ).reshape(-1, 2)
            # Don't simplify below 4 vertices; a 3-vertex polyline can't
            # carry meaningful geometry.
            if len(approx) >= 4:
                src = [(int(p[0]), int(p[1])) for p in approx]
    verts_mm = [
        ((p[0] - cx_px) * mm_per_px, -(p[1] - cy_px) * mm_per_px)
        for p in src
    ]
    return ClassifiedProfile(
        profile=Profile2D(
            shape="polyline",
            width_mm=max(bw * mm_per_px, 0.001),
            depth_mm=max(bh * mm_per_px, 0.001),
            vertices=verts_mm if len(verts_mm) >= 3 else None,
        ),
        label="polyline",
    )


# ---------------------------------------------------------------------------
# Stacked-extrude builder — one extrude per depth tier
# ---------------------------------------------------------------------------
# Tiers thinner than this collapse into the previous one. The threshold
# scales with part Z so small parts (1-2mm tall) keep all their tiers
# while huge parts (50mm+) still drop sub-millimetre histogram-jitter
# peaks. Floor at 0.05mm so a degenerate 0-thickness tier still gets
# culled even on the smallest parts.
_STACK_MIN_GAP_FRAC = 0.05
_STACK_MIN_GAP_FLOOR_MM = 0.05


def _stack_min_gap_mm(part_z_mm: float) -> float:
    return max(_STACK_MIN_GAP_FRAC * part_z_mm, _STACK_MIN_GAP_FLOOR_MM)
_STACK_POLYGON_CONTAINMENT_FRAC = 0.85   # tier[i] needs this fraction of vertices inside tier[i-1]
_STACK_BBOX_CONTAINMENT_FRAC = 0.90      # fallback when approxPolyDP makes a bad polygon


def _polygon_contains_fraction(
    outer_polygon: list[tuple[int, int]],
    inner_polygon: list[tuple[int, int]],
) -> float:
    """Fraction of inner_polygon's vertices that lie inside outer_polygon.

    1.0 = inner fully inside outer; 0.0 = inner completely outside.
    Uses cv2.pointPolygonTest for the per-vertex check.
    """
    import cv2
    import numpy as np
    if len(outer_polygon) < 3 or len(inner_polygon) < 1:
        return 0.0
    outer_np = np.array(outer_polygon, dtype=np.int32)
    inside = 0
    for v in inner_polygon:
        if cv2.pointPolygonTest(outer_np, (float(v[0]), float(v[1])), False) >= 0:
            inside += 1
    return inside / len(inner_polygon)


def _bbox_contains_fraction(
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
) -> float:
    """Fraction of inner_bbox area covered by the overlap with outer_bbox.

    This is a fallback for stacked-tier containment when the simplified
    approxPolyDP loop self-intersects and pointPolygonTest under-reports
    true containment.
    """
    ox, oy, ow, oh = outer_bbox
    ix, iy, iw, ih = inner_bbox
    if ow <= 0 or oh <= 0 or iw <= 0 or ih <= 0:
        return 0.0
    inter_x0 = max(ox, ix)
    inter_y0 = max(oy, iy)
    inter_x1 = min(ox + ow, ix + iw)
    inter_y1 = min(oy + oh, iy + ih)
    inter_w = max(inter_x1 - inter_x0, 0)
    inter_h = max(inter_y1 - inter_y0, 0)
    inter_area = float(inter_w * inter_h)
    inner_area = float(iw * ih)
    if inner_area <= 0.0:
        return 0.0
    return inter_area / inner_area


def _bottom_tier_thickness_from_side_view(
    side_views: list[ViewFeatures], frame: WorldFrame,
) -> float | None:
    """Estimate BIG-flange (bottom tier) thickness from a side-view
    silhouette by finding the Z position where the silhouette WIDTH
    drops sharply (transition from wide flange to narrow upper stack).
    Returns thickness in mm, or None if no clear step is detected.
    """
    import cv2
    import numpy as np
    for vfeat in side_views:
        if not vfeat or not vfeat.smooth_polyline_px:
            continue
        sx, sy, sw, sh = vfeat.bbox_px
        if sw < 10 or sh < 10:
            continue
        # Render the silhouette outline as a binary mask in its own bbox.
        verts_local = np.array(
            [(p[0] - sx, p[1] - sy) for p in vfeat.smooth_polyline_px],
            dtype=np.int32,
        )
        mask = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        cv2.fillPoly(mask, [verts_local], 255)
        # Width per Y row.
        widths = (mask > 0).sum(axis=1).astype(np.int32)
        if widths.max() == 0:
            continue
        max_w = int(widths.max())
        # Walk from BOTTOM (high y) UP looking for the first row where
        # width drops below 0.7 * max_w. The bottom span where width is
        # close to max_w is the wide flange.
        threshold = 0.7 * max_w
        flange_top_y = sh
        for y in range(sh - 1, -1, -1):
            if widths[y] >= threshold:
                flange_top_y = y
            else:
                break
        # If the flange spans the entire side view, no step → no clear
        # bottom tier; bail out.
        if flange_top_y <= 1 or flange_top_y >= sh - 1:
            continue
        flange_height_px = sh - flange_top_y
        if flange_height_px <= 0:
            continue
        # Side view's vertical axis IS the Z axis; convert px → mm.
        return float(flange_height_px) * frame.mm_per_px
    return None


def _build_stacked_extrudes(
    view: ViewFeatures, frame: WorldFrame,
    side_views: list[ViewFeatures] | None = None,
    source_png: str | Path | None = None,
) -> tuple[list[SketchOperation], str]:
    """Convert per-tier regions into an ordered list of extrude operations.

    Tier ordering: the segmenter returns regions FAR→NEAR (deepest first).
    For Top view (base on XY, looking down -Z), "far" = lowest Z = bottom
    of part. We build bottom up: deepest tier = XY base extrude; each
    subsequent tier extrudes on the >Z face. Distances come from the
    relative-depth gaps scaled by the part's Z extent.

    Two pre-flight gates protect against bogus stacks:
      1. Drop tiers whose extrude distance < ``_stack_min_gap_mm(frame.z_mm)`` (those
         are histogram-jitter peaks on a near-flat region).
      2. Each tier (after the first) must be contained within the previous
         tier's bbox (allowing a small slack). If not, the part isn't a
         clean stacked extrude — return empty so the caller falls back to
         a single-extrude path that won't crash CadQuery.
    """
    if not view.tier_regions:
        return [], "no tier regions"

    # PRE-FILTER 1 — same-depth dedup. When the gpt-image-2 cleanup produces
    # multiple disconnected regions at the same height (e.g. ring fragments
    # split by a hole), `_extract_tier_regions` returns each as a separate
    # tier with identical relative_depth. Keep only the LARGEST contour at
    # each unique depth so a single tier per height feeds the stack.
    by_depth: dict[float, TierRegion] = {}
    for tier in view.tier_regions:
        # Round to 3 decimals so jitter at the 4th-place doesn't keep duplicates.
        key = round(tier.relative_depth, 3)
        prev = by_depth.get(key)
        if prev is None or tier.area_px > prev.area_px:
            by_depth[key] = tier
    deduped = sorted(by_depth.values(), key=lambda t: -t.relative_depth)

    # PRE-FILTER 2 — monotonicity gate. For a stacked-extrude topology each
    # tier (going FAR→NEAR / bottom→top) must be SMALLER than the one
    # below it. A tier whose area is bigger than its predecessor is one of:
    #   (a) the real next tier and the predecessor was a sliver of noise →
    #       drop the predecessor and keep this one as the new floor;
    #   (b) cleanup noise itself, sandwiched between two real tiers, that
    #       happens to be smaller than the one BELOW but bigger than the
    #       one we'd otherwise advance to;
    #   (c) a counterbore step VISIBLE THROUGH a through-hole — predecessor
    #       is the counterbore floor (deeper Z, smaller area), this tier
    #       is the BOSS TOP annulus (higher Z, bigger area). The outside
    #       only has two step-offs (flange + boss); the apparent third
    #       "tier" is the counterbore floor seen through the hole. We
    #       handle this by dropping the predecessor (the counterbore is
    #       emitted as a CUT later by _detect_internal_counterbore) and
    #       keeping THIS tier as the legitimate boss top.
    # Heuristic: ≥ 2× the predecessor → (a); bbox of predecessor lies
    # INSIDE this tier's bbox (concentric annulus pattern) → (c) drop the
    # predecessor and keep this as the real boss top; else (b) drop.
    monotonic: list[TierRegion] = []
    # Tiers that were demoted from the stack because they were really a
    # counterbore floor seen through the hole. Each entry is the dropped
    # TierRegion paired with the BOSS tier it sits inside, so we can emit
    # it as a recess cut on the boss top after the main stack is built.
    counterbore_cuts: list[tuple[TierRegion, TierRegion]] = []
    for tier in deduped:
        if not monotonic:
            monotonic.append(tier)
            continue
        if tier.area_px <= monotonic[-1].area_px:
            monotonic.append(tier)
        elif tier.area_px >= 2.0 * monotonic[-1].area_px:
            # Predecessor was a noise sliver; replace it.
            monotonic[-1] = tier
        else:
            prev = monotonic[-1]
            prev_inside_tier = _bbox_contains_fraction(
                tier.bbox_px, prev.bbox_px,
            ) >= _STACK_BBOX_CONTAINMENT_FRAC
            if prev_inside_tier:
                # Counterbore-through-hole illusion: replace predecessor
                # with this tier (the real boss top). The discarded one
                # is re-emitted as a CUT later.
                monotonic[-1] = tier
                counterbore_cuts.append((prev, tier))
    deduped = monotonic

    # Filter tiers by minimum extrude distance.
    kept: list[TierRegion] = []
    prev_rel = 1.0
    for tier in deduped:
        gap_mm = (prev_rel - tier.relative_depth) * frame.z_mm
        if kept and gap_mm < _stack_min_gap_mm(frame.z_mm):
            continue  # drop this one as a duplicate / near-zero extrude
        kept.append(tier)
        prev_rel = tier.relative_depth
    if not kept:
        return [], "no tiers after gap filter"

    # Polygon-based containment check: each tier i must have most of its
    # vertices INSIDE the previous tier's polygon. If approxPolyDP made a
    # self-intersecting loop, fall back to strict bbox containment. Pure
    # bbox checks are too weak as the primary gate because they can pass
    # laterally offset tiers (e.g. side-by-side bosses on a bar).
    for i in range(1, len(kept)):
        poly_frac = _polygon_contains_fraction(
            kept[i - 1].polygon_px, kept[i].polygon_px,
        )
        bbox_frac = _bbox_contains_fraction(
            kept[i - 1].bbox_px, kept[i].bbox_px,
        )
        if (
            poly_frac < _STACK_POLYGON_CONTAINMENT_FRAC
            and bbox_frac < _STACK_BBOX_CONTAINMENT_FRAC
        ):
            return [], (
                f"tier {i} only {poly_frac*100:.0f}% inside tier {i-1} polygon "
                f"(bbox overlap {bbox_frac*100:.0f}%) — "
                "stacked path doesn't apply, falling back to single extrude"
            )

    base_bx, base_by, base_bw, base_bh = kept[0].bbox_px
    base_cx_px = base_bx + base_bw / 2.0
    base_cy_px = base_by + base_bh / 2.0

    # Pre-compute raw extrude heights from the luma-derived rel_d gaps,
    # then NORMALIZE so the stack sums exactly to frame.z_mm. The luma
    # sequence is non-monotonic in true depth (the colormap is
    # non-monotonic in luma), so the absolute proportions are noisy —
    # but their sum should still match the part's actual Z extent.
    raw_heights: list[float] = []
    prev_rel_h = 1.0
    for tier in kept:
        gap = max(prev_rel_h - tier.relative_depth, 0.0)
        raw_heights.append(gap * frame.z_mm)
        prev_rel_h = tier.relative_depth
    raw_total = sum(raw_heights)
    if raw_total > 0:
        scale = frame.z_mm / raw_total
        heights = [max(h * scale, _stack_min_gap_mm(frame.z_mm)) for h in raw_heights]
    else:
        heights = [_stack_min_gap_mm(frame.z_mm)] * len(kept)

    # STL-ANCHORED HEIGHTS + XY BBOXES — sample the recon STL directly:
    #   - tier-face peaks in the Z vertex histogram give tier transition
    #     Z values (true depth, not the non-monotonic luma proxy);
    #   - vertex X/Y bboxes within each Z band give true tier diameter
    #     and centre offset (no cleanup-time distortion).
    # Both are stored on `kept` for the per-tier loop below to use.
    stl_xy_bboxes_mm: list[tuple[float, float, float, float, float, float]] | None = None
    if source_png and len(kept) >= 2:
        z_anchors = _tier_z_anchors_from_stl(source_png, len(kept))
        if z_anchors and len(z_anchors) == len(kept) + 1:
            # z_anchors are in STL world coords (DeepCAD's unit cube,
            # roughly -0.5..0.5 — NOT mm). Compute raw gaps in STL units,
            # rescale ALL of them to mm by frame.z_mm / total_span, THEN
            # apply the per-tier min_gap floor (in mm). Doing the floor
            # on STL-unit values before scaling makes every tier saturate
            # to the floor (e.g. 0.07 < min_gap_mm), losing all relative
            # proportions and dividing the part Z evenly across tiers.
            raw_stl_heights = [
                z_anchors[i + 1] - z_anchors[i] for i in range(len(kept))
            ]
            stl_total = sum(raw_stl_heights)
            if stl_total > 0:
                cal_scale = frame.z_mm / stl_total
                heights = [
                    max(h * cal_scale, _stack_min_gap_mm(frame.z_mm))
                    for h in raw_stl_heights
                ]
                # Pull X/Y bboxes for each Z band, then convert STL→mm
                # using the SAME cal_scale (the bbox calibration already
                # forced the ratio between mm_per_unit on every axis).
                stl_xy_raw = _stl_xy_bboxes_per_band(source_png, z_anchors)
                if stl_xy_raw and len(stl_xy_raw) == len(kept):
                    stl_xy_bboxes_mm = [
                        (
                            x_lo * cal_scale, x_hi * cal_scale,
                            y_lo * cal_scale, y_hi * cal_scale,
                            cx * cal_scale,   cy * cal_scale,
                        )
                        for x_lo, x_hi, y_lo, y_hi, cx, cy in stl_xy_raw
                    ]
    # SIDE-VIEW FALLBACK — when the STL slicer can't find clean
    # transitions, fall back to the side-view silhouette anchor
    # (which detects the BIG flange's wide-bottom span).
    elif side_views and len(kept) >= 2:
        bottom_h_mm = _bottom_tier_thickness_from_side_view(side_views, frame)
        if bottom_h_mm and bottom_h_mm > _stack_min_gap_mm(frame.z_mm):
            remaining = max(frame.z_mm - bottom_h_mm, _stack_min_gap_mm(frame.z_mm))
            upper_raw_total = sum(heights[1:]) or 1.0
            upper_scale = remaining / upper_raw_total
            heights = (
                [bottom_h_mm]
                + [max(h * upper_scale, _stack_min_gap_mm(frame.z_mm)) for h in heights[1:]]
            )

    # Base centre from the STL XY bbox (tier 0) — this is the part's
    # true centre in the same mm units as our tier dimensions. We use
    # this to compute concentric position offsets for upper tiers so
    # they sit centred over the BIG flange the way the original part
    # does, ignoring any cleanup-time silhouette drift.
    if stl_xy_bboxes_mm:
        base_cx_mm = stl_xy_bboxes_mm[0][4]
        base_cy_mm = stl_xy_bboxes_mm[0][5]
    else:
        base_cx_mm = base_cx_px * frame.mm_per_px
        base_cy_mm = -base_cy_px * frame.mm_per_px

    operations: list[SketchOperation] = []
    notes_parts: list[str] = []
    upgrades = _harmonize_stacked_tier_classifications(kept, frame)
    for i, (tier, extrude_mm) in enumerate(zip(kept, heights)):
        if i in upgrades:
            cls = upgrades[i]
        else:
            cls = _profile_from_outline_polyline_bbox(
                tier.outline, tier.smooth_polyline_px, tier.bbox_px, frame.mm_per_px,
            )
        bx, by, bw, bh = tier.bbox_px
        cx_px = bx + bw / 2.0
        cy_px = by + bh / 2.0
        tier_bbox_area = float(bw * bh)
        tier_fill = (tier.area_px / tier_bbox_area) if tier_bbox_area > 0 else 1.0
        tier_aspect = bw / bh if bh else 0.0
        is_thin_ring = (
            tier_fill < 0.35
            and 0.90 <= tier_aspect <= 1.11
            and tier.area_px > 0
        )

        # STL-anchored XY for THIS tier (if available). Width/depth come
        # straight from the STL vertex bbox in this Z band; centre is
        # the bbox midpoint. These trump the cleaned-PNG silhouette
        # values whenever the STL slice is reliable.
        if stl_xy_bboxes_mm:
            sx_lo, sx_hi, sy_lo, sy_hi, scx, scy = stl_xy_bboxes_mm[i]
            stl_w_mm = max(sx_hi - sx_lo, 0.0)
            stl_d_mm = max(sy_hi - sy_lo, 0.0)
            stl_cx_mm = scx
            stl_cy_mm = scy
        else:
            stl_w_mm = bw * frame.mm_per_px
            stl_d_mm = bh * frame.mm_per_px
            # Absolute centroid in the same PNG-derived mm frame as
            # base_cx_mm / base_cy_mm. The position calc below subtracts
            # base from stl_cx_mm to get the on-workplane offset; storing
            # the absolute here keeps both branches consistent. Storing
            # (cx_px - base_cx_px) here — an offset — combined with the
            # downstream `stl_cx_mm - base_cx_mm` subtraction produces
            # `offset - absolute`, which is what put tier 2 of 117514 /
            # 000035 ~100mm off the part.
            stl_cx_mm = cx_px * frame.mm_per_px
            stl_cy_mm = -cy_px * frame.mm_per_px

        # TOPMOST THIN-RING → CUT (recess), not extrude.
        # When the LAST tier in the stack is a thin annular shape, it
        # represents a CIRCULAR RECESS cut down from the previous tier's
        # top face — not a raised disc.
        # The CUT'S DIAMETER comes from the cleaned PNG bbox (the ring
        # outline) — NOT the STL slab. The STL slab at the topmost Z
        # samples the part's TOP face (including the rim around the
        # ring), which would overestimate the recess opening.
        #
        # RIM HEIGHT FIX: the previous tier (the boss) was sized to extrude
        # only up to the RECESS FLOOR (z_anchors[i] from STL). The actual
        # part extends up to z_anchors[i+1] (top of the rim around the
        # recess). To preserve true Z extent we extend the previous extrude
        # by the cut depth — the solid boss reaches the rim, then we cut
        # the recess down from that rim. Without this the rebuild is short
        # by exactly the rim wall thickness.
        if i > 0 and i == len(kept) - 1 and is_thin_ring:
            d_mm = ((bw + bh) / 2.0) * frame.mm_per_px
            pos_x_mm = (cx_px - base_cx_px) * frame.mm_per_px
            pos_y_mm = -(cy_px - base_cy_px) * frame.mm_per_px
            cut_depth = max(extrude_mm, _stack_min_gap_mm(frame.z_mm))
            if operations and operations[-1].operation == "extrude":
                prev_op = operations[-1]
                operations[-1] = prev_op.model_copy(update={
                    "distance_mm": prev_op.distance_mm + cut_depth,
                })
                # Reflect the height bump in the notes for the previous tier.
                if notes_parts:
                    notes_parts[-1] = notes_parts[-1].replace(
                        f"h={prev_op.distance_mm:.1f}mm",
                        f"h={prev_op.distance_mm + cut_depth:.1f}mm (rim-extended)",
                    )
            operations.append(SketchOperation(
                order=i + 1,
                plane=">Z",
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                operation="cut",
                distance_mm=cut_depth,
                direction="negative",
                position_x=pos_x_mm,
                position_y=pos_y_mm,
            ))
            notes_parts.append(f"tier{i}(circle CUT d={d_mm:.1f}mm, depth={cut_depth:.1f}mm)")
            continue

        # Non-topmost thin ring → snap to solid circle so the rebuild
        # doesn't get a 30-vertex walled tube shape.
        if cls.profile.shape in ("arc_line", "polyline") and is_thin_ring:
            d_mm = (stl_w_mm + stl_d_mm) / 2.0
            cls = ClassifiedProfile(
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                label=f"circle(thin-ring snap, fill={tier_fill:.2f})",
            )

        # OVERRIDE PRIMITIVE DIMENSIONS with STL bbox — circles and
        # rectangles get their dimensions straight from the STL vertex
        # extents in this tier's Z band. arc_line / polyline keep their
        # cleaned-PNG vertex coords (those carry the actual outline
        # SHAPE that we don't want to lose) but we'll center them via
        # the STL centroid below.
        if cls.profile.shape == "circle":
            d_mm = (stl_w_mm + stl_d_mm) / 2.0
            cls = ClassifiedProfile(
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                label=f"{cls.label} [STL d={d_mm:.1f}mm]",
            )
        elif cls.profile.shape == "rectangle":
            cls = ClassifiedProfile(
                profile=Profile2D(
                    shape="rectangle",
                    width_mm=stl_w_mm,
                    depth_mm=stl_d_mm,
                ),
                label=f"{cls.label} [STL {stl_w_mm:.1f}x{stl_d_mm:.1f}mm]",
            )

        if i == 0:
            pos_x_mm = 0.0
            pos_y_mm = 0.0
            plane = "XY"
        else:
            pos_x_mm = stl_cx_mm - base_cx_mm
            pos_y_mm = stl_cy_mm - base_cy_mm
            plane = ">Z"
        operations.append(SketchOperation(
            order=i + 1,
            plane=plane,
            profile=cls.profile,
            operation="extrude",
            distance_mm=extrude_mm,
            direction="positive",
            position_x=pos_x_mm,
            position_y=pos_y_mm,
        ))
        notes_parts.append(f"tier{i}({cls.label}, h={extrude_mm:.1f}mm)")

    # COUNTERBORE CUTS — for each (counterbore_floor, boss_tier) pair
    # captured during the monotonicity gate, emit a partial-depth cut from
    # the boss top down to the counterbore floor. The diameter comes from
    # the counterbore region's CV bbox; the depth comes from the rel_depth
    # gap between the boss top and the counterbore floor, scaled by the
    # part's Z extent.
    if counterbore_cuts:
        bos_extrude = next(
            (op for op in reversed(operations) if op.operation == "extrude"),
            None,
        )
        boss_face = bos_extrude.plane if bos_extrude else ">Z"
        for cb_floor, boss_tier in counterbore_cuts:
            bx, by, bw, bh = cb_floor.bbox_px
            d_mm = ((bw + bh) / 2.0) * frame.mm_per_px
            depth_norm = max(boss_tier.relative_depth - cb_floor.relative_depth, 0.0)
            # relative_depth runs FAR (=1) → NEAR (=0); a counterbore
            # floor is FARTHER (deeper) than the boss top, so its
            # relative_depth is GREATER. depth_mm uses the absolute gap.
            cb_depth_mm = max(
                abs(cb_floor.relative_depth - boss_tier.relative_depth) * frame.z_mm,
                _stack_min_gap_mm(frame.z_mm),
            )
            cb_cx_px = bx + bw / 2.0
            cb_cy_px = by + bh / 2.0
            cb_pos_x = (cb_cx_px - base_cx_px) * frame.mm_per_px
            cb_pos_y = -(cb_cy_px - base_cy_px) * frame.mm_per_px
            operations.append(SketchOperation(
                order=len(operations) + 1,
                plane=boss_face,
                profile=Profile2D(
                    shape="circle", width_mm=d_mm, depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                operation="cut",
                distance_mm=cb_depth_mm,
                direction="negative",
                position_x=cb_pos_x,
                position_y=cb_pos_y,
            ))
            notes_parts.append(
                f"counterbore(circle d={d_mm:.1f}mm, depth={cb_depth_mm:.1f}mm)"
            )

    return operations, "stacked: " + " + ".join(notes_parts)


# ---------------------------------------------------------------------------
# Side-face features — bosses on the ±X / ±Y faces
# ---------------------------------------------------------------------------
# A side-view depth panel shows the side face as the "background" tier
# (largest area, deepest = farthest from this side's camera). Anything
# that's NEARER than the background AND contained inside it is a feature
# that protrudes from that face. For an MVP we emit those as additive
# bosses; cuts (counterbores, blind holes) come later.
_SIDE_BOSS_CONTAINMENT_FRAC = 0.7    # boss must lie this fraction inside background
_SIDE_BOSS_MIN_HEIGHT_MM = 1.0


# (view_name, face_selector, axis_extent_attr_name)
_SIDE_VIEW_SPECS = (
    ("Front", "<Y", "y_mm"),
    ("Back",  ">Y", "y_mm"),
    ("Right", "<X", "x_mm"),
    ("Left",  ">X", "x_mm"),
)


def _detect_side_face_bosses(
    views: dict[str, ViewFeatures], frame: WorldFrame, start_order: int,
) -> tuple[list[SketchOperation], list[str]]:
    """Walk the four side views and emit bosses (often cylindrical) on the
    matching face.

    The earlier "largest tier == background" heuristic broke when a
    cylinder boss dominates the silhouette (e.g. 128105 Right view: the
    cylinder end-face is BOTH the nearest AND the largest tier — the bar
    behind it only shows as a thin rim). The new approach is shape-first:
    identify tier regions whose outline is circular, treat them as
    cylinder bosses, and figure out the extrude length from the depth gap
    between this tier and the next-deepest tier in the same view.

    Tier ordering inside ``view.tier_regions``: FAR→NEAR (largest luma
    first; smaller luma = closer to camera). The "boss" tier is one of
    the NEAR tiers; the "face plane" it attaches to is one of the FAR
    tiers.
    """
    operations: list[SketchOperation] = []
    notes: list[str] = []
    order = start_order

    for view_name, face_selector, axis_attr in _SIDE_VIEW_SPECS:
        view = views.get(view_name)
        if view is None or len(view.tier_regions) < 2:
            continue
        axis_mm = getattr(frame, axis_attr)
        if axis_mm <= 0:
            continue

        # Reference depth = the deepest tier's relative_depth. Anything
        # nearer than this is a candidate boss. We don't pick a single
        # "face plane" because a cylinder boss can dominate the
        # silhouette and the bar behind it shows only as edge artifacts.
        deepest_rel = max(t.relative_depth for t in view.tier_regions)

        # Use the OVERALL silhouette centre (not a specific tier's centre)
        # as the reference for boss positions, because the workplane
        # `.faces(face_selector).workplane(centerOption="CenterOfBoundBox")`
        # places the workplane origin at the face's bbox centre — which
        # for a centred bar IS the silhouette centre.
        sil_bx, sil_by, sil_bw, sil_bh = view.bbox_px
        sil_cx_px = sil_bx + sil_bw / 2.0
        sil_cy_px = sil_by + sil_bh / 2.0

        boss_count = 0
        for tier in view.tier_regions:
            if tier.relative_depth >= deepest_rel:
                continue  # not nearer than the back face

            extrude_mm = (deepest_rel - tier.relative_depth) * axis_mm
            if extrude_mm < _SIDE_BOSS_MIN_HEIGHT_MM:
                continue

            cls = _profile_from_outline_polyline_bbox(
                tier.outline, tier.smooth_polyline_px, tier.bbox_px, frame.mm_per_px,
            )

            # Shape-first gate: only emit CIRCULAR tier regions as side
            # bosses (cylinder bosses). Non-circular tier regions on side
            # views are usually noise / face-plane fragments and emitting
            # them as polylines produces garbage.
            if cls.label != "circle":
                continue

            bx, by, bw, bh = tier.bbox_px
            cx_px = bx + bw / 2.0
            cy_px = by + bh / 2.0
            pos_x_mm = (cx_px - sil_cx_px) * frame.mm_per_px
            pos_y_mm = -(cy_px - sil_cy_px) * frame.mm_per_px

            operations.append(SketchOperation(
                order=order,
                plane=face_selector,
                profile=cls.profile,
                operation="extrude",
                distance_mm=extrude_mm,
                direction="positive",
                position_x=pos_x_mm,
                position_y=pos_y_mm,
            ))
            order += 1
            boss_count += 1
            notes.append(
                f"{view_name}→{face_selector}:cylinder(d={cls.profile.diameter_mm:.1f}mm,h={extrude_mm:.1f}mm)"
            )

    return operations, notes


def _match_holes(
    view_a: ViewFeatures, view_b: ViewFeatures, axis: str, frame: WorldFrame,
) -> list[ThroughHole]:
    """For every hole in view_a, look for a matching hole in view_b at the
    same relative position. Both views are assumed to be the +/-axis pair
    along the same axis."""
    if not view_a.interior_holes or not view_b.interior_holes:
        return []
    matches: list[ThroughHole] = []
    panel_w = max(view_a.bbox_px[2], view_b.bbox_px[2], 1)
    panel_h = max(view_a.bbox_px[3], view_b.bbox_px[3], 1)
    tol_x = panel_w * HOLE_MATCH_TOL_FRAC
    tol_y = panel_h * HOLE_MATCH_TOL_FRAC

    bx_a, by_a, bw_a, bh_a = view_a.bbox_px
    cx_a, cy_a = bx_a + bw_a / 2.0, by_a + bh_a / 2.0

    for ha in view_a.interior_holes:
        rel_a = (ha.centre_xy[0] - cx_a, ha.centre_xy[1] - cy_a)
        # Look for the closest hole in view_b at the mirrored relative
        # position. For the +/-Z axis (Top/Bottom) the U axis flips
        # because Bottom is looking the other way; same for Right/Left
        # (X axis flip) and Front/Back (X axis flip in the typical view
        # convention used here). To stay robust we just search both
        # mirror-flipped and identity positions.
        bx_b, by_b, bw_b, bh_b = view_b.bbox_px
        cx_b, cy_b = bx_b + bw_b / 2.0, by_b + bh_b / 2.0
        best: InteriorHole | None = None
        best_d = float("inf")
        for hb in view_b.interior_holes:
            rel_b = (hb.centre_xy[0] - cx_b, hb.centre_xy[1] - cy_b)
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    d = ((rel_a[0] - sx * rel_b[0]) ** 2
                         + (rel_a[1] - sy * rel_b[1]) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best = hb
        if best is None or best_d > (tol_x + tol_y):
            continue
        # Use the mean equivalent diameter for stability.
        d_px = (ha.equivalent_diameter_px + best.equivalent_diameter_px) / 2.0
        d_mm = d_px * frame.mm_per_px
        # Centre in mm relative to the silhouette centre of view_a.
        centre_mm = (rel_a[0] * frame.mm_per_px,
                     -rel_a[1] * frame.mm_per_px)
        matches.append(ThroughHole(axis=axis, diameter_mm=d_mm, centre_mm=centre_mm))
    return matches


# ---------------------------------------------------------------------------
# Step 4 — assemble the SketchPartDescription
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Slice-based stacked extrudes
# ---------------------------------------------------------------------------
# Map slicing axis -> base sketch plane + extrude axis attribute on
# WorldFrame + face selector for the extruded direction's "top".
_SLICE_AXIS_MAP = {
    "Z": {"plane": "XY", "axis_attr": "z_mm", "face_top": ">Z"},
    "Y": {"plane": "XZ", "axis_attr": "y_mm", "face_top": ">Y"},
    "X": {"plane": "YZ", "axis_attr": "x_mm", "face_top": ">X"},
}

_SLICE_MIN_AREA_DIFF_FRAC = 0.15  # adjacent zone area diff to count as a real step
_SLICE_MIN_RANK_SCORE = 0.30      # axis must have at least this score to be picked
# When consecutive tiers' bbox centres are within this fraction of the
# previous tier's bbox MIN dimension, treat them as exactly concentric.
# Cross-section bbox centres jitter by a few pixels between slices due
# to anti-aliasing of depth panel edges; without snapping, those tiny
# offsets accumulate down the workplane chain and the stack visibly
# drifts off-axis.
_SLICE_CONCENTRIC_TOL_FRAC = 0.20
# Position snap tolerances for the post-processing constraint pass.
_POS_SNAP_AXIS_FRAC = 0.05      # |x| or |y| < 5% of part_max_mm → snap to 0
_POS_SNAP_PAIR_TOL_FRAC = 0.05  # mirror pair tolerance (each component)
# Equal-dimension snap for circles on the same plane: if two circles'
# diameters agree within this fraction, snap both to their average.
# Helps express "two equal-diameter holes" as a parametric equality.
_DIAM_EQUAL_TOL_FRAC = 0.05


def _rank_axis(slices: AxisSlices) -> float:
    """Higher score = better candidate for the stepped-extrude axis.

    A real stepped extrude has area progressing MONOTONICALLY along the
    extrusion axis (each tier is smaller than the previous, or all the
    same — never up then down). When you slice a Z-extruded post along
    Y or X, you get an "up-then-down" pattern as you walk through the
    cylindrical side surface — that's NOT the extrusion axis even
    though it produces multiple zones. We disqualify any axis with a
    sign change in the area-vs-position progression.
    """
    if not slices.zones or len(slices.zones) < 2:
        return 0.0
    areas = [z.cross_section.area_px for z in slices.zones]
    max_area = max(areas)
    if max_area <= 0:
        return 0.0
    diffs = [areas[i] - areas[i - 1] for i in range(1, len(areas))]
    # Count sign changes treating very small diffs as 0 (within plateau noise).
    sig_threshold = 0.05 * max_area
    sign_changes = 0
    prev_sign = 0
    for d in diffs:
        if d > sig_threshold:
            sign = 1
        elif d < -sig_threshold:
            sign = -1
        else:
            sign = 0
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            sign_changes += 1
        if sign != 0:
            prev_sign = sign
    if sign_changes > 0:
        # Non-monotonic = WRONG axis (e.g. side view of an extrude).
        return 0.0
    significant_count = sum(
        1 for d in diffs if abs(d) / max_area > _SLICE_MIN_AREA_DIFF_FRAC
    )
    return float(significant_count)


_PERP_VIEW_FOR_AXIS = {"Z": "Top", "Y": "Front", "X": "Right"}


def _pick_slicing_axis(
    slices_dict: dict[str, AxisSlices],
    views: dict[str, ViewFeatures] | None = None,
) -> AxisSlices | None:
    """Pick the axis with the strongest stacked-cross-section signal.

    When multiple axes pass _rank_axis the raw score alone isn't enough:
    a stepped staircase like 002354 has monotonic area progressions BOTH
    along its true stack axis (Z) AND along the side-projection axis (X),
    and the side projection coincidentally scores higher because more of
    its transitions cross the 15%-area threshold. To break ties we
    consult the perpendicular base view's depth-tier count: Top reveals
    Z-tiers, Front reveals Y-tiers, Right reveals X-tiers. CAD parts are
    overwhelmingly Z-stacked, so when Z has perpendicular evidence
    (Top.depth_tier_count >= 2) we prefer Z even if X scores higher in
    raw transition count.

    Returns None when no axis has a meaningful stacked structure."""
    qualified: list[tuple[str, AxisSlices, float]] = []
    for axis_name in ("Z", "Y", "X"):
        s = slices_dict.get(axis_name)
        if s is None:
            continue
        score = _rank_axis(s)
        if score < _SLICE_MIN_RANK_SCORE:
            continue
        qualified.append((axis_name, s, score))

    if not qualified:
        return None
    if len(qualified) == 1:
        return qualified[0][1]

    # Multi-axis case: use perpendicular-view evidence + CAD-axis priority.
    def has_perp_evidence(axis_name: str) -> bool:
        if views is None:
            return False
        perp_name = _PERP_VIEW_FOR_AXIS.get(axis_name)
        perp = views.get(perp_name) if perp_name else None
        return bool(perp) and perp.depth_tier_count >= 2

    evidenced = [t for t in qualified if has_perp_evidence(t[0])]
    if len(evidenced) == 1:
        return evidenced[0][1]
    if evidenced:
        # Multiple axes have evidence: prefer Z > Y > X (CAD convention).
        priority = {"Z": 0, "Y": 1, "X": 2}
        evidenced.sort(key=lambda t: priority[t[0]])
        return evidenced[0][1]
    # No perpendicular evidence at all — fall back to raw score.
    return max(qualified, key=lambda t: t[2])[1]


def _polyline_to_regular_polygon(
    vertices_mm: list[tuple[float, float]],
) -> tuple[int, float, float, tuple[float, float]] | None:
    """If a polyline (in mm) IS a regular n-gon, return (N, radius_mm,
    rotation_rad, centre_mm). Used by the rotation-harmonisation pass
    to read back the polygon parameters from a Profile2D's vertex list.
    """
    n = len(vertices_mm)
    if n < 5 or n > 12:
        return None
    pts = np.array(vertices_mm, dtype=np.float64)
    centre = pts.mean(axis=0)
    rs = np.linalg.norm(pts - centre, axis=1)
    if rs.mean() <= 0:
        return None
    # Loose check — we already know it's a regular polygon at this point.
    if rs.std() / rs.mean() > 0.05:
        return None
    v0 = pts[0] - centre
    rotation = float(np.arctan2(v0[1], v0[0]))
    return n, float(rs.mean()), rotation, (float(centre[0]), float(centre[1]))


def _harmonise_polygon_rotations(
    operations: list[SketchOperation],
) -> list[SketchOperation]:
    """Walk extrude operations sharing a sketch plane; if multiple are
    regular n-gons (same N), force them all to share the same rotation
    (the parametric equivalent of a "parallel" constraint). Without
    this, each tier's regular polygon picks its own arbitrary rotation
    based on which contour vertex happened to be 'first' — producing
    visually-misaligned stacked tiers like 117514's pre-harmonisation
    output where every octagon was rotated a few degrees differently.
    """
    # Group by N only — NOT by (plane, N). A stacked extrude has its
    # base on an absolute plane (XY/XZ/YZ) but every subsequent tier on
    # a face selector (>Z, >Y, >X). Those still share the same axis of
    # symmetry, so they should share rotation. Grouping by plane alone
    # would keep tier 0 (XY) in a separate bucket from tiers 1+ (>Z),
    # leaving tier 0 with its own arbitrary rotation while the others
    # align to each other (visible: 117514's outer octagon rotated
    # relative to the inner two).
    by_n: dict[int, list[int]] = {}
    polygon_info: dict[int, tuple[int, float, float, tuple[float, float]]] = {}
    for i, op in enumerate(operations):
        if op.operation != "extrude":
            continue
        if op.profile.shape != "polyline" or not op.profile.vertices:
            continue
        info = _polyline_to_regular_polygon(list(op.profile.vertices))
        if info is None:
            continue
        n, r_mm, rot, ctr = info
        polygon_info[i] = info
        by_n.setdefault(n, []).append(i)

    out = list(operations)
    for n, idxs in by_n.items():
        if len(idxs) < 2:
            continue
        # Canonical rotation: pick the rotation MODULO (2π/N) of the
        # FIRST tier on this plane (the base extrude on XY/XZ/YZ — most
        # accurate detection because it has the most pixels). Others
        # snap to this, normalising the modulo offset.
        canonical_rot = polygon_info[idxs[0]][2]
        canonical_mod = canonical_rot % (2 * np.pi / n)
        for j in idxs:
            n_j, r_j, rot_j, ctr_j = polygon_info[j]
            current_mod = rot_j % (2 * np.pi / n)
            # Find the offset that aligns current_mod to canonical_mod.
            delta = canonical_mod - current_mod
            # Wrap delta into [-π/n, π/n] so we apply the smallest rotation.
            half_period = np.pi / n
            while delta > half_period:
                delta -= 2 * half_period
            while delta < -half_period:
                delta += 2 * half_period
            if abs(delta) < 1e-4:
                continue  # already aligned
            new_rot = rot_j + delta
            verts_mm = []
            for k in range(n):
                ang = new_rot + 2.0 * np.pi * k / n
                verts_mm.append((
                    ctr_j[0] + r_j * np.cos(ang),
                    ctr_j[1] + r_j * np.sin(ang),
                ))
            xs = [v[0] for v in verts_mm]
            ys = [v[1] for v in verts_mm]
            new_profile = out[j].profile.model_copy(update={
                "vertices": verts_mm,
                "width_mm": max(xs) - min(xs),
                "depth_mm": max(ys) - min(ys),
            })
            out[j] = out[j].model_copy(update={"profile": new_profile})
    return out


def _try_fit_regular_polygon(
    smooth_polyline_px: list[tuple[int, int]],
    n_target: int,
) -> tuple[float, float, float, tuple[float, float]] | None:
    """Force-fit a regular N-gon to a contour at a SPECIFIC N (used by
    the cross-tier consistency pass after a dominant N has been picked).

    Returns ``(score, radius_px, rotation_rad, centre_xy)`` if the fit
    is acceptable, where score is the radial RMS as a fraction of mean
    radius (smaller = better). Differs from ``detect_regular_polygon``
    in that we DON'T scan a range of N — we already know which N to
    target — and the tolerances are correspondingly looser.
    """
    import cv2 as _cv2
    if not smooth_polyline_px or len(smooth_polyline_px) < 8:
        return None
    dense_np = np.array(smooth_polyline_px, dtype=np.int32).reshape(-1, 1, 2)
    perim = float(_cv2.arcLength(dense_np, closed=True))

    def _score(approx_pts: np.ndarray) -> tuple[float, float, float, tuple[float, float]] | None:
        if len(approx_pts) != n_target:
            return None
        pts = approx_pts.astype(np.float64)
        ctr = pts.mean(axis=0)
        rs = np.linalg.norm(pts - ctr, axis=1)
        mean_r = rs.mean()
        if mean_r <= 0:
            return None
        s = float(rs.std() / mean_r)
        if s > 0.20:
            return None
        v0 = pts[0] - ctr
        return (s, float(mean_r), float(np.arctan2(v0[1], v0[0])),
                (float(ctr[0]), float(ctr[1])))

    # Pass 1: try the dense contour directly at several epsilons. Works
    # for clean polylines where each side approximates one straight edge.
    for eps_frac in (0.005, 0.012, 0.020, 0.030, 0.045, 0.060, 0.080):
        eps = max(2.0, perim * eps_frac)
        approx = _cv2.approxPolyDP(dense_np, eps, closed=True).reshape(-1, 2)
        result = _score(approx)
        if result is not None:
            return result

    # Pass 2: simplify via the CONVEX HULL first, then approxPolyDP. The
    # convex hull strips the interior wobbles a noisy octagon picks up
    # from cleanup-time rounded corners — what's left is the 8 corners
    # plus 0-2 stray hull points. A modest epsilon on the hull then
    # gives a clean N-gon for shapes whose dense polyline never reduced
    # to exactly N at any of the pass-1 epsilons.
    hull = _cv2.convexHull(dense_np)
    if hull is not None and len(hull) >= n_target:
        hull_perim = float(_cv2.arcLength(hull, closed=True))
        for eps_frac in (0.005, 0.012, 0.020, 0.035, 0.060):
            eps = max(2.0, hull_perim * eps_frac)
            approx = _cv2.approxPolyDP(hull, eps, closed=True).reshape(-1, 2)
            result = _score(approx)
            if result is not None:
                return result
    return None


def _harmonize_tier_classifications(
    zones: list, frame: "WorldFrame",
) -> dict[int, ClassifiedProfile]:
    """Cross-tier consistency: if multiple tiers fit the same regular N,
    promote the OTHER tiers to that same N if they almost qualify.

    The motivation is 117514's 3-tier octagonal post: tier 2 is detected
    as regular-8gon but tiers 0 and 1 come back as polyline / arc_line.
    All three are octagons of different sizes; they should all be
    rendered as 8-fold symmetric.

    Returns ``{zone_index: ClassifiedProfile}`` only for zones that get
    upgraded to a consistent regular N. Other zones stay with whatever
    the per-tier classifier picked.
    """
    upgrades: dict[int, ClassifiedProfile] = {}
    # Per-zone first-pass classification (label only).
    labels = []
    for z in zones:
        cs = z.cross_section
        cls = _profile_from_outline_polyline_bbox(
            cs.outline, cs.smooth_polyline_px, cs.bbox_px, frame.mm_per_px,
        )
        labels.append(cls.label)

    # Count regular-N hits.
    counts: dict[int, int] = {}
    for label in labels:
        if label.startswith("regular-"):
            try:
                n = int(label.split("regular-")[1].split("gon")[0])
            except (IndexError, ValueError):
                continue
            counts[n] = counts.get(n, 0) + 1

    if not counts:
        return upgrades

    # Pick the dominant N (most occurrences; tie-break by smaller N).
    dominant_n = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # Re-classify each non-matching tier by force-fitting at dominant_n.
    for i, (zone, label) in enumerate(zip(zones, labels)):
        if label.startswith(f"regular-{dominant_n}gon"):
            continue
        cs = zone.cross_section
        fit = _try_fit_regular_polygon(cs.smooth_polyline_px, dominant_n)
        if fit is None:
            continue
        score, r_px, rot_rad, ctr = fit
        # Build the perfect N-gon profile in mm.
        bx, by, bw, bh = cs.bbox_px
        cx_px = bx + bw / 2.0
        cy_px = by + bh / 2.0

        def _to_mm(p):
            return ((p[0] - cx_px) * frame.mm_per_px,
                    -(p[1] - cy_px) * frame.mm_per_px)

        verts_mm = []
        for k in range(dominant_n):
            ang = rot_rad + 2.0 * np.pi * k / dominant_n
            vx_px = ctr[0] + r_px * np.cos(ang)
            vy_px = ctr[1] + r_px * np.sin(ang)
            verts_mm.append(_to_mm((vx_px, vy_px)))
        xs = [v[0] for v in verts_mm]
        ys = [v[1] for v in verts_mm]
        upgrades[i] = ClassifiedProfile(
            profile=Profile2D(
                shape="polyline",
                width_mm=max(xs) - min(xs),
                depth_mm=max(ys) - min(ys),
                vertices=verts_mm,
            ),
            label=f"regular-{dominant_n}gon(harmonised)",
        )
    return upgrades


def _harmonize_stacked_tier_classifications(
    tiers: list[TierRegion], frame: "WorldFrame",
) -> dict[int, ClassifiedProfile]:
    """Cross-tier consistency for per-view stacked tiers.

    The per-view stack path can still classify one tier as a regular
    octagon while a neighbouring tier falls through to ``polyline`` or,
    worse, a near-circular cleanup pass makes the smallest tier trip the
    circle detector. When one regular N dominates the stack, force-fit
    the other tiers to that same N so a true stepped octagonal post
    stays octagonal all the way up.
    """
    upgrades: dict[int, ClassifiedProfile] = {}
    labels: list[str] = []
    fit_by_tier: dict[int, dict[int, tuple[float, float, float, tuple[float, float]]]] = {}
    counts: dict[int, int] = {}
    score_sums: dict[int, float] = {}

    for i, tier in enumerate(tiers):
        cls = _profile_from_outline_polyline_bbox(
            tier.outline, tier.smooth_polyline_px, tier.bbox_px, frame.mm_per_px,
        )
        labels.append(cls.label)
        # Only polygon-like tiers are allowed to vote for a dominant N.
        # Mixed arc/line silhouettes (D-cuts, obrounds, scalloped
        # flanges) can often be crudely approximated by a regular n-gon
        # but should not drag the whole stack into polygon mode.
        if cls.profile.shape not in {"polyline", "circle"} and not cls.label.startswith("regular-"):
            continue
        for n in range(5, 13):
            fit = _try_fit_regular_polygon(tier.smooth_polyline_px, n)
            if fit is None:
                continue
            score, _, _, _ = fit
            # Keep only high-confidence fits when voting for the stack's
            # dominant symmetry. Loose fits are still useful once the
            # dominant N is known, but they should not set it.
            if score > 0.08:
                continue
            fit_by_tier.setdefault(i, {})[n] = fit
            counts[n] = counts.get(n, 0) + 1
            score_sums[n] = score_sums.get(n, 0.0) + score

    # Also count any tier that was already classified as a regular N-gon.
    for label in labels:
        if not label.startswith("regular-"):
            continue
        try:
            n = int(label.split("regular-")[1].split("gon")[0])
        except (IndexError, ValueError):
            continue
        counts[n] = counts.get(n, 0) + 1

    if not counts:
        return upgrades

    dominant_n, dominant_count = sorted(
        counts.items(),
        key=lambda x: (
            -x[1],
            score_sums.get(x[0], float("inf")) / max(x[1], 1),
            -x[0],
        ),
    )[0]
    if dominant_count < 2:
        return upgrades

    for i, (tier, label) in enumerate(zip(tiers, labels)):
        if label.startswith(f"regular-{dominant_n}gon"):
            continue
        current_shape = _profile_from_outline_polyline_bbox(
            tier.outline, tier.smooth_polyline_px, tier.bbox_px, frame.mm_per_px,
        ).profile.shape
        # Allow arc_line outlines to be promoted too when the rest of the
        # stack is a clean N-gon — cleanup-time rounded corners give the
        # segmenter ~2-3 spurious arcs along an octagon perimeter, and
        # without this branch the largest tier stays as a 22-segment
        # arc_line while the smaller tiers all become clean octagons.
        # The follow-up fit-quality check (score ≤ 0.20) still rejects
        # genuine curved shapes like D-cuts and obrounds.
        if current_shape not in {"polyline", "circle", "arc_line"}:
            continue
        fit = fit_by_tier.get(i, {}).get(dominant_n)
        if fit is None:
            # Dominant N already exists elsewhere in the stack; allow a
            # looser per-tier fit now when trying to upgrade stragglers.
            fit = _try_fit_regular_polygon(tier.smooth_polyline_px, dominant_n)
        if fit is None:
            continue
        score, r_px, rot_rad, ctr = fit
        if score > 0.20:
            continue
        bx, by, bw, bh = tier.bbox_px
        cx_px = bx + bw / 2.0
        cy_px = by + bh / 2.0

        def _to_mm(p):
            return ((p[0] - cx_px) * frame.mm_per_px,
                    -(p[1] - cy_px) * frame.mm_per_px)

        verts_mm = []
        for k in range(dominant_n):
            ang = rot_rad + 2.0 * np.pi * k / dominant_n
            vx_px = ctr[0] + r_px * np.cos(ang)
            vy_px = ctr[1] + r_px * np.sin(ang)
            verts_mm.append(_to_mm((vx_px, vy_px)))
        xs = [v[0] for v in verts_mm]
        ys = [v[1] for v in verts_mm]
        upgrades[i] = ClassifiedProfile(
            profile=Profile2D(
                shape="polyline",
                width_mm=max(xs) - min(xs),
                depth_mm=max(ys) - min(ys),
                vertices=verts_mm,
            ),
            label=f"regular-{dominant_n}gon(harmonised)",
        )
    return upgrades


def _build_sliced_extrudes(
    slices: AxisSlices,
    frame: WorldFrame,
    base_view: ViewFeatures | None = None,
    through_hole_diameters_mm: list[float] | None = None,
) -> tuple[list[SketchOperation], str]:
    """Convert per-zone cross-sections into ordered extrude operations.

    Each zone = one extrude on the appropriate plane (XY/XZ/YZ for the
    first, then >Z/>Y/>X for subsequent stacked extrudes). Cross-section
    interior holes become cut operations on the same face.
    """
    if not slices.zones:
        return [], "no zones"
    spec = _SLICE_AXIS_MAP.get(slices.axis)
    if spec is None:
        return [], f"unknown axis {slices.axis}"
    base_plane = spec["plane"]
    face_top = spec["face_top"]
    axis_total_mm = max(getattr(frame, spec["axis_attr"]), 0.001)

    operations: list[SketchOperation] = []
    notes_parts: list[str] = []
    order = 1

    # Track the PREVIOUS tier's cross-section centre so each new tier's
    # offset is computed relative to the workplane that CadQuery is
    # actually placing it on (the previous tier's top face, centred on
    # its bbox via CenterOfBoundBox). Computing offsets relative to the
    # first tier instead caused cumulative drift down the stack.
    prev_cx_px: float | None = None
    prev_cy_px: float | None = None

    # Cross-tier consistency: if any tier is detected as a regular N-gon,
    # try to harmonise the OTHER tiers to the same N. A 3-tier octagonal
    # post should be three octagons of different sizes, not "polyline +
    # arc_line + octagon".
    upgrades = _harmonize_tier_classifications(slices.zones, frame)

    # Tier-height accounting: extrude each tier 1:1 with its actual
    # Z extent in the part. Tier 0 starts at the FIRST zone's start
    # (where the part actually begins, NOT at rel=0 which is below
    # the part). Each subsequent tier extends from previous zone's
    # end to its own zone end. Total height = part's actual Z extent.
    first_start_norm = slices.zones[0].start_norm if slices.zones else 0.0
    prev_end_norm = first_start_norm
    for i, zone in enumerate(slices.zones):
        cs = zone.cross_section
        tier_top_norm = zone.end_norm
        height_norm = max(tier_top_norm - prev_end_norm, 0.001)
        height_mm = max(height_norm * axis_total_mm, 0.5)
        prev_end_norm = tier_top_norm

        if i in upgrades:
            cls = upgrades[i]
        elif i == 0 and base_view is not None and base_view.outline:
            # CRITICAL: tier 0 (the BASE) uses the per-view OpenCV
            # outline directly — that's the cleanest extraction. The
            # slice cross-section can have small contour-finding
            # artifacts at the silhouette boundary; the per-view outline
            # is the same shape but cleaner.
            cls = _profile_from_outline_polyline_bbox(
                base_view.outline, base_view.smooth_polyline_px,
                base_view.bbox_px, frame.mm_per_px,
            )
        else:
            cls = _profile_from_outline_polyline_bbox(
                cs.outline, cs.smooth_polyline_px, cs.bbox_px, frame.mm_per_px,
            )

        cbx, cby, cbw, cbh = cs.bbox_px
        cx_px = cbx + cbw / 2.0
        cy_px = cby + cbh / 2.0
        if i == 0:
            pos_x_mm = 0.0
            pos_y_mm = 0.0
            plane = base_plane
            # Track the FIRST tier's bbox so subsequent tiers can snap
            # against it (the workplane is on tier 0's top face).
            base_bbox_min_dim = min(cbw, cbh)
        else:
            dx_px = cx_px - prev_cx_px
            dy_px = cy_px - prev_cy_px
            # Concentric snap: if the centre-to-centre offset is within
            # _SLICE_CONCENTRIC_TOL_FRAC of the LARGER of (current or
            # base tier) min-dim, treat as exactly concentric. Using the
            # base tier's size is more permissive — a small upper tier
            # whose centroid is pulled off by an asymmetric feature (e.g.
            # 117514's loop tab) still snaps to the central axis when
            # the offset is small relative to the part's overall scale.
            tol_px = max(min(cbw, cbh), base_bbox_min_dim) * _SLICE_CONCENTRIC_TOL_FRAC
            if abs(dx_px) < tol_px:
                dx_px = 0.0
            if abs(dy_px) < tol_px:
                dy_px = 0.0
            pos_x_mm = dx_px * frame.mm_per_px
            pos_y_mm = -dy_px * frame.mm_per_px
            plane = face_top

        operations.append(SketchOperation(
            order=order,
            plane=plane,
            profile=cls.profile,
            operation="extrude",
            distance_mm=height_mm,
            direction="positive",
            position_x=pos_x_mm,
            position_y=pos_y_mm,
        ))
        order += 1
        notes_parts.append(f"z{i}({cls.label},h={height_mm:.1f})")
        prev_cx_px = cx_px
        prev_cy_px = cy_px

        # Interior holes in this cross-section: real interior features
        # (counterbores, recesses, blind bores) need to be emitted, but
        # gpt-image-2 cleanup also leaves pinhole / boundary-clipping
        # artifacts that look like fake holes. Discriminator:
        #   - bbox aspect close to 1 (real recesses are roughly round)
        #   - position not at the silhouette boundary (avoid clipping)
        #   - substantial area relative to the tier (≥ 3%)
        # When a hole passes, emit as a clean CIRCLE of equivalent
        # diameter so the rebuild is parametric (no jagged edges).
        # Only emit slice-cross-section holes from the TOP tier (counter-
        # bores / surface recesses live there). Lower tiers' interior
        # "holes" are just the through-hole appearing in the cross-
        # section at that Z depth — emitting them creates duplicate cuts
        # that conflict with the per-view-matched through-hole. The
        # per-view through-hole (Top↔Bottom matched) is emitted later as
        # one clean cut through the full stack.
        is_top_tier = (i == len(slices.zones) - 1)
        cbx, cby, cbw, cbh = cs.bbox_px
        for hole in cs.holes if is_top_tier else []:
            bx_h, by_h, bw_h, bh_h = hole.bbox_px
            aspect_h = bw_h / bh_h if bh_h else 0
            if aspect_h < 0.6 or aspect_h > 1.7:
                continue           # weird-shaped → clipping artifact
            if hole.area_px / cs.area_px < 0.03:
                continue           # too small relative to the tier
            # Reject holes whose centre is too close to the silhouette
            # boundary (boundary clipping artifacts).
            margin_x = 0.10 * cbw
            margin_y = 0.10 * cbh
            if (hole.centre_xy[0] < cbx + margin_x
                    or hole.centre_xy[0] > cbx + cbw - margin_x
                    or hole.centre_xy[1] < cby + margin_y
                    or hole.centre_xy[1] > cby + cbh - margin_y):
                continue
            d_mm = hole.equivalent_diameter_px * frame.mm_per_px
            # DEDUP: skip slice holes that duplicate a per-view-matched
            # through-hole. The per-view matching produces a single clean
            # cut through the entire stack; emitting the same hole again
            # at every tier creates conflicting geometry that fills the
            # bore back in (boss above tier 0's hole-cut covers the hole).
            if through_hole_diameters_mm:
                hpx, hpy = hole.centre_xy
                # The per-view through-hole snaps to the tier centre after
                # the constraint pass; here we check raw distance.
                centre_dist_mm = (
                    ((hpx - cx_px) ** 2 + (hpy - cy_px) ** 2) ** 0.5
                ) * frame.mm_per_px
                near_centre = centre_dist_mm < 0.10 * d_mm
                duplicates_through = any(
                    abs(d_mm - tdm) / max(tdm, 0.001) < 0.10
                    for tdm in through_hole_diameters_mm
                )
                if near_centre and duplicates_through:
                    continue
            profile = Profile2D(
                shape="circle", width_mm=d_mm, depth_mm=d_mm, diameter_mm=d_mm,
            )
            hpx, hpy = hole.centre_xy
            hpos_x_mm = (hpx - cx_px) * frame.mm_per_px
            hpos_y_mm = -(hpy - cy_px) * frame.mm_per_px
            operations.append(SketchOperation(
                order=order,
                plane=face_top,
                profile=profile,
                operation="cut",
                distance_mm=height_mm + 1.0,
                direction="negative",
                position_x=hpos_x_mm,
                position_y=hpos_y_mm,
            ))
            order += 1
            notes_parts.append(f"z{i}.hole(circle d={d_mm:.1f})")

    return operations, f"sliced[{slices.axis}]: " + " + ".join(notes_parts)


_BASE_VIEW_SPECS: dict[str, dict] = {
    # view_name → {plane, axis_attr, opp_view, axis_label, hole_face}
    "Top":    {"plane": "XY", "axis_attr": "z_mm", "opp": "Bottom", "axis_label": "Z", "hole_face": ">Z"},
    "Front":  {"plane": "XZ", "axis_attr": "y_mm", "opp": "Back",   "axis_label": "Y", "hole_face": ">Y"},
    "Right":  {"plane": "YZ", "axis_attr": "x_mm", "opp": "Left",   "axis_label": "X", "hole_face": ">X"},
}


def _pick_base_view(views: dict[str, ViewFeatures]) -> tuple[ViewFeatures, str]:
    """Pick the view whose silhouette captures the most engineering intent.

    Heuristic: most outline segments + interior holes weighted heavily +
    depth-tier count weighted moderately. Tie-breaker prefers Top so
    parts that are extruded along Z (the common case) keep the
    historical XY base behaviour.

    For 128105 (rectangular bar with cylindrical bore-ends): Top is a
    plain 4-line rectangle, Front is the actual dumbbell silhouette
    (arc+line+arc+line + 2 holes). Front wins, base becomes XZ, extrude
    along Y → captures the dumbbell profile correctly.
    """
    def _score(v: ViewFeatures) -> float:
        # ARC presence is the only reliable signal that this view
        # captures geometric intent another view's projection would lose.
        # Holes are NOT counted: a through-hole on a side axis appears
        # as an interior hole in side views without that view being the
        # right base (e.g. 117514's octagonal post has a Y-axis hole
        # visible in Front, but Top is still the right base because
        # the post's cross-section is octagonal). When no view has any
        # arcs, all candidates tie and the rank table prefers Top.
        return float(sum(1 for s in v.outline if s.kind == "arc"))

    candidates = []
    for name in ("Top", "Front", "Right"):
        v = views.get(name)
        if v is not None and v.polygon_px:
            candidates.append((name, v, _score(v)))
    if not candidates:
        # Last resort: pick whichever view has any polygon at all.
        v = max(views.values(), key=lambda x: len(x.polygon_px))
        return v, "Top"

    # Tie-break: prefer Top (Z extrude is the historical default).
    # Within a tie, also keep Front higher than Right (more common axis).
    rank = {"Top": 0, "Front": 1, "Right": 2}
    candidates.sort(key=lambda c: (-c[2], rank.get(c[0], 9)))
    name, view, _ = candidates[0]
    return view, name


def _apply_geometric_constraints(
    operations: list[SketchOperation], frame: WorldFrame,
) -> list[SketchOperation]:
    """Post-process operation positions to enforce design intent constraints.

    Two constraints are inferred from the numbers:
      * AXIS-CENTRED: a position component very close to 0 (i.e. on the
        sketch plane's axis through the origin) snaps to exactly 0. This
        is the parametric equivalent of a "coincident with axis" or
        "centre on origin" constraint.
      * MIRROR-SYMMETRIC PAIR: two operations on the same plane whose
        centre vectors mirror each other (similar shapes, opposite sign
        on one component) snap to exact mirror coordinates. This is the
        parametric equivalent of a "symmetric across axis" constraint.

    Tolerances scale with the part's largest dimension so that small
    parts and big parts both get sensible snapping.
    """
    if not operations:
        return operations
    part_max_mm = max(frame.x_mm, frame.y_mm, frame.z_mm, 1.0)
    axis_tol = part_max_mm * _POS_SNAP_AXIS_FRAC
    pair_tol = part_max_mm * _POS_SNAP_PAIR_TOL_FRAC

    def _snapped(v: float) -> float:
        return 0.0 if abs(v) < axis_tol else v

    # Pass 1: axis-centred snap.
    snapped = []
    for op in operations:
        new_x = _snapped(op.position_x)
        new_y = _snapped(op.position_y)
        if new_x != op.position_x or new_y != op.position_y:
            snapped.append(op.model_copy(update={
                "position_x": new_x,
                "position_y": new_y,
            }))
        else:
            snapped.append(op)

    # Pass 1.5: equal-diameter snap for circles on the same plane.
    # Two cuts/extrudes of nearly-identical-diameter circles on the
    # same workplane usually express "matching holes" as a parametric
    # equality (two through-bores of the same drill bit).
    by_plane_circles: dict[str, list[int]] = {}
    for i, op in enumerate(snapped):
        if op.profile.shape == "circle" and op.profile.diameter_mm is not None:
            by_plane_circles.setdefault(op.plane, []).append(i)
    for plane_name, idxs in by_plane_circles.items():
        if len(idxs) < 2:
            continue
        diams = [snapped[i].profile.diameter_mm for i in idxs]
        max_d = max(diams)
        if max_d <= 0:
            continue
        # Cluster diameters: within tolerance, snap to cluster mean.
        clusters: list[list[int]] = []
        for j, i in enumerate(idxs):
            d = snapped[i].profile.diameter_mm
            placed = False
            for c in clusters:
                c_mean = sum(snapped[k].profile.diameter_mm for k in c) / len(c)
                if abs(d - c_mean) / max_d < _DIAM_EQUAL_TOL_FRAC:
                    c.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        for c in clusters:
            if len(c) < 2:
                continue
            mean_d = sum(snapped[i].profile.diameter_mm for i in c) / len(c)
            for i in c:
                op = snapped[i]
                new_profile = op.profile.model_copy(update={
                    "diameter_mm": mean_d,
                    "width_mm": mean_d,
                    "depth_mm": mean_d,
                })
                snapped[i] = op.model_copy(update={"profile": new_profile})

    # Pass 2: mirror-symmetric pair snap. Walk operations on the same
    # plane in order; for each pair (i, j) where positions look like
    # near-mirrors (matching profile shape, opposite-sign component),
    # snap to exact ±|component|.
    def _profile_signature(p: Profile2D) -> tuple:
        return (p.shape, round(p.width_mm, 1), round(p.depth_mm, 1),
                round(p.diameter_mm or 0, 1))

    by_plane: dict[str, list[int]] = {}
    for i, op in enumerate(snapped):
        by_plane.setdefault(op.plane, []).append(i)

    for plane, idxs in by_plane.items():
        if len(idxs) < 2:
            continue
        for i_a in idxs:
            a = snapped[i_a]
            sig_a = _profile_signature(a.profile)
            for i_b in idxs:
                if i_b <= i_a:
                    continue
                b = snapped[i_b]
                if _profile_signature(b.profile) != sig_a:
                    continue
                # Mirror across X axis: a.y ≈ -b.y, a.x ≈ b.x.
                # Mirror across Y axis: a.x ≈ -b.x, a.y ≈ b.y.
                for axis in ("x", "y"):
                    if axis == "x":
                        same_a, same_b = a.position_y, b.position_y
                        opp_a, opp_b = a.position_x, b.position_x
                    else:
                        same_a, same_b = a.position_x, b.position_x
                        opp_a, opp_b = a.position_y, b.position_y
                    if abs(same_a - same_b) > pair_tol:
                        continue
                    if abs(opp_a + opp_b) > pair_tol:
                        continue
                    avg_same = (same_a + same_b) / 2.0
                    avg_mag = (abs(opp_a) + abs(opp_b)) / 2.0
                    sign_a = 1.0 if opp_a >= 0 else -1.0
                    sign_b = -sign_a
                    if axis == "x":
                        snapped[i_a] = a.model_copy(update={
                            "position_x": sign_a * avg_mag,
                            "position_y": _snapped(avg_same),
                        })
                        snapped[i_b] = b.model_copy(update={
                            "position_x": sign_b * avg_mag,
                            "position_y": _snapped(avg_same),
                        })
                    else:
                        snapped[i_a] = a.model_copy(update={
                            "position_x": _snapped(avg_same),
                            "position_y": sign_a * avg_mag,
                        })
                        snapped[i_b] = b.model_copy(update={
                            "position_x": _snapped(avg_same),
                            "position_y": sign_b * avg_mag,
                        })
                    break  # one mirror axis at most per pair
    return snapped


from pathlib import Path  # noqa: E402


def _stl_z_slice_areas(
    png_path: str | Path, n_slices: int = 80,
) -> tuple[list[float], list[float]] | None:
    """Sample the recon STL directly along Z and return (z_bin_centres_mm,
    vertex_counts_per_bin) lists. We bin the mesh's vertices by Z height
    (rather than running planar sections, which fail on non-watertight
    point-cloud reconstructions). The vertex count per Z bin is a strong
    proxy for cross-section size: planar tier faces concentrate many
    vertices at one Z, vertical walls spread them thinly. Transitions
    in the count profile mark tier boundaries in actual STL space —
    far more reliable than luma proportions from the cleaned PNG.
    """
    from pathlib import Path as _Path
    import re as _re
    p = _Path(png_path)
    m = _re.match(r"(deepcadimg_\d+)_(?:geometry_clean|clean_input)\.png", p.name)
    if not m:
        return None
    sid = m.group(1)
    import json as _json
    manifest_path = p.parent.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _json.loads(manifest_path.read_text())
    except Exception:
        return None
    entry = next((e for e in manifest if e.get("sample_id") == sid), None)
    if not entry or not entry.get("recon_noisy_stl"):
        return None
    stl_path = p.parent.parent / entry["recon_noisy_stl"]
    if not stl_path.exists():
        return None
    try:
        import trimesh
        import numpy as np
        mesh = trimesh.load(str(stl_path))
        if hasattr(mesh, "is_empty") and mesh.is_empty:
            return None
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            return None
        z_vals = verts[:, 2]
        z_lo, z_hi = float(z_vals.min()), float(z_vals.max())
        if z_hi <= z_lo:
            return None
        # Histogram vertex counts in n_slices Z bins. Triangle-mesh
        # tier faces concentrate vertices at a fixed Z (lots of
        # triangles tessellate the planar cap), so high-count bins
        # mark planar tiers and gaps mark the vertical walls between.
        counts, edges = np.histogram(z_vals, bins=n_slices, range=(z_lo, z_hi))
        centres = (edges[:-1] + edges[1:]) / 2.0
        return ([float(c) for c in centres], [float(c) for c in counts])
    except Exception:
        return None


def _stl_xy_bboxes_per_band(
    png_path: str | Path, z_anchors: list[float],
) -> list[tuple[float, float, float, float, float, float]] | None:
    """For each tier defined by Z range [z_anchors[i], z_anchors[i+1]],
    sample a THIN SLAB just below the tier's top (its next-anchor Z) and
    return (x_min, x_max, y_min, y_max, x_centre, y_centre) of the STL
    vertices in that slab — all in STL world units.

    Why a slab below the top, not the full band: the tier's top face is
    a planar cap with the tier's TRUE cross-section. Its bottom edge
    coincides with the previous tier's top face — wider than this tier.
    Sampling the full band would include the previous tier's wide top
    face and overstate this tier's diameter (e.g. the MIDDLE disc would
    measure as the BIG flange because the BIG flange's top face vertices
    sit at the bottom of the MIDDLE Z band).
    """
    from pathlib import Path as _Path
    import re as _re
    p = _Path(png_path)
    m = _re.match(r"(deepcadimg_\d+)_(?:geometry_clean|clean_input)\.png", p.name)
    if not m:
        return None
    sid = m.group(1)
    import json as _json
    manifest_path = p.parent.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _json.loads(manifest_path.read_text())
    except Exception:
        return None
    entry = next((e for e in manifest if e.get("sample_id") == sid), None)
    if not entry or not entry.get("recon_noisy_stl"):
        return None
    stl_path = p.parent.parent / entry["recon_noisy_stl"]
    if not stl_path.exists():
        return None
    try:
        import trimesh
        import numpy as np
        mesh = trimesh.load(str(stl_path))
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            return None
        # Thin slab thickness = SLAB_FRAC of each band's height. 15% gives
        # enough vertices to define the bbox while staying clear of the
        # adjacent tier's wider face.
        SLAB_FRAC = 0.15
        out: list[tuple[float, float, float, float, float, float]] = []
        for i in range(len(z_anchors) - 1):
            z_lo, z_hi = float(z_anchors[i]), float(z_anchors[i + 1])
            band_h = z_hi - z_lo
            # Slab from (z_top - slab_h) up to z_top, where slab_h is
            # SLAB_FRAC of the band height (or a small floor).
            slab_h = max(band_h * SLAB_FRAC, abs(band_h) * 0.05, 1e-3)
            slab_lo = z_hi - slab_h
            slab_hi = z_hi
            slab = verts[(verts[:, 2] >= slab_lo) & (verts[:, 2] <= slab_hi)]
            if slab.size == 0:
                # Slab missed all vertices — widen progressively.
                for widen in (2.0, 4.0, 8.0):
                    slab_lo2 = z_hi - slab_h * widen
                    slab = verts[(verts[:, 2] >= slab_lo2) & (verts[:, 2] <= slab_hi)]
                    if slab.size > 0:
                        break
            if slab.size == 0:
                # Fall back to the full band, then to the entire mesh.
                slab = verts[(verts[:, 2] >= z_lo) & (verts[:, 2] <= z_hi)]
            if slab.size == 0:
                slab = verts
            x_lo, x_hi = float(slab[:, 0].min()), float(slab[:, 0].max())
            y_lo, y_hi = float(slab[:, 1].min()), float(slab[:, 1].max())
            cx = (x_lo + x_hi) / 2.0
            cy = (y_lo + y_hi) / 2.0
            out.append((x_lo, x_hi, y_lo, y_hi, cx, cy))
        return out
    except Exception:
        return None


def _tier_z_anchors_from_stl(
    png_path: str | Path, n_tiers: int,
) -> list[float] | None:
    """Find n_tiers+1 Z anchors marking tier boundaries [z_min, ...,
    z_max] from the STL vertex distribution.

    Strategy: planar tier-top faces concentrate many vertices at one Z
    (the cap is tessellated into many triangles). Locate the top peaks
    in the vertex-count histogram, drop peaks at the very bottom (the
    part's bottom face is not a tier transition), and use the surviving
    peaks as tier-top Z values. The first tier spans [z_min, peak_1],
    the second [peak_1, peak_2], etc.
    """
    sliced = _stl_z_slice_areas(png_path)
    if sliced is None:
        return None
    zs, counts = sliced
    if len(zs) < 8 or n_tiers < 1:
        return None
    import numpy as np
    c = np.array(counts, dtype=np.float32)
    if c.max() <= 0:
        return None
    z_min, z_max = float(zs[0]), float(zs[-1])
    z_span = z_max - z_min
    if z_span <= 0:
        return None
    # Find local maxima in the count profile. A bin counts as a peak
    # if it has more vertices than both adjacent bins. Then sort by
    # vertex count and filter:
    #   - drop peaks in the lowest 10% of Z (those are the part's
    #     bottom face, not a tier-top transition).
    #   - drop peaks in the highest 5% of Z (the part's topmost face is
    #     the LAST tier's top by definition — already at z_max).
    peaks: list[tuple[int, float]] = []  # (idx, count)
    for i in range(1, len(c) - 1):
        if c[i] > c[i - 1] and c[i] > c[i + 1]:
            peaks.append((i, float(c[i])))
    if not peaks:
        return None
    peaks.sort(key=lambda p: -p[1])
    drop_low_z = z_min + 0.10 * z_span
    drop_high_z = z_min + 0.95 * z_span
    candidates = [
        (idx, ct) for idx, ct in peaks
        if drop_low_z <= zs[idx] <= drop_high_z
    ]
    n_needed = max(n_tiers - 1, 0)
    if n_needed == 0:
        return [z_min, z_max]
    # Take the strongest n_needed candidates by vertex count, then
    # sort by Z so anchors come back in monotonic order.
    selected = sorted(candidates[:n_needed], key=lambda p: zs[p[0]])
    if len(selected) < n_needed:
        return None
    z_breaks = [float(zs[idx]) for idx, _ in selected]
    return [z_min] + z_breaks + [z_max]


def _stl_bbox_for_calibration(png_path: str | Path) -> tuple[float, float, float] | None:
    """Read the source recon STL's true world bbox so the rebuild can
    use 1:1 dimensions instead of the renderer's NORMALISE_LONGEST_MM
    (100 mm) approximation. Returns (x_mm, y_mm, z_mm) or None.

    The cleaned PNG lives next to a manifest that points to its
    recon STL — we look that up by sample id pattern.
    """
    import re, json
    p = Path(png_path)
    m = re.match(r"(deepcadimg_\d+)_(?:geometry_clean|clean_input)\.png", p.name)
    if not m:
        return None
    sid = m.group(1)
    manifest_path = p.parent.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    entry = next((e for e in manifest if e.get("sample_id") == sid), None)
    if not entry or not entry.get("recon_noisy_stl"):
        return None
    stl_path = p.parent.parent / entry["recon_noisy_stl"]
    if not stl_path.exists():
        return None
    try:
        import trimesh
        mesh = trimesh.load(str(stl_path))
        bx = mesh.bounding_box.extents
        return (float(bx[0]), float(bx[1]), float(bx[2]))
    except Exception:
        return None


def _stl_verts_projected_mm(
    png_path: str | Path, base_plane: str, frame: "WorldFrame",
) -> "np.ndarray | None":
    """Load the recon STL, project its vertices onto the base plane and
    rescale to mm using the same geometric-mean calibration as the rest
    of the rebuild. Returns an (N, 2) array of (u, v) coords in mm or
    None if the STL/manifest is missing.

    Plane mapping:
        XY -> (x, y), extrude axis = Z
        XZ -> (x, z), extrude axis = Y
        YZ -> (y, z), extrude axis = X
    The 2D coordinate convention matches the sketch frame used by
    _outline_to_arc_line_profile (image-y flipped to CAD-y on XY only,
    other planes keep STL signs).
    """
    import re, json
    p = Path(png_path)
    m = re.match(r"(deepcadimg_\d+)_(?:geometry_clean|clean_input)\.png", p.name)
    if not m:
        return None
    sid = m.group(1)
    manifest_path = p.parent.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    entry = next((e for e in manifest if e.get("sample_id") == sid), None)
    if not entry or not entry.get("recon_noisy_stl"):
        return None
    stl_path = p.parent.parent / entry["recon_noisy_stl"]
    if not stl_path.exists():
        return None
    try:
        import trimesh
        mesh = trimesh.load(str(stl_path))
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            return None
        # Geometric-mean scale: same calibration as _world_frame's STL
        # branch, so the projected mm values line up with arc centres
        # already computed in mm.
        ext = mesh.bounding_box.extents
        true_gm = float((ext[0] * ext[1] * ext[2]) ** (1.0 / 3.0)) or 1.0
        unit_to_mm = NORMALISE_GEOMETRIC_MEAN_MM / true_gm
        verts_mm = verts * unit_to_mm
        # Centre on the part's bbox centre (mm) so the projected coords
        # match arc centres expressed in bbox-centred frame.
        bbox_min = (np.asarray(mesh.bounds[0])) * unit_to_mm
        bbox_max = (np.asarray(mesh.bounds[1])) * unit_to_mm
        centre = (bbox_min + bbox_max) / 2.0
        verts_mm = verts_mm - centre
        if base_plane == "XY":
            return np.column_stack([verts_mm[:, 0], verts_mm[:, 1]])
        if base_plane == "XZ":
            return np.column_stack([verts_mm[:, 0], verts_mm[:, 2]])
        if base_plane == "YZ":
            return np.column_stack([verts_mm[:, 1], verts_mm[:, 2]])
        return None
    except Exception:
        return None


def _arc_centre_for_radius(
    start: tuple[float, float], end: tuple[float, float],
    radius: float, original_centre: tuple[float, float],
) -> tuple[float, float] | None:
    """Position an arc centre on the perpendicular bisector of (start,
    end) so that |start - centre| = |end - centre| = radius. Pick the
    side that matches the original centre (keeps the arc curving the
    same way as before). Returns None if the chord is longer than 2 ×
    radius (arc geometrically impossible at the requested size)."""
    s = np.array(start, dtype=np.float64)
    e = np.array(end, dtype=np.float64)
    mid = (s + e) / 2.0
    chord = float(np.linalg.norm(e - s))
    if chord <= 1e-9 or chord >= 2 * radius:
        return None
    h = float(np.sqrt(max(radius ** 2 - (chord / 2.0) ** 2, 0.0)))
    dir_chord = (e - s) / chord
    perp = np.array([-dir_chord[1], dir_chord[0]])
    c1 = mid + perp * h
    c2 = mid - perp * h
    oc = np.array(original_centre, dtype=np.float64)
    return tuple(c1.tolist()) if (
        np.linalg.norm(c1 - oc) < np.linalg.norm(c2 - oc)
    ) else tuple(c2.tolist())


def _override_arc_radii_from_stl(
    profile: "Profile2D", source_png: str | Path,
    base_plane: str, frame: "WorldFrame",
    scale_log: list | None = None,
) -> "Profile2D":
    """Restore asymmetric arc radii on an arc_line base profile by
    sampling the noisy recon STL at each arc's centre.

    The cleaned PNG averages similar-looking features (e.g. two
    dumbbell ends of slightly different sizes) into near-identical
    silhouettes, so the segmented arcs come back with similar radii.
    The noisy STL still has the original sizes — projecting it onto
    the base plane lets us measure the LOCAL part extent around each
    arc centre and rebuild the arc to that scale.

    The arc endpoints are held fixed (they sit on lines that must
    still meet at the same place), and the centre is repositioned on
    the perpendicular bisector of (start, end) to keep the arc
    well-formed. If the required radius is shorter than half the
    chord, the override is skipped for that arc.
    """
    if profile.shape != "arc_line" or not profile.arc_line_segments:
        return profile
    verts2d = _stl_verts_projected_mm(source_png, base_plane, frame)
    if verts2d is None or len(verts2d) < 16:
        return profile
    # Identify the long axis of the projection — for a dumbbell-like
    # part the long axis runs through both ball centres, and each
    # ball's RADIUS is the local extent PERPENDICULAR to that axis.
    # Sampling along the long axis avoids contaminating the ball
    # radius with material from the connecting rod (which would
    # otherwise dominate any direct distance-from-centre measure).
    proj_extent = verts2d.max(axis=0) - verts2d.min(axis=0)
    long_axis_idx = 0 if proj_extent[0] >= proj_extent[1] else 1
    perp_axis_idx = 1 - long_axis_idx

    new_segs: list[ArcLineSegment] = []
    overrides = 0
    for s in profile.arc_line_segments:
        if (s.kind != "arc" or s.arc_centre is None
                or not s.arc_radius_mm):
            new_segs.append(s)
            continue
        r_old = float(s.arc_radius_mm)
        centre = np.array(s.arc_centre, dtype=np.float64)
        # Window along the long axis: ±r_old captures just the ball,
        # not the connecting rod. The connecting rod's vertices sit
        # FURTHER along the long axis than r_old from the ball centre.
        window = r_old
        mask = np.abs(verts2d[:, long_axis_idx] - centre[long_axis_idx]) < window
        local = verts2d[mask]
        if len(local) < 8:
            new_segs.append(s)
            continue
        # Ball radius = local perpendicular extent / 2 (the ball is
        # approximately circular, so its perpendicular-axis range
        # equals its diameter).
        perp_lo = float(local[:, perp_axis_idx].min())
        perp_hi = float(local[:, perp_axis_idx].max())
        r_new = (perp_hi - perp_lo) / 2.0
        # Sanity-bound: keep within 0.5x..2.0x of the original arc
        # radius. Bigger jumps usually mean the projection picked up
        # an unrelated tier (multi-Z-level features) and we'd warp
        # the loop badly.
        if r_new < r_old * 0.5 or r_new > r_old * 2.0:
            new_segs.append(s)
            continue
        new_centre = _arc_centre_for_radius(
            s.start, s.end, r_new, s.arc_centre,
        )
        if new_centre is None:
            new_segs.append(s)
            continue
        new_segs.append(ArcLineSegment(
            kind="arc", start=s.start, end=s.end,
            arc_centre=new_centre, arc_radius_mm=r_new,
            arc_ccw=s.arc_ccw,
        ))
        if scale_log is not None and r_old > 0:
            # Record the per-arc scale so callers can rescale matching
            # features (e.g. through-holes drilled through this ball)
            # by the same factor.
            scale_log.append({
                "centre": tuple(s.arc_centre),
                "scale": r_new / r_old,
                "long_axis_idx": long_axis_idx,
            })
        overrides += 1
    if overrides == 0:
        return profile
    return profile.model_copy(update={"arc_line_segments": new_segs})


_OPPOSITE_VIEW = {
    "Top": "Bottom", "Bottom": "Top",
    "Front": "Back", "Back": "Front",
    "Right": "Left", "Left": "Right",
}


def _drop_unpaired_silhouette_holes(views: dict[str, ViewFeatures]) -> dict[str, ViewFeatures]:
    """Defense against gpt-image-2 hallucinating a circular hole into the
    silhouette half of a single view.

    A real through-hole shows as an interior_hole in BOTH end-on views
    (Top+Bottom, Front+Back, Right+Left). A hole that only appears in one
    of the pair is either a hallucination by cleanup or a real blind
    feature; either way it should not break slice-axis volume
    reconstruction. We drop any interior_hole whose opposite view has no
    matching hole at the mirrored relative position.
    """
    cleaned: dict[str, ViewFeatures] = {}
    for vname, vfeat in views.items():
        opp_name = _OPPOSITE_VIEW.get(vname)
        opp = views.get(opp_name) if opp_name else None
        if not vfeat.interior_holes or opp is None or not opp.interior_holes:
            # No holes here, or opposite view is missing / has zero holes
            # -> drop ALL holes on this view if opp has none and we have
            # any (defensive: keep when both empty so we don't churn).
            if vfeat.interior_holes and (opp is None or not opp.interior_holes):
                cleaned[vname] = replace(vfeat, interior_holes=[])
            else:
                cleaned[vname] = vfeat
            continue
        # Both views have at least one hole. Keep only holes on this view
        # that have a matching hole on the opposite view at a near-mirror
        # relative position (10% panel-size tolerance).
        bx_a, by_a, bw_a, bh_a = vfeat.bbox_px
        cx_a, cy_a = bx_a + bw_a / 2.0, by_a + bh_a / 2.0
        bx_b, by_b, bw_b, bh_b = opp.bbox_px
        cx_b, cy_b = bx_b + bw_b / 2.0, by_b + bh_b / 2.0
        tol = 0.10 * max(bw_a, bh_a, bw_b, bh_b, 1.0)
        kept: list = []
        for ha in vfeat.interior_holes:
            ra = (ha.centre_xy[0] - cx_a, ha.centre_xy[1] - cy_a)
            matched = False
            for hb in opp.interior_holes:
                rb = (hb.centre_xy[0] - cx_b, hb.centre_xy[1] - cy_b)
                # Try identity and mirror flips along U/V (opposite view
                # may have axis flipped depending on the ortho convention).
                for sx in (1.0, -1.0):
                    for sy in (1.0, -1.0):
                        d = ((ra[0] - sx * rb[0]) ** 2
                             + (ra[1] - sy * rb[1]) ** 2) ** 0.5
                        if d < tol:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                kept.append(ha)
        cleaned[vname] = replace(vfeat, interior_holes=kept)
    return cleaned


def infer_sketches(features: OrthoFeatures) -> SketchPartDescription:
    views = _drop_unpaired_silhouette_holes(features.views)
    frame = _world_frame(views)

    # Calibrate the per-axis ASPECT against the source STL's bbox, then
    # rescale so the longest axis is NORMALISE_LONGEST_MM (100mm). The
    # recon STL is in DeepCAD's normalized "unit cube" units (longest
    # axis ≈ 1-2), so reading the STL bbox literally would give a 1mm
    # part. We trust the STL for the X/Y/Z ratios (those carry the real
    # geometry) but override the absolute scale to a sensible display
    # size, mirroring the renderer's normalisation convention.
    stl_bbox = _stl_bbox_for_calibration(features.source_png)
    if stl_bbox is not None:
        x_mm_true, y_mm_true, z_mm_true = stl_bbox
        # Use the geometric mean of the STL bbox as the calibration anchor,
        # matching the cleaned-PNG pass above. Picks the same "neutral"
        # absolute size for any aspect ratio; the STL ratios still flow
        # through unchanged.
        true_gm = (x_mm_true * y_mm_true * z_mm_true) ** (1.0 / 3.0)
        if true_gm <= 0:
            true_gm = max(x_mm_true, y_mm_true, z_mm_true, 0.001)
        px_gm = (frame.x_px * frame.y_px * frame.z_px) ** (1.0 / 3.0)
        if px_gm <= 0:
            px_gm = max(frame.x_px, frame.y_px, frame.z_px, 1.0)
        # mm_per_px = (mm-per-unit) * (unit-per-px).
        unit_to_mm = NORMALISE_GEOMETRIC_MEAN_MM / true_gm
        frame = WorldFrame(
            x_px=frame.x_px,
            y_px=frame.y_px,
            z_px=frame.z_px,
            mm_per_px=(true_gm * unit_to_mm) / px_gm,
        )

    # PASS 0: try slice-based stacked extrudes. Slicing computes per-axis
    # cross-sections from opposite-view depth panels and finds the real
    # area-vs-position transitions — much cleaner than depth-tier-peak
    # detection for stepped parts. Only fires when at least one axis has
    # a strong stacked-cross-section signal.
    sliced_ops: list[SketchOperation] = []
    sliced_note = ""
    sliced_axis = ""
    try:
        axis_slices = compute_axis_slices(features.source_png)
        chosen = _pick_slicing_axis(axis_slices, views=views)
        if chosen is not None:
            # Pass the BASE view (matched to the chosen axis) so the
            # base tier can use its per-view outline directly.
            base_view_for_axis = views.get({"Z": "Top", "Y": "Front", "X": "Right"}.get(chosen.axis, "Top"))
            # Pre-compute the per-view through-hole diameters along the
            # chosen axis so we can dedup slice cross-section holes that
            # duplicate them.
            view_pair = {"Z": ("Top", "Bottom"), "Y": ("Front", "Back"), "X": ("Right", "Left")}.get(chosen.axis)
            through_hole_diams: list[float] = []
            if view_pair:
                va, vb = view_pair
                if views.get(va) and views.get(vb):
                    matched = _match_holes(views[va], views[vb], chosen.axis, frame)
                    through_hole_diams = [h.diameter_mm for h in matched]
            sliced_ops, sliced_note = _build_sliced_extrudes(
                chosen, frame, base_view=base_view_for_axis,
                through_hole_diameters_mm=through_hole_diams,
            )
            sliced_axis = chosen.axis
    except Exception as exc:
        sliced_note = f"slicing skipped: {exc}"

    # Pick the base view by silhouette complexity (only used when slicing
    # didn't produce a usable construction).
    top, base_view_name = _pick_base_view(views)
    spec = _BASE_VIEW_SPECS[base_view_name]
    base_plane = spec["plane"]
    base_axis_attr = spec["axis_attr"]
    base_axis_label = spec["axis_label"]
    base_extrude_axis_mm = max(getattr(frame, base_axis_attr), 0.001)

    operations: list[SketchOperation] = []
    construction_note: str

    # If slicing succeeded, use that construction (it's the cleanest
    # signal) — UNLESS the base view's per-pixel depth-tier detector
    # found strictly more tiers than slicing did. That case usually
    # means the part has stacked tiers whose Z transitions are too
    # smooth (rounded fillets / cleanup-softened edges) for the area-
    # gradient slicer to localize. Per-view tier_regions reads each
    # tier from a luma cluster directly, so it doesn't suffer from
    # the smearing problem.
    tier_ops: list[SketchOperation] = []
    construction_note = ""
    # Per-arc scale factors applied by the STL-anchored radius override.
    # Populated in the _base_profile branch below when the base is
    # arc_line; consumed later by the through-hole loop to rescale hole
    # diameters that match a ball whose arc got resized.
    arc_scale_log: list = []
    sliced_n_zones = sum(1 for op in (sliced_ops or []) if op.operation == "extrude")
    perview_n_tiers = len(top.tier_regions)
    use_sliced = bool(sliced_ops) and (
        sliced_axis != "Z" or sliced_n_zones >= perview_n_tiers
    )
    if use_sliced:
        tier_ops = sliced_ops
        construction_note = sliced_note
        # Map the chosen axis back to the base plane so subsequent code
        # that references base_plane / base_extrude_axis_mm uses the
        # axis the slicing picked instead of the view-based default.
        if sliced_axis in _SLICE_AXIS_MAP:
            sl_spec = _SLICE_AXIS_MAP[sliced_axis]
            base_plane = sl_spec["plane"]
            base_extrude_axis_mm = max(
                getattr(frame, sl_spec["axis_attr"]), 0.001,
            )
    elif perview_n_tiers >= 2:
        # Pass side views (Front/Right preferred, then Back/Left) so
        # the stacked builder can anchor BIG-flange thickness against
        # the side silhouette's actual Z extent, not luma proportions.
        side_views_for_stack = [
            v for v in (
                views.get("Front"), views.get("Right"),
                views.get("Back"),  views.get("Left"),
            ) if v is not None
        ]
        tier_ops, construction_note = _build_stacked_extrudes(
            top, frame, side_views=side_views_for_stack,
            source_png=features.source_png,
        )
        construction_note = (
            f"per-view tier_regions ({perview_n_tiers} tiers) chosen over "
            f"slicing ({sliced_n_zones} zones); "
        ) + construction_note
    if tier_ops:
        operations.extend(tier_ops)
        base = BaseProfile(
            profile=tier_ops[0].profile,
            flat_cuts=[],
            flat_cut_positions_mm=[],
            notes=construction_note,
        )
        extrude_distance_mm = sum(op.distance_mm for op in tier_ops) or base_extrude_axis_mm
        order = len(operations) + 1
    else:
        base = _base_profile(top, frame)
        extrude_distance_mm = base_extrude_axis_mm
        # If the base is arc_line, sample the noisy STL at each arc's
        # centre and rebuild arcs at the LOCAL part extent. Restores
        # size asymmetries (e.g. dumbbell ends of different diameters)
        # that the cleanup PNG smoothed into near-identical arcs.
        if base.profile.shape == "arc_line":
            anchored = _override_arc_radii_from_stl(
                base.profile, features.source_png, base_plane, frame,
                scale_log=arc_scale_log,
            )
            if anchored is not base.profile:
                base = BaseProfile(
                    profile=anchored,
                    flat_cuts=base.flat_cuts,
                    flat_cut_positions_mm=base.flat_cut_positions_mm,
                    notes=base.notes + " [STL-anchored arc radii]",
                )

        order = 1
        # 1) Base extrude on the chosen plane.
        operations.append(SketchOperation(
            order=order,
            plane=base_plane,
            profile=base.profile,
            operation="extrude",
            distance_mm=extrude_distance_mm,
            direction="positive",
            position_x=0.0,
            position_y=0.0,
        ))
        order += 1

        # 2) Flat cuts on the matching face selector.
        for cut_profile, (cx, cy) in zip(base.flat_cuts, base.flat_cut_positions_mm):
            operations.append(SketchOperation(
                order=order,
                plane=spec["hole_face"],
                profile=cut_profile,
                operation="cut",
                distance_mm=extrude_distance_mm + 1.0,
                direction="negative",
                position_x=cx,
                position_y=cy,
            ))
            order += 1

    # 3) Through-holes — match interior holes between opposite views.
    #    Z axis: Top ↔ Bottom (cut on >Z face).
    #    Y axis: Front ↔ Back (cut on >Y face).
    #    X axis: Right ↔ Left (cut on >X face).
    axis_specs = (
        ("Z", "Top",   "Bottom", ">Z", extrude_distance_mm),
        ("Y", "Front", "Back",   ">Y", max(frame.y_mm, 0.001)),
        ("X", "Right", "Left",   ">X", max(frame.x_mm, 0.001)),
    )
    hole_count_by_axis: dict[str, int] = {}
    for axis, va, vb, face_sel, axis_extent_mm in axis_specs:
        view_a = views.get(va)
        view_b = views.get(vb)
        if view_a is None or view_b is None:
            hole_count_by_axis[axis] = 0
            continue
        matched = _match_holes(view_a, view_b, axis, frame)
        hole_count_by_axis[axis] = len(matched)
        for hole in matched:
            d_mm = max(hole.diameter_mm, 0.001)
            # If a STL-anchored arc override resized a ball at this
            # hole's position, scale the hole diameter by the same
            # factor. Without this the dumbbell's two ends end up
            # different sizes (good) but with identical holes (bad).
            if arc_scale_log:
                hx, hy = hole.centre_mm
                best_scale = None
                best_d = float("inf")
                for entry in arc_scale_log:
                    cx, cy = entry["centre"]
                    d = ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_scale = entry["scale"]
                # Only apply if the hole sits inside a resized ball
                # (within ~1.5x the typical arc radius from its centre).
                if best_scale is not None and best_d < d_mm * 1.5:
                    d_mm = max(d_mm * best_scale, 0.001)
            operations.append(SketchOperation(
                order=order,
                plane=face_sel,
                profile=Profile2D(
                    shape="circle",
                    width_mm=d_mm,
                    depth_mm=d_mm,
                    diameter_mm=d_mm,
                ),
                operation="cut",
                distance_mm=axis_extent_mm + 1.0,
                direction="negative",
                position_x=hole.centre_mm[0],
                position_y=hole.centre_mm[1],
            ))
            order += 1
    z_holes_count = hole_count_by_axis.get("Z", 0)
    y_holes_count = hole_count_by_axis.get("Y", 0)
    x_holes_count = hole_count_by_axis.get("X", 0)

    # 4) Side-face bosses (-Y / +Y / -X / +X). Each side view's depth
    #    panel may contain features nearer than the face plane; those
    #    are bosses extruded outward from that face.
    side_ops, side_notes = _detect_side_face_bosses(views, frame, order)
    operations.extend(side_ops)
    order += len(side_ops)

    # Geometric constraint pass: snap near-axis positions to 0 (centre),
    # snap mirror pairs to exact ±same position, and align the rotation
    # of stacked regular polygons (parallel constraint). Together these
    # are the "concentric / symmetric / parallel" constraints a CAD
    # designer would set explicitly; we infer them from the numerics.
    operations = _harmonise_polygon_rotations(operations)
    operations = _apply_geometric_constraints(operations, frame)

    notes = (
        f"deterministic CV inference. Base: {base.notes}. "
        f"Bbox px (X/Y/Z) = {frame.x_px:.0f}/{frame.y_px:.0f}/{frame.z_px:.0f}, "
        f"mm/px = {frame.mm_per_px:.3f}. "
        f"D-cuts: {len(base.flat_cuts)}. "
        f"Through-holes Z/Y/X: {z_holes_count}/{y_holes_count}/{x_holes_count}. "
        f"Side-face: {', '.join(side_notes) if side_notes else 'none'}. "
        f"Dimensions auto-calibrated from noisy mesh "
        f"(geometric-mean = {NORMALISE_GEOMETRIC_MEAN_MM:.0f}mm); "
        f"override per-tier mm values to use exact specs."
    )

    # Confidence heuristic: high if base classified as a primitive AND we
    # found at least the expected matched holes; medium for D-cut
    # (geometric approximation); low for polyline fallback.
    if base.notes.startswith("polyline"):
        confidence = "low"
    elif base.notes.startswith("D-cut"):
        confidence = "medium"
    else:
        confidence = "high"

    return SketchPartDescription(
        sketches=operations,
        bounding_box_mm=(frame.x_mm, frame.y_mm, frame.z_mm),
        confidence=confidence,
        notes=notes,
    )
