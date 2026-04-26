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

from dataclasses import dataclass
from typing import Iterable

from backend.ai_infra.ortho_view_segmenter import (
    InteriorHole,
    OrthoFeatures,
    OutlineSegment,
    StraightEdge,
    TierRegion,
    ViewFeatures,
)
from backend.ai_infra.sketch_models import (
    ArcLineSegment,
    Profile2D,
    SketchOperation,
    SketchPartDescription,
)


NORMALISE_LONGEST_MM = 100.0

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

    longest = max(x_px, y_px, z_px, 1.0)
    mm_per_px = NORMALISE_LONGEST_MM / longest
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
    a depth-tier region)."""
    label, _ = _classify_outline_perimeter_fraction(outline)
    bx, by, bw, bh = bbox_px
    cx_px = bx + bw / 2.0
    cy_px = by + bh / 2.0

    def _to_mm(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] - cx_px) * mm_per_px, -(p[1] - cy_px) * mm_per_px)

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
        return ClassifiedProfile(
            profile=Profile2D(
                shape="arc_line",
                width_mm=bw * mm_per_px,
                depth_mm=bh * mm_per_px,
                arc_line_segments=seg_models,
            ),
            label="arc_line",
        )
    # polyline / empty fallback
    src = smooth_polyline_px or []
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
_STACK_MIN_GAP_MM = 1.0          # tiers thinner than this collapse into the previous one
_STACK_POLYGON_CONTAINMENT_FRAC = 0.85   # tier[i] needs this fraction of vertices inside tier[i-1]


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


def _build_stacked_extrudes(
    view: ViewFeatures, frame: WorldFrame,
) -> tuple[list[SketchOperation], str]:
    """Convert per-tier regions into an ordered list of extrude operations.

    Tier ordering: the segmenter returns regions FAR→NEAR (deepest first).
    For Top view (base on XY, looking down -Z), "far" = lowest Z = bottom
    of part. We build bottom up: deepest tier = XY base extrude; each
    subsequent tier extrudes on the >Z face. Distances come from the
    relative-depth gaps scaled by the part's Z extent.

    Two pre-flight gates protect against bogus stacks:
      1. Drop tiers whose extrude distance < ``_STACK_MIN_GAP_MM`` (those
         are histogram-jitter peaks on a near-flat region).
      2. Each tier (after the first) must be contained within the previous
         tier's bbox (allowing a small slack). If not, the part isn't a
         clean stacked extrude — return empty so the caller falls back to
         a single-extrude path that won't crash CadQuery.
    """
    if not view.tier_regions:
        return [], "no tier regions"

    # Filter tiers by minimum extrude distance.
    raw = view.tier_regions
    kept: list[TierRegion] = []
    prev_rel = 1.0
    for tier in raw:
        gap_mm = (prev_rel - tier.relative_depth) * frame.z_mm
        if kept and gap_mm < _STACK_MIN_GAP_MM:
            continue  # drop this one as a duplicate / near-zero extrude
        kept.append(tier)
        prev_rel = tier.relative_depth
    if not kept:
        return [], "no tiers after gap filter"

    # Polygon-based containment check: each tier i must have most of its
    # vertices INSIDE the previous tier's polygon. bbox-only checks pass
    # for laterally offset tiers (e.g. side-by-side bosses on a bar) that
    # cannot actually stack.
    for i in range(1, len(kept)):
        frac = _polygon_contains_fraction(
            kept[i - 1].polygon_px, kept[i].polygon_px,
        )
        if frac < _STACK_POLYGON_CONTAINMENT_FRAC:
            return [], (
                f"tier {i} only {frac*100:.0f}% inside tier {i-1} polygon — "
                "stacked path doesn't apply, falling back to single extrude"
            )

    base_bx, base_by, base_bw, base_bh = kept[0].bbox_px
    base_cx_px = base_bx + base_bw / 2.0
    base_cy_px = base_by + base_bh / 2.0

    operations: list[SketchOperation] = []
    prev_rel = 1.0
    notes_parts: list[str] = []
    for i, tier in enumerate(kept):
        gap = max(prev_rel - tier.relative_depth, 0.0)
        extrude_mm = max(gap * frame.z_mm, _STACK_MIN_GAP_MM)
        cls = _profile_from_outline_polyline_bbox(
            tier.outline, tier.smooth_polyline_px, tier.bbox_px, frame.mm_per_px,
        )
        bx, by, bw, bh = tier.bbox_px
        cx_px = bx + bw / 2.0
        cy_px = by + bh / 2.0
        if i == 0:
            pos_x_mm = 0.0
            pos_y_mm = 0.0
            plane = "XY"
        else:
            pos_x_mm = (cx_px - base_cx_px) * frame.mm_per_px
            pos_y_mm = -(cy_px - base_cy_px) * frame.mm_per_px
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
        prev_rel = tier.relative_depth

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


def infer_sketches(features: OrthoFeatures) -> SketchPartDescription:
    views = features.views
    frame = _world_frame(views)

    # Pick the base view by silhouette complexity rather than always
    # using Top. This lets parts like 128105 (uniform rectangle from Top
    # but dumbbell silhouette from Front) get constructed from the
    # right axis projection.
    top, base_view_name = _pick_base_view(views)
    spec = _BASE_VIEW_SPECS[base_view_name]
    base_plane = spec["plane"]
    base_axis_attr = spec["axis_attr"]
    base_axis_label = spec["axis_label"]
    base_extrude_axis_mm = max(getattr(frame, base_axis_attr), 0.001)

    operations: list[SketchOperation] = []
    construction_note: str

    # Stacked-extrude path: when the base view has 2+ depth tiers AND
    # the tiers form a valid containment chain, emit one extrude per
    # tier. Captures T-shapes, flange+boss, stepped bars. If the
    # builder rejects the stack (over-segmented or non-contained tiers),
    # we fall through to the single-extrude path so CadQuery doesn't
    # try to extrude on a face that doesn't exist.
    tier_ops: list[SketchOperation] = []
    construction_note = ""
    if len(top.tier_regions) >= 2:
        tier_ops, construction_note = _build_stacked_extrudes(top, frame)
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

    notes = (
        f"deterministic CV inference. Base: {base.notes}. "
        f"Bbox px (X/Y/Z) = {frame.x_px:.0f}/{frame.y_px:.0f}/{frame.z_px:.0f}, "
        f"mm/px = {frame.mm_per_px:.3f}. "
        f"D-cuts: {len(base.flat_cuts)}. "
        f"Through-holes Z/Y/X: {z_holes_count}/{y_holes_count}/{x_holes_count}. "
        f"Side-face: {', '.join(side_notes) if side_notes else 'none'}."
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
