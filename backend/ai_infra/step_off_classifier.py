"""Classifier helpers for step-off features.

Two pure-geometry predicates plus two higher-level helpers (axis picker and
SketchOperation assembler) used by later refactor agents. The geometry
predicates are intentionally small and dependency-light so they can be unit
tested in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from backend.ai_infra.sketch_models import Profile2D, SketchOperation

if TYPE_CHECKING:
    from backend.ai_infra.ortho_feature_inferencer import WorldFrame
    from backend.ai_infra.step_offs import StepOff


def polygon_touches_silhouette(
    poly: list[tuple[float, float]],
    silhouette_bbox: tuple[int, int, int, int],
    tol_px: float = 3.0,
) -> bool:
    """True if any vertex of ``poly`` is within ``tol_px`` of the silhouette
    bbox boundary."""
    if not poly:
        return False
    bx, by, bw, bh = silhouette_bbox
    if bw <= 0 or bh <= 0:
        return False
    x0, y0 = float(bx), float(by)
    x1, y1 = float(bx + bw), float(by + bh)
    for vx, vy in poly:
        if (
            abs(vx - x0) <= tol_px
            or abs(vx - x1) <= tol_px
            or abs(vy - y0) <= tol_px
            or abs(vy - y1) <= tol_px
        ):
            return True
    return False


def polygon_strictly_inside(
    poly: list[tuple[float, float]],
    outer_poly: list[tuple[float, float]],
) -> bool:
    """True if ALL vertices of ``poly`` are strictly inside ``outer_poly``
    (``cv2.pointPolygonTest`` returns +1, not 0). Returns False if
    ``outer_poly`` has fewer than 3 vertices."""
    if len(outer_poly) < 3 or not poly:
        return False
    outer_np = np.array(outer_poly, dtype=np.float32)
    for vx, vy in poly:
        if cv2.pointPolygonTest(outer_np, (float(vx), float(vy)), False) <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Axis selection (used by Agent 1.2)
# ---------------------------------------------------------------------------
_AXIS_VIEWS: dict[str, tuple[str, str]] = {
    "Z": ("Top", "Bottom"),
    "Y": ("Front", "Back"),
    "X": ("Right", "Left"),
}


_AXIS_TO_PERP_VIEW: dict[str, str] = {"Z": "Top", "Y": "Front", "X": "Right"}


def pick_axis_from_step_offs(
    step_offs: list["StepOff"],
    features: "OrthoFeatures | None" = None,  # noqa: F821
) -> Literal["Z", "Y", "X"]:
    """Pick the slicing axis from external step-offs and perpendicular-view
    evidence.

    The raw tally is unreliable on its own: side views of a Z-stacked staircase
    expose X- or Y-direction tier transitions too (the staircase profile
    projected into XZ creates X-depth tiers in the Right view) and can
    out-vote the true stack axis. So we combine the tally with the
    perpendicular base-view's ``depth_tier_count`` — Top reveals Z-tiers,
    Front reveals Y-tiers, Right reveals X-tiers.

    Scoring per axis::

        score = (perp_evidence[axis] >= 2) * 1000 + tally[axis]

    The 1000 bias makes perpendicular evidence dominate while the tally still
    breaks ties when multiple axes have perpendicular evidence. Final tiebreak
    is the CAD convention Z > Y > X. Returns ``"Z"`` when all scores are zero
    (flat-extrude / through-only parts default to Z-stacked).
    """
    tallies = {"Z": 0, "Y": 0, "X": 0}
    for s in step_offs:
        if s.kind != "external":
            continue
        if s.axis in tallies:
            tallies[s.axis] += 1

    perp_evidence: dict[str, int] = {"Z": 0, "Y": 0, "X": 0}
    if features is not None:
        for axis, view_name in _AXIS_TO_PERP_VIEW.items():
            try:
                view = features.views[view_name]
            except (KeyError, AttributeError):
                continue
            count = getattr(view, "depth_tier_count", 0) or 0
            perp_evidence[axis] = int(count)
    else:
        # Fall back to distinct outer-tier relative-depths from external step-offs.
        seen: dict[str, set[float]] = {"Z": set(), "Y": set(), "X": set()}
        for s in step_offs:
            if s.kind != "external" or s.axis not in seen:
                continue
            for token in s.notes.split():
                if token.startswith("outer_rel="):
                    try:
                        seen[s.axis].add(round(float(token.split("=", 1)[1]), 3))
                    except ValueError:
                        pass
        for axis in perp_evidence:
            # Each external step-off adds a pair of tier depths, so a single
            # transition implies two distinct tiers.
            perp_evidence[axis] = len(seen[axis]) + (
                1 if seen[axis] and tallies[axis] >= 1 else 0
            )

    scores = {
        axis: (1000 if perp_evidence[axis] >= 2 else 0) + tallies[axis]
        for axis in ("Z", "Y", "X")
    }
    max_score = max(scores.values())
    if max_score == 0:
        return "Z"
    # Mirror _pick_slicing_axis: when multiple axes have perpendicular
    # evidence (depth_tier_count >= 2), CAD priority Z > Y > X dominates
    # over the tally (which double-counts side-projection staircases).
    # The tally only breaks ties when perpendicular evidence is absent.
    evidenced = [a for a in ("Z", "Y", "X") if perp_evidence[a] >= 2]
    if len(evidenced) >= 2:
        return evidenced[0]  # type: ignore[return-value]
    if len(evidenced) == 1:
        return evidenced[0]  # type: ignore[return-value]
    # No perpendicular evidence: fall back to tally with Z > Y > X tiebreak.
    for axis in ("Z", "Y", "X"):
        if scores[axis] == max_score:
            return axis  # type: ignore[return-value]
    return "Z"


# ---------------------------------------------------------------------------
# Op assembler (used by Agent 1.3 — basic implementation only)
# ---------------------------------------------------------------------------
_AXIS_TO_FIRST_PLANE: dict[str, str] = {"Z": "XY", "Y": "XZ", "X": "YZ"}
_AXIS_TO_FACE_SELECTOR: dict[str, str] = {"Z": ">Z", "Y": ">Y", "X": ">X"}


def _bbox_centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    bx, by, bw, bh = bbox
    return (bx + bw / 2.0, by + bh / 2.0)


def _profile_from_outer_bbox_polygon(
    polygon: list[tuple[float, float]],
    bbox: tuple[int, int, int, int],
    mm_per_px: float,
) -> Profile2D:
    """Minimal profile classifier for an outer (base) outline. Tries
    rectangle first, then circle (via min-enclosing-circle bbox-area check),
    then falls back to polyline."""
    bx, by, bw, bh = bbox
    width_mm = max(bw * mm_per_px, 0.001)
    depth_mm = max(bh * mm_per_px, 0.001)

    # Rectangle: 4 axis-aligned vertices with bbox-coverage ~ 1.
    if len(polygon) == 4:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_uniques = sorted({round(x) for x in xs})
        y_uniques = sorted({round(y) for y in ys})
        if len(x_uniques) == 2 and len(y_uniques) == 2:
            return Profile2D(
                shape="rectangle", width_mm=width_mm, depth_mm=depth_mm
            )

    if polygon:
        contour = np.array(polygon, dtype=np.float32)
        try:
            (_cx, _cy), r = cv2.minEnclosingCircle(contour)
            circle_area = float(np.pi * r * r)
            bbox_area = float(bw * bh)
            if bbox_area > 0 and 0.85 <= circle_area / bbox_area <= 1.18:
                d_mm = max(2.0 * r * mm_per_px, 0.001)
                return Profile2D(
                    shape="circle",
                    width_mm=d_mm,
                    depth_mm=d_mm,
                    diameter_mm=d_mm,
                )
        except cv2.error:
            pass

    cx_px, cy_px = _bbox_centroid(bbox)
    verts_mm = [
        ((px - cx_px) * mm_per_px, -(py - cy_px) * mm_per_px)
        for px, py in polygon
    ]
    if len(verts_mm) < 3:
        return Profile2D(shape="rectangle", width_mm=width_mm, depth_mm=depth_mm)
    return Profile2D(
        shape="polyline",
        width_mm=width_mm,
        depth_mm=depth_mm,
        vertices=verts_mm,
    )


def _profile_circle_from_inner_bbox(
    bbox: tuple[int, int, int, int], mm_per_px: float
) -> Profile2D:
    bw, bh = bbox[2], bbox[3]
    d_px = float(min(bw, bh)) if min(bw, bh) > 0 else float(max(bw, bh, 1))
    d_mm = max(d_px * mm_per_px, 0.001)
    return Profile2D(
        shape="circle", width_mm=d_mm, depth_mm=d_mm, diameter_mm=d_mm
    )


def _inner_offset_mm(
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
    mm_per_px: float,
) -> tuple[float, float]:
    icx, icy = _bbox_centroid(inner_bbox)
    ocx, ocy = _bbox_centroid(outer_bbox)
    return ((icx - ocx) * mm_per_px, -(icy - ocy) * mm_per_px)


def assemble_ops_from_step_offs(
    step_offs: list["StepOff"],
    axis: str,
    frame: "WorldFrame",
) -> list[SketchOperation]:
    """Basic mechanical dispatch on (kind, step_direction). The first emitted
    op uses an absolute plane (XY/XZ/YZ); subsequent ops reference the live
    top face for the slicing axis. Through-cuts always use the Z+ face
    selector since the seed-tag rewrite inside ``sketch_builder`` handles the
    cut-relative selection."""
    mm_per_px = float(frame.mm_per_px)

    target_axis_step_offs = [
        s for s in step_offs if s.axis == axis or s.kind == "through"
    ]

    externals = [s for s in target_axis_step_offs if s.kind == "external"]
    internals = [s for s in target_axis_step_offs if s.kind == "internal"]
    throughs = [s for s in target_axis_step_offs if s.kind == "through"]

    # Deepest-first: largest relative_depth = base (farthest from camera) =
    # the bottom of the stack. Subsequent (external, up) extrudes stack
    # toward the camera. ``notes`` carries the inner relative depth.
    def _inner_rel(s) -> float:
        for token in s.notes.split():
            if token.startswith("inner_rel="):
                try:
                    return float(token.split("=", 1)[1])
                except ValueError:
                    return 0.0
        return 0.0

    externals.sort(key=_inner_rel, reverse=True)

    ops: list[SketchOperation] = []
    order = 1
    first_plane = _AXIS_TO_FIRST_PLANE.get(axis, "XY")
    face_selector = _AXIS_TO_FACE_SELECTOR.get(axis, ">Z")
    saw_base = False

    for s in externals:
        if s.step_direction == "up":
            if not saw_base:
                profile = _profile_from_outer_bbox_polygon(
                    s.outer_polygon_px, s.outer_bbox_px, mm_per_px
                )
                ops.append(
                    SketchOperation(
                        order=order,
                        plane=first_plane,  # type: ignore[arg-type]
                        profile=profile,
                        operation="extrude",
                        distance_mm=max(s.depth_mm, 0.001),
                        direction="positive",
                        position_x=0.0,
                        position_y=0.0,
                    )
                )
                order += 1
                saw_base = True
            else:
                profile = _profile_from_outer_bbox_polygon(
                    s.inner_polygon_px, s.inner_bbox_px, mm_per_px
                )
                pos_x, pos_y = _inner_offset_mm(
                    s.inner_bbox_px, s.outer_bbox_px, mm_per_px
                )
                ops.append(
                    SketchOperation(
                        order=order,
                        plane=face_selector,  # type: ignore[arg-type]
                        profile=profile,
                        operation="extrude",
                        distance_mm=max(s.depth_mm, 0.001),
                        direction="positive",
                        position_x=pos_x,
                        position_y=pos_y,
                    )
                )
                order += 1
        else:  # external, down: blind recess pocket
            profile = _profile_from_outer_bbox_polygon(
                s.inner_polygon_px, s.inner_bbox_px, mm_per_px
            )
            pos_x, pos_y = _inner_offset_mm(
                s.inner_bbox_px, s.outer_bbox_px, mm_per_px
            )
            ops.append(
                SketchOperation(
                    order=max(order, 2),
                    plane=face_selector,  # type: ignore[arg-type]
                    profile=profile,
                    operation="cut",
                    distance_mm=max(s.depth_mm, 0.001),
                    direction="negative",
                    position_x=pos_x,
                    position_y=pos_y,
                )
            )
            order = max(order, 2) + 1

    # If we never saw an external base, synthesise one from the largest
    # outer polygon we have so the resulting SketchPartDescription still has
    # a valid first absolute-plane extrude. (Through-only samples like
    # 128105 land here.)
    if not saw_base:
        base_source = throughs[0] if throughs else None
        if base_source is None and target_axis_step_offs:
            base_source = target_axis_step_offs[0]
        if base_source is not None:
            profile = _profile_from_outer_bbox_polygon(
                base_source.outer_polygon_px,
                base_source.outer_bbox_px,
                mm_per_px,
            )
            ops.append(
                SketchOperation(
                    order=1,
                    plane=first_plane,  # type: ignore[arg-type]
                    profile=profile,
                    operation="extrude",
                    distance_mm=max(float(frame.z_mm), 0.001),
                    direction="positive",
                    position_x=0.0,
                    position_y=0.0,
                )
            )
            order = 2
            saw_base = True

    for s in internals:
        profile = _profile_circle_from_inner_bbox(s.inner_bbox_px, mm_per_px)
        pos_x, pos_y = _inner_offset_mm(
            s.inner_bbox_px, s.outer_bbox_px, mm_per_px
        )
        ops.append(
            SketchOperation(
                order=order,
                plane=face_selector,  # type: ignore[arg-type]
                profile=profile,
                operation="cut",
                distance_mm=max(s.depth_mm, 0.001),
                direction="negative",
                position_x=pos_x,
                position_y=pos_y,
            )
        )
        order += 1

    for s in throughs:
        profile = _profile_circle_from_inner_bbox(s.inner_bbox_px, mm_per_px)
        pos_x, pos_y = _inner_offset_mm(
            s.inner_bbox_px, s.outer_bbox_px, mm_per_px
        )
        ops.append(
            SketchOperation(
                order=order,
                plane=">Z",
                profile=profile,
                operation="cut",
                distance_mm=max(float(frame.z_mm) + 1.0, 0.001),
                direction="negative",
                position_x=pos_x,
                position_y=pos_y,
            )
        )
        order += 1

    return ops
