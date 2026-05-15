"""Step-off feature extraction for the ortho-view CAD pipeline.

A "step-off" is a discrete depth transition between two adjacent tier regions
(stacked extrudes, counterbores, pockets) or a through-hole that pierces a
silhouette from one side to the other. Each StepOff captures the outer/inner
polygons, the bbox containment confidence, the depth delta in mm, and a
classification (external/internal/through) so downstream agents can either
audit the segmenter or build a SketchPartDescription deterministically.

This module is purposely additive: it consumes ``OrthoFeatures`` and
``WorldFrame`` produced by the existing inferencer without mutating them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TYPE_CHECKING

from backend.ai_infra.ortho_feature_inferencer import (
    _STACK_BBOX_CONTAINMENT_FRAC,
    _bbox_contains_fraction,
    _match_holes,
)
from backend.ai_infra.step_off_classifier import (
    polygon_strictly_inside,
    polygon_touches_silhouette,
)

if TYPE_CHECKING:
    from backend.ai_infra.ortho_feature_inferencer import WorldFrame
    from backend.ai_infra.ortho_view_segmenter import OrthoFeatures, TierRegion


_VIEW_TO_AXIS: dict[str, Literal["Z", "Y", "X"]] = {
    "Top": "Z",
    "Bottom": "Z",
    "Front": "Y",
    "Back": "Y",
    "Right": "X",
    "Left": "X",
}

_VIEW_ORDER: tuple[str, ...] = ("Top", "Bottom", "Front", "Back", "Right", "Left")


@dataclass
class StepOff:
    """One discrete depth transition between two tier regions on a single
    ortho view, OR a through-hole that pierces the silhouette end-to-end.

    Coordinates are in panel pixels (the segmenter's native frame).
    ``depth_mm`` is computed by multiplying the |relative_depth| delta by the
    relevant world-axis span in mm.
    """

    id: str
    view: str
    axis: Literal["Z", "Y", "X"]
    outer_polygon_px: list[tuple[float, float]]
    inner_polygon_px: list[tuple[float, float]]
    outer_bbox_px: tuple[int, int, int, int]
    inner_bbox_px: tuple[int, int, int, int]
    step_direction: Literal["up", "down"]
    depth_mm: float
    kind: Literal["external", "internal", "through"]
    confidence: float
    notes: str


_FIELD_ORDER: tuple[str, ...] = (
    "id",
    "view",
    "axis",
    "kind",
    "step_direction",
    "depth_mm",
    "confidence",
    "outer_bbox_px",
    "inner_bbox_px",
    "outer_polygon_px",
    "inner_polygon_px",
    "notes",
)


def step_off_to_dict(s: StepOff) -> dict:
    """Convert a StepOff to a dict with a deterministic field order so audit
    JSON files diff cleanly between runs."""
    raw = asdict(s)
    return {k: raw[k] for k in _FIELD_ORDER}


def _axis_mm(axis: Literal["Z", "Y", "X"], frame: "WorldFrame") -> float:
    if axis == "Z":
        return float(frame.z_mm)
    if axis == "Y":
        return float(frame.y_mm)
    return float(frame.x_mm)


def _polygon_as_float(poly: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in poly]


def _emit_tier_pairs(
    name: str,
    axis: Literal["Z", "Y", "X"],
    tiers: list["TierRegion"],
    silhouette_bbox: tuple[int, int, int, int],
    frame: "WorldFrame",
) -> list[StepOff]:
    """Pair each (outer, inner) tier by bbox area where the inner bbox is
    >=_STACK_BBOX_CONTAINMENT_FRAC contained inside the outer bbox. Emits at
    most one StepOff per unordered pair (i, j)."""
    out: list[StepOff] = []
    n = len(tiers)
    if n < 2:
        return out

    axis_mm = _axis_mm(axis, frame)

    for a in range(n):
        for b in range(a + 1, n):
            ta = tiers[a]
            tb = tiers[b]
            area_a = float(ta.bbox_px[2] * ta.bbox_px[3])
            area_b = float(tb.bbox_px[2] * tb.bbox_px[3])
            if area_a <= 0 or area_b <= 0:
                continue
            if area_a >= area_b:
                outer_idx, inner_idx = a, b
                outer, inner = ta, tb
            else:
                outer_idx, inner_idx = b, a
                outer, inner = tb, ta

            frac = _bbox_contains_fraction(outer.bbox_px, inner.bbox_px)
            if frac < _STACK_BBOX_CONTAINMENT_FRAC:
                continue

            step_direction: Literal["up", "down"] = (
                "up" if inner.relative_depth < outer.relative_depth else "down"
            )

            if polygon_touches_silhouette(
                _polygon_as_float(inner.polygon_px), silhouette_bbox, tol_px=3.0
            ):
                kind: Literal["external", "internal", "through"] = "external"
            elif polygon_strictly_inside(
                _polygon_as_float(inner.polygon_px),
                _polygon_as_float(outer.polygon_px),
            ):
                kind = "internal"
            else:
                # Degenerate: neither external (touches silhouette) nor cleanly
                # inside the outer region. Skip so the audit JSON stays clean.
                continue

            depth_mm = abs(inner.relative_depth - outer.relative_depth) * axis_mm

            out.append(
                StepOff(
                    id=f"step:{name}:{outer_idx}->{inner_idx}",
                    view=name,
                    axis=axis,
                    outer_polygon_px=_polygon_as_float(outer.polygon_px),
                    inner_polygon_px=_polygon_as_float(inner.polygon_px),
                    outer_bbox_px=tuple(int(x) for x in outer.bbox_px),  # type: ignore[arg-type]
                    inner_bbox_px=tuple(int(x) for x in inner.bbox_px),  # type: ignore[arg-type]
                    step_direction=step_direction,
                    depth_mm=float(depth_mm),
                    kind=kind,
                    confidence=float(frac),
                    notes=(
                        f"tier_pair view={name} "
                        f"outer_rel={outer.relative_depth:.3f} "
                        f"inner_rel={inner.relative_depth:.3f}"
                    ),
                )
            )
    return out


def _emit_through_holes(
    features: "OrthoFeatures", frame: "WorldFrame"
) -> list[StepOff]:
    """Match Top<->Bottom interior holes via the existing ``_match_holes``
    helper and emit one StepOff per matched through-hole. Z axis only in this
    initial pass; the inner bbox is reconstructed from ``diameter_mm`` and
    ``centre_mm`` so a human auditor can sanity-check the back-conversion."""
    view_top = features.views.get("Top")
    view_bot = features.views.get("Bottom")
    if view_top is None or view_bot is None:
        return []
    if view_top.bbox_px == (0, 0, 0, 0) or view_bot.bbox_px == (0, 0, 0, 0):
        return []

    holes = _match_holes(view_top, view_bot, "Z", frame)
    if not holes:
        return []

    mm_per_px = frame.mm_per_px
    if mm_per_px <= 0:
        return []

    bx, by, bw, bh = view_top.bbox_px
    cx_px = bx + bw / 2.0
    cy_px = by + bh / 2.0

    out: list[StepOff] = []
    for i, hole in enumerate(holes):
        # Back-convert centre_mm into pixel space. centre_mm uses the
        # inferencer's CAD convention (+y up), so invert y when reading back
        # into the segmenter's top-left pixel frame.
        cx_mm, cy_mm = hole.centre_mm
        px = cx_px + (cx_mm / mm_per_px)
        py = cy_px - (cy_mm / mm_per_px)
        d_px = hole.diameter_mm / mm_per_px
        inner_bbox = (
            int(round(px - d_px / 2.0)),
            int(round(py - d_px / 2.0)),
            int(round(d_px)),
            int(round(d_px)),
        )
        out.append(
            StepOff(
                id=f"step:Top:through:{i}",
                view="Top",
                axis="Z",
                outer_polygon_px=_polygon_as_float(view_top.polygon_px),
                inner_polygon_px=[],
                outer_bbox_px=tuple(int(x) for x in view_top.bbox_px),  # type: ignore[arg-type]
                inner_bbox_px=inner_bbox,
                step_direction="down",
                depth_mm=float(frame.z_mm),
                kind="through",
                confidence=1.0,
                notes="through-hole match Top<->Bottom",
            )
        )
    return out


def extract_step_offs(
    features: "OrthoFeatures", frame: "WorldFrame"
) -> list[StepOff]:
    """Walk every populated view, emit a StepOff per qualifying tier-pair, and
    append through-hole StepOffs from the Top<->Bottom match. Pure function:
    does not mutate ``features`` or ``frame``."""
    out: list[StepOff] = []
    for name in _VIEW_ORDER:
        view = features.views.get(name)
        if view is None:
            continue
        if view.bbox_px == (0, 0, 0, 0):
            continue
        tiers = list(view.tier_regions or [])
        if not tiers:
            continue
        axis = _VIEW_TO_AXIS[name]
        out.extend(_emit_tier_pairs(name, axis, tiers, view.bbox_px, frame))

    out.extend(_emit_through_holes(features, frame))
    return out
