"""Pydantic v2 models for the sketch-plane architecture.

This is the *replacement* for ``backend.ai_infra.models`` on the
``sketch-plane-architecture`` branch. The existing ``BaseBody +
features`` schema is deliberately untouched so the two architectures
can be A/B tested.

Core idea: a part is fully described by an ORDERED LIST of 2D-sketch +
extrude/cut/revolve operations. The CadQuery builder applies them in
order. There is no privileged "base body" — the first operation just
happens to be the largest.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


ProfileShape = Literal[
    # Clean primitives — described compactly by bbox + (optional) diameter.
    "rectangle", "circle",
    # Legacy enum hints that the LLM may still emit. Builder treats any
    # of these as a fallback rectangle UNLESS ``vertices`` is set, in
    # which case it routes to the polyline path. Kept for backward
    # compatibility with earlier prompts.
    "polygon", "l_shape", "t_shape",
    # The general case: an arbitrary closed 2D polygon defined by
    # explicit vertex coordinates. ``vertices`` is REQUIRED and is in
    # world-mm relative to the sketch-plane centre (workplane origin
    # after CenterOfBoundBox), with +x along the plane's horizontal
    # axis and +y along its vertical axis. Use this when the profile
    # isn't a clean rectangle/circle.
    "polyline",
    # Hybrid outline made of straight segments AND circular arcs in a
    # single closed loop. ``arc_line_segments`` is REQUIRED. This is the
    # right primitive for D-cuts (arc + line + arc + line), rounded
    # rectangles, racetracks / obrounds, and any silhouette that mixes
    # straight edges with curved sections without being a single primitive.
    "arc_line",
]
SketchPlane = Literal[
    # Absolute world planes (only valid for the FIRST operation).
    "XY", "XZ", "YZ",
    # Face selectors for subsequent operations. These are CadQuery
    # selector strings the builder forwards to ``.faces(...)``.
    ">Z", "<Z", ">X", "<X", ">Y", "<Y",
]
OperationType = Literal["extrude", "cut", "revolve"]
Direction = Literal["positive", "negative"]
Confidence = Literal["high", "medium", "low"]


def _check_positive(v: Optional[float], field: str) -> Optional[float]:
    if v is not None and v <= 0:
        raise ValueError(f"{field} must be > 0 mm when provided")
    return v


# ---------------------------------------------------------------------------
# ArcLineSegment — one piece of a hybrid arc+line closed outline
# ---------------------------------------------------------------------------
class ArcLineSegment(BaseModel):
    """One segment of an arc_line profile.

    All coordinates are in world mm relative to the sketch-plane centre
    (workplane origin after CenterOfBoundBox), matching the convention
    used by ``polyline`` profiles. The closed loop is implied by the
    order of segments: each segment's ``end`` should equal the next
    segment's ``start`` (within builder tolerance).

    For ``kind == "line"``: ``start`` and ``end`` define the segment;
    ``arc_centre`` / ``arc_radius_mm`` are ignored.

    For ``kind == "arc"``: ``start``, ``end``, ``arc_centre`` and
    ``arc_radius_mm`` together define the arc. Builder emits
    ``.threePointArc((midpoint), (end))`` where ``midpoint`` is computed
    on the arc.
    """

    kind: Literal["line", "arc"]
    start: Tuple[float, float]
    end: Tuple[float, float]
    arc_centre: Optional[Tuple[float, float]] = None
    arc_radius_mm: Optional[float] = None
    # Sweep convention: positive = CCW (counter-clockwise) around the
    # centre. Builder uses this to pick the correct mid-arc point.
    arc_ccw: Optional[bool] = None

    @model_validator(mode="after")
    def _check_arc_fields(self) -> "ArcLineSegment":
        if self.kind == "arc":
            if self.arc_centre is None or self.arc_radius_mm is None:
                raise ValueError(
                    "arc segment requires arc_centre and arc_radius_mm"
                )
            if self.arc_radius_mm <= 0:
                raise ValueError("arc_radius_mm must be > 0")
        return self


# ---------------------------------------------------------------------------
# Profile2D — what gets drawn on a sketch plane
# ---------------------------------------------------------------------------
class Profile2D(BaseModel):
    """The 2D shape drawn on a sketch plane.

    Two ways to describe a profile:

    * **Clean primitives** (``shape="rectangle"`` or ``"circle"``)
      — fully described by ``width_mm`` / ``depth_mm`` (and
      ``diameter_mm`` for circles). Use these whenever the outline
      really is a rect or a circle, because the resulting CadQuery
      code is shorter and editable by humans.
    * **Polyline** (``shape="polyline"``) — a general n-gon defined
      by an explicit ``vertices`` list in **world mm** relative to
      the sketch-plane centre (the workplane's
      ``CenterOfBoundBox`` origin). Vertices are listed in order
      around the outline and the builder closes the loop
      automatically. Use this for L/T/U/wedge profiles or anything
      else that isn't a clean primitive — the OpenCV contour
      extractor produces these directly.

    ``width_mm`` and ``depth_mm`` are still required for ``polyline``
    (they describe the *bounding box* of the vertex list) so
    downstream code can size scale-bars and run sanity checks
    without re-walking the vertex array.
    """

    shape: ProfileShape
    width_mm: float = Field(..., description="Bounding box width (X span on the sketch plane).")
    depth_mm: float = Field(..., description="Bounding box depth (Y span on the sketch plane).")
    vertices: Optional[list[Tuple[float, float]]] = Field(
        None,
        description="For shape='polyline' (REQUIRED): closed-polygon vertex "
        "list in world mm, relative to the sketch-plane centre, in order "
        "around the outline. Builder calls .polyline(verts).close() so do "
        "not repeat the first vertex at the end. For 'polygon' (legacy): "
        "may be normalised 0-1 inside the bbox; if present, builder treats "
        "it as polyline. Ignored for clean primitives.",
    )
    diameter_mm: Optional[float] = Field(
        None,
        description="Diameter for shape='circle'. If omitted, builder uses "
        "min(width_mm, depth_mm).",
    )
    arc_line_segments: Optional[list[ArcLineSegment]] = Field(
        None,
        description="For shape='arc_line' (REQUIRED): ordered list of line "
        "and arc segments forming a closed loop. Each segment's end should "
        "equal the next segment's start; builder closes the loop after the "
        "final segment.",
    )

    @field_validator("width_mm", "depth_mm", mode="after")
    @classmethod
    def _wd_positive(cls, v: float) -> float:
        return _check_positive(v, "width_mm/depth_mm")  # type: ignore[return-value]

    @field_validator("diameter_mm", mode="after")
    @classmethod
    def _dia_positive(cls, v: Optional[float]) -> Optional[float]:
        return _check_positive(v, "diameter_mm")

    @model_validator(mode="after")
    def _polyline_needs_vertices(self) -> "Profile2D":
        if self.shape == "polyline":
            if not self.vertices or len(self.vertices) < 3:
                raise ValueError(
                    "shape='polyline' requires at least 3 vertices in mm"
                )
        if self.shape == "arc_line":
            if not self.arc_line_segments or len(self.arc_line_segments) < 2:
                raise ValueError(
                    "shape='arc_line' requires at least 2 arc_line_segments"
                )
        return self


# ---------------------------------------------------------------------------
# SketchOperation — one entry in the construction sequence
# ---------------------------------------------------------------------------
class SketchOperation(BaseModel):
    """One sketch + extrude/cut/revolve operation.

    The whole part is just a list of these, executed in ascending
    ``order``. No state is implicit between operations beyond the
    accumulated CadQuery solid.
    """

    order: int = Field(
        ...,
        ge=1,
        description="1-based construction order. Sequential, no gaps.",
    )
    plane: SketchPlane = Field(
        ...,
        description='"XY"/"XZ"/"YZ" for the first op; ">Z" / "<Z" / ">X" '
        '/ "<X" / ">Y" / "<Y" face selector for subsequent ops.',
    )
    profile: Profile2D
    operation: OperationType = Field(
        ...,
        description='"extrude" adds material, "cut" removes it, "revolve" '
        "spins the profile (revolve is reserved; builder MVP does not "
        "implement it yet).",
    )
    distance_mm: float = Field(
        ...,
        description="How far to extrude or cut along the plane normal, in mm.",
    )
    direction: Direction = Field(
        "positive",
        description='"positive" extrudes/cuts along the plane normal; '
        '"negative" goes the other way (into the existing body).',
    )
    position_x: float = Field(
        0.0,
        description="Sketch centre offset from plane centre, X axis (mm).",
    )
    position_y: float = Field(
        0.0,
        description="Sketch centre offset from plane centre, Y axis (mm).",
    )

    @field_validator("distance_mm", mode="after")
    @classmethod
    def _dist_positive(cls, v: float) -> float:
        return _check_positive(v, "distance_mm")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SketchPartDescription — the whole part
# ---------------------------------------------------------------------------
class SketchPartDescription(BaseModel):
    """A part as an ordered list of sketch operations + bounding box."""

    sketches: list[SketchOperation] = Field(
        ...,
        description="Ordered construction sequence. Must contain at least "
        "one operation. Order field must be sequential 1, 2, 3, ...",
    )
    bounding_box_mm: Tuple[float, float, float] = Field(
        ...,
        description="Overall (width X, depth Y, height Z) of the finished "
        "part in mm. Longest dimension should be ~100 mm because the "
        "renderer normalises inputs to that scale.",
    )
    confidence: Confidence
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_order(self) -> "SketchPartDescription":
        if not self.sketches:
            raise ValueError("at least one sketch operation is required")
        # Sort by order so downstream consumers don't have to.
        self.sketches.sort(key=lambda s: s.order)
        # Sequentiality check (1, 2, 3, ...): catches off-by-one or
        # duplicate orders coming back from Claude.
        expected = list(range(1, len(self.sketches) + 1))
        actual = [s.order for s in self.sketches]
        if actual != expected:
            raise ValueError(
                f"sketch order numbers must be sequential starting at 1; "
                f"got {actual}, expected {expected}"
            )
        # The first op MUST start in an absolute world plane; subsequent
        # ops should reference an existing face.
        first = self.sketches[0]
        if first.plane not in {"XY", "XZ", "YZ"}:
            raise ValueError(
                f"first sketch operation must use an absolute plane "
                f'(XY/XZ/YZ); got "{first.plane}"'
            )
        return self
