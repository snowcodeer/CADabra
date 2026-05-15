"""Translate a ``SketchPartDescription`` into runnable CadQuery code.

This is the sketch-plane-architecture counterpart of
``backend.ai_infra.cadquery_builder``. It is intentionally a *pure*
code generator: ``build_from_sketches`` returns Python source text and
never imports ``cadquery`` itself, so it can be unit-tested without an
OCCT install.

The resulting script:
  * starts with ``import cadquery as cq``
  * always leaves the assembled solid in a variable named ``result``
  * applies the sketch operations in ascending ``order``
  * uses one ``result = ...`` assignment per operation so a CadQuery
    traceback points at the offending step
"""

from __future__ import annotations

from backend.ai_infra.sketch_models import (
    Profile2D,
    SketchOperation,
    SketchPartDescription,
)


# CadQuery face selectors. The sketch model's `plane` field already
# uses these literals for non-first ops, so the mapping is just the
# identity — but keeping it explicit makes invalid values easy to spot.
FACE_SELECTORS = {">Z", "<Z", ">X", "<X", ">Y", "<Y"}
ABSOLUTE_PLANES = {"XY", "XZ", "YZ"}

# Tag names applied to the seed solid's six cardinal faces. Subsequent
# operations look up the face by tag instead of using the live
# ``.faces(">Z")`` selector. Without this, two pillars extruded in
# sequence on the same face land on TOP of each other (the second
# extrude sees the new ``>Z`` face, which is the top of the first
# pillar) instead of side-by-side on the original base. We hit and
# fixed the same bug class on the BaseBody branch — the rationale
# carries over verbatim.
SEED_FACE_TAGS: dict[str, str] = {
    ">Z": "seed+Z", "<Z": "seed-Z",
    ">X": "seed+X", "<X": "seed-X",
    ">Y": "seed+Y", "<Y": "seed-Y",
}

# Tagging chain emitted right after the seed extrude. Each tag is
# followed by ``.end()`` so the previous selection state does not
# carry over and accidentally tag the same face six times.
SEED_FACE_TAG_LINES: list[str] = [
    "result = (",
    "    result",
    *[
        f'    .faces("{sel}").tag("{tag}").end()'
        for sel, tag in SEED_FACE_TAGS.items()
    ],
    ")",
]


def _fmt(v: float) -> str:
    """Render a float without trailing-zero noise."""
    return f"{float(v)}"


def _polyline_call(vertices: list[tuple[float, float]]) -> str:
    """Emit '.polyline([(x, y), ...]).close()' for a vertex list in mm.

    The builder leaves it up to the caller to make sure the workplane
    is positioned correctly (CenterOfBoundBox + .center(...)) before
    this is appended; vertices themselves are interpreted as
    workplane-local mm.
    """
    pts = ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in vertices)
    return f".polyline([{pts}]).close()"


def _arc_midpoint(
    start: tuple[float, float], end: tuple[float, float],
    centre: tuple[float, float], radius: float, ccw: bool | None,
) -> tuple[float, float]:
    """Pick a midpoint on the arc through start and end with the given
    centre and radius. CadQuery's threePointArc takes a midpoint, not a
    centre; we sample the arc at its midway angle."""
    import math

    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    # Pick sweep direction. If ccw is set use it; otherwise pick the
    # shorter arc.
    if ccw is True:
        if a1 < a0:
            a1 += 2 * math.pi
    elif ccw is False:
        if a1 > a0:
            a1 -= 2 * math.pi
    else:
        # Shorter arc by default.
        diff = a1 - a0
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        a1 = a0 + diff
    am = (a0 + a1) / 2.0
    return (centre[0] + radius * math.cos(am),
            centre[1] + radius * math.sin(am))


def _arc_line_call(segments: list) -> str:
    """Emit '.moveTo(...).threePointArc(...).lineTo(...).close()' for a
    closed loop of arc and line segments. Vertices are workplane-local mm.
    """
    if not segments:
        return ".rect(1, 1)"  # safety; validator should prevent this
    first = segments[0]
    parts: list[str] = [f".moveTo({_fmt(first.start[0])}, {_fmt(first.start[1])})"]
    for seg in segments:
        if seg.kind == "line":
            parts.append(f".lineTo({_fmt(seg.end[0])}, {_fmt(seg.end[1])})")
        else:
            mid = _arc_midpoint(
                seg.start, seg.end, seg.arc_centre, seg.arc_radius_mm, seg.arc_ccw,
            )
            parts.append(
                f".threePointArc(({_fmt(mid[0])}, {_fmt(mid[1])}), "
                f"({_fmt(seg.end[0])}, {_fmt(seg.end[1])}))"
            )
    parts.append(".close()")
    return "".join(parts)


def _profile_2d_call(profile: Profile2D) -> str:
    """Return the CadQuery sketch call for a 2D profile.

    Routing rules:

    * ``rectangle``                 -> ``.rect(w, d)``
    * ``circle``                    -> ``.circle(d/2)``
    * ``polyline``                  -> ``.polyline(verts).close()``
    * ``polygon`` / ``l_shape`` / ``t_shape`` with vertices set -> same
      as polyline (legacy enums get the upgrade for free)
    * ``polygon`` / ``l_shape`` / ``t_shape`` WITHOUT vertices -> bbox
      rectangle fallback so the builder always returns valid CadQuery
    """
    w = profile.width_mm
    d = profile.depth_mm

    if profile.shape == "rectangle":
        return f".rect({_fmt(w)}, {_fmt(d)})"
    if profile.shape == "circle":
        diameter = profile.diameter_mm if profile.diameter_mm is not None else min(w, d)
        return f".circle({_fmt(diameter / 2)})"
    if profile.shape == "polyline":
        # validator guarantees >= 3 vertices in mm
        return _polyline_call(list(profile.vertices))  # type: ignore[arg-type]
    if profile.shape == "arc_line":
        # validator guarantees >= 2 arc_line_segments
        return _arc_line_call(list(profile.arc_line_segments))  # type: ignore[arg-type]
    # legacy enums: prefer the vertex list when the LLM provided one
    if profile.vertices and len(profile.vertices) >= 3:
        return _polyline_call(list(profile.vertices))
    # Last-resort bbox rectangle — keeps generated code runnable.
    return f".rect({_fmt(w)}, {_fmt(d)})"


def _first_op_lines(op: SketchOperation) -> list[str]:
    """Emit the lines that build the FIRST operation as the seed solid.

    The first op uses an absolute plane (XY/XZ/YZ) and must produce a
    solid — i.e. it has to be an extrude in this MVP. A first-op cut
    has nothing to cut into.
    """
    if op.plane not in ABSOLUTE_PLANES:
        raise ValueError(
            f"first sketch operation must use an absolute plane "
            f"(XY/XZ/YZ); got {op.plane!r}"
        )
    if op.operation != "extrude":
        raise ValueError(
            f"first sketch operation must be 'extrude' (nothing to cut "
            f"into yet); got {op.operation!r}"
        )

    sketch_call = _profile_2d_call(op.profile)
    distance = op.distance_mm if op.direction == "positive" else -op.distance_mm
    center = ""
    if op.position_x or op.position_y:
        center = f".center({_fmt(op.position_x)}, {_fmt(op.position_y)})"
    return [
        f'result = cq.Workplane("{op.plane}"){center}{sketch_call}'
        f".extrude({_fmt(distance)})"
    ]


def _subsequent_op_lines(op: SketchOperation) -> list[str]:
    """Emit the lines for an op that builds on the existing ``result`` solid.

    For a face selector plane (">Z" etc.) we look the face up by the
    seed-time TAG instead of the live selector, otherwise consecutive
    extrudes on ``>Z`` stack on top of each other. We also pass
    ``centerOption="CenterOfBoundBox"`` — the default
    ``ProjectedOrigin`` would project the previous workplane's origin
    onto the new face and silently rotate the workplane back to the
    original sketch plane (so a hole "into +Y" would actually drill
    straight down through the part).
    """
    sketch_call = _profile_2d_call(op.profile)
    distance = op.distance_mm

    if op.plane in FACE_SELECTORS:
        # ROUTING for face-selector planes (">Z" / "<Z" / "±X" / "±Y"):
        #
        #   - >Z / <Z (top / bottom):
        #       Centered stacked extrudes must use the LIVE selector
        #       (`result.faces(">Z")`) so each new tier sits on the
        #       current top. An OFFSET top extrude belongs on the
        #       ORIGINAL seed face. CUTS always use the SEED tag so
        #       (a) the workplane is anchored to the world frame
        #       regardless of how the live face has shifted after
        #       stacking, and (b) cutThruAll punches through the whole
        #       stack from the seed plane.
        #
        #   - ±X / ±Y (side faces):
        #       After vertical stacking, the live ">Y" selector matches
        #       multiple non-coplanar faces, so the live workplane is
        #       ambiguous. Use the seed tag for both EXTRUDE and CUT —
        #       the seed face is unique and pre-tagged, and cutThruAll
        #       handles "drill through both halves of the part" without
        #       caring whether the workplane sits inside or outside the
        #       solid.
        if op.plane in {">Z", "<Z"}:
            # A circular blind recess (counterbore) must anchor to the
            # CURRENTLY LIVE top face, not the seed face: it sits ON TOP
            # of the boss / latest stacked tier, recessing partway DOWN
            # into it. Routing through the seed tag drills the recess
            # from the flange top instead, so the counterbore floats in
            # the middle of the part where nobody can see it.
            #
            # Through-holes / through-cuts (cutThruAll) keep using the
            # seed tag — they're two-sided so the workplane anchor only
            # needs to be SOMEWHERE in the part; the seed face is the
            # stable reference that doesn't move when tiers stack on top.
            cut_diam = (
                op.profile.diameter_mm
                if op.profile.diameter_mm is not None
                else min(op.profile.width_mm, op.profile.depth_mm)
            ) if op.profile.shape == "circle" else 0.0
            is_blind_recess = (
                op.operation == "cut"
                and op.profile.shape == "circle"
                and op.direction == "negative"
                and 0 < op.distance_mm < cut_diam
            )
            use_seed_face = (
                (op.operation == "cut" and not is_blind_recess)
                or (
                    op.operation == "extrude"
                    and (abs(op.position_x) > 1e-9 or abs(op.position_y) > 1e-9)
                )
            )
            if use_seed_face:
                tag = SEED_FACE_TAGS[op.plane]
                wp = (
                    f'result.faces(tag="{tag}")'
                    f'.workplane(centerOption="CenterOfBoundBox")'
                )
            else:
                wp = (
                    f'result.faces("{op.plane}")'
                    f'.workplane(centerOption="CenterOfBoundBox")'
                )
        else:
            # Side-face CUT or EXTRUDE — use the seed tag as a stable
            # anchor so we attach to the original base side, not whatever
            # live selector resolves to.
            tag = SEED_FACE_TAGS[op.plane]
            wp = (
                f'result.faces(tag="{tag}")'
                f'.workplane(centerOption="CenterOfBoundBox")'
            )
    elif op.plane in ABSOLUTE_PLANES:
        # Treat as a fresh workplane in world space — useful for adding
        # disjoint geometry that gets unioned in by being part of result.
        wp = f'result.copyWorkplane(cq.Workplane("{op.plane}"))'
    else:
        raise ValueError(f"unknown sketch plane: {op.plane!r}")

    if op.position_x or op.position_y:
        wp = f"{wp}.center({_fmt(op.position_x)}, {_fmt(op.position_y)})"

    if op.operation == "extrude":
        signed = distance if op.direction == "positive" else -distance
        return [f"result = {wp}{sketch_call}.extrude({_fmt(signed)})"]

    if op.operation == "cut":
        if op.profile.shape == "circle":
            diameter = (
                op.profile.diameter_mm
                if op.profile.diameter_mm is not None
                else min(op.profile.width_mm, op.profile.depth_mm)
            )
            # SHALLOW recess vs THROUGH-HOLE classification:
            # Through-hole cuts are emitted by `_match_holes` with
            # distance_mm = axis_extent + 1 (i.e. slightly longer than
            # the part along that axis). Recess cuts (from
            # `_build_stacked_extrudes` thin-ring detection) get a
            # MUCH smaller distance — the rim wall thickness, typically
            # <10% of the part's Z extent.
            #
            # Use cutThruAll for through-holes — that's two-sided and
            # independent of where the workplane sits inside the part,
            # so a Y-axis hole through a thin slab works even though
            # the absolute XZ workplane is on one face. For shallow
            # recesses, keep `.hole(d, depth=X)` which carves a blind
            # cylinder of the requested depth from the workplane.
            #
            # Through-vs-blind heuristic: through-cuts are as deep as
            # the diameter OR deeper (a hole drilled 50mm deep in a
            # 38mm-wide part is asking to pierce the back face).
            # Recesses are FAR shallower (depth/diameter typically
            # 0.05-0.15), so this threshold cleanly separates them
            # without needing a part-extent reference.
            is_through = (
                op.direction == "positive"
                or distance >= diameter
            )
            if is_through:
                return [
                    f"result = {wp}.circle({_fmt(diameter / 2)}).cutThruAll()"
                ]
            return [
                f"result = {wp}.hole({_fmt(diameter)}, depth={_fmt(distance)})"
            ]
        # Rectangular / polygon cut.
        signed = -distance if op.direction == "positive" else distance
        return [f"result = {wp}{sketch_call}.cutBlind({_fmt(signed)})"]

    if op.operation == "revolve":
        # Revolve is reserved in the schema but not implemented in the
        # MVP builder. Skip with a marker comment so the generated
        # script still runs.
        return [
            f"# skipped 'revolve' op order={op.order}: "
            f"revolve not implemented in sketch_builder MVP"
        ]

    raise ValueError(f"unknown operation type: {op.operation!r}")


def build_from_sketches(part: SketchPartDescription) -> str:
    """Return a complete, runnable CadQuery script for ``part``.

    The script follows the same conventions as
    ``backend.ai_infra.cadquery_builder.build_cadquery``:

      * one ``import cadquery as cq`` line at the top
      * ``result`` is the final variable
      * each operation is on its own line
    """
    ops = sorted(part.sketches, key=lambda s: s.order)
    if not ops:
        raise ValueError("SketchPartDescription has no operations")

    lines: list[str] = [
        "import cadquery as cq",
        "",
        "# Dimensions auto-calibrated from noisy mesh; override per-tier "
        "mm values to use exact specs.",
        "",
    ]
    lines.extend(_first_op_lines(ops[0]))
    # Tag the seed solid's six cardinal faces BEFORE any subsequent
    # operation runs, so face lookups by tag stay anchored to the
    # original base regardless of features added on top.
    lines.extend(SEED_FACE_TAG_LINES)
    for op in ops[1:]:
        lines.extend(_subsequent_op_lines(op))
    return "\n".join(lines) + "\n"


__all__ = ["build_from_sketches"]
