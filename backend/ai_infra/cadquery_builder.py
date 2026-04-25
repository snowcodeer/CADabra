"""Translate a ``PartDescription`` into a runnable CadQuery code string.

This is a pure code generator: ``build_cadquery`` returns Python source
text that the caller can ``exec`` (typically inside a sandboxed
subprocess in ``scripts/roundtrip_demo.py``). This module never calls
``exec`` itself, never imports ``cadquery``, and never spawns a
subprocess. Keeping it side-effect-free means it can be tested without
having an OCCT install handy.

Conventions of the emitted code:

* The first line is always ``import cadquery as cq``.
* The final variable is always named ``result`` and is a
  ``cq.Workplane`` carrying the assembled solid.
* Each base-shape branch and each feature is on its own line so a
  CadQuery traceback points at the offending step in the generated
  source rather than a single mile-long expression.

CLI:

    python -m backend.ai_infra.cadquery_builder path/to/part.json
"""

import json
import sys
from pathlib import Path

from backend.ai_infra.models import EdgeTreatment, Feature, PartDescription


SELECTOR_MAP: dict[str, str] = {
    "+Z": ">Z",
    "-Z": "<Z",
    "+X": ">X",
    "-X": "<X",
    "+Y": ">Y",
    "-Y": "<Y",
}

# Maps the EdgeSet enum values to CadQuery edge selector strings.
# `None` means call `.edges()` with no selector (= every edge).
EDGE_SELECTOR_MAP: dict[str, str | None] = {
    "all": None,
    "top_outer": ">Z",
    "bottom_outer": "<Z",
    "vertical": "|Z",
    "horizontal": "#Z",
}

# Tag names applied to the six cardinal faces of the base body. Every
# feature is then placed by tag (e.g. .faces(tag="base+Z")) so that
# adding one boss does NOT change which face the next feature lands on.
# Without these tags, .faces(">Z") drifts to the top of the most recent
# extrusion and turns "two pillars side by side" into "one stacked tower".
def _tag_for(face: str) -> str:
    return f"base{face}"


# Tagging chain emitted right after the base body.
#
# Two non-obvious details:
#   1. Each tag is followed by ``.end()``. Without it CadQuery accumulates
#      the previous tag's selection state and every subsequent
#      ``.faces(">X").tag(...)`` ends up tagging the SAME face as the
#      first tag.
#   2. When the feature step does ``.faces(tag="base+Y").workplane()``
#      it MUST pass ``centerOption="CenterOfBoundBox"`` (see
#      ``_feature_line``). Otherwise CadQuery defaults to
#      ``ProjectedOrigin``, which projects the original ``XY`` plane
#      origin onto the side face and silently rotates the workplane
#      back to ``+Z`` — so a hole that should be drilled into +Y ends
#      up drilled straight down through the part.
TAG_BASE_FACES_LINES: list[str] = [
    "result = (",
    "    result",
    *[
        f'    .faces("{sel}").tag("{_tag_for(face)}").end()'
        for face, sel in SELECTOR_MAP.items()
    ],
    ")",
]


def _fmt(v: float) -> str:
    """Render a float without trailing-zero noise (e.g. 30.0 -> '30.0')."""
    return f"{float(v)}"


def _base_body_lines(part: PartDescription) -> list[str]:
    """Emit the lines that build the base body and tag its 6 cardinal faces.

    The tagging chain at the bottom is what makes feature placement
    deterministic — see ``_tag_for`` for the rationale.
    """
    base = part.base
    plane = base.sketch_plane
    w, d, h = base.width_mm, base.depth_mm, base.height_mm

    if base.shape == "rectangle":
        body: list[str] = [
            f'result = cq.Workplane("{plane}").box({_fmt(w)}, {_fmt(d)}, {_fmt(h)})'
        ]

    elif base.shape == "circle":
        body = [
            f'result = cq.Workplane("{plane}").circle({_fmt(w / 2)}).extrude({_fmt(h)})'
        ]

    elif base.shape == "l_shape":
        # Box first, then cut the top-right quadrant (positive X, positive Y)
        # so the +Z footprint reads as an L from above. Tagging happens
        # AFTER the cut so "+Z" refers to the L-shaped top, not the
        # original full rectangle top.
        corner_w = w / 2
        corner_d = d / 2
        corner_cx = w / 4
        corner_cy = d / 4
        body = [
            f'result = cq.Workplane("{plane}").box({_fmt(w)}, {_fmt(d)}, {_fmt(h)})',
            (
                f'result = result.faces(">Z").workplane()'
                f'.moveTo({_fmt(corner_cx)}, {_fmt(corner_cy)})'
                f'.rect({_fmt(corner_w)}, {_fmt(corner_d)}).cutThruAll()'
            ),
        ]

    elif base.shape == "t_shape":
        # Vertical bar unioned with a horizontal bar pushed to the +Y edge
        # so the silhouette reads as a T from above.
        v_w = w / 3
        h_d = d / 3
        h_cy = (d - h_d) / 2
        body = [
            f'_vbar = cq.Workplane("{plane}").box({_fmt(v_w)}, {_fmt(d)}, {_fmt(h)})',
            (
                f'_hbar = cq.Workplane("{plane}").center(0, {_fmt(h_cy)})'
                f'.box({_fmt(w)}, {_fmt(h_d)}, {_fmt(h)})'
            ),
            "result = _vbar.union(_hbar)",
        ]

    elif base.shape == "polygon":
        # Use the part's longest XY extent as the circumscribed-circle
        # diameter for the polygon. This means the polygon's flat-to-flat
        # or vertex-to-vertex span (depending on n) is approximately
        # max(width_mm, depth_mm) — which is what the +Z view label
        # shows. CadQuery's `.polygon(nSides, diameter)` extrudes a
        # regular n-gon inscribed in that diameter.
        sides = base.sides if base.sides is not None else 6
        diameter = max(w, d)
        body = [
            (
                f'result = cq.Workplane("{plane}")'
                f'.polygon({sides}, {_fmt(diameter)}).extrude({_fmt(h)})'
            )
        ]

    else:  # unknown shape -> rectangle fallback so we still produce *something*
        body = [
            f'result = cq.Workplane("{plane}").box({_fmt(w)}, {_fmt(d)}, {_fmt(h)})'
        ]

    return body + TAG_BASE_FACES_LINES


def _feature_line(feat: Feature) -> str | None:
    """Emit the single chained line that applies one feature to ``result``.

    Selects the face by tag (so the placement does not drift as more
    features are added), then uses ``moveTo`` for absolute placement on
    that workplane (``.center`` is cumulative across consecutive
    workplane chains, which silently misplaces every feature after the
    first one on a given face).

    Returns ``None`` when the feature combination is unsupported (e.g. a
    boss with no height, a hole with no diameter); the caller skips
    those rather than emitting broken code.
    """
    tag = _tag_for(feat.face)
    chain = (
        f'result = result.faces(tag="{tag}")'
        f'.workplane(centerOption="CenterOfBoundBox")'
        f".moveTo({_fmt(feat.position_x)}, {_fmt(feat.position_y)})"
    )

    if feat.type == "hole" and feat.shape == "circle":
        if feat.diameter_mm is None:
            return None
        if feat.depth_type == "through":
            return chain + f".hole({_fmt(feat.diameter_mm)})"
        if feat.depth_type == "blind":
            if feat.height_mm is None:
                return None
            return (
                chain
                + f".circle({_fmt(feat.diameter_mm / 2)})"
                + f".cutBlind(-{_fmt(feat.height_mm)})"
            )
        return None

    if feat.type == "boss" and feat.shape == "circle":
        if feat.diameter_mm is None or feat.height_mm is None:
            return None
        return (
            chain
            + f".circle({_fmt(feat.diameter_mm / 2)})"
            + f".extrude({_fmt(feat.height_mm)})"
        )

    if feat.type == "boss" and feat.shape == "rectangle":
        if feat.width_mm is None or feat.depth_mm is None or feat.height_mm is None:
            return None
        return (
            chain
            + f".rect({_fmt(feat.width_mm)}, {_fmt(feat.depth_mm)})"
            + f".extrude({_fmt(feat.height_mm)})"
        )

    if feat.type == "pocket":
        if feat.width_mm is None or feat.depth_mm is None or feat.height_mm is None:
            return None
        return (
            chain
            + f".rect({_fmt(feat.width_mm)}, {_fmt(feat.depth_mm)})"
            + f".cutBlind(-{_fmt(feat.height_mm)})"
        )

    if feat.type == "slot":
        # Schema convention for slot:
        #     width_mm = NARROW dimension (rounded-end diameter)
        #     depth_mm = LONG dimension (overall slot length)
        # CadQuery's slot2D signature is slot2D(length, diameter), so we
        # pass them in that order. Getting this wrong silently produces
        # a slot that's wide instead of long (or vice-versa) but builds
        # without error — exactly the kind of dimension swap that's hard
        # to diagnose visually.
        if feat.width_mm is None or feat.depth_mm is None:
            return None
        return (
            chain
            + f".slot2D({_fmt(feat.depth_mm)}, {_fmt(feat.width_mm)})"
            + ".cutThruAll()"
        )

    return None


def _edge_treatment_line(treat: EdgeTreatment) -> str:
    """Emit one line that applies a fillet or chamfer to a coherent edge set.

    ``edges="all"`` becomes ``.edges()`` (every edge); the other selectors
    map through ``EDGE_SELECTOR_MAP``. We deliberately put each treatment
    on its own ``result = ...`` line so a CadQuery exception during build
    points at the failing treatment instead of a single mile-long expression.
    """
    sel = EDGE_SELECTOR_MAP[treat.edges]
    op = "fillet" if treat.type == "fillet" else "chamfer"
    if sel is None:
        return f"result = result.edges().{op}({_fmt(treat.size_mm)})"
    return f'result = result.edges("{sel}").{op}({_fmt(treat.size_mm)})'


def build_cadquery(part: PartDescription) -> str:
    """Return a complete, runnable CadQuery script as a single string.

    The script imports ``cadquery as cq``, builds the base body, tags
    its six cardinal faces, applies every feature in order, and finally
    applies any base-body edge treatments (fillets/chamfers). Edge
    treatments come LAST because they would otherwise change which
    edges/faces the feature face-tags resolve to.
    """
    lines: list[str] = ["import cadquery as cq", ""]
    lines.extend(_base_body_lines(part))

    for feat in part.features:
        line = _feature_line(feat)
        if line is not None:
            lines.append(line)
        else:
            lines.append(
                f"# skipped unsupported feature: type={feat.type} "
                f"shape={feat.shape} face={feat.face}"
            )

    for treat in part.base.edge_treatments:
        lines.append(_edge_treatment_line(treat))

    return "\n".join(lines) + "\n"


def preview_code(part: PartDescription) -> None:
    """Generate the CadQuery code for ``part`` and print it to stdout."""
    print(build_cadquery(part))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m backend.ai_infra.cadquery_builder path/to/part_description.json",
            file=sys.stderr,
        )
        sys.exit(2)

    data = json.loads(Path(sys.argv[1]).read_text())
    preview_code(PartDescription(**data))
