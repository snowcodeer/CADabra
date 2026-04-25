"""Anthropic SDK wrapper for the face-geometry pipeline.

The contract is intentionally narrower than ``sketch_llm_client``:

  Claude's job here is *construction-sequence reasoning only*.

Every numeric value Claude needs is already in the
``ExtractedGeometry`` JSON we attach to the user message — face
vertices, areas, cylinder radii, axis directions, all measured
directly from the mesh. Claude must NOT re-measure pixels and MUST
copy these numbers verbatim into its output. What Claude IS for is
deciding which face group is the base sketch, which features are
additive vs subtractive, and what order they go in.

Reuses ``SketchPartDescription`` from ``sketch_models`` so that
``sketch_builder.build_from_sketches`` (and the comparison harness)
stay completely unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from backend.ai_infra.face_extractor import ExtractedGeometry, summarise_geometry
from backend.ai_infra.sketch_models import SketchPartDescription


_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_DOTENV_PATH)


CLAUDE_MODEL = "claude-opus-4-7"
# Lower than the sketch-plane client because the geometry is
# pre-extracted: Claude is doing pure ordering reasoning, not
# numeric extraction. 4000 tokens leaves room for the Q1-Q5
# explanations plus the JSON envelope.
MAX_TOKENS = 4000

JSON_BEGIN = "===JSON_BEGIN==="
JSON_END = "===JSON_END==="


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are a CAD construction-planning assistant.

You receive TWO inputs:

(1) A 2400 × 1000 px FACE GEOMETRY DIAGRAM image. Each of the six
    panels shows the part projected onto one principal plane:

        Row 1: +Z (top)   +X (right)   +Y (front)
        Row 2: -Z (bottom)  -X (left)    -Y (back)

    Inside each panel:
      * Filled coloured polygons are the part's PLANAR FACES, with
        their face id printed at the centroid (e.g. "F12") and the
        face area underneath (e.g. "1042 mm²").
      * Solid grey filled circles are CYLINDRICAL BOSSES (convex —
        the surface points outward from the part).
      * Dashed red-outlined circles are CYLINDRICAL HOLES (concave —
        the surface points inward).
      * A short labelled "10 mm" / "20 mm" bar is the panel's scale
        bar; you do NOT need it because all measurements are also
        attached as exact JSON.

(2) An ExtractedGeometry JSON describing every face exactly. Vertex
    coordinates, areas, cylinder radii / heights / centres, and
    bounding box are all geometrically precise — they came straight
    from the mesh, not from any pixel measurement. USE THESE NUMBERS
    DIRECTLY IN YOUR OUTPUT. Do not round, re-estimate, or alter them.

YOUR JOB IS REASONING, NOT MEASUREMENT
======================================
Every value you put in the output JSON must come from the
ExtractedGeometry. Your contribution is figuring out:

  Q1. Which face group is the BASE SKETCH?
      Usually the face group with the largest total area, lying on a
      principal plane (typically +Z or -Z). Its outer boundary
      polygon (already in vertices_2d_mm) IS the base profile.

  Q2. Which features are ADDITIVE? (extrudes ON TOP of the base)
      * Planar faces parallel to the base but at a different Z height
        whose outline does not match the base outline = pillar tops
        / boss tops.
      * Cylindrical bosses (face_type == "boss") = circular extrudes
        on a face.

  Q3. Which features are SUBTRACTIVE? (cuts INTO the base)
      * Cylindrical holes (face_type == "hole") = .hole() cuts.
      * Recessed planar faces bounded by walls = pocket cuts.

  Q4. Which side faces (±X, ±Y normals) are JUST RESULTS of the base
      extrude and should NOT be added as separate operations?
      A side face whose UV bounding box equals (base_extrude_height,
      base_outer_dimension) is a result. Skip it.

  Q5. What is the CORRECT ORDER?
      1. Base sketch + extrude.
      2. Additive features in ascending Z order.
      3. Subtractive features last.

CONSTRUCTION RULES
==================
Rule 1: BASE PROFILE comes from the largest planar face on a
        principal plane. Use that face's vertices_2d_mm exactly.
        - 4-vertex face with shape_type == "rectangle" → use
          width_mm = bounding_box_mm[0], depth_mm = bounding_box_mm[1].
          Profile shape = "rectangle".
        - Anything else → Profile shape = "polyline" and the
          vertices field MUST be a verbatim copy of vertices_2d_mm.

Rule 2: BASE EXTRUDE distance = height of the part along the axis
        perpendicular to the base. Read it from
        ExtractedGeometry.bounding_box_mm: if base is on ±Z, the
        extrude distance is bounding_box_mm[2].

Rule 3: ADDITIVE PILLAR / BOSS extrudes go on the face selector that
        matches the base's plane (e.g. ">Z" if the base sits on +Z).
        - Planar pillar top: profile width/depth = its bounding_box_mm.
          distance_mm = pillar top z - base top z.
          position_x / position_y = pillar centre_3d_mm minus base
          face's centre_3d_mm, projected onto the base plane.
        - Cylindrical boss: profile shape "circle",
          diameter_mm = 2 * radius_mm. distance_mm = boss height_mm.

Rule 4: HOLE CUTS use shape "circle", diameter_mm = 2 * radius_mm,
        operation = "cut", direction matches the cut direction on the
        chosen face. distance_mm = at LEAST the part thickness through
        which the hole goes (use cylinder height_mm, padded slightly
        if you suspect it goes further). position_x / position_y =
        cylinder centre_3d_mm projected onto the base face plane,
        minus the base face's centroid.

Rule 5: SIDE FACES (±X, ±Y) that match (base_extrude_height ×
        base_dimension) are NOT operations. Drop them.

Rule 6: NEVER MERGE separate features. Two holes with the same
        radius at different XY positions = two separate
        SketchOperation entries.

POSITION CONVENTION
===================
position_x and position_y are the offset from the WORKING FACE's
centroid to the FEATURE's centroid, measured in the working face's
local UV (which matches the world axes, see face_extractor docs).

A feature at the exact centre of the face → (0, 0).
A feature 30 mm to the right of the centre on a +Z face → (30, 0).

OUTPUT FORMAT
=============
First write your reasoning briefly:

  Q1: ...
  Q2: ...
  Q3: ...
  Q4: ...
  Q5: ...

Then the JSON wrapped EXACTLY like this (markers required):

{JSON_BEGIN}
{{ ...SketchPartDescription JSON... }}
{JSON_END}

SKETCHPARTDESCRIPTION SCHEMA
============================
{{
  "sketches": [
    {{
      "order": <int starting at 1>,
      "plane": "XY" | "XZ" | "YZ" | ">Z" | "<Z" | ">X" | "<X" | ">Y" | "<Y",
      "profile": {{
        "shape": "rectangle" | "circle" | "polyline",
        "width_mm":   <float or null>,
        "depth_mm":   <float or null>,
        "diameter_mm":<float or null>,
        "vertices":   <list of [u,v] pairs in mm or null>
      }},
      "operation": "extrude" | "cut",
      "distance_mm": <float>,
      "direction":   "positive" | "negative",
      "position_x":  <float>,
      "position_y":  <float>
    }}
  ],
  "bounding_box_mm": [<float>, <float>, <float>],
  "confidence": "high" | "medium" | "low",
  "notes": <string or null>
}}

The first sketch must use an ABSOLUTE plane (XY / XZ / YZ).
Subsequent sketches that act on a base face should use the face
selector form (>Z, <Z, >X, ...).
"""


USER_PROMPT_TEMPLATE = """\
Here is the exact face geometry extracted from the mesh:

ExtractedGeometry JSON:
{geometry_json}

Plain-English summary:
{summary}

Reason briefly through Q1-Q5 (one short paragraph each, max ~3
sentences) then emit the JSON envelope wrapped in {JSON_BEGIN} /
{JSON_END} markers.
"""


# ---------------------------------------------------------------------------
# JSON parsing — same envelope/bare-ops/largest-blob fallback chain as
# sketch_llm_client so a single Claude quirk does not break the run.
# ---------------------------------------------------------------------------
def _all_balanced_json_objects(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    found: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                _, end = decoder.raw_decode(text[i:])
                found.append(text[i : i + end])
                i += end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return found


def _between_markers(text: str) -> str | None:
    """Pull the substring between ``JSON_BEGIN`` / ``JSON_END`` markers
    when Claude follows the instruction; otherwise return None."""
    if JSON_BEGIN in text and JSON_END in text:
        start = text.index(JSON_BEGIN) + len(JSON_BEGIN)
        end = text.index(JSON_END, start)
        return text[start:end].strip()
    return None


def _pick_part_description(text: str) -> dict:
    """Same fallback strategy as sketch_llm_client. Markers preferred,
    then "any object with a 'sketches' key", then bare-ops synthesis,
    then largest-blob.
    """
    marked = _between_markers(text)
    candidates: list[str] = []
    if marked:
        candidates.append(marked)
    candidates.extend(_all_balanced_json_objects(text))

    parsed: list[dict] = []
    for blob in candidates:
        try:
            parsed.append(json.loads(blob))
        except json.JSONDecodeError:
            continue
    if not parsed:
        raise ValueError(
            "No JSON object could be parsed from the model response:\n" + text
        )

    envelopes = [p for p in parsed if isinstance(p, dict) and "sketches" in p]
    if envelopes:
        return max(envelopes, key=lambda d: len(d.get("sketches", [])))

    bare_ops = [p for p in parsed if isinstance(p, dict) and "order" in p and "profile" in p]
    if bare_ops:
        bare_ops.sort(key=lambda d: d.get("order", 0))
        return {
            "sketches": bare_ops,
            "bounding_box_mm": [0.0, 0.0, 0.0],
            "confidence": "low",
            "notes": "synthesised from bare operations (no envelope)",
        }

    return max(parsed, key=lambda d: len(json.dumps(d)))


def _fill_required_profile_dims(data: dict) -> dict:
    """Fill ``width_mm`` / ``depth_mm`` for circle and polyline
    profiles where Claude (correctly, per my prompt) emitted ``null``.

    ``Profile2D`` keeps width/depth as REQUIRED non-nullable floats —
    even for circles and polylines, where the values are derived
    rather than primary. Rather than modify ``sketch_models.py``
    (which is shared with the sketch-plane pipeline) we synthesise
    sensible defaults so the same JSON schema works for both clients:

      * circle  -> width = depth = diameter
      * polyline -> width = U-bbox, depth = V-bbox of vertices

    Mutates the dict in place AND returns it so callers can chain.
    """
    for sketch in data.get("sketches", []):
        prof = sketch.get("profile") or {}
        shape = prof.get("shape")
        w, d = prof.get("width_mm"), prof.get("depth_mm")
        if shape == "circle" and prof.get("diameter_mm") is not None:
            if w is None:
                prof["width_mm"] = float(prof["diameter_mm"])
            if d is None:
                prof["depth_mm"] = float(prof["diameter_mm"])
        elif shape == "polyline" and prof.get("vertices"):
            verts = prof["vertices"]
            us = [float(v[0]) for v in verts]
            vs = [float(v[1]) for v in verts]
            if w is None:
                prof["width_mm"] = max(us) - min(us)
            if d is None:
                prof["depth_mm"] = max(vs) - min(vs)
    return data


def _validate(data: dict) -> SketchPartDescription:
    data = _fill_required_profile_dims(data)
    try:
        return SketchPartDescription(**data)
    except ValidationError as exc:
        raise ValueError(
            "SketchPartDescription validation failed.\n"
            f"Raw data: {data}\n"
            f"Pydantic error: {exc}"
        ) from exc


def _encode_image(path: str | Path) -> dict:
    image_bytes = Path(path).read_bytes()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(image_bytes).decode("ascii"),
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def call_claude_faces(
    diagram_path: str | Path,
    geometry: ExtractedGeometry,
) -> SketchPartDescription:
    """Run the face-geometry vision pipeline.

    ``diagram_path``: PNG produced by ``face_diagram_renderer``. The
        sibling ``.json`` is read automatically and embedded in the
        user message so Claude never has to re-extract numbers.
    ``geometry``: the same ExtractedGeometry that was used to render
        the diagram (we accept it directly to avoid double-loading).

    Returns a validated ``SketchPartDescription`` ready to feed
    straight into ``sketch_builder.build_from_sketches``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to backend/.env or "
            "export it in your shell before running the pipeline."
        )

    client = anthropic.Anthropic()

    user_text = USER_PROMPT_TEMPLATE.format(
        geometry_json=geometry.model_dump_json(indent=2),
        summary=summarise_geometry(geometry),
        JSON_BEGIN=JSON_BEGIN,
        JSON_END=JSON_END,
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    _encode_image(diagram_path),
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Print Claude's reasoning + envelope so the orchestrator can
    # show it without a second API call.
    print("--- Claude (face-geometry) ---")
    print(raw)
    print("--- end Claude ---", flush=True)

    data = _pick_part_description(raw)
    return _validate(data)


# ---------------------------------------------------------------------------
# CLI: python -m backend.ai_infra.face_llm_client diagram.png
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Send a face-diagram PNG + its JSON to Claude.",
    )
    p.add_argument("diagram", type=Path, help="PNG produced by face_diagram_renderer.")
    args = p.parse_args()

    json_path = args.diagram.with_suffix(".json")
    if not json_path.exists():
        print(
            f"error: expected geometry JSON at {json_path}. "
            "Re-run face_diagram_renderer first.",
            file=sys.stderr,
        )
        sys.exit(2)
    geometry = ExtractedGeometry.model_validate_json(json_path.read_text())

    part = call_claude_faces(args.diagram, geometry)
    print(part.model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
