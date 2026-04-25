"""Anthropic SDK wrapper for the sketch-plane architecture.

Two-stage pipeline:

  Stage A (no API call): OpenCV reads the 6-view grid PNG and produces
                         a structured per-view contour summary.
  Stage B (one API call): Claude receives BOTH the raw image AND the
                          OpenCV summary. It is told NOT to re-extract
                          2D shapes, only to reason about depth ordering
                          and construction sequence.

This is a sibling of ``backend.ai_infra.llm_client``. We deliberately
do not import or modify the ``BaseBody+features`` client so the two
architectures can be benchmarked head-to-head.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from backend.ai_infra.contour_extractor import (
    MeshBounds,
    extract_all_views,
    summarise_contours,
)
from backend.ai_infra.sketch_models import SketchPartDescription


_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_DOTENV_PATH)


CLAUDE_MODEL = "claude-opus-4-7"
# Bumped from 3000 because the hierarchical-decomposition prompt now
# encourages explicit per-step reasoning and Opus was running out of
# tokens BEFORE emitting the JSON envelope on complex parts (e.g.
# 061490). 8000 leaves headroom for ~6-step reconstructions.
MAX_TOKENS = 8000

JSON_BEGIN = "===JSON_BEGIN==="
JSON_END = "===JSON_END==="


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are a CAD construction-planning assistant. You receive TWO inputs:

  (1) A 6-view orthographic PNG of a single rigid mechanical part.
      Layout (always identical, 2400 x 1000 px):
        Row 1: +Z top  | +X right | +Y front
        Row 2: -Z bot  | -X left  | -Y back
      Each cell has an RGB panel (left 400 px) and a depth panel
      (right 400 px).

  (2) A pre-computed OpenCV contour analysis of (1). For each view it
      gives you, IN MILLIMETRES:
        - the 2D outline's bounding box (width x height)
        - the simplified outline as an explicit vertex list
          (panel-local frame, +x right, +y up, origin at panel centre)
        - the bounding boxes of "near-camera" warm regions in the
          depth panel
      These mm values are GROUND TRUTH. Use them. Do NOT remeasure
      pixels.

================================================================
YOUR JOB — HIERARCHICAL GREEDY DECOMPOSITION
================================================================
Reconstruct the ORDERED construction sequence a CAD engineer would
use to build the part from scratch. Repeat this question until the
shape is fully accounted for:

    "What is the largest 2D cross-section of what remains, and on
     which plane does it lie?"

  Step 1 — Pick the dominant base profile.
    Look at the +Z view. Its 2D outline IS the base profile.
    Use OpenCV's vertex list verbatim. Extrude it along Z by the
    object's Z extent (read off any +X / +Y / -X / -Y bbox).
    -> Operation 1, plane="XY", operation="extrude".

  Step 2 — Mentally subtract the base. Look at what is left:
    * Warm regions in the +Z depth panel that sit ABOVE the base top
      face = additive features (bosses, pillars, ribs).
    * Dark-blue regions in the +Z depth panel = pockets / holes that
      go INTO the base.
    * Stepped silhouettes in side views = features at different
      heights or features on side faces.
    Each one becomes ONE SketchOperation, in order:
    additive first, subtractive last.

  Step 3 — For each remaining feature, recover its sketch plane:
    - Boss/pocket on the top face       -> plane=">Z"
    - Boss/pocket on the bottom face    -> plane="<Z"
    - Hole entering from a side face    -> plane=">X" / "<X" /
                                                  ">Y" / "<Y"
    Use the warm-region centre and size (in mm) for position and
    profile dimensions. Through-hole vs blind: a hole that appears
    as a dark-blue circle in BOTH the +Z and -Z depth panels is
    THROUGH; a hole visible only in +Z is BLIND.

  Step 4 — Order the operations. Additive operations that create
  geometry others depend on come first; subtractive operations
  (holes, pockets) come last. The first sketch (order==1) MUST be
  the absolute-plane extrude that produces the seed solid.

================================================================
PROFILE REPRESENTATION — POLYLINE FIRST
================================================================
For each profile, choose the most precise representation:

  * If the outline is a clean rectangle (4 vertices, area_ratio
    > 0.85), use shape="rectangle" with width_mm and depth_mm.
  * If the outline is a clean circle (>= 9 vertices, near-square
    aspect, area_ratio near 0.785), use shape="circle" with
    diameter_mm.
  * OTHERWISE — anything with 5+ vertices and area_ratio < 0.85
    (L, T, U, wedge, irregular polygon) — use shape="polyline"
    and COPY OpenCV's vertices_mm list verbatim into
    profile.vertices. Set width_mm and depth_mm to the bbox
    dimensions OpenCV reports. The builder will call
    .polyline(vertices).close() on those points.

  When in doubt, prefer polyline. A polyline with the correct
  vertices is always safer than a rectangle approximation.

================================================================
DEPTH MAP RULES (RdYlBu_r colormap)
================================================================
  Red / orange    = surface CLOSEST to that camera
  Yellow          = mid-distance
  Dark blue       = surface FURTHEST from that camera
  #1e1e1e (very dark gray) = background, NOT part of the object

  Distinct colour bands across a depth panel = distinct depth levels
  in that view. Use this to spot multiple stacked features.

================================================================
FEATURE-SEPARATION RULE (CRITICAL)
================================================================
If the OpenCV summary reports TWO OR MORE distinct warm regions in
any view, those are TWO OR MORE separate SketchOperations. Do NOT
merge them into one block.

================================================================
DIMENSIONS — WHEN OPENCV BBOXES ARE PRESENT
================================================================
When the OpenCV summary lists "Bounding box: W x H mm" and
vertices_mm for a view, those numbers ARE the object dimensions in
that view. You do not need the visible_mm / 1.2 trick. Read mm
values directly off the summary.

  * Part X extent  = +Z view bbox width
  * Part Y extent  = +Z view bbox height
  * Part Z extent  = +X / +Y / -X / -Y view bbox height

  Feature dimensions and positions: read directly off the
  warm-region (cx_mm, cy_mm, w_mm, h_mm) tuples. They are already in
  the same panel-local frame (origin at panel centre, +x right,
  +y up) as profile.vertices, so you can use them as
  position_x / position_y without further conversion.

Round all dimensions to the nearest 1 mm.

================================================================
OUTPUT FORMAT
================================================================
After your reasoning, emit EXACTLY one JSON object that matches the
schema below, wrapped between the markers shown. Nothing else after
the closing marker.

{JSON_BEGIN}
{{
  "sketches": [
    {{
      "order":       <int, 1-based, sequential>,
      "plane":       "XY" | "XZ" | "YZ" |
                     ">Z" | "<Z" | ">X" | "<X" | ">Y" | "<Y",
      "profile": {{
        "shape":       "rectangle" | "circle" | "polyline",
        "width_mm":    <float>,
        "depth_mm":    <float>,
        "diameter_mm": <float | null>,
        "vertices":    <list of [x_mm, y_mm] | null>
      }},
      "operation":   "extrude" | "cut" | "revolve",
      "distance_mm": <float>,
      "direction":   "positive" | "negative",
      "position_x":  <float>,
      "position_y":  <float>
    }}
  ],
  "bounding_box_mm": [<width X>, <depth Y>, <height Z>],
  "confidence":      "high" | "medium" | "low",
  "notes":           <string | null>
}}
{JSON_END}

Field rules and downstream usage:

  * The FIRST sketch (order==1) MUST use an absolute plane
    (XY / XZ / YZ) and operation "extrude". It is the seed solid.
  * Every subsequent sketch SHOULD use a face selector (">Z" etc.)
    so the builder lands the operation on an existing face.
  * profile.vertices, when present, is a list of [x_mm, y_mm] in the
    sketch plane's local frame (the same frame OpenCV reported the
    vertices in). Do NOT close the loop yourself — the builder
    appends .close().
  * For a circular hole, set profile.shape="circle",
    profile.diameter_mm=<diameter>, operation="cut", and
    distance_mm equal to the part thickness in that direction. The
    builder converts that into a through-hole automatically.
  * position_x / position_y are sketch-centre offsets from the
    workplane centre, in mm. (0, 0) means perfectly centred.
  * bounding_box_mm is the (X, Y, Z) extent of the FINISHED part —
    use it as a self-check.

Be honest about confidence. If something cannot be reconciled
(e.g. the +Z and -Z bbox sizes disagree, or a side view does not
match the implied Z extent), set confidence to "medium" or "low"
and explain in notes.
"""

USER_PROMPT_TEMPLATE = """\
OpenCV contour analysis of the attached grid:

{contour_summary}

Reason BRIEFLY (<= 8 short sentences total). State, in order:
  1. The dominant base profile and which view it came from.
  2. Each subsequent operation (additive features then subtractive
     holes), one short sentence each.
  3. Total operation count.

Then IMMEDIATELY emit the JSON envelope — a single ``{{...}}`` object
with the keys "sketches", "bounding_box_mm", "confidence", "notes" —
wrapped between the markers {begin} and {end}. Do not write anything
after the closing marker. Do not emit individual operations as
top-level JSON; they MUST be inside the "sketches" array.
"""


# ---------------------------------------------------------------------------
# JSON extraction (mirrors llm_client.py's behaviour for parity)
# ---------------------------------------------------------------------------
def _all_balanced_json_objects(text: str) -> list[str]:
    """Return EVERY top-level balanced ``{...}`` blob in ``text``."""
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


def _pick_part_description(text: str) -> dict:
    """Pick the right JSON object out of Claude's response.

    Claude sometimes emits per-operation JSON snippets in its
    reasoning section AND the full envelope; sometimes it skips the
    envelope and emits a list of bare operations; sometimes it nests
    everything inside one envelope. Handle all three:

    1. If any balanced object contains a top-level ``"sketches"`` key,
       prefer that one (it IS the envelope).
    2. Otherwise, collect every object that looks like a single
       SketchOperation (has ``"order"``) and synthesise an envelope
       around them. The caller will set sensible defaults for the
       envelope-only fields.
    3. If neither matches, fall back to the largest balanced object.
    """
    blobs = _all_balanced_json_objects(text)
    if not blobs:
        raise ValueError(
            "No JSON object could be parsed from the model response:\n" + text
        )

    parsed: list[dict] = []
    for b in blobs:
        try:
            parsed.append(json.loads(b))
        except json.JSONDecodeError:
            continue

    # 1. Direct envelope hit.
    envelopes = [p for p in parsed if isinstance(p, dict) and "sketches" in p]
    if envelopes:
        # If there are several, take the one with the most sketches.
        return max(
            envelopes,
            key=lambda e: len(e["sketches"]) if isinstance(e.get("sketches"), list) else 0,
        )

    # 2. Synthesise an envelope from bare operations. Deduplicate by
    #    ``order`` so reasoning-step duplicates do not blow up the list.
    op_blobs = [p for p in parsed if isinstance(p, dict) and "order" in p and "profile" in p]
    if op_blobs:
        by_order: dict[int, dict] = {}
        for op in op_blobs:
            try:
                by_order[int(op["order"])] = op
            except (TypeError, ValueError):
                continue
        if by_order:
            ordered = [by_order[k] for k in sorted(by_order)]
            print(
                "[warn] Claude omitted the SketchPartDescription envelope; "
                f"synthesised one from {len(ordered)} bare operation(s).",
                file=sys.stderr,
            )
            return {
                "sketches": ordered,
                # Compute a rough bbox from the largest profile so the
                # envelope validates; the renderer will recompute the
                # real bounds from the resulting STL anyway.
                "bounding_box_mm": _guess_bbox_from_ops(ordered),
                "confidence": "low",
                "notes": (
                    "Envelope auto-synthesised by sketch_llm_client because "
                    "Claude returned bare operations rather than a wrapping "
                    "object. Treat the construction sequence with extra care."
                ),
            }

    # 3. Last resort.
    return json.loads(max(blobs, key=len))


def _guess_bbox_from_ops(ops: list[dict]) -> list[float]:
    """Conservative bbox guess from a list of bare op dicts (mm)."""
    w = h = d = 1.0
    for op in ops:
        prof = op.get("profile") or {}
        try:
            w = max(w, float(prof.get("width_mm") or 0))
            h = max(h, float(prof.get("depth_mm") or 0))
        except (TypeError, ValueError):
            pass
        try:
            d = max(d, float(op.get("distance_mm") or 0))
        except (TypeError, ValueError):
            pass
    return [w, h, d]


def _extract_json(text: str) -> dict:
    begin = text.find(JSON_BEGIN)
    end = text.find(JSON_END, begin + 1) if begin != -1 else -1
    if begin != -1 and end != -1:
        candidate = text[begin + len(JSON_BEGIN) : end].strip()
        if candidate.startswith("```"):
            candidate = (
                candidate.split("\n", 1)[1] if "\n" in candidate else candidate
            )
            if candidate.endswith("```"):
                candidate = candidate.rsplit("```", 1)[0]
        return json.loads(candidate)
    return _pick_part_description(text)


def _validate(data: dict) -> SketchPartDescription:
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
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_b64,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def call_claude_sketch(
    image_path: str | Path,
    mesh_bounds_mm: Optional[MeshBounds] = None,
) -> SketchPartDescription:
    """Run the sketch-plane vision pipeline on a 6-view grid PNG.

    Stage A: OpenCV pre-processes the image into a contour summary.
    Stage B: Claude reasons about construction order and emits a
             validated ``SketchPartDescription``.

    Pass ``mesh_bounds_mm`` (the original mesh's
    ``(xmin, xmax, ymin, ymax, zmin, zmax)`` AS RENDERED — i.e. after
    any normalisation the renderer applied) to enable the world-mm
    fields on the contour summary. With those fields Claude can copy
    OpenCV's vertices straight into ``polyline`` profiles instead of
    guessing dimensions from pixels. Without them the prompt still
    works but Claude has to fall back to the visible_mm/1.2 trick.

    Raises ``FileNotFoundError`` if the image is missing, ``ValueError``
    if Claude's response cannot be parsed or validated, and propagates
    ``anthropic.APIError`` subclasses untouched.
    """
    contours = extract_all_views(image_path, mesh_bounds_mm=mesh_bounds_mm)
    summary = summarise_contours(contours)

    user_text = USER_PROMPT_TEMPLATE.format(
        contour_summary=summary,
        begin=JSON_BEGIN,
        end=JSON_END,
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    _encode_image(image_path),
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    return _validate(_extract_json(text))


def warn_if_uncertain(part: SketchPartDescription, label: str = "sketch-vision") -> None:
    if part.confidence == "high":
        return
    marker = "!!" if part.confidence == "low" else "**"
    print(
        f"{marker} {label}: confidence={part.confidence}"
        + (f"  notes={part.notes!r}" if part.notes else ""),
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run the sketch-plane vision pipeline.")
    p.add_argument("image", type=Path, help="Path to a 6-view grid PNG.")
    p.add_argument("--print-summary", action="store_true",
                   help="Also print the OpenCV contour summary that was sent to Claude.")
    args = p.parse_args()

    if args.print_summary:
        contours = extract_all_views(args.image)
        print("=== OpenCV summary ===")
        print(summarise_contours(contours))
        print()

    part = call_claude_sketch(args.image)
    warn_if_uncertain(part)
    print("=== SketchPartDescription ===")
    print(part.model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
