"""Anthropic SDK wrapper that turns a 6-view grid PNG into a PartDescription.

Public surface:

    call_claude(image_path)                 -> PartDescription   (initial pass)
    call_claude_refine(orig, recon, curr)   -> PartDescription   (review pass)
    test_call(image_path)                   -> None              (CLI helper)

CLI:

    python -m backend.ai_infra.llm_client path/to/grid.png

Requires ``anthropic`` (pip install anthropic) and ANTHROPIC_API_KEY in
the environment (loaded from ``backend/.env`` via python-dotenv).

Design notes for both passes:

* The model is asked to *reason out loud* per view first, then emit the
  JSON wrapped in ``===JSON_BEGIN===`` / ``===JSON_END===`` markers.
  More chain-of-thought tokens dramatically improve part decomposition
  on busy 6-view grids; the markers let us pull the JSON out reliably
  even when prose surrounds it.
* If the markers are missing (older runs, model variability) we fall
  back to extracting the last balanced ``{...}`` block in the response.
* The system prompt explicitly warns against the "merge two close
  features into one" failure mode that we observed on real samples.
"""

import base64
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from backend.ai_infra.models import PartDescription


_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_DOTENV_PATH)


CLAUDE_MODEL = "claude-opus-4-7"
MAX_TOKENS = 4000  # was 1500 — chain-of-thought needs the headroom

JSON_BEGIN = "===JSON_BEGIN==="
JSON_END = "===JSON_END==="


SYSTEM_PROMPT = f"""\
You are a mechanical-CAD vision model. The user gives you a single PNG
that contains six orthographic renders of one rigid mechanical part,
arranged in a fixed grid. Your job is to read the part's geometry off
those renders and emit a strict JSON description of it.

================================================================
1. GRID LAYOUT (the image is always 2400 x 1000 pixels)
================================================================
The image has a thin header bar at the top, a 3 x 2 cell grid in the
middle, and a thin legend bar at the bottom. The grid cells, indexed
left-to-right then top-to-bottom, are exactly:

    Row 1, column 1 = +Z view (top, looking DOWN the Z axis)
    Row 1, column 2 = +X view (right side, looking along -X)
    Row 1, column 3 = +Y view (front,    looking along -Y)
    Row 2, column 1 = -Z view (bottom, looking UP the Z axis)
    Row 2, column 2 = -X view (left side, looking along +X)
    Row 2, column 3 = -Y view (back,     looking along +Y)

Each cell has a text label at the top naming the view direction.
Trust the layout above; treat the labels as a sanity check, not the
source of truth.

================================================================
2. PER-CELL PANEL LAYOUT
================================================================
Every cell contains TWO panels side by side:
    Left panel  = RGB render  (lit gray mesh on a white background)
    Right panel = Depth map   (false-coloured RdYlBu_r heatmap)

================================================================
3. DEPTH COLOUR KEY (identical for every view)
================================================================
    Red / orange    = surface CLOSEST to that camera (near)
    Yellow          = mid-distance surface
    Dark blue       = surface FURTHEST from that camera (far)
    #1e1e1e (very dark gray) = background, NOT part of the object
    Solid red with a faint radial darkening at the edges = a perfectly
        flat surface facing the camera (the "flat-surface fallback"
        the renderer uses when the entire surface sits at one depth).
        Treat that as a flat top/face, not as a feature.

A text legend along the bottom of the image confirms this key.

================================================================
4. WHAT EACH VIEW REVEALS
================================================================
+Z TOP RGB:
    The 2D footprint of the part from above. White gaps WITHIN the
    silhouette outline = holes or empty space. Aspect ratio of the
    outline = width-to-depth (X:Y) ratio of the base.

+Z TOP DEPTH:
    Red/orange patch = a raised boss (sticks up toward this camera).
    Dark-blue patch  = a through-hole (camera sees the background
                       through it; the depth there is the far clip).
    Uniform mid-colour or the radial-fallback red = a flat top face,
    no features.

+X RIGHT-SIDE RGB:
    Side silhouette from +X. Shows overall HEIGHT.
    Height-to-width ratio in the silhouette = height-to-width ratio
    of the part. Steps or bumps in the silhouette outline = features
    sitting at different heights.
    A circular dark hole INSIDE the silhouette (not on the outline) =
    a HOLE on the +X face (not a feature on +Z). A circular bump
    sticking out of the silhouette outline = a BOSS on the +X face.

+Y FRONT RGB:
    Cross-check on height. Compare with +X. If +X and +Y heights
    disagree the part is taller along one axis than the other.
    Same side-face feature rule as +X: an internal dark patch is a
    feature ON THIS FACE, not on +Z.

+X DEPTH and +Y DEPTH:
    Count the distinct colour bands across the surface = number of
    distinct depth levels in the part. An abrupt colour change =
    a step or ledge in the geometry. A small RED patch surrounded by
    blue = a boss on +X / +Y. A small BLUE patch surrounded by red =
    a hole on that face (or a pocket).

SIDE-FACE COORDINATES:
    When you emit a feature with face="+X", "-X", "+Y" or "-Y", the
    position_x/position_y are FACE-LOCAL coordinates on that side
    face, not world X/Y. Convention used by the builder:
        face="+X" or "-X": position_x = world Y offset,
                           position_y = world Z offset (height)
        face="+Y" or "-Y": position_x = world X offset,
                           position_y = world Z offset (height)
        face="+Z" or "-Z": position_x = world X offset,
                           position_y = world Y offset
    A perfectly centred feature is always (0.0, 0.0). Read the offset
    off the view in which the feature is most visible (the +X RGB for
    a +X feature, etc.).

-Z BOTTOM RGB and DEPTH:
    Compare with +Z. A hole visible in BOTH +Z and -Z = a through
    hole (depth_type "through"). A hole visible in +Z only and
    absent in -Z = a blind pocket (depth_type "blind").

-X vs +X, -Y vs +Y:
    Identical silhouettes = symmetric along that axis. Different
    silhouettes mean an asymmetric feature exists (boss only on one
    side, hole only on one face, etc.).

================================================================
5. DIMENSION READING RULE  (READ, DO NOT GUESS)
================================================================
The renderer puts a real-world scale into every cell so you do NOT
have to estimate proportions. Use the annotations:

  (a) Cell label (top of every cell). Format:
          "<view> ... | visible: <H_AXIS> <H>mm \u00d7 <V_AXIS> <V>mm"
      Example: "+Z \u2014 top view ... | visible: X 100mm \u00d7 Y 80mm"
      That tells you exactly how many millimetres the panel covers
      horizontally and vertically of the world. Read those numbers.

  (b) Scale bar (bottom-left of every RGB panel). A short black bar
      with a label like "10mm" or "20mm". Use it as a cross-check on
      the cell label and as a quick visual ruler when reading feature
      sizes by eye.

IMPORTANT: the cell label states the VISIBLE region of the panel,
which is the object's bounding box plus 10% padding on each side.
The actual object extent along that axis is therefore:

    object_extent_mm = visible_mm / 1.2

How to derive each dimension you need to emit:

  * base.width_mm  = (H mm value from the +Z cell label) / 1.2.
                     This is the X extent of the part.
  * base.depth_mm  = (V mm value from the +Z cell label) / 1.2.
                     This is the Y extent of the part.
  * base.height_mm = (V mm value from the +X or +Y cell label) / 1.2.
                     This is the Z extent of the part. +X and +Y must
                     agree; if they do not, prefer the larger and
                     lower confidence.
  * For a feature, use the cell label of whichever view shows that
    feature most clearly, then measure pixels against the panel size
    (RGB panel is 400 px wide / 400 px tall) to get its mm size.
    The pixel-to-mm scale is direct (no 1.2 correction needed because
    you measure pixels of the feature itself, not of the bounding box):
        feature_mm = feature_pixels * (visible_mm / 400)

  * position_x and position_y are offsets from the FACE CENTRE in
    millimetres. A perfectly centred feature is (0.0, 0.0). Pixel
    distance from the panel centre, scaled by the visible mm/px,
    gives the offset.

Rounding: round all length / diameter / width / depth / height
estimates to the nearest 5 mm; round position offsets to the
nearest 5 mm as well (was 10; with annotations you can be tighter).

Sanity check: bounding boxes of all features must fit inside the
base. If a derived feature size is larger than the corresponding
base dimension, you are reading the scale wrong — re-check the
cell label of that view.

================================================================
6. REASONING ORDER
================================================================
Do this OUT LOUD before emitting JSON (the user prompt asks for it):

Step 1: Identify the base footprint shape from the +Z RGB
        (rectangle, circle, l_shape, t_shape, polygon). READ the
        +Z cell label to get base.width_mm (X) and base.depth_mm (Y).
Step 2: READ the +X cell label V-span to get the part's HEIGHT
        (Z span). Cross-check against the +Y cell label V-span;
        they should agree. If they do not, lower confidence.
Step 3: Find secondary features:
            - +Z DEPTH: warm patches = bosses, dark-blue patches = holes
            - +X / +Y RGB: bumps on the silhouette outline = bosses on
              that side face; steps on the silhouette = ledges; INTERNAL
              dark patches inside the silhouette (not on the outline) =
              holes on that side face. Emit those as features with
              face="+X"/"-X"/"+Y"/"-Y".
Step 4: For each hole found in +Z, compare against -Z:
            - Visible in BOTH = depth_type "through"
            - Visible in +Z only = depth_type "blind"
              (estimate the blind depth from the +Z depth-map shade)
        Same logic for side-face holes: a hole on +X is "through" if it
        also shows in -X.
Step 5: Compare +X vs -X and +Y vs -Y to confirm symmetry. Add any
        asymmetric features you only see on one side.
Step 6: COUNT features one more time before writing JSON. If a side
        view shows two distinct vertical bumps separated by a gap of
        background, that is TWO features, not one.
Step 7: Look for EDGE TREATMENTS (fillets / chamfers) on the base body:
            - Top (+Z) edges rounded? RGB shows the +Z perimeter as a
              soft, gradual transition (no sharp dark line); depth map
              shows a smooth gradient from red (top) to mid-tone at the
              perimeter. Emit edge_treatments=[{{type:"fillet",
              edges:"top_outer", size_mm:R}}].
            - Bottom (-Z) edges rounded? Same test on the -Z view.
            - Vertical edges (the "corners" of a box) rounded? The +X
              and +Y silhouettes lose their sharp corner at the top
              and bottom, OR the +Z RGB shows a curved outline at the
              corners instead of a sharp 90 degree angle. Emit
              edges:"vertical".
            - Same edges show a NARROW BAND of intermediate shade
              between the top face and the side face (not a smooth
              gradient but a flat angled band)? That is a CHAMFER, not
              a fillet — emit type:"chamfer" with size_mm = the band
              width measured against the scale bar.
            - If none of the edges look modified, leave
              edge_treatments empty.

================================================================
7. COMMON PITFALLS — AVOID THESE
================================================================
DO NOT MERGE separate features into one. Specifically:

* Two narrow vertical features in the +Y or -Y RGB view, separated by
  a visible gap of WHITE background between them, are TWO separate
  bosses (typically corner pillars), NOT one wider boss spanning the
  gap. Emit them as two features with different position_x values.
* Two distinct warm-coloured patches in the +Z DEPTH panel separated
  by mid-tone or background pixels are two separate raised features,
  not one.
* Two distinct dark-blue patches in the +Z DEPTH are two separate
  holes, not one slot — unless they share an obvious connecting
  channel of dark blue, in which case it is a slot.
* When the silhouette could be explained EITHER by one wide feature OR
  by two narrow features with a gap, prefer TWO features. The downstream
  builder handles two features cleanly; one feature in the wrong place
  is harder to fix than two features that are slightly mispositioned.
* If +Z and -Z show identical hole shapes, the hole MUST be set to
  depth_type "through". A blind hole is visible only in +Z.
* If you detect TWO corner pillars on the back edge of the +Z view
  (small rectangles tucked against +Y), they likely sit at
  position_y near +depth/2 and position_x near +/- width/4.

================================================================
8. OUTPUT FORMAT (STRICT)
================================================================
After your reasoning, return EXACTLY one JSON object that matches the
schema below, wrapped between the markers shown. Nothing else after
the closing marker.

{JSON_BEGIN}
{{
  "base": {{
    "shape": "rectangle" | "circle" | "l_shape" | "t_shape" | "polygon",
    "sides": <int | null>,
    "width_mm":  <float>,
    "depth_mm":  <float>,
    "height_mm": <float>,
    "sketch_plane": "XY" | "XZ" | "YZ",
    "edge_treatments": [
      {{
        "type":    "fillet" | "chamfer",
        "edges":   "all" | "top_outer" | "bottom_outer" | "vertical" | "horizontal",
        "size_mm": <float>
      }}
    ]
  }},
  "features": [
    {{
      "type":  "boss" | "hole" | "pocket" | "slot",
      "face":  "+Z" | "-Z" | "+X" | "-X" | "+Y" | "-Y",
      "shape": "circle" | "rectangle",
      "sketch_plane": "XY" | "XZ" | "YZ",
      "diameter_mm": <float | null>,
      "width_mm":    <float | null>,
      "depth_mm":    <float | null>,
      "height_mm":   <float | null>,
      "depth_type":  "through" | "blind" | null,
      "position_x":  <float>,
      "position_y":  <float>
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "notes": <string | null>
}}
{JSON_END}

Field rules and downstream usage:

  base.shape — must be one of the five enum values exactly. If the part
      is an n-gon, use "polygon" (NOT "octagon", "hexagon",
      "regular_polygon" — those are NOT valid). Put the side count in
      base.sides.
  base.sides — required when shape="polygon"; the integer number of
      sides (3-12 for typical hardware). Null/omitted for non-polygon
      shapes.
  base.width_mm / depth_mm — for polygon and circle bases these are
      both the diameter of the circumscribed circle (i.e. width=depth=
      OD); the builder uses max(width,depth) as the polygon diameter.
  feature.slot — convention: width_mm = NARROW dimension (rounded-end
      diameter); depth_mm = LONG dimension (overall slot length). The
      builder calls slot2D(depth_mm, width_mm) accordingly. Always
      check which axis the slot is long along — the slot's long axis
      is by default along X on its sketch plane.
  base.edge_treatments — list of fillets/chamfers applied to the BASE
      body (not features). Each entry needs a coarse edge selector:
        "top_outer"    = perimeter of the +Z face only.
        "bottom_outer" = perimeter of the -Z face only.
        "vertical"     = the side edges (parallel to Z), e.g. the four
                         vertical corners of a box.
        "horizontal"   = top + bottom rim edges (perpendicular to Z).
        "all"          = every edge of the base body.
      For a fillet, size_mm = radius. For a chamfer, size_mm = leg
      length. Round to the nearest 1 mm (fillets are usually small).
      OMIT the field (or pass []) when no edge looks modified — adding
      a tiny fillet "just in case" makes builds fail.
  feature side faces — when face is "+X", "-X", "+Y" or "-Y",
      position_x and position_y are face-local: they map to (world_Y,
      world_Z) on +/-X faces and (world_X, world_Z) on +/-Y faces.
      Read them off the SAME view that the feature is most visible in.

Other downstream usage:
    - sketch_plane decides which argument is passed to cq.Workplane().
    - face decides which face tag the feature lands on.
    - depth_type "through" -> .hole(diameter_mm); "blind" -> .cutBlind.
    - position_x / position_y -> .moveTo(position_x, position_y).

Be honest about confidence. If you had to guess on feature count or
positions, set confidence to "medium" or "low" and explain in notes.
"""


REFINE_SYSTEM_PROMPT = f"""\
You are reviewing a CAD reconstruction.

You are given:
  1. The ORIGINAL 6-view orthographic grid of a mechanical part.
  2. A RECONSTRUCTED 6-view orthographic grid generated by building the
     part from the JSON description in the user message.
  3. The current JSON description.

Both grids use the identical layout, camera, and depth colourmap
documented in the standard rules: 3x2 cell grid (+Z, +X, +Y on top
row; -Z, -X, -Y on bottom row), each cell split into RGB and Depth
panels, RdYlBu_r colormap (red=near, blue=far, #1e1e1e=background).

Your job: identify what is wrong with the reconstruction and return a
CORRECTED JSON description in the same schema. Common errors to look
for:

  * Missing features (clearly visible in original, absent in
    reconstruction).
  * Extra or duplicated features in the reconstruction.
  * Wrong feature COUNT (one boss in reconstruction where the
    original shows two separate bosses, etc.).
  * Wrong dimensions (reconstruction silhouette is the wrong size or
    aspect ratio).
  * Wrong positions (feature is in the right region but offset
    incorrectly).
  * Wrong depth_type ("blind" where it should be "through" or
    vice-versa — check whether the hole appears in both +Z and -Z).
  * Missing or wrong-radius edge_treatments. If the original shows
    rounded top edges and the reconstruction shows sharp ones, add
    {{type:"fillet", edges:"top_outer", size_mm:R}} to base.edge_treatments
    (R measured against the scale bar). Same logic for chamfers and
    for the bottom/vertical edge sets. If the reconstruction has a
    fillet that the original does NOT, remove that entry.
  * Side-face features missed. If the original +X (or +Y) RGB shows a
    circular dark patch INSIDE the silhouette but the reconstruction
    does not, that feature was incorrectly placed on +Z (or omitted).
    Re-emit it with face="+X"/"-X"/"+Y"/"-Y" and face-local
    position_x/position_y as defined in the standard rules.

Same dimension reading rule applies. Every cell label states the
VISIBLE region in mm (which is the bounding box plus 10% padding on
each side); the actual object extent is the visible value DIVIDED BY
1.2:

    object_extent_mm = visible_mm / 1.2

So if the +Z label says "visible: X 80mm \u00d7 Y 120mm" the part is
~67 mm wide (X) and ~100 mm deep (Y). Each RGB panel also has a
scale bar in its bottom-left corner — use it to cross-check feature
sizes. Round lengths and positions to the nearest 5 mm.

Cross-check before emitting the corrected JSON: compute the visible
spans your refined description WOULD produce
(``visible_mm = object_extent_mm * 1.2``) and confirm they match the
ORIGINAL grid's labels within 5 mm. If they don't, you misread the
labels — fix it before emitting.

Reason out loud first: list each visible difference between the two
grids, then write the corrected JSON wrapped between the markers.

Output format:

{JSON_BEGIN}
{{ ...corrected PartDescription JSON... }}
{JSON_END}

Set "confidence" to "high" only if you are now fully confident the
description matches the original. Otherwise "medium" or "low", and
note remaining uncertainty in the "notes" field.
"""


USER_PROMPT = (
    "Examine all six orthographic views of this part.\n\n"
    "First, briefly describe what you see in each of the six views, one\n"
    "or two short lines per view (top, right, front, bottom, left, back)\n"
    "— focus on overall silhouette, feature count, and feature locations.\n\n"
    "Then count the features one final time.\n\n"
    f"Then emit the part description as JSON wrapped exactly between\n"
    f"{JSON_BEGIN} and {JSON_END} markers, with nothing after the\n"
    f"closing marker."
)


REFINE_USER_PROMPT_TEMPLATE = (
    "The FIRST image is the ORIGINAL part.\n"
    "The SECOND image is the RECONSTRUCTION built from the JSON below.\n\n"
    "Current JSON description:\n"
    "{current_json}\n\n"
    "Step 1: For each of the six views, describe in 1-2 lines what is\n"
    "different between the original and the reconstruction (missing,\n"
    "extra, wrong count, wrong size, wrong position).\n\n"
    "Step 2: Emit a CORRECTED JSON description wrapped exactly between\n"
    f"{JSON_BEGIN} and {JSON_END} markers, with nothing after the\n"
    "closing marker."
)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _last_balanced_json_object(text: str) -> str:
    """Return the last syntactically valid balanced ``{...}`` block in text.

    Used as a fallback when the JSON delimiter markers are missing from
    Claude's response. Walks left-to-right, attempts ``raw_decode`` at
    every ``{``, keeps the most recent successful decode.
    """
    decoder = json.JSONDecoder()
    last: str | None = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                _, end = decoder.raw_decode(text[i:])
                last = text[i : i + end]
                i += end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    if last is None:
        raise ValueError(
            "No JSON object could be parsed from the model response:\n"
            + text
        )
    return last


def _extract_json(text: str) -> dict:
    """Pull the JSON payload out of Claude's response.

    Prefers content between ``===JSON_BEGIN===`` and ``===JSON_END===``
    markers; falls back to the last balanced ``{...}`` if the markers
    are missing.
    """
    begin = text.find(JSON_BEGIN)
    end = text.find(JSON_END, begin + 1) if begin != -1 else -1
    if begin != -1 and end != -1:
        candidate = text[begin + len(JSON_BEGIN) : end].strip()
        # Trim leading "json" hint / code fence remnants if Claude added them.
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
            if candidate.endswith("```"):
                candidate = candidate.rsplit("```", 1)[0]
        return json.loads(candidate)
    return json.loads(_last_balanced_json_object(text))


def _validate(data: dict) -> PartDescription:
    try:
        return PartDescription(**data)
    except ValidationError as exc:
        raise ValueError(
            "PartDescription validation failed.\n"
            f"Raw data: {data}\n"
            f"Pydantic error: {exc}"
        ) from exc


def warn_if_uncertain(part: PartDescription, label: str = "vision") -> None:
    """Emit a stderr warning if the model itself flagged uncertainty."""
    if part.confidence == "high":
        return
    marker = "!!" if part.confidence == "low" else "**"
    print(
        f"{marker} {label}: confidence={part.confidence}"
        + (f"  notes={part.notes!r}" if part.notes else ""),
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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


def call_claude(image_path: str) -> PartDescription:
    """Initial pass: send one grid PNG to Claude, return a validated PartDescription.

    Raises ``FileNotFoundError`` if the image is missing, ``ValueError``
    if Claude returns non-JSON or JSON that fails Pydantic validation,
    and propagates ``anthropic.APIError`` subclasses on transport
    failures so the caller can decide whether to retry.
    """
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
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    return _validate(_extract_json(text))


def call_claude_refine(
    original_image_path: str,
    reconstruction_image_path: str,
    current: PartDescription,
) -> PartDescription:
    """Refinement pass: show Claude the original + the reconstruction
    rendered from ``current``, get back a corrected PartDescription.

    The model receives two images in one user turn — the order matters
    and is reflected in the user prompt ("FIRST image" / "SECOND image").
    """
    user_text = REFINE_USER_PROMPT_TEMPLATE.format(
        current_json=current.model_dump_json(indent=2)
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=REFINE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    _encode_image(original_image_path),
                    _encode_image(reconstruction_image_path),
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    return _validate(_extract_json(text))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_call(image_path: str) -> None:
    """Run ``call_claude`` on ``image_path`` and pretty-print the result."""
    result = call_claude(image_path)

    base = result.base
    print(
        f"Base: shape={base.shape} "
        f"width={base.width_mm}mm depth={base.depth_mm}mm "
        f"height={base.height_mm}mm sketch_plane={base.sketch_plane}"
    )

    print(f"Features: {len(result.features)} found")
    for i, feat in enumerate(result.features, start=1):
        if feat.shape == "circle":
            dims = f"diameter={feat.diameter_mm}mm"
        else:
            dims = f"width={feat.width_mm}mm depth={feat.depth_mm}mm"
        print(
            f"  [{i}] type={feat.type} face={feat.face} shape={feat.shape} "
            f"{dims} height={feat.height_mm}mm "
            f"depth_type={feat.depth_type} "
            f"position=({feat.position_x}, {feat.position_y})"
        )

    print(f"Confidence: {result.confidence}")
    print(f"Notes: {result.notes}")
    warn_if_uncertain(result, "test_call")

    print()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m backend.ai_infra.llm_client path/to/grid.png",
            file=sys.stderr,
        )
        sys.exit(2)
    test_call(sys.argv[1])
