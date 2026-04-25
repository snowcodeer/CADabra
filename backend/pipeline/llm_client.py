from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from anthropic import Anthropic

from models import LLMReasoningOutput

SYSTEM_PROMPT = """You are a mechanical CAD expert. You will be shown 6 orthographic views of a 3D part rendered from a point cloud.
Each view has two panels: RGB surface rendering (left) and depth map (right, blue=far, red=near).
Views shown: Top, Bottom, Front, Back, Right, Left.
The part bounding box is provided. All measurements are in millimetres.

Your task:
1. Analyse each view pair to understand the geometry
2. Identify the base body and all features (extrudes, cuts, bosses, pockets)
3. Infer sketch planes and extrude heights by comparing depth maps across adjacent views
4. Write complete, executable CadQuery Python code that reconstructs the part

Rules for CadQuery output:
- Use only sketch-and-extrude operations (no loft, sweep, or shell)
- All dimensions must be numeric literals in mm, not variables
- Start from: result = cq.Workplane(\"XY\")
- End with: show_object(result) on the last line
- The code must be a single self-contained block, no functions, no classes
- Import only: import cadquery as cq

Return your response as valid JSON matching this schema exactly:
{
  \"reasoning\": \"<your step-by-step analysis of what you see in each view>\",
  \"features\": [
    {
      \"type\": \"extrude|cut|pocket|boss\",
      \"sketch_plane\": \"XY|XZ|YZ\",
      \"profile_description\": \"<brief description>\",
      \"height_mm\": <number>,
      \"position_mm\": [<x>, <y>, <z>]
    }
  ],
  \"cadquery_code\": \"<complete python code>\",
  \"confidence\": \"high|medium|low\"
}

Return ONLY the JSON. No preamble, no markdown fences, no explanation outside the JSON."""


def _extract_text_block(response) -> str:
    parts = []
    for block in response.content:
        txt = getattr(block, "text", None)
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def infer_cadquery(
    render_grid_path: str | Path,
    bbox_mm: dict[str, float],
) -> LLMReasoningOutput:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    render_grid_path = Path(render_grid_path)
    image_b64 = base64.b64encode(render_grid_path.read_bytes()).decode("utf-8")

    client = Anthropic(api_key=api_key)
    user_text = (
        f"Part bounding box: X={bbox_mm['x_mm']}mm, Y={bbox_mm['y_mm']}mm, Z={bbox_mm['z_mm']}mm. "
        "Reconstruct this part."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
            ],
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    payload_text = _extract_text_block(response)
    try:
        return LLMReasoningOutput.model_validate(json.loads(payload_text))
    except Exception:
        retry_messages = messages + [
            {
                "role": "user",
                "content": "Your previous response was not valid JSON. Return only the JSON object.",
            }
        ]
        retry_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=retry_messages,
        )
        retry_text = _extract_text_block(retry_response)
        return LLMReasoningOutput.model_validate(json.loads(retry_text))
