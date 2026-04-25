"""ExtractedGeometry -> clean 2400x1000 technical face diagram PNG.

This is the second stage of the face-geometry pipeline. The whole
point of the new approach is that Claude no longer has to read pixel
photos of a 3D part — instead it sees clean 2D engineering drawings
of the actual mesh faces, with the underlying geometry attached as
JSON. So the renderer here is deliberately minimalist: filled
polygons, dashed circles for holes, face IDs, and a scale bar. No
shading, no perspective, no decorative chrome.

Layout matches the 6-view PNG grid that ``stl_renderer.py`` produces
so a side-by-side comparison in ``scripts/face_roundtrip.py`` can
just paste the two images next to each other:

    Row 1: +Z     +X     +Y
    Row 2: -Z     -X     -Y

Final canvas is 2400x1000 px to match.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .face_extractor import (
    CylindricalFace,
    ExtractedGeometry,
    PlanarFace,
    PrincipalAxis,
)


# ---------------------------------------------------------------------------
# Layout constants — all in pixels because matplotlib + PIL play nicely
# at 100 dpi -> 1 inch == 100 px.
# ---------------------------------------------------------------------------
CANVAS_W_PX = 2400
CANVAS_H_PX = 1000
DPI = 100
HEADER_PX = 60
FOOTER_PX = 60

PANEL_DIRECTIONS: list[list[PrincipalAxis]] = [
    ["+Z", "+X", "+Y"],
    ["-Z", "-X", "-Y"],
]

# Distinct, accessible colour palette (matplotlib tab10) recycled
# round-robin for face fills. Strong borders ensure even faint pastels
# remain readable on the white canvas.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# Padding around the bounding box of the panel content.
PANEL_PADDING_FRAC = 0.10


# ---------------------------------------------------------------------------
# Helpers — group faces by panel, project cylinders onto a plane, etc.
# ---------------------------------------------------------------------------
def _planar_uv_axes_for_direction(direction: PrincipalAxis) -> tuple[str, str]:
    """Match ``face_extractor._planar_uv_axes`` so axis labels read true."""
    if direction in ("+Z", "-Z"):
        return "X", "Y"
    if direction in ("+X", "-X"):
        return "Y", "Z"
    if direction in ("+Y", "-Y"):
        return "X", "Z"
    raise ValueError(direction)


def _project_centre_to_plane(
    centre: tuple[float, float, float],
    direction: PrincipalAxis,
) -> tuple[float, float]:
    """Drop the world coordinate that's perpendicular to the panel's plane."""
    x, y, z = centre
    if direction in ("+Z", "-Z"):
        return x, y
    if direction in ("+X", "-X"):
        return y, z
    if direction in ("+Y", "-Y"):
        return x, z
    raise ValueError(direction)


def _planar_world_vertices(face: PlanarFace) -> list[tuple[float, float]]:
    """Lift the face's local-2D vertices back to the world UV plane.

    ``face_extractor`` stores ``vertices_2d_mm`` *relative to the face
    centroid* (so a 100x50 mm face centred at world (10, -5) emits
    vertices like (-50, -25)..(+50, +25)). For the diagram we want to
    plot vertices in the *world* UV coords of the panel so that
    multiple faces in the same panel land in the right spots relative
    to one another.
    """
    cu, cv = _project_centre_to_plane(face.centre_3d_mm, face.normal.direction)
    return [(cu + u, cv + v) for u, v in face.vertices_2d_mm]


def _group_planar(
    faces: Iterable[PlanarFace],
) -> dict[PrincipalAxis, list[PlanarFace]]:
    out: dict[PrincipalAxis, list[PlanarFace]] = defaultdict(list)
    for f in faces:
        out[f.normal.direction].append(f)
    return out


def _group_cylinders_by_panel(
    cylinders: Iterable[CylindricalFace],
) -> dict[PrincipalAxis, list[CylindricalFace]]:
    """A cylinder along axis ``+Z`` should appear in BOTH the +Z and
    -Z panels (it's visible from above and below)."""
    out: dict[PrincipalAxis, list[CylindricalFace]] = defaultdict(list)
    for c in cylinders:
        for sign in ("+", "-"):
            label = f"{sign}{c.axis_direction[1]}"  # "+Z" -> "+Z"/"-Z"
            out[label].append(c)  # type: ignore[arg-type]
    return out


def _panel_extents(
    direction: PrincipalAxis,
    planar: list[PlanarFace],
    cylinders: list[CylindricalFace],
    overall_bbox: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Returns ((u_min, u_max), (v_min, v_max)) for the panel.

    Falls back to the part's overall bbox when the panel has no faces,
    so empty panels still show the *part scale* not zoom-to-nothing.
    """
    points: list[tuple[float, float]] = []
    for f in planar:
        points.extend(_planar_world_vertices(f))
    for c in cylinders:
        cu, cv = _project_centre_to_plane(c.centre_3d_mm, direction)
        r = c.radius_mm
        points.extend([(cu - r, cv - r), (cu + r, cv + r)])
    if not points:
        bw, bd, bh = overall_bbox
        if direction in ("+Z", "-Z"):
            return (-bw / 2, bw / 2), (-bd / 2, bd / 2)
        if direction in ("+X", "-X"):
            return (-bd / 2, bd / 2), (-bh / 2, bh / 2)
        return (-bw / 2, bw / 2), (-bh / 2, bh / 2)
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)
    pad_u = max((u_max - u_min) * PANEL_PADDING_FRAC, 1.0)
    pad_v = max((v_max - v_min) * PANEL_PADDING_FRAC, 1.0)
    return (u_min - pad_u, u_max + pad_u), (v_min - pad_v, v_max + pad_v)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_panel(
    ax: Axes,
    direction: PrincipalAxis,
    planar: list[PlanarFace],
    cylinders: list[CylindricalFace],
    overall_bbox: tuple[float, float, float],
) -> None:
    u_label, v_label = _planar_uv_axes_for_direction(direction)
    (u_min, u_max), (v_min, v_max) = _panel_extents(
        direction, planar, cylinders, overall_bbox
    )

    if not planar and not cylinders:
        ax.set_facecolor("#f1f1f1")
        ax.text(
            0.5, 0.5, f"{direction}\n(no faces)",
            ha="center", va="center",
            color="#7a7a7a", fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Filled planar faces
    for i, face in enumerate(planar):
        verts = _planar_world_vertices(face)
        colour = PALETTE[i % len(PALETTE)]
        patch = mpatches.Polygon(
            verts, closed=True, facecolor=colour, alpha=0.40,
            edgecolor=colour, linewidth=2.0, joinstyle="miter",
        )
        ax.add_patch(patch)
        cu, cv = _project_centre_to_plane(face.centre_3d_mm, direction)
        ax.text(
            cu, cv, f"F{face.face_id}",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222",
        )
        ax.text(
            cu, cv - (v_max - v_min) * 0.025,
            f"{face.area_mm2:.0f} mm²",
            ha="center", va="top", fontsize=8, color="#444",
        )

    # Cylinders — bosses solid, holes dashed
    for c in cylinders:
        cu, cv = _project_centre_to_plane(c.centre_3d_mm, direction)
        if c.face_type == "boss":
            patch = mpatches.Circle(
                (cu, cv), c.radius_mm,
                facecolor="#000000", alpha=0.20,
                edgecolor="#000000", linewidth=2.0,
            )
            label = f"F{c.face_id} boss\nr={c.radius_mm:.1f}"
        else:
            patch = mpatches.Circle(
                (cu, cv), c.radius_mm,
                facecolor="none",
                edgecolor="#c0392b", linewidth=2.0, linestyle="--",
            )
            label = f"F{c.face_id} hole\nr={c.radius_mm:.1f}"
        ax.add_patch(patch)
        ax.text(
            cu, cv, label,
            ha="center", va="center", fontsize=8, color="#222",
        )

    # Axis decorations
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_min, v_max)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=-1)
    ax.set_title(_panel_title(direction), fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{u_label} (mm)")
    ax.set_ylabel(f"{v_label} (mm)")
    ax.tick_params(labelsize=8)

    _draw_scale_bar(ax, u_min, u_max, v_min)


def _panel_title(direction: PrincipalAxis) -> str:
    name = {
        "+Z": "+Z (top)",
        "-Z": "-Z (bottom)",
        "+X": "+X (right)",
        "-X": "-X (left)",
        "+Y": "+Y (front)",
        "-Y": "-Y (back)",
    }[direction]
    return name


def _draw_scale_bar(
    ax: Axes,
    u_min: float, u_max: float, v_min: float,
) -> None:
    """A short horizontal "10mm" or "20mm" bar in the lower-left.

    Picks a length that's a "clean" multiple (1/2/5 × 10^k) and
    roughly 15-25 % of the panel width.
    """
    target = (u_max - u_min) * 0.20
    candidates = []
    for exp in range(-1, 4):
        for mant in (1.0, 2.0, 5.0):
            candidates.append(mant * (10 ** exp))
    bar_len = min(candidates, key=lambda v: abs(v - target))

    bar_u0 = u_min + (u_max - u_min) * 0.05
    bar_v0 = v_min + (u_max - u_min) * 0.04
    ax.plot(
        [bar_u0, bar_u0 + bar_len], [bar_v0, bar_v0],
        color="#222", linewidth=3,
    )
    ax.text(
        bar_u0 + bar_len / 2, bar_v0 + (u_max - u_min) * 0.015,
        f"{bar_len:g} mm",
        ha="center", va="bottom", fontsize=8, color="#222",
    )


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------
def render_face_diagrams(
    geometry: ExtractedGeometry,
    output_path: str | Path,
    part_id: str = "unknown",
) -> str:
    """Compose a 2400x1000 6-panel face diagram and the sibling JSON.

    Saves:
      * ``output_path`` — the PNG diagram.
      * ``output_path.with_suffix('.json')`` — the full
        ``ExtractedGeometry`` so ``face_llm_client`` can attach exact
        numbers to the prompt.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    planar_by_dir = _group_planar(geometry.planar_faces)
    cyl_by_dir = _group_cylinders_by_panel(geometry.cylindrical_faces)

    fig = plt.figure(
        figsize=(CANVAS_W_PX / DPI, CANVAS_H_PX / DPI),
        dpi=DPI,
        facecolor="white",
    )

    # GridSpec: header row + 2 panel rows + footer row.
    # Header / footer get fixed pixel heights.
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        height_ratios=[HEADER_PX, (CANVAS_H_PX - HEADER_PX - FOOTER_PX) / 2,
                       (CANVAS_H_PX - HEADER_PX - FOOTER_PX) / 2, FOOTER_PX],
        left=0.04, right=0.99, top=0.99, bottom=0.01,
        hspace=0.35, wspace=0.18,
    )

    # Header
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(
        0.01, 0.5,
        f"FACE GEOMETRY DIAGRAM — {part_id}",
        ha="left", va="center", fontsize=18, fontweight="bold",
    )
    header_ax.text(
        0.99, 0.5,
        f"{geometry.face_count} faces  •  bbox "
        f"{geometry.bounding_box_mm[0]:.1f} × "
        f"{geometry.bounding_box_mm[1]:.1f} × "
        f"{geometry.bounding_box_mm[2]:.1f} mm  •  longest "
        f"{geometry.longest_dimension_mm:.1f} mm",
        ha="right", va="center", fontsize=11, color="#444",
    )

    # Panels
    for r, row in enumerate(PANEL_DIRECTIONS):
        for c, direction in enumerate(row):
            ax = fig.add_subplot(gs[r + 1, c])
            _draw_panel(
                ax, direction,
                planar_by_dir.get(direction, []),
                cyl_by_dir.get(direction, []),
                geometry.bounding_box_mm,
            )

    # Footer / legend
    foot_ax = fig.add_subplot(gs[3, :])
    foot_ax.axis("off")
    foot_ax.text(
        0.01, 0.7,
        "Filled polygon = planar face   "
        "Solid grey circle = cylindrical boss   "
        "Dashed red circle = cylindrical hole",
        ha="left", va="center", fontsize=11,
    )
    foot_ax.text(
        0.01, 0.25,
        "Longest part dimension normalised to 100 mm. All distances "
        "are exact mm (extracted from mesh, not estimated from pixels).",
        ha="left", va="center", fontsize=10, color="#444",
    )

    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)

    # Sibling JSON: identical stem, .json suffix.
    json_out = out.with_suffix(".json")
    json_out.write_text(geometry.model_dump_json(indent=2))

    return str(out)


# ---------------------------------------------------------------------------
# CLI: python -m backend.ai_infra.face_diagram_renderer mesh.stl out.png
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    from .face_extractor import extract_faces

    p = argparse.ArgumentParser(
        description="Render an STL's face geometry as a 2400x1000 diagram."
    )
    p.add_argument("mesh", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--part-id", default=None,
        help="Override the part id shown in the header (defaults to mesh stem).",
    )
    args = p.parse_args()

    pid = args.part_id or args.mesh.stem
    geom = extract_faces(args.mesh)
    out = render_face_diagrams(geom, args.out, part_id=pid)
    print(f"wrote {out}")
    print(f"wrote {Path(out).with_suffix('.json')}")


if __name__ == "__main__":
    _main()
