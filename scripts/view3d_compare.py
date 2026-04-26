#!/usr/bin/env python3
"""Open the input STL and the rebuilt STL side-by-side in one window.

Two viewports inside a single PyVista window with cameras LINKED — rotate
or zoom either side and the other follows, so you can eyeball geometric
differences without manually aligning two windows.

The script resolves both files automatically from a single stem:

    INPUT  -> backend/outputs/deepcad_selected_stl/<stem>.stl
    OUTPUT -> backend/outputs/face_<stem>.stl

If the OUTPUT STL is missing, the script tells you which command will
generate it instead of opening a half-empty window.

Usage:
    # Pass either the stem or any path that contains it:
    python scripts/view3d_compare.py deepcadimg_000017
    python scripts/view3d_compare.py backend/outputs/deepcad_selected_stl/deepcadimg_000017.stl

    # Pass two explicit paths if you want to compare arbitrary STLs:
    python scripts/view3d_compare.py path/to/a.stl path/to/b.stl

Controls (PyVista trackball default):
    Left-click + drag   -> rotate
    Right-click + drag  -> pan
    Scroll wheel        -> zoom
    R                   -> reset camera

Lighting matches backend.pipeline.stl_renderer so the shading you see
here is the same as what the offscreen 6-view renderer captures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_selected_stl"
OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"

INPUT_COLOR = "#AAAAAA"   # neutral grey — same as view3d.py
OUTPUT_COLOR = "#A8C8FF"  # cool blue — easy to tell at a glance which pane is the rebuild
BG_COLOR = "white"

# face_extractor normalises every input mesh to a 100 mm longest edge
# before extraction, so the rebuilt STL lives at this scale. The
# original DeepCAD STLs are at arbitrary export scales (30 mm, 200 mm,
# etc.). Without normalising at display time, the linked camera frames
# one pane correctly and makes the other look tiny / huge. Match the
# extractor's rule so both panes render at the same physical size.
DISPLAY_NORMALISE_MM = 100.0


def _setup_lights(plotter: pv.Plotter) -> None:
    """Three-point rig identical to backend.pipeline.stl_renderer and view3d.py."""
    plotter.remove_all_lights()

    key = pv.Light(position=(1.0, 1.0, 2.0), focal_point=(0.0, 0.0, 0.0))
    key.intensity = 0.8
    plotter.add_light(key)

    fill = pv.Light(position=(-1.0, -0.5, 0.5), focal_point=(0.0, 0.0, 0.0))
    fill.intensity = 0.3
    plotter.add_light(fill)

    ambient = pv.Light(light_type="headlight")
    ambient.intensity = 0.2
    plotter.add_light(ambient)


def _resolve_pair(arg: str) -> tuple[Path, Path]:
    """Return (input_stl, output_stl) for a stem-or-path argument.

    Accepts:
        - bare stem            "deepcadimg_000017"
        - input path           ".../deepcad_selected_stl/deepcadimg_000017.stl"
        - output path          ".../face_deepcadimg_000017.stl"
    All three forms map to the same canonical pair.
    """
    p = Path(arg)
    name = p.name if p.suffix else arg
    stem = name[:-4] if name.endswith(".stl") else name
    if stem.startswith("face_"):
        stem = stem[len("face_"):]

    return (
        INPUT_DIR / f"{stem}.stl",
        OUTPUT_DIR / f"face_{stem}.stl",
    )


def _normalise_for_display(mesh: pv.PolyData) -> tuple[pv.PolyData, float, float]:
    """Center the mesh at the origin and scale longest bbox edge to
    DISPLAY_NORMALISE_MM. Returns (normalised_mesh, original_longest_mm, scale_applied)
    so the panel label can show the true physical size."""
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    extents = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    longest = max(extents)
    if longest <= 0:
        return mesh, 0.0, 1.0
    scale = DISPLAY_NORMALISE_MM / longest
    centered = mesh.translate(
        (
            -(bounds[0] + bounds[1]) / 2.0,
            -(bounds[2] + bounds[3]) / 2.0,
            -(bounds[4] + bounds[5]) / 2.0,
        ),
        inplace=False,
    )
    scaled = centered.scale(scale, inplace=False)
    return scaled, longest, scale


def _add_panel(
    plotter: pv.Plotter,
    row: int,
    col: int,
    mesh_path: Path,
    label: str,
    color: str,
    *,
    normalise: bool,
    smooth_shading: bool,
) -> None:
    plotter.subplot(row, col)
    plotter.set_background(BG_COLOR)
    _setup_lights(plotter)
    mesh = pv.read(str(mesh_path))

    if normalise:
        mesh, original_mm, _scale = _normalise_for_display(mesh)
        size_note = f"  [original longest edge: {original_mm:.1f} mm → shown at {DISPLAY_NORMALISE_MM:.0f} mm]"
    else:
        bx = mesh.bounds
        original_mm = max(bx[1] - bx[0], bx[3] - bx[2], bx[5] - bx[4])
        size_note = f"  [longest edge: {original_mm:.1f} mm  (raw scale)]"

    # Default smooth_shading=False: per-triangle flat shading matches CAD
    # facet structure and avoids fake radial banding on coarse tessellated
    # cylinders / hex faces. Pass --smooth for interpolated normals.
    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=smooth_shading,
        ambient=0.15,
        diffuse=0.85,
        specular=0.0,
    )
    plotter.add_text(label + size_note, position="upper_edge", font_size=11, color="black")
    plotter.add_axes()
    plotter.enable_trackball_style()
    plotter.reset_camera()


def view_pair(
    input_stl: Path,
    output_stl: Path,
    *,
    show: bool = True,
    normalise: bool = True,
    smooth_shading: bool = False,
) -> None:
    plotter = pv.Plotter(shape=(1, 2), window_size=(2000, 900), off_screen=not show)

    _add_panel(plotter, 0, 0, input_stl,  f"INPUT  —  {input_stl.name}",
               INPUT_COLOR,  normalise=normalise, smooth_shading=smooth_shading)
    _add_panel(plotter, 0, 1, output_stl, f"OUTPUT —  {output_stl.name}",
               OUTPUT_COLOR, normalise=normalise, smooth_shading=smooth_shading)

    plotter.link_views()
    plotter.reset_camera()

    if show:
        plotter.show()
    else:
        plotter.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open the input STL and the face-pipeline rebuilt STL side-by-side "
            "in one window with synchronised cameras."
        ),
    )
    parser.add_argument(
        "args",
        nargs="+",
        help=(
            "Either ONE arg (a stem like 'deepcadimg_000017' or any path "
            "containing it), or TWO arbitrary STL paths to compare directly."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Construct the plotter but don't open a window (smoke-test only).",
    )
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help=(
            "Show meshes at their raw on-disk scale (do NOT normalise to "
            f"{DISPLAY_NORMALISE_MM:.0f} mm longest edge). Useful for spotting "
            "scale mismatches; default is to normalise so both panes line up."
        ),
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help=(
            "Use smooth (interpolated) vertex normals — can look nicer on "
            "organic shapes but often shows fake radial bands on faceted "
            "CAD meshes. Default is flat per-triangle shading."
        ),
    )
    ns = parser.parse_args(argv[1:])

    if len(ns.args) == 1:
        input_stl, output_stl = _resolve_pair(ns.args[0])
    elif len(ns.args) == 2:
        input_stl = Path(ns.args[0]).resolve()
        output_stl = Path(ns.args[1]).resolve()
    else:
        print("Pass either ONE stem/path, or TWO explicit STL paths.", file=sys.stderr)
        return 2

    if not input_stl.is_file():
        print(f"[error] input STL not found: {input_stl}", file=sys.stderr)
        return 2

    if not output_stl.is_file():
        stem = input_stl.stem
        print(f"[error] rebuilt STL not found: {output_stl}", file=sys.stderr)
        print(file=sys.stderr)
        print("Generate it first with:", file=sys.stderr)
        print(f"    python scripts/face_roundtrip.py {input_stl}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"...then re-run:    python scripts/view3d_compare.py {stem}", file=sys.stderr)
        return 2

    print(f"INPUT : {input_stl.relative_to(REPO_ROOT) if str(input_stl).startswith(str(REPO_ROOT)) else input_stl}")
    print(f"OUTPUT: {output_stl.relative_to(REPO_ROOT) if str(output_stl).startswith(str(REPO_ROOT)) else output_stl}")
    print("(cameras are linked — rotate one pane and the other follows)")
    if ns.raw_scale:
        print("(raw scale — input and output may differ in physical size)")
    else:
        print(f"(both meshes normalised to {DISPLAY_NORMALISE_MM:.0f} mm longest edge for fair shape comparison)")
    if ns.smooth:
        print("(smooth shading — try without --smooth if you see odd radial patterns)")

    view_pair(
        input_stl, output_stl,
        show=not ns.no_show,
        normalise=not ns.raw_scale,
        smooth_shading=ns.smooth,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
