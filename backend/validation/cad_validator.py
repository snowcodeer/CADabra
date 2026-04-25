from __future__ import annotations

from pathlib import Path

import open3d as o3d

from models import ValidationResult


def validate_outputs(step_path: str | Path, stl_path: str | Path) -> ValidationResult:
    step_path = Path(step_path)
    stl_path = Path(stl_path)

    if not step_path.exists() or step_path.stat().st_size == 0:
        return ValidationResult(
            executes=False,
            produces_solid=False,
            is_watertight=False,
            vertex_count=None,
            face_count=None,
            bounding_box_mm=None,
            error="STEP file missing or empty",
        )

    if not stl_path.exists() or stl_path.stat().st_size == 0:
        return ValidationResult(
            executes=False,
            produces_solid=False,
            is_watertight=False,
            vertex_count=None,
            face_count=None,
            bounding_box_mm=None,
            error="STL file missing or empty",
        )

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    executes = mesh.has_triangles()
    vertex_count = len(mesh.vertices) if executes else 0
    face_count = len(mesh.triangles) if executes else 0
    produces_solid = executes and vertex_count > 0 and face_count > 0
    is_watertight = bool(mesh.is_watertight()) if produces_solid else False

    bbox = None
    if produces_solid:
        aabb = mesh.get_axis_aligned_bounding_box()
        ext = aabb.get_extent()
        bbox = (float(ext[0]), float(ext[1]), float(ext[2]))

    return ValidationResult(
        executes=executes,
        produces_solid=produces_solid,
        is_watertight=is_watertight,
        vertex_count=vertex_count,
        face_count=face_count,
        bounding_box_mm=bbox,
        error=None,
    )
