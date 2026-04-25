"""STL mesh -> exact face geometry, no OpenCV, no re-rendered images.

The face-geometry-approach branch's foundation. Replaces the
``contour_extractor.py`` pipeline (which OCR-style guessed 2D
geometry from rendered PNGs) with direct geometric extraction from
the actual mesh.

Real-world wrinkle the FACE_GEOMETRY_MIGRATION_PROMPT.md skips
=============================================================
The prompt says:

    "Loading strategy: ... If .stl file: use CadQuery's Shape import
     ...  shape = Shape.importBrep(path)"

That path does not exist. STL is a tessellated *triangle soup* with
no notion of B-Rep faces, so CadQuery's ``face.geomType()`` will not
return ``"PLANE"`` or ``"CYLINDER"`` for an STL-loaded shape — every
"face" is a single triangle. We have to *recover* logical faces from
adjacency + normal clustering ourselves.

Algorithm
---------
1. Load the mesh with ``open3d.io.read_triangle_mesh`` and compute
   per-triangle normals + areas.
2. **Triangle clustering** — region-grow over the triangle adjacency
   graph: two triangles join the same cluster when they share an
   edge AND their normals agree within ``PLANAR_NORMAL_TOL_DEG``.
3. For each cluster decide what kind of face it represents:

   * **Planar**: the cluster's average normal is within
     ``PRINCIPAL_AXIS_DOT_MIN`` of one of the six principal axes
     (±X / ±Y / ±Z). We extract the boundary polygon (edges that
     appear in only one triangle of the cluster) and project to 2D.
   * **Cylindrical**: the cluster's per-triangle normals fan around
     a single principal axis (the cluster has a clear axis-aligned
     "spin direction"). We recover the axis position, radius, height
     and classify boss vs hole by checking whether the cluster
     normals point away from or toward the cluster centroid axis.
   * **Other**: skipped for the MVP.

4. Filter noise (clusters whose total area is < a fraction of the
   largest cluster's area).
5. Normalise everything so the longest bbox dimension = 100 mm —
   matches what ``stl_renderer.py`` does so the comparison render
   lands at the same scale.

The output is a fully self-contained ``ExtractedGeometry`` that
``face_diagram_renderer`` and ``face_llm_client`` can consume
without ever touching the original mesh again.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
import open3d as o3d
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Two adjacent triangles join the same cluster iff their normals agree
# to within this many degrees. Small enough that a sharp 90° edge is a
# hard cluster boundary; big enough that floating-point noise on a
# perfectly flat face does not explode into many micro-clusters.
PLANAR_NORMAL_TOL_DEG = 5.0

# A cluster's average normal is "axis-aligned" if its dot product with
# the closest principal axis is at least this. cos(18°) ≈ 0.95 — the
# prompt's suggested value.
PRINCIPAL_AXIS_DOT_MIN = 0.95

# Drop clusters whose total area is below this fraction of the largest
# cluster's area. Filters tessellation seams and slivers.
MIN_AREA_FRAC = 0.005

# Pass-2 normal tolerance: must be larger than the per-step normal
# difference of a tessellated cylinder (~5–11° on a typical 32-to-64
# facet hole) but smaller than 90° so cylinder walls do NOT bleed
# into the adjacent flat top/bottom faces. 20° is a comfortable
# middle ground.
CYLINDER_PASS_TOL_DEG = 20.0

# Cluster considered cylindrical if its per-triangle normals span at
# least this much around a candidate axis (i.e. the surface really
# wraps, not just two flats meeting at a sharp angle).
CYL_MIN_ANGULAR_SPAN_DEG = 30.0

# Cylinder axis must be near-perpendicular to the per-triangle normals.
# If the average |dot(normal, axis_candidate)| stays below this, we
# accept the candidate axis.
CYL_AXIS_NORMAL_DOT_MAX = 0.20

# After the part is normalised so longest_dim == 100 mm, anything
# below this is treated as "could be either boss or hole, default to
# boss" — purely a safety cap, never hit on real CAD data.
NORMALISE_LONGEST_MM = 100.0

PRINCIPAL_AXES: list[Tuple[str, np.ndarray]] = [
    ("+X", np.array([1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, 1.0])),
    ("-Z", np.array([0.0, 0.0, -1.0])),
]

PrincipalAxis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


# ---------------------------------------------------------------------------
# Pydantic models — the contract with face_diagram_renderer and
# face_llm_client. Every value is exact mm in the post-normalisation
# frame; downstream code must never re-scale.
# ---------------------------------------------------------------------------
class FaceNormal(BaseModel):
    direction: PrincipalAxis = Field(
        ...,
        description="Closest principal axis to the face normal.",
    )
    vector: Tuple[float, float, float] = Field(
        ...,
        description="Unit normal as it actually came out of the mesh "
        "(post-snap, but the raw vector pre-snap is logged here so "
        "the LLM can tell perfectly axis-aligned faces from slightly "
        "tilted ones).",
    )


class PlanarFace(BaseModel):
    """One axis-aligned planar face recovered from triangle clustering."""

    face_id: int = Field(..., description="0-based id, unique within the part.")
    normal: FaceNormal
    centre_3d_mm: Tuple[float, float, float]
    area_mm2: float
    vertices_2d_mm: list[Tuple[float, float]] = Field(
        ...,
        description="Boundary polygon projected onto the face plane. "
        "Origin = face centroid. U axis = the principal in-plane axis "
        "with the smaller world index (X<Y<Z), V = the other. So a "
        "+Z face uses (X, Y), a +X face uses (Y, Z), a +Y face uses "
        "(X, Z). Vertices are listed once around the loop, no repeated "
        "first vertex.",
    )
    bounding_box_mm: Tuple[float, float] = Field(
        ...,
        description="(U_extent_mm, V_extent_mm) of the boundary polygon.",
    )
    shape_type: Literal["rectangle", "polygon"] = Field(
        ...,
        description='"rectangle" iff 4 vertices AND area_ratio > 0.85 '
        '(face area / bbox area). Otherwise "polygon". Circles do not '
        "appear here — they are recovered as CylindricalFace.",
    )


class CylindricalFace(BaseModel):
    face_id: int
    axis_direction: PrincipalAxis = Field(
        ...,
        description="Principal axis the cylinder spins around. For a "
        "vertical hole this would be +Z (the axis points up out of the "
        "top face).",
    )
    radius_mm: float
    height_mm: float
    centre_3d_mm: Tuple[float, float, float] = Field(
        ...,
        description="World-mm position of the cylinder axis MIDPOINT — "
        "i.e. halfway up the cylinder, on the axis.",
    )
    face_type: Literal["boss", "hole"] = Field(
        ...,
        description='"boss" if triangle normals point AWAY from the '
        'axis (convex), "hole" if they point TOWARD it (concave).',
    )


class ExtractedGeometry(BaseModel):
    planar_faces: list[PlanarFace]
    cylindrical_faces: list[CylindricalFace]
    bounding_box_mm: Tuple[float, float, float] = Field(
        ...,
        description="Overall (X, Y, Z) extent in mm AFTER normalisation.",
    )
    longest_dimension_mm: float = Field(
        ...,
        description="Always close to 100 mm — we normalise the part to "
        "longest-edge=100 before any measurement is taken.",
    )
    face_count: int
    source_file: str


# ---------------------------------------------------------------------------
# Mesh loading + normalisation
# ---------------------------------------------------------------------------
def _load_normalised_mesh(mesh_path: Path) -> Tuple[o3d.geometry.TriangleMesh, float]:
    """Load an STL, weld its duplicate vertices, centre it, scale
    longest edge to ``NORMALISE_LONGEST_MM``.

    The vertex-merge step is critical. STL files store *triangle
    soup*: every triangle owns its own three vertices and there are
    no shared indices, so naive edge-adjacency on the raw mesh sees
    every edge as belonging to exactly one triangle and the cluster
    pass tops out at 1-triangle "faces". ``merge_close_vertices``
    deduplicates within a small tolerance so adjacent triangles share
    vertex indices and the cluster pass actually walks the surface.

    Returns ``(mesh, scale_factor)``. The scale factor is the
    multiplier that was applied — useful for the caller if it ever
    needs to map measurements back to the original coordinate system.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError(f"{mesh_path}: no triangles in mesh")
    # Tolerance ~ 0.1% of the model's smallest extent. STL precision
    # is single-float so adjacent triangles' "shared" vertices can
    # disagree by a few μm.
    extents = np.asarray(
        mesh.get_axis_aligned_bounding_box().get_extent(), dtype=float
    )
    weld_tol = max(float(np.min(extents[extents > 0]) * 1e-4), 1e-6)
    mesh = mesh.merge_close_vertices(weld_tol)
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    bbox = mesh.get_axis_aligned_bounding_box()
    extents = np.asarray(bbox.get_extent(), dtype=float)
    longest = float(np.max(extents))
    if longest <= 0:
        raise ValueError(f"{mesh_path}: degenerate mesh (zero extent)")
    scale = NORMALISE_LONGEST_MM / longest
    if scale != 1.0:
        mesh.scale(scale, center=bbox.get_center())
    return mesh, scale


# ---------------------------------------------------------------------------
# Triangle clustering by adjacency + normal agreement
# ---------------------------------------------------------------------------
def _build_edge_to_triangles(triangles: np.ndarray) -> dict[Tuple[int, int], list[int]]:
    """Map every undirected edge to the list of triangles containing it.

    A manifold mesh has exactly two triangles per edge; non-manifold
    edges (overlapping geometry, T-junctions) have more or fewer. The
    cluster step happily ignores edges with !=2 incidence, treating
    them as cluster boundaries — this keeps a slightly broken mesh
    from collapsing every face into one cluster.
    """
    edge_map: dict[Tuple[int, int], list[int]] = defaultdict(list)
    for ti, tri in enumerate(triangles):
        a, b, c = sorted((int(tri[0]), int(tri[1]), int(tri[2])))
        edge_map[(a, b)].append(ti)
        edge_map[(b, c)].append(ti)
        edge_map[(a, c)].append(ti)
    return edge_map


def _cluster_triangles(
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    tol_deg: float = PLANAR_NORMAL_TOL_DEG,
    candidate_mask: Optional[np.ndarray] = None,
) -> list[list[int]]:
    """Region-grow over edge-adjacent triangles.

    With the default ``tol_deg`` two adjacent triangles join the same
    cluster only if their normals also agree to within that many
    degrees — recovers planar facets. Set ``tol_deg=180.0`` to disable
    the normal check and group purely by adjacency, which is what the
    cylinder pass needs (a 40-facet cylinder has 9° steps between
    adjacent triangles, which would otherwise tear the cylinder into
    one cluster per triangle).

    ``candidate_mask`` (optional) restricts the BFS to triangles where
    the mask is True, leaving the others untouched. Used to run pass 2
    over only the triangles that pass 1 didn't claim.
    """
    edge_map = _build_edge_to_triangles(triangles)

    cos_tol = math.cos(math.radians(min(tol_deg, 179.9)))
    use_normal_check = tol_deg < 179.0

    # tri -> [neighbour tri ids that share an edge AND (optionally) agree on normal]
    neighbours: list[list[int]] = [[] for _ in range(len(triangles))]
    for tris in edge_map.values():
        if len(tris) < 2:
            continue
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                a, b = tris[i], tris[j]
                if use_normal_check:
                    dot = float(np.dot(triangle_normals[a], triangle_normals[b]))
                    if dot < cos_tol:
                        continue
                neighbours[a].append(b)
                neighbours[b].append(a)

    visited = [False] * len(triangles)
    if candidate_mask is not None:
        # Pre-mark non-candidates as visited so they're skipped.
        for i in range(len(triangles)):
            if not candidate_mask[i]:
                visited[i] = True

    clusters: list[list[int]] = []
    for start in range(len(triangles)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        cluster: list[int] = []
        while queue:
            t = queue.popleft()
            cluster.append(t)
            for n in neighbours[t]:
                if not visited[n]:
                    visited[n] = True
                    queue.append(n)
        clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Planar face: snap to axis, walk boundary loop, project to 2D
# ---------------------------------------------------------------------------
def _snap_to_principal_axis(
    normal: np.ndarray,
) -> Optional[Tuple[PrincipalAxis, np.ndarray]]:
    """Return (label, axis_unit_vector) if normal is within ~18° of an axis."""
    best_label: Optional[PrincipalAxis] = None
    best_axis: Optional[np.ndarray] = None
    best_dot = -1.0
    for label, axis in PRINCIPAL_AXES:
        d = float(np.dot(normal, axis))
        if d > best_dot:
            best_dot = d
            best_label = label  # type: ignore[assignment]
            best_axis = axis
    if best_dot < PRINCIPAL_AXIS_DOT_MIN or best_label is None or best_axis is None:
        return None
    return best_label, best_axis


def _planar_uv_axes(direction: PrincipalAxis) -> Tuple[np.ndarray, np.ndarray]:
    """The (U, V) world-axis pair we use to flatten a face to 2D.

    The convention matches ``backend.pipeline.stl_renderer.VIEW_AXES``
    so a +Z face's local (u, v) lines up with what the orthographic
    +Z render would show.
    """
    if direction in ("+Z", "-Z"):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if direction in ("+X", "-X"):
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if direction in ("+Y", "-Y"):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    raise ValueError(f"unknown principal axis: {direction}")


def _cluster_boundary_loops(
    cluster: list[int],
    triangles: np.ndarray,
) -> list[list[int]]:
    """Return ALL boundary loops of a planar cluster as ordered vertex
    chains. A face with N holes will have N+1 loops (one outer, N inner).

    A boundary edge appears in exactly one of the cluster's triangles.
    We walk those edges head-to-tail, then start a fresh walk on any
    remaining unvisited boundary edges to pick up additional loops.
    """
    edge_count: dict[Tuple[int, int], int] = defaultdict(int)
    for ti in cluster:
        a, b, c = (int(x) for x in triangles[ti])
        for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))):
            edge_count[e] += 1

    boundary_edges = {e for e, n in edge_count.items() if n == 1}
    if not boundary_edges:
        return []

    # Adjacency on boundary vertices
    nbrs: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        nbrs[a].append(b)
        nbrs[b].append(a)

    used: set[Tuple[int, int]] = set()

    def _take_edge(u: int, v: int) -> None:
        used.add(tuple(sorted((u, v))))  # type: ignore[arg-type]

    def _next(curr: int, prev: Optional[int]) -> Optional[int]:
        for n in nbrs[curr]:
            if n == prev:
                continue
            if tuple(sorted((curr, n))) in used:
                continue
            return n
        return None

    loops: list[list[int]] = []
    # Walk every loop. A loop ends when we either return to its start
    # or run out of unused outgoing edges.
    for start in sorted({v for e in boundary_edges for v in e}):
        # Skip if the vertex has no remaining unused edges.
        if all(tuple(sorted((start, n))) in used for n in nbrs[start]):
            continue
        loop = [start]
        prev: Optional[int] = None
        current = start
        for _ in range(len(boundary_edges) + 1):
            nxt = _next(current, prev)
            if nxt is None:
                break
            _take_edge(current, nxt)
            if nxt == start:
                break
            loop.append(nxt)
            prev = current
            current = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _simplify_collinear(
    polyline: list[Tuple[float, float]],
    tol_mm: float = 0.05,
) -> list[Tuple[float, float]]:
    """Drop intermediate vertices that lie on the same line as their
    neighbours within ``tol_mm``. After tessellation a flat edge often
    carries dozens of tessellation vertices; we want the corner points
    only so the LLM-facing JSON stays compact.
    """
    if len(polyline) < 3:
        return polyline
    out: list[Tuple[float, float]] = []
    n = len(polyline)
    for i in range(n):
        prev = np.asarray(polyline[(i - 1) % n], dtype=float)
        curr = np.asarray(polyline[i], dtype=float)
        nxt = np.asarray(polyline[(i + 1) % n], dtype=float)
        d1 = curr - prev
        d2 = nxt - curr
        # Cross product magnitude in 2D; small => collinear.
        cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
        seg_len = max(np.linalg.norm(d1), np.linalg.norm(d2), 1e-9)
        if cross / seg_len > tol_mm:
            out.append(tuple(curr))  # type: ignore[arg-type]
    if len(out) < 3:
        return polyline
    return out


def _planar_face_from_cluster(
    face_id: int,
    cluster: list[int],
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    triangle_areas: np.ndarray,
    vertices: np.ndarray,
) -> Optional[PlanarFace]:
    avg_normal = np.average(
        triangle_normals[cluster], axis=0, weights=triangle_areas[cluster]
    )
    norm = np.linalg.norm(avg_normal)
    if norm < 1e-9:
        return None
    avg_normal = avg_normal / norm
    snapped = _snap_to_principal_axis(avg_normal)
    if snapped is None:
        return None
    direction, axis = snapped

    loops = _cluster_boundary_loops(cluster, triangles)
    if not loops:
        return None

    u_axis, v_axis = _planar_uv_axes(direction)
    centre_3d = np.average(
        np.mean(vertices[triangles[cluster]], axis=1),
        axis=0,
        weights=triangle_areas[cluster],
    )

    def _project(loop_idx: list[int]) -> list[Tuple[float, float]]:
        relative = vertices[loop_idx] - centre_3d
        return [(float(np.dot(p, u_axis)), float(np.dot(p, v_axis))) for p in relative]

    # Pick the loop with the largest 2D bounding-box area as the OUTER
    # boundary. Inner loops (= holes punched through this face) reappear
    # as cylindrical features in the second pass, so we silently drop
    # them here.
    candidates = [_project(loop) for loop in loops]
    def _bbox_area(poly: list[Tuple[float, float]]) -> float:
        us = [p[0] for p in poly]
        vs = [p[1] for p in poly]
        return (max(us) - min(us)) * (max(vs) - min(vs))
    poly_2d = max(candidates, key=_bbox_area)
    poly_2d = _simplify_collinear(poly_2d)

    us = [p[0] for p in poly_2d]
    vs = [p[1] for p in poly_2d]
    bbox_w = max(us) - min(us)
    bbox_h = max(vs) - min(vs)
    bbox_area = max(bbox_w * bbox_h, 1e-9)
    cluster_area = float(np.sum(triangle_areas[cluster]))
    area_ratio = cluster_area / bbox_area

    shape_type: Literal["rectangle", "polygon"] = (
        "rectangle" if (len(poly_2d) == 4 and area_ratio > 0.85) else "polygon"
    )

    return PlanarFace(
        face_id=face_id,
        normal=FaceNormal(
            direction=direction,
            vector=(float(axis[0]), float(axis[1]), float(axis[2])),
        ),
        centre_3d_mm=(float(centre_3d[0]), float(centre_3d[1]), float(centre_3d[2])),
        area_mm2=cluster_area,
        vertices_2d_mm=poly_2d,
        bounding_box_mm=(bbox_w, bbox_h),
        shape_type=shape_type,
    )


# ---------------------------------------------------------------------------
# Cylindrical face: detect axis from normal-fan, recover radius / height
# ---------------------------------------------------------------------------
def _cluster_cylindrical_or_none(
    face_id: int,
    cluster: list[int],
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    triangle_areas: np.ndarray,
    vertices: np.ndarray,
) -> Optional[CylindricalFace]:
    """Heuristic cylinder detection over a leftover cluster.

    The cluster is "cylindrical around principal axis A" iff:
      * the per-triangle normals stay mostly perpendicular to A
        (|dot(n, A)| <= CYL_AXIS_NORMAL_DOT_MAX on average), AND
      * the angular span of the normals around A is at least
        ``CYL_MIN_ANGULAR_SPAN_DEG`` (so two flats meeting at an edge
        do not get mistaken for a quarter-cylinder).

    For each principal axis candidate we score it and pick the best.
    """
    if len(cluster) < 4:
        return None

    norms = triangle_normals[cluster]
    areas = triangle_areas[cluster]
    centroids = np.mean(vertices[triangles[cluster]], axis=1)

    best: Optional[Tuple[float, PrincipalAxis, np.ndarray]] = None
    for label, axis in PRINCIPAL_AXES:
        # Skip the negative copies of an axis — both ±X give the same
        # axis line. We only need the half-set; sign comes from
        # boss/hole later.
        if label.startswith("-"):
            continue
        dots = np.abs(norms @ axis)
        avg_dot = float(np.average(dots, weights=areas))
        if avg_dot > CYL_AXIS_NORMAL_DOT_MAX:
            continue
        # Angular span: project normals onto the plane perpendicular to
        # axis, then measure atan2 spread.
        proj = norms - np.outer(norms @ axis, axis)
        proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
        proj_norm[proj_norm < 1e-9] = 1.0
        proj_unit = proj / proj_norm
        # Pick a reference direction in the perpendicular plane.
        if abs(axis[0]) < 0.9:
            ref = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        else:
            ref = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        ref = ref / np.linalg.norm(ref)
        cross = np.cross(ref, proj_unit)
        sins = cross @ axis
        coss = proj_unit @ ref
        angles = np.arctan2(sins, coss)
        span = float(np.max(angles) - np.min(angles))
        # Wrap into [0, 2π] in case all angles cluster near ±π.
        if span < 0:
            span += 2 * math.pi
        if math.degrees(span) < CYL_MIN_ANGULAR_SPAN_DEG:
            continue
        # Score: prefer larger angular span and lower avg_dot.
        score = math.degrees(span) - avg_dot * 100
        if best is None or score > best[0]:
            best = (score, label, axis)  # type: ignore[assignment]

    if best is None:
        return None
    _, axis_label, axis = best  # type: ignore[misc]

    # Project centroids onto a plane perpendicular to the axis to find
    # the centre point of the cylinder (centroid of the projected
    # points). Radius = mean distance from each centroid to that
    # centre (in the perpendicular plane).
    along = centroids @ axis
    centre_along = float(np.average(along, weights=areas))
    perp = centroids - np.outer(along, axis)
    perp_centre = np.average(perp, axis=0, weights=areas)
    radii = np.linalg.norm(perp - perp_centre, axis=1)
    radius = float(np.average(radii, weights=areas))
    height = float(np.max(along) - np.min(along))

    # Boss vs hole: do triangle normals point AWAY from the axis (boss)
    # or TOWARD it (hole)? Inspect each triangle: compute its centroid
    # offset from the axis, normalise, then dot with its normal.
    radial = perp - perp_centre
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm[radial_norm < 1e-9] = 1.0
    radial_unit = radial / radial_norm
    outward = np.einsum("ij,ij->i", radial_unit, norms)
    outward_score = float(np.average(outward, weights=areas))
    face_type: Literal["boss", "hole"] = "boss" if outward_score > 0 else "hole"

    centre_3d = perp_centre + axis * centre_along

    # Snap axis_label to the +/- form. Cylinders are bidirectional so
    # we always emit the "+" axis label; downstream consumers can read
    # ``face_type`` to know if they're looking at a boss or hole.
    return CylindricalFace(
        face_id=face_id,
        axis_direction=axis_label,
        radius_mm=radius,
        height_mm=height,
        centre_3d_mm=(float(centre_3d[0]), float(centre_3d[1]), float(centre_3d[2])),
        face_type=face_type,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_faces(mesh_path: str | Path) -> ExtractedGeometry:
    """Run the full extraction pipeline on an STL.

    Returns a fully populated ``ExtractedGeometry``. All distances are
    in mm in the post-normalisation frame (longest bbox extent ~100 mm).
    """
    path = Path(mesh_path)
    mesh, _scale = _load_normalised_mesh(path)

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    norms = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    triangle_normals = triangle_normals / norms

    # Per-triangle area via cross product magnitude.
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # ---- Pass 1: planar clustering with tight normal tolerance.
    planar_clusters = _cluster_triangles(triangles, triangle_normals)
    cluster_areas = np.array(
        [float(np.sum(triangle_areas[c])) for c in planar_clusters], dtype=np.float64
    )
    if cluster_areas.size == 0 or cluster_areas.max() <= 0:
        raise ValueError(f"{path}: clustering produced no usable faces")
    area_threshold = MIN_AREA_FRAC * cluster_areas.max()

    planar: list[PlanarFace] = []
    cylindrical: list[CylindricalFace] = []
    next_face_id = 0
    consumed = np.zeros(len(triangles), dtype=bool)

    for cluster, total_area in zip(planar_clusters, cluster_areas):
        if total_area < area_threshold:
            continue
        face = _planar_face_from_cluster(
            next_face_id, cluster, triangles,
            triangle_normals, triangle_areas, vertices,
        )
        if face is None:
            continue
        planar.append(face)
        next_face_id += 1
        for ti in cluster:
            consumed[ti] = True

    # ---- Pass 2: cylinder detection on leftover triangles.
    # Group remaining triangles purely by adjacency (no normal check)
    # so the curved surface of a cylinder lands in ONE cluster instead
    # of being torn apart by the 5° threshold.
    remaining_mask = ~consumed
    if remaining_mask.any():
        cyl_clusters = _cluster_triangles(
            triangles, triangle_normals,
            tol_deg=CYLINDER_PASS_TOL_DEG, candidate_mask=remaining_mask,
        )
        for cluster in cyl_clusters:
            total_area = float(np.sum(triangle_areas[cluster]))
            if total_area < area_threshold:
                continue
            cyl = _cluster_cylindrical_or_none(
                next_face_id, cluster, triangles,
                triangle_normals, triangle_areas, vertices,
            )
            if cyl is not None:
                cylindrical.append(cyl)
                next_face_id += 1

    bbox_extents = mesh.get_axis_aligned_bounding_box().get_extent()
    bbox_tuple = (float(bbox_extents[0]), float(bbox_extents[1]), float(bbox_extents[2]))
    longest = float(max(bbox_tuple))

    return ExtractedGeometry(
        planar_faces=planar,
        cylindrical_faces=cylindrical,
        bounding_box_mm=bbox_tuple,
        longest_dimension_mm=longest,
        face_count=len(planar) + len(cylindrical),
        source_file=str(path),
    )


# ---------------------------------------------------------------------------
# Plain-English summary for the LLM prompt
# ---------------------------------------------------------------------------
def _group_planar_by_direction(
    faces: Iterable[PlanarFace],
) -> dict[PrincipalAxis, list[PlanarFace]]:
    out: dict[PrincipalAxis, list[PlanarFace]] = defaultdict(list)
    for f in faces:
        out[f.normal.direction].append(f)
    return out


def summarise_geometry(geometry: ExtractedGeometry) -> str:
    """Plain-English block grouping faces by principal direction."""
    lines: list[str] = []
    bw, bd, bh = geometry.bounding_box_mm
    lines.append(
        f"Bounding box: {bw:.1f} x {bd:.1f} x {bh:.1f} mm "
        f"(longest = {geometry.longest_dimension_mm:.1f} mm). "
        f"{geometry.face_count} faces extracted."
    )

    grouped = _group_planar_by_direction(geometry.planar_faces)
    for direction in ("+Z", "-Z", "+X", "-X", "+Y", "-Y"):
        faces = grouped.get(direction, [])  # type: ignore[arg-type]
        if not faces:
            continue
        total_area = sum(f.area_mm2 for f in faces)
        biggest = max(faces, key=lambda f: f.area_mm2)
        bw, bh = biggest.bounding_box_mm
        line = (
            f"{direction} planar faces ({len(faces)} face"
            f"{'s' if len(faces) != 1 else ''}, total area "
            f"{total_area:.0f} mm²): largest is F{biggest.face_id} "
            f"{biggest.area_mm2:.0f} mm² {biggest.shape_type} "
            f"{bw:.1f}x{bh:.1f} mm centred at "
            f"({biggest.centre_3d_mm[0]:+.1f}, "
            f"{biggest.centre_3d_mm[1]:+.1f}, "
            f"{biggest.centre_3d_mm[2]:+.1f}) mm."
        )
        if biggest.shape_type == "polygon":
            verts = ", ".join(f"({u:+.1f}, {v:+.1f})" for u, v in biggest.vertices_2d_mm)
            line += f" Vertices_2d_mm: [{verts}]"
        lines.append(line)

    if geometry.cylindrical_faces:
        cy_lines = []
        for c in geometry.cylindrical_faces:
            cy_lines.append(
                f"F{c.face_id} {c.face_type} on {c.axis_direction} axis: "
                f"radius={c.radius_mm:.1f} mm, height={c.height_mm:.1f} mm, "
                f"axis-midpoint=({c.centre_3d_mm[0]:+.1f}, "
                f"{c.centre_3d_mm[1]:+.1f}, {c.centre_3d_mm[2]:+.1f}) mm"
            )
        lines.append("Cylinders:")
        for cy in cy_lines:
            lines.append(f"  - {cy}")
    else:
        lines.append("Cylinders: none detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: ``python -m backend.ai_infra.face_extractor path/to/part.stl``
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Extract exact face geometry from an STL mesh.",
    )
    p.add_argument("mesh", type=Path, help="Path to .stl file.")
    p.add_argument("--json", action="store_true",
                   help="Print the full ExtractedGeometry JSON.")
    args = p.parse_args()

    geom = extract_faces(args.mesh)
    print(summarise_geometry(geom))
    if args.json:
        print()
        print(geom.model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
