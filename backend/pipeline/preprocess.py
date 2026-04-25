from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

LOGGER = logging.getLogger(__name__)


def _pca_rotation(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def preprocess_mesh(mesh_path: str | Path) -> tuple[o3d.geometry.TriangleMesh, tuple[float, float, float]]:
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")

    mesh = _cleanup_mesh(mesh)
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] < 4:
        raise ValueError("Mesh too sparse for normalization")

    rotation = _pca_rotation(vertices)
    rotated_vertices = (vertices - vertices.mean(axis=0)) @ rotation
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)

    aabb = mesh.get_axis_aligned_bounding_box()
    min_corner = aabb.get_min_bound()
    mesh.translate(-min_corner)

    mesh = _cleanup_mesh(mesh)
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    dims = (float(extent[0]), float(extent[1]), float(extent[2]))

    if not mesh.is_watertight():
        LOGGER.warning("Preprocess output is not watertight: %s", mesh_path)

    return mesh, dims


def derive_visualization_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    point_count: int = 5000,
) -> o3d.geometry.PointCloud:
    if point_count < 500:
        raise ValueError("point_count must be at least 500")
    pcd = mesh.sample_points_uniformly(number_of_points=point_count)
    if len(pcd.points) < 500:
        raise ValueError("Derived point cloud too sparse for visualization")
    return pcd


def save_mesh(mesh: o3d.geometry.TriangleMesh, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"Failed to write mesh to {output_path}")


def _default_sample_stl() -> Path:
    return Path(__file__).resolve().parents[1] / "sample_data" / "block.stl"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_path = _default_sample_stl()
    processed_mesh, bbox_mm = preprocess_mesh(sample_path)
    print(f"Sample: {sample_path.name}")
    print(f"Watertight: {processed_mesh.is_watertight()}")
    print(f"Bounding box (mm): X={bbox_mm[0]:.2f}, Y={bbox_mm[1]:.2f}, Z={bbox_mm[2]:.2f}")

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    save_mesh(processed_mesh, outputs_dir / f"{sample_path.stem}_preprocessed.stl")
    pcd = derive_visualization_point_cloud(processed_mesh, point_count=5000)
    o3d.io.write_point_cloud(str(outputs_dir / f"{sample_path.stem}_viz.ply"), pcd)
    print("Wrote debug preprocessed mesh and visualization point cloud.")
