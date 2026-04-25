import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

try:
    import torch
    from pytorch3d.ops import sample_farthest_points
    HAS_PYTORCH3D = True
except Exception:
    HAS_PYTORCH3D = False


def fps_numpy_fallback(points: np.ndarray, n: int) -> np.ndarray:
    selected = [0]
    distances = np.full(len(points), np.inf)
    for _ in range(n - 1):
        last = points[selected[-1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))
    return points[selected]


def mesh_to_point_cloud(mesh: trimesh.Trimesh, n_points: int = 256, n_pre_points: int = 8192, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)

    if HAS_PYTORCH3D:
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
        _, ids = sample_farthest_points(vertices_tensor, K=n_points)
        ids = ids[0].numpy()
        return np.asarray(vertices[ids])

    return fps_numpy_fallback(np.asarray(vertices), n_points)


def normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(2.0 / max(mesh.extents))
    return mesh


def stl_to_ply(stl_path: Path, ply_path: Path, n_points: int = 256) -> dict:
    mesh = trimesh.load_mesh(str(stl_path))
    raw_extents = mesh.extents.tolist()

    if len(mesh.vertices) == 0:
        raise ValueError(f'Empty mesh: {stl_path}')

    raw_watertight = bool(mesh.is_watertight)
    mesh = normalise_mesh(mesh)
    points = mesh_to_point_cloud(mesh, n_points=n_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path), pcd)

    return {
        'ply_path': str(ply_path),
        'n_points': n_points,
        'raw_extents_mm': raw_extents,
        'is_watertight_before_normalization': raw_watertight,
        'used_pytorch3d_fps': HAS_PYTORCH3D,
    }


def validate_ply(ply_path: Path) -> bool:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    assert points.shape == (256, 3), f'Wrong shape: {points.shape}'
    assert not np.any(np.isnan(points)), 'NaN values in point cloud'
    assert points.min() >= -1.1, f'Points outside expected range: {points.min()}'
    assert points.max() <= 1.1, f'Points outside expected range: {points.max()}'
    return True


def run_batch():
    stl_dir = Path('/workspace/backend/outputs/deepcad_selected_stl')
    out_dir = Path('/workspace/backend/sample_data/deepcad_selected_ply')
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_manifest = json.loads((stl_dir / 'manifest.json').read_text())
    succeeded = []
    for row in stl_manifest:
        if not row.get('success'):
            continue
        sid = int(row['id'])
        stl_file = stl_dir / row['stl_file']
        ply_file = out_dir / f'deepcadimg_{sid:06d}.ply'
        meta_file = out_dir / f'deepcadimg_{sid:06d}_meta.json'

        meta = stl_to_ply(stl_file, ply_file, n_points=256)
        meta['source_stl'] = str(stl_file)
        meta['sample_id'] = f'deepcadimg_{sid:06d}'
        meta_file.write_text(json.dumps(meta, indent=2))
        validate_ply(ply_file)
        succeeded.append({'sample_id': meta['sample_id'], 'ply_file': ply_file.name, 'meta_file': meta_file.name})

    (out_dir / 'manifest.json').write_text(json.dumps(succeeded, indent=2))
    print('generated', len(succeeded), 'point clouds')


if __name__ == '__main__':
    run_batch()
