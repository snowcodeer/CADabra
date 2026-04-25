import json
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore', message='All-NaN slice encountered')

REPO_ROOT = Path(__file__).resolve().parents[1]
POINT_COUNT = 4048

# Noise calibrated to roughly match a structured-light depth scanner (e.g. RealSense / Kinect-class)
# on a 2-unit-diameter object: ~0.3% lateral, ~0.7% axial, sharp grazing-angle dropoff,
# no free-floating outliers (real scanners produce flying pixels only at depth discontinuities,
# and we already cull those via the edge mask).
SCAN_PROFILE = {
    'n_views': 8,
    'image_resolution': 180,
    'sigma_z': 0.0030,
    'sigma_xy': 0.0008,
    'grazing_cos_cutoff': float(np.cos(np.deg2rad(75.0))),
    'grazing_clip_floor': 0.40,
    'edge_depth_threshold': 0.025,
    'orthographic_extent': 1.6,
    'camera_distance': 3.0,
    'sor_k': 12,
    'sor_std_ratio': 1.5,
}

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    HAS_PYEMBREE = True
except Exception:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    HAS_PYEMBREE = False


def fps_numpy_fallback(points: np.ndarray, n: int) -> np.ndarray:
    selected = [0]
    distances = np.full(len(points), np.inf)
    for _ in range(n - 1):
        last = points[selected[-1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))
    return points[selected]


def sample_to_count(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError('No points available for sampling')

    if len(points) >= n_points:
        return fps_numpy_fallback(points, n_points)

    extra_ids = rng.choice(len(points), size=n_points - len(points), replace=True)
    padded = np.concatenate([points, points[extra_ids]], axis=0)
    return fps_numpy_fallback(padded, n_points)


def fibonacci_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    # Random offset prevents identical viewpoints across samples while keeping uniform coverage.
    offset = rng.uniform(0.0, 1.0)
    indices = np.arange(n) + offset
    phi = np.arccos(1 - 2 * indices / n)
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _orthonormal_basis(view_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view_dir, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, view_dir)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(view_dir, right)
    up /= np.linalg.norm(up) + 1e-12
    return right, up


def virtual_scan_view(
    intersector: RayMeshIntersector,
    camera_pos: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    res = SCAN_PROFILE['image_resolution']
    extent = SCAN_PROFILE['orthographic_extent']

    view_dir = -camera_pos / (np.linalg.norm(camera_pos) + 1e-12)
    right, up = _orthonormal_basis(view_dir)

    coords = np.linspace(-extent / 2.0, extent / 2.0, res)
    grid_u, grid_v = np.meshgrid(coords, coords)
    origins = (
        camera_pos[None, :]
        + grid_u.reshape(-1, 1) * right[None, :]
        + grid_v.reshape(-1, 1) * up[None, :]
    )
    directions = np.tile(view_dir, (origins.shape[0], 1))

    locations, ray_indices, tri_indices = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )

    if len(locations) == 0:
        return np.empty((0, 3), dtype=np.float64)

    face_normals = intersector.mesh.face_normals[tri_indices]
    hit_directions = directions[ray_indices]

    cos_grazing = np.abs(np.einsum('ij,ij->i', face_normals, hit_directions))
    grazing_mask = cos_grazing >= SCAN_PROFILE['grazing_cos_cutoff']
    if not np.any(grazing_mask):
        return np.empty((0, 3), dtype=np.float64)

    locations = locations[grazing_mask]
    face_normals = face_normals[grazing_mask]
    ray_indices = ray_indices[grazing_mask]
    cos_grazing = cos_grazing[grazing_mask]

    depth = -((locations - camera_pos) @ view_dir)
    depth_image = np.full(res * res, np.nan, dtype=np.float64)
    depth_image[ray_indices] = depth
    depth_image = depth_image.reshape(res, res)

    pad = np.pad(depth_image, 1, constant_values=np.nan)
    neighborhood = np.stack(
        [
            pad[0:res, 1:res + 1],
            pad[2:res + 2, 1:res + 1],
            pad[1:res + 1, 0:res],
            pad[1:res + 1, 2:res + 2],
        ],
        axis=0,
    )
    diffs = np.abs(neighborhood - depth_image[None, :, :])
    edge_mask_image = np.nanmax(diffs, axis=0) > SCAN_PROFILE['edge_depth_threshold']
    edge_mask_image = np.where(np.isnan(diffs).all(axis=0), False, edge_mask_image)

    edge_per_hit = edge_mask_image.reshape(-1)[ray_indices]
    keep_mask = ~edge_per_hit
    if not np.any(keep_mask):
        return np.empty((0, 3), dtype=np.float64)

    locations = locations[keep_mask]
    face_normals = face_normals[keep_mask]
    cos_grazing = cos_grazing[keep_mask]

    sigma_z = SCAN_PROFILE['sigma_z'] / np.clip(cos_grazing, SCAN_PROFILE['grazing_clip_floor'], 1.0)
    along_ray_noise = rng.normal(scale=1.0, size=len(locations)) * sigma_z
    lateral_u_noise = rng.normal(scale=SCAN_PROFILE['sigma_xy'], size=len(locations))
    lateral_v_noise = rng.normal(scale=SCAN_PROFILE['sigma_xy'], size=len(locations))

    noise_vec = (
        along_ray_noise[:, None] * view_dir[None, :]
        + lateral_u_noise[:, None] * right[None, :]
        + lateral_v_noise[:, None] * up[None, :]
    )
    return locations + noise_vec


def statistical_outlier_removal(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Drop points whose mean distance to k nearest neighbors exceeds the global mean by std_ratio*std.

    Emulates per-pixel sensor confidence filtering that real depth scanners apply on-device.
    Critical to run BEFORE FPS sampling, otherwise FPS preferentially picks isolated outliers
    because they maximize the minimum-distance criterion.
    """
    n = len(points)
    if n <= k + 1:
        return points
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_d = dists[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    return points[mean_d <= threshold]


def virtual_scan(mesh: trimesh.Trimesh, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    intersector = RayMeshIntersector(mesh)
    camera_dirs = fibonacci_sphere(SCAN_PROFILE['n_views'], rng)
    camera_positions = camera_dirs * SCAN_PROFILE['camera_distance']

    all_points: list[np.ndarray] = []
    for cam_pos in camera_positions:
        pts = virtual_scan_view(intersector, cam_pos, rng)
        if len(pts):
            all_points.append(pts)

    if not all_points:
        raise RuntimeError('virtual scan produced no hits')

    points = np.concatenate(all_points, axis=0)
    raw_count = int(len(points))
    points = statistical_outlier_removal(
        points,
        k=int(SCAN_PROFILE['sor_k']),
        std_ratio=float(SCAN_PROFILE['sor_std_ratio']),
    )
    return points, {'raw_scanned_points': raw_count, 'after_sor_points': int(len(points))}


def normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(2.0 / max(mesh.extents))
    return mesh


def stl_to_ply(stl_path: Path, ply_path: Path, n_points: int = POINT_COUNT, seed: int = 0) -> dict:
    mesh = trimesh.load_mesh(str(stl_path))
    raw_extents = mesh.extents.tolist()

    if len(mesh.vertices) == 0:
        raise ValueError(f'Empty mesh: {stl_path}')

    raw_watertight = bool(mesh.is_watertight)
    mesh = normalise_mesh(mesh)

    rng = np.random.default_rng(seed)
    scanned, scan_stats = virtual_scan(mesh, rng)
    points = sample_to_count(scanned, n_points, rng)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path), pcd)

    return {
        'ply_path': str(ply_path),
        'n_points': n_points,
        'raw_extents_mm': raw_extents,
        'is_watertight_before_normalization': raw_watertight,
        'used_pyembree': HAS_PYEMBREE,
        'scan_profile': SCAN_PROFILE,
        **scan_stats,
    }


def validate_ply(ply_path: Path) -> bool:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    assert points.shape == (POINT_COUNT, 3), f'Wrong shape: {points.shape}'
    assert not np.any(np.isnan(points)), 'NaN values in point cloud'
    assert points.min() >= -1.5, f'Points outside expected range: {points.min()}'
    assert points.max() <= 1.5, f'Points outside expected range: {points.max()}'
    return True


def run_batch():
    stl_dir = REPO_ROOT / 'backend' / 'outputs' / 'deepcad_selected_stl'
    out_dir = REPO_ROOT / 'backend' / 'sample_data' / 'deepcad_selected_ply'
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_manifest = json.loads((stl_dir / 'manifest.json').read_text())
    succeeded = []
    failed = []
    for row in stl_manifest:
        if not row.get('success'):
            continue
        sid = int(row['id'])
        stl_file = stl_dir / row['stl_file']
        ply_file = out_dir / f'deepcadimg_{sid:06d}.ply'
        meta_file = out_dir / f'deepcadimg_{sid:06d}_meta.json'

        try:
            meta = stl_to_ply(stl_file, ply_file, n_points=POINT_COUNT, seed=sid)
            meta['source_stl'] = str(stl_file)
            meta['sample_id'] = f'deepcadimg_{sid:06d}'
            meta_file.write_text(json.dumps(meta, indent=2))
            validate_ply(ply_file)
            succeeded.append({'sample_id': meta['sample_id'], 'ply_file': ply_file.name, 'meta_file': meta_file.name})
            print(
                f"deepcadimg_{sid:06d}: scanned={meta['raw_scanned_points']} "
                f"sor={meta['after_sor_points']} -> kept={POINT_COUNT}"
            )
        except Exception as exc:
            failed.append({'sample_id': f'deepcadimg_{sid:06d}', 'error': str(exc)})
            print(f'deepcadimg_{sid:06d}: failed - {exc}')

    (out_dir / 'manifest.json').write_text(json.dumps(succeeded, indent=2))
    print(f'generated {len(succeeded)} point clouds, {len(failed)} failed | pyembree={HAS_PYEMBREE}')


if __name__ == '__main__':
    run_batch()
