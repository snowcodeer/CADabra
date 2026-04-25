import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

warnings.filterwarnings('ignore', category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MANIFEST = REPO_ROOT / 'backend' / 'sample_data' / 'deepcad_selected_ply' / 'manifest.json'
SRC_DIR = SRC_MANIFEST.parent
OUT_DIR = REPO_ROOT / 'backend' / 'outputs' / 'deepcad_pc_recon_stl'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PARAMS = {
    'statistical_nb_neighbors': 24,
    'statistical_std_ratio': 2.0,
    'radius_nb_points': 8,
    'radius_factor': 3.0,
    'voxel_size': 0.012,
    'normal_radius': 0.06,
    'normal_max_nn': 40,
    'denoise_k': 20,
    'orient_k': 32,
}

RECON_PARAMS = {
    'poisson_depths': (8, 9),
    'poisson_density_quantile': 0.06,
    'alpha_factors': (0.06, 0.10, 0.16),
}

DECIMATION_PARAMS = {
    'start_fraction': 0.5,
    'min_triangles': 24,
    'hausdorff_budget': 0.005,
    'sample_count': 4096,
}


def _stat(pcd: o3d.geometry.PointCloud) -> int:
    return len(pcd.points)


def remove_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=CLEAN_PARAMS['statistical_nb_neighbors'],
        std_ratio=CLEAN_PARAMS['statistical_std_ratio'],
    )

    distances = np.asarray(cleaned.compute_nearest_neighbor_distance())
    if distances.size:
        radius = float(np.mean(distances)) * CLEAN_PARAMS['radius_factor']
        cleaned, _ = cleaned.remove_radius_outlier(
            nb_points=CLEAN_PARAMS['radius_nb_points'],
            radius=radius,
        )
    return cleaned


def voxel_downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=CLEAN_PARAMS['voxel_size'])


def pca_plane_denoise(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(pcd.points)
    k = CLEAN_PARAMS['denoise_k']
    if len(points) < k:
        return pcd

    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    _, indices = nn.kneighbors(points)

    neigh = points[indices]
    centroids = neigh.mean(axis=1)
    centered = neigh - centroids[:, None, :]
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[..., 0]
    delta = points - centroids
    proj = np.einsum('ni,ni->n', delta, normals)
    smoothed = points - normals * proj[:, None]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(smoothed)
    return out


def estimate_and_orient_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=CLEAN_PARAMS['normal_radius'],
            max_nn=CLEAN_PARAMS['normal_max_nn'],
        )
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(CLEAN_PARAMS['orient_k'])
    except Exception:
        pass
    return pcd


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        return mesh

    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size > 0:
        largest = int(np.argmax(cluster_n_triangles))
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def _poisson_candidates(pcd: o3d.geometry.PointCloud) -> list[tuple[str, o3d.geometry.TriangleMesh]]:
    out: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    for depth in RECON_PARAMS['poisson_depths']:
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            densities = np.asarray(densities)
            if densities.size:
                threshold = float(np.quantile(densities, RECON_PARAMS['poisson_density_quantile']))
                mesh.remove_vertices_by_mask(densities < threshold)
            mesh = _cleanup_mesh(mesh)
            if len(mesh.triangles) > 0:
                out.append((f'poisson_depth{depth}', mesh))
        except Exception:
            continue
    return out


def _alpha_candidates(pcd: o3d.geometry.PointCloud) -> list[tuple[str, o3d.geometry.TriangleMesh]]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return []
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    out: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    for factor in RECON_PARAMS['alpha_factors']:
        alpha = max(diag * factor, 1e-4)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh = _cleanup_mesh(mesh)
            if len(mesh.triangles) > 0:
                out.append((f'alpha_{factor:.2f}', mesh))
        except Exception:
            continue
    return out


def _mesh_score(mesh: o3d.geometry.TriangleMesh) -> tuple[int, int, int]:
    # Watertight first, then manifold, then prefer smaller mesh (negative tris).
    return (
        int(mesh.is_watertight()),
        int(mesh.is_edge_manifold()),
        -len(mesh.triangles),
    )


def _to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )


def _from_trimesh(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    out.compute_vertex_normals()
    return out


def fill_holes_if_needed(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if mesh.is_watertight():
        return mesh
    tm = _to_trimesh(mesh)
    try:
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fill_holes(tm)
        trimesh.repair.fix_inversion(tm)
    except Exception:
        return mesh
    if len(tm.faces) == 0:
        return mesh
    fixed = _from_trimesh(tm)
    return _cleanup_mesh(fixed)


def _hausdorff_estimate(reference: trimesh.Trimesh, candidate: trimesh.Trimesh, rng: np.random.Generator) -> float:
    n = DECIMATION_PARAMS['sample_count']
    if len(reference.vertices) == 0 or len(candidate.vertices) == 0:
        return float('inf')

    ref_points, _ = trimesh.sample.sample_surface(reference, n)
    cand_points, _ = trimesh.sample.sample_surface(candidate, n)

    _, ref_to_cand, _ = trimesh.proximity.closest_point(candidate, ref_points)
    _, cand_to_ref, _ = trimesh.proximity.closest_point(reference, cand_points)
    return float(max(ref_to_cand.max(), cand_to_ref.max()))


def decimate(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict]:
    triangles = len(mesh.triangles)
    if triangles <= DECIMATION_PARAMS['min_triangles']:
        return mesh, {'decimation_iterations': 0, 'triangles_before': triangles, 'triangles_after': triangles, 'hausdorff': 0.0}

    rng = np.random.default_rng(0)
    reference_tm = _to_trimesh(mesh)
    best = mesh
    best_tris = triangles
    iterations = 0
    last_hausdorff = 0.0

    target = max(DECIMATION_PARAMS['min_triangles'], int(triangles * DECIMATION_PARAMS['start_fraction']))

    while target < best_tris and target >= DECIMATION_PARAMS['min_triangles']:
        try:
            candidate = best.simplify_quadric_decimation(target_number_of_triangles=target)
            candidate = _cleanup_mesh(candidate)
        except Exception:
            break

        if len(candidate.triangles) == 0:
            break

        if not candidate.is_watertight():
            candidate = fill_holes_if_needed(candidate)
        if not candidate.is_watertight():
            break

        cand_tm = _to_trimesh(candidate)
        try:
            haus = _hausdorff_estimate(reference_tm, cand_tm, rng)
        except Exception:
            break

        if haus > DECIMATION_PARAMS['hausdorff_budget']:
            break

        best = candidate
        best_tris = len(candidate.triangles)
        last_hausdorff = haus
        iterations += 1
        target = max(DECIMATION_PARAMS['min_triangles'], int(best_tris * 0.5))

    return best, {
        'decimation_iterations': iterations,
        'triangles_before': triangles,
        'triangles_after': best_tris,
        'hausdorff': last_hausdorff,
    }


def reconstruct_noisy_for_sample(ply_path: Path) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Naive reconstruction from raw noisy points: only normals + multi-strategy recon + manifold cleanup.

    No outlier removal, no denoise, no hole fill, no decimation. This represents the
    baseline 'what if we skip the cleaning pipeline' visual for the comparison panel.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError('empty point cloud')

    pcd = estimate_and_orient_normals(pcd)

    candidates: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    candidates.extend(_poisson_candidates(pcd))
    candidates.extend(_alpha_candidates(pcd))
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(candidates, key=lambda item: _mesh_score(item[1]))
    stats = {
        'strategy': strategy,
        'triangles': len(mesh.triangles),
        'vertices': len(mesh.vertices),
        'watertight': bool(mesh.is_watertight()),
        'edge_manifold': bool(mesh.is_edge_manifold()),
    }
    return mesh, stats


def reconstruct_for_sample(ply_path: Path) -> tuple[o3d.geometry.TriangleMesh, dict]:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError('empty point cloud')

    stats: dict = {'input_points': _stat(pcd)}

    pcd = remove_outliers(pcd)
    stats['after_outlier_removal'] = _stat(pcd)

    pcd = voxel_downsample(pcd)
    stats['after_voxel_downsample'] = _stat(pcd)

    pcd = pca_plane_denoise(pcd)
    stats['after_denoise'] = _stat(pcd)

    pcd = estimate_and_orient_normals(pcd)

    candidates: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    candidates.extend(_poisson_candidates(pcd))
    candidates.extend(_alpha_candidates(pcd))
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(candidates, key=lambda item: _mesh_score(item[1]))
    stats['strategy'] = strategy
    stats['triangles_post_recon'] = len(mesh.triangles)
    stats['watertight_post_recon'] = bool(mesh.is_watertight())

    mesh = fill_holes_if_needed(mesh)
    stats['watertight_post_holefill'] = bool(mesh.is_watertight())

    mesh, deci_stats = decimate(mesh)
    stats.update(deci_stats)

    stats['final_triangles'] = len(mesh.triangles)
    stats['final_vertices'] = len(mesh.vertices)
    stats['watertight'] = bool(mesh.is_watertight())
    stats['edge_manifold'] = bool(mesh.is_edge_manifold())

    return mesh, stats


def run_single(sample_id: str, ply_file: str) -> dict:
    ply_path = SRC_DIR / ply_file
    mesh, stats = reconstruct_for_sample(ply_path)

    out_name = f'{sample_id}_recon.stl'
    out_path = OUT_DIR / out_name
    ok = o3d.io.write_triangle_mesh(str(out_path), mesh)
    if not ok or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError('failed to write reconstructed stl')

    noisy_mesh, noisy_stats = reconstruct_noisy_for_sample(ply_path)
    noisy_name = f'{sample_id}_recon_noisy.stl'
    noisy_path = OUT_DIR / noisy_name
    ok_noisy = o3d.io.write_triangle_mesh(str(noisy_path), noisy_mesh)
    if not ok_noisy or not noisy_path.exists() or noisy_path.stat().st_size == 0:
        raise RuntimeError('failed to write noisy reconstructed stl')

    return {
        'sample_id': sample_id,
        'success': True,
        'recon_stl': out_name,
        'recon_noisy_stl': noisy_name,
        'noisy': noisy_stats,
        **stats,
    }


def _invoke_child_once(row: dict) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--sample-id',
        row['sample_id'],
        '--ply-file',
        row['ply_file'],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f'child exited with code {proc.returncode}'
        return {'sample_id': row['sample_id'], 'success': False, 'error': err}

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {'sample_id': row['sample_id'], 'success': False, 'error': 'child produced no output'}

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {'sample_id': row['sample_id'], 'success': False, 'error': f'invalid child json: {exc}'}


def _run_child_for_row(row: dict, max_attempts: int = 3) -> dict:
    last = {'sample_id': row['sample_id'], 'success': False, 'error': 'no attempts'}
    for _ in range(max_attempts):
        last = _invoke_child_once(row)
        if last.get('success'):
            return last
    return last


def run_batch() -> None:
    rows = json.loads(SRC_MANIFEST.read_text())
    out_rows = []

    for row in rows:
        result = _run_child_for_row(row)
        out_rows.append(result)
        if result.get('success'):
            noisy = result.get('noisy', {})
            print(
                f"{result['sample_id']}: strat={result['strategy']} "
                f"noisy={noisy.get('triangles', 0)} clean={result['triangles_post_recon']}"
                f"->{result['final_triangles']} verts={result['final_vertices']} "
                f"wt={result['watertight']} haus={result.get('hausdorff', 0):.4f}"
            )
        else:
            print(f"{result['sample_id']}: failed - {result.get('error', 'unknown error')}")

    (OUT_DIR / 'manifest.json').write_text(json.dumps(out_rows, indent=2))
    success = [r for r in out_rows if r.get('success')]
    watertight = sum(1 for r in success if r.get('watertight'))
    if success:
        avg_tris = int(round(sum(r['final_triangles'] for r in success) / len(success)))
        avg_verts = int(round(sum(r['final_vertices'] for r in success) / len(success)))
    else:
        avg_tris = avg_verts = 0
    print(
        f'reconstructed {len(success)} of {len(out_rows)} | watertight: {watertight} | '
        f'avg tris: {avg_tris} | avg verts: {avg_verts}'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-id')
    parser.add_argument('--ply-file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_id and args.ply_file:
        result = run_single(args.sample_id, args.ply_file)
        print(json.dumps(result))
    else:
        run_batch()


if __name__ == '__main__':
    main()
