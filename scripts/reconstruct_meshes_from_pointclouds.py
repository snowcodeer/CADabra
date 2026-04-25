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
    'voxel_size': 0.010,
    'normal_radius': 0.06,
    'normal_max_nn': 40,
    'denoise_k': 24,
    'denoise_sigma_n': 0.30,  # normal-similarity bandwidth (radians-ish via 1-cos)
    'denoise_iterations': 2,
    'denoise_step': 0.5,       # fraction of bilateral offset to apply per iteration
    'orient_k': 32,
}

RECON_PARAMS = {
    'poisson_depths': (8, 9, 10),
    'poisson_density_quantile': 0.015,   # lower -> keeps more wall coverage, fewer false holes
    'envelope_radius_factor': 2.2,       # x avg NN distance; crops Poisson shrink-wrap membrane
    'alpha_factors': (0.05, 0.075, 0.11),
    # ball pivoting kept only as a fallback - it cannot bridge gaps so it tends to leave swiss-cheese
    'ball_pivot_radius_factors': (1.6, 2.4),
}

SCORE_PARAMS = {
    'chamfer_sample_count': 4096,
    'chamfer_weight': 1.0,
    'boundary_penalty': 0.6,             # weight x (boundary_edges / triangle_count) - punishes gappy meshes
    'poisson_bonus': 0.012,               # strongly prefer poisson family over BP at similar fidelity
    'alpha_bonus': 0.004,
    'watertight_bonus': 0.006,           # closed meshes win ties
    'manifold_bonus': 0.002,
    'size_penalty': 1e-6,
}

HOLE_FILL_PARAMS = {
    # First pass closes small/medium boundary loops, second pass aggressively closes
    # any remaining membrane-fringe artifacts from envelope cropping. Two passes work
    # better than a single huge threshold because Open3D's fill_holes seems to behave
    # differently on a freshly cleaned-up mesh.
    'pass1_frac': 0.30,
    'pass2_frac': 0.80,
    'min_loop_length_abs': 0.20,
}

DECIMATION_PARAMS = {
    'start_fraction': 0.7,
    'shrink_per_iter': 0.75,
    'max_iterations': 4,
    'min_triangles': 24,
    'chamfer_growth_budget': 1.4,        # stop when chamfer-to-cloud grows beyond 1.4x baseline
    'sample_count': 4096,
    'boundary_weight': 4.0,              # protects hole rims and silhouettes during simplification
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


def _local_normals(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Eigen-decomposition normals (smallest-eigenvalue eigenvector) per point."""
    neigh = points[indices]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / indices.shape[1]
    _, eigvecs = np.linalg.eigh(cov)
    return eigvecs[..., 0]


def bilateral_normal_denoise(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Bilateral filter that weights neighbors by both spatial and normal similarity.

    Critical for hole/edge retention: points on opposite sides of a sharp edge or
    hole rim have very different surface normals, so the normal-similarity weight
    suppresses cross-edge averaging that the older PCA-plane filter caused.
    """
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(pcd.points)
    k = CLEAN_PARAMS['denoise_k']
    if len(points) < k + 1:
        return pcd

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
    dists, indices = nn.kneighbors(points)
    neigh_idx = indices[:, 1:]
    neigh_dist = dists[:, 1:]

    sigma_d = float(np.mean(neigh_dist))
    if sigma_d <= 0:
        return pcd
    sigma_n2 = CLEAN_PARAMS['denoise_sigma_n'] ** 2

    normals = _local_normals(points, indices)

    smoothed = points.copy()
    for _ in range(CLEAN_PARAMS['denoise_iterations']):
        neigh_pts = smoothed[neigh_idx]
        neigh_norms = normals[neigh_idx]

        spatial = np.exp(-(neigh_dist ** 2) / (2.0 * sigma_d ** 2))
        normal_sim = np.einsum('nki,ni->nk', neigh_norms, normals)
        normal_sim = np.clip(normal_sim, -1.0, 1.0)
        normal_w = np.exp(-((1.0 - normal_sim) ** 2) / (2.0 * sigma_n2))

        weights = spatial * normal_w
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12

        weighted_centroid = np.einsum('nk,nki->ni', weights, neigh_pts)
        delta = weighted_centroid - smoothed
        proj = np.einsum('ni,ni->n', delta, normals)[:, None]
        offset = normals * proj
        smoothed = smoothed + CLEAN_PARAMS['denoise_step'] * offset

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


def _avg_nn_distance(pts: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    if len(pts) < 4:
        return 1e-3
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(pts)
    dists, _ = nn.kneighbors(pts)
    return float(dists[:, 1].mean())


def _envelope_crop(mesh: o3d.geometry.TriangleMesh, pts: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
    """Drop mesh vertices that are farther than `radius` from any input point.

    Removes the 'shrink-wrap' membrane Poisson reconstruction adds across
    concavities and holes, which is the single biggest hole-killer in the pipeline.
    """
    from sklearn.neighbors import NearestNeighbors

    verts = np.asarray(mesh.vertices)
    if len(verts) == 0 or len(pts) == 0:
        return mesh
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts)
    d, _ = nn.kneighbors(verts)
    far_mask = d[:, 0] > radius
    if not np.any(far_mask):
        return mesh
    mesh.remove_vertices_by_mask(far_mask)
    return mesh


def _poisson_candidates(pcd: o3d.geometry.PointCloud) -> list[tuple[str, o3d.geometry.TriangleMesh]]:
    pts = np.asarray(pcd.points)
    avg_nn = _avg_nn_distance(pts)
    envelope_r = avg_nn * RECON_PARAMS['envelope_radius_factor']
    out: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    for depth in RECON_PARAMS['poisson_depths']:
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            densities = np.asarray(densities)
            if densities.size:
                threshold = float(np.quantile(densities, RECON_PARAMS['poisson_density_quantile']))
                mesh.remove_vertices_by_mask(densities < threshold)
            mesh = _envelope_crop(mesh, pts, envelope_r)
            mesh = _cleanup_mesh(mesh)
            if len(mesh.triangles) > 0:
                out.append((f'poisson_d{depth}', mesh))
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
                out.append((f'alpha_{factor:.3f}', mesh))
        except Exception:
            continue
    return out


def _ball_pivot_candidates(pcd: o3d.geometry.PointCloud) -> list[tuple[str, o3d.geometry.TriangleMesh]]:
    """Ball pivoting cannot bridge gaps wider than its radius, so it naturally
    preserves holes and concave openings - complementary to Poisson and alpha.
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 8:
        return []
    if not pcd.has_normals():
        return []
    avg_nn = _avg_nn_distance(pts)
    out: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    for factor in RECON_PARAMS['ball_pivot_radius_factors']:
        radius = max(avg_nn * factor, 1e-4)
        try:
            radii = o3d.utility.DoubleVector([radius, radius * 1.5])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            mesh = _cleanup_mesh(mesh)
            if len(mesh.triangles) > 0:
                out.append((f'bp_{factor:.1f}', mesh))
        except Exception:
            continue
    return out


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


def _chamfer_to_cloud(mesh: o3d.geometry.TriangleMesh, cloud_pts: np.ndarray) -> float:
    """Symmetric Chamfer distance between mesh surface samples and the input cloud.
    Lower = mesh follows the actual data better (rewards meshes that respect holes
    instead of filling them in)."""
    from sklearn.neighbors import NearestNeighbors

    if len(mesh.triangles) == 0 or len(cloud_pts) == 0:
        return float('inf')
    n = SCORE_PARAMS['chamfer_sample_count']
    sampled = mesh.sample_points_uniformly(number_of_points=min(n, max(64, len(mesh.triangles))))
    samples = np.asarray(sampled.points)
    if len(samples) == 0:
        return float('inf')

    nn_to_cloud = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(cloud_pts)
    d_m_to_c, _ = nn_to_cloud.kneighbors(samples)
    nn_to_mesh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(samples)
    d_c_to_m, _ = nn_to_mesh.kneighbors(cloud_pts)
    return float(d_m_to_c.mean() + d_c_to_m.mean())


def _boundary_edge_count(mesh: o3d.geometry.TriangleMesh) -> int:
    """Edges that bound exactly one triangle. Many boundary edges = swiss-cheese."""
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return 0
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _mesh_score(mesh: o3d.geometry.TriangleMesh, cloud_pts: np.ndarray, strategy: str) -> float:
    """Higher is better.

    Primary signal: Chamfer fidelity to the input cloud.
    Secondary: penalize mesh that has many boundary edges relative to its size
        (swiss-cheese ball-pivoting outputs).
    Tertiary: bonuses for Poisson/alpha (fill naturally) over BP, and for watertight/manifold.
    """
    chamfer = _chamfer_to_cloud(mesh, cloud_pts)
    score = -SCORE_PARAMS['chamfer_weight'] * chamfer

    n_tris = max(len(mesh.triangles), 1)
    boundary_density = _boundary_edge_count(mesh) / n_tris
    score -= SCORE_PARAMS['boundary_penalty'] * boundary_density

    if strategy.startswith('poisson'):
        score += SCORE_PARAMS['poisson_bonus']
    elif strategy.startswith('alpha'):
        score += SCORE_PARAMS['alpha_bonus']

    if mesh.is_watertight():
        score += SCORE_PARAMS['watertight_bonus']
    if mesh.is_edge_manifold():
        score += SCORE_PARAMS['manifold_bonus']
    score -= SCORE_PARAMS['size_penalty'] * len(mesh.triangles)
    return score


def _bbox_diag(mesh: o3d.geometry.TriangleMesh) -> float:
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return 0.0
    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))


def _fill_with_tensor(mesh: o3d.geometry.TriangleMesh, hole_size: float) -> o3d.geometry.TriangleMesh | None:
    """Use Open3D tensor TriangleMesh.fill_holes(hole_size) to close any boundary
    loop whose perimeter is < hole_size. Returns None if the operation fails or
    returns an empty mesh."""
    try:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled = t_mesh.fill_holes(hole_size=float(hole_size))
        out = filled.to_legacy()
    except Exception:
        return None
    if len(out.triangles) == 0:
        return None
    return out


def _fill_with_trimesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Fallback hole patcher using trimesh.repair.fill_holes (handles up to
    triangle-shaped holes). Only used if the Open3D tensor path fails."""
    tm = _to_trimesh(mesh)
    try:
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fill_holes(tm)
        trimesh.repair.fix_inversion(tm)
    except Exception:
        return mesh
    if len(tm.faces) == 0:
        return mesh
    return _from_trimesh(tm)


def aggressive_fill_holes(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Close as many holes as possible while preserving truly large openings.

    Two-pass strategy:
      pass 1: hole_size = pass1_frac * bbox_diag - closes simple noise loops cleanly
      pass 2: hole_size = pass2_frac * bbox_diag - mops up the shredded membrane fringe
              that envelope-cropped Poisson reconstructions tend to leave behind.

    Open3D's tensor fill_holes(hole_size) gates on a geometric size of the loop
    region (not perimeter). The second, more aggressive pass on the
    already-cleaned mesh is what closes the bulk of envelope-crop artifacts in
    practice.
    """
    diag = _bbox_diag(mesh)
    boundary_before = _boundary_edge_count(mesh)
    stats = {
        'hole_fill_diag': round(diag, 6),
        'hole_fill_boundary_before': boundary_before,
    }

    if boundary_before == 0:
        stats['hole_fill_method'] = 'noop'
        stats['hole_fill_pass1_size'] = 0.0
        stats['hole_fill_pass2_size'] = 0.0
        stats['hole_fill_boundary_after_pass1'] = 0
        stats['hole_fill_boundary_after'] = 0
        return mesh, stats

    pass1 = max(diag * HOLE_FILL_PARAMS['pass1_frac'], HOLE_FILL_PARAMS['min_loop_length_abs'])
    pass2 = max(diag * HOLE_FILL_PARAMS['pass2_frac'], HOLE_FILL_PARAMS['min_loop_length_abs'])
    stats['hole_fill_pass1_size'] = round(pass1, 6)
    stats['hole_fill_pass2_size'] = round(pass2, 6)

    filled = _fill_with_tensor(mesh, pass1)
    if filled is None:
        fallback = _cleanup_mesh(_fill_with_trimesh(mesh))
        stats['hole_fill_method'] = 'trimesh_fallback'
        stats['hole_fill_boundary_after_pass1'] = _boundary_edge_count(fallback)
        stats['hole_fill_boundary_after'] = stats['hole_fill_boundary_after_pass1']
        return fallback, stats

    filled = _cleanup_mesh(filled)
    stats['hole_fill_boundary_after_pass1'] = _boundary_edge_count(filled)

    if stats['hole_fill_boundary_after_pass1'] > 0:
        filled2 = _fill_with_tensor(filled, pass2)
        if filled2 is not None:
            filled = _cleanup_mesh(filled2)
        stats['hole_fill_method'] = 'open3d_tensor_2pass'
    else:
        stats['hole_fill_method'] = 'open3d_tensor_1pass'

    stats['hole_fill_boundary_after'] = _boundary_edge_count(filled)
    return filled, stats




def _simplify_with_boundary(mesh: o3d.geometry.TriangleMesh, target: int) -> o3d.geometry.TriangleMesh:
    """Open3D's simplify_quadric_decimation accepts boundary_weight on newer
    versions; older builds don't. Try the keyword first, fall back to default."""
    try:
        return mesh.simplify_quadric_decimation(
            target_number_of_triangles=target,
            boundary_weight=DECIMATION_PARAMS['boundary_weight'],
        )
    except TypeError:
        return mesh.simplify_quadric_decimation(target_number_of_triangles=target)


def decimate(mesh: o3d.geometry.TriangleMesh, cloud_pts: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Iterative quadric decimation bounded by Chamfer fidelity to the input
    cloud (not to the pre-decimation mesh). This avoids the trap where each step
    looks fine relative to the previous step but the overall fidelity to the
    real data degrades silently."""
    triangles = len(mesh.triangles)
    if triangles <= DECIMATION_PARAMS['min_triangles']:
        return mesh, {
            'decimation_iterations': 0,
            'triangles_before': triangles,
            'triangles_after': triangles,
            'chamfer_baseline': 0.0,
            'chamfer_after': 0.0,
        }

    baseline_chamfer = _chamfer_to_cloud(mesh, cloud_pts)
    chamfer_cap = baseline_chamfer * DECIMATION_PARAMS['chamfer_growth_budget']
    was_watertight = mesh.is_watertight()

    best = mesh
    best_tris = triangles
    best_chamfer = baseline_chamfer
    iterations = 0
    target = max(DECIMATION_PARAMS['min_triangles'], int(triangles * DECIMATION_PARAMS['start_fraction']))

    while iterations < DECIMATION_PARAMS['max_iterations'] and target < best_tris and target >= DECIMATION_PARAMS['min_triangles']:
        try:
            candidate = _simplify_with_boundary(best, target)
            candidate = _cleanup_mesh(candidate)
        except Exception:
            break
        if len(candidate.triangles) == 0:
            break

        # If we started watertight, refuse to introduce holes from decimation.
        if was_watertight and not candidate.is_watertight():
            candidate, _ = aggressive_fill_holes(candidate)
            if not candidate.is_watertight():
                break

        try:
            cand_chamfer = _chamfer_to_cloud(candidate, cloud_pts)
        except Exception:
            break

        if cand_chamfer > chamfer_cap:
            break

        best = candidate
        best_tris = len(candidate.triangles)
        best_chamfer = cand_chamfer
        iterations += 1
        target = max(DECIMATION_PARAMS['min_triangles'], int(best_tris * DECIMATION_PARAMS['shrink_per_iter']))

    return best, {
        'decimation_iterations': iterations,
        'triangles_before': triangles,
        'triangles_after': best_tris,
        'chamfer_baseline': round(baseline_chamfer, 6),
        'chamfer_after': round(best_chamfer, 6),
    }


def _gather_candidates(pcd: o3d.geometry.PointCloud) -> list[tuple[str, o3d.geometry.TriangleMesh]]:
    candidates: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    candidates.extend(_poisson_candidates(pcd))
    candidates.extend(_alpha_candidates(pcd))
    candidates.extend(_ball_pivot_candidates(pcd))
    return candidates


def reconstruct_noisy_for_sample(ply_path: Path) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Naive reconstruction from raw noisy points: normals + multi-strategy recon + manifold cleanup.

    No outlier removal, no denoise, no hole fill, no decimation. This represents the
    baseline 'what if we skip the cleaning pipeline' visual for the comparison panel.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError('empty point cloud')

    pcd = estimate_and_orient_normals(pcd)
    cloud_pts = np.asarray(pcd.points)

    candidates = _gather_candidates(pcd)
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(candidates, key=lambda item: _mesh_score(item[1], cloud_pts, item[0]))
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

    pcd = bilateral_normal_denoise(pcd)
    stats['after_denoise'] = _stat(pcd)

    pcd = estimate_and_orient_normals(pcd)
    cloud_pts = np.asarray(pcd.points)

    candidates = _gather_candidates(pcd)
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(candidates, key=lambda item: _mesh_score(item[1], cloud_pts, item[0]))
    stats['strategy'] = strategy
    stats['candidates_evaluated'] = len(candidates)
    stats['triangles_post_recon'] = len(mesh.triangles)
    stats['watertight_post_recon'] = bool(mesh.is_watertight())
    stats['boundary_edges_post_recon'] = _boundary_edge_count(mesh)
    stats['chamfer_pre_decimate'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

    mesh, fill_stats = aggressive_fill_holes(mesh)
    stats.update(fill_stats)
    stats['watertight_post_holefill'] = bool(mesh.is_watertight())
    stats['boundary_edges_post_holefill'] = _boundary_edge_count(mesh)

    mesh, deci_stats = decimate(mesh, cloud_pts)
    stats.update(deci_stats)

    stats['final_triangles'] = len(mesh.triangles)
    stats['final_vertices'] = len(mesh.vertices)
    stats['watertight'] = bool(mesh.is_watertight())
    stats['edge_manifold'] = bool(mesh.is_edge_manifold())
    stats['chamfer_final'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

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
                f"wt={result['watertight']} chamfer={result.get('chamfer_final', 0):.4f}"
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
