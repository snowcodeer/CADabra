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
    # Higher max depth + more depths -> better resolution on small/shallow features.
    # Poisson respects vertex normals, so it produces vertical walls between
    # surfaces of different heights instead of diagonal bridges.
    'poisson_depths': (8, 9, 10, 11),
    'poisson_density_quantile': 0.015,   # lower -> keeps more wall coverage, fewer false holes
    'envelope_radius_factor': 2.2,       # x avg NN distance; crops Poisson shrink-wrap membrane
    # Smaller alpha values added: when a feature is shallower than alpha, alpha
    # shape connects the top of the feature to the surrounding base at a
    # diagonal instead of forming a vertical wall. Adding 0.025 / 0.04 lets us
    # capture step heights down to ~2.5% of bbox diag (e.g. 2-3mm on a 100mm
    # part) without the bridging artifact.
    'alpha_factors': (0.025, 0.04, 0.06, 0.085),
    'ball_pivot_radius_factors': (1.0, 1.5, 2.0, 2.8),
}

SCORE_PARAMS = {
    'chamfer_sample_count': 4096,
    'chamfer_weight': 1.0,
    'boundary_penalty': 0.6,             # weight x (boundary_edges / triangle_count) - punishes gappy meshes
    # Catches the "shallow-extrude diagonal connection" failure mode: alpha
    # shape with too-large alpha produces long triangles bridging across short
    # vertical walls. Score = penalty * (long_edges / total_edges) where a long
    # edge is one with length > long_edge_threshold * avg_NN of the cloud.
    'long_edge_penalty': 1.5,
    'long_edge_threshold': 3.5,          # x avg NN distance
    'poisson_bonus': 0.020,              # strongly prefer poisson family - it respects normals so no diagonal bridges
    'alpha_bonus': 0.002,
    'watertight_bonus': 0.006,           # closed meshes win ties
    'manifold_bonus': 0.002,
    'size_penalty': 1e-6,
    # Reward candidates whose boundary topology preserves visible real holes.
    # This is what stops alpha-shape candidates from winning by closing CAD
    # bores during construction (before hole_fill protection ever runs). One
    # detected real hole is worth more than a watertight bonus, so a part with
    # actual holes will always beat a candidate that bridged them.
    # Capped to avoid rewarding fragmented-topology candidates that produce
    # many false-positive "real holes" (e.g. an alpha-shape that splattered
    # the boundary into dozens of small loops).
    'real_hole_bonus': 0.025,
    'real_hole_bonus_cap': 8,
}

HOLE_FILL_PARAMS = {
    # First pass closes small/medium boundary loops, second pass aggressively closes
    # any remaining membrane-fringe artifacts from envelope cropping. Two passes work
    # better than a single huge threshold because Open3D's fill_holes seems to behave
    # differently on a freshly cleaned-up mesh.
    'pass1_frac': 0.30,
    'pass2_frac': 0.80,
    'min_loop_length_abs': 0.20,
    # When True, reconstruct_noisy_for_sample runs aggressive_fill_holes on the
    # winning candidate before reporting metrics. Off by default to preserve the
    # raw-recon look; the vision tuner can flip this per-sample to chase
    # watertightness when a recon is mostly closed but has a few stubborn holes.
    'enable_in_noisy': False,
    # Real-hole protection. Before any fill pass we extract every boundary loop
    # and classify it. A loop is treated as a "real through-hole" if it is
    # large+circular OR very large regardless of shape; loops in that set are
    # NEVER closed (we cap effective pass1/pass2 below their smallest perimeter).
    # This is what stops the tuner from inadvertently sealing CAD bores.
    'protect_real_holes': True,
    'circular_protect_perim_frac': 0.14,   # min perim to even consider for circular protection (frac of bbox diag)
    'circularity_threshold': 0.65,         # 4*pi*A / P^2 above this counts as "circular"
    'planarity_threshold': 0.025,          # max planar residual / bbox_diag for circular classification
    'absolute_protect_perim_frac': 0.30,   # any loop above this perimeter is protected regardless of shape
    'protect_safety_margin': 0.90,         # cap fill threshold to (margin * smallest protected perim)
    # "Meshy/Tripo-style cohesive close": after detecting protected real holes,
    # iteratively fill EVERY remaining boundary loop (with hole_size capped just
    # below the smallest protected perimeter) until the boundary count converges.
    # Off by default for the legacy raw-recon path; the cohesion tuner enables
    # it automatically because the contract there is "clean orthographic views",
    # not "low chamfer to noisy points".
    'meshy_close': False,
    'meshy_max_iters': 4,
    'meshy_unprotected_cap_frac': 1.5,     # if no real holes detected, cap = bbox_diag * this (effectively unlimited)
    # Edge-aware safety brake for meshy_close: after each fill iteration, drop any
    # newly added triangle whose longest edge spans more than this fraction of the
    # bbox diagonal. This is what stops the "connect dots far across" bridging
    # pattern even when the model enables meshy_close on a part with no
    # protected real holes detected. Lower = safer, higher = more aggressive
    # closure.
    'meshy_close_max_edge_frac': 0.18,
    # Topological through-hole detector. For every boundary loop we cast rays
    # along its best-fit normal axis from the loop centroid, slightly inside the
    # part. If a ray hits the mesh nearby on either side, the loop is a true
    # through-bore (or a deep cavity opening to it) and is force-protected
    # regardless of perimeter / circularity heuristics. This is what catches
    # the small CAD bores that the geometric heuristics miss.
    'through_hole_detection': True,
    'through_hole_max_depth_frac': 0.55,   # ray hit must come back within bbox_diag * this
    'through_hole_inset_frac': 0.003,      # nudge ray origin this far inside the loop along normal
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

PLANE_SNAP_PARAMS = {
    # Distance, as a fraction of the mesh bbox diagonal, within which a vertex is
    # considered an inlier of a detected plane. RANSAC uses this for plane
    # acceptance; smaller -> only well-aligned vertices snap to the plane.
    'inlier_distance_frac': 0.008,
    # Multiplier on inlier_distance for deciding which planes a vertex is "near"
    # for the snap step. Kept tight (=1.0) so a vertex only snaps to a plane
    # when it really is on it, and only ends up at a corner when it really sits
    # near 2-3 planes simultaneously (avoids collapsing curved or off-plane
    # vertices).
    'snap_distance_multiplier': 1.0,
    # Minimum inlier count for a plane to be accepted. Filters tiny noise planes.
    'min_plane_inliers': 80,
    # Maximum planes to detect. CAD parts rarely have more than this.
    'max_planes': 20,
    # RANSAC iterations per plane.
    'ransac_iterations': 400,
    # How many full snap passes to run.
    'iterations': 2,
    # Cap on per-vertex movement as a multiplier of snap distance (defensive
    # against ill-conditioned multi-plane intersections producing wild points).
    'max_move_multiplier': 1.5,
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


def _detect_planes(verts: np.ndarray, distance: float) -> list[np.ndarray]:
    """Iterative RANSAC plane segmentation. Returns list of normalized
    (a, b, c, d) plane equations, largest first."""
    if len(verts) < PLANE_SNAP_PARAMS['min_plane_inliers']:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    used = np.zeros(len(verts), dtype=bool)
    planes: list[np.ndarray] = []

    for _ in range(PLANE_SNAP_PARAMS['max_planes']):
        if (~used).sum() < PLANE_SNAP_PARAMS['min_plane_inliers']:
            break
        active_idx = np.where(~used)[0]
        sub = pcd.select_by_index(active_idx.tolist())
        try:
            model, inliers = sub.segment_plane(
                distance_threshold=float(distance),
                ransac_n=3,
                num_iterations=int(PLANE_SNAP_PARAMS['ransac_iterations']),
            )
        except Exception:
            break
        if len(inliers) < PLANE_SNAP_PARAMS['min_plane_inliers']:
            break
        a, b, c, d = model
        norm = float(np.sqrt(a * a + b * b + c * c))
        if norm < 1e-9:
            break
        planes.append(np.array([a / norm, b / norm, c / norm, d / norm], dtype=np.float64))
        used[active_idx[np.asarray(inliers)]] = True

    return planes


def _snap_vertex_to_planes(v: np.ndarray, near_planes: np.ndarray) -> np.ndarray | None:
    """Project a vertex onto the intersection of nearby planes.

    1 plane  -> orthogonal projection onto plane
    2 planes -> orthogonal projection onto their line of intersection
    3+ planes -> least-squares solution (closest point to all planes,
                 effectively the corner where 3 planes meet)
    """
    n_planes = len(near_planes)
    if n_planes == 0:
        return None
    if n_planes == 1:
        a, b, c, d = near_planes[0]
        normal = np.array([a, b, c])
        signed = a * v[0] + b * v[1] + c * v[2] + d
        return v - signed * normal
    if n_planes == 2:
        n1, n2 = near_planes[0, :3], near_planes[1, :3]
        d1, d2 = near_planes[0, 3], near_planes[1, 3]
        direction = np.cross(n1, n2)
        dn = float(np.linalg.norm(direction))
        if dn < 1e-6:
            a, b, c, d = near_planes[0]
            normal = np.array([a, b, c])
            return v - (a * v[0] + b * v[1] + c * v[2] + d) * normal
        direction /= dn
        # Solve [n1; n2] x = [-d1; -d2] in least squares for a point on the line.
        A = np.stack([n1, n2], axis=0)
        rhs = np.array([-d1, -d2])
        x0, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        t = float(np.dot(v - x0, direction))
        return x0 + t * direction
    # >= 3 planes: corner is the least-squares closest point to all of them.
    A = near_planes[:, :3]
    rhs = -near_planes[:, 3]
    x_corner, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    return x_corner


def plane_snap(
    mesh: o3d.geometry.TriangleMesh,
    cached_planes: list[np.ndarray] | None = None,
) -> tuple[o3d.geometry.TriangleMesh, list[np.ndarray], dict]:
    """Detect dominant planes via RANSAC and snap mesh vertices to them.

    For CAD-style prismatic parts this sharpens edges and reconstructs corners
    that the noisy reconstruction left rounded. Vertices near a single plane
    are projected onto that plane; vertices near two planes get pulled to the
    intersection line (sharp edge); vertices near 3+ planes get pulled to the
    corner intersection point.

    `cached_planes`: if provided, plane RE-DETECTION is skipped and the supplied
    planes are used. This is critical for the post-hole-fill snap pass:
    re-fitting on the filled mesh would let bridging triangles bias the planes
    away from the original surface and pull good vertices off-target.
    """
    verts = np.asarray(mesh.vertices)
    if len(verts) < PLANE_SNAP_PARAMS['min_plane_inliers']:
        return mesh, [], {'plane_snap_planes': 0, 'plane_snap_iterations': 0, 'plane_snap_moved_pct': 0.0}

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    distance = max(diag * PLANE_SNAP_PARAMS['inlier_distance_frac'], 1e-5)
    snap_dist = distance * PLANE_SNAP_PARAMS['snap_distance_multiplier']
    cap = snap_dist * PLANE_SNAP_PARAMS['max_move_multiplier']

    snapped = verts.astype(np.float64).copy()
    moved_pct = 0.0
    detected_planes: list[np.ndarray] = list(cached_planes) if cached_planes else []
    iterations = 1 if cached_planes else PLANE_SNAP_PARAMS['iterations']

    for _it in range(iterations):
        if cached_planes:
            planes = detected_planes
        else:
            planes = _detect_planes(snapped, distance)
            detected_planes = planes
        if not planes:
            break
        plane_array = np.stack(planes, axis=0)
        normals = plane_array[:, :3]
        ds = plane_array[:, 3]

        signed = snapped @ normals.T + ds
        abs_signed = np.abs(signed)

        new_verts = snapped.copy()
        moved = 0
        for i in range(len(snapped)):
            mask = abs_signed[i] <= snap_dist
            if not np.any(mask):
                continue
            near = plane_array[mask]
            target = _snap_vertex_to_planes(snapped[i], near)
            if target is None:
                continue
            delta = target - snapped[i]
            d_move = float(np.linalg.norm(delta))
            if d_move > cap:
                target = snapped[i] + delta * (cap / d_move)
            new_verts[i] = target
            moved += 1

        moved_pct = moved / len(snapped)
        snapped = new_verts

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(snapped)
    out.triangles = mesh.triangles
    out.compute_vertex_normals()
    # Skip merge_close_vertices here - it can collapse triangles into degenerate
    # / non-manifold configurations that _cleanup_mesh then removes, re-opening
    # the mesh. The basic dedupe inside _cleanup_mesh is enough.
    out = _cleanup_mesh(out)
    return out, detected_planes, {
        'plane_snap_planes': int(len(detected_planes)),
        'plane_snap_iterations': int(iterations),
        'plane_snap_moved_pct': round(moved_pct, 4),
    }


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


def _long_edge_density(mesh: o3d.geometry.TriangleMesh, avg_nn: float) -> float:
    """Fraction of unique edges whose length exceeds long_edge_threshold * avg_nn.

    Long edges typically come from alpha-shape bridges across shallow features
    (e.g. a 3mm extrude turning into a diagonal slope to the base) or from
    skinny triangles spanning concavities. Penalizing this density steers the
    selector away from those bridge artifacts.
    """
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    if len(tris) == 0 or len(verts) == 0 or avg_nn <= 0:
        return 0.0
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    p0 = verts[edges[:, 0]]
    p1 = verts[edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    threshold = avg_nn * SCORE_PARAMS['long_edge_threshold']
    return float((lengths > threshold).mean())


def _mesh_score(
    mesh: o3d.geometry.TriangleMesh,
    cloud_pts: np.ndarray,
    strategy: str,
    avg_nn: float | None = None,
) -> float:
    """Higher is better.

    Primary: Chamfer fidelity to the input cloud.
    Secondary: penalize boundary-edge density (swiss-cheese) and long-edge
        density (diagonal bridges across shallow features).
    Tertiary: prefer Poisson (respects normals) over alpha/BP at similar fidelity,
        plus small bonuses for watertight/manifold.
    """
    chamfer = _chamfer_to_cloud(mesh, cloud_pts)
    score = -SCORE_PARAMS['chamfer_weight'] * chamfer

    n_tris = max(len(mesh.triangles), 1)
    boundary_density = _boundary_edge_count(mesh) / n_tris
    score -= SCORE_PARAMS['boundary_penalty'] * boundary_density

    if avg_nn is None:
        avg_nn = _avg_nn_distance(cloud_pts)
    long_density = _long_edge_density(mesh, avg_nn)
    score -= SCORE_PARAMS['long_edge_penalty'] * long_density

    if strategy.startswith('poisson'):
        score += SCORE_PARAMS['poisson_bonus']
    elif strategy.startswith('alpha'):
        score += SCORE_PARAMS['alpha_bonus']

    if mesh.is_watertight():
        score += SCORE_PARAMS['watertight_bonus']
    if mesh.is_edge_manifold():
        score += SCORE_PARAMS['manifold_bonus']
    score -= SCORE_PARAMS['size_penalty'] * len(mesh.triangles)

    # Reward candidates that retain real-hole topology. _smallest_protected_perim
    # also reports the count, so we reuse it. This dominates over the modest
    # boundary_density penalty so candidates that bridge bores can't win on
    # cleanliness alone. The count is capped to avoid rewarding fragmented
    # candidates that produce many false-positive "real holes".
    real_hole_bonus = SCORE_PARAMS.get('real_hole_bonus', 0.0)
    if real_hole_bonus > 0:
        diag = _bbox_diag(mesh)
        if diag > 0:
            _, n_real, _ = _smallest_protected_perim(mesh, diag)
            cap = int(SCORE_PARAMS.get('real_hole_bonus_cap', 8))
            score += real_hole_bonus * min(n_real, cap)
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


def _extract_boundary_loops(mesh: o3d.geometry.TriangleMesh) -> list[list[int]]:
    """Return each closed boundary loop as an ordered list of vertex indices.

    Walks the boundary-edge graph one edge at a time so vertices that belong to
    multiple distinct loops (pinch points) are handled correctly.
    """
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return []
    e1 = triangles[:, [0, 1]]
    e2 = triangles[:, [1, 2]]
    e3 = triangles[:, [2, 0]]
    all_edges = np.concatenate([e1, e2, e3], axis=0)
    sorted_edges = np.sort(all_edges, axis=1)
    keys, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_keys = keys[counts == 1]
    if len(boundary_keys) == 0:
        return []

    from collections import defaultdict
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in boundary_keys:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    for seed in list(adj.keys()):
        for first in adj[seed]:
            ekey = (seed, first) if seed < first else (first, seed)
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)
            loop = [seed, first]
            prev, cur = seed, first
            while True:
                next_step = None
                for nbr in adj[cur]:
                    if nbr == prev:
                        continue
                    nk = (cur, nbr) if cur < nbr else (nbr, cur)
                    if nk in visited_edges:
                        continue
                    next_step = nbr
                    visited_edges.add(nk)
                    break
                if next_step is None:
                    break
                if next_step == seed:
                    break
                loop.append(next_step)
                prev, cur = cur, next_step
                if len(loop) > len(adj):
                    break
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _classify_boundary_loop(loop_pts: np.ndarray, bbox_diag: float) -> dict:
    """Compute perimeter, planar-fit area, circularity, planarity residual, and
    the loop's centroid + best-fit normal so the through-hole detector can reuse
    them without re-running SVD."""
    n = len(loop_pts)
    if n < 3:
        return {'perim': 0.0, 'area': 0.0, 'circularity': 0.0, 'planarity': 1.0,
                'centroid': np.zeros(3), 'normal': np.array([0.0, 0.0, 1.0]),
                'through_hole': False}
    edges = loop_pts[(np.arange(n) + 1) % n] - loop_pts
    perim = float(np.linalg.norm(edges, axis=1).sum())
    centroid = loop_pts.mean(axis=0)
    centered = loop_pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {'perim': perim, 'area': 0.0, 'circularity': 0.0, 'planarity': 1.0,
                'centroid': centroid, 'normal': np.array([0.0, 0.0, 1.0]),
                'through_hole': False}
    normal = vh[-1]
    basis_u = vh[0]
    basis_v = vh[1]
    proj_u = centered @ basis_u
    proj_v = centered @ basis_v
    area = float(0.5 * abs(np.sum(proj_u * np.roll(proj_v, -1) - np.roll(proj_u, -1) * proj_v)))
    planar_residual = float(np.abs(centered @ normal).max() / max(bbox_diag, 1e-9))
    circularity = float(4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    return {'perim': perim, 'area': area, 'circularity': circularity, 'planarity': planar_residual,
            'centroid': centroid, 'normal': normal, 'through_hole': False}


def _detect_through_holes(mesh: o3d.geometry.TriangleMesh, classified: list[dict],
                          bbox_diag: float) -> None:
    """Mark any boundary loop that visually corresponds to a through-bore.

    For each loop we shoot two rays from a point slightly inside the part along
    the loop's best-fit normal axis (i.e. into the part). If a ray hits the
    mesh again within bbox_diag * through_hole_max_depth_frac, the loop opens
    onto an internal cavity / through-bore. Marks `through_hole=True` in place.

    This is the safety net that catches the small/oblong CAD bores the
    perimeter+circularity heuristic misses (so meshy_close / aggressive_fill
    can't bridge them later).
    """
    if not classified or not HOLE_FILL_PARAMS.get('through_hole_detection', True):
        return
    if len(mesh.triangles) == 0:
        return
    try:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
    except Exception:
        return

    inset = float(HOLE_FILL_PARAMS.get('through_hole_inset_frac', 0.003)) * bbox_diag
    max_depth = float(HOLE_FILL_PARAMS.get('through_hole_max_depth_frac', 0.55)) * bbox_diag
    eps = 1e-5 * max(bbox_diag, 1e-6)

    rays = []
    for c in classified:
        n = c.get('normal')
        cen = c.get('centroid')
        if n is None or cen is None or float(np.linalg.norm(n)) < 1e-9:
            rays.extend([[0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, -1.0]])
            continue
        n_unit = n / (np.linalg.norm(n) + 1e-12)
        # Origin nudged BOTH ways along the normal so we can see whether either
        # side of the loop has more mesh further along (which is the topological
        # signature of a tunnel / cavity vs. a flat-face loop).
        o_pos = (cen + n_unit * (eps + inset)).astype(np.float32)
        d_pos = n_unit.astype(np.float32)
        o_neg = (cen - n_unit * (eps + inset)).astype(np.float32)
        d_neg = (-n_unit).astype(np.float32)
        rays.append([*o_pos, *d_pos])
        rays.append([*o_neg, *d_neg])
    rays_arr = np.asarray(rays, dtype=np.float32)
    try:
        ans = scene.cast_rays(o3d.core.Tensor(rays_arr))
        t_hit = ans["t_hit"].numpy()
    except Exception:
        return

    for i, c in enumerate(classified):
        t_pos = float(t_hit[2 * i])
        t_neg = float(t_hit[2 * i + 1])
        # A through-hole has mesh on the opposite side reachable within
        # max_depth on EITHER probe direction (we don't require both because
        # blind cavities still count as real openings).
        if (np.isfinite(t_pos) and t_pos < max_depth) or (np.isfinite(t_neg) and t_neg < max_depth):
            c['through_hole'] = True


def _is_real_hole(c: dict, bbox_diag: float) -> bool:
    if c.get('through_hole'):
        return True
    perim = c['perim']
    if perim <= 0:
        return False
    abs_thresh = HOLE_FILL_PARAMS['absolute_protect_perim_frac'] * bbox_diag
    if perim >= abs_thresh:
        return True
    circ_perim_thresh = HOLE_FILL_PARAMS['circular_protect_perim_frac'] * bbox_diag
    if (
        perim >= circ_perim_thresh
        and c['circularity'] >= HOLE_FILL_PARAMS['circularity_threshold']
        and c['planarity'] <= HOLE_FILL_PARAMS['planarity_threshold']
    ):
        return True
    return False


def _smallest_protected_perim(mesh: o3d.geometry.TriangleMesh, bbox_diag: float) -> tuple[float, int, list[dict]]:
    """Return (smallest_protected_perim, protected_count, classified_loops).

    smallest_protected_perim is +inf if no loops are protected. When the
    protect_real_holes flag is off, we still classify (so callers like
    _mesh_score can use n_real) but report inf so the hole-fill cap is unbounded.

    Each loop is first classified geometrically (perim/circularity/planarity);
    we then run the topological through-hole detector to mark any loop that
    opens onto an internal cavity even if the geometric heuristics missed it.
    """
    loops = _extract_boundary_loops(mesh)
    if not loops:
        return float('inf'), 0, []
    verts = np.asarray(mesh.vertices)
    classified: list[dict] = []
    for loop in loops:
        classified.append(_classify_boundary_loop(verts[loop], bbox_diag))
    _detect_through_holes(mesh, classified, bbox_diag)
    smallest = float('inf')
    protected = 0
    for c in classified:
        c['protected'] = _is_real_hole(c, bbox_diag)
        if c['protected']:
            protected += 1
            if c['perim'] < smallest:
                smallest = c['perim']
    if not HOLE_FILL_PARAMS.get('protect_real_holes', True):
        smallest = float('inf')
    return smallest, protected, classified


def aggressive_fill_holes(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Close as many holes as possible while preserving truly large openings.

    Two-pass strategy:
      pass 1: hole_size = pass1_frac * bbox_diag - closes simple noise loops cleanly
      pass 2: hole_size = pass2_frac * bbox_diag - mops up the shredded membrane fringe
              that envelope-cropped Poisson reconstructions tend to leave behind.

    Real-hole protection (HOLE_FILL_PARAMS['protect_real_holes']): we extract
    every boundary loop, classify each by perimeter / circularity / planarity,
    and cap pass1/pass2 below the smallest protected loop's perimeter so we
    never bridge a true through-hole.
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
        stats['hole_fill_protected_loops'] = 0
        return mesh, stats

    pass1 = max(diag * HOLE_FILL_PARAMS['pass1_frac'], HOLE_FILL_PARAMS['min_loop_length_abs'])
    pass2 = max(diag * HOLE_FILL_PARAMS['pass2_frac'], HOLE_FILL_PARAMS['min_loop_length_abs'])

    smallest_prot, prot_count, _ = _smallest_protected_perim(mesh, diag)
    if smallest_prot != float('inf'):
        cap = HOLE_FILL_PARAMS['protect_safety_margin'] * smallest_prot
        pass1 = min(pass1, cap)
        pass2 = min(pass2, cap)
    stats['hole_fill_protected_loops'] = prot_count
    stats['hole_fill_protected_min_perim'] = round(smallest_prot, 6) if smallest_prot != float('inf') else None
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


def _drop_long_fill_triangles(before: o3d.geometry.TriangleMesh,
                              after: o3d.geometry.TriangleMesh,
                              max_edge: float) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Strip any triangles in `after` that did not exist in `before` AND whose
    longest edge is longer than `max_edge`. Returns the filtered mesh and the
    number of triangles dropped.

    This is the safety brake against fill_holes "connecting dots far across".
    A triangle is considered "newly added" if any of its three vertex indices
    is >= len(before.vertices) (Open3D appends new fill vertices at the end)
    OR if the (sorted) vertex triple isn't in `before`'s triangle set.
    """
    if max_edge <= 0:
        return after, 0
    n_old_v = len(before.vertices)
    after_tris = np.asarray(after.triangles)
    after_verts = np.asarray(after.vertices)
    if len(after_tris) == 0:
        return after, 0

    before_tri_keys: set[tuple[int, int, int]] = set()
    if len(before.triangles) > 0:
        before_arr = np.sort(np.asarray(before.triangles), axis=1)
        for row in before_arr:
            before_tri_keys.add((int(row[0]), int(row[1]), int(row[2])))

    a, b, c = after_tris[:, 0], after_tris[:, 1], after_tris[:, 2]
    e0 = np.linalg.norm(after_verts[a] - after_verts[b], axis=1)
    e1 = np.linalg.norm(after_verts[b] - after_verts[c], axis=1)
    e2 = np.linalg.norm(after_verts[c] - after_verts[a], axis=1)
    longest = np.maximum.reduce([e0, e1, e2])

    sorted_tris = np.sort(after_tris, axis=1)
    keep = np.ones(len(after_tris), dtype=bool)
    dropped = 0
    for i in range(len(after_tris)):
        if longest[i] <= max_edge:
            continue
        # Long edge: only drop if this triangle wasn't already in `before` (so
        # we never delete original mesh triangles, only newly synthesized fills).
        is_new = (
            after_tris[i, 0] >= n_old_v
            or after_tris[i, 1] >= n_old_v
            or after_tris[i, 2] >= n_old_v
            or (int(sorted_tris[i, 0]), int(sorted_tris[i, 1]), int(sorted_tris[i, 2])) not in before_tri_keys
        )
        if is_new:
            keep[i] = False
            dropped += 1
    if not dropped:
        return after, 0

    filtered = o3d.geometry.TriangleMesh()
    filtered.vertices = after.vertices
    filtered.triangles = o3d.utility.Vector3iVector(after_tris[keep])
    filtered.remove_unreferenced_vertices()
    return filtered, dropped


def meshy_close_mesh(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Meshy/Tripo-style cohesive close (edge-guarded).

    Detects which boundary loops are "real holes" (large or circular+planar OR
    raycast-confirmed through-bores) and treats them as protected. Then
    iteratively closes other boundary loops by invoking the tensor fill_holes
    API with hole_size capped just below the smallest protected perimeter,
    AND drops any newly synthesized triangle whose longest edge exceeds
    `meshy_close_max_edge_frac * bbox_diag`. The edge guard is the safety net
    that stops fill_holes from "connecting dots far across" intentional gaps.

    Result: a more cohesive mesh whose only remaining openings are the real
    through-holes plus any genuinely large gaps. Disabled by default in the
    cohesion baseline now; downstream image-gen handles cosmetic gap cleanup
    on the rendered orthographic views instead of forcing geometric closure.
    """
    diag = _bbox_diag(mesh)
    boundary_before = _boundary_edge_count(mesh)
    stats = {
        'meshy_close_diag': round(diag, 6),
        'meshy_close_boundary_before': boundary_before,
    }
    if boundary_before == 0:
        stats['meshy_close_method'] = 'noop'
        stats['meshy_close_iterations'] = 0
        stats['meshy_close_boundary_after'] = 0
        stats['meshy_close_protected_loops'] = 0
        stats['meshy_close_cap'] = 0.0
        stats['meshy_close_protected_min_perim'] = None
        stats['meshy_close_long_tris_dropped'] = 0
        return mesh, stats

    smallest_prot, prot_count, _ = _smallest_protected_perim(mesh, diag)
    if smallest_prot != float('inf'):
        cap = HOLE_FILL_PARAMS['protect_safety_margin'] * smallest_prot
    else:
        cap = diag * float(HOLE_FILL_PARAMS.get('meshy_unprotected_cap_frac', 1.5))

    stats['meshy_close_protected_loops'] = prot_count
    stats['meshy_close_protected_min_perim'] = round(smallest_prot, 6) if smallest_prot != float('inf') else None
    stats['meshy_close_cap'] = round(cap, 6)

    max_edge = float(HOLE_FILL_PARAMS.get('meshy_close_max_edge_frac', 0.18)) * diag

    max_iters = max(1, int(HOLE_FILL_PARAMS.get('meshy_max_iters', 4)))
    current = mesh
    last_boundary = boundary_before
    iters_done = 0
    total_dropped = 0
    for _ in range(max_iters):
        filled = _fill_with_tensor(current, cap)
        if filled is None:
            break
        filled, dropped = _drop_long_fill_triangles(current, filled, max_edge)
        total_dropped += dropped
        filled = _cleanup_mesh(filled)
        new_boundary = _boundary_edge_count(filled)
        iters_done += 1
        current = filled
        if new_boundary > 0:
            smallest_prot, prot_count, _ = _smallest_protected_perim(current, diag)
            if smallest_prot != float('inf'):
                cap = HOLE_FILL_PARAMS['protect_safety_margin'] * smallest_prot
        if new_boundary == 0 or new_boundary == last_boundary:
            last_boundary = new_boundary
            break
        last_boundary = new_boundary

    stats['meshy_close_iterations'] = iters_done
    stats['meshy_close_boundary_after'] = last_boundary
    stats['meshy_close_method'] = 'open3d_tensor_iterative_edgeguard'
    stats['meshy_close_final_protected_loops'] = prot_count
    stats['meshy_close_long_tris_dropped'] = total_dropped
    stats['meshy_close_max_edge'] = round(max_edge, 6)
    return current, stats


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
    avg_nn = _avg_nn_distance(cloud_pts)

    candidates = _gather_candidates(pcd)
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(
        candidates,
        key=lambda item: _mesh_score(item[1], cloud_pts, item[0], avg_nn=avg_nn),
    )

    fill_stats: dict | None = None
    if HOLE_FILL_PARAMS.get('enable_in_noisy'):
        mesh, fill_stats = aggressive_fill_holes(mesh)
        # Re-run cleanup so any non-manifold remnants from the fill are dropped before metrics.
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    meshy_stats: dict | None = None
    if HOLE_FILL_PARAMS.get('meshy_close'):
        mesh, meshy_stats = meshy_close_mesh(mesh)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    strategy_tag = strategy
    if HOLE_FILL_PARAMS.get('enable_in_noisy'):
        strategy_tag += '+holefill'
    if HOLE_FILL_PARAMS.get('meshy_close'):
        strategy_tag += '+meshyclose'

    stats = {
        'strategy': strategy_tag,
        'triangles': len(mesh.triangles),
        'vertices': len(mesh.vertices),
        'watertight': bool(mesh.is_watertight()),
        'edge_manifold': bool(mesh.is_edge_manifold()),
        'long_edge_density': round(_long_edge_density(mesh, avg_nn), 4),
        'boundary_edges': _boundary_edge_count(mesh),
        'chamfer': round(_chamfer_to_cloud(mesh, cloud_pts), 6),
        'hole_fill': fill_stats,
        'meshy_close': meshy_stats,
        'candidates_evaluated': len(candidates),
        'candidate_summary': [
            {
                'strategy': cand_strat,
                'triangles': len(cand_mesh.triangles),
                'long_edge_density': round(_long_edge_density(cand_mesh, avg_nn), 4),
                'chamfer': round(_chamfer_to_cloud(cand_mesh, cloud_pts), 6),
                'score': round(_mesh_score(cand_mesh, cloud_pts, cand_strat, avg_nn=avg_nn), 6),
            }
            for cand_strat, cand_mesh in candidates
        ],
    }
    return mesh, stats


def reconstruct_for_sample(ply_path: Path) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Sharp-preserving reconstruction.

    Pipeline (no smoothing or aggressive decimation that would round corners):
      1. Light statistical outlier removal only - no voxel downsample, no
         bilateral denoise (those are the corner-killers).
      2. Estimate normals.
      3. Multi-strategy reconstruction (Poisson + alpha + ball-pivot).
      4. Plane snap pass A - sharpens existing edges/corners on the raw recon.
      5. Aggressive two-pass hole fill - extrapolates surface across gaps.
      6. Plane snap pass B - the new fill vertices snap onto the same planes
         so the filled regions extend cleanly into corners instead of bowing.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError('empty point cloud')

    stats: dict = {'input_points': _stat(pcd)}

    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=CLEAN_PARAMS['statistical_nb_neighbors'],
        std_ratio=CLEAN_PARAMS['statistical_std_ratio'],
    )
    pcd = cleaned
    stats['after_outlier_removal'] = _stat(pcd)

    pcd = estimate_and_orient_normals(pcd)
    cloud_pts = np.asarray(pcd.points)
    avg_nn = _avg_nn_distance(cloud_pts)

    candidates = _gather_candidates(pcd)
    if not candidates:
        raise RuntimeError('reconstruction produced empty mesh')

    strategy, mesh = max(
        candidates,
        key=lambda item: _mesh_score(item[1], cloud_pts, item[0], avg_nn=avg_nn),
    )
    stats['strategy'] = strategy
    stats['candidates_evaluated'] = len(candidates)
    stats['triangles_post_recon'] = len(mesh.triangles)
    stats['watertight_post_recon'] = bool(mesh.is_watertight())
    stats['boundary_edges_post_recon'] = _boundary_edge_count(mesh)
    stats['chamfer_post_recon'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

    mesh, planes, snap_stats_a = plane_snap(mesh)
    stats['plane_snap_a'] = snap_stats_a
    stats['triangles_post_snap_a'] = len(mesh.triangles)
    stats['chamfer_post_snap_a'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

    mesh, fill_stats = aggressive_fill_holes(mesh)
    stats.update(fill_stats)
    stats['watertight_post_holefill'] = bool(mesh.is_watertight())
    stats['boundary_edges_post_holefill'] = _boundary_edge_count(mesh)
    stats['chamfer_post_holefill'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

    # Reuse the planes detected before fill so the bridging triangles can't
    # bias the plane fit; fill-vertices snap onto the same surfaces as the
    # original recon.
    mesh, _, snap_stats_b = plane_snap(mesh, cached_planes=planes)
    stats['plane_snap_b'] = snap_stats_b

    # Final hole fill to close anything snap pass B may have re-opened (snap
    # can create degenerate / non-manifold configurations that _cleanup_mesh
    # removes; this final fill catches the resulting tiny holes).
    mesh, final_fill_stats = aggressive_fill_holes(mesh)
    stats['final_fill'] = final_fill_stats
    stats['boundary_edges_final'] = _boundary_edge_count(mesh)

    stats['final_triangles'] = len(mesh.triangles)
    stats['final_vertices'] = len(mesh.vertices)
    stats['watertight'] = bool(mesh.is_watertight())
    stats['edge_manifold'] = bool(mesh.is_edge_manifold())
    stats['chamfer_final'] = round(_chamfer_to_cloud(mesh, cloud_pts), 6)

    return mesh, stats


def run_single(sample_id: str, ply_file: str) -> dict:
    """Generate the raw/noisy reconstruction only - that's what the selector
    page shows now. The cleaned/sharp+filled path was removed because the raw
    recon already preserves sharp edges; we just need to keep tuning candidate
    selection so it doesn't bridge shallow features at a diagonal.
    """
    ply_path = SRC_DIR / ply_file
    mesh, stats = reconstruct_noisy_for_sample(ply_path)

    out_name = f'{sample_id}_recon_noisy.stl'
    out_path = OUT_DIR / out_name
    ok = o3d.io.write_triangle_mesh(str(out_path), mesh)
    if not ok or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError('failed to write reconstructed stl')

    return {
        'sample_id': sample_id,
        'success': True,
        'recon_noisy_stl': out_name,
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
            print(
                f"{result['sample_id']}: strat={result['strategy']:<14} "
                f"tris={result['triangles']:>5} verts={result['vertices']} "
                f"wt={result['watertight']} bnd={result.get('boundary_edges', 0):>4} "
                f"long_edge={result.get('long_edge_density', 0)*100:.2f}% "
                f"chamfer={result.get('chamfer', 0):.4f}"
            )
        else:
            print(f"{result['sample_id']}: failed - {result.get('error', 'unknown error')}")

    (OUT_DIR / 'manifest.json').write_text(json.dumps(out_rows, indent=2))
    success = [r for r in out_rows if r.get('success')]
    watertight = sum(1 for r in success if r.get('watertight'))
    if success:
        avg_tris = int(round(sum(r['triangles'] for r in success) / len(success)))
        avg_verts = int(round(sum(r['vertices'] for r in success) / len(success)))
        avg_long = sum(r.get('long_edge_density', 0) for r in success) / len(success)
    else:
        avg_tris = avg_verts = 0
        avg_long = 0.0
    print(
        f'reconstructed {len(success)} of {len(out_rows)} | watertight: {watertight} | '
        f'avg tris: {avg_tris} | avg verts: {avg_verts} | avg long-edge: {avg_long*100:.2f}%'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-id')
    parser.add_argument('--ply-file')
    parser.add_argument(
        '--params-override',
        help='JSON string of {"RECON_PARAMS": {...}, "SCORE_PARAMS": {...}} to override module defaults for this run only.',
    )
    return parser.parse_args()


_TUPLE_KEYS = {
    'poisson_depths',
    'alpha_factors',
    'ball_pivot_radius_factors',
}


def _apply_param_overrides(payload: dict) -> dict:
    """Mutate RECON_PARAMS / SCORE_PARAMS / HOLE_FILL_PARAMS with caller-supplied overrides.
    Returns the merged effective config so the caller can record what was used.
    """
    applied: dict = {'RECON_PARAMS': {}, 'SCORE_PARAMS': {}, 'HOLE_FILL_PARAMS': {}}
    for key, val in (payload.get('RECON_PARAMS') or {}).items():
        if key not in RECON_PARAMS:
            continue
        if key in _TUPLE_KEYS and isinstance(val, list):
            val = tuple(val)
        RECON_PARAMS[key] = val
        applied['RECON_PARAMS'][key] = val
    for key, val in (payload.get('SCORE_PARAMS') or {}).items():
        if key not in SCORE_PARAMS:
            continue
        SCORE_PARAMS[key] = val
        applied['SCORE_PARAMS'][key] = val
    for key, val in (payload.get('HOLE_FILL_PARAMS') or {}).items():
        if key not in HOLE_FILL_PARAMS:
            continue
        HOLE_FILL_PARAMS[key] = val
        applied['HOLE_FILL_PARAMS'][key] = val
    return applied


def main() -> None:
    args = parse_args()
    applied_overrides: dict | None = None
    if args.params_override:
        try:
            payload = json.loads(args.params_override)
        except json.JSONDecodeError as exc:
            print(json.dumps({'success': False, 'error': f'invalid params-override json: {exc}'}))
            sys.exit(1)
        applied_overrides = _apply_param_overrides(payload)

    if args.sample_id and args.ply_file:
        result = run_single(args.sample_id, args.ply_file)
        if applied_overrides:
            result['applied_overrides'] = applied_overrides
        print(json.dumps(result))
    else:
        run_batch()


if __name__ == '__main__':
    main()
