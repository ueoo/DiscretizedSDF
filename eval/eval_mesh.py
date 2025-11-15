import multiprocessing as mp

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln

from chamferdist import ChamferDistance
from tqdm import tqdm


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def evaluation_shinyblender(
    gt_mesh,
    out_mesh,
    vis_out_dir,
    downsample_density=0.3,
    patch_size=60,
    max_dist_d=100,
    max_dist_t=10,
    visualize_threshold=10,
    points_for_plane=None,
    nonvalid_bbox=None,
):
    mp.freeze_support()

    thresh = downsample_density

    pbar = tqdm(total=9)
    pbar.set_description("read data mesh")

    vertices = np.asarray(gt_mesh.vertices)
    triangles = np.asarray(gt_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description("sample pcd from mesh")

    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i : i + 1], v2[i : i + 1], tri_vert[i : i + 1, 0]) for i in range(len(n1))),
            chunksize=1024,
        )

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description("random shuffle pcd index")
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description("downsample pcd")
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description("read STL pcd")
    stl = np.asarray(out_mesh.points)
    BB = np.array([stl.min(0), stl.max(0)])

    # compute lowest surface
    p1 = np.array(points_for_plane[0])
    p2 = np.array(points_for_plane[1])
    p3 = np.array(points_for_plane[2])
    v1 = p1 - p2
    v2 = p3 - p2

    normal = np.cross(v1, v2)
    # make sure the normal toward positive z
    if normal[-1] < 0:
        normal = np.cross(v2, v1)
    D = np.dot(normal, p1)

    pbar.update(1)
    pbar.set_description("masking data pcd")

    BB = BB.astype(np.float32)

    patch = patch_size
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    above = (data_in @ normal - D) > 0
    data_in_above = data_in[above]

    above_stl = (stl @ normal - D) > 0
    stl_above = stl[above_stl]

    if nonvalid_bbox is not None:
        aa = nonvalid_bbox[0]
        bb = nonvalid_bbox[1]

        mask_bbox = ((data_in_above >= bb) & (data_in_above <= aa)).sum(axis=-1) == 3
        mask_val = ~mask_bbox
    else:
        mask_val = np.ones_like(data_in_above)
        mask_val = mask_val.astype(bool)[:, 0]
    data_in_above = data_in_above[mask_val]

    pbar.update(1)
    pbar.set_description("compute data2stl")
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_above, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist_d].mean()

    pbar.update(1)
    pbar.set_description("compute stl2data")
    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist_t].mean()

    pbar.update(1)
    pbar.set_description("visualize error")
    vis_dist = visualize_threshold
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 0, 1]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist

    data_color[np.where(inbound)[0][above][mask_val]] = R * data_alpha + W * (1 - data_alpha)
    data_color[np.where(inbound)[0][above][mask_val][dist_d2s[:, 0] >= max_dist_d]] = G
    write_vis_pcd(f"{vis_out_dir}/vis_d2s.ply", data_down, data_color)

    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[np.where(above_stl)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[np.where(above_stl)[0][dist_s2d[:, 0] >= max_dist_t]] = G
    write_vis_pcd(f"{vis_out_dir}/vis_s2d.ply", stl, stl_color)

    pbar.update(1)
    pbar.set_description("done")
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    print(mean_d2s, mean_s2d, over_all)

    return mean_d2s, mean_s2d, over_all


import torch


def preprocess_mesh(data_pcd):
    thresh = 0.3
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    return data_down


scenes = ["car", "coffee", "helmet", "toaster"]

for scene in scenes:
    gt_mesh = o3d.io.read_triangle_mesh(f"data1/ShinyBlender_eval/{scene}/points_of_interest.ply")

    gt_pcd = np.asarray(gt_mesh.vertices)
    print(np.asarray(gt_mesh.vertices).shape)
    gt_pcd = torch.from_numpy(gt_pcd).unsqueeze(0).float()
    out_mesh = o3d.io.read_triangle_mesh(f"output/shiny/{scene}/meshes/fuse_post.ply")
    out_pcd = torch.from_numpy(np.asarray(out_mesh.vertices)).unsqueeze(0).float()

    chamferDist = ChamferDistance()
    cd = chamferDist(out_pcd.cuda(), gt_pcd.cuda(), bidirectional=True, point_reduction="mean") * 0.5
    print(scene, cd)
