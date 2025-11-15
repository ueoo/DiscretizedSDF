#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

from argparse import ArgumentParser

import open3d as o3d
import torch

from scene import Scene


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import RENDER_DICT
from scene import GaussianModel
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh


def reconstruct_set(model_path, scene, extractor):
    mesh_path = os.path.join(model_path, "meshes")
    os.makedirs(mesh_path, exist_ok=True)
    extractor.reconstruction(scene.getTrainCameras())
    name = "fuse.ply"
    depth_trunc = (extractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    o3d.io.write_triangle_mesh(os.path.join(mesh_path, name), mesh)
    print("mesh saved at {}".format(os.path.join(mesh_path, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(mesh_path, name.replace(".ply", "_post.ply")), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_path, name.replace(".ply", "_post.ply"))))


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(dataset.sh_degree, dataset.env_mode, dataset.env_res, dataset.use_sdf, True, True)
    render_fn = RENDER_DICT[pipeline.gaussian_type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    extractor = GaussianExtractor(scene, render_fn, pipeline, bg_color=bg_color)
    reconstruct_set(dataset.model_path, scene, extractor)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF")
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF")
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF")
    parser.add_argument("--num_cluster", default=50, type=int, help="Mesh: number of connected clusters to export")
    parser.add_argument("--unbounded", action="store_true", help="Mesh: using unbounded mode for meshing")
    parser.add_argument("--mesh_res", default=2048, type=int, help="Mesh: resolution for unbounded mesh extraction")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
