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

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.NVDIFFREC import load_env, load_latlong_env, save_env_map
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.sh_utils import eval_sh
from utils.system_utils import mkdir_p, searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.env_mode = args.env_mode

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print(os.path.join(args.source_path, "transforms_train.json"))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            print("No data")

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling

        try:
            self.cameras_extent = self.scene_info.nerf_normalization["radius"]
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                    self.scene_info.train_cameras, resolution_scale, args
                )
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                    self.scene_info.test_cameras, resolution_scale, args
                )
        except:
            pass
        print(self.loaded_iter)
        if self.loaded_iter:
            self.resume()
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.env_mode == "envmap":
            envmap_path = os.path.join(self.model_path, f"envmap/iteration_{iteration}/envmap.hdr")
            mkdir_p(os.path.dirname(envmap_path))
            save_env_map(envmap_path, self.gaussians.envmap)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def resume(self):
        self.gaussians.load_ply(
            os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
        )

        env_mode = self.args.env_mode
        if env_mode == "envmap":
            fn = os.path.join(self.model_path, "envmap", "iteration_" + str(self.loaded_iter), "envmap.hdr")
            if os.path.exists(fn):
                self.gaussians.envmap = load_env(fn, scale=1.0)
                print(f"Load envmap from: {fn}")
            else:
                print("Env. map does not exist.")
