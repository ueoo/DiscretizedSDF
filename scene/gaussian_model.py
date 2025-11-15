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

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from tqdm import tqdm

from arguments import OptimizationParams
from scene.NVDIFFREC import create_trainable_env_rnd, load_env
from scene.NVDIFFREC.util import simple_fibonacci_sphere_sampling
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    flip_align_view,
    get_const_lr_func,
    get_expon_lr_func,
    get_maximum_axis,
    get_minimum_axis,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud, normal2quat
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(
        self, sh_degree: int, env_mode: str, env_res: int, use_sdf=False, use_metallic=True, sphere_init=False
    ):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def build_full_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance

        self.use_metallic = use_metallic
        self.use_sdf = use_sdf
        self.sphere_init = sphere_init

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._diffuse = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._sdfs = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)

        # brdf setting
        self.env_mode = env_mode
        self.env_res = env_res
        self._albedo = torch.empty(0)
        self._metallic = torch.empty(0)
        self._roughness = torch.empty(0)

        if self.env_mode == "envmap":
            self.envmap = create_trainable_env_rnd(self.env_res, scale=0.0, bias=0.8)
        else:
            raise NotImplementedError

        self.albedo_activation = torch.sigmoid
        self.metallic_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.roughness_bias = 0.0
        self.default_roughness = 0.6
        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.sdf_activation = lambda x: x

        def learnable_sigmoid(s, inv_deviation):
            tmp = torch.exp(-inv_deviation * s)
            return tmp / ((1 + tmp) ** 2) * 4

        self.opacity_projection = learnable_sigmoid
        self.inverse_deviation = torch.empty(0)
        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.full_cov_activation = build_full_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def set_envmap(self, envmap):
        self.envmap.set_base(envmap)
        self.envmap.build_mips()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_minimum_scaling(self):
        return torch.sort(self.scaling_activation(self._scaling), descending=False, dim=-1)[0][:, 0]

    @property
    def get_maximum_scaling(self):
        return torch.sort(self.scaling_activation(self._scaling), descending=False, dim=-1)[0][:, -1]

    @property
    def get_medium_scaling(self):
        return torch.sort(self.scaling_activation(self._scaling), descending=False, dim=-1)[0][:, 1]

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_sdf(self):
        return self.sdf_activation(self._sdfs)

    @property
    def get_sdf_grad(self):
        RS = build_scaling_rotation(self.get_scaling, self.get_rotation)
        return F.normalize(RS[:, :3, 2], p=2, dim=-1)

    @property
    def get_shift_xyz(self):
        sdf = self.get_sdf
        sdf_grad = self.get_sdf_grad
        xyz = self.get_xyz

        return xyz - sdf_grad * sdf

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self.use_sdf:
            return self.opacity_projection(self.get_sdf, self.inverse_deviation)
        else:
            return self.opacity_activation(self._opacity)

    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)

    @property
    def get_metallic(self):
        if self.use_metallic:
            return self.metallic_activation(self._metallic)
        else:
            return torch.zeros_like(self._metallic)

    def get_covariance(self, scaling_modifier=1):
        if self.ges_splat:
            return self.covariance_activation(self.get_scaling * self.get_shape, scaling_modifier, self._rotation)
        else:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_full_covariance(self, scaling_modifier):
        return self.full_cov_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling, 1 / scaling_modifier, self.get_rotation)

    def get_full_inverse_covariance(self, scaling_modifier=1):
        return self.full_cov_activation(1 / self.get_scaling, 1 / scaling_modifier, self.get_rotation)

    def get_normal(self):
        if self.use_sdf:
            return self.get_sdf_grad, None

        normal = self.get_minimum_axis
        normal = normal / normal.norm(dim=1, keepdim=True)  # (N, 3)
        return normal

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness + self.roughness_bias)

    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)

    @property
    def get_maximum_axis(self):
        return get_maximum_axis(self.get_scaling, self.get_rotation)

    @property
    def get_shape(self):
        return self.shape_activation(self._shape, self.shape_strngth)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5
        if self.sphere_init:
            print("Using sphere init...")
            num_points = np.asarray(pcd.points).shape[0]
            fused_point_cloud = torch.tensor(simple_fibonacci_sphere_sampling(num_points)).float().cuda() * 1.1
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if self.max_sh_degree > -1:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3 * (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3] = fused_color
            features[:, 3:] = 0.0
        else:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
            features[:, :3] = fused_color
            features[:, 3:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        if self.sphere_init:
            sdfs = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda") + 0.2
            n = F.normalize(fused_point_cloud, dim=-1)
            rots = normal2quat(n)
        else:
            sdfs = torch.rand((fused_point_cloud.shape[0], 1), device="cuda") * 2
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._sdfs = nn.Parameter(sdfs.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :3].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, 3:].contiguous().requires_grad_(True))

        albedo = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        metallic = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        self._roughness = nn.Parameter(
            self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True)
        )
        self._metallic = nn.Parameter(metallic.to(self._xyz.device).requires_grad_(True))
        self._albedo = nn.Parameter(albedo.to(self._xyz.device).requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args: OptimizationParams, scene):
        self.spatial_lr_scale = 5
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.inverse_deviation = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float, device="cuda"))

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.exclude_list = ["envmap", "inverse_deviation"]

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._sdfs], "lr": training_args.sdf_lr, "name": "sdf"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self.inverse_deviation], "lr": training_args.deviation_lr, "name": "inverse_deviation"},
        ]

        l.extend(
            [
                {"params": list(self.envmap.parameters()), "lr": training_args.envmap_lr_init, "name": "envmap"},
                {"params": [self._roughness], "lr": training_args.roughness_lr, "name": "roughness"},
                {"params": [self._albedo], "lr": training_args.albedo_lr, "name": "albedo"},
                {"params": [self._metallic], "lr": training_args.metallic_lr, "name": "metallic"},
            ]
        )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.envmap_scheduler_args = get_expon_lr_func(
            lr_init=training_args.envmap_lr_init,
            lr_final=training_args.envmap_lr_final,
            lr_delay_mult=training_args.envmap_lr_delay_mult,
            max_steps=training_args.envmap_lr_max_steps,
        )

    def _update_learning_rate(self, iteration, param):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == param:
                try:
                    lr = getattr(self, f"{param}_scheduler_args", self.envmap_scheduler_args)(iteration)
                    param_group["lr"] = lr
                    return lr
                except AttributeError:
                    pass

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        self._update_learning_rate(iteration, "xyz")
        for param in ["envmap", "roughness", "f_dc", "f_rest", "metallic", "albedo"]:
            lr = self._update_learning_rate(iteration, param)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        # All channels except the 3 DC

        for i in range(self._features_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        features_rest_len = self._features_rest.shape[1]
        for i in range(features_rest_len):
            l.append("f_rest_{}".format(i))
        l.append("sdf")
        l.append("inverse_deviation")
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("roughness")
        for i in range(self._albedo.shape[1]):
            l.append("albedo{}".format(i))
        l.append("metallic")

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        sdfs = self._sdfs.detach().cpu().numpy()
        N = self._rotation.shape[0]
        inverse_deviation = self.inverse_deviation.detach().view(1, 1).repeat(N, 1).cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()
        f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [
            xyz,
            f_dc,
            f_rest,
            sdfs,
            inverse_deviation,
            opacities,
            scale,
            rotation,
            roughness,
            albedo,
            metallic,
        ]

        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(f_dc)
        return pcd_o3d

    def get_invs_ref(self):
        def inv_learnable_sigmoid(o, s):
            tmp = 2 / o - 1 - 2 / o * torch.sqrt(1 - o)
            inv_dev = -torch.log(tmp) / s
            return inv_dev

        sdfs = torch.abs(self.get_sdf)
        med_sdf = torch.median(sdfs.squeeze())
        if med_sdf <= 0.1:
            return torch.tensor(4.5490, device=self.inverse_deviation.device)
        inv_s = inv_learnable_sigmoid(torch.ones_like(med_sdf) * 0.5, med_sdf)
        return inv_s

    def reset_opacity(self):
        if self.use_sdf:

            def inv_learnable_sigmoid(o, s):
                tmp = 2 / o - 1 - 2 / o * torch.sqrt(1 - o)
                inv_dev = -torch.log(tmp) / s
                return inv_dev

            sdfs_threshold = inv_learnable_sigmoid(
                torch.ones_like(self.get_sdf) * 0.01, torch.ones_like(self.get_sdf) * self.inverse_deviation
            )
            sdfs_new = torch.max(self.get_sdf, sdfs_threshold)
            optimizable_tensors = self.replace_tensor_to_optimizer(sdfs_new, "sdf")
            self._sdfs = optimizable_tensors["sdf"]
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        try:
            sdfs = np.asarray(plydata.elements[0]["sdf"])[..., np.newaxis]
            inverse_deviation = np.asarray(plydata.elements[0]["inverse_deviation"])
        except:
            sdfs = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            inverse_deviation = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        if self.max_sh_degree > -1:
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3 * (self.max_sh_degree + 1) ** 2 - 3))
        else:
            features_extra = np.zeros((xyz.shape[0], 3))
            if len(extra_f_names) == 3:
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                features_extra = features_extra.reshape((features_extra.shape[0], 3))
            else:
                print(f"NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
                features_extra = features_extra.reshape((features_extra.shape[0], 3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo")]
        albedo = np.zeros((xyz.shape[0], len(albedo_names)))
        for idx, attr_name in enumerate(albedo_names):
            albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sdfs = nn.Parameter(torch.tensor(sdfs, dtype=torch.float, device="cuda").requires_grad_(True))
        self.inverse_deviation = nn.Parameter(
            torch.tensor(inverse_deviation[0], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.exclude_list:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.exclude_list:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._sdfs = optimizable_tensors["sdf"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._albedo = optimizable_tensors["albedo"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.exclude_list:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_sdfs,
        new_opacities,
        new_scaling,
        new_rotation,
        new_roughness,
        new_albedo=None,
        new_metallic=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "sdf": new_sdfs,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "roughness": new_roughness,
            "albedo": new_albedo,
            "metallic": new_metallic,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sdfs = optimizable_tensors["sdf"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._roughness = optimizable_tensors["roughness"]
        self._albedo = optimizable_tensors["albedo"]
        self._metallic = optimizable_tensors["metallic"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
    ):
        N = 2
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        if torch.sum(selected_pts_mask) == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_sdf = self._sdfs[selected_pts_mask].repeat(N, 1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_sdf,
            new_opacity,
            new_scaling,
            new_rotation,
            new_roughness,
            new_albedo,
            new_metallic,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        if torch.sum(selected_pts_mask) == 0:
            return
        new_xyz = self._xyz[selected_pts_mask]
        new_sdfs = self._sdfs[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_sdfs,
            new_opacities,
            new_scaling,
            new_rotation,
            new_roughness,
            new_albedo,
            new_metallic,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
