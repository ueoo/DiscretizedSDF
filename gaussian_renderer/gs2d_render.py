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

import math

import torch
import torch.nn.functional as F

from diff_surfel_sdf_rasterization import GaussianRasterizationSettings as GSDFSettings
from diff_surfel_sdf_rasterization import GaussianRasterizer as GSDFRasterizer

from scene.NVDIFFREC import extract_env_map
from utils.graphics_utils import depths_to_points, get_rays


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


# render 360 lighting for a single gaussian
def render_lighting(pc, resolution=(512, 1024), sampled_index=None):
    if pc.env_mode == "envmap":
        lighting = extract_env_map(pc.envmap, resolution)  # (H, W, 3)
        lighting = lighting.permute(2, 0, 1)  # (3, H, W)
    else:
        raise NotImplementedError

    return lighting


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None, ...] > 0.0).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def render(
    viewpoint_camera, scene, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, debug=False, rescale=1.0, **kwargs
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if pipe.render_mode.find("defer") > -1:
        out_pbr = defer_render(viewpoint_camera, scene, pipe, bg_color, scaling_modifier, debug, rescale)
    else:
        raise NotImplementedError
    return out_pbr


def defer_render(
    viewpoint_camera, scene, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, debug=False, rescale=1.0
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    pc = scene.gaussians
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GSDFSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True,
    )

    rasterizer = GSDFRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[:, :2]
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    roughness = pc.get_roughness  # (N, 1)
    metallic = pc.get_metallic
    albedo = pc.get_albedo
    render_extras = [albedo, roughness, metallic]
    render_extras = torch.cat(render_extras, -1)

    out_extras = {}
    out_extras["pos"], radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=pc.get_xyz,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        pbr_params=render_extras,
        cov3D_precomp=cov3D_precomp,
    )

    out_extras["alpha"] = alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    normal = allmap[2:5]
    normal = (normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)
    out_extras["normal"] = normal * alpha.detach()

    # get expected depth map
    depth_expected = allmap[0:1]
    depth_expected = depth_expected / (alpha + 1e-8)
    depth_expected = torch.nan_to_num(depth_expected, 0, 0)
    out_extras["depth"] = depth_expected

    out_extras["distortion"] = allmap[6:7]
    out_extras["albedo"] = allmap[7:10] * rescale
    out_extras["roughness"] = allmap[10:11]
    out_extras["metallic"] = allmap[11:12]

    if pipe.render_mode.find("split_sum") > -1:
        defer_normal = F.normalize(out_extras["normal"].permute(1, 2, 0), dim=-1).reshape(1, 1, -1, 3)
        rendered_image, brdf_pkg = pc.envmap.shade_ss(
            out_extras["pos"].permute(1, 2, 0).reshape(1, 1, -1, 3),
            defer_normal,
            out_extras["albedo"].permute(1, 2, 0).reshape(1, 1, -1, 3),
            out_extras["metallic"].permute(1, 2, 0).reshape(1, 1, -1, 1),
            out_extras["roughness"].permute(1, 2, 0).reshape(1, 1, -1, 1),
            viewpoint_camera.camera_center[None, None, :]
            .repeat(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 1)
            .reshape(1, 1, -1, 3),
        )
    elif pipe.render_mode.find("ss_v2") > -1:
        view_dir = viewpoint_camera.get_rays().reshape(1, 1, -1, 3)
        defer_normal = F.normalize(out_extras["normal"].permute(1, 2, 0), dim=-1).reshape(1, 1, -1, 3)
        rendered_image, brdf_pkg = pc.envmap.shade_ss_v2(
            view_dir,
            defer_normal,
            out_extras["albedo"].permute(1, 2, 0).reshape(1, 1, -1, 3),
            out_extras["metallic"].permute(1, 2, 0).reshape(1, 1, -1, 1),
            out_extras["roughness"].permute(1, 2, 0).reshape(1, 1, -1, 1),
        )
    else:
        raise NotImplementedError

    rendered_image = rendered_image.view(
        int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 3
    ).permute(2, 0, 1)
    rendered_image = rendered_image * out_extras["alpha"] + (1 - out_extras["alpha"]) * bg_color[:, None, None]

    # Render normal from depth image, and alpha blend with the background.
    out_extras["normal_ref"] = depth_to_normal(viewpoint_camera, out_extras["depth"]).permute(2, 0, 1) * alpha.detach()
    normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
    normalize_normal_inplace(out_extras["normal_ref"], out_extras["alpha"][0])

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    out.update(out_extras)
    return out
