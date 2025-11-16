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
import uuid

from argparse import ArgumentParser, Namespace
from random import randint
from time import time

import torch
import torchvision

from fused_ssim import fused_ssim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import RENDER_DICT, render_lighting
from render import render_set
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap, psnr
from utils.loss_utils import (
    base_smooth_loss,
    bilateral_smooth_loss,
    l1_loss,
    predicted_normal_loss,
)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, load_iteration):
    render_fn = RENDER_DICT[pipe.gaussian_type]
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.env_mode,
        dataset.env_res,
        dataset.use_sdf,
        opt.metallic,
        opt.sphere_init,
    )
    if load_iteration > -1:
        scene = Scene(dataset, gaussians, load_iteration=load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, scene)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    load_iteration = load_iteration if load_iteration > 0 else 0
    progress_bar = trange(load_iteration, opt.iterations, desc="Training progress")

    for iteration in range(load_iteration + 1, opt.iterations + 1):
        if dataset.random_background == True:
            background = torch.rand((3)).cuda()

        iter_start = time()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        mask = viewpoint_cam.gt_alpha_mask.cuda()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        gt_image = viewpoint_cam.original_image.cuda()

        if gaussians.env_mode == "envmap":
            gaussians.envmap.build_mips()

        # Render
        render_pkg = render_fn(viewpoint_cam, scene, pipe, background, debug=False)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = gt_image * mask + background[:, None, None] * (1 - mask)
        loss = torch.tensor(0.0).cuda()

        Ll1 = l1_loss(image, gt_image)
        loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )
        losses_extra = {}

        if (
            iteration > opt.normal_reg_from_iter
            and iteration < opt.normal_reg_util_iter
            and "normal_ref" in render_pkg.keys()
        ):
            losses_extra["predicted_normal"] = predicted_normal_loss(
                render_pkg["normal"], render_pkg["normal_ref"], mask
            )
        o = render_pkg["alpha"].clamp(1e-6, 1 - 1e-6)
        losses_extra["zero_one"] = -(mask * torch.log(o) + (1 - mask) * torch.log(1 - o)).mean()

        if opt.lambda_brdf_smoothness > 0:
            l_roughness = bilateral_smooth_loss(render_pkg["roughness"], gt_image, mask)
            l_albedo = bilateral_smooth_loss(render_pkg["albedo"], gt_image, mask)
            l_metallic = bilateral_smooth_loss(render_pkg["metallic"], gt_image, mask)
            losses_extra["brdf_smoothness"] = l_roughness + l_albedo + l_metallic

        if opt.lambda_base_smoothness > 0:
            l_roughness = base_smooth_loss(render_pkg["roughness"], gt_image, mask)
            l_albedo = base_smooth_loss(render_pkg["albedo"], gt_image, mask)
            l_metallic = base_smooth_loss(render_pkg["metallic"], gt_image, mask)
            losses_extra["base_smoothness"] = l_roughness + l_albedo + l_metallic

        if opt.lambda_light_reg > 0.0:
            losses_extra["light_reg"] = gaussians.envmap.regularizer()

        if opt.lambda_distortion > 0.0 and iteration > opt.dist_from_iteration:
            losses_extra["distortion"] = render_pkg["distortion"].mean()

        proj_error = None
        if opt.lambda_proj > 0.0 and iteration > opt.proj_from_iteration:
            points = gaussians.get_shift_xyz[visibility_filter]
            points = torch.cat([points, torch.ones_like(points[:, -1:])], -1)
            points_view = points @ viewpoint_cam.world_view_transform
            points_proj = points @ viewpoint_cam.full_proj_transform
            points_depth = points_view[:, 2:3]
            uv = points_proj[:, :2] / (points_proj[:, -1:] + 1e-8)
            gaussian_proj_depth = torch.nn.functional.grid_sample(
                input=render_pkg["depth"].unsqueeze(0),
                grid=uv.view(1, -1, 1, 2),
                mode="bilinear",
                padding_mode="border",  # 'reflection', 'zeros'
            )[0, 0]
            # Detach the projected depth to avoid unstable gradient.
            proj_error = torch.abs(gaussian_proj_depth.detach() - points_depth)
            loss_proj = (proj_error * (proj_error < opt.proj_thres)).mean()
            losses_extra["proj"] = loss_proj

        if gaussians.use_sdf and iteration > 1000:
            ref_dev = gaussians.get_invs_ref()
            loss_dev = torch.relu(ref_dev - gaussians.inverse_deviation)
            losses_extra["dev"] = loss_dev

        for k in losses_extra.keys():
            loss += getattr(opt, f"lambda_{k}") * losses_extra[k]
        loss.backward()
        iter_end = time()

        scene.gaussians._roughness.grad[scene.gaussians._roughness.grad.isnan()] = 0.0

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}, {gaussians.get_xyz.shape[0]}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            # Log and save
            losses_extra["psnr"] = psnr(image, gt_image).mean()
            test_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                losses_extra,
                l1_loss,
                iter_end - iter_start,
                testing_iterations,
                scene,
                render_fn,
                (pipe, test_bg),
                {"is_training": False},
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (
                pipe.env_mode == "envmap"
                and pipe.render_mode.find("split_sum") < 0
                and pipe.render_mode.find("disney") < 0
            ):
                gaussians.envmap.clamp_(min=0.0, max=1.0)


def prepare_output_and_logger(args, opt, pipe):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))
    with open(os.path.join(args.model_path, "pipe_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    losses_extra,
    l1_loss,
    elapsed,
    testing_iterations,
    scene,
    renderFunc,
    renderArgs,
    renderArgDict={},
):
    number_gaussian = int((scene.gaussians.get_xyz).shape[0])
    if tb_writer:
        tb_writer.add_scalar("train/number_gaussian", number_gaussian, iteration)
        tb_writer.add_scalar("train/inv_deviation", scene.gaussians.inverse_deviation.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f"train_loss_patches/{k}_loss", losses_extra[k].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )
        for config in validation_configs:
            if not config["cameras"] or len(config["cameras"]) == 0:
                continue

            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            vis_idx = [i for i in range(0, len(config["cameras"]), len(config["cameras"]) // 4)]
            for idx, viewpoint in enumerate(config["cameras"]):
                if idx not in vis_idx:
                    continue
                mask = viewpoint.gt_alpha_mask.cuda()
                gt_image = viewpoint.original_image.cuda()
                H, W = gt_image.shape[1:]
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                render_pkg = renderFunc(viewpoint, scene, *renderArgs, **renderArgDict)

                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)
                gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                if tb_writer and (idx in vis_idx):
                    tb_writer.add_images(
                        config["name"] + "_view_{}/render".format(viewpoint.image_name),
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                            gt_image[None],
                            global_step=iteration,
                        )
                    for k in render_pkg.keys():
                        if render_pkg[k].dim() < 3 or k == "delta_normal_norm":
                            continue
                        if "depth" in k:
                            image_k = apply_depth_colormap(-render_pkg[k][0][..., None])
                            image_k = image_k.permute(2, 0, 1)
                        elif k == "alpha":
                            image_k = apply_depth_colormap(render_pkg[k][0][..., None], min=0.0, max=1.0)
                            image_k = image_k.permute(2, 0, 1)
                        elif k == "render":
                            image_k = render_pkg["render"]
                        else:
                            if "normal" in k:
                                render_pkg[k] = 0.5 + (0.5 * render_pkg[k])  # (-1, 1) -> (0, 1)
                            image_k = torch.clamp(render_pkg[k], 0.0, 1.0)
                        tb_writer.add_images(
                            config["name"] + "_view_{}/{}".format(viewpoint.image_name, k),
                            image_k[None],
                            global_step=iteration,
                        )
                        save_dir = f'{scene.model_path}/eval/iteration_{iteration:05d}/{config["name"]}'
                        os.makedirs(save_dir, exist_ok=True)
                        torchvision.utils.save_image(image_k, f"{save_dir}/{idx:03d}_{k}.png")
                    lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                    if tb_writer:
                        tb_writer.add_images(config["name"] + "/lighting", lighting[None], global_step=iteration)
            l1_test = l1_loss(images, gts)
            psnr_test = psnr(images, gts).mean()
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
        render_set(
            scene.model_path,
            "test",
            iteration,
            scene.getTestCameras(),
            scene,
            renderArgs[0],
            background=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        )
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6019)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    start_time = time()
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.iteration,
    )
    end_time = time()
    print(end_time - start_time)
    # All done
    print("\nTraining complete.")
