import os
from typing import Tuple, Dict, Optional, Literal
from typing_extensions import Literal, assert_never
import numpy as np
import gsplat
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio
import numpy as np

import lpips # perceptual loss

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
# from fused_ssim import fused_ssim
from pytorch_msssim import ssim, 

from . import incremental_pipeline
from .config import TrainingConfig
from .utils.colmap_datahandling import Parser, Dataset
from .utils.utils import create_splats_with_optimizers



VERBOSE     = False

class Trainer():
    """
    trainer for gaussian splats.

    inspired by https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
    """
    def __init__(
        self,
        config: TrainingConfig,
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"parser data dir: {config.colmap_data_dir}")

        self.parser = Parser(
            # data_dir=os.path.join(config.colmap_data_dir,"images"),
            data_dir=config.colmap_data_dir,
            factor=config.data_factor,
            normalize=True,
            test_every=8,
        )
        self.dataset = Dataset(
            self.parser,
            split="train",
        )
        print(f"length of the dataset = {len(self.dataset)}")

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=4,
        )

        # scene scale needed for strategy initialization
        self.scene_scale = self.parser.scene_scale * 1.1
        print(f"scene scale: {self.scene_scale}")

        feature_dim = None
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser,
            init_type="sfm",
            init_opacity=0.1,
            init_scale=1.0,
            scene_scale=self.scene_scale,
            sh_degree=config.sh_degree,
            feature_dim=feature_dim,
            device=self.device,
        )
        print(f"number of splats: {len(self.splats['means'])}")

        self.loss_fn_lpips = lpips.LPIPS(net="alex").to(self.device)

        # initialize strategy
        self.strategy = config.strategy
        self.strategy_state = self.strategy.initialize_state(
            scene_scale=self.scene_scale
        )

    def __data_step(self):
        if not hasattr(self, "data_iterator"):
            self.data_iterator = iter(self.data_loader)
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)
        return batch

    def forward_pass(
        self,
        camtoworlds:torch.Tensor,
        Ks:torch.Tensor,
        width:int, height:int,
        masks: Optional[torch.Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # prepare the splat's current params
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        # TODO: implement appearance optimization
        # if self.cfg.app_opt:
        #     colors = self.app_module(
        #         features=self.splats["features"],
        #         embed_ids=image_ids,
        #         dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        #         sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
        #     )
        #     colors = colors + self.splats["colors"]
        #     colors = torch.sigmoid(colors)
        # else:
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "classic"
        if camera_model is None:
            camera_model = "pinhole"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=self.strategy.absgrad,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


    def train_step(self, step:int):
        batch = self.__data_step()
        Ks          = batch["K"].to(self.device)        # [bs,3,3]
        camtoworlds = batch["camtoworld"].to(self.device) # [bs,4,4]
        image       = batch["image"].to(self.device) / 255.0  # [bs,h,w,3], normalize to [0,1]
        image_ids   = batch["image_id"]

        # sh schedule
        sh_degree_to_use = min(step//self.config.sh_degree_interval, self.config.sh_degree)

        # forward pass
        renders, alpha, info = self.forward_pass(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=image.shape[2],
            height=image.shape[1],
            image_ids=image_ids,
            sh_degree=sh_degree_to_use,
        )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        # pre-backward hook
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # compute loss
        l1loss = F.l1_loss(colors, image)
        # fused_ssim needs to build from source, so it does work with the philosophy of this project
        # ssim_loss = 1.0 - fused_ssim(
        #     colors.permute(0,3,1,2), image.permute(0,3,1,2), padding="valid",
        # )
        ssim_loss = 1.0 - ssim(colors.permute(0,3,1,2), image.permute(0,3,1,2), data_range=1.0, size_average=True)
        lpips_loss = self.loss_fn_lpips(
            colors.permute(0,3,1,2) * 2 - 1,  # normalize to [-1,1]
            image.permute(0,3,1,2) * 2 - 1
        ).mean()


        # loss from og paper is l1 + ssim loss (perceptual + structure)
        # loss = (1.0-self.config.ssim_lambda)* l1loss + self.config.ssim_lambda * ssim_loss
        loss = 0.8  * l1loss + 0.1 * ssim_loss + 0.1 * lpips_loss

        loss.backward()

        # optimize
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()

        # post-backward: densification (split/clone/prune)
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
            packed=False,
        )

        return loss.item()


    def train(self, steps):

        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / steps)
            ),
        ]
        print(f"\ntraining for {steps} steps...")
        pbar = tqdm(range(steps))
        losses_buffer = []
        for step in pbar:
            loss = self.train_step(step=step)
            losses_buffer.append(loss)
            # update progress
            if step % 100 == 0:
                num_gs = len(self.splats['means'])
                pbar.set_description(f"loss: {np.array(losses_buffer).mean():.4f} | gaussians: {num_gs}")
                losses_buffer = []

            if step % 2000 == 0:
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]
                results_path = os.path.join(self.config.results_dir, f"step-{step}.ply")
                os.makedirs(self.config.results_dir, exist_ok=True)
                gsplat.export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=results_path,
                )



        print(f"\ntraining complete! final num gaussians: {len(self.splats['means'])}")

        means = self.splats["means"]
        scales = self.splats["scales"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]

        results_path = os.path.join(self.config.results_dir, f"final.ply")
        os.makedirs(self.config.results_dir, exist_ok=True)
        gsplat.export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=results_path,
        )

    def render_orbit(self, num_frames: int = 120):
        """Render 360° orbit around the scene."""

        # compute scene center from camera positions
        camera_positions = self.parser.camtoworlds[:, :3, 3]  # [N, 3]
        scene_center = camera_positions.mean(axis=0)

        # compute radius
        radius = np.linalg.norm(camera_positions - scene_center, axis=1).mean()

        # use first camera's intrinsics
        K = torch.from_numpy(
            list(self.parser.Ks_dict.values())[0]
        ).float().unsqueeze(0).to(self.device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # create orbit path
        frames = []
        with torch.no_grad():
            for i in range(num_frames):
                angle = 2 * np.pi * i / num_frames
                r_factor = 1.8
                # camera position on circle
                cam_pos = scene_center + np.array([
                    r_factor *radius * np.cos(angle),
                    r_factor *radius * np.sin(angle),
                    scene_center[2] /6 # keep same height
                ])

                # look at center
                # copy_cam_pos = cam_pos.copy()
                # copy_cam_pos[2] += 0.8*cam_pos[2] # slight downward angle
                # forward = scene_center - (copy_cam_pos)
                # forward = forward / np.linalg.norm(forward)
                target = scene_center.copy()
                target += np.array([0,0,0.6])
                forward = target - cam_pos
                forward = forward / np.linalg.norm(forward)

                # up vector
                up = np.array([0, 0, 1])
                right = np.cross(up, forward)
                right = right / np.linalg.norm(right)
                up = np.cross(forward, right)

                # build camera-to-world matrix
                camtoworld = np.eye(4)
                camtoworld[:3, 0] = right
                camtoworld[:3, 1] = up
                camtoworld[:3, 2] = forward
                camtoworld[:3, 3] = cam_pos

                camtoworld = torch.from_numpy(camtoworld).float().unsqueeze(0).to(self.device)

                # render
                renders, _, _ = self.forward_pass(
                    camtoworlds=camtoworld,
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=self.config.sh_degree,
                )

                frame = renders[0].cpu().numpy()
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                frames.append(frame)

        # save video
        output_path = os.path.join(self.config.results_dir, "orbit.mp4")
        imageio.mimsave(output_path, frames, fps=30)


    def eval(self):
        # TODO: implement evaluation
        ...
