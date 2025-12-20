
from typing import Tuple, Dict, Optional, Literal
import gsplat
import torch
import torch.nn.functional as F

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from fused_ssim import fused_ssim

import incremental_pipeline
from config import TrainingConfig
from utils.colmap_datahandling import Parser, Dataset
from utils.utils import create_splats_with_optimizers


DATA_DIR    = "res/output/"
RESULTS_DIR = "res/results/"

class Trainer():
    """
    trainer for gaussian splats.

    inspired by https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
    """
    def __init__(
        self,
        config: TrainingConfig,

    ):
        self.strategy = gsplat.strategy.DefaultStrategy(verbose=True)
        self.parser = Parser(
            data_dir=config.data_dir,
            factor=config.data_factor,
            #TODO: those need to be done
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

        feature_dim = None
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser
            #TODO: fix all of the settings here, maybe add to config
        )
        print(f"number of splats: {len(self.splats['means'])}")
        self.config = config

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
        #prepare the splat's current params
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
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
            rasterize_mode = "classic"#"antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = "pinhole"#self.cfg.camera_model
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
            packed=False,#self.cfg.packed,
            absgrad=(
                self.config.strategy.absgrad
                if isinstance(self.config.strategy, DefaultStrategy)
                else False
            ),
            # sparse_grad=self.cfg.sparse_grad,
            # rasterize_mode=rasterize_mode,
            # distributed=self.world_size > 1,
            # camera_model=self.cfg.camera_model,
            # with_ut=self.cfg.with_ut,
            # with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


    def train_step(self, step:int):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = self.__data_step()
        Ks          = batch["K"].to(device) # [bs,3,3]
        camtoworlds = batch["camtoworld"]   # [bs,4,4]
        camtoworlds = camtoworlds.to(device)# [bs,4,4]
        iamge       = batch["image"].to(device)\
                            / 255.0         # [bs,h,w,3], normalize to [0,1]
        # print(f"shape Ks: {Ks.shape}, camtoworlds: {camtoworlds.shape}, image: {iamge.shape}")
        image_ids = batch["image_id"]


        #sh schedule
        sh_degree_to_use = min(step//self.config.sh_degree_interval,self.config.sh_degree)
        #forward pass
        renders, alpha, info = self.forward_pass(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=iamge.shape[2],
            height=iamge.shape[1],
            image_ids=image_ids,
            sh_degree=sh_degree_to_use,
        )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        l1loss = F.l1_loss(colors, iamge)
        ssim_loss = 1.0 - fused_ssim(
            colors.permute(0,3,1,2), iamge.permute(0,3,1,2), padding="valid",
        )
        # loss from og paper is l1 + ssim loss (perceptual + structure)
        loss = (1.0-self.config.ssim_lambda)* l1loss + self.config.ssim_lambda * ssim_loss

        loss.backward()
        # optimize
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()



    def train(self, steps):
        for i in range(steps):
            self.train_step(step=i)


    def eval(self):
        ...




def main():
    video_path = "res/input/test_video.MOV"
    ip = incremental_pipeline.COLMAP_Processor()

    ip.create_colmap(video_path, frames_modulo=10, mode="sequential")
    ip.clean_up()
    config = TrainingConfig(data_dir=DATA_DIR, data_factor=1, results_dir=RESULTS_DIR)
    trainer = Trainer(config=config)
    trainer.train(10)









if __name__ == "__main__":
    main()
