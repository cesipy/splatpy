
import incremental_pipeline
import gsplat
import torch


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

        feature_dim = 32
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser
            #TODO: fix all of the settings here, maybe add to config
        )
        print(f"number of splats: {len(self.splats['means'])}")


    def train_step(self, ):
        ...

    def train(self, steps):
        for i in range(steps):
            self.train_step

    def eval(self):
        ...




def main():
    video_path = "res/input/test_video.MOV"
    ip = incremental_pipeline.COLMAP_Processor()

    ip.create_colmap(video_path, frames_modulo=10, mode="sequential")
    ip.clean_up()
    config = TrainingConfig(data_dir=DATA_DIR, data_factor=1, results_dir=RESULTS_DIR)
    trainer = Trainer(config=config)









if __name__ == "__main__":
    main()
