import os
from . import incremental_pipeline
from .config import TrainingConfig
from .trainer import Trainer, DATA_DIR, RESULTS_DIR
from .utils.colmap_datahandling import Parser, Dataset, test


def video_to_gsplat(path: str):
    assert os.path.exists(path), f"video path {path} does not exist"
    try:
        ip = incremental_pipeline.COLMAP_Processor()
        ip.clean_up()
        ip.create_colmap(path, frames_modulo=20, mode="sequential")

        # step 2: train
        config = TrainingConfig(data_dir=DATA_DIR, data_factor=1, results_dir=RESULTS_DIR,)
        trainer = Trainer(config=config)
        trainer.train(100000)
        trainer.render_orbit(num_frames=240)
    except Exception as e:
        raise e
    finally:
        ip.clean_up()

