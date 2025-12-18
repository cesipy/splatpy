from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_dir: str = "res/output/"       # where the colmap is stored
    data_factor:int = 1                 # downscaling factor for images
    results_dir: str = "res/results/"   # where to store results
