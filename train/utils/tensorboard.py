import wandb
from dataclasses import dataclass

@dataclass
class TensorboardConfig:
    log_dir: str
    name : str