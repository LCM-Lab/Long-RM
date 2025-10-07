import wandb
from dataclasses import dataclass

@dataclass
class WandbConfig:
    api_key: str
    entity : str
    project: str
    group  : str
    name   : str
    config : dict
    reinit : bool = True