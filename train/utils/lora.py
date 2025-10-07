from train.utils.tools import read_json, save_json, path_join
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig, 
    TaskType, 
    PeftModel, 
    get_peft_model
)
from safetensors import safe_open
import torch

from dataclasses import dataclass

@dataclass
class LoRAConfig(LoraConfig):
    task_type: str = TaskType.CAUSAL_LM
    save_merged: bool = True
    target_modules: list | str = "all-linear"
    def __post_init__(self):
        super().__post_init__()
        self.task_type = TaskType.CAUSAL_LM
        self.bias = "none"
