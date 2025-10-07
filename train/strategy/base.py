import os, torch
# from torch import nn
from typing import Callable
import torch.utils.data as tud
# import deepspeed

from transformers import PreTrainedModel
from train.dataset.base import CollateDataset
from train.model.gpt import GPT
from train.utils.lora import LoRAConfig, get_peft_model

# TBD
class Strategy:
    def __init__(self, lora_config: LoRAConfig = None):
        self.lora_config = lora_config

    def get_local_rank(self):
        return int(os.environ["LOCAL_RANK"])
    
    def get_local_world_size(self):
        return int(os.environ["LOCAL_WORLD_SIZE"])

    def setup_dataloader(self, 
                         dataset: CollateDataset,
                         batch_size: int,
                         pin_memory: bool = False,
                         shuffle: bool = True,
                         collate_fn: Callable = None):
        return tud.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            pin_memory = pin_memory,
            shuffle = shuffle,
            collate_fn = dataset.collate_fn
        )

    def set_submodule(self, 
                      module: torch.nn.Module, 
                      submodule_name: str,
                      submodule: torch.nn.Module):
        submodules = module.__dict__.get("_modules")
        submodules[submodule_name] = submodule
        module.__dict__[submodule_name] = submodule

    def prepare_if_lora(self, model: GPT):
        if self.lora_config is not None:
            model.model.enable_input_require_grads()
            peft_model = get_peft_model(model.model, self.lora_config)
            self.set_submodule(model, "model", peft_model)
            
        return model

    def prepare_train(self, model, optimizer, scheduler):
        return model, optimizer, scheduler
    
    def prepare_eval(self, model):
        return model


    def backward(self, loss: torch.Tensor, model: GPT):
        model.model.backward(loss)
    
    def optimizer_step(self, model: GPT):
        model.model.step()


    def zero_grad(self, model: GPT):
        engine = model.model
        
        if engine.bfloat16_enabled():
            if engine.zero_optimization() and hasattr(engine.optimizer, "zero_grad"):
                engine.optimizer.zero_grad()
            else:
                pass
        elif engine.zero_optimization() or engine.fp16_enabled() or engine.amp_enabled():
            engine.optimizer.zero_grad()
        else:
            engine.zero_grad()
