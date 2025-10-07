from __future__ import annotations
from typing import Literal, Union, TYPE_CHECKING
import wandb, math, os
import torch
import torch.optim as opt
import torch.utils.data as tud
import torch.distributed as dist
from torch import nn

# from einops import rearrange,reduce,repeat
# from einops.layers.torch import Rearrange,Reduce
from dataclasses import dataclass
from transformers.trainer import get_scheduler

from train.utils.tools import rank0only_decorator, rank0print, mkdir, read_path, remove_path
from train.model.gpt import GPT
from train.dataset.base import CollateDataset
if TYPE_CHECKING:
    from train.strategy.deepspeed.deepspeed import DeepspeedStrategy
from train.utils.wandb import WandbConfig
from train.utils.tensorboard import TensorboardConfig
from train.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainerConfig:
    batch_size: int = 1
    bucket_size: int  = None
    micro_batch_size: int = 1
    max_epochs: int = 1
    
    scheduler: Literal["linear",
                       "cosine",
                       "cosine_with_restarts",
                       "polynomial",
                       "constant",
                       "constant_with_warmup",
                       "inverse_sqrt",
                       "reduce_lr_on_plateau",
                       "cosine_with_min_lr",
                       "warmup_stable_decay"] = "cosine_with_min_lr"
    learing_rate: float = 5e-6
    lr_warmup_ratio: float = 0.03

    eval_steps: int = 1
    save_steps: int = float("inf")
    logging_steps: int = 1 # wandb

    ckpt_dir: str = None
    max_ckpts: int = 1
    max_ckpt_GB: int = 1000
    weights_dir: str = None # intermediate model weight path
    max_weights:  int = 1 
    weights_saving_interval: int = 1 # the global_steps between two saving weight process
    save_dir: str = None # final model weight path

    enable_micro_bar: bool = False # enable a bar showing the progress of micro batch
    

    def update_steps_per_epoch(self, train_dataloader: tud.DataLoader):
        if self.bucket_size is None:
            return len(train_dataloader.dataset) // self.batch_size
        assert isinstance(train_dataloader.batch_sampler, DistributedBucketSampler)
        return train_dataloader.batch_sampler.total_size // self.batch_size

    def max_steps(self, train_dataloader: tud.DataLoader):
        return math.ceil(self.max_epochs * self.update_steps_per_epoch(train_dataloader))
    
    def lr_warmup_steps(self, train_dataloader: tud.DataLoader):
        return math.ceil(self.lr_warmup_ratio * self.max_steps(train_dataloader))

class Trainer: 
    def __init__(
            self,
            model: Union[GPT],
            train_dataset: CollateDataset,
            eval_dataset: CollateDataset,
            optimizer: opt.Optimizer,
            strategy: DeepspeedStrategy,
            config: TrainerConfig,
            wandb_config: WandbConfig = None,
            tensorboard_config: TensorboardConfig = None,
    ):
        self.model = model
        self.tokenizer = train_dataset.tokenizer
        self.optimizer = optimizer
        self.strategy = strategy
        
        self.config = config

        self.train_dataloader = strategy.setup_dataloader(
            dataset = train_dataset,
            batch_size = config.micro_batch_size,
            bucket_size = config.bucket_size,
            pin_memory = True,
            shuffle = True,
            collate_fn = train_dataset.collate_fn,
        )
        self.eval_dataloader = strategy.setup_dataloader(
            dataset = eval_dataset,
            batch_size = config.micro_batch_size,
            bucket_size = config.bucket_size,
            pin_memory = True,
            shuffle = False,
            collate_fn = eval_dataset.collate_fn
        )


        self.scheduler = get_scheduler(
            name = config.scheduler,
            optimizer = optimizer,
            num_warmup_steps = config.lr_warmup_steps(self.train_dataloader),
            num_training_steps = config.max_steps(self.train_dataloader),
            scheduler_specific_kwargs = dict(min_lr = config.learing_rate * 0.1)
        )

        (self.model, 
         self.optimizer, 
         self.scheduler) = strategy.prepare_train(self.model,
                                                  self.optimizer, 
                                                  self.scheduler)
        

        assert (wandb_config is not None) | (tensorboard_config is not None),\
        "You should only use at least one visualization." 

        self._init_wandb(wandb_config = wandb_config)
        self._init_tensorboard(tensorboard_config = tensorboard_config)

    def to_current_device(self, *args):
        if len(args) == 1:
            if isinstance(args[0], list): return [self.to_current_device(k) for k in args[0]]
            if isinstance(args[0], tuple): return tuple(self.to_current_device(k) for k in args[0])
            if isinstance(args[0], torch.Tensor): return args[0].to(torch.cuda.current_device())
            return args[0]
        return tuple(self.to_current_device(k) for k in args)
    
    @rank0only_decorator
    def _init_tensorboard(self, tensorboard_config: TensorboardConfig):
        self.tensorboard_config = tensorboard_config
        if tensorboard_config:
            mkdir(tensorboard_config.log_dir)
            log_dir = os.path.join(tensorboard_config.log_dir, tensorboard_config.name)
            self._tensorboard = SummaryWriter(log_dir = log_dir)
    
    @rank0only_decorator
    def _update_tensorboard(self, logs_dict:dict, global_step: int, logging_steps:int = 1, step: int = 0):
        if self.tensorboard_config and global_step % logging_steps == 0:
            for k, v in logs_dict.items():
                self._tensorboard.add_scalar(k, v, global_step)
            

    @rank0only_decorator
    def _finish_tensorboard(self):
        if self.tensorboard_config:
            self._tensorboard.close()

    @rank0only_decorator
    def _init_wandb(self, wandb_config: WandbConfig):
        self.wandb_config = wandb_config
        if wandb_config:
            wandb.login(key = wandb_config.api_key)
            wandb.init(
                entity = wandb_config.entity,
                project = wandb_config.project,
                group = wandb_config.group,
                name = wandb_config.name,
                config = wandb_config.config,
                reinit = wandb_config.reinit
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", 
                                step_metric = "train/global_step",
                                step_sync = True)
            
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", 
                                step_metric = "eval/global_step",
                                step_sync = True)
            

    
    @rank0only_decorator
    def _update_wandb(self, logs_dict:dict, global_step:int, logging_steps:int = 1, step:int = 0):
        if self.wandb_config and global_step % logging_steps == 0:
            wandb.log(logs_dict, step = step)



    @rank0only_decorator
    def _finish_wandb(self):
        if self.wandb_config:
            wandb.finish()


    @rank0only_decorator
    def update_visualization(self, logs_dict:dict, global_step:int, logging_steps:int = 1, step:int = 0):
        self._update_tensorboard(logs_dict, global_step, logging_steps, step)
        self._update_wandb(logs_dict, global_step, logging_steps, step)
        

    @rank0only_decorator
    def finish_visualization(self):
        self._finish_tensorboard()
        self._finish_wandb()


    def load_ckpt(self, load_ckpt: bool = True) -> dict:
        client_states = dict(consumed_samples = 0,
                             total_tokens = 0,
                             loss_tokens = 0
                            #  train_batch_size = 1,
                            #  accumulated_gradient = 1,
                            #  update_steps_per_epoch = 1
                             )
        if load_ckpt and self.config.ckpt_dir:
            if not os.path.exists(self.config.ckpt_dir) or \
                (not os.path.exists(f"{self.config.ckpt_dir}/latest")):
                rank0print(f"Make sure that this is the first training process,"
                           f" because ckpt path:`{self.config.ckpt_dir}` doesn't exist .")
                return client_states
                
            _, client_states = self.strategy.load_ckpt(model = self.model.model,
                                                       load_dir = self.config.ckpt_dir)

            print( f"Rank: {dist.get_rank()} has loaded the checkpoint:{self.config.ckpt_dir}")
        return client_states
    
    def save_ckpt(self, global_step: int = 1, client_state:dict = dict()):
        if self.config.ckpt_dir is None: return
        if global_step % self.config.save_steps:return
        tag = f"global_step{global_step}"
        self.strategy.save_ckpt(
            self.model.model, 
            save_dir = self.config.ckpt_dir, 
            tag = tag,
            max_ckpts = self.config.max_ckpts,
            max_ckpt_GB = self.config.max_ckpt_GB,
            client_state = client_state
        )

    def save_model(self, global_step:int = 1, finished: bool = False):

        if self.config.weights_dir is None: return
        if global_step % self.config.weights_saving_interval and (not finished): return
        
        weight_dir = self.config.weights_dir
        max_weights = self.config.max_weights

        if dist.get_rank() == 0 and (not finished):
            os.makedirs(weight_dir, exist_ok = True)
            subdirs = sorted([k for k in read_path(weight_dir) if os.path.isdir(k)],
                             key = lambda x: os.path.getmtime(x))
            
            while True:
                if len(subdirs) < max_weights:
                    break
                remove_path(subdirs.pop(0))

        self.strategy.save_model(
            self.model,
            self.tokenizer,
            os.path.join(self.config.weights_dir, f"global_step{global_step}")
        ) 