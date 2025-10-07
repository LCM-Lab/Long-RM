from __future__ import annotations
import os
from copy import deepcopy
from typing import Union, Literal, Callable, TYPE_CHECKING
from datetime import timedelta
from dataclasses import dataclass
import torch, transformers, deepspeed
import torch.distributed as dist
import torch.utils.data as tud
from ring_flash_attn import substitute_hf_flash_attn
from transformers import PreTrainedTokenizer
# from deepspeed import PipelineEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from train.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)
if TYPE_CHECKING:
    from train.trainer.base import TrainerConfig
    from train.utils.lora import LoRAConfig
from peft import get_peft_model_state_dict, get_peft_model, PeftModel
from train.strategy.base import Strategy
from train.strategy.deepspeed.utils import *
from train.model.gpt import GPT
from train.model.rm import RM
from train.utils.init_hf import init_model
from train.utils.tools import remove_path, read_path, copy_path

@dataclass
class DeepspeedConfig:
    zero_stage: Literal[2, 3] = 2,
    enable_bf16     : bool = True,
    offload         : bool = False
    adam_offload    : bool = False
    ref_offload     : bool = False # reference model offload
    train_batch_size: int = 1
    train_micro_batch_size_per_gpu: int = 1
    grad_clip       : float = 1. 
    zpg             : int = 1
    grad_accum_dtype: Literal["fp16", "bf16", "fp32"] = None
    overlap_comm    : bool = False
    load_univeral   : bool = False
    torch_compile   : bool = False #useless ?


@dataclass
class RingAttnConfig:
    ring_attn_size  : int = 1
    ring_head_stride: int = 1

class DeepspeedStrategy(Strategy):
    def __init__(
            self,
            seed: int = 42,
            full_determinism: bool = False,
            deepspeed_config: DeepspeedConfig = None,
            ring_attn_config: RingAttnConfig = None,
            lora_config: LoRAConfig = None,
    ):
        super().__init__(lora_config) #TBD
        self.seed = seed
        self.full_determinism = full_determinism
        self.config = deepspeed_config
        
        self.batch_size = self.config.train_batch_size
        self.micro_batch_size = self.config.train_micro_batch_size_per_gpu
    

        self.ring_attn_config = ring_attn_config
    
    def set_seed(self):
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)
    
    def set_device(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def init_distributed(self, timeout = timedelta(minutes = 60)):
        self.set_seed()
        self.set_device()

        deepspeed.init_distributed(timeout = timeout)
        
        self.setup_ring_attn()
        # (batch_size // (micro_batch_size * ring_groups))
        self.accumulated_gradient = self.batch_size * self.ring_attn_size // self.micro_batch_size // dist.get_world_size()



    def setup_ring_attn(self):
        self.ring_attn_size = self.ring_attn_config.ring_attn_size
        self.world_size = dist.get_world_size()
        self.ring_groups = self.world_size // self.ring_attn_size
        if self.ring_attn_size == 1:
            self.ring_attn_group = None
            self.ring_attn_rank = 0
            return
        
        ring_head_stride = self.ring_attn_config.ring_head_stride

        for i in range(0, dist.get_world_size(), self.ring_attn_size):
            ring_attn_ranks = list(range(i, i + self.ring_attn_size))

            group = dist.new_group(ranks = ring_attn_ranks, backend = 'nccl')

            if dist.get_rank() in ring_attn_ranks:
                self.ring_attn_group = group

                self.ring_attn_rank = dist.get_rank(group = group)
                self.ring_attn_ranks = ring_attn_ranks
    
        substitute_hf_flash_attn(self.ring_attn_group, ring_head_stride)
        
    
    def setup_optimizer(self, model: GPT, **kwargs):
        AdamOptimizer = DeepSpeedCPUAdam if self.config.adam_offload else FusedAdam

        optim_params = optimizer_grouped_parameters(model.model, kwargs["weight_decay"])

        return AdamOptimizer(optim_params, **kwargs)
    
    def setup_dataloader(self, 
                         dataset: tud.Dataset,
                         batch_size: int = 1,
                         bucket_size: int = None,
                         pin_memory: bool = False,
                         shuffle: bool = True,
                         collate_fn: Callable = None,
                         drop_last: bool = True,
                         drop_exceed: bool = False):
        
        '''
        batch_size is actually micro_batch_size
        '''
        num_replicas = dist.get_world_size() // self.ring_attn_size
        rank = dist.get_rank() // self.ring_attn_size

        Sampler = DistributedSampler
        sampler_type = 'sampler'
        dataloader_kwargs = dict(
            batch_size = batch_size,
            collate_fn = collate_fn,
            pin_memory = pin_memory
        )
        if bucket_size is not None:
            Sampler = DistributedBucketSampler
            sampler_type = 'batch_sampler'
            dataloader_kwargs.pop('batch_size')
        
        dataloader_kwargs[sampler_type] = Sampler(
            dataset,
            bucket_size = bucket_size,
            num_replicas = num_replicas,
            rank = rank,
            shuffle = shuffle,
            seed = self.seed,
            drop_last  = drop_last,
            drop_exceed = drop_exceed
        )

        return tud.DataLoader(
            dataset,
            ** dataloader_kwargs
        )

    def prepare_train(self, model: GPT, optimizer, scheduler):
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model = model.model,
            optimizer = optimizer,
            lr_scheduler = scheduler,
            config = deepspeed_train_config(
                        offload = self.config.offload,
                        adam_offload = self.config.adam_offload,
                        stage = self.config.zero_stage,
                        enable_bf16 = self.config.enable_bf16,
                        # train_batch_size = self.config.train_batch_size,
                        gradient_accumulation_steps = self.accumulated_gradient,
                        train_micro_batch_size_per_gpu = self.config.train_micro_batch_size_per_gpu,
                        grad_clip = self.config.grad_clip,
                        zpg = self.config.zpg,
                        grad_accum_dtype = self.config.grad_accum_dtype,
                        overlap_comm = self.config.overlap_comm,
                        load_univeral = self.config.load_univeral
                    ),
            args = dict(local_rank = int(os.environ.get("LOCAL_RANK", "-1"))),
            dist_init_required = True,
        ) 

        if self.config.torch_compile: engine.compile()

        model.model = engine

        return model, optimizer, scheduler

    
    def prepare_eval(self, model: GPT):
        engine, *_ = deepspeed.initialize(
            model = model.model,
            args = dict(local_rank = int(os.environ.get("LOCAL_RANK", "-1"))),
            config = deepspeed_eval_config(
                offload = self.config.ref_offload,
                stage = self.config.zero_stage if self.config.stage == 3 else 0,
                enable_bf16 = self.config.enable_bf16,
                train_batch_size = self.config.train_batch_size,
                train_micro_batch_size_per_gpu = self.config.train_micro_batch_size_per_gpu
            ),
            dist_init_required = True
        )
        if self.config.torch_compile: engine.compile()
        model.model = engine
        return model
    

    def save_ckpt(self, 
                  model:deepspeed.DeepSpeedEngine, 
                  save_dir: str,
                  tag: str = None,
                  max_ckpts:int = 1,
                  max_ckpt_GB: int =1000,
                  client_state: dict = dict(),
                  save_latest: bool = True):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
            MAX_SIZE = max_ckpt_GB * 1024**3
            subdirs = sorted([k for k in read_path(save_dir) if os.path.isdir(k)],
                             key = lambda x: os.path.getmtime(x))
            
            while True:
                total_size = sum(
                    os.path.getsize(os.path.join(dir_path, file_name))
                    for subdir in subdirs
                    for dir_path, folder_names, file_names in os.walk(subdir)
                    for file_name in file_names
                )

                if len(subdirs) < max_ckpts and total_size <= MAX_SIZE:
                    break

                remove_path(subdirs.pop(0))


        dist.barrier()
        model.save_checkpoint(save_dir = save_dir, 
                              tag = tag,
                              client_state = client_state,
                              save_latest = save_latest)
        
        dist.barrier()
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print(f"Model Checkpoint: {save_dir}/{tag} is ready !!!")

            
    def load_ckpt(self,
                  model: deepspeed.DeepSpeedEngine,
                  load_dir: str,
                  tag: str = None,
                  load_modudle_strict: bool = True,
                  load_optimizer_states: bool = True,
                  load_lr_scheduler_states: bool = True,
                  load_module_only: bool = False):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        with open(os.path.join(load_dir,"latest"), "r") as f:
            tag = f.read().strip()

        load_path, states = model.load_checkpoint(
            load_dir = load_dir,
            tag = tag,
            load_module_strict = load_modudle_strict,
            load_optimizer_states = load_optimizer_states,
            load_lr_scheduler_states = load_lr_scheduler_states,
            load_module_only = load_module_only
        )

        assert load_path, f"DeepSpeed failed to resume from checkoutoint `{load_dir}`"
        return load_path, states
    

    def restore_ckpt(self,
                     states: dict,
                     train_dataloader: tud.DataLoader,
                     config: TrainerConfig):
        consumed_samples = states['consumed_samples']
        total_tokens = states["total_tokens"] / self.ring_groups * (10**9)
        loss_tokens = states["loss_tokens"] / self.ring_groups * (10**9)
        update_steps_per_epoch = config.update_steps_per_epoch(train_dataloader)

        step = consumed_samples // config.batch_size * self.accumulated_gradient + 1
        start_epoch = consumed_samples // config.batch_size // update_steps_per_epoch
        consumed_samples %= (update_steps_per_epoch * config.batch_size)

        return step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens


    def save_model(self,
                   model: Union[GPT, RM],
                   tokenizer: PreTrainedTokenizer,
                   save_dir: str,
                   **kwargs):
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        model_to_save = model.model
        
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        gathered_state_dict = dict()
        for k, v in model_to_save.named_parameters():

            with deepspeed.zero.GatheredParameters([v], enabled = \
                                                   hasattr(v, "ds_id") and v.ds_status == ZeroParamStatus.NOT_AVAILABLE):
                if dist.get_rank() == 0:
                    gathered_state_dict[k] = v.data.cpu()
        
        if dist.get_rank() == 0:
            state_dict = model_to_save.state_dict()
            for k, v in model_to_save.named_buffers():
                if k in state_dict:
                    gathered_state_dict[k] = v.data.cpu()
            
            state_dict_keys = set(state_dict.keys())
            gathered_state_dict_keys = set(gathered_state_dict.keys())

            assert state_dict_keys.issubset(gathered_state_dict), \
            f"Mismatch keys: {gathered_state_dict_keys.symmetric_difference(state_dict_keys)}"

            if isinstance(model_to_save, PeftModel):
                if isinstance(model, GPT):
                    model_to_save = deepcopy(model_to_save)
                elif isinstance(model, RM):
                    base_model = init_model(model_to_save.base_model.model._load_path,
                                            model_type = "classifier")
                    cloned = get_peft_model(base_model, self.lora_config)
                    cloned.load_state_dict(model_to_save.state_dict(), strict=True)
                    model_to_save = cloned


                if self.lora_config.save_merged:
                    model_to_save = model_to_save.merge_and_unload()
                    model_to_save.save_pretrained(save_dir, ** kwargs)
                else:
                    adapter_save_dir = save_dir
                    model_to_save.save_pretrained(adapter_save_dir, ** kwargs)
                    if self.config.zero_stage == 3:
                        torch.save(
                            get_peft_model_state_dict(model_to_save, gathered_state_dict),
                            os.path.join(save_dir, "adapter_model.bin"),
                        )
                    
            else:
                model_to_save.save_pretrained(save_directory = save_dir,
                                              state_dict = gathered_state_dict,
                                              **kwargs)

            model_to_save.config.to_json_file(os.path.join(save_dir, "config.json"))

            tokenizer.save_pretrained(save_dir)


            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for file_name in read_path(train_from_model_path, concat_root = False):
                    if file_name.endswith(".py"):
                        copy_path(os.path.join(train_from_model_path, file_name),
                                os.path.join(save_dir, file_name))

        dist.barrier()
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print(f"Model Weight: {save_dir} is ready !!!")