from typing import Literal

# from deepspeed import PipelineEngine
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def deepspeed_train_config(
    offload: bool,
    adam_offload: bool = True,
    stage: Literal[1, 2, 3] = 2,
    enable_bf16: bool = True,
    # train_batch_size: int = 1,
    gradient_accumulation_steps: int  = 1,
    train_micro_batch_size_per_gpu: int = 1,
    grad_clip: float = 1.0,
    zpg: int = 8,
    grad_accum_dtype: Literal["fp16", "bf16", "fp32"] = None,
    overlap_comm: bool = False,
    load_univeral: bool =False,
    activation_checkpoint: bool = True,
    cpu_checkpointing: bool = False,
):
    offload = bool(stage == 3)
    device = "cpu" if offload else "none"

    zero_opt_dict = dict(
        stage = stage,
        offload_param = dict(device = device),
        offload_optimizer = dict(
            device = "cpu" if adam_offload else "none",
            pin_memory = True
        ),
        sub_group_size = "auto",
        stage3_max_live_parameters = "auto", 
        stage3_max_reuse_distance = "auto",
        stage3_param_persistence_threshold = "auto",
        stage3_prefetch_bucket_size = "auto",
        reduce_bucket_size = "auto",
        zero_hpz_partition_size = zpg,
        zero_quantized_weights = False,
        zero_quantized_gradients = False,
    )

    if overlap_comm:
        zero_opt_dict.update(dict(
            overlap_comm = True,
            contiguous_gradients = True,
        ))
    if stage == 3:
        zero_opt_dict.update(dict(reduce_scatter = True))

    config_dict = dict(
        steps_per_print = 100,
        zero_optimization = zero_opt_dict,
        bf16 = dict(enabled = enable_bf16),
        # train_batch_size = train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu,
        gradient_clipping =  grad_clip,
        prescale_gradients = False,
        wall_clock_breakdown = False,
        data_types = dict(grad_accum_dtype = grad_accum_dtype),
        checkpoint = dict(load_universal = load_univeral),
        activation_checkpointing = dict(
            partition_activations = False,
            contiguous_memory_optimization = True,
            cpu_checkpointing = cpu_checkpointing
        )
    )

    if not enable_bf16:
        config_dict.update(fp16 = dict(enabled = True))

    if activation_checkpoint:
        config_dict.update(dict(
            partition_activations = False,
            contiguous_memory_optimization = True,
            cpu_checkpointing = cpu_checkpointing
        ))
    elif cpu_checkpointing:
        print("Warning: ", f"`cpu_checkpointing` will not "
              "be set because you didn't set `activation_checkpoint = True`")
    
    return config_dict


def deepspeed_eval_config(
    offload: bool,
    stage: Literal[0, 1, 2, 3] = 0, # 默认不启用
    enable_bf16: bool = True,
    train_batch_size: int = 1,
    train_micro_batch_size_per_gpu: int = 1,
):
    zero_opt_dict = dict(
        stage = stage,
        stage3_max_live_parameters = "auto",
        stage3_max_reuse_distance = "auto",
        stage3_param_persistence_threshold = "auto",
        stage3_prefetch_bucket_size = "auto",
        offload_param = dict(
            device = "cpu" if offload else "none",
            pin_memory = True
        )
    )
    return dict(
        steps_per_print = 100,
        zero_optimization = zero_opt_dict,
        bf16 = dict(enabled = enable_bf16),
        fp16 = dict(enabled = not enable_bf16),
        train_batch_size = train_batch_size,
        train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu,
        gradient_clipping = 1.0,
        prescale_gradients = False,
        wall_clock_breakdown = False,

    )


def optimizer_grouped_parameters(
    model: deepspeed.DeepSpeedEngine,
    weight_decay: float,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def zero3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]



def offload_deepspeed_states(model: deepspeed.DeepSpeedEngine, 
                             pin_memory: bool = True, 
                             non_blocking: bool = True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    model.optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()



def reload_deepspeed_states(model: deepspeed.DeepSpeedEngine, 
                            non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    model.reload_states(non_blocking=non_blocking)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()

