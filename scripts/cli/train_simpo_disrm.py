import sys, pickle, os
os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from datasets import load_from_disk, load_dataset, Dataset
import argparse
from train.strategy import (
    DeepspeedConfig, 
    RingAttnConfig, 
    LoRAConfig,
    DeepspeedStrategy
)
from train.utils.tools import cached, basename, path_join
from train.utils.tensorboard import TensorboardConfig
from train.trainer import SimPOTrainerConfig, SimPOBradleyTerryRMTrainer
from train.model import RM
from train.dataset import PreferenceDataset, BlendedPreferenceDataset
from train.utils.init_hf import init_model, init_tokenizer
import torch.distributed as dist

def train(args):


    trainer_config = SimPOTrainerConfig(
        gamma =  args.simpo_gamma,
        beta = args.simpo_beta,
        nll_loss_weight = args.simpo_nll_loss_weight,
        batch_size = args.batch_size,
        bucket_size = getattr(args, "bucket_size", None),
        micro_batch_size = args.micro_batch_size,
        max_epochs = args.max_epochs,
        scheduler = "cosine_with_min_lr",
        learing_rate = getattr(args, "learning_rate", 5e-6),
        lr_warmup_ratio = getattr(args, "lr_warmup_ratio", 0.03),
        save_steps = getattr(args, "save_steps", -1),
        ckpt_dir = getattr(args,"ckpt_dir", None),
        max_ckpts = getattr(args,"max_ckpts", 1),
        weights_dir = getattr(args, "weights_dir", None),
        eval_steps = getattr(args, "eval_steps", 20),
        max_weights = args.max_weights,
        weights_saving_interval = args.weights_saving_interval,
    )

    deepspeed_config = DeepspeedConfig(
        zero_stage = getattr(args, "zero_stage", 2),
        enable_bf16 = getattr(args, "enable_bf16", False),
        offload = getattr(args, "offload", False),
        adam_offload = getattr(args, "adam_offload", False),
        train_batch_size = trainer_config.batch_size,
        train_micro_batch_size_per_gpu = trainer_config.micro_batch_size,
        grad_clip = getattr(args, "grad_clip", 1.0),
        zpg = getattr(args,"zpg", 1),
    )

    ring_attn_config = RingAttnConfig(
        ring_attn_size = getattr(args,"ring_attn_size", 8),
        ring_head_stride = getattr(args, "ring_head_stride", 1),    
    )

    lora_config = LoRAConfig(
        r = getattr(args, "lora_r", 8),
        lora_alpha = getattr(args, "lora_alpha", 8),
        target_modules = getattr(args, "lora_target_modules", "all-linear"),
        lora_dropout = getattr(args, "lora_dropout", 0.0),
    )

    tensorboard_config = TensorboardConfig(
        log_dir = args.tensorboard_logdir,
        name = args.tensorboard_name
    )

    strategy = DeepspeedStrategy(
        seed = getattr(args, "seed", 42),
        deepspeed_config = deepspeed_config,
        ring_attn_config = ring_attn_config,
        lora_config = lora_config
    )

    strategy.init_distributed()

    if dist.get_rank() == 0:
        print("WORLD_SIE####     : ", dist.get_world_size())

    simpo_model = init_model(getattr(args, "load_from", args.model_path), model_type = "classifier")
    tokenizer =init_tokenizer(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = strategy.prepare_if_lora(RM(simpo_model))

    print(f"DATASETS: {args.dataset_paths}")
    datasets_dicts = [
        load_from_disk(pth) for pth in args.dataset_paths
    ]
    
    train_set,eval_set = None, None
    if dist.get_rank() == 0:
        [
            cached(
                PreferenceDataset,
                kwargs = dict(
                    dataset = Dataset.from_dict(dataset['train'][:6000]),
                    prompt_key = "chat_template",
                    chosen_key = "chosen",
                    rejected_key ="rejects",
                    num_rejects = args.num_rejects,
                    sample_rejects = lambda x, n: x[:n],
                    tokenizer = tokenizer,
                    max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size,
                    num_processors = 1,
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_length//1000}k=r{args.num_rejects}"
            ) for dataset, data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

        [
            cached(
                PreferenceDataset,
                kwargs = dict(
                    dataset = Dataset.from_dict(dataset['eval'][:150]),
                    prompt_key = "chat_template",
                    chosen_key = "chosen",
                    rejected_key ="rejects",
                    num_rejects = args.num_rejects,
                    sample_rejects = lambda x, n: x[:n],
                    tokenizer = tokenizer,
                    max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size,
                    num_processors = 1,
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_length//1000}k=r{args.num_rejects}=eval"
            ) for dataset, data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    dist.barrier()


    prefer_datasets = [
        cached(
            PreferenceDataset,
            kwargs = dict(
                dataset = train_set,
                prompt_key = "chat_template",
                chosen_key = "chosen",
                rejected_key ="rejects",
                num_rejects = args.num_rejects,
                sample_rejects = lambda x, n: x[:n],
                tokenizer = tokenizer,
                max_length = args.max_length,
                ring_attn_size = ring_attn_config.ring_attn_size,
                num_processors = 1,
            ),
            cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
            f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_length//1000}k=r{args.num_rejects}"
        ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
    ]


    prefer_datasets_eval = [
            cached(
                PreferenceDataset,
                kwargs = dict(
                    dataset = Dataset.from_dict(dataset['eval'][:150]),
                    prompt_key = "chat_template",
                    chosen_key = "chosen",
                    rejected_key ="rejects",
                    num_rejects = args.num_rejects,
                    sample_rejects = lambda x, n: x[:n],
                    tokenizer = tokenizer,
                    max_length = args.max_length,
                    ring_attn_size = ring_attn_config.ring_attn_size,
                    num_processors = 1,
                ),
                cache_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset_files/" + \
                f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_length//1000}k=r{args.num_rejects}=eval"
            ) for dataset, data_pth in zip(datasets_dicts, args.dataset_paths)
        ]


    train_dataset = BlendedPreferenceDataset(prefer_datasets,
                                      sample_ratios = args.sample_ratios,
                                      sample_counts = args.sample_counts)    
    eval_dataset = BlendedPreferenceDataset(prefer_datasets_eval,
                                      sample_counts = args.sample_counts_eval)


    trainer = SimPOBradleyTerryRMTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        optimizer = strategy.setup_optimizer(model,
                                             lr = args.learning_rate,
                                             betas = (0.9, 0.95),
                                             weight_decay = args.L2),
        strategy = strategy,
        config = trainer_config,
        tokenizer = tokenizer,
        tensorboard_config = tensorboard_config
        
    )


    trainer.fit(load_ckpt = True) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default = -1)
    parser.add_argument("--workspace_path", type = str, default = f"{os.path.dirname(os.path.abspath(__file__))}/")

    parser.add_argument("--save_name", type=str, required=True)

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--dataset_paths", type = str, nargs = "+")
    parser.add_argument("--sample_ratios", type = int, nargs = "+")
    parser.add_argument("--sample_counts", type = int, nargs = "+")
    parser.add_argument("--sample_counts_eval", type = int, nargs = "+")
    parser.add_argument("--save_steps", type = int, default = 20, 
                        help = "ckpt saving interval")
    parser.add_argument("--weights_saving_interval", type = int, default = 10)
    parser.add_argument("--max_weights", type = int, default = 10)
    parser.add_argument("--max_ckpts", type = int, default = 2)
    parser.add_argument("--max_length", type = int, default = 128000)
    parser.add_argument("--bucket_size", type = int, default = 128000)
    parser.add_argument("--num_rejects", type = str, default = 1)

    parser.add_argument("--simpo_gamma", type = float, default = 0.5)
    parser.add_argument("--simpo_beta", type = float, default = 2.5)
    parser.add_argument("--simpo_nll_loss_weight", type = float, default = 0.0)

    parser.add_argument("--zero_stage", type = int, default = 2)
    parser.add_argument("--micro_batch_size", type = int, default = 1)
    parser.add_argument("--batch_size", type = int, default = 16)

    parser.add_argument("--ring_attn_size", type = int, default = 8)
    parser.add_argument("--ring_head_stride", type = int, default = 4)
    parser.add_argument("--max_norm", type = float, default = 1.0)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--enable_bf16", type = bool, default = True)
    parser.add_argument("--zpg", type = int, default = 1)
    parser.add_argument("--over_lap_comm", type = bool, default = True)
    parser.add_argument("--max_epochs", type = int, default = 1)
    parser.add_argument("--learning_rate", type = float, default = 2e-6)
    parser.add_argument("--lr_warmup_ratio", type = float, default = 0.05)
    parser.add_argument("--L2", type = float, default = 0.0)
    
    args = parser.parse_args()


    if not hasattr(args,"bucket_size"):
        args.bucket_size = None
    

    args.weights_dir = f"{args.workspace_path}/{args.save_name}-weight"
    args.ckpt_dir = f"{args.workspace_path}/{args.save_name}-ckpt"
    args.tensorboard_name = args.save_name
    args.tensorboard_logdir = f"{args.workspace_path}/tensorboard"


    train(args)