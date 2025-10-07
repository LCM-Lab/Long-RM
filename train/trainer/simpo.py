import torch
import torch.distributed as dist
import torch.optim as opt
from torch.nn import functional as F
from typing import Literal
from tqdm import tqdm
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from train.utils.tools import all_reduce
from flash_attn.utils.distributed import all_gather
from train.utils.wandb import WandbConfig
from train.utils.tensorboard import TensorboardConfig
from train.trainer.base import Trainer, TrainerConfig
from train.model.gpt import GPT
from train.model.rm import RM
from train.model.rl import SimPOLoss, logps_from_logits
from train.model.rm import BradleyTerryLoss
from train.strategy import DeepspeedStrategy
from train.dataset.preference import PreferenceDataset
from train.utils.distributed_sampler import DistributedSampler


@dataclass
class SimPOTrainerConfig(TrainerConfig):
    beta: float = 2
    gamma: float = 0.5
    label_smoothing: float = 0.
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    # memory_save: bool = False TODO: try to use kv cache to compute common prefix
    nll_loss_weight: float = 0.


class SimPOTrainer(Trainer):
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model: GPT,
      train_dataset: PreferenceDataset,
      eval_dataset: PreferenceDataset,
      optimizer: opt.Optimizer,
      strategy: DeepspeedStrategy,
      config: SimPOTrainerConfig,
      tokenizer: PreTrainedTokenizer = None,
      save_hf_ckpt: bool = False,
      disable_ds_ckpt: bool = False,
      wandb_config: WandbConfig = None,
      tensorboard_config: TensorboardConfig = None     
    ):
        super().__init__(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            optimizer = optimizer,
            strategy = strategy,
            config = config,
            wandb_config = wandb_config,
            tensorboard_config = tensorboard_config
        )
        
        self.config = config

        self.tokenzier = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.loss_fn = SimPOLoss(beta = config.beta,
                                 gamma = config.gamma,
                                 label_smoothing = config.label_smoothing,
                                 ltype = config.loss_type)

    def fit(self, load_ckpt: bool = True):
        states = self.load_ckpt(load_ckpt = load_ckpt)
        step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens = \
            self.strategy.restore_ckpt(states, self.train_dataloader, self.config)
        
        assert len(self.train_dataloader.dataset) and update_steps_per_epoch, \
            f"train_dataset_len: {len(self.train_dataloader.dataset)} < batch_size: {self.config.batch_size}"
        if self.config.eval_steps < 0:
            self.config.eval_steps = update_steps_per_epoch


        epoch_bar = tqdm(range(start_epoch, self.config.max_epochs), 
                         desc = "Train epoch", 
                         disable = dist.get_rank() != 0)

        loss, batch_correct, batch_samples = 0, 0, 0
        chosen_rewards, reject_rewards = 0, 0
        total_tokens, loss_tokens = 0, 0
        for epoch in range(start_epoch, self.config.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )

            inital_global_step = step // self.strategy.accumulated_gradient % update_steps_per_epoch

            step_bar = tqdm(range(update_steps_per_epoch),
                            desc = f"Train step of epoch {epoch}",
                            initial = inital_global_step,
                            disable = dist.get_rank() != 0)
            if self.config.enable_micro_bar:
                micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                desc = f"Micro Batch of Step {inital_global_step}", initial = 0,
                                disable = dist.get_rank() != 0)
            self.model.train()
            for (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) in self.train_dataloader:
                inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens = self.to_current_device(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )


                preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                final_loss = preference_loss + nll_loss
                preference_correct = (chosen_reward > reject_reward).float().sum()


                self.strategy.backward(final_loss, self.model)
                self.strategy.optimizer_step(self.model)

                loss += final_loss.item()
                chosen_rewards += chosen_reward.sum().item()
                reject_rewards += reject_reward.sum().item()
                batch_correct += preference_correct.item()
                batch_samples += len(seq_lens_list[0])

                loss_token = loss_masks.int().sum().item()
                loss_tokens += loss_token
                total_tokens += (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                      /(len(seq_lens_list))

                if self.config.enable_micro_bar:
                    micro_bar.set_postfix(dict(
                        preference_loss = preference_loss.item(),
                        nll_loss = nll_loss.item() if hasattr(nll_loss, "item") else nll_loss,
                        loss = final_loss.item()
                    ))
                    micro_bar.update()
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict = dict(
                        mean_loss = loss / self.strategy.accumulated_gradient,
                        preference_loss = preference_loss.item(),
                        rewards_chosen = chosen_rewards,
                        rewards_reject = reject_rewards,
                        preference_acc = batch_correct,
                        batch_samples = batch_samples,
                        lr = self.scheduler.get_last_lr()[0]
                    )
                    if self.config.nll_loss_weight > 1e-8:
                        logs_dict["nll_loss"] = nll_loss.item()

                    logs_dict = all_reduce(logs_dict)
                    logs_dict["rewards_chosen"] /= logs_dict["batch_samples"]
                    logs_dict["rewards_reject"] /= logs_dict["batch_samples"]
                    logs_dict["preference_acc"] /= logs_dict["batch_samples"]
                    logs_dict["batch_samples"] *= self.strategy.ring_groups

                    loss, batch_correct, batch_samples = 0, 0, 0
                    chosen_rewards, reject_rewards = 0, 0
                    
                    step_bar.set_postfix(logs_dict)
                    step_bar.update()
                    
                    global_step = step // self.strategy.accumulated_gradient

                    if self.config.enable_micro_bar:
                        micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                        desc = f"Micro Batch of Step {inital_global_step}", initial = 0,
                                        disable = dist.get_rank() != 0)
                

                    tokens_dict = dict(
                        total_tokens = all_reduce(total_tokens) * self.strategy.ring_groups / 10**9,
                        loss_tokens = all_reduce(loss_tokens) * self.strategy.ring_groups / 10**9
                    )

                    visualized_dict = { **{f"train/{k}": v for k, v in \
                                           {**logs_dict, "global_step": global_step}.items()},
                                        **{f"train/{k}(B)": v for k, v in tokens_dict.items()}
                                       }

                    self.update_visualization(visualized_dict,
                                              global_step,
                                              self.config.logging_steps,
                                              step = global_step)
                    finished = (epoch + 1 == self.config.max_epochs) \
                        and (global_step == update_steps_per_epoch)
                    self.save_model(global_step = global_step, finished = finished)
                    self.evaluate(global_step, finished = finished)
                    self.save_ckpt(global_step = global_step, 
                                   client_state = dict(
                                       consumed_samples = global_step * self.config.batch_size,
                                       ** tokens_dict
                                    )
                                )
                    
                    torch.cuda.empty_cache()

                step += 1
            epoch_bar.update()
    
        self.finish_visualization()

    def evaluate(self, global_step:int = 0, finished: bool = False):
        if global_step % self.config.eval_steps and (not finished): return
        self.model.eval()
        with torch.no_grad():
            loss, batch_correct, batch_samples = 0, 0, 0
            chosen_rewards, reject_rewards = 0, 0
            total_tokens, loss_tokens = 0, 0
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = dist.get_rank() == 0,
            )
            bar_dict = dict()

            for times, (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) in enumerate(self.eval_dataloader):
                inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens = self.to_current_device(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                final_loss = preference_loss + nll_loss
                preference_correct = (chosen_reward > reject_reward).float().sum()
                
                loss += final_loss.item()
                chosen_rewards += chosen_reward.sum().item()
                reject_rewards += reject_reward.sum().item()
                batch_correct += preference_correct.item()
                batch_samples += len(seq_lens_list[0])

                loss_token = loss_masks.int().sum().item()
                loss_tokens += loss_token
                total_tokens += (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                      /(len(seq_lens_list))


                bar_dict = {"eval_loss": loss / (times + 1),
                            "eval_acc" : batch_correct / batch_samples,
                            "eval_chosen_rewards": chosen_rewards / batch_samples,
                            "eval_reject_rewards": reject_rewards / batch_samples}

                step_bar.update()
            if not bar_dict:return
            logs = all_reduce(bar_dict)
            step_bar.set_postfix(logs)
            
            tokens_dict = dict(
                total_tokens = all_reduce(total_tokens, op = "sum") / 10**9,
                loss_tokens = all_reduce(loss_tokens, op = "sum") / 10**9
            )
            visualized_dict = { **{f"eval/{k}": v for k, v in {**logs, "global_step": global_step}.items()},
                                **{f"eval/{k}(B)": v for k, v in tokens_dict.items()}
                               }

            self.update_visualization(
                visualized_dict, 
                global_step, 
                self.config.logging_steps, 
                step = global_step
            )
        
        self.model.train()

    
    def _get_loss_and_reward(self, 
                             inputs_ids: torch.LongTensor, 
                             attention_masks: torch.LongTensor, 
                             loss_masks: torch.BoolTensor, 
                             seq_lens_list: list[list[int]], 
                             packed_seq_lens: list[int], 
                             merged_seq_lens: list[int]
                            #  inputs_list: list[torch.LongTensor], 
                            #  attention_mask_list: list[torch.LongTensor], 
                            #  loss_mask_list: list[torch.BoolTensor], 
                            #  seq_lens_list: list[list[int]]
                             ):
        '''The first element of each input belongs to chosen, others belong to rejected'''
        
        chosen_logps, reject_logps, nll_loss = self._get_logps(
            self.model, inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
        )
        loss, chosen_reward, reject_reward = self.loss_fn(
            chosen_logps, reject_logps
        )

        return loss, nll_loss, chosen_reward, reject_reward

    def _get_logps(self, 
                   model: GPT,
                   inputs_ids: torch.LongTensor, 
                   attention_masks: torch.LongTensor, 
                   loss_masks: torch.BoolTensor, 
                   seq_lens_list: list[list[int]],
                   packed_seq_lens: list[int], 
                   merged_seq_lens: list[int]):
        
        packed_local_logits = model(
            sequences = inputs_ids,
            seq_lens = packed_seq_lens,
            attention_mask= attention_masks,
            ring_attn_group = self.strategy.ring_attn_group
        )["logits"]

        
        logps_sums_list, logps_means_list = self._get_logps_from_packed_logits(
            logits = packed_local_logits, labels = inputs_ids,
            masks = attention_masks.bool() & loss_masks.bool(),
            seq_lens_list = seq_lens_list,
            merged_seq_lens = merged_seq_lens

        )


        chosen_logps = logps_means_list[0]
        reject_logps = torch.stack(logps_means_list[1:]).mean(dim = 0)

        nll_loss = -logps_means_list[0].mean() * self.config.nll_loss_weight \
            if self.config.nll_loss_weight > 1e-8 else 0

        return chosen_logps, reject_logps, nll_loss



    def _get_logps_from_packed_logits(self,
                                      logits: torch.FloatTensor,
                                      labels: torch.LongTensor,
                                      masks: torch.BoolTensor,
                                      seq_lens_list: list[int],
                                      merged_seq_lens: list[int]):
        '''logits: [1, local_seq_len, vocab_size], labels: [1, seq_len]'''
        if self.strategy.ring_attn_group is None:
            full_logps = logps_from_logits(logits = logits[:, :-1, :], 
                                           labels = labels[:, 1:])
        else:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.strategy.ring_attn_size
            start_idx = self.strategy.ring_attn_rank * seq_len_per_process + 1
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)

            local_label = labels[:, start_idx: end_idx]

            if self.strategy.ring_attn_rank + 1 == self.strategy.ring_attn_size:
                local_label = F.pad(local_label, (0, 1), value = 0)
            
            
            local_logps = logps_from_logits(logits = logits.squeeze(0), 
                                            labels = local_label.squeeze(0))

            assert local_logps.ndim == 1, f"local_logps shape: {local_logps.shape}" 

            full_logps = all_gather(local_logps, self.strategy.ring_attn_group).reshape(1,-1)[:, : -1]
        
        logps_list = [full_logps[:, l: r] for l, r in zip(merged_seq_lens[:-1], merged_seq_lens[1:])]
        loss_masks_list = [masks[:, l: r] for l, r in zip(merged_seq_lens[:-1],merged_seq_lens[1:])]
        logps_sums_list, logps_means_list = [], []

        for logps, mask, seq_lens in zip(logps_list, loss_masks_list, seq_lens_list):
            loss_masks = mask[:, 1:].float()
            logps_sums, logps_means = [], []
            start_idx = 0
            for seq_len in seq_lens:
                seq = logps[0, start_idx: start_idx + seq_len - 1]
                mask = loss_masks[0, start_idx: start_idx + seq_len - 1]
                logps_sums +=[(seq @ mask)]
                logps_means += [(seq @ mask)/ mask.sum()]

                start_idx += seq_len
    
            logps_sums_list += [torch.stack(logps_sums)]
            logps_means_list += [torch.stack(logps_means)]
        
        return logps_sums_list, logps_means_list




    def _get_logps_from_logits(self,
                   logits: torch.FloatTensor,
                   labels: torch.LongTensor,
                   masks: torch.BoolTensor,
                   seq_lens: list[int]):
        '''legacy, not availble'''
        '''logits: [1, local_seq_len, vocab_size], labels: [1, seq_len]'''
        if self.strategy.ring_attn_group is None:
            logps = logps_from_logits(logits = logits[:, :-1, :], labels = labels[:, 1:])
        else:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.strategy.ring_attn_size
            start_idx = self.strategy.ring_attn_rank * seq_len_per_process + 1
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)

            local_label = labels[:, start_idx: end_idx]

            if self.strategy.ring_attn_rank + 1 == self.strategy.ring_attn_size:
                local_label = F.pad(local_label, (0, 1), value = 0)
            
            
            local_logps = logps_from_logits(logits = logits.squeeze(0), 
                                            labels = local_label.squeeze(0))

            assert local_logps.ndim == 1, f"local_logps shape: {local_logps.shape}" 

            logps = all_gather(local_logps, self.strategy.ring_attn_group).reshape(1,-1)[:, : -1]
        
        loss_masks = masks[:, 1:].float()
        logps_sums, logps_means = [], []
        start_idx = 0
        for seq_len in seq_lens:
            seq = logps[0, start_idx: start_idx + seq_len - 1]
            mask = loss_masks[0, start_idx: start_idx + seq_len - 1]
            logps_sums +=[(seq @ mask)]
            logps_means += [(seq @ mask)/ mask.sum()]

            start_idx += seq_len
        
        return torch.stack(logps_sums), torch.stack(logps_means)
    

    def _get_lops_legacy(
            self,
            model: GPT,
            inputs_list: list[torch.LongTensor], 
            attention_mask_list: list[torch.LongTensor], 
            loss_mask_list: list[torch.BoolTensor], 
            seq_lens_list: list[list[int]]
        ):

        '''forward each chosen/reject batch seperately, but not avaible'''
        logits_list = [
            model(
                sequences = inputs,
                seq_lens = seq_lens,
                attention_mask = attention_mask,
                ring_attn_group = self.strategy.ring_attn_group
            )["logits"] for inputs, seq_lens, attention_mask \
                in zip(inputs_list, seq_lens_list, attention_mask_list)
        ]

        # get the results of gathered full sequences
        logps_sums_list, logps_means_list = zip(*[
            self._get_logps_from_logits(
                logits = logits, 
                labels = inputs, 
                masks = attention_mask.bool() & loss_mask.bool(),
                seq_lens = seq_lens
            ) for logits, inputs, attention_mask, loss_mask, seq_lens in zip(
                logits_list, inputs_list, attention_mask_list, loss_mask_list, seq_lens_list
                )
        ])

        ...




class SimPOBradleyTerryRMTrainer(Trainer):
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model: RM,
      train_dataset: PreferenceDataset,
      eval_dataset: PreferenceDataset,
      optimizer: opt.Optimizer,
      strategy: DeepspeedStrategy,
      config: SimPOTrainerConfig,
      tokenizer: PreTrainedTokenizer = None,
      save_hf_ckpt: bool = False,
      disable_ds_ckpt: bool = False,
      wandb_config: WandbConfig = None,
      tensorboard_config: TensorboardConfig = None     
    ):
        super().__init__(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            optimizer = optimizer,
            strategy = strategy,
            config = config,
            wandb_config = wandb_config,
            tensorboard_config = tensorboard_config
        )
        
        self.config = config

        self.tokenzier = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.loss_fn = BradleyTerryLoss()


    def fit(self, load_ckpt: bool = True):
        states = self.load_ckpt(load_ckpt = load_ckpt)
        step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens = \
            self.strategy.restore_ckpt(states, self.train_dataloader, self.config)
        
        assert len(self.train_dataloader.dataset) and update_steps_per_epoch, \
            f"train_dataset_len: {len(self.train_dataloader.dataset)} < batch_size: {self.config.batch_size}"
        if self.config.eval_steps < 0:
            self.config.eval_steps = update_steps_per_epoch


        epoch_bar = tqdm(range(start_epoch, self.config.max_epochs), 
                         desc = "Train epoch", 
                         disable = dist.get_rank() != 0)

        loss, batch_correct, batch_samples = 0, 0, 0
        chosen_rewards, reject_rewards = 0, 0
        total_tokens, loss_tokens = 0, 0
        for epoch in range(start_epoch, self.config.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )

            inital_global_step = step // self.strategy.accumulated_gradient % update_steps_per_epoch

            step_bar = tqdm(range(update_steps_per_epoch),
                            desc = f"Train step of epoch {epoch}",
                            initial = inital_global_step,
                            disable = dist.get_rank() != 0)
            if self.config.enable_micro_bar:
                micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                desc = f"Micro Batch of Step {inital_global_step}", initial = 0,
                                disable = dist.get_rank() != 0)
            self.model.train()
            for (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) in self.train_dataloader:
                inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens = self.to_current_device(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )


                preference_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs_ids, attention_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                final_loss = preference_loss
                preference_correct = (chosen_reward > reject_reward).float().sum()


                self.strategy.backward(final_loss, self.model)
                self.strategy.optimizer_step(self.model)

                loss += final_loss.item()
                chosen_rewards += chosen_reward.sum().item()
                reject_rewards += reject_reward.sum().item()
                batch_correct += preference_correct.item()
                batch_samples += len(seq_lens_list[0])

                loss_token = loss_masks.int().sum().item()
                loss_tokens += loss_token
                total_tokens += (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                      /(len(seq_lens_list))

                if self.config.enable_micro_bar:
                    micro_bar.set_postfix(dict(
                        preference_loss = preference_loss.item(),
                        loss = final_loss.item()
                    ))
                    micro_bar.update()
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict = dict(
                        mean_loss = loss / self.strategy.accumulated_gradient,
                        preference_loss = preference_loss.item(),
                        rewards_chosen = chosen_rewards,
                        rewards_reject = reject_rewards,
                        preference_acc = batch_correct,
                        batch_samples = batch_samples,
                        lr = self.scheduler.get_last_lr()[0]
                    )

                    logs_dict = all_reduce(logs_dict)
                    logs_dict["rewards_chosen"] /= logs_dict["batch_samples"]
                    logs_dict["rewards_reject"] /= logs_dict["batch_samples"]
                    logs_dict["preference_acc"] /= logs_dict["batch_samples"]
                    logs_dict["batch_samples"] *= self.strategy.ring_groups

                    loss, batch_correct, batch_samples = 0, 0, 0
                    chosen_rewards, reject_rewards = 0, 0
                    
                    step_bar.set_postfix(logs_dict)
                    step_bar.update()
                    
                    global_step = step // self.strategy.accumulated_gradient

                    if self.config.enable_micro_bar:
                        micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                        desc = f"Micro Batch of Step {inital_global_step}", initial = 0,
                                        disable = dist.get_rank() != 0)
                

                    tokens_dict = dict(
                        total_tokens = all_reduce(total_tokens) * self.strategy.ring_groups / 10**9,
                        loss_tokens = all_reduce(loss_tokens) * self.strategy.ring_groups / 10**9
                    )

                    visualized_dict = { **{f"train/{k}": v for k, v in \
                                           {**logs_dict, "global_step": global_step}.items()},
                                        **{f"train/{k}(B)": v for k, v in tokens_dict.items()}
                                       }

                    self.update_visualization(visualized_dict,
                                              global_step,
                                              self.config.logging_steps,
                                              step = global_step)
                    finished = (epoch + 1 == self.config.max_epochs) \
                        and (global_step == update_steps_per_epoch)
                    self.save_model(global_step = global_step, finished = finished)
                    self.evaluate(global_step, finished = finished)
                    self.save_ckpt(global_step = global_step, 
                                   client_state = dict(
                                       consumed_samples = global_step * self.config.batch_size,
                                       ** tokens_dict
                                    )
                                )
                    
                    torch.cuda.empty_cache()

                step += 1
            epoch_bar.update()
    
        self.finish_visualization()

    def evaluate(self, global_step:int = 0, finished: bool = False):
        if global_step % self.config.eval_steps and (not finished): return
        self.model.eval()
        with torch.no_grad():
            loss, batch_correct, batch_samples = 0, 0, 0
            chosen_rewards, reject_rewards = 0, 0
            total_tokens, loss_tokens = 0, 0
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = dist.get_rank() == 0,
            )
            bar_dict = dict()

            for times, (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) in enumerate(self.eval_dataloader):
                inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens = self.to_current_device(
                    inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                preference_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs_ids, attention_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
                )

                final_loss = preference_loss
                preference_correct = (chosen_reward > reject_reward).float().sum()
                
                loss += final_loss.item()
                chosen_rewards += chosen_reward.sum().item()
                reject_rewards += reject_reward.sum().item()
                batch_correct += preference_correct.item()
                batch_samples += len(seq_lens_list[0])

                loss_token = loss_masks.int().sum().item()
                loss_tokens += loss_token
                total_tokens += (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                      /(len(seq_lens_list))


                bar_dict = {"eval_loss": loss / (times + 1),
                            "eval_acc" : batch_correct / batch_samples,
                            "eval_chosen_rewards": chosen_rewards / batch_samples,
                            "eval_reject_rewards": reject_rewards / batch_samples}

                step_bar.update()
            
            if not bar_dict: return

            logs = all_reduce(bar_dict)
            step_bar.set_postfix(logs)
            
            tokens_dict = dict(
                total_tokens = all_reduce(total_tokens, op = "sum") / 10**9,
                loss_tokens = all_reduce(loss_tokens, op = "sum") / 10**9
            )
            visualized_dict = { **{f"eval/{k}": v for k, v in {**logs, "global_step": global_step}.items()},
                                **{f"eval/{k}(B)": v for k, v in tokens_dict.items()}
                               }

            self.update_visualization(
                visualized_dict, 
                global_step, 
                self.config.logging_steps, 
                step = global_step
            )
        
        self.model.train()

    
    def _get_loss_and_reward(self, 
                             inputs_ids: torch.LongTensor, 
                             attention_masks: torch.LongTensor, 
                             seq_lens_list: list[list[int]], 
                             packed_seq_lens: list[int], 
                             merged_seq_lens: list[int]
                             ):
        '''The first element of each input belongs to chosen, others belong to rejected'''
        
        chosen_scores, reject_scores = self._get_scores(
            self.model, inputs_ids, attention_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
        )
        loss, chosen_reward, reject_reward = self.loss_fn(
            chosen_scores, reject_scores
        )

        return loss, chosen_reward, reject_reward

    def _get_scores(self, 
                    model: RM,
                    inputs_ids: torch.LongTensor, 
                    attention_masks: torch.LongTensor,
                    seq_lens_list: list[list[int]],
                    packed_seq_lens: list[int], 
                    merged_seq_lens: list[int]):
        
        packed_local_scores = model(
            sequences = inputs_ids,
            seq_lens = packed_seq_lens,
            attention_mask= attention_masks,
            ring_attn_group = self.strategy.ring_attn_group
        )["logits"]

        
        logps_list = self._get_scores_from_packed_rewards(
            scores = packed_local_scores,
            seq_lens_list = seq_lens_list,
            merged_seq_lens = merged_seq_lens
        )

        chosen_scores = logps_list[0]
        reject_scores = torch.stack(logps_list[1:]).mean(dim = 0)



        return chosen_scores, reject_scores



    def _get_scores_from_packed_rewards(self,
                                        scores: torch.FloatTensor,
                                        seq_lens_list: list[int],
                                        merged_seq_lens: list[int]):
        '''logits: [1, local_seq_len, 1], labels: [1, seq_len]'''
        if self.strategy.ring_attn_group is None:
            full_scores = scores.reshape(1, -1)
        else:
            full_scores = all_gather(scores.flatten(), self.strategy.ring_attn_group).reshape(1,-1)
        
        rewards_list = [full_scores[:, l: r] for l, r in zip(merged_seq_lens[:-1], merged_seq_lens[1:])]

        scores_list = []
        for rewards, seq_lens in zip(rewards_list, seq_lens_list):
            seq_scores = []
            start_idx = 0
            for seq_len in seq_lens:
                seq_scores += [rewards[0, start_idx + seq_len - 1]]
                start_idx += seq_len
    
            scores_list += [torch.stack(seq_scores)]
        
        return scores_list

