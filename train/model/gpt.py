from typing import Optional, Union
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import deepspeed
from transformers import PreTrainedModel
from train.utils.sequence import mask2position
from train.utils.ring_attn import set_cu_seqlens_for_ring_attn

class GPT(nn.Module):
    def __init__(self, 
                 model: Union[PreTrainedModel, deepspeed.DeepSpeedEngine]):
        super().__init__()
        self.model = model

    
    @torch.no_grad()
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)



    def forward(self,
                sequences: torch.LongTensor, #packed_sequences
                seq_lens: Optional[list[int]] = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                ring_attn_group: Optional[dist.ProcessGroup] = None,
                ):
        '''
        input: sequences[..., :-1]
        label: sequences[..., 1: ]
        mask: control causal relationship
        '''

        if ring_attn_group is None:
            position_ids = mask2position(attention_mask)
        else:
            labels = sequences
            sequences, attention_mask, position_ids = set_cu_seqlens_for_ring_attn(
                sequences, attention_mask, seq_lens, ring_attn_group
            )

        output = self.model(sequences, 
                            attention_mask = attention_mask, 
                            position_ids = position_ids)

        output["logits"] = output["logits"].to(torch.float32)

        return output








class GPTCELoss(nn.Module):
    def __init__(self, ring_attn_group = None, ignore_index:int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index = ignore_index)

        self.ring_attn_group = ring_attn_group
        if ring_attn_group:
            self.ring_attn_rank = dist.get_rank(ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(ring_attn_group)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        logits: ring local logits (including the last useless logit)
        '''
        if self.ring_attn_group is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        total_seq_len = labels.size(-1)
        seq_len_per_process = total_seq_len // self.ring_attn_world_size
        start_idx = self.ring_attn_rank * seq_len_per_process + 1
        
        end_idx = min(start_idx + seq_len_per_process, total_seq_len)
        
        full_labels = labels
        labels = labels[..., start_idx: end_idx] #shift_logits is automatic between star_idx and end_idx

        if self.ring_attn_rank + 1 == self.ring_attn_world_size:
            labels = F.pad(labels, (0, 1), value = self.loss.ignore_index)

        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()

        if torch.all(shift_labels == self.ignore_index):
            loss = shift_logits.mean() * 0
        else:
            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        calculated = (shift_labels != self.ignore_index).long().sum().item()
        all_calculated = (full_labels != self.ignore_index).long().sum().item()

        loss *= (calculated / max(all_calculated, 1))

        dist.all_reduce(loss, op = dist.ReduceOp.SUM, group = self.ring_attn_group)

        return loss#/ self.ring_attn_world_size



