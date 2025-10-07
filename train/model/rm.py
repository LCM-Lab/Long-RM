from typing import Optional, Union
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import deepspeed
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast
)
from train.model.customs.rm_modeling import train_forwards
from train.utils.sequence import mask2position
from train.utils.ring_attn import set_cu_seqlens_for_ring_attn

class RM(nn.Module):
    def __init__(self, 
                 model: Union[PreTrainedModel, deepspeed.DeepSpeedEngine]):
        super().__init__()
        model._org_forward = model.forward

        train_forward = getattr(train_forwards,  model.config.architectures[0],
                                self.train_forward)


        model.forward = partial(train_forward, model = model)
        
        self.model = model
        

    
    @torch.no_grad()
    def score(self, **kwargs):
        return self.model(**kwargs)


    def train_forward(self,
                      input_ids: torch.LongTensor = None, #packed_sequences
                      attention_mask: Optional[torch.BoolTensor] = None,
                      position_ids: Optional[torch.LongTensor]= None,                     
                      model = None,
                      ** kwargs,
                      ):

        output: BaseModelOutputWithPast = model.model(input_ids = input_ids,
                                                      attention_mask = attention_mask,
                                                      position_ids = position_ids)
        hidden_states = output.last_hidden_state
        logits = model.score(hidden_states)

        output = SequenceClassifierOutputWithPast(
            logits = logits
        )        

        return output



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



class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,
                chosen_scores: torch.FloatTensor,
                rejected_scores: torch.FloatTensor):
        '''
        policy_rejected_logps may be the mean of multiple rejected_logps 
        '''

        policy_delta = (chosen_scores - rejected_scores).flatten()

        loss = -F.logsigmoid(policy_delta)
        
        chosen_rewards = chosen_scores.detach()
        rejected_rewards = rejected_scores.detach()

        return loss.mean(), chosen_rewards, rejected_rewards




