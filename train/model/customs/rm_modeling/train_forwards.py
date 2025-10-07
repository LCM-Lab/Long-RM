from typing import Optional, Union
from functools import partial
import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast
)


def LlamaForSequenceClassificationWithNormal_Weights(
    input_ids: torch.LongTensor = None, #packed_sequences
    attention_mask: Optional[torch.BoolTensor] = None, 
    position_ids: Optional[torch.LongTensor]= None,
    model = None,
    ** kwargs,
):

    output: BaseModelOutputWithPast = model.model(input_ids = input_ids,
                                                    attention_mask = attention_mask,
                                                    position_ids = position_ids)
    hidden_states = output[0]
    logits = model.score(hidden_states).detach()
    weights = model.weights(hidden_states.detach())

    rews = logits.view(-1, 5, 2)[:, :, 0].view(1, -1, 5)
    scores = (rews * weights).sum(dim = -1).view(1, -1)


    output = SequenceClassifierOutputWithPast(
        logits = scores,
    )        

    return output
