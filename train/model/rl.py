from typing import  Literal
import torch
from torch import nn
from torch.nn import functional as F


from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

def logps_from_logits(logits: torch.FloatTensor,
                      labels: torch.LongTensor,
                      temperature: float = 1.0,
                      ignore_index: int = -100):
    assert logits.ndim == 2 and labels.ndim == 1, \
        f"logits shape: {logits.shape}, labels shape: {labels.shape}"
    if temperature != 1.0:
        logits.div_(temperature)
    
    if logits.dtype in (torch.float32, torch.float64):
        dims = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logps, _ = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1),
                                      ignore_index = ignore_index)
        
        return -logps.view(*dims)
    safe_mask = (labels != ignore_index)
    safe_labels = labels.clone().masked_fill_(~safe_mask, 0)
    return (F.log_softmax(logits, dim = -1)\
            .gather(dim = -1, index = safe_labels.unsqueeze(-1)).squeeze(-1)) * safe_mask.float()


class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
    
    def forward(self,
                policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                reference_chosen_logps: torch.FloatTensor,
                reference_rejected_logps: torch.FloatTensor):
        policy_delta = policy_chosen_logps - policy_rejected_logps
        reference_delta = reference_chosen_logps - reference_chosen_logps

        logits = policy_delta - reference_delta

        if self.ipo:
            loss = (logits - 1 / (2 * self.beta)) ** 2
        else:
            loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards, rejected_rewards


class SimPOLoss(nn.Module):
    def __init__(self, 
                 beta: float = 2.0, gamma: float = 0.5, 
                 label_smoothing: float = 0.0, 
                 ltype: Literal["sigmoid", "hing"] = "sigmoid"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.gamma_beta_ratio = gamma / beta
        self.label_smoothing = label_smoothing        
        self.ltype = ltype
    def forward(self,
                policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor):
        '''
        policy_rejected_logps may be the mean of multiple rejected_logps 
        '''

        policy_delta = (policy_chosen_logps - policy_rejected_logps).flatten()
        logits = policy_delta - self.gamma_beta_ratio


        if self.ltype == "hinge":
            loss = torch.relu(1 - self.beta * logits)
        elif self.ltype == "sigmoid":
            loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.ltype}. Should be one of ['sigmoid', 'hinge']"
            )

        
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return loss.mean(), chosen_rewards, rejected_rewards

