import math

import torch
import torch.optim.lr_scheduler as tol

class WarmupCosineScheduler(tol._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            return [
                0.5 * base_lr * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps))
                for base_lr in self.base_lrs
            ]

class WarmupCosLR(tol._LRScheduler):
    def __init__(self,
                 optimizer,  
                 max_lr,
                 min_lr,
                 total_epochs,
                 
                 warmup_steps=None,
                 warmup_rate=None,
                 last_epoch=-1):
        
        if warmup_steps is not None:
            self.warmup_steps=warmup_steps
        if warmup_rate is not None:
            self.warmup_steps=int(total_epochs*warmup_rate)
        
        assert self.warmup_steps > 1
        self.max_lr=max_lr
        self.min_lr=min_lr
        self.total_epochs=total_epochs
        
        super().__init__(
            optimizer,
            last_epoch
        )

    def get_lr(self) -> float:
        assert self.max_lr>self.base_lrs[0]
        if self.last_epoch<self.warmup_steps:
            return [lr+self.last_epoch*(self.max_lr-self.base_lrs[0])/(self.warmup_steps-1)\
                    for lr in self.base_lrs]
        else:
            lr=self.min_lr+\
                0.5*(self.max_lr-self.min_lr)*(1.+ math.cos(math.pi * (self.last_epoch+1 - self.warmup_steps)/\
                                                (self.total_epochs-self.warmup_steps)))
            
            return [lr for _ in self.base_lrs]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "+ \
            f"base_lr={self.base_lrs[0]}, max_lr={self.max_lr}, min_lr={self.min_lr})"

    
    
    
    
    
    
class WarmupLinearLR(tol._LRScheduler):
    def __init__(self,
                 optimizer,  
                 max_lr,
                 min_lr,
                 total_epochs,
                 
                 warmup_steps=None,
                 warmup_rate=None,
                 last_epoch=-1):
        
        if warmup_steps is not None:
            self.warmup_steps=warmup_steps
        if warmup_rate is not None:
            self.warmup_steps=int(total_epochs*warmup_rate)
        
        assert self.warmup_steps > 1
        self.max_lr=max_lr
        self.min_lr=min_lr
        self.total_epochs=total_epochs
        
        super().__init__(
            optimizer,
            last_epoch
        )

    def get_lr(self) -> float:
        assert self.max_lr>self.base_lrs[0]
        if self.last_epoch<self.warmup_steps:
            return [lr+self.last_epoch*(self.max_lr-self.base_lrs[0])/(self.warmup_steps-1)\
                    for lr in self.base_lrs]
        else:
            lr=self.min_lr+\
                (self.max_lr-self.min_lr)*(self.total_epochs-self._step_count)/(self.total_epochs-self.warmup_steps)
            
            return [lr for _ in self.base_lrs]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "+ \
            f"base_lr={self.base_lrs[0]}, max_lr={self.max_lr}, min_lr={self.min_lr})"
