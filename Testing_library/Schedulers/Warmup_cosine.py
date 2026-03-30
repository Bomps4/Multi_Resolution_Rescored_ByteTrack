from torch.optim.lr_scheduler import _LRScheduler
import math
from .base_warmup import Warmup_Base_Scheduler
from loguru import logger
__all__=['Linear_Warmup_Cosine_Schedule']

class Linear_Warmup_Cosine_Schedule(Warmup_Base_Scheduler):
    def __init__(self,optimizer,warmup_steps,lr_after_warmup,T_max,eta_min=0, last_epoch=-1, verbose=False):
        print('started')
        self.T_max=T_max
        self.lr_after_warmup=lr_after_warmup
        self.eta_min=eta_min
        self.group_step=[(self.lr_after_warmup-group)/warmup_steps for group in [group['lr'] for group in optimizer.param_groups]]
        super(Linear_Warmup_Cosine_Schedule,self).__init__(optimizer,warmup_steps, last_epoch, verbose)
        
        print(self.__dict__)
        
    def lr_warmup(self):
        return [group['lr'] + self.group_step[idx] for idx,group in enumerate(self.optimizer.param_groups)]
    def lr_normal(self):
        if self._step_count == self.warmup_steps:
            return [self.eta_min + (group['lr'] - self.eta_min) *
                (1 + math.cos((self._step_count) * math.pi / self.T_max)) / 2
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self._step_count - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self._step_count / self.T_max)) /
                (1 + math.cos(math.pi * (self._step_count - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
    
    
    