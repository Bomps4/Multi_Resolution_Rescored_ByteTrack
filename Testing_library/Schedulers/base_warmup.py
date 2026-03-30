from torch.optim.lr_scheduler import _LRScheduler



__all__=["Warmup_Base_Scheduler"]

class Warmup_Base_Scheduler(_LRScheduler):
    def __init__(self,optimizer,warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps=warmup_steps
        super(Warmup_Base_Scheduler,self).__init__(optimizer, last_epoch, verbose)
        
    def lr_warmup(self):
        raise NotImplementedError
    def lr_normal(self):
        raise NotImplementedError
    def get_lr(self):
        if(self._step_count<=self.warmup_steps):
            return self.lr_warmup()
        else:
            return self.lr_normal()