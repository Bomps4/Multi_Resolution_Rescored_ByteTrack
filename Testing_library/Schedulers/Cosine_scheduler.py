from .Base_scheduler import BaseLRScheduler
import math


class CosineLRScheduler(BaseLRScheduler):
    """
    Cosine learning rate scheduler.
    """
    def __init__(self, optimizer, total_iters,last_epoch=-1):
        super().__init__(optimizer, total_iters, last_epoch)

    def get_lr(self):
        """Compute learning rate using cosine annealing."""

        return [
            base_lr * 0.5 * (1 + math.cos(math.pi * self.current_step / self.total_iters))
            for base_lr in self._last_lr
        ]


class WarmCosineLRScheduler(BaseLRScheduler):
    """
    Cosine with warmup learning rate scheduler.
    """
    def __init__(self, optimizer, total_iters, warmup_iters, warmup_lr_start, last_epoch=-1):
        super().__init__(optimizer, total_iters, last_epoch)
        self.warmup_iters = warmup_iters
        self.warmup_lr_start = warmup_lr_start

    def get_lr(self):
        """Compute learning rate with warmup and cosine annealing."""
        if self.current_step < self.warmup_iters:
            return [
                (base_lr - self.warmup_lr_start) * (self.current_step / self.warmup_iters) + self.warmup_lr_start
                for base_lr in self._last_lr
            ]
        else:
            adjusted_step = self.current_step - self.warmup_iters
            adjusted_total_iters = self.total_iters - self.warmup_iters
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * adjusted_step / adjusted_total_iters))
                for base_lr in self._last_lr
            ]
    
    def state_dict(self):
        """Returns the state of the scheduler."""
        base_dict = super().state_dict()
        base_dict["warmup_iters"] =  self.warmup_iters
        base_dict['warmup_lr_start'] = self.warmup_lr_start

        return base_dict

    def load_state_dict(self, state_dict):
        """Loads the state of the scheduler."""
        self.current_step = state_dict['current_step']
        self.total_iters = state_dict['total_iters']
        self._last_lr = state_dict['last_lr']
        self.warmup_iters = state_dict['warmup_iters']
        self.warmup_lr_start = state_dict['warmup_lr_start']

        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr

