import math
import torch
from typing import Optional

class BaseLRScheduler(object):
    """
    Base class for learning rate schedulers with an internal step counter.
    Includes save/load functionality for seamless resumption of training.
    """
    def __init__(self, optimizer, total_iters:int, last_epoch:int=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
            total_iters (int): Total number of iterations in the training job.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.current_step = 0  # Internal step counter
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

        # Initialize with last_epoch if resuming training
        if last_epoch >= 0:
            self.current_step = last_epoch * total_iters

    def get_lr(self):
        """Override this method in derived classes to compute new learning rates."""
        raise NotImplementedError

    def step(self, step:Optional[int]=None):
        """
        Updates the learning rate of the optimizer.
        
        Args:
            step (int, optional): Manually specify the current step. If not provided,
                                  the internal step counter is used and incremented.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        self._last_lr = new_lrs

    @property
    def last_lr(self):
        """Returns the most recent learning rates."""
        return self._last_lr

    def state_dict(self):
        """Returns the state of the scheduler."""
        return {
            'current_step': self.current_step,
            'total_iters': self.total_iters,
            'last_lr': self._last_lr,
        }

    def load_state_dict(self, state_dict):
        """Loads the state of the scheduler."""
        self.current_step = state_dict['current_step']
        self.total_iters = state_dict['total_iters']
        self._last_lr = state_dict['last_lr']
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


