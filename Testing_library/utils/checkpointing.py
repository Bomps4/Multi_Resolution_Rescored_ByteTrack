# Copyright (c) 2021-2022 Megvii Inc. and its affiliates. 
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

from typing import Type
import torch
from loguru import logger

SAVING_KEYS=['model','optimizer','lr_scheduler']

def save_experiment_checkpoint(path:str,**kwargs):
    
    for key in kwargs:
        if key not in SAVING_KEYS:
            logger.warning(f"{key} is not recognized. The key:value pair will be saved but remember to add it in SAVING_KEYS")

    torch.save(kwargs,path)

def load_checkpoint(ckpt_dict,exp):
    print('exp',exp.__dict__.keys())
    print('ckpt',ckpt_dict.keys())
    print('ckpt',ckpt_dict['lr_scheduler'])
    print('exp',exp.__dict__['lr_scheduler'])
    for key in SAVING_KEYS:
        try:
            exp.__dict__[key].load_state_dict(ckpt_dict[key])
        except AttributeError:
            exp.__dict__[key]=ckpt_dict[key]



