# Copyright (c) 2021-2022 Megvii Inc. and its affiliates. 
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

import torch
import torch.nn as nn

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)



def get_activation(name="silu", inplace=True):
    if issubclass(type(name), nn.Module):
        return name
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'swish':
        module = nn.SiLU(inplace=inplace)
    elif name == 'hsigm':
        module = nn.Hardsigmoid(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
