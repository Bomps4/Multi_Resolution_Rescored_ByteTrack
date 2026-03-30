
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
    elif name == 'relu6':
        module = nn.ReLU6(inplace=inplace)
    elif name == 'sigmoid':
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module