from .Convolutional_Base import DWConv,BaseConv
import torch 
from .Activations import get_activation
from torch import nn



def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x






class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="relu"):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1,groups=inp
                ),
                nn.BatchNorm2d(inp),
                BaseConv(
                    inp, branch_features, ksize=1, stride=1, bias=False,act=activation
                )
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            get_activation(activation),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=3,
                groups=branch_features,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            BaseConv(
                branch_features,
                branch_features,
                ksize=1,
                stride=1,
                bias=False,
                act=activation
            )
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
