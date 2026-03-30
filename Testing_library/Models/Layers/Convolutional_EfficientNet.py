from .Activations import get_activation
from .Convolutional_Base import BaseConv,SqueezeExcite
import torch 
from torch import nn
import math

def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(
        self,
        inp,
        final_oup,
        k,
        s,
        expand_ratio,
        se_ratio,
        has_se=False,
        activation="relu",
        drop_connect_rate=None
    ):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self.first_stage=BaseConv(in_channels=inp, out_channels=oup,stride=1, ksize=1, bias=False,act=activation)
            

        # Depthwise convolution phase
        self._depthwise_stage = BaseConv(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            ksize=k,
            stride=s,
            bias=False,
            act=activation
        )
        

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            self.se_stage=SqueezeExcite(
                inp,
                se_ratio,
            )
            
        # Output phase
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._momentum, eps=self._epsilon
        )
        if drop_connect_rate is not None:
            self.dropout=nn.Dropout(drop_connect_rate)
        else:
            self.dropout= drop_connect_rate

    def forward(self, x):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self.first_stage(x)
        x = self._depthwise_stage(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self.se_stage(x)

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            if self.dropout is not None:
                x = self.dropout(x)
            x += identity  # skip connection
        return x