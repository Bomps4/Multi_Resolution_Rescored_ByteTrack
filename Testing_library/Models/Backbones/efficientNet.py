from torch import nn
import torch 
from ..Layers.Convolutional_Base import BaseConv 
from ..Layers.Convolutional_EfficientNet import MBConvBlock,round_filters,round_repeats
import math 

from typing import List

efficientnet_lite_params = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    "efficientnet_lite0": [1.0, 1.0, 224, 0.2],
    "efficientnet_lite1": [1.0, 1.1, 240, 0.2],
    "efficientnet_lite2": [1.1, 1.2, 260, 0.3],
    "efficientnet_lite3": [1.2, 1.4, 280, 0.3],
    "efficientnet_lite4": [1.4, 1.8, 300, 0.3],
}


class EfficientNetLite(nn.Module):
    def __init__(
        self, model_name, out_stages=(2, 4, 6), activation="relu", pretrain=True
    ):
        super(EfficientNetLite, self).__init__()
        assert set(out_stages).issubset(i for i in range(0, 7))
        assert model_name in efficientnet_lite_params

        self.model_name = model_name
        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        width_multiplier, depth_multiplier, _, dropout_rate = efficientnet_lite_params[
            model_name
        ]
        self.drop_connect_rate = 0.2
        self.out_stages = out_stages

        mb_block_settings = [
            # repeat|kernel_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],  # stage0
            [2, 3, 2, 6, 16, 24, 0.25],  # stage1 - 1/4
            [2, 5, 2, 6, 24, 40, 0.25],  # stage2 - 1/8
            [3, 3, 2, 6, 40, 80, 0.25],  # stage3
            [3, 5, 1, 6, 80, 112, 0.25],  # stage4 - 1/16
            [4, 5, 2, 6, 112, 192, 0.25],  # stage5
            [1, 3, 1, 6, 192, 320, 0.25],  # stage6 - 1/32
        ]

        # Stem
        out_channels = 32
        self.stem = BaseConv(3, out_channels, ksize=3, stride=2, bias=False,act='relu')

        # Build blocks
        self.blocks = nn.ModuleList([])
        idx=0
        total_number_of_blocks=sum([i[0] for i in mb_block_settings ])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            (
                num_repeat,
                kernal_size,
                stride,
                expand_ratio,
                input_filters,
                output_filters,
                se_ratio,
            ) = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = (
                input_filters
                if i == 0
                else round_filters(input_filters, width_multiplier)
            )
            output_filters = round_filters(output_filters, width_multiplier)
            num_repeat = (
                num_repeat
                if i == 0 or i == len(mb_block_settings) - 1
                else round_repeats(num_repeat, depth_multiplier)
            )

            # The first block needs to take care of stride and filter size increase.
            stage.append(
                MBConvBlock(
                    input_filters,
                    output_filters,
                    kernal_size,
                    stride,
                    expand_ratio,
                    se_ratio,
                    has_se=False,
                    drop_connect_rate=self.drop_connect_rate*i/total_number_of_blocks
                )
            )
            idx+=1
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                
                stage.append(
                    MBConvBlock(
                        input_filters,
                        output_filters,
                        kernal_size,
                        stride,
                        expand_ratio,
                        se_ratio,
                        has_se=False,
                        drop_connect_rate=self.drop_connect_rate*idx/total_number_of_blocks
                    )
                )
                idx+=1

            self.blocks.append(stage)
        self._initialize_weights(pretrain)

    def forward(self, x:torch.Tensor)->List[torch.Tensor]:
        x = self.stem(x)
        output = []
        idx = 0
        for j, stage in enumerate(self.blocks):
            for block in stage:
                x = block(x)
                idx += 1
            if j in self.out_stages:
                output.append(x)
        return output

    def _initialize_weights(self, pretrain=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        