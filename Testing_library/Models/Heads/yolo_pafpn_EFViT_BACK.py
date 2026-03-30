#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..Layers.Convolutional_Base import BaseConv,DWConv
from ..Layers.Convolutional_YOLO import  CSPLayer
from timm.models import create_model

class YOLOPAFPN_EFVT(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        backbone_name:str,
        pretrained:bool,
        depth=1.0,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = create_model(
            backbone_name,
            features_only=True,
            pretrained=False,
            img_size=None,no_jit=True)
            # pretrained_cfg_overlay={'file': '/leonardo/home/userexternal/lbompani/models--timm--efficientvit_b0.r224_in1k/snapshots/4844777adc2285905eeb022672ba81620b52bd0c/model.safetensors'})
        
        print(self.backbone.feature_info.channels())
        self.in_channels = self.backbone.feature_info.channels()[-3:]
        in_channels = self.in_channels
        self.strides = [8,16,32]
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] ), int(in_channels[1] ), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] ),
            int(in_channels[1] ),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] ), int(in_channels[0] ), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] ),
            int(in_channels[0] ),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] ), int(in_channels[0] ), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] ),
            int(in_channels[1] ),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] ),
            int(in_channels[2] ),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, _input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        

        # Print all named parameters (weights and biases)
        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}:\n{param.data}\n")
        
        #  backbone
        out_features = self.backbone(_input)[-3:]
        
        
        features = out_features
        [x2, x1, x0] = features
  

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
