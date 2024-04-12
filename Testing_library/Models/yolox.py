#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

import torch.nn as nn
import torch
from typing import Tuple
from .Heads.yolo_head import YOLOXHead
from .Heads.yolo_pafpn import YOLOPAFPN
from ..utils.postprocess import postprocess

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    
    def out_to_detectron(self,outputs:torch.Tensor,image_sizes:Tuple[int,int]):
        from detectron2.structures.boxes import Boxes
        from detectron2.structures.instances import Instances
        bboxes = outputs[:, 0:4]
        cls = outputs[:, 5]
        scores = outputs[:, 4] 
        #print(f'sono outputs 3 e 4 {outputs[:, 4]},{ outputs[:, 5]}')
        new_output={}
        nn_results={'pred_boxes':Boxes(bboxes),'scores':scores,'pred_classes':cls}
        new_output['instances']=Instances(image_size=image_sizes,**nn_results)

        return new_output
    
    def post(self,prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        outputs = postprocess(prediction, num_classes, conf_thre, nms_thre, class_agnostic,xyxy=self.xyxy)
        outputs=[i for i in outputs if i is not None]
        current = [torch.empty_like(i).to(i.device) for i in outputs]
        for idx,i in enumerate(outputs):
            current[idx][...,:4]=i[...,:4]
            current[idx][..., 4]=i[...,4]*i[...,5]
            current[idx][..., 5]=i[...,6]
            current[idx]=current[idx][...,:-1]
        # print('sono current',current)
        # input()
        return current

    
    def from_nntool_conv(self,outputs,conf_thre,nm_thre):
        correct_out=torch.from_numpy(outputs[0])[None]
        return self.head.decode_outputs(correct_out,torch.float32)


    def __init__(self, backbone=None, head=None,xyxy=True):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.xyxy=xyxy
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        self.head.decode_in_inference=not torch.onnx.is_in_onnx_export()
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
