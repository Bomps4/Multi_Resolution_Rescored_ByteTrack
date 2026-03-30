from torchvision.models.detection.ssd import SSDScoringHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import torchvision.models.detection._utils as det_utils
import math
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import List,Union,Tuple,Callable,Dict,Optional
from torch import nn
from ..Layers.Convolutional_Base import DWConv
from ...utils.general_functions import format_outputs
from ...utils.Box_encoder import BoxCoder
from ...utils.box_ops import box_xyxy_to_cxcywh,decode_boxes


def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

def _uniform_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.uniform_(layer.weight, a=0.0, b=1.0/layer.weight.numel())
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def _SSDLITE_prediction_block (input_channel,output_channels):
    layers=[nn.Conv2d(input_channel,input_channel,kernel_size = 3,groups = input_channel,stride=1, padding=1, dilation=1)]
    layers+=[nn.BatchNorm2d(input_channel,eps=0.001, momentum=0.03),nn.ReLU6()]
    layers+=[nn.Conv2d(input_channel,output_channels,kernel_size=1)]#pointwise
    return nn.Sequential(*layers)
    


class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(
        self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(_SSDLITE_prediction_block(channels,num_classes*anchors))
        _normal_init(cls_logits)
        super().__init__(cls_logits, num_classes)
    @torch.compile
    def forward (self,x):
        return super().forward(x)


class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int],ksize:Union[Tuple,int]=3,stride:int=1,activation:str='relu6',bias:bool=False):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(_SSDLITE_prediction_block(channels, 4* anchors))
        _normal_init(bbox_reg)
        super().__init__(bbox_reg, 4)
    @torch.compile
    def forward (self,x):
        return super().forward(x).sigmoid()

class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, criterion):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors)
        self.criterion = criterion

        
        

    
    def forward(self,anchors,features,targets):
        classes_outputs = self.classification_head(features['classification_features'])
        regression_outputs = self.regression_head(features['regression_features'])

        boxes_per_feature = [i.size(-1)*i.size(-2) for i in features['classification_features']]

        regression_outputs = decode_boxes(regression_outputs,anchors[None])

        outputs = format_outputs(regression_outputs,classes_outputs)

       
        if self.training:

            return self.criterion(outputs,targets,anchors[None],boxes_per_feature)

        else:

            return outputs


