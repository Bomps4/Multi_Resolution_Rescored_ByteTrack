from torch import nn
from ..Layers.Convolutional_Base import BaseConv,DWConv

from typing import List
import torch

class Head_Net(nn.Module):
    def __init__(self,input_channels,num_repeats,num_outputs,num_anchors,act='lrelu'):
        super(Head_Net, self).__init__()
        self.num_outputs=num_outputs
        self.feat_conv=nn.Sequential(*[BaseConv(input_channels,input_channels,ksize=3,stride=1,act=act) for _ in range(num_repeats)])
        self.last_conv=BaseConv(input_channels,num_outputs*num_anchors,ksize=3,stride=1,act=act)
    def forward(self,x:List[torch.Tensor]):
        output=[]
        for idx,i in enumerate(x):
            output.append(self.feat_conv(i))
            output[-1]=self.last_conv(output[-1])
            output[-1]=output[-1].view(-1,self.num_outputs)
        return torch.cat(output,dim=0)
