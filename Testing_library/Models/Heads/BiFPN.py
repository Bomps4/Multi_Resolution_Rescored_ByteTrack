from torch import nn 
import torch
from typing import List,Iterable
from ..Layers.Convolutional_Base import BaseConv,DWConv



class Fpn_UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize,conv_type,stride,act):
        super(Fpn_UpBlock, self).__init__()
        self.convolutions=conv_type(in_channels,out_channels,ksize,stride=stride,act=act)
        self.edge_weights = nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=True) for  _ in range(2)])
        
        self.upsample=nn.Upsample(scale_factor=2,mode='nearest')
        self.act=nn.ReLU6(inplace=False)
    def forward(self,x:Iterable[torch.Tensor]):

        x[0]=self.upsample(x[0])
        comp=x[0]*self.edge_weights[0]
        for idx,i in enumerate(self.edge_weights[1:]):
            comp+=x[idx+1]*i
        comp=self.act(comp)
        comp=self.convolutions(comp)
        return comp


        

class Fpn_DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize,conv_type,stride,act,weight_size=3):
        super(Fpn_DownBlock, self).__init__()

        self.convolutions=conv_type(out_channels,out_channels,ksize,stride=stride,act=act)
        self.edge_weights = nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(weight_size)])
        self.downsample=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.act=nn.ReLU6()
    def forward(self,x:Iterable[torch.Tensor]):
        
        x[0]=self.downsample(x[0])
        comp=x[0]*self.edge_weights[0]
        for idx,i in enumerate(self.edge_weights[1:]):
            comp+=x[idx+1]*i
        comp=self.act(comp)
        comp=self.convolutions(comp)
        return comp
        


        
        


class BiFpnLayer(nn.Module):
    def __init__(self,in_channels,out_channels,k_sizes,separable_conv,act):
        super(BiFpnLayer, self).__init__()
        self.k_sizes=k_sizes
        self.in_channels=in_channels
        self.out_channels=out_channels
        conv_type=BaseConv if not separable_conv else DWConv
        self.up_blocks=nn.ModuleList ([Fpn_UpBlock(i,j,ksize,conv_type,stride=1,act=act) for i,j,ksize in zip(self.in_channels[1:],self.out_channels[1:],self.k_sizes[1:])])
        self.down_blocks=nn.ModuleList([Fpn_DownBlock(i,j,ksize,conv_type,stride=1,act=act) if idx<len(self.out_channels)-2 else Fpn_DownBlock(i,j,ksize,conv_type,stride=1,act=act,weight_size=2) for idx,(i,j,ksize) in enumerate(zip(self.out_channels[1:],self.out_channels[1:],self.k_sizes[1:]))])
    def forward(self,x:Iterable[torch.Tensor]):
        outputs=[]
        x=x[::-1]
        outputs.append(x[0])
        for idx,x_feature in enumerate(x[:-1]):
            if(idx==0):
                outputs.append(self.up_blocks[idx]([x_feature,x[idx+1]]))
            else:
                outputs.append(self.up_blocks[idx]([outputs[-1],x[idx+1]]))
        outputs=outputs[::-1]
        for idx,middle_feature in enumerate(outputs[:-1]):
            
            if(idx<len(outputs)-2):
                outputs[idx+1]=self.down_blocks[idx]([middle_feature,outputs[idx+1],x[idx+1]])
            else:
                outputs[idx+1]=self.down_blocks[idx]([middle_feature,outputs[idx+1]])
        return outputs





class BiFpn(nn.Module):
    def __init__(self,in_channels,out_channels,k_sizes,num_modules=3,separable_conv=False,act='relu'):
        super(BiFpn, self).__init__()
        if(isinstance(k_sizes,int)):
            k_sizes=[k_sizes for _ in in_channels]
        if(isinstance(out_channels,int)):
            out_channels=[out_channels for _ in in_channels]
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.squeezing=nn.ModuleList([BaseConv(i,j,ksize=1,stride=1,act=act) for i,j in zip(in_channels,out_channels)])
        self.Fpn_layers=[]
        
        for _ in range(num_modules):
            self.Fpn_layers.append(BiFpnLayer(out_channels,out_channels,k_sizes,separable_conv,act))
        self.Fpn_layers=nn.Sequential(*self.Fpn_layers)
    def forward(self, x: List[torch.Tensor]):
        for idx,i in enumerate(x):
            x[idx]=self.squeezing[idx](i)
        x=self.Fpn_layers(x)

        return x


