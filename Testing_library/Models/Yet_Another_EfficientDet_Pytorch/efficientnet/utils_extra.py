# Author: Zylo117

import math

from torch import nn
import torch
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)


        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        
        self.conv.stride=self.stride

        self.left=int(self.kernel_size[0]-1)//(self.stride[0]*2)
        self.top=int(self.kernel_size[0]-1)//(self.stride[0]*2)

        self.right=int((self.kernel_size[0] - 1)//self.stride[0] - self.left//self.stride[0])
        self.bottom=int((self.kernel_size[0]- 1)//self.stride[0]  - self.top//self.stride[0])

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=(self.top,self.left),
        #                       bias=bias, groups=groups)
        # self.left,self.top=(self.kernel_size[0]-1)//(self.stride[0]*2),(self.kernel_size[0]-1)//(self.stride[0]*2)

        # self.right=((self.kernel_size[0] - 1)//self.stride[0] - self.left//self.stride[0])
        # self.bottom=((self.kernel_size[0]- 1)//self.stride[0]  - self.top//self.stride[0])

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=[(self.left, self.right),(self.top, self.bottom)],
        #                       bias=bias, groups=groups)
        
        
        

        

        # self.pad=torch.nn.ZeroPad2d((self.left, self.right, self.top, self.bottom))

        
        # print(f'kernel size Conv2dStaticSamePadding  {self.kernel_size}')
        # print(f' stride Conv2dStaticSamePadding {self.stride}')
        # print('-------------------------------------------------')


    def forward(self, x):
        
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        
        # print(f'calculated MaxPool2dStaticSamePadding {[self.left,self.right, self.top,self.bottom]}')
        # print(f'padding applied MaxPool2dStaticSamePadding {[left, right, top, bottom]}')
        # print('--------------------------------------------------------------------')
        # input()
        # x = F.pad(x, [left, right, top, bottom])
        x = F.pad(x, (self.left, self.right, self.top, self.bottom))#self.pad(x)

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)

        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride,self.stride] 
        elif len(self.stride) == 1:
            self.stride = [self.stride[0],self.stride[0]] 

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2



        self.left=int(self.kernel_size[0]-1)//(self.stride[0]*2)
        self.top=int(self.kernel_size[0]-1)//(self.stride[0]*2)

        self.right=int((self.kernel_size[0] - 1)//self.stride[0] - self.left//self.stride[0])
        self.bottom=int((self.kernel_size[0]- 1)//self.stride[0]  - self.top//self.stride[0])
        # kernel_size, stride,kernel_size,stride,

        # kwargs={'kernel_size':kernel_size,'stride':stride,'padding':(kernel_size-1)//2,**kwargs} ,padding='same'
        # self.pool = nn.MaxPool2d(*args,padding=(self.top,self.left), **kwargs)
        # padding=[self.left, self.right, self.top, self.bottom],
        
        
        

        # self.pad=torch.nn.ZeroPad2d((self.left, self.right, self.top, self.bottom))

        

    def forward(self, x):

        #if(torch.onnx.is_in_onnx_export()):
        
        h, w = x.shape[-2:]
        # # print(h,w)
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]


        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        # print(f'kernel size Conv2dStaticSamePadding  {self.kernel_size}')
        # print(f' stride Conv2dStaticSamePadding {self.stride}')
        # print(f'calculated MaxPool2dStaticSamePadding {[self.left,self.right, self.top,self.bottom]}')
        # print(f'padding applied MaxPool2dStaticSamePadding {[left, right, top, bottom]}')
        # print('--------------------------------------------------------------------')
        
        
        x = F.pad(x, [left, right, top, bottom])
        # x=self.pad(x)
        # x = F.pad(x, (self.left, self.right, self.top, self.bottom))
        x = self.pool(x)
        # h, w = x.shape[-2:]
        # print(h,w)
        # print('-------------------------------------------------')
        # input()
        return x
