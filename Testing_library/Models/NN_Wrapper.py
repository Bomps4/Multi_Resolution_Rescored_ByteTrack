from torch.nn import Module
from typing import Union,Iterable
import torch


class NetworkWrapper(Module):

    def __init__(self,multiresolution,experiment):
        super().__init__()
        self.multiresolution=multiresolution
        self.experiment = experiment

    def format_input (self,input:dict[str,torch.Tensor],target,*args,**kwargs):
        raise NotImplementedError('The input_formaltting method needs to be overwritten in derived classes')


    def format_output(self,output:dict[str,torch.Tensor],target,*args,**kwargs):
        raise NotImplementedError('The output_formaltting method needs to be overwritten in derived classes')
    
    def filter_output(self,output:Iterable[torch.Tensor],thresholds:Union[dict,None],*args,**kwargs):
        """
        This function will always assume to work on non batched data
        
        """
        raise NotImplementedError('The output_formaltting method needs to be overwritten in derived classes')