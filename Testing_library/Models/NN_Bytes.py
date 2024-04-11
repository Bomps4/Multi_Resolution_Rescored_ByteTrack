
from typing import Union,List,Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
from ..utils.postprocess import postprocess
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from torchvision.ops import nms
from ..My_transforms.functional import resize_bounding_box

class NN_Augmented(nn.Module):
    
    def out_to_detectron(self,outputs:Union[np.ndarray,torch.Tensor],image_sizes:Tuple[int,int]):
        # print(outputs)
        # input()
        outputs[...,4]=outputs[...,6]
        # print(f'sono outputs {outputs}')
        # input()
        to_return=self.model.out_to_detectron(outputs,image_sizes)
        # print(to_return)
        # input()
        return to_return
    
    def post(self,prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        """
        empty operation used for keeping consistency with other classes
        """
        
        return prediction

    def __init__(self,NN_model,PP_method,conf_threshold,nms_threshold,num_classes,classes_present,working_size=None,no_low=True,class_agnostic=False):
        super().__init__()
        self.model=NN_model
        self.working_size=working_size
        self.PP=PP_method
        self.conf_threshold=conf_threshold
        self.nmsthre=nms_threshold
        self.num_classes=num_classes
        self.classes_present=torch.tensor(classes_present)
        
        self.no_low=no_low
        self.class_agnostic=class_agnostic
        
    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict)
    
    def clear(self):
        self.PP.clear()
        return self
    
    def to(self,dest):
        self.model.to(dest)
        return self
    
    def eval(self):
        self.model.eval()
        return self

    def forward (self,inputs:Tensor)->List[Tensor]:
        
        if self.model.training:
           self.model.eval()
        if(isinstance(inputs,torch.Tensor)):
            size=inputs.shape[-2:]
        if (self.no_low or np.prod(self.working_size)==np.prod(size) ):
            # print(f'input shape {inputs.tensors.shape}')
            # input()   
            outputs=self.model(inputs)
            # print(f'this are the outputs {outputs}')
            # input()
            outputs=self.model.post(outputs,self.num_classes,self.conf_threshold,self.nmsthre,class_agnostic=self.class_agnostic)
            # print(f'this are the outputs after post {outputs}')
            # input()
        else:
            outputs=[None for i in range(inputs.shape[0])]
        
        

        out_list=[]

        for output_1_image in outputs:
            if(output_1_image is None):
                output_1_image=torch.empty((0,7))
            else:
                if(self.working_size is not None):
                    
                    output_1_image[...,:4]=resize_bounding_box(output_1_image[...,:4],size,self.working_size)
             
            if(output_1_image.shape[-1]==7):
                output_1_image=output_1_image.detach().cpu()
                selected=torch.isin(output_1_image[:,6],self.classes_present)
                output_1_image=output_1_image[selected]
                # nms_selected=nms(output_1_image[...,:4],(output_1_image[...,4]*output_1_image[...,5]),0.3)
                # output_1_image=output_1_image[nms_selected]
                output_1_image=output_1_image.numpy()
                bboxes_scores,classes=np.concatenate((output_1_image[...,:4],(output_1_image[...,4]*output_1_image[...,5])[:,None]),axis=-1),output_1_image[...,6]
                # print(f'sono boxes scores and classes {bboxes_scores} ,{classes}')
                # input()
            elif(output_1_image.shape[-1]==6):
                output_1_image=output_1_image.detach().cpu()
                selected=torch.isin(output_1_image[:,5],self.classes_present)
                output_1_image=output_1_image[selected]
                # nms_selected=nms(output_1_image[...,:4],output_1_image[...,4],0.3)
                # output_1_image=output_1_image[nms_selected]
                output_1_image=output_1_image.numpy()
                bboxes_scores,classes=np.concatenate((output_1_image[...,:4],(output_1_image[...,4])[:,None]),axis=-1),output_1_image[...,5]
                # print(f'sono boxes scores and classes {bboxes_scores} ,{classes}')
                # input()
            
            
            PP_out=self.PP.update(bboxes_scores,classes)
            # print(f'sono PP_out {PP_out}')
            # input()

            '''
            
            PP_out[:,:4]=output_1_image[...,:4]
            PP_out[:,5]=output_1_image[...,5]
            PP_out[:,6]=output_1_image[...,4]
            remember PP_out is an array which already has the correct size to be put inside the out_list need only to be converted to tensor
            '''
            
            
            
            out_list.append(torch.from_numpy(PP_out))

        
        return out_list
        
