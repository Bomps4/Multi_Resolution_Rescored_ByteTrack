import torch 
from torch import nn
import torch.nn.functional as F
import torchvision as thcv
from timm.models import create_model
from typing import List,Callable,Union,Tuple,Dict
from .NN_Wrapper import NetworkWrapper
from .Losses.Criterion import Criterion
from ..utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy,box_cxcywh_to_xywh,box_xywh_to_xyxy
from loguru import logger
from ..utils.Box_encoder import BoxCoder
from .Heads.SSDLite_head import SSDLiteHead

from ..utils.general_functions import format_outputs,sum_multiple_nested_dicts,sum_nested_dict_with_exclusions

if __name__ == '__main__':
    import sys
    import os

    SCRIPT_DIR = os.path.dirname('/home/bomps/Scrivania/workfile/CNN_TRAINING_REFACTORING/new_version/NN_Train_test/Models')
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from .Layers.Convolutional_Base import DWConv

from torchvision.models.detection.ssd import SSDScoringHead
from ..utils.Defaul_box_generator import MyDefaultBoxGenerator as DefaultBoxGenerator

from .Heads.Anchor import AnchorGenerator
from .Layers.Convolutional_Base import BaseConv

import math

class EfficientViTSSDLite(NetworkWrapper):

    def format_input (self,_input:Union[torch.Tensor,Dict],targets:Union[torch.Tensor,Dict]):
        
        if(isinstance(_input,dict)):
            for target_key in targets:
                target=targets[target_key]
                h,w=target['height'],target['width']
                targets[target_key]['boxes']=box_xyxy_to_cxcywh(target['boxes']) #Ugly to do all this format changes but the loss as i've taken it uses this format and so i will use it 
                targets[target_key]['boxes']/=torch.stack([w,h,w,h],dim=-1)
        else:
            h,w=targets['height'],targets['width']

            targets['boxes']=box_xyxy_to_cxcywh(targets["boxes"])

            targets['boxes']/=(torch.stack([w,h,w,h],dim=-1))


        return _input,targets


    def format_output(self,output,targets,to_coco=False):
        torch.set_printoptions(threshold=10000000)

        # print('images ids ',targets['images_id'])

        
        

        if (isinstance(output['pred_boxes'],torch.Tensor)):

            
            
            h,w=targets['height'],targets['width']

            print(output['pred_boxes'])
            
            if to_coco:
                output['pred_boxes'] = box_cxcywh_to_xywh(output['pred_boxes']) #coco uses xywh format
                output['pred_boxes'] = torch.clip(output['pred_boxes'],min=0,max=1)
                
                
            output['pred_boxes'] *= (torch.cat([w,h,w,h],dim=-1))[:,None]#Batch dimension number of detection last is size (cx,cy,w,h) 
            max_type = torch.max(output['pred_scores'],dim=-1) #xyxy format 417,   7, 606, 172  xcycwh format 511.5000,  89.5000, 189.0000, 165.0000  [303.25, 39.75, 350.5, 229.75]
            output['pred_classes'] = max_type[1]

            
            # print(output['pred_classes'].shape)
            # print(output['pred_classes'][0])
            # input()
            output['pred_scores'] = max_type[0]

            # background_mask = output['pred_classes']!=0 #this model considers the 0 class as background 

            # output['pred_classes'] = output['pred_classes'][background_mask]
            # output['pred_scores'] = output['pred_scores'][background_mask]
            # output['pred_boxes'] = output['pred_boxes'][background_mask]

            

        elif (isinstance(output['pred_boxes'],list)):
            keys=targets.keys()
            output['pred_classes']=[]
            for box_idx,box in enumerate(output['pred_boxes']):
                target=targets[keys[box_idx]]
                box = torch.clip(box,min=0,max=1)



                h,w=target['height'],target['width']

                box *= (torch.cat([w,h,w,h],dim=-1))[:,None]
                if to_coco:
                    box = box_cxcywh_to_xywh(box)#coco uses xywh format
                    # box[...,2:]-=box[...,:2]
                
                max_type = [i for i in torch.max(output['pred_scores'][box_idx],dim=-1)]
                background_mask = max_type[1]!=0 #this model considers the 0 class as background 
                max_type[0]=max_type[0][background_mask]
                box = box[background_mask]
                max_type[1]=max_type[1][background_mask]
                output['pred_boxes'][box_idx]=box
                output['pred_scores'][box_idx] = max_type[0]
                output['pred_classes'].append(max_type[1])

        return output
    
    def filter_output(self,output):
        """
        This function assumes to work on non batched data to filter the output the inputs are tensors 
        """
        experiment = self.experiment
        bboxes_image,score_image,classes_image=output

        masked_by_class = classes_image!=0

        bboxes_image = bboxes_image[masked_by_class]
        score_image = score_image[masked_by_class]
        classes_image = classes_image[masked_by_class]

        indexes=thcv.ops.nms(box_xywh_to_xyxy(bboxes_image),score_image,experiment.nmsthre)

        bboxes_image = bboxes_image[indexes]
        score_image = score_image[indexes].softmax(dim=-1)

        classes_image = classes_image[indexes]


        # print('bboxes after nms ', bboxes_image,'classes_image')
        

        # print('after nms boxes ',bboxes_image)

        # print('after nms classes ',classes_image)

        # input()


        score_mask=score_image>experiment.test_conf

        bboxes_image = bboxes_image[score_mask]
        score_image = score_image[score_mask]
        classes_image = classes_image[score_mask]

        

         #functions expects to be called after the format output

        

        
       

        return bboxes_image.tolist(),score_image.tolist(),classes_image.tolist()
    
            

    def __init__(self,backbone_name:str,pretrained:bool,criterion:Criterion,num_classes:int,multiresolution:bool,experiment,onnx_export:bool=False,input_size:Union[int,Tuple,None]=None,drop_rate:float=0.):
        """ok
        backbone_name : name of the backbone as provided by timm
        pretrained: bool if to load also the pretrained version of the module in timm 
        criterion : loss class must be a module or a subclass of Criterion
        num_classes: number of classes 
        input_size: None to use the default defined in the timm model 
        drop_rate: TODO check the value in Timm
        drop_path_rate: dropout rate 
        """
        super().__init__(multiresolution,experiment)
        self.backbone=create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_block_rate=None,
            img_size=input_size)
        
        
        # print(str(self.backbone.feature_info.get_dicts()))
        # print(str(self.backbone.feature_info.channels()))
        # input()
        
        

        self.num_features=self.backbone.feature_info.channels()[-3:]


        channels = [512,512,512]

        self.classification_expansion = nn.ModuleList([BaseConv(i,channel,ksize=1,stride=1,act='relu6',norm=True,bias=True) for channel,i in zip(channels,self.num_features)])
        self.regression_expansion = nn.ModuleList([BaseConv(i,channel,ksize=1,stride=1,act='relu6',norm=True,bias=True) for channel,i in zip(channels,self.num_features)])

        self.criterion=criterion

        self.anchor_generator = AnchorGenerator(((8,),(16,),(32,),(64,)), ((1,),(1,),(1,),(1,)))

        anchors=self.anchor_generator.num_anchors_per_location()

        self.head = SSDLiteHead([512 for _ in anchors],anchors,num_classes+1,criterion)

        self.num_anchors_layers=len(anchors)

        self.pooling=nn.AvgPool2d(kernel_size=(2,2),stride=2)

        self.near_upscale = nn.Upsample(scale_factor=2, mode='nearest')

        self.bilinear_upscale = nn.Upsample(scale_factor=2, mode='bilinear')

        self.max_pooling=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        
    
    
    def foward_tensor_block(self,x,targets):
        feature_tensor=self.backbone(
            x,
            # indices=self.backbone.out_indices,
            # norm=self.backbone.norm,
            # output_fmt=self.backbone.output_fmt,
            # intermediates_only=True,
            # return_prefix_tokens=True
        )

        feature_tensor = feature_tensor[-3:]

        classification_features = [module(result) for result,module in zip(feature_tensor,self.classification_expansion)]

        regression_features = [module(result) for result,module in zip(feature_tensor,self.regression_expansion)]

        # for idx,(class_feat,reg_feat) in enumerate(zip(classification_features[:-1:-1],regression_features[:-1:-1])):

        #     classification_features[-idx-2]+=self.max_pooling(classification_features[idx])

        #     regression_features[-idx-2]+=self.pooling(regression_features[idx])

        
        
        for idx,(class_feat,reg_feat) in enumerate(zip(classification_features[::-1][:-1],regression_features[::-1][:-1])):

            classification_features[-idx-2] += self.near_upscale(classification_features[-1-idx])

            regression_features[-idx-2] += self.bilinear_upscale(regression_features[-1-idx])
                
        

        classification_features.append(self.max_pooling(classification_features[-1]))

        regression_features.append(self.pooling(regression_features[-1]))

            
        anchors = box_xyxy_to_cxcywh(self.anchor_generator(x,classification_features,targets))

        
        features = {'regression_features':regression_features,'classification_features':classification_features}
        
                
                
        return anchors,features,targets

    def forward(self,x,targets=None):

        #if we are in multiresolution we divide it in blocks based on their size else we take all in a single blcok
        total_outputs=format_outputs([],[])
        all_losses = None 
        if self.multiresolution and isinstance(x,dict):
            keys=list(x.keys())
            
            all_losses=None
            for key in keys:
                anchors,features,targets = self.foward_tensor_block(x[key],targets[key])
                output = self.head(anchors,features,targets)
                if not self.training:
                    for out_key in total_outputs:
                        total_outputs[out_key].extend(output[out_key])
                    return total_outputs
                else:
                    if(all_losses is None):
                        all_losses=output
                    else:
                        all_losses=sum_multiple_nested_dicts(all_losses,output) #summing dictionaries across blocks (have the same keys so it should not be a problem)
        else:
            anchors,features,targets = self.foward_tensor_block(x,targets)
            current_output = self.head(anchors,features,targets)
            if not self.training:
                total_outputs = current_output

                
                if torch.onnx.is_in_onnx_export(): #avoiding dictionary output in onnx export
                    return torch.cat(total_outputs['pred_boxes'],total_outputs['pred_scores'].softmax(dim = -1),dim=-1)

                return  total_outputs
            else:
                all_losses = current_output

        loss=sum_nested_dict_with_exclusions(all_losses,set(['cardinality_error','class_error'])) #summation of losses except cardinarlity loss 
        
        # if torch.isnan(scores).any():
        #     import pdb; pdb.set_trace()

        # if torch.isnan(boxes).any():
        #     import pdb; pdb.set_trace()    
       
        

        return all_losses,loss 
        


