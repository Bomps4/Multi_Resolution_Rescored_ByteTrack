#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import torch
from typing import Tuple,Union,Dict
from .Heads.yolo_head import YOLOXHead
from .Heads.yolo_pafpn import YOLOPAFPN
from ..utils.postprocess import postprocess
from .NN_Wrapper import NetworkWrapper
from ..utils.box_ops import box_xyxy_to_xywh,box_xywh_to_xyxy,box_xyxy_to_cxcywh,box_cxcywh_to_xywh,box_cxcywh_to_xyxy
import torchvision as thcv
from ..utils.general_functions import map_tensor_values,format_outputs
import json

class YOLOX(NetworkWrapper):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    
    # def out_to_detectron(self,outputs:torch.Tensor,image_sizes:Tuple[int,int]):
    #     from detectron2.structures.boxes import Boxes
    #     from detectron2.structures.instances import Instances
    #     bboxes = outputs[:, 0:4]
    #     cls = outputs[:, 6]
    #     scores = outputs[:, 4] * outputs[:, 5]
    #     new_output={}
    #     nn_results={'pred_boxes':Boxes(bboxes),'scores':scores,'pred_classes':cls}
    #     new_output['instances']=Instances(image_size=image_sizes,**nn_results)

    #     return new_output
    
    # def post(self,prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    #     return postprocess(prediction, num_classes, conf_thre, nms_thre, class_agnostic,xyxy=self.xyxy)

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

    
    def from_nntool_conv(self,outputs,conf_thre,nm_thre):
        correct_out=torch.from_numpy(outputs[0])[None]
        return self.head.decode_outputs(correct_out,torch.float32)
    


    def format_output(self,output:Union[Dict[str,torch.Tensor],Dict[str,list]],targets:Dict[str,torch.Tensor],to_coco:bool=False):
        torch.set_printoptions(threshold=10000000)
        
        
        h,w = targets['height'],targets['width']
        # print('images ids ',targets['images_id'])
        if (isinstance(output['pred_boxes'],torch.Tensor)):
            
        
            h,w = targets['height'],targets['width']

            
            

            postprocessed = postprocess (torch.cat((output['pred_boxes'],output['pred_scores']),dim=-1) ,self.experiment.num_classes, conf_thre=self.experiment.test_conf, nms_thre=self.experiment.nmsthre, class_agnostic=False,xyxy=self.experiment.xyxy) 
            
            # print(postprocessed[0])
            # input()

            bboxes = [post_proc[:, 0:4] if post_proc is not None else None for post_proc in postprocessed]
            cls = [post_proc[:, 6] if post_proc is not None else []  for post_proc in postprocessed]
            scores = [post_proc[:, 4] * post_proc[:, 5] if post_proc is not None else []   for post_proc in postprocessed] 

            # print(scores[0])
            # input()

            
            if to_coco:
                bboxes = [i/targets['resize_factor'][idx] if i is not None else None for idx,i in enumerate(bboxes)]

                
                bboxes = [torch.clip(box_xyxy_to_xywh(i),min=0) if i is not None else [] for i in bboxes] #coco uses xywh format
                
            output['pred_boxes'] = bboxes    
            output['pred_classes'] = cls

            
            # print(output['pred_classes'].shape)
            # print(output['pred_classes'][0])
            # input()
            output['pred_scores'] = scores

            # print(output['pred_scores'][0])
            # input()

            # background_mask = output['pred_classes']!=0 #this model considers the 0 class as background 

            # output['pred_classes'] = output['pred_classes'][background_mask]
            # output['pred_scores'] = output['pred_scores'][background_mask]
            # output['pred_boxes'] = output['pred_boxes'][background_mask]

            

        elif (isinstance(output['pred_boxes'],list)):
            keys=targets.keys()
            output['pred_classes']=[]
            for box_idx,box in enumerate(output['pred_boxes']):
                targets_current = targets[keys[box_idx]]            
                postprocessed = postprocess (torch.cat((output['pred_boxes'][box_idx],output['pred_scores'][box_idx]),dim=-1) ,self.experiment.num_classes, conf_thre=self.experiment.test_conf, nms_thre=self.experiment.nmsthre, class_agnostic=True,xyxy=self.xyxy)        
        

                bboxes = [post_proc[:, 0:4] if post_proc is not None else None for post_proc in postprocessed]
                cls = [post_proc[:, 6] if post_proc is not None else None  for post_proc in postprocessed]
                scores = [post_proc[:, 4] * post_proc[:, 5] if post_proc is not None else None   for post_proc in postprocessed] 

                
                if to_coco:
                    bboxes = [i/targets_current['resize_factor'][idx] if i is not None else None for idx,i in enumerate(bboxes)]
                    
                    bboxes = [torch.clip(box_xyxy_to_xywh(i),min=0) if i is not None else [] for i in bboxes] #coco uses xywh format
                    # box[...,2:]-=box[...,:2]
                
                
                output['pred_boxes'][box_idx]=bboxes
                output['pred_scores'][box_idx] = scores
                output['pred_classes'].append(cls)

        return output
    
    def filter_output(self,output:Tuple):
        """
        This function assumes to work on non batched data to filter the output the inputs are a tuple of tensors 
        """
        experiment = self.experiment

        bboxes_image,score_image,classes_image=output

        if (len(bboxes_image)==0):
            return [],[],[]
        
        # print(classes_image[classes_image>=30])

        # print(classes_image)
        # input()

        masked_by_class =  torch.isin(classes_image,experiment.selected_classes)

        bboxes_image = bboxes_image[masked_by_class]
        score_image = score_image[masked_by_class]
        classes_image = classes_image[masked_by_class]

        # print(bboxes_image)
        # print(score_image)
        # print(classes_image)
        # input()

        #indexes=thcv.ops.nms(box_xywh_to_xyxy(bboxes_image),score_image,experiment.nmsthre)

        

        # bboxes_image = bboxes_image[indexes]
        # score_image = score_image[indexes]
        # classes_image = classes_image[indexes]

        

        # print(experiment.from_coco_to_imgvid)
        # input()
        
        classes_image = map_tensor_values(classes_image.long(),experiment.from_coco_to_imgvid)   
        # print(classes_image)
        # input()




        return bboxes_image.tolist(),score_image.tolist(),classes_image.tolist()





    def __init__(self, multiresolution, experiment,backbone=None, head=None,xyxy=False):
        super().__init__(multiresolution,experiment)

        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        self.head.decode_in_inference=not torch.onnx.is_in_onnx_export()

        fpn_outs = self.backbone(x)

        # for name, param in self.backbone.named_parameters():
        #     print(f"{name}:\n{param.data}\n")
        
        # input()

        # for name, param in self.head.named_parameters():
        #     print(f"{name}:\n{param.data}\n")

        # input()

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = ({
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            },loss)
        else:
            outputs = self.head(fpn_outs)
            if not torch.onnx.is_in_onnx_export():
                outputs = format_outputs(outputs[...,:4],outputs[...,4:])
            else:
                print(outputs[...,4].shape)
                print(outputs[...,5:].shape)
                print(outputs[...,4][...,None]*outputs[...,5:])
                outputs = torch.concat((outputs[...,:4],outputs[...,4][...,None]*outputs[...,5:]),dim=-1)

        

        return outputs
