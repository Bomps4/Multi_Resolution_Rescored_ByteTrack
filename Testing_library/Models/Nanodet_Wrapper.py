
import copy

import torch
from torch import nn
from torchvision.ops import nms
import numpy as np
from typing import Tuple
from .nanodet.nanodet.model.arch import build_model
from typing import Mapping,Any
from loguru import logger
import torch.nn.functional as F
from .NN_Wrapper import NetworkWrapper
from ..utils.general_functions import format_outputs,map_tensor_values
import math
from typing import Union,List,Tuple,Dict
from ..utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_xywh,box_cxcywh_to_xywh,box_xywh_to_xyxy,distance2bbox
import torchvision as thcv

def load_model_weight(model, checkpoint):
    state_dict = checkpoint["state_dict"].copy()
    for k in checkpoint["state_dict"]:
        # convert average model weights
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            logger.info("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            logger.info("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model

def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels
    
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep],scores[keep]




def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    if len(boxes_for_nms) < split_thr:
        keep = nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep

class NanoDetPlusWrapper(NetworkWrapper):
    """
    Nanodet Plus class produces detection as a single tensor with subdivision:


    Tensor[batch_size,n_detection,0:4](bounding boxes)
    Tensor[batch_size,n_detection,4](score)
    Tensor[batch_size,n_detection,5](label)
    
    """


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
        
        
        h,w = targets['height'],targets['width']
        # print('images ids ',targets['images_id'])
        if (isinstance(output['pred_boxes'],torch.Tensor)):
            
            if isinstance(self.experiment.test_size,tuple):
                (h,w) = self.experiment.test_size
            elif isinstance(self.experiment.test_size,list) and len(self.test_size)==1:
                (h,w) = self.experiment.test_size[0][1]
            

            nms_result = self.get_bboxes (output['pred_scores'], output['pred_boxes'], (h,w) ,self.experiment.test_conf) 
            
        

            bboxes = [i[0][:,:4] for i in nms_result]
            scores = [i[0][:,4] for i in nms_result]
            labels = [i[1] for i in nms_result]

            
            if to_coco:
                

                bboxes = [i/targets['resize_factor'][idx] for idx,i in enumerate(bboxes)]
                bboxes = [torch.clip(box_xyxy_to_xywh(i),min=0) for i in bboxes] #coco uses xywh format

            output['pred_boxes'] = bboxes
                
                
                
            # output['pred_boxes'] *= (torch.cat([w,h,w,h],dim=-1))[:,None]#Batch dimension number of detection last is size (cx,cy,w,h) 
            max_type = torch.max(output['pred_scores'],dim=-1) #xyxy format 417,   7, 606, 172  xcycwh format 511.5000,  89.5000, 189.0000, 165.0000  [303.25, 39.75, 350.5, 229.75]
            output['pred_classes'] = labels
            # print(output['pred_classes'].shape)
            # print(output['pred_classes'][0])
            # input()
            output['pred_scores'] = scores
            

            # background_mask = output['pred_classes']!=0 #this model considers the 0 class as background 

            # output['pred_classes'] = output['pred_classes'][background_mask]
            # output['pred_scores'] = output['pred_scores'][background_mask]
            # output['pred_boxes'] = output['pred_boxes'][background_mask]

            

        elif (isinstance(output['pred_boxes'],list)):
            keys=targets.keys()
            output['pred_classes']=[]
            for box_idx,box in enumerate(output['pred_boxes']):
                

            
                nms_result = self.get_bboxes (output['pred_scores'][box_idx], output['pred_boxes'][box_idx], keys[box_idx] ,self.experiment.test_conf) 
            
        

                bboxes = [i[0][:,:4] for i in nms_result]
                scores = [i[0][:,4] for i in nms_result]
                labels = [i[1] for i in nms_result]

                
                if to_coco:
                    

                    bboxes = [i/(targets[keys[box_idx]]['resize_factor'][idx]) for idx,i in enumerate(nms_result)]
                    bboxes = [torch.clip(box_xyxy_to_xywh(i),min=0) for i in bboxes] #coco uses xywh format
                    # box[...,2:]-=box[...,:2]
                
                
                output['pred_boxes'][box_idx]=bboxes
                output['pred_scores'][box_idx] = scores
                output['pred_classes'].append(labels)

        return output
    
    def filter_output(self,output):
        """
        This function assumes to work on non batched data to filter the output the inputs are tensors 
        """
        experiment = self.experiment

        bboxes_image,score_image,classes_image=output

        print(experiment.selected_classes)

        masked_by_class = torch.isin(classes_image,experiment.selected_classes)

        bboxes_image = bboxes_image[masked_by_class]
        score_image = score_image[masked_by_class]
        classes_image = classes_image[masked_by_class]

        

        # indexes=thcv.ops.nms(box_xywh_to_xyxy(bboxes_image),score_image,experiment.nmsthre)

        

        # bboxes_image = bboxes_image[indexes]
        # score_image = score_image[indexes]
        # classes_image = classes_image[indexes]


        # print('bboxes after nms ', bboxes_image,'classes_image')
        

        # print('after nms boxes ',bboxes_image)

        # print('after nms classes ',classes_image)

        # input()

        classes_image = map_tensor_values(classes_image,experiment.from_coco_to_imgvid)

        score_mask=score_image>experiment.test_conf

        bboxes_image = bboxes_image[score_mask]
        score_image = score_image[score_mask]
        classes_image = classes_image[score_mask]

        

        #functions expects to be called after the format output

        

        
       

        return bboxes_image.tolist(),score_image.tolist(),classes_image.tolist()
    



    
    def out_to_detectron(self,outputs:np.ndarray,image_sizes:Tuple[int,int]):
        from detectron2.structures.boxes import Boxes
        from detectron2.structures.instances import Instances
        bboxes = outputs[:, 0:4]
              
        cls = outputs[:, 5]
        scores = outputs[:, 4] 
        new_output={}
        nn_results={'pred_boxes':Boxes(bboxes),'scores':scores,'pred_classes':cls.astype(int)}

        new_output['instances']=Instances(image_size=image_sizes[::-1],**nn_results)
        return new_output
    

    def get_bboxes(self, cls_preds, reg_preds, input_shape,score_threshold):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = input_shape
        input_shape = (input_height, input_width)
        

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.model.head.strides
        ]

        


        # get grid cells of one image
        mlvl_center_priors = [
            self.model.head.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.model.head.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        
        dis_preds = self.model.head.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=score_threshold,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    
    
    
    def from_nntool_conv(self,outputs,conf_thre,nm_thre):
        return torch.from_numpy(outputs[0])[None]

        
    def __init__(
        self,multisize, experiments, cfg
    ):
        super(NanoDetPlusWrapper, self).__init__(multisize, experiments)
        
        self.model=build_model(cfg)
       
    def execute_network(self,x):
        if hasattr(self.model, "fpn"):
                x = self.model.fpn(x)
        if hasattr(self.model, "head"):
            x = self.model.head(x)

        cls_scores, bbox_preds = x.split(
            [self.model.head.num_classes, 4 * (self.model.head.reg_max + 1)], dim=-1
        )
        

        return cls_scores, bbox_preds 
       

    def forward(self, x,target=None):
        # self.img=torch.empty(x.shape)
        if(self.model.training):
            if target is None:
                raise Exception("targets cannot be None in training mode")
            gt_meta={'bboxes':target[...,1:],'labels':target[...,0],'img':x,"gt_bboxes_ignore":None}
            self.forward_train(gt_meta)
        
        # if (isinstance(x,list)):
        #     x=x[0]
        #     x=x.unsqueeze(0)
        #     print('sono la shape',x.shape)
        #     input()
       

        
        cls_scores = []
        bbox_preds = []

        x = self.model.backbone(x)
        if not self.multiresolution:
            
            cls_scores, bbox_preds = self.execute_network(x)
        

        else:
            for key in x:
                cls_scores_block,bbox_preds_block = self.execute_network(x[key])

                bbox_preds.append(bbox_preds_block)
                cls_scores.append(cls_scores_block)
    


        if not torch.onnx.is_in_onnx_export():
            return format_outputs(bbox_preds, cls_scores)

        else:
            if self.multiresolution:
                raise Exception('Cannot use multiresolution while exporting')

            import math
            from .nanodet.nanodet.util.box_transform import distance2bbox
            # cls_preds, = x.split(
            #     [80, 4 * (self.model.head.reg_max + 1)], dim=-1
            # )
            
            
            cls_preds = x[:,:,:80]
            
            scores = cls_preds.sigmoid()
            
            reg_preds = x[:,:,80:]
            print('reg_preds ',reg_preds.shape)
            input_shape = (input_height, input_width)

            featmap_sizes = [
                (math.ceil(input_height / stride), math.ceil(input_width / stride))
                for stride in self.model.head.strides
            ]
            
            # get grid cells of one image
            mlvl_center_priors = [
                self.model.head.get_single_level_center_priors(
                    1,
                    featmap_sizes[i],
                    stride,
                    dtype=torch.float32,
                    device=torch.device('cpu'),
                )
                for i, stride in enumerate(self.model.head.strides)
            ]
            center_priors = torch.cat(mlvl_center_priors, dim=1)
            print('sono centerreg_preds priors', center_priors.shape)

            a=torch.linspace(0, self.model.head.reg_max, self.model.head.reg_max + 1)
            shape=reg_preds.shape
            print('region prediction shape ',shape)
            reg_preds=reg_preds.reshape(*shape[:-1], 4, self.model.head.reg_max + 1)
            reg_preds=F.softmax(torch.tensor(reg_preds),dim=-1)
            
            
            print('a shape ',a.shape)
            reg_preds = reg_preds* a.type_as(reg_preds)[None,None,None]
            reg_preds = torch.sum(reg_preds,dim=-1)
            dis_preds = reg_preds* center_priors[..., 2, None]
            
            bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)

            # out_scores,out_labels=torch.max(scores,dim=-1)
            # out_scores,out_labels = out_scores[...,None],out_labels[...,None]

            x=torch.cat([bboxes,scores],dim=-1)
            
            x=x.squeeze()
            
            return x
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        try:
            self.model.load_state_dict(state_dict, strict)
            return self
        except RuntimeError:
            self.model=load_model_weight(self.model,state_dict)
            return self

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.model.backbone(img)
        fpn_feat = self.model.fpn(feat)
        if self.model.epoch >= self.model.detach_epoch:
            aux_fpn_feat = self.model.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = (
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.model.aux_fpn(feat)
            dual_fpn_feat = (
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        head_out = self.model.head(fpn_feat)
        aux_head_out = self.model.aux_head(dual_fpn_feat)
        loss = self.model.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        loss={'total_loss':loss[0],**loss[1]}
        return loss
    

