
from torch import nn, Tensor
import torch 
from collections import OrderedDict
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from ..My_transforms.Transforms import T_Identity
from torchvision.ops import nms
from typing import Tuple,List,Dict
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.image_list import ImageList
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteHead
from copy import deepcopy
import numpy as np
from functools import partial
import gc
import torch.nn.functional as F

class MY_BoxCoder(det_utils.BoxCoder):
    def __init__(self):
        super(MY_BoxCoder,self).__init__(weights=(10.0, 10.0, 5.0, 5.0))
        self.half_const=torch.tensor(0.5)[None,None,None]
    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """


        boxes = boxes.to(rel_codes.dtype)
        widths = boxes[:,:, 2] - boxes[:,:, 0]
        heights = boxes[:,:, 3] - boxes[:,:, 1]
        ctr_x = boxes[:,:, 0] + self.half_const * widths
        ctr_y = boxes[:,:, 1] + self.half_const * heights

        wx, wy, ww, wh = self.weights

        
        rel_coxes_0,rel_coxes_1,rel_coxes_2,rel_coxes_3=torch.split(rel_codes,1,dim=-1)

        dx =  torch.squeeze(torch.div(rel_coxes_0,torch.tensor(wx)))
        dy =  torch.squeeze(torch.div(rel_coxes_1, torch.tensor(wy)))
        dw =  torch.squeeze(torch.div(rel_coxes_2, torch.tensor(ww)))
        dh = torch.squeeze(torch.div(rel_coxes_3 , torch.tensor(wh)))

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw,min=-1000, max=self.bbox_xform_clip)
        dh = torch.clamp(dh,min=-1000, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # Distance from center to box's corner.
        
        c_to_c_h = torch.tensor(self.half_const, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(self.half_const, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = torch.unsqueeze(pred_ctr_x - c_to_c_w,dim=-1)
        pred_boxes2 = torch.unsqueeze(pred_ctr_y - c_to_c_h,dim=-1)
        pred_boxes3 = torch.unsqueeze(pred_ctr_x + c_to_c_w,dim=-1)
        pred_boxes4 = torch.unsqueeze(pred_ctr_y + c_to_c_h,dim=-1)

        
        #input()
        pred_boxes = torch.cat((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
        return pred_boxes#pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4

class Mobilnet_v3_SSDLite_wrapper(nn.Module):
    '''
    inputs:
    images:tensor of images
    targets:tensor of ground truth targets[BATCH SIZE,NUM OF BBOXES == 50 PADDED,0] labels ,targets[BATCH SIZE,NUM OF BBOXES == 50 PADDED,1:5] BBoxes
    outputs:List[Dict
    - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
    - labels (Int64Tensor[N]): the predicted labels for each detection
    - scores (Tensor[N]): the scores for each detection
    '''
    
    def out_to_detectron(self,outputs:np.ndarray,image_sizes:Tuple[int,int]):
        from detectron2.structures.boxes import Boxes
        from detectron2.structures.instances import Instances
        bboxes = outputs[:, 0:4]
        
              
        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]
        new_output={}
        nn_results={'pred_boxes':Boxes(bboxes),'scores':scores,'pred_classes':cls.astype(int)}

        new_output['instances']=Instances(image_size=image_sizes[::-1],**nn_results)
        return new_output
    
    def post(self,prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        """
        num_classes,nms_thre,class_agnostic unused retained for compatibility with YOLOX 
        """
        output=[]
        
        for i in prediction:
            selected=nms(i['boxes'],i['scores'],0.5)
            i['boxes'],i['scores'],i['labels']=i['boxes'][selected],i['scores'][selected],i['labels'][selected]
            bool_kept_prediction=i['scores']>=conf_thre
            
            num_kep_prediction=torch.sum(bool_kept_prediction)
            if(num_kep_prediction==0):
                output.append(None)
                continue
            single_image_output=torch.ones((num_kep_prediction,7),dtype=float)
            single_image_output[:,:4]=i['boxes'][bool_kept_prediction]
            single_image_output[:,6]=i['labels'][bool_kept_prediction]
            single_image_output[:,4]=i['scores'][bool_kept_prediction]
            output.append(single_image_output)
        
        return output
    
    def from_nntool_conv(self,outputs,conf_thre,nm_thre):
        """ this function assumes that the batch size is one as the onnx exported always exports with a size of 1"""
        boxes=torch.from_numpy(outputs[0]).squeeze()
        scores=torch.from_numpy(outputs[1]).squeeze()
        image_boxes = []
        image_scores = []
        image_labels = []
        detections = []
        for label in range(1,self.number_classes):
            score = scores[:, label]
            keep_idxs = score > conf_thre
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(score, self.model.detections_per_img, 0)
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64))
        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nm_thre)
        keep = keep[: self.model.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )
        return detections

        
    

    def __init__(self,number_classes:int,pretrained_start=False,usepretrained=False,size=(320,320)):
        super(Mobilnet_v3_SSDLite_wrapper,self).__init__()
        self.number_classes=number_classes
        self.model=ssdlite320_mobilenet_v3_large(num_classes=self.number_classes)
        self.box_coder=MY_BoxCoder()
        if(pretrained_start):
            temp_model=ssdlite320_mobilenet_v3_large(pretrained=pretrained_start)
            
            
            if(usepretrained):
                self.model=temp_model
            else:
                out_channels = det_utils.retrieve_out_channels(temp_model.backbone, size)
                norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
                num_anchors = temp_model.anchor_generator.num_anchors_per_location()

                head=SSDLiteHead(out_channels, num_anchors, number_classes, norm_layer)   
                self.model.backbone=temp_model.backbone
                self.model.head=head
                
        self.model.transform=T_Identity()

    def postprocess_onnx_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],image_shape: Tuple[int, int]=(320,240)  ) -> List[Dict[str, Tensor]]:
        
        boxes = head_outputs["bbox_regression"]
        #head_outputs["cls_logits"],indices=torch.max(head_outputs["cls_logits"],dim=-1)
        head_outputs["cls_logits"]=head_outputs["cls_logits"]
        scores = F.softmax(head_outputs["cls_logits"], dim=-1)
        print('boxes_shape',boxes.shape)
        num_classes = self.number_classes
        device = scores.device

        detections: List[Dict[str, Tensor]] = []

        anchors=torch.unsqueeze(image_anchors[0],dim=0)
        print(anchors.shape)
        #image_shape=image_shapes[0]
        #input()
        boxes = self.box_coder.decode_single(boxes, anchors)
        #print(boxes[0].shape)
        #boxes =torch.cat(boxes,dim=-1)
        return boxes,scores
    
    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.sigmoid(head_outputs["cls_logits"])

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections
        
    def forward(self,images,targets=None):

        if targets is not None:
            targets=[{'labels':i[i.sum(dim=1)!=0][...,0].long(),'boxes':i[i.sum(dim=1)!=0][...,1:]} for i in targets]

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.model.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(ImageList(images,[i.shape[-2:]for i in images]), features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.model.proposal_matcher(match_quality_matrix))

                losses = self.model.compute_loss(targets, head_outputs, anchors, matched_idxs)
                losses['total_loss']=losses['classification']+4*losses['bbox_regression']
        else:
            if(torch.onnx.is_in_onnx_export()):
                
                detections= self.postprocess_onnx_detections(head_outputs, anchors, [i.shape[-2:]for i in images])
                return  detections
            detections = self.model.postprocess_detections(head_outputs, anchors, [i.shape[-2:]for i in images])

        
        return self.model.eager_outputs(losses, detections)

    
    














