import torch 
from torch import nn
from .Backbones.efficientNet import EfficientNetLite
from .Heads.EfficientDet_Head import Head_Net
from .Heads.BiFPN import BiFpn
from typing import Tuple 
import numpy as np
from .Yet_Another_EfficientDet_Pytorch.backbone import EfficientDetBackbone
from torchvision.ops.boxes import batched_nms

def bboxes_regression(anchors,regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return torch.stack([xmin, ymin, xmax, ymax], dim=2)

class EfficientDet_Wrapper(nn.Module):
    
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
    
    def post(self,prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        """
        num_classes,nms_thre,class_agnostic unused retained for compatibility with YOLOX 
        """
        transformed_anchors,classification = prediction
        
        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > conf_thre)[:, :, 0]
        out = []
        for i in range(classification.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                out.append(None)
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=nms_thre)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]
                output=torch.ones((boxes_.shape[0],6),dtype=torch.float32)
                output[:,:4]=boxes_.cpu()
                output[:,5]=classes_.cpu()
                output[:,4]=scores_.cpu()
                out.append(output)
            else:
                out.append(None)

        return out

    def __init__(self,compound_coef, num_classes,
                             anchor_ratios, anchor_scales):
        super(EfficientDet_Wrapper, self).__init__()
        self.num_classes=num_classes
        self.model=EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes,
                             ratios=anchor_ratios, scales=anchor_scales)
    def forward(self,x):
        features, regression, classification, anchors = self.model(x)

       
        bboxes=bboxes_regression(anchors,regression)

        return (bboxes,classification)
    def load_state_dict(self,ckp):
        try:
            super().load_state_dict(ckp)
        except RuntimeError:
            self.model.load_state_dict(ckp)
            


        



