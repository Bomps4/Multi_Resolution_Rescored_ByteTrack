import torch 
from torch import Tensor
from typing import List,Dict
import torch.nn.functional as F
from .Criterion import Criterion
torch.set_printoptions(threshold=10000000)


class SSD_Criterion(Criterion):

    def __init__(self,box_encoder,eos_class_weight,matcher,neg_to_pos_ratio=3):
        
        super().__init__()
        self.matcher = matcher  
        self.box_encoder = box_encoder
        self.eos_class = eos_class_weight
        
        self.neg_to_pos_ratio = neg_to_pos_ratio

    
    def forward(self,
        targets: Dict[str, Tensor],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        
        # print('size of ancors',torch.stack(anchors,dim=0).shape)
        # input()

        # print('max : ',torch.max(anchors[0],dim=0))
        # print('min : ',torch.min(anchors[0],dim=0))

        matched_idxs = self.matcher({'pred_boxes':torch.stack(anchors,dim=0)},targets)

        bbox_regression = head_outputs["pred_boxes"]
        cls_logits = head_outputs["pred_scores"]

        gt_labels,gt_boxes = targets['labels'].get_org_tensors(),targets['boxes'].get_org_tensors()
        # print('sono len di matched idxs',len(matched_idxs))
        # print()
        # input()

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            gt_labels_per_image,
            gt_boxes_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(gt_labels,gt_boxes, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets

            # print(matched_idxs_per_image[1])

            # print('gt_boxes_per_image',gt_boxes_per_image.shape)
            # print('bbox_regression_per_image',bbox_regression_per_image.shape)
            # input()
            matched_idxs_per_image = matched_idxs_per_image[1]
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = gt_boxes_per_image[foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            # print(anchors_per_image)
            # print(matched_gt_boxes_per_image)
            # input()



            target_regression = self.box_encoder.encode_single(matched_gt_boxes_per_image, anchors_per_image)


            print('target_regression',target_regression)
            input()

            # print(target_regression)
            # input()quei due sul server
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            
            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=gt_labels_per_image.dtype,
                device=gt_labels_per_image.device,
            )

            gt_classes_target[foreground_idxs_per_image] = gt_labels_per_image[
                foreground_matched_idxs_per_image
            ]
           
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size()
        )

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }



