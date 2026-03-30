

from .Criterion import Criterion
import torch
import torch.nn.functional as F

from ...utils.env import is_dist_avail_and_initialized,get_world_size
from ...utils.general_functions import accuracy
from ...utils import box_ops


class Detection_One_to_Many_Criterion(Criterion):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef,neg_to_pos_ratio=3):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes #i don't include the background in the number of classes 
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef #eight applied to the 0 class (background)
        self.losses = ['labels','boxes']
        empty_weight = torch.ones(self.num_classes+1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def _get_src_permutation_idx(self, indices):
        
        # permute predictions following iself.matching_thrsndices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_scores' in outputs

        torch.set_printoptions(threshold=100000)

        src_logits = outputs['pred_scores']
        
        bs, n_bboxes , _ = src_logits.shape

        target_classes_o = targets['labels'].get_org_tensors()

        target_classes_o = torch.stack([target_classes_o_one_image[tgt] for target_classes_o_one_image,(_,tgt) in zip(target_classes_o,indices)],dim=0)


        not_objects_mask = torch.stack([ tgt <0 for (_,tgt) in indices],dim=0)

        objects_mask = torch.stack([ tgt >=0 for (_,tgt) in indices],dim=0)

        assert target_classes_o.size(0) == bs, "batching dimensions are not matching"

        target_classes_o[not_objects_mask]=0

        cls_loss = F.cross_entropy(src_logits.view(-1, self.num_classes+1), target_classes_o.view(-1), weight=self.empty_weight,reduction="none").view( bs, n_bboxes)

        num_positives_by_image = objects_mask.to(int).sum(1,keepdim=True)

        num_negative = self.neg_to_pos_ratio * num_positives_by_image.sum(1, keepdim=True)

        negative_loss = cls_loss.clone()
        negative_loss[objects_mask] = -float("inf")  # use -inf to detect positive values that creeped in the sample

        values, idx = negative_loss.sort(1, descending=True)

        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))

        background_idxs = idx.sort(1)[1] < num_negative

        # print(torch.sum(background_idxs.to(int)))

        # print(num_positives_by_image)

        # print(torch.sum(num_positives_by_image))

        # print(background_idxs.shape)

        # print('source logit of real objects ',torch.argmax(src_logits[objects_mask],dim=-1))

        # print('real classes ',target_classes_o[objects_mask])

        # print('negative part ',cls_loss[background_idxs].mean())

        # print('positive part',cls_loss[objects_mask].mean())

        # input()

        losses = {}

        losses ['loss_ce'] = (cls_loss[objects_mask].sum() + cls_loss[background_idxs].sum())/num_boxes

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[objects_mask], target_classes_o[objects_mask])[0]

        return  losses

    
    def loss_bboxes (self, outputs, targets, indices, num_boxes, log=True):

        assert 'pred_boxes' in outputs

        src_boxes = outputs['pred_boxes']

        

        target_boxes = targets['boxes'].get_org_tensors()

        objects_indexes = torch.stack([ tgt for (_,tgt) in indices],dim=0)


        #equality_among_objects = torch.stack([ tgt[tgt>=0].unsqueeze(0) == tgt[tgt>=0].unsqueeze(1) for (_,tgt) in indices],dim=0)

        #tgt are indexes with the same size as num_of_src_boxes with an index that goes from 0 to num ground truths (there is also -1 that indicates no match)

        # print(equality_among_objects)

        # print(equality_among_objects[0].shape)

        # input()

        objects_mask = objects_indexes >= 0

        # print('sono src_boxes matched ',src_boxes[0,objects_mask[0]])

        # print('sono target boxes ', target_boxes[0][0])

        # input()

        target_boxes_o = torch.stack([target_boxes_image[tgt] for target_boxes_image,(_,tgt) in zip(target_boxes,indices)])

        target_boxes_o = target_boxes_o[objects_mask]

        src_boxes = src_boxes[objects_mask]

        

        bbox_loss = F.l1_loss(src_boxes, target_boxes_o, reduction="none").sum(dim=-1)

        weights = torch.ones_like(bbox_loss,device=bbox_loss.device)    

        ends = torch.cumsum(torch.sum(objects_mask.to(int),dim=-1),dim=0)

        starts = ends -torch.sum(objects_mask.to(int),dim=-1)

        # for idx,(objects_indexes_by_batch,object_mask_single_image) in enumerate(zip(objects_indexes,objects_mask)):

        #     indexes,inverse_indices,counts = torch.unique(objects_indexes_by_batch[object_mask_single_image],return_counts=True,return_inverse=True)

        #     # print(indexes)
        #     # if (indexes.shape.numel()==0):
        #     #     print(objects_indexes_by_batch)
        #     # input()
        #     if (object_mask_single_image.to(int).sum()==0):
        #         continue

        #     total_counts = object_mask_single_image.to(int).sum()    

    
        #     #relative_bonus = counts.float() ** (-2)
        #     raw_weights = (1.0 / counts.float())#*relative_bonus

        #     normalization_factor = total_counts / torch.sum(raw_weights * counts.float())
        #     normalized_weights = raw_weights * normalization_factor

        #     normalized_weights = normalized_weights[inverse_indices]

        #     normalized_weights.max() 
            
        #     weights[starts[idx]:ends[idx]] = normalized_weights #torch.max(torch.ones(1,device=bbox_loss.device),(sum_count - counts_considered))/max([1,object_mask_single_image.to(int).sum()-1])   
            

        bbox_loss = (bbox_loss).sum()
        

        objects_indexes = objects_indexes[objects_mask]

        # bbox_distance = torch.cdist(src_boxes, target_boxes_o, p=1)

        iou_matrix = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes_o))
        

        loss_giou = 1 - ((torch.diag(iou_matrix)).sum() / num_boxes)


        return {'loss_bbox': (bbox_loss)/num_boxes,'loss_giou': loss_giou}

        
    
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_bboxes,
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        
    
    def forward(self,outputs,targets,anchors,num_anchors_feat):

        indices = self.matcher(anchors, targets,num_anchors_feat)

        num_boxes = sum([torch.sum((tgt>=0).to(int)) for (_,tgt) in indices])


        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)

        losses = {}
        for loss in self.losses:
            calculated = self.get_loss(loss, outputs, targets, indices, num_boxes)
            
            for k in calculated:
                if k in self.weight_dict:
                    calculated[k]*=self.weight_dict[k]

            losses.update(calculated)
        
        return losses



        


















        # ends = torch.cumsum(torch.sum(objects_mask.to(int),dim=-1),dim=0)
        
        # starts = ends -torch.sum(objects_mask.to(int),dim=-1)


        # for start,stop,eq in zip(starts,ends,equality_among_objects):
        
        #     print(iou_matrix[start:stop,start:stop])

        #     print(iou_matrix[start:stop,start:stop].shape)

        #     print(eq)

        #     print(eq.shape)

        #     input()

        # stop_start_mask = (ends-starts) > 1


        # starts = starts [stop_start_mask]

        # ends = ends[stop_start_mask]

        # num_of_differences = sum([((~eq).to(int)).sum() for eq in equality_among_objects])

        # num_of_differences = torch.where (num_of_differences>=1,num_of_differences,1)

        

        #         loss_difference = sum ([iou_matrix[start:stop,start:stop][~eq].sum()/num_of_differences for start,stop,eq in zip(starts,ends,equality_among_objects)])
        
        # repulsion_l1_loss = sum ([bbox_distance[start:stop,start:stop][~eq].sum()/num_of_differences for start,stop,eq in zip(starts,ends,equality_among_objects)])
        


        # # diag_view= torch.diagonal(iou_matrix)  

        # # diag_view[:] = 0 

        

        # # loss_difference = sum([sum([iou_matrix[start:stop,start:stop][~eq_mask]/torch.sum((~eq_mask).to(int)) for eq_mask in equality_among_objects if torch.sum((~eq_mask).to(int))!=0])/ num_boxes for start,stop in zip(starts,ends) ])

        
        
        # loss_giou += loss_difference

      
        
        
        # print(targets['boxes'].sizes)

        # for idx,(target_boxes_image,(_,tgt)) in enumerate(zip(target_boxes,indices)):
        #     for i in tgt[tgt!=-1]:
        #         print( box_ops.generalized_box_iou(
        #         box_ops.box_cxcywh_to_xyxy(src_boxes[tgt[tgt!=-1]]),
        #         box_ops.box_cxcywh_to_xyxy(target_boxes_image[tgt[tgt!=-1]]))[i])
        #         print(torch.diag(iou_matrix)[idx])
        #     print(tgt[tgt!=-1])
        #     print(target_boxes_image[tgt[tgt!=-1]])
        #     input()
        
        # print(target_boxes_o)

        # print(iou_matrix.shape)
        # print(src_boxes.shape)
        # print(target_boxes_o.shape)
        # input()




        