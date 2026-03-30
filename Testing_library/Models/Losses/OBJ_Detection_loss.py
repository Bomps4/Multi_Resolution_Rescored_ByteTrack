from torch import nn
from .Criterion import Criterion
import torch
import torch
import torch.nn.functional as F
from torch import nn
from ...utils import box_ops
from ...utils.env import is_dist_avail_and_initialized,get_world_size
from ...utils.general_functions import accuracy
from loguru import logger 


    # def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert 'pred_scores' in outputs
    #     src_logits = outputs['pred_scores']

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = targets['labels'].get_tensor() 
    #     target_classes = torch.full(src_logits.shape[:2], 0,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     target_classes[idx] = target_classes_o

    #     flat_logits=torch.flatten(src_logits,start_dim=0,end_dim=1)
    #     target_classes=torch.flatten(target_classes,start_dim=0,end_dim=1)


    #     loss_ce = F.cross_entropy(flat_logits, target_classes, self.empty_weight)

    #     losses = {'loss_ce': loss_ce}

    #     # import pdb; pdb.set_trace()
    #     if log:
    #         # TODO this should probably be a separate loss, not hacked in this one here
    #         losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     return losses

class Detection_Criterion(Criterion):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = ['labels','boxes','cardinality_error']
        empty_weight = torch.ones(self.num_classes+1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.to_print=0
        


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_scores' in outputs
        src_logits = outputs['pred_scores'].sigmoid()
        
        bs, n_bboxes , _ = src_logits.shape

        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = targets['labels'].get_org_tensors()

        target_classes_sizes = torch.as_tensor([len(indices) for _,index in indices],device=src_logits.device)
        # Initialize target classes with zeros in one step and set the required indices
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)

        target_classes_o = torch.cat([t[i] for t, (_, i) in zip(target_classes_o , indices)], dim=0).long()

        
        target_classes[idx] = target_classes_o
        

        
        # logger.info('this are the target labels ',target_classes_o)
        
        

        # target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.view(-1, src_logits.size(-1)), target_classes.view(-1), self.empty_weight,reduction='none')

        loss_ce = loss_ce.reshape(bs, n_bboxes)

        negative_indexes_for_loss=torch.ones(src_logits.shape[:2],dtype=bool)
        negative_indexes_for_loss[idx] = False

        positive_indexes_for_loss=torch.zeros(src_logits.shape[:2],dtype=bool)
        positive_indexes_for_loss[idx] = True


        target_classes_sizes = target_classes_sizes*3

        target_classes_sizes[target_classes_sizes==0] = 1

        total_negative_loss=0

        # probs = F.softmax(predictions, dim=-1)
        # pt = torch.where(targets == 1, probs, 1 - probs)
        # loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-6)

        for batch_number in range(bs):

            negatives_losses , _ = loss_ce[batch_number][negative_indexes_for_loss[batch_number]].sort(dim=0, descending=True)

            # if(torch.isnan(negatives_losses).any()):
            #     import pdb;pdb.trace()

            taken_losses=negatives_losses[:target_classes_sizes[batch_number]]

            total_negative_loss+=taken_losses.sum()


        
            
        negative_ce = total_negative_loss.sum()/(target_classes_sizes.sum())

        # if(torch.isnan(loss_ce.view(-1)[positive_indexes_for_loss.view(-1)]).any()):
        #         import pdb;pdb.trace()

        positive_indexes_for_loss

        positive_ce=loss_ce.view(-1)[positive_indexes_for_loss.view(-1)].mean()

        # if(torch.isnan(negative_ce).any()):
        #     import pdb;pdb.trace()
            
        # if(torch.isnan(positive_ce).any()):
        #     import pdb;pdb.trace()

        loss_ce = (negative_ce + positive_ce)/2



        # if (torch.isnan(loss_ce).any()):
        #     print('something went wrong')
        #     torch.cuda.synchronize()
        #     import pdb;pdb.set_trace()
        
        # selected_src_logits = 1 - src_logits[idx].view(-1,self.num_classes+1)[:,target_classes_o]
        # loss_ce = selected_src_logits.mean([0,1])


        losses = {'loss_ce': loss_ce}

        # Compute class error if log is enabled
        if log:
            # Compute class error accurassigned_prdacy
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            
        return losses

    @torch.no_grad()
    def cardinality_error(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_scores']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(targets["labels"].sizes, device=device)
        # Count the number of predictions that are NOT "no-object" (which is class zero, (for consistency with COCO format))
        card_pred = (pred_logits.argmax(-1) !=  0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        pred_logits have a shape of batch_size,num_boxes,num_classes+1(the +1 is the zero class indicating the background (index 0))
        """

        pred_logits = outputs['pred_scores'].sigmoid()

        softmax_logits = pred_logits.softmax(-1)
        softmax_logits_classes = torch.max(softmax_logits,dim=-1,keepdim=True)[0]
        


        predicted_objects = torch.sum(softmax_logits[...,1:] == softmax_logits_classes,dim=[-1,-2])

        

        predicted_non_objects = torch.sum(softmax_logits[...,0] == softmax_logits_classes[...,0],dim=-1)

        


        device = pred_logits.device
        tgt_lengths = torch.as_tensor(targets["labels"].sizes, device=device)
        # non_zero_elements = torch.nn.functional.gumbel_softmax(pred_logits,hard=True,dim=-1)[...,1:].sum([-1,-2]) #Softmax give to each label a value whose sum is 1 so avoiding the zero we are measuring if the sum of the others labels equals the total number of elements
        card_err = F.l1_loss(predicted_objects.float(), tgt_lengths.float())
        card_error_no_obj = F.l1_loss(predicted_non_objects.float(), softmax_logits.shape[1]-tgt_lengths.float())
        
        card_loss = ((1- 1/(1+torch.abs(card_err)))).mean() #((1 - torch.exp(-torch.abs(card_err)/100))+(1 - torch.exp(-torch.abs(card_error_no_obj)/100))) +(1-1/(1+torch.abs(card_error_no_obj)))
        losses = {'loss_cardinality': card_loss}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """


        assert 'pred_boxes' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        target_boxes=targets['boxes'].get_org_tensors()
        
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

        

        # Calculate L1 loss for bounding boxes
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum') / num_boxes

        

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))).sum() / num_boxes
        
        # import pdb; pdb.set_trace()
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    

    def _get_src_permutation_idx(self, indices):
        
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        device=self.empty_weight.device
        
        # permute targets following indices
        
        sizes = torch.as_tensor(list((tgt.shape[0] for (_, tgt) in indices)),dtype=int,device=device)
        
        indices_tensor = torch.arange(sizes.shape[0],device=device)
        
        batch_idx = torch.repeat_interleave(indices_tensor, sizes)
        
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality_error': self.cardinality_error,
            'boxes': self.loss_boxes,
            'cardinality':self.loss_cardinality,
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: dict each element is a 
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
       
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        sizes=[i[0].size(0) for i in indices]
     
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(sizes)
      
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)


        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            calculated = self.get_loss(loss, outputs, targets, indices, num_boxes)
            
            for k in calculated:
                if k in self.weight_dict:
                    calculated[k]*=self.weight_dict[k]

            losses.update(calculated)

         

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # loss_sum=sum(losses['labels'].values())+sum(losses['boxes'].values())

        return losses
    



# class Detection_One_to_Many_Criterion(Criterion):

    



#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_scores' in outputs
#         src_logits = outputs['pred_scores'].sigmoid()
        
#         bs, n_bboxes , _ = src_logits.shape

        

#         idx = self._get_src_permutation_idx(indices)
        
#         # logger.info(f'permutations {idx}')
#         target_classes_o = targets['labels'].get_tensor().long()
#         target_classes_sizes = torch.as_tensor(targets['labels'].sizes,device=src_logits.device)
#         # logger.info('this are the target labels ',target_classes_o)
#         # Initialize target classes with zeros in one step and set the required indices
#         target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)

#         target_classes[idx] = target_classes_o

#         loss_ce = F.cross_entropy(src_logits.transpose(1,2), target_classes, self.empty_weight,reduction='none')

#         loss_ce = loss_ce.reshape(bs, n_bboxes)

#         negative_indexes_for_loss=torch.ones(src_logits.shape[:2],dtype=bool)
#         negative_indexes_for_loss[idx] = False

#         positive_indexes_for_loss=torch.zeros(src_logits.shape[:2],dtype=bool)
#         positive_indexes_for_loss[idx] = True


#         target_classes_sizes = target_classes_sizes*3

#         target_classes_sizes[target_classes_sizes==0]=1

#         total_negative_loss=0

#         # probs = F.softmax(predictions, dim=-1)
#         # pt = torch.where(targets == 1, probs, 1 - probs)
#         # loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-6)

#         for batch_number in range(bs):

#             negatives_losses , _ = loss_ce[batch_number][negative_indexes_for_loss[batch_number]].sort(dim=0, descending=True)

#             # if(torch.isnan(negatives_losses).any()):
#             #     import pdb;pdb.trace()

#             taken_losses=negatives_losses[:target_classes_sizes[batch_number]]

#             total_negative_loss+=taken_losses.sum()


        
            
#         negative_ce = total_negative_loss.sum()/(target_classes_sizes.sum())

#         # if(torch.isnan(loss_ce.view(-1)[positive_indexes_for_loss.view(-1)]).any()):
#         #         import pdb;pdb.trace()

#         positive_indexes_for_loss

#         positive_ce=loss_ce.view(-1)[positive_indexes_for_loss.view(-1)].mean()

#         # if(torch.isnan(negative_ce).any()):
#         #     import pdb;pdb.trace()
            
#         # if(torch.isnan(positive_ce).any()):
#         #     import pdb;pdb.trace()

#         loss_ce = (negative_ce + positive_ce)/2



#         # if (torch.isnan(loss_ce).any()):
#         #     print('something went wrong')
#         #     torch.cuda.synchronize()
#         #     import pdb;pdb.set_trace()
        
#         # selected_src_logits = 1 - src_logits[idx].view(-1,self.num_classes+1)[:,target_classes_o]
#         # loss_ce = selected_src_logits.mean([0,1])


#         losses = {'loss_ce': loss_ce}

#         # Compute class error if log is enabled
#         if log:
#             # Compute class error accuracy
#             losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            
#         return losses

        
#     def forward(self, outputs, targets):
#         pass