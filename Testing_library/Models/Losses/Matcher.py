import torch
from scipy.optimize import linear_sum_assignment
from torch import nn,Tensor

from ...utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from .Hungarian_algorithm_pytorch import calculate_assignment
from loguru import logger

from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import concurrent

import torch.multiprocessing as multiprocessing

N_CPUS=multiprocessing.cpu_count()

def async_lsa(inp):
    i, c,=inp
    device=c.device
    indices = linear_sum_assignment(c[i].cpu().numpy())
    return (torch.as_tensor(indices[0], dtype=torch.int32,device=device), torch.as_tensor(indices[1], dtype=torch.int32,device=device))
    
    #(torch.as_tensor(indices[0], dtype=torch.int32,device=device), torch.as_tensor(indices[1], dtype=torch.int32,device=device)) 

def move_data_to_cpu(gpu_data):
    # Simulate the time it takes to move data from GPU to CPU
    # Replace with actual GPU-to-CPU data transfer code
    cpu_data = gpu_data.cpu()  # Assuming numpy array; replace with actual data transfer
    return cpu_data

def compute_on_cpu(cpu_data):
    # Run the CPU-bound linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cpu_data)
    return row_ind, col_ind

def move_result_to_gpu(result):
    # Assume result is a tuple of numpy arrays (row_ind, col_ind)
    row_ind, col_ind,device = result
    # Transfer result to GPU
    gpu_row_ind = torch.as_tensor(row_ind,device=device)
    gpu_col_ind = torch.as_tensor(col_ind,device=device)
    return gpu_row_ind, gpu_col_ind



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = nn.parameter.Parameter(torch.Tensor([cost_class]),False)
        self.cost_bbox = nn.parameter.Parameter(torch.Tensor([cost_bbox]),False)
        self.cost_giou = nn.parameter.Parameter(torch.Tensor([cost_giou]),False)
        # Thread and process executors
        self.thread_executor = concurrent.futures.ThreadPoolExecutor()
        
        self.process_executor = concurrent.futures.ProcessPoolExecutor()

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    # async def process_batch(self,gpu_data_batch):

    #     # Store results as they are processed, along with their index
    #     results = []

    #     device = gpu_data_batch

    #     async def handle_data(index, gpu_data):
    #         device = gpu_data.device
    #         # Step 1: Move data from GPU to CPU asynchronously
    #         cpu_data = await asyncio.get_running_loop().run_in_executor(self.thread_executor, move_data_to_cpu, gpu_data)
    #         print(f'i finished transfer {index}')
    #         # Step 2: Submit CPU-bound task to ProcessPoolExecutor
    #         future = asyncio.get_running_loop().run_in_executor(self.process_executor, compute_on_cpu, cpu_data)
            
    #         # Step 3: Collect the result from the CPU computation
    #         result = await future
    #         print(f'i finished computation {index}')
            
    #         # Step 4: Asynchronously bring the result back to the GPU
    #         gpu_result = await asyncio.get_running_loop().run_in_executor(self.thread_executor, move_result_to_gpu, (result[0],result[1],device)) 
    #         print(f'i finished to gpu {index}')
    #         # Store the GPU result with its original index
    #         results.append((index, gpu_result))

    #     # Schedule all data transfers and processing concurrently
    #     tasks = [handle_data(index, gpu_data[index]) for index, gpu_data in enumerate(gpu_data_batch)]
    #     await asyncio.gather(*tasks)

    #     print('i finieshed everything ')
    #     input()
    #     # Sort results by index to maintain the original order
    #     results.sort(key=lambda x: x[0])  # Sort by the first element (index)
    #     ordered_gpu_results = [result for _, result in results]  # Extract sorted GPU results

    #     return ordered_gpu_results    

    def cuda(self,device=None, non_blocking=False, memory_format=torch.preserve_format):        
        super().cuda(device=device,non_blocking=non_blocking,memory_format=memory_format)
        self.cost_class = self.cost_class.cuda(device=device,non_blocking=non_blocking,memory_format=memory_format)
        self.cost_bbox = self.cost_bbox.cuda(device=device,non_blocking=non_blocking,memory_format=memory_format)
        self.cost_giou = self.cost_giou.cuda(device=device,non_blocking=non_blocking,memory_format=memory_format)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # islist = isinstance(outputs['pred_scores'],list)
        
        # output=[]
        # if islist:
        #     #if the scores or the boxes are lists this means we have multiple resolutions
        #     # (each element in a list is the output for a specific resolution)
        #     target_labels=list(targets.keys())
        #     for block_id,(score_bock,box_block) in enumerate(outputs['pred_scores'],outputs["pred_boxes"]):
                
        #         target_label=target_labels[block_id]

        #         matching=self.matching_function(score_bock,box_block,targets[target_label])
                
        #         output.append(matching)
        # else:

        matching=self.matching_function(outputs['pred_scores'],outputs["pred_boxes"],targets)
        output=matching
        
        return output

    
    @torch.no_grad()
    def matching_function(self,score_bock,box_block,targets):

        block_size, num_queries = score_bock.shape[:2]

        # logger.info(f'score block shape {score_bock.shape}')

        out_prob = score_bock.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = box_block.flatten(0, 1)  # [batch_size * num_queries, 4]

       

        # Also concat the target labels and boxes
        tgt_ids =   targets["labels"].get_tensor().long()
        
        tgt_bbox =  targets["boxes"].get_tensor() 

        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.       

        
        cost_class = -out_prob[:, tgt_ids]
        
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # logger.info(f'queste sono le out_prob shape{out_prob.shape}')
        # logger.info(f'cost_bboxes {cost_bbox.shape}')
        # logger.info(f'queste sono le out_bbox shape {out_bbox.shape}')
        # logger.info(f'queste sono le target box shape {tgt_bbox.shape}')
        # print('cost_bbox shape',cost_bbox.shape)
        

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # logger.info(f'cost_giou {cost_giou.shape}')

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        device = C.device
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        
        # logger.info(f'sizes of cost function {block_size}, {num_queries}')
        

        C = C.view(block_size, num_queries, -1)

        sizes = targets['boxes'].sizes

        split_costs=C.split(sizes, -1)



        # logger.info(f'sizes {sizes}')
        # logger.info(f'cost {C.split(sizes, -1)[0].shape}')
        # input()
        # print('sizes ',sizes)
        # def async_lsa(i, c):
            
        #     indices = linear_sum_assignment(c[i].cpu().numpy())
        #     return (torch.as_tensor(indices[0], dtype=torch.int32,device=device), torch.as_tensor(indices[1], dtype=torch.int32,device=device)) 
        def process_chunk(chunk):
            i,chunck=chunk
            device=chunck.device

            # Move chunk to CPU and convert to NumPy
            chunk_cpu = chunck[i].cpu()
            print(chunk_cpu.shape)
            indices = linear_sum_assignment(chunk_cpu.numpy())
            print(indices)

            input()
            
            # Return indices as tensors back on the original device
            return (
                torch.as_tensor(indices[0],device=device),
                torch.as_tensor(indices[1],device=device)            )
       

        # indices = [process_chunk((index, chunk)) for index, chunk in enumerate(split_costs)]

        with ThreadPoolExecutor() as executor:

            futures = [executor.submit(process_chunk, (index, chunk)) for index, chunk in enumerate(split_costs)]
            indices = [f.result() for f in futures]

        print(indices[0])
        
        return indices#[(torch.as_tensor(i, dtype=torch.int32,device=device), torch.as_tensor(j, dtype=torch.int32,device=device)) for i,j in indices] 




# class One_To_Many_Matcher (nn.Module):
#     def __init__(self,top_k=20):
#         super().__init__()
#         self.top_k = top_k
    


#     def assign_fixed_boxes(self,iou_matrix):
#         """
#         Assign at most `n` fixed boxes to each n_box without overlaps across assignments.
        
#         :param iou_matrix: Tensor of shape (n_boxes, fixed_boxes_size) representing IoU values.
#         :param n: Maximum number of fixed boxes to assign to each n_box.
#         :return: List of tensors with indices of the assigned fixed boxes for each n_box.
#         """
#         device = iou_matrix.device
#         n_tgt_boxes, n_predictd_boxes = iou_matrix.shape
        
#         masked_iou = iou_matrix.clone()
#         n=self.top_k
#         # Track the availability of fixed boxes
#         available = torch.ones(n_predictd_boxes, dtype=torch.bool, device=iou_matrix.device)
#         assigned_indices = []
#         assigned_boxes = []

#         for i in range(n_tgt_boxes):
#             # Mask unavailable boxes by setting their IoU to a large negative value
#             selected_iou = masked_iou[i]
#             selected_iou[~available] = -float('inf')
            
#             # Select up to `n` highest IoU values for the current n_box
#             top_values, top_indices = torch.topk(selected_iou, min(n, selected_iou.size(0)), dim=0)

#             # Filter out invalid assignments (where IoU was set to -inf)
#             valid_indices = top_indices[top_values != -float('inf')]
#             val_box=torch.as_tensor([i for _ in range(len(valid_indices))],device=device)
            
#             # Append valid indices to the result
#             assigned_indices.append(valid_indices)
#             assigned_boxes.append(val_box)



#             # Mark these indices as unavailable
#             available[valid_indices] = False

#         return torch.cat(assigned_indices),torch.cat(assigned_boxes)
    

#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         box_block = outputs["pred_boxes"]
#         block_size, num_queries = box_block.shape[:2]

#         out_bbox = box_block.flatten(0, 1)  # [batch_size * num_queries, 4]
#         tgt_bbox =  targets["boxes"].get_tensor() 

#         tgt_sizes = targets["boxes"].sizes

#         C_IOU = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

#         C_IOU = C_IOU.view(block_size, num_queries,-1)

#         C_IOU_list = C_IOU.split(tgt_sizes,-1)

#         return self.batch_assign_fixed_boxes(C_IOU_list)



#     def batch_assign_fixed_boxes(self, iou_matrices):
#         """
#         Assign at most `n` fixed boxes for each n_box in a batch of IoU matrices.

#         :param iou_matrices: Tensor of shape (batch_size, n_boxes, fixed_boxes_size),
#                             where each slice along the batch represents an IoU matrix.
#         :param n: Maximum number of fixed boxes to assign to each n_box.
#         :return: List of length batch_size, where each element is a list of tensors with
#                 assigned fixed box indices for the corresponding batch IoU matrix.
#         """
#         results = []
        

#         for iou_idx,iou_matrix in enumerate(iou_matrices):
            
#             # Process each IoU matrix in the batch independently
#             assigned_pd,assigned_tg = self.assign_fixed_boxes(iou_matrix[iou_idx].permute(1,0))  # Use the function defined earlier
            
#             results.append((assigned_pd,assigned_tg))
            
#         return results



    

class One_To_Many_Matcher (nn.Module):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_THRESHOLD = -1
    

    __annotations__ = {
        "BELOW_THRESHOLD": int,
    }

    def __init__(self,threshold:float, n:int = 30) -> None:
        """
        Args:
        threshold: float value indicating the minimal value to have a match all other are considered no object 
           
        """
        super().__init__()
        self.threshold = threshold
        self.n = n
    
    def forward(self, outputs, targets):
        box_block = outputs["pred_boxes"]
        block_size, num_queries = box_block.shape[:2]

        out_bbox = box_block.flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox =  targets["boxes"].get_tensor() 

        tgt_sizes = targets["boxes"].sizes

        C_IOU = generalized_box_iou(box_cxcywh_to_xyxy(tgt_bbox),box_cxcywh_to_xyxy(out_bbox))

        C_IOU = C_IOU.view(block_size, num_queries,-1)

        C_IOU_list = C_IOU.split(tgt_sizes,-1)    

        return self.batch_assignment_calculation(C_IOU_list)
    
    def batch_assignment_calculation(self, cost_matrix):
        results = []
        

        for iou_idx,iou_matrix in enumerate(cost_matrix):
            
            # Process each IoU matrix in the batch independently
            assigned_prd,assigned_tg = self.assign_boxes(iou_matrix[iou_idx].permute(1,0))  # Use the function defined earlier
            
            results.append((assigned_prd,assigned_tg))
        
        print(results)
            
        return results


    def assign_boxes(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        tgt_elements,dt_elements = match_quality_matrix.shape[:2]

        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction

        matched_vals, matches = match_quality_matrix.max(dim=0)



        

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.threshold
        matches[below_low_threshold] = self.BELOW_THRESHOLD

        
        trasposed_matching_matrix = match_quality_matrix.clone()

        highest_quality_pred_foreach_gt = []

        ### force to have at least one match per ground truth

        # for idx,_ in enumerate(trasposed_matching_matrix):

        #     _, highest_quality_each_gt = trasposed_matching_matrix[idx][None].max(dim=-1) #none needed to fool the max operation when working on size 1 tensors


        #     highest_quality_pred_foreach_gt.append(highest_quality_each_gt)

            
            
            
        #     trasposed_matching_matrix[idx:,highest_quality_each_gt] = -1

        

        for idx in range(tgt_elements):  # Iterate over each ground truth
            # Use topk to get the top-n predictions for the current ground truth
            topk_vals, topk_indices = trasposed_matching_matrix[idx].topk(k=self.n, largest=True)

            # Force top-n predictions to be associated with this ground truth
            highest_quality_pred_foreach_gt.append(topk_indices)

            # Update match assignments
            matches[topk_indices] = idx

            # Invalidate these predictions to avoid duplicate assignments
            trasposed_matching_matrix[:, topk_indices] = -1  # Mark these predictions as used


        
        # highest_quality_pred_foreach_gt = torch.cat(highest_quality_pred_foreach_gt)
        

        # matches[highest_quality_pred_foreach_gt] = torch.arange(
        #     tgt_elements, dtype=torch.int64, device=trasposed_matching_matrix.device
        # )

            

        return torch.arange(0,matches.size(0)),matches
    


class FCos_Matcher (nn.Module):
    def __init__(self,center_sampling_radius):
        super().__init__()
        self.center_sampling_radius = center_sampling_radius
    
    def forward(self,anchors,targets,num_anchors_per_level):
        "in my framework all the bounding boxes are the moment that enter the flux in the cx,cy,w,h format!"

        matched_idxs = []

        w,h = targets['width'],targets['height']

        gt_bboxes = targets['boxes'].get_org_tensors()

        anchors = anchors.repeat((len(gt_bboxes),1,1))


        
        # print('ancore ',anchors.shape)
        # input()

        for anchors_per_image, gt_bboxes_per_image in zip(anchors, gt_bboxes):


            if gt_bboxes_per_image.numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue
            

            gt_boxes = gt_bboxes_per_image
            gt_centers = (gt_boxes[:, :2])   # Nx2
            anchor_centers = anchors_per_image[:, :2]  # N
            anchor_sizes = anchors_per_image[:, 2]
            # center sampling: anchor point must be close enough to gt center.

           

            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(dim=2).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1) (number of detections)+

            x0, y0, x1, y1 = box_cxcywh_to_xyxy(gt_boxes).unsqueeze(dim=0).unbind(dim=2)  # (1, M) number of ground truth per image 
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M) 0.4167, 0.4583   0.0198 4583


            minimum_pair_dist = pairwise_dist.min(dim=2).values

            matched_centers = minimum_pair_dist > 0

            # matched_centers = (matched_centers.sum()>0)*matched_centers + (matched_centers.sum()<=0)*(minimum_pair_dist == minimum_pair_dist.max().values)

            

            # anchor point must be inside gt
            pairwise_match &= matched_centers

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes.clone()/2 
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes.clone() * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] ) * (gt_boxes[:, 3])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            # if ((matched_idx==-1).all()):
            #     print('pairwise_match ',pairwise_match)
            #     import pdb; pdb.set_trace()
            #     input()

            matched_idxs.append((torch.arange(0,matched_idx.size(0)),matched_idx))
        
        return matched_idxs
    
