import torch
from loguru import logger 
from ..utils.ragged_tensor import RaggedTensor


def stacking(tensor_list:list,key,dim=0):
    sizes=[tensor.shape[0] for tensor in tensor_list]
    if(key in {'boxes','labels'}):
        return RaggedTensor(tensor_list,sizes)
    if all([size==sizes[0] for size in sizes]):
        return torch.stack(tensor_list,dim=dim)
    else:
        return RaggedTensor(tensor_list,sizes)


def collate_normal(batch):
    images = []
    ground_truths = {i:[] for i in list(batch[0][1].keys())}
    
    for  i in batch:
        image,ground_truth=i
        images.append(image)
        for key in ground_truth:
            ground_truths[key].append(ground_truth[key])

    images = torch.stack(images)
    ground_truths = {gt_key:stacking(ground_truths[gt_key],gt_key,dim=0) for gt_key in ground_truths}
    
    return images,ground_truths

def collate_multiresolution(batch):

    images={}
    ground_truths = {}

    for i in batch:
        image,ground_truth=i
        size=tuple(image.shape[-2:])


        images[size]=[]
        ground_truths[size]={gt_key:[] for gt_key in list(batch[0][1].keys())}
    


    for i in batch:
        image,ground_truth=i
        size=tuple(image.shape[-2:])
        images[size].append(image)
        ground_by_size=ground_truths[size]
        for key in ground_truth:
            ground_by_size[key].append(ground_truth[key])

    ground_truths = {i:stacking(ground_truths[i],dim=0) for i in ground_truths}

    return images,ground_truths



