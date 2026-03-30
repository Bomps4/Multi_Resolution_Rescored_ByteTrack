# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch 
import torch.nn.functional as F
import random 
from typing import Sequence
from torch import nn
from copy import deepcopy

def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)



def format_outputs(boxes,scores):
    return {'pred_boxes':boxes,'pred_scores':scores}

def sum_multiple_nested_dicts(dicts):
    def sum_recursive(*dicts):
        summed = {}
        for d in dicts:
            for key, value in d.items():
                if isinstance(value, dict):
                    # If the value is a dictionary, recurse into the dictionaries
                    summed[key] = sum_recursive(*[d[key] for d in dicts if key in d])
                else:
                    # Otherwise, sum the values
                    summed[key] = summed.get(key, 0) + value
        return summed

    return sum_recursive(*dicts)



def sum_nested_dict_with_exclusions(nested_dict, exclude_keys=None):
    # Initialize exclude_keys as an empty set if not provided
    if exclude_keys is None:
        exclude_keys = set()

    def sum_recursive(d):
        total_sum = torch.tensor([0]).float()
        for key, value in d.items():
            if key in exclude_keys:
                # Skip this key and all its nested values
                continue
            if isinstance(value, dict):
                # Recursively sum nested dictionary
                total_sum += sum_recursive(value)
            else:
                # Add the value for non-dictionary items
                total_sum=total_sum.to(value.device)
                total_sum += value
        return total_sum

    return sum_recursive(nested_dict)

def unravel_nested_dict(nested_dict):
    # Dictionary to store the flattened result
    flat_dict = {}

    # Helper function to recursively unravel
    def unravel_recursive(d):
        for key, value in d.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                unravel_recursive(value)
            else:
                # If it's a value, append it to the list under the corresponding key
                if key in flat_dict:
                    flat_dict[key].append(value)
                else:
                    flat_dict[key] = [value]

    # Start the recursion
    unravel_recursive(nested_dict)

    return flat_dict

def total_size(nested_list):
    total = 0
    for element in nested_list:
        if isinstance(element, list):
            total += total_size(element)  # Recursive call for nested lists
        else:
            total += 1  # Count non-list element as 1
    return total


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

@torch.compile
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def all_equal(lst):
    # Base case: if lst is empty or only one element, all are trivially equal
    if not lst or len(lst) == 1:
        return True
    
    # Recursively flatten and compare each item in the list
    def flatten(item):
        # If item is a list, recursively flatten each element in it
        if isinstance(item, list):
            return [x for subitem in item for x in flatten(subitem)]
        else:
            return [item]
    
    # Flatten the first sublist to serve as a reference
    reference = flatten(lst[0])
    
    # Compare each flattened list to the reference
    for sublist in lst[1:]:
        if flatten(sublist) != reference:
            return False
    return True

def map_tensor_values(tensor, mapping):
    """
    Replace all elements in the tensor based on a dictionary mapping.

    works only if all the elements in the tensor are within the mapping (as it should be our case)

    Args:
        tensor (torch.Tensor): The input 1D tensor.
        mapping (dict): A dictionary with `int:int` format.

    Returns:
        torch.Tensor: A tensor with values replaced based on the mapping.
    """

    tensor_device = tensor.device
    # Extract keys and values from the mapping
    keys = torch.tensor(list(mapping.keys()), dtype=torch.long,device = tensor_device)
    values = torch.tensor(list(mapping.values()), dtype=torch.long,device = tensor_device )

    # Create a lookup table
    max_key = keys.max().item()  # Find the maximum key value

    # Check if all tensor values are within the valid range
    if (tensor > max_key).any():
        raise ValueError("The tensor contains values not covered by the dictionary mapping.")

    lookup_table = torch.full((max_key + 1,), -1, dtype=torch.long,device = tensor_device)  # Initialize with -1 (or another default)
    lookup_table[keys] = values  # Populate the lookup table with the mapping

    # Map the input tensor using the lookup table
    result = lookup_table[tensor]
    return result

def random_value(interval, center=0.0):
    """
    Generate a random value from a distribution with given bounds.

    :param interval: Either a tuple (min, max) defining the bounds, or a single value.
                     If a single value is provided, the bounds will be
                     (center - value, center + value).
    :param center: The center of the distribution (default is 0.0).
    :return: A random value within the specified bounds.
    """
    # Validate input types and values
    if isinstance(interval, tuple):
        assert len(interval) == 2, "Tuple 'interval' must have exactly two elements."
        assert interval[0] <= interval[1], "'interval' tuple must have min <= max."
        min_val, max_val = interval
    elif isinstance(interval, (int, float)):
        assert interval > 0, "'interval' must be a positive number if provided as a single value."
        min_val = center - interval
        max_val = center + interval
    else:
        raise TypeError("'interval' must be a tuple of two numbers or a single positive number.")

    return random.uniform(min_val, max_val)


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

if __name__ =='__main__':
    pass 