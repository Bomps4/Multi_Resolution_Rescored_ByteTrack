from torch import nn
import torch
from typing import *
from enum import Enum
from loguru import logger
import PIL
from torchvision.transforms.functional import _get_inverse_affine_matrix
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

def get_size(img):
    if(isinstance(img,PIL.Image.Image)):
        return img.size[-1],img.size[0] #normally pil returns width height so we switch it to be consistent with pytorch
    if(isinstance(img,torch.Tensor)):
        return img.shape[-2:] #height,width

def _is_pil_image(img):
    return isinstance(img,PIL.Image.Image)


def horizontal_flip_bounding_box(
    bounding_box: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:    

    bounding_box[:, [0, 2]] = image_size[1] - bounding_box[:, [2, 0]]

    return bounding_box

def vertical_flip_bounding_box(
    bounding_box: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:    

    bounding_box[:, [1, 3]] = image_size[0] - bounding_box[:, [3, 1]]

    return bounding_box

def resize_bounding_box(bounding_box: torch.Tensor, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> torch.Tensor:
    old_height, old_width = old_size
    new_height, new_width = new_size
    r=min([new_width / old_width, new_height / old_height])
    #ratios = torch.tensor(r, device=bounding_box.device)
    return bounding_box*r#.view(-1, 2, 2).mul(ratios).view(bounding_box.shape)


def rotate_bounding_box(
    bounding_box: torch.Tensor,
    image_size: Tuple[int, int],
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    if center is not None and expand:
        logger.warning("The provided center argument has no effect on the result if expand is True")
        center = None
    out_bboxes = _affine_bounding_box_xyxy(bounding_box, image_size, angle=-angle, center=center, expand=expand)

    return out_bboxes

def _affine_bounding_box_xyxy(
    bounding_box: torch.Tensor,
    image_size: Tuple[int, int],
    angle: float,
    translate: Optional[List[float]] = None,
    scale: Optional[float] = None,
    shear: Optional[List[float]] = None,
    center: Optional[List[float]] = None,
    expand: bool = False,
) -> torch.Tensor:
    dtype = bounding_box.dtype if torch.is_floating_point(bounding_box) else torch.float32
    device = bounding_box.device

    if translate is None:
        translate = [0.0, 0.0]

    if scale is None:
        scale = 1.0

    if shear is None:
        shear = [0.0, 0.0]

    if center is None:
        height, width = image_size
        center_f = [width * 0.5, height * 0.5]
    else:
        center_f = [float(c) for c in center]

    translate_f = [float(t) for t in translate]
    affine_matrix = torch.tensor(
        _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear, inverted=False),
        dtype=dtype,
        device=device,
    ).view(2, 3)
    # 1) Let's transform bboxes into a tensor of 4 points (top-left, top-right, bottom-left, bottom-right corners).
    # Tensor of points has shape (N * 4, 3), where N is the number of bboxes
    # Single point structure is similar to
    # [(xmin, ymin, 1), (xmax, ymin, 1), (xmax, ymax, 1), (xmin, ymax, 1)]
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].view(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    # 2) Now let's transform the points using affine matrix
    transformed_points = torch.matmul(points, affine_matrix.T)
    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.view(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)

    if expand:
        # Compute minimum point for transformed image frame:
        # Points are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
        height, width = image_size
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0 * height, 1.0],
                [1.0 * width, 1.0 * height, 1.0],
                [1.0 * width, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        new_points = torch.matmul(points, affine_matrix.T)
        tr, _ = torch.min(new_points, dim=0, keepdim=True)
        # Translate bounding boxes
        out_bboxes[:, 0::2] = out_bboxes[:, 0::2] - tr[:, 0]
        out_bboxes[:, 1::2] = out_bboxes[:, 1::2] - tr[:, 1]

    return out_bboxes



def resize_as_transvod(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = nn.functional.interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target




def apply_affine_to_boxes(bboxes, matrix_torch):
    """
    Apply the same transformation to bounding boxes.

    :param bboxes: Torch tensor of shape (N, 4) where each row is (x1, y1, x2, y2).
    :param matrix_torch: 3x3 transformation matrix (torch tensor).
    :return: Transformed bounding boxes as a torch tensor of shape (N, 4).
    """
    # Extract the top-left and bottom-right corners of the bounding boxes
    corners = bboxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)

    # Convert to homogeneous coordinates (add a third coordinate = 1)
    ones = torch.ones(corners.shape[0], 1, dtype=torch.float32, device=bboxes.device)
    points = torch.cat([corners, ones], dim=1)

    # Apply the transformation matrix
    transformed_points = torch.matmul(points, matrix_torch.T)
    # 3) Reshape transformed points to [N boxes, 4 points, x/y coords]
    # and compute bounding box from 4 transformed points:
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).reshape(-1, 4)


    return out_bboxes


def generate_affine_matrix(scale:Union[Tuple[float,float],float]=(1.0, 1.0), shear:Union[Tuple[float,float],float]=(0.0, 0.0), translation:Union[Tuple[float,float],float]=(0.0, 0.0), rotation:float=0.0):
    """
    Generate a 3x3 affine transformation matrix based on scale, shear, translation, and rotation.

    :param scale: Tuple (scale_x, scale_y) for scaling. if float the same value will be used for both axis
    :param shear: Tuple (shear_x, shear_y) for shearing.
    :param translation: Tuple (translate_x, translate_y) for translation.
    :param rotation: Rotation angle in degrees.
    :return: 3x3 affine transformation matrix as a numpy array.
    """
    # Convert rotation to radians
    theta = np.radians(rotation)


    #if float use the same value for both axis
    if isinstance(scale,float):
        scale = (scale,scale)

    if isinstance(shear,float):
        shear = (shear,shear)
    
    if isinstance(translation,float):
        translation = (translation,translation)


    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Scaling matrix
    scale_matrix = np.array([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1]
    ])

    # Shearing matrix
    shear_matrix = np.array([
        [1, np.tan(np.radians(shear[0])), 0],
        [np.tan(np.radians(shear[1])), 1, 0],
        [0, 0, 1]
    ])

    # Translation matrix
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    # Combine transformations: T * S * Sh * R
    affine_matrix = translation_matrix @ scale_matrix @ shear_matrix @ rotation_matrix

    return torch.from_numpy(affine_matrix)