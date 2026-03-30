from .functional import *
from .functional import _affine_bounding_box_xyxy,_is_pil_image
from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip,RandomRotation,Resize
from torch import nn,Tensor
import torch
from random import choice
import numpy as np 
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as T
import torchvision

import random
from PIL.Image import Image
import PIL 
from PIL import ImageDraw
from typing import *
from loguru import logger
from collections import OrderedDict
from typing import Type,Dict
from ..utils.box_ops import box_xyxy_to_cxcywh,clamp_boxes,erase_zero_boxes
from ..utils.general_functions import random_value
from itertools import cycle, chain
import cv2 



class Selective_Compose(object):
    def __init__(self, transforms:List[nn.Module]):
        prepared = [(i.__class__.__name__,i) for i in transforms]
        self.transforms =  OrderedDict(prepared)
        self.reorder_transform()
    def __len__(self):
        return len(self.transforms)

    def __getitem__(self,idx:str):
        return self.transforms[idx]
    
    def __contains__(self,idx:str):
        return idx in self.transforms
    
    def check_if_multiple(self):
        names=[i.__class__.__name__ for i in self.transforms]
        resize_num=np.sum(np.isin(names,['T_Resize_as_TransVOD','T_Resize_Multires','T_Resize_as_YOLO','T_Resize']))
        return resize_num>1    


    def reorder_transform(self):

        assert not self.check_if_multiple() ,"cannot have multiple resizes use T_Resize_Multires"

        if 'Mosaic_Augment' in self.transforms:
            self.transforms.move_to_end('Mosaic_Augment',last=False)

        if 'T_Resize_Multires' in self.transforms:
            self.transforms.move_to_end('T_Resize_Multires',last=True)
        
        if 'T_Resize' in self.transforms:
            self.transforms.move_to_end('T_Resize',last=True)
        
        if 'T_Resize_as_YOLO' in self.transforms:
            self.transforms.move_to_end('T_Resize_as_YOLO',last=True)

        if 'T_Resize_as_TransVOD' in self.transforms:
            self.transforms.move_to_end('T_Resize_as_TransVOD',last=True)

        if 'T_To_tensor' in self.transforms:
            self.transforms.move_to_end('T_To_tensor',last=True)
        
        if 'T_Normalize' in self.transforms:
            self.transforms.move_to_end('T_Normalize',last=True)

    

    def add_trasform(self,transform:Union[List[Type[nn.Module]],Type[nn.Module]]):
        if(isinstance(transform,list)):
            for i in transform:
                self.transforms[i.__class__.__name__]=i
        else:
            self.transforms[transform.__class__.__name__]=transform
        self.reorder_transform()

    def remove_transform(self,idx:Union[str,List[str]]):
        if(isinstance(idx,list)):
            for i in idx:
                if i in self.transforms:
                    del self.transforms[i]
        else:
            if idx in self.transforms:
                del self.transforms[idx]
        self.reorder_transform()


    def __call__(self, image:Union[torch.Tensor,Image], target:Optional[Union[Dict[str,Tensor],None]]=None):
        for t in self.transforms:

            image, target = self.transforms[t](image, target)
            
        return image, target
    


def pil_to_tensor(img:Image,swap=(2, 0, 1),bgr=False)->Tensor:
    
    img=np.array(img) #when moved to array image goes to whc format to hwc 
    img=img.transpose(swap)
    img = np.ascontiguousarray(img,dtype=np.float32)
    if bgr:
        img = img[::-1,:,:].copy()

    return torch.from_numpy(img)




class T_Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,image:Union[Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[Tensor,Image], Union[Dict[str,Tensor],None]]:
        
        return image, target
    
class T_QuadResize(nn.Module):
    def __init__(self,size):
        super().__init__()
        if isinstance(size,int):
            size = (size,size)
        self.size = size
    def forward (self,image:PIL.Image,target:Optional[Dict]=None):
        image = image.resize(self.size)


        height,width=image.size

        if target is not None:
            target['resize_factor']=torch.tensor([1])
            target['mask']=torch.zeros(( height, width), dtype=torch.bool) 

        return image,target

    
class T_Resize_as_TransVOD(nn.Module):
    def __init__ (self,min_size:Union[int,list],max_size:int):
        super(T_Resize_as_TransVOD,self).__init__()
        self.min_size=min_size
        self.max_size=max_size
    def forward(self,image:Image,target:Optional[Union[Dict[str,Tensor],None]]=None):
        image,_ = resize_as_transvod(image,target,size=self.min_size,max_size=self.max_size)
        
        height,width=image.size
        if target is not None:
            target['resize_factor']=torch.tensor([1])
            target['mask']=torch.zeros(( height, width), dtype=torch.bool)
        return image,target

class T_RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target
    

class T_To_tensor(nn.Module):
    def __init__(self,swap=(2, 0, 1),bgr=False):
        super().__init__()
        self.bgr = bgr #if true goes to RGB to GBR
        self.swap = swap
    def forward(self,image:Union[Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[Tensor,Image], Union[Dict[str,Tensor],None]]:
        is_pil = _is_pil_image(image)
        if is_pil:
            image = pil_to_tensor(image,self.swap,self.bgr)
        
        # reverse_image = image.numpy().transpose((1,2,0))
        # PIL.Image.fromarray(reverse_image.astype(np.uint8)).save('after_to_tensor.png')
        # input()
        image = image.float()  
        
        
        return image, target

class T_HSV_Augment(nn.Module):
    def __init__(self,p:float,hgain:int=5,sgain:int=30,vgain:int=30):
        super().__init__()
        self.p = p
        self.hgain = hgain
        self.sgain = sgain 
        self.vgain = vgain
    def forward(self,image,targets):
        r = torch.rand(1)
        
        if r>self.p:
            
            return image,targets
        
        img_hsv = image.convert("HSV")
        image_converted = np.array(img_hsv, dtype=np.int16)
        hsv_augs = np.random.uniform(-1, 1, 3) * np.array([self.hgain, self.sgain, self.vgain])  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        

        image_converted[..., 0] = (image_converted[..., 0] + hsv_augs[0]) % 180
        image_converted[..., 1] = np.clip(image_converted[..., 1] + hsv_augs[1], 0, 255)
        image_converted[..., 2] = np.clip(image_converted[..., 2] + hsv_augs[2], 0, 255)

        return PIL.Image.fromarray(image_converted.astype(np.uint8),mode='HSV').convert("RGB"),targets
   
    

class T_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        # target = target.copy()
        # h, w = target['height'],target['width']
        # if "boxes" in target:
        #     boxes = target["boxes"]
        #     # boxes = box_xyxy_to_cxcywh(boxes)
        #     boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        #     target["boxes"] = boxes
        return image, target

class T_Resize_as_YOLO(nn.Module):
    def __init__(self,
                 out_size: Union[List[Tuple[int, Tuple[int, int]]],Tuple[int, int]],
                 base_value=0,
                 evaluation=False,
                 ):
        """
        Args:
            out_size (Tuple[int, int] or List[Tuple[int, Tuple[int, int]]]): Default output size.
            swap (Tuple[int, int, int]): Order of axes for image transposition.
            base_value (float): Base value for padding the image.
            evaluation (bool): Indicates evaluation mode.
            resize_schedule (List[Tuple[int, Tuple[int, int]]]): Schedule for resizing.
                Format: [(repeat_count, (width, height)), ...]
        """
    
        self.base_value = base_value
        self.evaluation = evaluation
        self.resize_schedule = [(1, out_size)] if isinstance(out_size,tuple) else out_size  # Default schedule
        self.size_cycle = self._initialize_schedule_cycle()  # Initialize cycling schedule
        self.current_vid = None
        super().__init__()

    def _initialize_schedule_cycle(self):
        """Expand the resize_schedule into a repeating cycle."""
        flat_schedule = list(chain.from_iterable([size] * count for count, size in self.resize_schedule))
        return cycle(flat_schedule)
    
    def reset_cycle(self):
        self.size_cycle = self._initialize_schedule_cycle()

    def forward(self, img: Image, target: Optional[Union[Dict[str, torch.Tensor], None]] = None, override_out_size: Optional[Tuple[int, int]] = None):

        size = get_size(img) #remember pytorch uses CHW so size is in HW format

        if target is not None and self.evaluation and 'vid_index' in target:

            if self.current_vid is None:
                self.current_vid = target['vid_index']
            
            if self.current_vid != target['vid_index']:
                self.reset_cycle()


        if  override_out_size is None:
            # Get the next size from the schedule
            current_out_size = next(self.size_cycle)
        else:
            current_out_size = override_out_size #this one is in wh which is the format usually used outside of pytorch

        # Get original image size
        size = np.array(size).astype(int)#nunmpy uses hw format

        # Prepare padding and mask
        padded_img = np.ones((current_out_size[0], current_out_size[1], 3), dtype=float) * self.base_value
        mask = np.ones((current_out_size[0], current_out_size[1]), dtype=bool)

        # Calculate the resize ratio
        r = min(current_out_size[1] / size[0], current_out_size[0] / size[1])

        # Compute the new size
        new_size = (size * r).astype(int)

        # Handle image as Tensor or PIL Image
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Resize the image
        # img = img.resize((new_size[-1], new_size[0]), PIL.Image.BILINEAR)

        
        
        img = np.array(img)
       
        
        img = cv2.resize(
        img,
        (int(new_size[-1]), int(new_size[0])),
        interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        
        img = img.transpose((1, 0, 2))
        
        
        # Place the resized image into the padded array
        padded_img[:new_size[1], :new_size[0]] = img
        
        # new_img = PIL.Image.fromarray(padded_img.transpose(1,0,2).astype(np.uint8))
        # draw_img = ImageDraw.Draw(new_img)
        # for i in target['boxes']:
        #     draw_img.rectangle(list(i*r), outline ="red") 
        # new_img.save('after_resize_and_padding.png')
        # input()


        mask = mask.astype(int) - 1
        mask = mask.astype(bool)
                
        
        # input()
        

        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        

        # Handle target resizing information
        if target is not None:
            target['boxes']=target['boxes'].to(torch.float32)
            target['boxes']*=r
            if 'resize_factor' in target:
                target['resize_factor']*= torch.tensor([r])
            else:
                target['resize_factor'] = torch.tensor([r])
            target['mask'] = torch.from_numpy(mask)

        

        return PIL.Image.fromarray(padded_img.astype(np.uint8).transpose((1, 0, 2))), target

        


class T_RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float] = (0.5, 1.5),
        saturation: Tuple[float] = (0.5, 1.5),
        hue: Tuple[float] = (-0.05, 0.05),
        brightness: Tuple[float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        # Check the type of p and handle accordingly
        if isinstance(p, tuple):
            assert len(p) == 6, "The tuple 'p' must have exactly 7 elements."
            self.p = p
        elif isinstance(p, float):
            self.p = (p,) * 6
        else:
            raise TypeError("'p' must be either a float or a tuple of 7 elements.")

    def forward(
        self, image:Union[torch.Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[torch.Tensor,Image], Union[Dict[str,Tensor],None]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p[0]:
            image = self._brightness(image)
        return image, target    

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p[1]:
                image = self._contrast(image)

        if r[3] < self.p[2]:
            image = self._saturation(image)

        if r[4] < self.p[3]:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p[4]:
                image = self._contrast(image)

        if r[6] < self.p[5]:
            is_pil = _is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            
            channels, _, _ = image.shape
            permutation = torch.randperm(channels)
            image = image[permutation, :, :]
            
            if is_pil:
                image = F.to_pil_image(image)

        return image, target
    

class T_RandomVerticalFlip(RandomVerticalFlip):
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__()
        self.p=p
    def forward(self,img:Union[torch.Tensor,Image],target:Union[Dict[str,Tensor],None])->Tuple[Union[torch.Tensor,Image],Union[Dict[str,Tensor],None]]:
        r=torch.rand(1)
        if r>=self.p:
            return img,target
        img_out=super().forward(img)
        img_size=get_size(img) #remember hw format
        if(target is not None):
            
            target['boxes']=vertical_flip_bounding_box(target['boxes'],img_size)
            
        return img_out,target

class T_RandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.p=p
    def forward(self,img:Union[torch.Tensor,Image],target:Union[Dict[str,Tensor],None])->Tuple[Union[torch.Tensor,Image],Union[Dict[str,Tensor],None]]:
        r=torch.rand(1)
        if r>=self.p:
            return img,target
        img_out=super().forward(img)
        img_size=get_size(img) #remember hw format
        if(target is not None):
            
            target['boxes']=horizontal_flip_bounding_box(target['boxes'],img_size)
        
        # draw = ImageDraw.Draw(img_out)
        # for box in target['boxes'].tolist():
        #     draw.rectangle(box,outline='red')
        # img_out.save('after_h_flip.png')
        # input()

        return img_out,target
    
class T_RandomRotate(RandomRotation):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        super().init(degrees, interpolation, expand, center, fill)
    
    def forward(self, img:Union[Image,Tensor],target:Union[Dict[str,Tensor],None])->Tuple[Union[Image,Union[Dict[str,Tensor],None]]]:
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)
        size=get_size(img)
        if(target is not None):
            target['boxes']=_affine_bounding_box_xyxy(target['boxes'],size,angle, center=self.center, expand=self.expand)
        return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill),target

class T_Resize(Resize):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None,resize_as_tensor=False):
        super().__init__(size, interpolation, max_size, antialias)
        self.size=(size[-1],size[0])
        self.resize_as_tensor=resize_as_tensor
    def forward(self, img:Union[torch.Tensor,Image],target:Union[Dict[str,Tensor],None])->Tuple[Union[torch.Tensor,Image],Union[Dict[str,Tensor],None]]:
        old_size=get_size(img)
        if(self.resize_as_tensor and isinstance(img,Image)):
            img=pil_to_tensor(img)
            img=img/255
        r = min(self.out_size[1] / old_size[0], self.out_size[0] / old_size[1])

        img=super().forward(img)
        if (target is not None):
            target['boxes']=resize_bounding_box(target['boxes'],old_size,self.size)

        return img,target 

class T_Resize_Multires(nn.Module):

    def __init__(self,first_resizer:Union[T_Resize,T_Resize_as_TransVOD,T_Resize_as_YOLO],second_resizer:Union[T_Resize,T_Resize_as_TransVOD,T_Resize_as_YOLO],frequency:int,resize_as_tensor:bool=False):
        super().__init__()
        self.first_resizer=first_resizer
        self.second_resizer=second_resizer
        self.frequency=frequency
        self.counter=0
        self.vid_id_previous=None

    def check(self,targets):
        
        if(self.vid_id_previous is None):
            self.vid_id_previous = targets['vid_index']
            return True
        
        result = targets['vid_index'] == self.vid_id_previous
        self.vid_id_previous = targets['vid_index']

        return not result
         
    def forward(self,image,targets):
        if(self.check(targets)):
            self.counter=0
            
        if(self.counter==0):
            self.counter+=1
            return self.first_resizer(image,targets)
        else:
            self.counter=(self.counter+1)%self.frequency
            return self.second_resizer(image,targets)
        


class T_Random_Affine (nn.Module):
    def __init__(self, shear,translate,rotation_degree,p:float=0.3,fill:int=114,scale:float=1.0):
        """
        Random Affine Transformation applied to both image and bboxes.
        shear: float indicates the shear angle range (-shear,+shear) can be a tuple in that case it will be the whole range
        translate = float indicates the translation amout (-translate,+translate) can be a tuple in that case it will be the whole range
        rotations_degrees = float indicates the rotation angle range (-rotations_degrees,+rotations_degrees) can be a tuple in that case it will be the whole range
        scale = if a signle value will be considered a fixed amout if a tuple indicates a range between a value will be selected 
        """
        super().__init__()
        self.shear = shear
        self.translate = translate
        self.rotation_degree = rotation_degree
        self.scale = scale 
        self.fill = fill
        self.p = p


    def get_affine_parameters_and_matrix(self,center):
        shear_x,shear_y = random_value(self.shear),random_value(self.shear)
        translate_x,translate_y = random_value(self.translate*2*center[0]),random_value(self.translate*2*center[1])
        rotation_degree = random_value(self.rotation_degree)
    
        if isinstance(self.scale,tuple):
            assert self.scale[0]>0, "scale lower bound must be a positive number!"
            scale = random_value(self.scale)
        else:
            assert self.scale>0, "scale must be a positive number!"
            scale = self.scale
        
        shear = (shear_x,shear_y)
        translate = (translate_x,translate_y)

        matrix = F._get_inverse_affine_matrix(center,rotation_degree,(translate_x,translate_y),scale,(shear_x,shear_y),inverted=False)
        
        matrix = torch.tensor(matrix).reshape(2, 3)
        

        return shear,translate,rotation_degree,scale,matrix

    def forward(self, image,targets):
        """
        Apply the transformation to a PIL image.

        :param image: PIL.Image instance.
        :return: Transformed PIL.Image instance.
        """
        
        r = torch.rand(1)

        if r>self.p:
            return image,targets


        h,w = get_size(image)

        center = (0.5*w,0.5*h)

        shear,translate,rotation_degree,scale,affine_matrix = self.get_affine_parameters_and_matrix(center)

        image_transformed = F.affine(image,angle=rotation_degree,translate=translate,scale=scale,shear=shear,fill=self.fill)

        if targets is not None:
            
            targets['boxes'] = apply_affine_to_boxes(targets['boxes'],affine_matrix)
            
            new_h,new_w = get_size(image_transformed)
            targets['boxes']=clamp_boxes ( targets['boxes'],new_w,new_h)
            targets['boxes'],targets['labels'] = erase_zero_boxes(targets['boxes'],targets['labels'])
        
        # draw = ImageDraw.Draw(image_transformed)
        # for box in targets['boxes'].tolist():
        #     draw.rectangle(box,outline='red')
        # image_transformed.save('after_affine.png')
        # input()

        return image_transformed,targets






        