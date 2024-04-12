# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

from .functional import *
from .functional import _affine_bounding_box_xyxy
from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip,RandomRotation,Resize
from torch import nn,Tensor
import torch
from random import choice
import numpy as np 
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as T
import torchvision
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP
from PIL.Image import Image
import PIL 
from typing import *
from loguru import logger
from collections import OrderedDict
from typing import Type

np.random.seed(42)
torch.manual_seed(42)

class Selective_Compose(object):
    def __init__(self, transforms:List[nn.Module]):
        prepared=[(i.__class__.__name__,i) for i in transforms]
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

def pil_to_tensor(img:Image)->Tensor:
    
    img=np.array(img)
    img=img.transpose((2,0,1))

    return torch.from_numpy(img)

class T_Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,image:Union[Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[Tensor,Image], Union[Dict[str,Tensor],None]]:
        
        return image, target
    
class T_Resize_as_TransVOD(nn.Module):
    def __init__ (self,min_size:Union[int,list],max_size:int):
        super(T_Resize_as_TransVOD,self).__init__()
        self.min_size=min_size
        self.max_size=max_size
    def forward(self,image:Image,target:Optional[Union[Dict[str,Tensor],None]]=None):
        image,_ = resize_as_transvod(image,target,size=self.min_size,max_size=self.max_size)
        
        height,width=image.size
        if target is not None:
            target['resize_factor']=1
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
    def __init__(self,normalize:bool,mean:Union[float,int,Collection[int],Collection[float],Tensor],std:Union[float,int,Collection[int],Collection[float],Tensor]):
        self.mean=mean[:,None,None]
        self.std=std[:,None,None]
        self.normalize=normalize
        super().__init__()
    def forward(self,image:Union[Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[Tensor,Image], Union[Dict[str,Tensor],None]]:
        is_pil = _FP._is_pil_image(image)
        if is_pil:
            image = pil_to_tensor(image)
        image = image.float()
        if self.normalize:
            image = (image - self.mean)/self.std
        
        return image, target

class T_Resize_as_YOLO(nn.Module):
    def __init__(self,out_size:Tuple[int,int],swap=(2,1,0),base_value=0,bgr=True):
        self.out_size=out_size
        self.bgr=bgr
        self.swap=swap
        self.base_value=base_value
        super().__init__()
    def forward(self,img:Image,target: Optional[Union[Dict[str,Tensor],None]]=None ):
        size=get_size(img)

        size=np.array(size).astype(int)
        padded_img = np.ones((self.out_size[0], self.out_size[1], 3), dtype=float) * self.base_value
        mask = np.ones((self.out_size[0], self.out_size[1]),dtype=bool)
        r = min(self.out_size[1] / size[0], self.out_size[0] / size[1])

        
        new_size=(size*r).astype(int)
        
        if(isinstance(img,torch.Tensor)):
            img=F.to_pil_image(img)
        
        
        img=img.resize((new_size[-1],new_size[0]),PIL.Image.BILINEAR)
        
        img=np.array(img)
        if(self.bgr):
            img=img[...,::-1]#rgb to bgr
        img=img.transpose((1,0,2))
        
        padded_img[: new_size[1], : new_size[0]] = img
        mask=mask.astype(int)-1
        mask=mask.astype(bool)
        padded_img = padded_img.transpose(self.swap)
        
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        tensor=torch.from_numpy(padded_img)
        target['resize_factor']=r
        target['mask']=torch.from_numpy(mask)

        # if target is not None:
        #     target['boxes']=resize_bounding_box(target['boxes'],size,new_size)
        
        
        return tensor,target

        


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
        self.p = p

    def forward(
        self, image:Union[torch.Tensor,Image], target: Optional[Union[Dict[str,Tensor],None]] = None
    ) -> Tuple[Union[torch.Tensor,Image], Union[Dict[str,Tensor],None]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)
        return image, target    

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            is_pil = _FP._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = _FT.convert_image_dtype(image)
            
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
        img_size=get_size(img)
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
        img_size=get_size(img)
        if(target is not None):
            
            target['boxes']=horizontal_flip_bounding_box(target['boxes'],img_size)
            
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
        



        

















