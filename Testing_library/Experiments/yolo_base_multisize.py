# Copyright (c) 2021-2022 Megvii Inc. and its affiliates. 
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

from loguru import logger
import torch
from torch import nn
from torch.nn import Module
from yaml import load, dump    
import uuid
import random 
import numpy as np
import ast
import pprint
from math import ceil
from abc import ABCMeta, abstractmethod
from typing import Dict,Union,Type
from tabulate import tabulate
from torch import distributed as dist
import os
from functools import partial



class SingletonABCMeta(ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]   

class BaseExp(metaclass=SingletonABCMeta):
    """Basic class for any experiment."""
    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10
        self.experiment_name="Experiment name"
        

    @abstractmethod
    def get_model(self) -> Module:
        pass


    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def _init_with_Yaml(self,Yaml_file):
        properties_dict=load(Yaml_file)
        for i in properties_dict:
            setattr(self,i,properties_dict[i])
        

def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'YOLOX'
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 30
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        self.multisize=True
        self.base_value=114 #used to fill the empty part of an image in resize
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        #experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost  a lot of memory, reduce this value.
        self.data_num_workers = 4
        # name of annotation file for evaluation
        self.val_dat_dir = "/path/to/ILSVRC2015/Data/VID/"
        # name of annotation file for testing
        self.val_ann_dir= "/path/to/Annotations/VID/"
        #define if images are read in the bgr format or rgb if false
        self.bgr=True   
        # full resolution output image size during evaluation/test
        self.test_size = (576, 576)
        # low resolution image size for the reduced boxes
        self.reduced_size = (256,256)
        
        # -----------------  testing config ------------------ #
        #use resizing of pytorch o PIL
        self.resize_as_tensor=True
        #mean and standard deviation for images
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.normalize=True #if to normalize the images pixels or leave it in the 0 255 range
        self.class_agnostic=False #define the type of nms applied 
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.003
        # nms threshold
        self.nmsthre = 0.65
        #INDICATES COCO RESTRICTION TO BE USED (IF Imgvid no restriction else COCO_V1 XOr COCO_V2 )
        self.COCO='Imgvid'
        #swapping to be applied to have CHW format
        self.swap=(2, 1, 0)
        #consider a background class 
        self.Add_Background=False
        self.resize_frequency=5 #number of frames before a frame is not resized (TODO change name)
        self.seq_lenght=1
        

    def get_model(self):
        from ..Models.Heads.yolo_head import   YOLOXHead
        from ..Models.Heads.yolo_pafpn import YOLOPAFPN
        from ..Models.yolox import YOLOX

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head,xyxy=self.xyxy)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
    
    
    def get_boxes_resizer(self):
        from ..My_transforms.functional import resize_bounding_box
        if(hasattr(self,'val_loader')or hasattr(self,'nntool_val_loader')): 
            if 'T_Resize' in self.val_dataset.transform :
                p_resize=partial(resize_bounding_box,old_size=self.test_size[::-1])
                return p_resize
            elif 'T_Resize_as_YOLO' in self.val_dataset.transform :
                def resize_bboxes_as_yolo(bboxes,new_size):
                    height,width=new_size
                    r=min([self.test_size[0]/width,self.test_size[1]/height])
                    return bboxes/r
                
                return resize_bboxes_as_yolo
            else:
                logger.warning("no resizing in validation a fictitious operation was added to keep interface concistency")
                p_resize=partial(resize_bounding_box,old_size=self.test_size[::-1])
                return p_resize


        else:
            logger.error("you cannot instancuiate the validation resizer without the corresponding dataloader")
            raise Exception("Tried to create resize operation for bounding boxes without the corresponding dataloader")

    def _get_second_resize(self):
        from ..My_transforms.Transforms import T_Resize_as_YOLO,T_To_tensor,T_Resize
        if (self.resize_as_tensor):
            return T_Resize(self.reduced_size,resize_as_tensor=self.resize_as_tensor)
        else:
            return T_Resize_as_YOLO(self.reduced_size,base_value=self.base_value,swap=self.swap,bgr=self.bgr)
        
    def _mean_std_dim(self):
        indice_b=self.swap.index(2)
        indice_a=len(self.swap)-indice_b -1 
        current_mean=self.mean
        current_std=self.std

        for _ in range(indice_b):
            current_mean,current_std=current_mean[None],current_std[None]
        for _ in range(indice_a):
            current_mean,current_std=current_mean[...,None],current_std[...,None] 
        
        return current_mean,current_std
        
    def _get_resizer_normalization(self):
        # if(len(self.mean.shape)==1):
        #     self.mean,self.std=self._mean_std_dim()
        from ..My_transforms.Transforms import T_Resize_as_YOLO,T_To_tensor,T_Resize
        if (self.resize_as_tensor):
            return [T_To_tensor(self.normalize,self.mean,self.std),T_Resize(self.test_size,resize_as_tensor=self.resize_as_tensor)]
        else:
            return [T_Resize_as_YOLO(self.test_size,base_value=self.base_value,swap=self.swap,bgr=self.bgr),T_To_tensor(self.normalize,self.mean,self.std)]


    
    def _get_resize_multires(self):
        from ..My_transforms.Transforms import T_Resize_Multires
        first_resize = self._get_resizer_normalization()[0]
        second_resize = self._get_second_resize()
        
        return T_Resize_Multires(first_resize,second_resize,self.resize_frequency)

    def get_eval_loader(self, batch_size, is_distributed):
        from ..Dataset.Imagenet import Imagenet_VID_Dataset,NAMES
        from ..My_transforms.Transforms import T_Resize_Multires
        
        assert batch_size==1 ,"cannot have batch size different than 1 we have multiple resolutions"
        transforms=self._get_resizer_normalization()
        self.val_dataset = Imagenet_VID_Dataset(self.val_ann_dir,self.val_dat_dir,val=True,transform=transforms,COCO=self.COCO,Add_Background=self.Add_Background)
        self.val_dataset.remove_transform(['T_Resize_as_YOLO','T_Resize'])
        resize_multires = self._get_resize_multires()
        self.val_dataset.add_transform(resize_multires)
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(self.val_dataset)
       
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": sampler,
        }
        
        dataloader_kwargs["batch_size"] = batch_size
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,**dataloader_kwargs)

        return self.val_loader
    
    def get_trainer(self, args,val=False):
        __package__="NN_Train_test.Experiments.yolo_base"
        from ..Evaluator.Evaluator import Evaluator
        trainer = Evaluator(self, args,val=val)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer




   
