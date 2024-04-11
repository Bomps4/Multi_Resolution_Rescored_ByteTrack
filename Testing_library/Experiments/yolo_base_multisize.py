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

    @abstractmethod
    def get_train_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> Type[torch.optim.lr_scheduler._LRScheduler]:
        pass

    @abstractmethod
    def get_evaluator(self):
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
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.normalize=True
        self.class_agnostic=False
        self.bgr=True
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (320,240)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = "/scratch/ILSVRC2015/Data/VID/train/"
        # name of annotation file for training
        self.train_ann_dir = "/scratch/ILSVRC2015/Annotations/VID/train/"
        # name of annotation file for evaluation
        self.val_dat_dir = "/scratch/ILSVRC2015/Data/VID/"
        # name of annotation file for testing
        self.val_ann_dir= "/scratch/ILSVRC2015/Annotations/VID/"

        # --------------- transform config ----------------- #
        ''' currently ignored
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0
        '''

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 3
        # max training epoch
        self.max_epoch = 20
        # minimum learning rate during warmup
        self.warmup_lr = 0.005
        
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64 
        # name of LRScheduler
        
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 5
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 4e-5
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 100
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 3
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        #use resizing of pytorch o PIL
        self.resize_as_tensor=True
        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (576, 576)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.003
        # nms threshold
        self.nmsthre = 0.65
        #zero value for resize
        self.base_value=114
        #INDICATES COCO RESTRICTION TO BE USED (IF Imgvid no restriction else COCO_V1 XOr COCO_V2 )
        self.COCO='Imgvid'
        self.swap=(2, 1, 0)
        #consider a background class 
        self.Add_Background=False
        #use a pretrained in the model
        self.pretrained=True
        self.reduced_size = (256,256)
        self.resize_frequency=5 #number of frames before a frame is not resized(need to change name)
        self.seq_lenght=1
        self.pretrained_file = '/home/bompani/CNN_Training/Training_experiements/NN_Train_test/Models/Pretrained/yolox_nano.pth'
        self.mosaic_augment=True
        self.nntool_val_ann_dir = '/home/bompani/CNN_Training/mini_dataset/Annotations/'
        self.nntool_val_dat_dir = '/home/bompani/CNN_Training/mini_dataset/Data/'


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
    
    def get_train_augmentations(self):
        from ..Dataset.Imagenet import Mosaic_Augment
        if('augmentations' in self.__dict__):
            return self.augmentations
        self.augmentations=[Mosaic_Augment(self.dataset.data,self.input_size,(0.3,0.7))]
        return self.augmentations
        
    def get_train_transformations(self):
        from ..My_transforms.Transforms import T_RandomPhotometricDistort,T_RandomVerticalFlip,T_RandomHorizontalFlip,T_RandomIoUCrop
        if('transformations' in self.__dict__):
            return self.transformations
        self.transformations=[T_RandomIoUCrop(),T_RandomVerticalFlip(p=0.3),T_RandomHorizontalFlip(p=0.3),T_RandomPhotometricDistort(p=0.3)]
        return self.transformations

    def get_train_loader(self, batch_size, is_distributed, transforms=[], cache_img=False):
        from ..Dataset.Imagenet import Imagenet_VID_Dataset,Mosaic_Augment
        from torch.utils.data import DataLoader
        print(len(transforms))
        transforms=transforms+self._get_resizer_normalization()
        if(not hasattr(self,'dataset')):    
            self.dataset = Imagenet_VID_Dataset(self.train_ann_dir,self.data_dir,transform=transforms,COCO=self.COCO,Add_Background=self.Add_Background)
        if self.mosaic_augment:
            self.dataset.add_transform(Mosaic_Augment(self.dataset.data,self.input_size,(0.3,1)))
        
        
        '''
        def collate(seq_of_seq):
            images=[ i[0].unsqueeze(0) for i in seq_of_seq ]
            labels=[ j for i in seq_of_seq  for j in i[1]]
            return images,labels
        

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        '''
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True,"shuffle":True}
        if is_distributed:
            batch_size = batch_size // dist.get_world_size() 
            
            dataloader_kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=False
            )

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["timeout"] = 100 # maximum number of seconds allowed to recover a batch

        train_loader = DataLoader(self.dataset,**dataloader_kwargs)

        return train_loader,int(ceil(len(self.dataset)/batch_size))
    
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


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2,"weight_decay": self.weight_decay})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self,batch_size):
        from ..Schedulers.Warmup_cosine import Linear_Warmup_Cosine_Schedule 
        if('lr_scheduler' in self.__dict__):
            return self.lr_scheduler
        if("optimizer" not in self.__dict__):
            logger.error("you cannot instancuiate the learning rate scheduler without having the optimizer execution will stop")
            raise Exception("Tried to create lr-scheduler without first instanciating the optimizer")
        if(self.warmup_epochs!=0):
            self.lr_scheduler = Linear_Warmup_Cosine_Schedule(self.optimizer,self.warmup_epochs,self.basic_lr_per_img * batch_size,self.max_epoch+1)
        else:
            self.lr_scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch+1)
        return self.lr_scheduler
    
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
    
    def get_nntool_eval_loader(self, batch_size, is_distributed):
        from ..Dataset.Imagenet import Imagenet_VID_Dataset,NAMES
        
        
        transforms=self._get_resizer_normalization()
        self.val_dataset = Imagenet_VID_Dataset(self.nntool_val_ann_dir,self.nntool_val_dat_dir,val=True,transform=transforms,COCO=self.COCO,Add_Background=self.Add_Background,seq_lenght=self.seq_lenght)
        
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
        self.nntool_val_loader = torch.utils.data.DataLoader(self.val_dataset,**dataloader_kwargs)

        return self.nntool_val_loader


    def get_evaluator(self,subcaption=1):
        from detectron2.evaluation import COCOEvaluator
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from ..Dataset.Imagenet import set_sub_captioning,DATASET_NAMES

        if( hasattr(self,'val_loader')):
            
            NAMES=DATASET_NAMES[self.COCO]
            NAMES= NAMES if self.Add_Background else NAMES[1:]
            
            DATASET_NAME=self.COCO+f'coco_image_net_{subcaption}'+(''if not self.Add_Background else '_with_background')

            ds_image_net_val=set_sub_captioning(self.val_dat_dir,self.val_ann_dir,subcaption,COCO=self.COCO,Add_Background=self.Add_Background)
            DatasetCatalog.register(DATASET_NAME,ds_image_net_val)	
            MetadataCatalog.get(DATASET_NAME ).set(thing_classes=NAMES)
            self.evaluator = COCOEvaluator(DATASET_NAME, output_dir="./output",use_fast_impl=False)
        else:
            logger.error("you cannot instanciate the evaluator without its corresponding dataloader")
            raise Exception("Tried to create evaluator without evaluator dataloader")
        return self.evaluator
        
    def get_nntool_evaluator(self,subcaption=1):
        from detectron2.evaluation import COCOEvaluator
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from ..Dataset.Imagenet import set_sub_captioning,DATASET_NAMES

        if( hasattr(self,'nntool_val_loader')):
            
            NAMES=DATASET_NAMES[self.COCO]
            NAMES= NAMES if self.Add_Background else NAMES[1:]
            
            DATASET_NAME=self.COCO+f'NNTOOL_coco_image_net_{subcaption}'+(''if not self.Add_Background else '_with_background')

            ds_image_net_val=set_sub_captioning(self.nntool_val_dat_dir,self.nntool_val_ann_dir,subcaption,COCO=self.COCO,Add_Background=self.Add_Background)
            DatasetCatalog.register(DATASET_NAME,ds_image_net_val)	
            MetadataCatalog.get(DATASET_NAME ).set(thing_classes=NAMES)
            self.evaluator = COCOEvaluator(DATASET_NAME, output_dir="./output",use_fast_impl=False)
        else:
            logger.error("you cannot instanciate the evaluator without its corresponding dataloader")
            raise Exception("Tried to create evaluator without evaluator dataloader")


        
        return self.evaluator

    def get_trainer(self, args,val=False):
        __package__="NN_Train_test.Experiments.yolo_base"
        from ..Trainer.Trainer import Trainer
        trainer = Trainer(self, args,val=val)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer



'''
def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets
'''

   