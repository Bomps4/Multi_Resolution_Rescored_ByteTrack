import datetime
import os
import time
import gc
from loguru import logger
import torch
from ..Models.Heads.yolo_head import YOLOXHead
from ..Dataset.DataPrefetcher import DataPrefetcher
from ..utils.logger import setup_logger
from ..utils.env import get_world_size,get_rank,get_local_rank,synchronize,is_parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from ..utils.checkpointing import save_experiment_checkpoint,load_checkpoint
from .metric import MeterBuffer
from ..Dataset.Imagenet import Mosaic_Augment,NAMES,data_dict,DATASET_NAMES
import numpy as np
from tabulate import tabulate
import random
import pprint
from PIL import Image,ImageDraw
from ..Models.Postprocess.similaritymetrics import iou_batch
import contextlib
from ..Models.NN_Bytes import NN_Augmented
from NN_Train_test.My_transforms.functional import resize_bounding_box
from torchvision.ops import nms

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import copy
from torchinfo import summary
from ..utils.general_functions import total_size
from ..utils.box_ops import box_xyxy_to_xywh
from collections.abc import MutableMapping

def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))

def return_size(input):
            size=0
            if(isinstance(input,dict)):
                for i in input:
                    size+=i.shape[0]
                return size
            return input.shape[0]

# def return_number #finire funzione per returnare parametri

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.debug(line.rstrip())

    def flush(self):
        pass


class Trainer:
    def __init__(self, exp, args, val=False):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        
        

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        # if self.amp_training:
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        #     torch.set_float32_matmul_precision('high')
        # else:
        #     torch.set_float32_matmul_precision('high')
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
       

        #device related variables
        self.rank = get_rank()
        self.is_distributed = get_world_size() > 1
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)

        #val loader and prefetcher
        self.val_loader = self.exp.get_eval_loader(self.args.batch_size,self.is_distributed)
        self.val_prefetcher = DataPrefetcher(self.val_loader)
        #self.use_model_ema = exp.ema not converted yet


        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_mAP = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.saving_dir=exp.saving_dir

        # evaluator 
        self.evaluator = self.exp.get_evaluator(self.saving_dir)
        self.eval_interval = self.exp.eval_interval
        self.eval_only=val

        self.start_epoch=0
        self.generate_main_properties()
        self.printing_interval=self.exp.print_interval

        

        if self.args.ckpt != '':
            self.resume()
        
        if self.eval_only:
            logger.info("args: {}".format(self.args))
            logger.info("exp value:\n{}".format(self.exp))
            self.val_prefetcher = DataPrefetcher(self.val_loader)
            self.evaluate()




    def before_train(self):
        
        #setup device
       
        self.no_aug = True
        
        if(self.no_aug):
            logger.info("Augmentation removed from training")

            # self.train_loader = self.exp.get_train_loader(
            #     self.args.batch_size,self.is_distributed
            # )
            self.train_loader = self.exp.get_eval_loader(self.args.batch_size,self.is_distributed,for_train=True)
            
            
        else:
            augm=self.exp.get_train_transformations()
            self.train_loader = self.exp.get_train_loader(
                self.args.batch_size,self.is_distributed,transforms=augm
            )
        
        
        
        self.model.train()


        
        logger.info("Training start...")

    def generate_main_properties(self):
        model = self.exp.get_model()
        if (torch.cuda.is_available()):
            torch.cuda.set_device(self.local_rank)
            self.exp.selected_classes = self.exp.selected_classes.to(self.device)
        
        model=model.to(self.device)

        if self.amp_training:
            model = model.to(torch.float16)
            

        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        
        self.model = model

        if not self.eval_only:
            self.optimizer=self.exp.get_optimizer(self.args.batch_size,self.is_distributed)
            self.lr_scheduler = self.exp.get_lr_scheduler(self.args.batch_size)

            logger.info("args: {}".format(self.args))
            logger.info("exp value:\n{}".format(self.exp))




    def resume (self):
        logger.info("loading checkpoint for evaluation")
        ckpt_file = self.args.ckpt
        
        self.start_epoch = 0
        
        model=self.model
        
        if(not (ckpt_file =='')):
            ckpt = torch.load(ckpt_file, map_location=self.device)
            try:
                logger.info('loading state')
                model.load_state_dict(ckpt['model'])
            except KeyError:
                logger.info('no "model" key present changing to "state_dict"')
                try:
                    model.load_state_dict(ckpt["state_dict"])
                except KeyError:
                    logger.info('checking without any key passing the dictionary as is')
                    model.load_state_dict(ckpt)

            except AttributeError:
                logger.info('checking if the save file is not just a state but the model itself')
                model=ckpt['model']
        if self.eval_only:
            self.model=model
            return 
        
        # optimizer related init
        self.optimizer = self.optimizer.load_state_dict(ckpt['optimizer'])
        # value of epoch will be set in `resume_train`
        self.lr_scheduler = self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.start_epoch=ckpt['epoch']
        return


    def before_epoch(self):

        if torch.cuda.is_available():

            self.train_prefetcher = DataPrefetcher(self.train_loader)   
            self.val_prefetcher = DataPrefetcher(self.val_loader)

        # logger.info(f"---> start train epoch{epoche}")
        # transform=self.exp.get_train_transformations()
        # augment=self.exp.get_train_augmentations()
        # if (self.epoch + 1- self.exp.warmup_epochs ) % 4 ==0 \
        #         and (self.epoch + 1- self.exp.warmup_epochs ) \
        #         and (self.epoch + 1 < self.max_epoch - self.exp.no_aug_epochs):
        #     if(self.no_aug):
        #         idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
        #         self.train_loader.dataset.remove_transform(idx)
        #     else:   
        #         self.train_loader.dataset.add_transform(transform+augment)
        #     logger.info('Refreshing dataloader')
                
        # if self.epoch + 1 >= self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
        #     logger.info("--->Removing all Augmentations!")
        #     logger.info("--->USING L1 Loss")
           
        #     if('head' in self.model._modules and 'use_l1' in self.model.head.__dict__):
        #         self.model.head.use_l1=True
        #     idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
        #     self.train_loader.dataset.remove_transform(idx)
        # elif 0< self.epoch + 1 and  self.epoch + 1 <= self.exp.warmup_epochs:
        #     logger.info("--->No Augmentation in warmup")
        #     idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
        #     self.train_loader.dataset.remove_transform(idx)
        # else:
        #     logger.info("--->Including aug now!")
        #     self.train_loader.dataset.add_transform(augment)
        #     self.train_loader.dataset.add_transform(transform)
        #     logger.info("--->Training full model now!")

        #     self.train_prefetcher = DataPrefetcher(self.train_loader)

    def train(self):

        torch.cuda.set_device(self.local_rank)

        self.before_train()

        for epoch in range(self.start_epoch,self.start_epoch+self.max_epoch):
            self.before_epoch()
        
            self.train_one_epoch()

            self.after_epoch(epoch)


    def after_epoch(self,epoch):        
        
        synchronize()
        
        torch.cuda.empty_cache()

        self.lr_scheduler.step()
        if self.exp.save_history_ckpt:
            self.save_ckpt(f"epoch_{epoch + 1}",epoch)

        if (epoch % self.exp.eval_interval) == 0 and epoch!=0:
            mAP=self.evaluate()
            if mAP>self.best_mAP:
                self.best_mAP=mAP
                self.save_ckpt('best_checkpoint',epoch)

        
   
    def train_one_epoch(self):
        """
        From the prefetcher there are 2 possible outputs if we are not in ù
        a multiresolution size input are a single tensor with size Batch X Channel x Height X Width 
        Else the input is a dictionary with keys the size of the images and values tensors
        with Block_size X Channel X Height x Width

        (la sommatoria delle varie Block size è sempre <= Batch size)
        
        """

        with torch.cuda.amp.autocast(enabled=self.amp_training): #enabling fp16 (and also tensor cores)
                
            with tqdm.tqdm(total=len(self.train_prefetcher)*self.train_loader.batch_size) as pbar:
                for idx,(_input,target) in enumerate(self.train_prefetcher):
                    
                    
                    
                    size=return_size(_input)
                    pbar.update(size)



                    _input,target=self.model.format_input(_input,target)
                    
                    #losses is a dictionary of all the losses while loss is the sum of all the selected losses
                    losses,loss=self.model(_input,target)
                     

                    self.scaler.scale(loss).backward() 
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    #flattening dictionary and updating the metes
                    flat_losses = flatten_dict(losses)
                    flat_losses = {key:flat_losses[key].detach().item() for key in flat_losses}
                    flat_losses['total loss'] = loss.detach().item()
                    self.meter.update(**flat_losses)
                    self.optimizer.zero_grad()
                    
                    if idx % self.printing_interval==0:
                        table = [(key,self.meter.get_filtered_meter(key)[key].global_avg) for key in flat_losses]
                        table  = tabulate(table,['loss_name','value'],tablefmt="outline")                       
                        logger.info(str(table))
                       
                            
                                

             
    def evaluate(self):

        torch.cuda.set_device(self.local_rank)

        if self.amp_training:
            self.model = self.model.to(torch.float16)

        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                with tqdm.tqdm(total=len(self.val_prefetcher)*self.val_loader.batch_size) as pbar:
                    for _input,target in self.val_prefetcher:

                        size=return_size(_input)
                        pbar.update(size)
                        if self.amp_training:
                            if isinstance(_input,dict):
                                for key in _input:
                                    _input[key]=_input[key].to(torch.float16)
                            else:
                                _input=_input.to(torch.float16)
                        

                        _input,target = self.model.format_input(_input,target)

                        
                       
                        out = self.model(_input,target)

                        
                        out = self.model.format_output(out,target,to_coco=True)

                        
                    
                        image_ids=[target[key]['images_id'] for key in target]  if(isinstance(_input,dict)) else target['images_id'] 

                        image_ids = target['images_id']

                       
                        

                        # target_labels = target['labels'].to('cuda:0').get_org_tensors()

                        # target_scores = [torch.ones(i.size(0),dtype=torch.float32,device=torch.device('cuda:0')) for i in target_labels]
                               
                        # target_boxes =box_xyxy_to_xywh( target['boxes']).to('cuda:0').to(torch.float32).get_org_tensors()

                        # self.fill_evaluator(image_ids,target_boxes,target_scores,target_labels)
                        
                        self.fill_evaluator(image_ids,out['pred_boxes'],out['pred_scores'],out['pred_classes'])


        mAP,printable=self.evaluator.evaluation(self.local_rank)
        logger.info(printable)
        
        self.model.train() 

        return mAP         
        
    
    def fill_evaluator(self,image_ids,pred_boxes,pred_scores,pred_class):

        torch.set_printoptions(threshold=1000000)
        if (isinstance(image_ids,list)):
            image_ids=[j.cpu().item() for i in image_ids for j in torch.split(i,i.shape[0])]
            pred_boxes = [j  for i in pred_boxes for j in torch.split(i,i.shape[0])]
            pred_scores = [j for i in pred_scores for j in torch.split(i,i.shape[0])]
            pred_class=[j  for i in pred_class for j in torch.split(i,i.shape[0])]
        else:
            image_ids=image_ids.squeeze(-1).tolist()
            pred_boxes=pred_boxes
            pred_scores=pred_scores
            pred_class=pred_class
        
        
        for idx,image_id in enumerate(image_ids): #iterazione sulla batch size

            bboxes_image,score_image,classes_image = pred_boxes[idx],pred_scores[idx],pred_class[idx]
            
            bboxes_image,score_image,classes_image = self.model.filter_output([torch.tensor(i) for i in [bboxes_image,score_image,classes_image]])


            self.evaluator.add_detections(image_id,classes_image,score_image,bboxes_image)
            # for classe,score,bbox in zip(classes_image,score_image,bboxes_image):
                


    def save_ckpt(self, ckpt_name,epoch):

        if self.rank == 0:

            model_state_dict=(self.model.module if is_parallel(self.model) else self.model).state_dict()
            optimizer_state_dict=self.optimizer.state_dict()
            lr_scheduler_state_dict=self.lr_scheduler.state_dict()

            ckpt_dict={'model':model_state_dict,'optimizer':optimizer_state_dict,'lr_scheduler':lr_scheduler_state_dict,'epoch':epoch}
            logger.info("Save weights to {}".format(self.saving_dir+ckpt_name))

            save_experiment_checkpoint(
                    self.saving_dir+ckpt_name,
                    **ckpt_dict
            )
        







