
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
from ..My_transforms.functional import resize_bounding_box
from tidecv import TIDE, datasets, Data
from torchvision.ops import nms

import tqdm
import copy
from torchinfo import summary


'''

def table_expansion(k):
        if (isinstance(k,dict)):
            return [(str(i),table_expansion(j)) for i,j in k.items()]     
        else:
            return pprint.pformat(k)
    
if self.rank == 0:
    table_header = ['metrics', "values"]
    exp_table = table_expansion(summary['bbox'])
    table_summary=tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
    logger.info('\n'+table_summary)
'''

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.debug(line.rstrip())

    def flush(self):
        pass

class COCO_EVAL(object):
    def __init__(self,classes,ids,saving_directory,add_background):
        self.saving_directory=saving_directory
        self.categories=[{'id':0,'name':'__background__','supercategory':''}]+[{'id':int(i),'name':str(classes[i-int(add_background)]),'supercategory':''}for i in ids]
        self.license={"id": 3,"name": 'Fittizia',"url": '',}
        self.image_ids=set()
        self.image_ids_gt=set()
        self.images=[]
        self.gts=[]
        self.detections=[]
        self.current_id=0
    def add_image_informations(self,id,width,height,file_name):
        self.image_ids.add(id)
        im={"id": id,"width": width,"height": height,"file_name": file_name,"license": 3}
        self.images.append(im)

    def add_gt_annotations(self,id,classe,bbox):
        #assert id in self.image_ids
        self.image_ids_gt.add(id)
        gt={'id':self.current_id,"image_id": id,"category_id":classe,"area": int(bbox[-1]*bbox[-2]),"iscrowd": 0,"bbox":[int(i) for i in bbox]}
        self.gts.append(gt)
        self.current_id+=1
    def add_detection(self,id,classe,score,bbox):
        #assert id in self.image_ids

        record={}
        record['bbox'] = [int(i) for i in bbox]
        record['category_id']= classe
        record["image_id"] = id
        record['score']=score

        self.detections.append(record)

    def evaluation(self,classe=None):
        import json
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        info={
        "year": 2023,
        "version": 1,
        "description": 'boh',
        "contributor": 'io',
        "url": '',
        "date_created": '',
        }
        ground_truths={
        "info": info,
        "images": self.images,
        "annotations": self.gts,
        "licenses": [self.license],
        "categories":self.categories
        }
        print('faccio il primo')
        json_gts=json.dumps(ground_truths)
        print('faccio il secondo')
        json_dts=json.dumps(self.detections)
        print('scrivo nel file gt')
        if(classe is None):
            fil=open(self.saving_directory+'ground_truth.json','w')
        else:
            fil=open(self.saving_directory+f'ground_truth_{classe}.json','w')
        fil.write(json_gts)
        print('scrivo nel file det')
        if(classe is None):
            fil_b=open(self.saving_directory+'detections.json','w')
        else:
            fil_b=open(self.saving_directory+f'detections_{classe}.json','w')
        fil_b.write(json_dts)
        fil.close()
        fil_b.close()
        cocoGt=COCO(self.saving_directory+'ground_truth.json')
        
        cocoDt=cocoGt.loadRes(self.saving_directory+'detections.json')
        annType  = 'bbox'
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        #cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        cocoEval.summarize()
        precision=cocoEval.eval['absolute_precision']
        recall=cocoEval.eval['recall']
        A_P=[f" \n Average Precision  (AP) @[ IoU={j} | area=   {k} | maxDets=100 ] = {i} \n" for j,k,i in zip(['0.5:0.95','0.5','0.75','0.5:0.95','0.5:0.95','0.5:0.95'],['all','all','all','small','medium','large'],cocoEval.stats[:6])]
        A_R=[f" \n Average Recall     (AR) @[ IoU={j} | area=   {k} | maxDets=  1 ] = {i} \n" for j,k,i in zip(['0.5:0.95','0.5','0.75','0.5:0.95','0.5:0.95','0.5:0.95'],['all','all','all','small','medium','large'],cocoEval.stats[6:])]
        out_pre = [f"output precision {np.mean(precision[0,:,0,2])}"]
        out_recall=[f"output recall {np.mean(recall[0,:,0,2])}"]
        return A_P+A_R+out_pre+out_recall
    

class Trainer:
    def __init__(self, exp, args, val=False):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        #self.prefetcher = vid.DataPrefetcher(train_loader)
        #self.train_loader = train_loader
        
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.val_loader = self.exp.get_eval_loader(self.args.batch_size,self.is_distributed)
        print(self.val_loader)
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        #self.use_model_ema = exp.ema not converted yet
        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        #gc.set_debug(gc.DEBUG_LEAK|gc.DEBUG_STATS)

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.saving_dir=exp.saving_dir
        # evaluator 
        self.evaluator = self.exp.get_evaluator()
        self.bboxes_resizer=self.exp.get_boxes_resizer()

        
        
        if self.rank==0:
            setup_logger(
                self.saving_dir,
                distributed_rank=0,
                filename=self.saving_dir+"train_log.txt",
                mode="a",
            )


        if val and self.exp.multisize:
            self.evaluate_multires()
            return 
        if val:
            self.evaluate()
            return
        

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))
        #setup device
        torch.cuda.set_device(self.local_rank)
        #getting model from experiment
        model = self.exp.get_model()

        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        # value of epoch will be set in `resume_train`
        self.lr_scheduler = self.exp.get_lr_scheduler(self.args.batch_size)
        model = self.resume_train(model)

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        if(self.no_aug):
            logger.info("Augmentation removed from training")

            self.train_loader,self.max_iter = self.exp.get_train_loader(
                self.args.batch_size,self.is_distributed
            )
            
        else:
            augm=self.exp.get_train_transformations()
            self.train_loader,self.max_iter = self.exp.get_train_loader(
                self.args.batch_size,self.is_distributed,transforms=augm
            )
        
        self.prefetcher = DataPrefetcher(self.train_loader)   
        

        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        self.model = model
        self.model.train()


        
        logger.info("Training start...")

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        transform=self.exp.get_train_transformations()
        augment=self.exp.get_train_augmentations()
        if (self.epoch + 1- self.exp.warmup_epochs ) % 4 ==0 \
                and (self.epoch + 1- self.exp.warmup_epochs ) \
                and (self.epoch + 1 < self.max_epoch - self.exp.no_aug_epochs):
            if(self.no_aug):
                idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
                self.train_loader.dataset.remove_transform(idx)
            else:   
                self.train_loader.dataset.add_transform(transform+augment)
            logger.info('Refreshing dataloader')
                
        if self.epoch + 1 >= self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->Removing all Augmentations!")
            logger.info("--->USING L1 Loss")
           
            if('head' in self.model._modules and 'use_l1' in self.model.head.__dict__):
                self.model.head.use_l1=True
            idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
            self.train_loader.dataset.remove_transform(idx)
        elif 0< self.epoch + 1 and  self.epoch + 1 <= self.exp.warmup_epochs:
            logger.info("--->No Augmentation in warmup")
            idx=[i.__class__.__name__ for i in transform]+['Mosaic_Augment']
            self.train_loader.dataset.remove_transform(idx)
        else:
            logger.info("--->Including aug now!")
            self.train_loader.dataset.add_transform(augment)
            self.train_loader.dataset.add_transform(transform)
            logger.info("--->Training full model now!")
            
        self.prefetcher = DataPrefetcher(self.train_loader)

        

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def before_iter(self):
        pass

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        
        
        #with torch.cuda.amp.autocast(enabled=self.amp_training):
        outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        '''
        not implemented yet
        if self.use_model_ema:
            self.ema_model.update(self.model)
        '''
        
        lr = self.lr_scheduler.get_last_lr()[-1]
        

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
    
    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if self.rank==0 and (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            loss_meter = self.meter.get_filtered_meter("classification")
            loss_str += ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            loss_meter = self.meter.get_filtered_meter("bbox_regression")
            loss_str += ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )
            logger.info(
                "{}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str)))

            self.meter.clear_meters()
    
    
    def after_epoch(self):        
        gc.collect()
        synchronize()
        
        self.prefetcher.free()
        
        torch.cuda.empty_cache()
        
        

        self.lr_scheduler.step()
        if self.exp.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            self.evaluate_and_save_model()
    
    def evaluate_and_save_model(self):
        '''
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            OrderedDict([('bbox', {'AP': 21.38509001802317, 'AP50':
        '''
        evalmodel = self.model
        
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module
        evalmodel.eval()
        self.evaluator.reset()
        with torch.no_grad():
            for _, (inputs,originals) in enumerate(self.val_loader):
                outputs=[]
                cuda_inputs=inputs.cuda()

                output=evalmodel(cuda_inputs)


                
                output=self.model.post(output,self.exp.num_classes,self.exp.test_conf,self.exp.nmsthre ,class_agnostic=True)
                
                for output_1_image,width,height in zip(output,originals["width"],originals["height"]):
                    
                    if(output_1_image is None):
                        output_1_image=np.empty((0,7))
                    else:
                        output_1_image=torch.clamp(output_1_image,min=0)
                        output_1_image[...,:4]=self.bboxes_resizer(output_1_image[...,:4],new_size=(height,width))
                        output_1_image=output_1_image.detach().cpu().numpy()
                        
                    
                    
                    outputs+=[output_1_image]
                    

                    
                detectron_originals=[{'labels':an[an.sum(dim=1)!=0][...,0].numpy(),'boxes':an[an.sum(dim=1)!=0][...,1:],"image_id":idx.cpu().item()} for an,idx in zip(originals['annotations'],originals["image_id"])]
                
                outputs_list=[self.model.out_to_detectron(i,(width,height)) for i,width,height in zip(outputs,originals["width"],originals["height"])]
                
                self.evaluator.process(detectron_originals,outputs_list)

        summary=self.evaluator.evaluate()    

        self.model.train()

        ap50_95 = summary['bbox']['AP']
        ap50 = summary['bbox']['AP50']
        '''
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            self.tblogger.add_scalar("lr", self.lr, self.epoch + 1)
        '''
        logger.info('model AP '+ str(ap50_95)+'\n'+' model AP50 '+str(ap50)+'\n')

        synchronize()
        self.save_ckpt("best_epoch", ap50_95 > self.best_ap)
        
        self.best_ap = max(self.best_ap, ap50_95)

    

    def evaluate(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        
        model = self.resume_for_evaluation()
        model = model.to(self.device)
        model = model.to(self.data_type)
        self.model = model
        evalmodel = self.model
        evalmodel=evalmodel

        evalmodel.eval()
        self.evaluator.reset()
        COCO_labels=torch.tensor([data_dict[self.exp.COCO][i]  for i in data_dict[self.exp.COCO]])
        
        print(f'in trainer classes {COCO_labels}')
        
        tide = TIDE()
        gt_data = Data('gt_data')
        det_data= Data('det_data')
        with torch.no_grad():
            for indice, (inputs,originals) in enumerate(self.val_loader):
                
                outputs=[]
                if(len(inputs.shape)>4):
                    inputs=inputs.squeeze()
                inputs=inputs.to(self.data_type)
                if (isinstance (originals,list)):
                    originals={key: torch.stack([i[key] for i in originals],dim=0).squeeze() for key in originals[0] if key !='file_name'}

                

                for idx,inp in enumerate(inputs):
                    
                    inp=inp.unsqueeze(0)
                    output=evalmodel(inp.to(self.device))

                    output=self.model.post(output,self.exp.num_classes,self.exp.test_conf,self.exp.nmsthre ,class_agnostic=True)
                    
                    for output_1_image in output:
                        width,height= originals["width"][idx],originals["height"][idx]
                        if(output_1_image is None):
                            output_1_image=np.empty((0,7))
                        else:
                            #v=torch.clamp(output_1_image,min=0,max=max)
                            
                            
                            #output_1_image[...,:4]=self.bboxes_resizer(output_1_image[...,:4],new_size=(height,width))
                            #
                            # print(originals['resize_factor'])
                            # input()
                            output_1_image[...,:4]/=originals['resize_factor']#resize_bounding_box(output_1_image[:,:4],self.exp.test_size[::-1],(height,width))

                            if (not isinstance(evalmodel,NN_Augmented)):
                                if (output_1_image.shape[-1]==6):
                                    selected=torch.isin(output_1_image[:,5],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                    # nms_selected=nms(output_1_image[...,:4],(output_1_image[...,4]),0.3)
                                    # output_1_image=output_1_image[nms_selected]
                                elif (output_1_image.shape[-1]==7):
                                    selected=torch.isin(output_1_image[:,6],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                    # nms_selected=nms(output_1_image[...,:4],(output_1_image[...,4]*output_1_image[...,5]),0.3)
                                    # output_1_image=output_1_image[nms_selected]
                            else:
                                selected=torch.isin(output_1_image[:,5],COCO_labels.to(output_1_image.device))
                                output_1_image=output_1_image[selected]
                            
                            output_1_image=output_1_image.detach().cpu().numpy()
                            
                            output_1_image=output_1_image
                            
                           
                            
                        
                        outputs+=[output_1_image]
                
                if (isinstance(evalmodel,NN_Augmented)):
                    evalmodel.clear()
                
                 
                detectron_originals=[{'labels':an[an.sum(dim=1)!=0][...,0].numpy(),'boxes':an[an.sum(dim=1)!=0][...,1:],"image_id":idx.cpu().item()} for an,idx in zip(originals['annotations'],originals["image_id"])]
                
                
                outputs_list=[self.model.out_to_detectron(i,(width,height)) for i,width,height in zip(outputs,originals["width"],originals["height"])]

                for gts,detect in zip(detectron_originals,outputs_list):

                    for lab,gt_box in zip(gts['labels'],gts['boxes']):
                        gt_data.add_ground_truth(gts['image_id'],int(lab),list(gt_box),None)
                    
                    for box,scores,clas in zip(detect['instances'].get('pred_boxes'),detect['instances'].get('scores'),detect['instances'].get('pred_classes')):
                        det_data.add_detection(gts['image_id'],int(clas),float(scores),list(box),None)

                
                
                
                #self.evaluator.process(detectron_originals,outputs_list)
                
                #ann_ids = self.evaluator._coco_api.getAnnIds(imgIds=detectron_originals[0]["image_id"])
                #anno = self.evaluator._coco_api.loadAnns(ann_ids)




        tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
        tide.summarize()
        tide.plot(self.saving_dir)
        for run_name, run in tide.runs.items():
            ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh*100))
            logger.info('\n{}: {:.2f}'.format(ap_title, run.ap))
        #summary=self.evaluator.evaluate()
        #print(summary)

        
        synchronize()
        return None
    
    def evaluate_multires(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        
        model = self.resume_for_evaluation()
        model = model.to(self.device)
        model = model.to(self.data_type)
        self.model = model
        evalmodel = self.model
        evalmodel.eval()

        # summary(evalmodel,input_size=[1,3,320,320])
        # input()
        
        # first_resizer=self.exp. _get_resizer_normalization()[0]
        # second_resizer=self.exp._get_second_resize()
        COCO_labels=torch.tensor([data_dict[self.exp.COCO][i] - int(not self.exp.Add_Background)  for i in data_dict[self.exp.COCO]])
        print(COCO_labels)
        self.coco_eval=COCO_EVAL(classes=DATASET_NAMES[self.exp.COCO][1:],ids=COCO_labels,saving_directory=self.saving_dir,add_background=self.exp.Add_Background)
        
        tide = TIDE()
        gt_data = Data('gt_data')
        det_data= Data('det_data')
       
        previous_index=None
        current_index=None
        taken_labels=set()
        # if 'sequence' in self.exp.__dict__ and self.exp.sequence:
        #     from ..Models.TransVOD_Lite_2.util.misc_multi import NestedTensor
        # else:
        #     from ..Models.TransVOD_plusplus.util.misc import NestedTensor
        with torch.no_grad():
            # gt_data_local = Data('gt_data_local')
            # det_data_local= Data('det_data_local')
            older_input=[]
            current_input=[]
            older_ids=[]
            current_ids=[]
            left_out_im=None 
            left_out_idx=None
            left_out_size=None
            current_sizes=[]
            older_sizes=[]
            for indice, (inputs,originals) in tqdm.tqdm(enumerate(self.val_loader)):
                
                
                if(previous_index is None):
                    video_name=originals['file_name'][0].split('/')[-2]
                current_index=originals['vid_index']
                if(previous_index is not None and previous_index !=current_index):
                    if(isinstance(evalmodel,NN_Augmented)):
                        print('cleared')
                        
                        evalmodel=evalmodel.clear()
                    
                    # tide.evaluate(gt_data_local, det_data_local, mode=TIDE.BOX)
                    # tide.summarize()
                    # for run_name, run in tide.runs.items():
                    #     ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh*100))
                    #     logger.info('\n\n video name {} {}: {:.2f}'.format(video_name,ap_title, run.ap))
                    
                    # video_name=originals['file_name'][0].split('/')[-2]
                    # gt_data_local = Data('gt_data_local')
                    # det_data_local= Data('det_data_local')
                    # tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
                    # tide.summarize()
                    
                    # for run_name, run in tide.runs.items():
                    #     ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh*100))
                    #     logger.info('\n\n cumulative {}: {:.2f}'.format(ap_title, run.ap))

                    
                outputs=[]
                if(len(inputs.shape)>4):
                    inputs=inputs.squeeze()
                inputs=inputs.to(self.data_type)
                size=inputs.shape[-2:]
                if (isinstance (originals,list)):
                    originals={key: torch.stack([i[key] for i in originals],dim=0).squeeze() for key in originals[0] if key !='file_name'}

                detectron_originals=[{'labels':an[an.sum(dim=1)!=0][...,0].numpy(),'boxes':an[an.sum(dim=1)!=0][...,1:],"image_id":idx.cpu().item()} for an,idx in zip(originals['annotations'],originals["image_id"])]
                for gts in detectron_originals:
                    for lab,gt_box in zip(gts['labels'],gts['boxes']):
                        
                        
                        gt_box[2:]=gt_box[2:]-gt_box[:2]
                        # print(f'sono groundtruth box {gt_box} {lab}')
                        self.coco_eval.add_gt_annotations(int(gts['image_id']),int(lab),list(gt_box))
                        # print(lab)
                        gt_data.add_ground_truth(int(gts['image_id']),int(lab),list(gt_box),None)
                        # gt_data_local.add_ground_truth(gts['image_id'],int(lab),list(gt_box),None)
                        # input()
                
                for idx,inp in enumerate(inputs):

                    width,height= int(originals["width"][idx]),int(originals["height"][idx])
                    
                    image_id=int(originals["image_id"][idx])
                    file_name=str(originals['file_name'][idx])

                    # print(type(width),type(height),type(image_id),type(file_name))
                    # an=originals['annotations'][idx]
                    
                    # for gt_lab in self.coco_evaluator:
                        
                    self.coco_eval.add_image_informations(image_id,width,height,file_name)

                    if ('sequence' in self.exp.__dict__):
                        
                        if( self.exp.sequence and (previous_index ==current_index or previous_index is None) and len(current_input)<self.exp.num_frames):
                            current_input.append(inp)
                            current_ids.append(image_id)
                            current_sizes.append((width,height))
                            
                        else:
                            
                            if(previous_index!=current_index and len(current_input)<self.exp.num_frames):
                                left_out_im=inp
                                left_out_idx=image_id
                                left_out_size=(width,height)
                                current_input=older_input[-(self.exp.num_frames-len(current_input)):]+current_input
                                
                                current_ids=older_ids[-(self.exp.num_frames-len(current_ids)):]+current_ids

                                
                                current_sizes=older_sizes[-(self.exp.num_frames-len(current_sizes)):]+current_sizes
                                

                        

                        if len(current_input)<self.exp.num_frames:
                            continue
                    
                    # if(counter_resize!=0):
                    #     inp,resized_originals=first_resizer(inp,{ key:originals[key][idx]for key in originals})
                    #     for i in resized_originals:
                    #         originals[i][idx]=resized_originals[i]
                    # else:
                    #     inp,resized_originals=second_resizer(inp,{ key:originals[key][idx]for key in originals})
                    #     for i in resized_originals:
                    #         originals[i][idx]=resized_originals[i]

                    

                    # if self.exp.masked:
                    #     if 'sequence' in self.exp.__dict__ and self.exp.sequence:
                            
                    #             masks=torch.stack([torch.zeros(( i.shape[1],i.shape[2]), dtype=torch.bool) for i in current_input],dim=0)
                                
                    #             inp=NestedTensor(torch.stack(current_input,dim=0),masks)
                    #     else:
                    #         if 'mask' in originals: 
                                
                    #             mask=originals['mask']
                    #             # print(f'sono inp {type(inp)}')
                    #             my_inp=NestedTensor(inp.unsqueeze(0),mask)
                    #             inp=my_inp
                    # else:
                    inp=inp.unsqueeze(0)
                    
                    output=evalmodel(inp.to(self.device))
                    
                    
                    # print(f'sono output {output}')
                    # input()
                    output=self.model.post(output,self.exp.num_classes,self.exp.test_conf,self.exp.nmsthre ,class_agnostic=self.exp.class_agnostic)
                    # print(f'sono output dopo post {output}')
                    # input()

                    # print('output',output)
                    # print('ground truth',originals[])

                    for output_1_image in output:
                        if(output_1_image is None):
                            output_1_image=np.empty((0,7))
                        else:
                            
                            # if (output_1_image.shape[-1]==6):
                            #     output_1_image[:,5]+= int(self.exp.Add_Background)
                            #     for label_to_check in output_1_image[:,5]:
                            #         taken_labels.add(label_to_check.item())
                            # elif (output_1_image.shape[-1]==7):
                            #     output_1_image[:,6]+= int(self.exp.Add_Background)
                            #     for label_to_check in output_1_image[:,6]:
                            #         taken_labels.add(label_to_check.item())
                            
                            #v=torch.clamp(output_1_image,min=0,max=max)
                            
                            
                            #output_1_image[...,:4]=self.bboxes_resizer(output_1_image[...,:4],new_size=(height,width))
                            if (not isinstance(evalmodel,NN_Augmented)):
                                # print(f'width:{ width} height:{height}')
                                # print(f"resize factor {originals['resize_factor']}")
                                # print(f'size before {output_1_image[...,:4]}')
                                output_1_image[...,:4]/=originals['resize_factor'].to(output_1_image.device)
                                # print(f'size after {output_1_image[...,:4]}')
                                # input()
                            else:
                                if(evalmodel.working_size is not None):
                                    new_width,new_height=evalmodel.working_size
                                    r=min([new_width / width, new_height / height])
                                    
                                    output_1_image[...,:4]/=r
                            
                            if (not isinstance(evalmodel,NN_Augmented)):
                                if (output_1_image.shape[-1]==6):
                                    selected=torch.isin(output_1_image[:,5],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                    # nms_selected=nms(output_1_image[...,:4],(output_1_image[...,4]),0.3)
                                    # output_1_image=output_1_image[nms_selected]
                                elif (output_1_image.shape[-1]==7):
                                    selected=torch.isin(output_1_image[:,6],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                    # nms_selected=nms(output_1_image[...,:4],(output_1_image[...,4]*output_1_image[...,5]),0.3)
                                    # output_1_image=output_1_image[nms_selected]
                            #sel=np.isin(output_1_image[...,5],COCO_labels)
                            #output_1_image=output_1_image[sel]
                            # else:
                                
                            #     selected=torch.isin(output_1_image[:,5],COCO_labels.to(output_1_image.device))
                            #     output_1_image=output_1_image[selected]
                            output_1_image=output_1_image.detach().cpu().numpy()
                        outputs+=[output_1_image]
                    
                           
                            
                        
                
                
                   
                
                # if (isinstance(evalmodel,NN_Augmented)):
                    #print(size)
                    # originals['annotations'][...,1:]=resize_bounding_box(originals['annotations'][...,1:],size,self.exp.test_size)  
                    #print([an[an.sum(dim=1)!=0] for an in originals['annotations']])


                

                
                #print('originals',detectron_originals)
                
                
                
                if(len(outputs)!=0):
                    

                    if('sequence' in self.exp.__dict__ and self.exp.sequence):
                        originals["width"]=[i[0] for i,ids in zip(current_sizes,current_ids) if (ids not in older_ids) ]
                        originals["height"]=[i[1] for i,ids in zip(current_sizes,current_ids) if (ids not in older_ids) ]
                        outputs = [i for i,ids in zip(outputs,current_ids) if (ids not in older_ids) ]
                        originals['image_id']=[i for i in current_ids if (i not in older_ids)]

                        # print(f"sono image ids {originals['image_id']}")
                    
                        if(len(current_input)==self.exp.num_frames):
                                older_input=copy.deepcopy(current_input)
                                older_ids=copy.deepcopy(current_ids)

                                               
                    outputs_list=[self.model.out_to_detectron(i,(width,height)) for i,width,height in zip(outputs,originals["width"],originals["height"])]
                    # print(f'final outputs lists{outputs_list}')
                    # input()
                    
                    for detect,ids in zip(outputs_list,originals['image_id']):
                        
                        for box,scores,clas in zip(detect['instances'].get('pred_boxes'),detect['instances'].get('scores'),detect['instances'].get('pred_classes')):
                            
                            box[2:]=box[2:]-box[:2]
                            det_data.add_detection(int(ids),int(clas),float(scores),list(box),None)
                            self.coco_eval.add_detection(int(ids),int(clas),float(scores),list(box))
                            # print(f'sono prediction box {box} {clas}')
                            # input()
                            # if(int(gts['image_id']) in self.coco_evaluator[int(clas)].image_ids_gt):
                            #     self.coco_evaluator[int(clas)].add_detection(int(gts['image_id']),int(clas),float(scores),list(gt_box))
                            # else:
                            #     added=False
                            #     for label in self.coco_evaluator:
                            #         if(int(gts['image_id']) in self.coco_evaluator[label].image_ids_gt):
                            #             added=True
                            #             self.coco_evaluator[label].add_detection(int(gts['image_id']),int(clas),float(scores),list(gt_box))
                            #             break
                            #     if(not added):
                            #         r_label=random.choice(COCO_labels)
                            #         self.coco_evaluator[int(r_label)].add_detection(int(gts['image_id']),int(clas),float(scores),list(gt_box))
                    if 'sequence' in self.exp.__dict__:
                        if previous_index!=current_index and previous_index is not None:
                            current_input=[left_out_im]
                            current_ids=[left_out_idx]
                            current_sizes=[left_out_size]
                        else:
                            current_input=[]
                            current_ids=[]
                            current_sizes=[]
                    previous_index=current_index             
                                

                            
                        # input()
                        # if(len(detect['instances'])==0):
                            
                        #     for lab,gt_box in zip(gts['labels'],gts['boxes']):
                        #         det_data.add_detection(gts['image_id'],int(0),float(0),[0,0,0,0],None)
                        #         det_data_local.add_detection(gts['image_id'],int(0),float(0),[0,0,0,0],None)
                    
                    
                    
                    #self.evaluator.process(detectron_originals,outputs_list)
                    
                    #ann_ids = self.evaluator._coco_api.getAnnIds(imgIds=detectron_originals[0]["image_id"])
                    #anno = self.evaluator._coco_api.loadAnns(ann_ids)
        logger.info(f'All detected labels {taken_labels}')
        results=self.coco_eval.evaluation()
        for i in results:
            logger.info(f'Accumulated results {i}')
        # total=0
        # for i in self.coco_evaluator:
        #     AP_Value=self.coco_evaluator[i].evaluation(i)
        #     scritta_AP_singolo=f'classe {i} {AP_Value[1]}'
        #     total+=AP_Value[-1]
        #     logger.info('\n'+scritta_AP_singolo)
        # total=total/len(self.coco_evaluator)   
        # logger.info('\n totale AP_50 '+str(total))
        tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
        tide.summarize()
        # tide.plot(self.saving_dir)
        for run_name, run in tide.runs.items():
            ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh*100))
            logger.info('\n{}: {:.2f}'.format(ap_title, run.ap))
        #summary=self.evaluator.evaluate()
        #print(summary)

       
        return None

    def resume_for_evaluation(self):
        logger.info("loading checkpoint for evaluation")
        ckpt_file = self.args.ckpt
        
        self.start_epoch = 0
        
        model=self.exp.get_model()
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
        
        
        return model

    def resume_train(self,model):
        
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt == '':
                ckpt_file = self.saving_dir+"latest" + "_ckpt.pth"
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)

            # resume the model/optimizer state dict
            load_checkpoint(ckpt,self.exp)
            
            self.model=self.exp.get_model()
            self.optimizer=self.exp.get_optimizer(self.args.batch_size)
            self.lr_scheduler=self.exp.get_lr_scheduler(self.args.batch_size)
            
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["epoch"]  )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt != '':
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)
                load_checkpoint(ckpt,self.exp)
            self.start_epoch = 0
        
        self.model=self.exp.get_model()
        
        

        return self.model
    
        
    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap)
        )
    def save_ckpt(self, ckpt_name, update_best_ckpt=False):

        if self.rank == 0:

            model_state_dict=(self.model.module if is_parallel(self.model) else self.model).state_dict()
            optimizer_state_dict=self.optimizer.state_dict()
            lr_scheduler_state_dict=self.lr_scheduler.state_dict()
            ckpt_dict={'model':model_state_dict,'optimizer':optimizer_state_dict,'lr_scheduler':lr_scheduler_state_dict,'epoch':self.epoch}
            logger.info("Save weights to {}".format(self.saving_dir+ckpt_name))
            if((not ckpt_name=="best_epoch" ) or  update_best_ckpt):
                save_experiment_checkpoint(
                    self.saving_dir+ckpt_name,
                    **ckpt_dict
            )
            

    

    


    

    
        


