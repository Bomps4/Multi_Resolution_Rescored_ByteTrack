
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
from .COCO_EVAL import COCO_EVAL
import tqdm
import copy
from torchinfo import summary




class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.debug(line.rstrip())

    def flush(self):
        pass
    

class Evaluator:
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
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        
        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.saving_dir=exp.saving_dir
        # evaluator 
       
        
        
        if self.rank==0:
            setup_logger(
                self.saving_dir,
                distributed_rank=0,
                filename=self.saving_dir+"train_log.txt",
                mode="a",
            )


        if val :
            self.evaluate_multires()
            return 
        

   
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

        
        COCO_labels=torch.tensor([data_dict[self.exp.COCO][i] - int(not self.exp.Add_Background)  for i in data_dict[self.exp.COCO]])
        self.coco_eval=COCO_EVAL(classes=DATASET_NAMES[self.exp.COCO][1:],ids=COCO_labels,saving_directory=self.saving_dir,add_background=self.exp.Add_Background)
        
        tide = TIDE()
        gt_data = Data('gt_data')
        det_data= Data('det_data')
       
        previous_index=None
        current_index=None
        taken_labels=set()
     
        with torch.no_grad():
            
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
                        evalmodel=evalmodel.clear()

                    
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
                        
                        self.coco_eval.add_gt_annotations(int(gts['image_id']),int(lab),list(gt_box))
                        
                        gt_data.add_ground_truth(int(gts['image_id']),int(lab),list(gt_box),None)
                        
                
                for idx,inp in enumerate(inputs):

                    width,height= int(originals["width"][idx]),int(originals["height"][idx])
                    
                    image_id=int(originals["image_id"][idx])
                    file_name=str(originals['file_name'][idx])
                        
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
                    
                    
                    inp=inp.unsqueeze(0)
                    
                    output=evalmodel(inp.to(self.device))
                    
    
                    output=self.model.post(output,self.exp.num_classes,self.exp.test_conf,self.exp.nmsthre ,class_agnostic=self.exp.class_agnostic)


                    for output_1_image in output:
                        if(output_1_image is None):
                            output_1_image=np.empty((0,7))
                        else:
                            
                           
                            if (not isinstance(evalmodel,NN_Augmented)):

                                output_1_image[...,:4]/=originals['resize_factor'].to(output_1_image.device)

                            else:
                                if(evalmodel.working_size is not None):
                                    new_width,new_height=evalmodel.working_size
                                    r=min([new_width / width, new_height / height])
                                    
                                    output_1_image[...,:4]/=r
                            
                            if (not isinstance(evalmodel,NN_Augmented)):
                                if (output_1_image.shape[-1]==6):
                                    selected=torch.isin(output_1_image[:,5],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                   

                                elif (output_1_image.shape[-1]==7):
                                    selected=torch.isin(output_1_image[:,6],COCO_labels.to(output_1_image.device))
                                    output_1_image=output_1_image[selected]
                                   
                            output_1_image=output_1_image.detach().cpu().numpy()
                        outputs+=[output_1_image]

                             
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
          
        logger.info(f'All detected labels {taken_labels}')
        results=self.coco_eval.evaluation()
        for i in results:
            logger.info(f'Accumulated results {i}')
       
        tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
        tide.summarize()
        # tide.plot(self.saving_dir)
        for run_name, run in tide.runs.items():
            ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh*100))
            logger.info('\n{}: {:.2f}'.format(ap_title, run.ap))

       
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

    
            

    

    


    

    
        


