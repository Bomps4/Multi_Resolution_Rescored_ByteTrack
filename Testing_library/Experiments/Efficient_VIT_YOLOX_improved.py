#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn


from .yolo_base import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 80
        self.width = 0.25 #same parameter as yoloX Nano
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.multiscale_range = 0
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (320, 320)
        self.mosaic_prob = 0.5

        self.data_num_workers = 2

        self.no_aug_epochs = 20
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
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0
        self.seed = None #NOT SEEDING TRAINING only used for debug (will come to it later)ù

        self.bgr=True
        self.normalize = False
        self.bgr = True 
        self.base_value=114
        self.xyxy=False

        self.resize_as_tensor=False
        self.backbone_name = 'efficientvit_b0.r224_in1k'
        self.pretrained_backbone = True 
        self.COCO='COCO_V1'

        self.nmsthre = 0.6 

        # name of annotation file for evaluation
        # self.val_dat_dir = "/media/bomps/UB_WIN_SHARED/COCO/val2017"
        # # name of annotation file for testing
        # self.val_ann_dir= "/media/bomps/UB_WIN_SHARED/ILSVRC2015/ILSVRC2015/Annotations/VID/"
        # self.val_annot_file = '/media/bomps/UB_WIN_SHARED/COCO/annotations/instances_val2017.json'
        # self.gt_path = '/media/bomps/UB_WIN_SHARED/COCO/annotations/instances_val2017.json'
            
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
            
        if "model" not in self.__dict__:
            from ..Models.Heads.yolo_pafpn_EFViT_BACK import YOLOPAFPN_EFVT
            from ..Models.Heads.yolo_head import YOLOXHead
            from ..Models.yolox import YOLOX
            # from ..yolox.models import EfficientViTSSDLite
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN_EFVT(self.backbone_name,self.pretrained_backbone,depth=self.depth,depthwise=True,act=self.act)
            
            input_feature  = [int(round(i/self.width)) for i in backbone.in_channels]
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=input_feature,strides=backbone.strides,
                act=self.act, depthwise=True)

            head.use_l1 = True

            self.model = YOLOX(self.multisize,self,backbone, head)

        

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()

        return self.model
    
    # def get_evaluator(self,saving_dir):
    #     from ..Evaluators.Vid_Coco_Evaluator import COCO_EVAL
    #     from ..Dataset.Imagenet import data_dict,DATASET_NAMES
    #     import torch


    #     self.selected_classes=torch.tensor(list(range(80)))

    #     print(self.selected_classes)

    #     categories = [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]



    #     self.from_coco_to_imgvid = {i:categories[i]['id'] for i in range(len(categories))}


    #     numeral_classes =  [i['id']for i in categories]
        
    #     classes=DATASET_NAMES[self.COCO][1:]

        

        

    #     if 'evaluator' not in self.__dict__:

            

    #         self.evaluator = COCO_EVAL(classes,numeral_classes,saving_dir,self.Add_Background,self.gt_path)
        
    #     return self.evaluator
