#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from .yolo_base import Exp as MyExp


# COCO V1 category list — used for class mapping and evaluator setup.
# fmt: off
COCO_V1_CATEGORIES = [
    {"supercategory": "person",      "id": 1,  "name": "person"},
    {"supercategory": "vehicle",     "id": 2,  "name": "bicycle"},
    {"supercategory": "vehicle",     "id": 3,  "name": "car"},
    {"supercategory": "vehicle",     "id": 4,  "name": "motorcycle"},
    {"supercategory": "vehicle",     "id": 5,  "name": "airplane"},
    {"supercategory": "vehicle",     "id": 6,  "name": "bus"},
    {"supercategory": "vehicle",     "id": 7,  "name": "train"},
    {"supercategory": "vehicle",     "id": 8,  "name": "truck"},
    {"supercategory": "vehicle",     "id": 9,  "name": "boat"},
    {"supercategory": "outdoor",     "id": 10, "name": "traffic light"},
    {"supercategory": "outdoor",     "id": 11, "name": "fire hydrant"},
    {"supercategory": "outdoor",     "id": 13, "name": "stop sign"},
    {"supercategory": "outdoor",     "id": 14, "name": "parking meter"},
    {"supercategory": "outdoor",     "id": 15, "name": "bench"},
    {"supercategory": "animal",      "id": 16, "name": "bird"},
    {"supercategory": "animal",      "id": 17, "name": "cat"},
    {"supercategory": "animal",      "id": 18, "name": "dog"},
    {"supercategory": "animal",      "id": 19, "name": "horse"},
    {"supercategory": "animal",      "id": 20, "name": "sheep"},
    {"supercategory": "animal",      "id": 21, "name": "cow"},
    {"supercategory": "animal",      "id": 22, "name": "elephant"},
    {"supercategory": "animal",      "id": 23, "name": "bear"},
    {"supercategory": "animal",      "id": 24, "name": "zebra"},
    {"supercategory": "animal",      "id": 25, "name": "giraffe"},
    {"supercategory": "accessory",   "id": 27, "name": "backpack"},
    {"supercategory": "accessory",   "id": 28, "name": "umbrella"},
    {"supercategory": "accessory",   "id": 31, "name": "handbag"},
    {"supercategory": "accessory",   "id": 32, "name": "tie"},
    {"supercategory": "accessory",   "id": 33, "name": "suitcase"},
    {"supercategory": "sports",      "id": 34, "name": "frisbee"},
    {"supercategory": "sports",      "id": 35, "name": "skis"},
    {"supercategory": "sports",      "id": 36, "name": "snowboard"},
    {"supercategory": "sports",      "id": 37, "name": "sports ball"},
    {"supercategory": "sports",      "id": 38, "name": "kite"},
    {"supercategory": "sports",      "id": 39, "name": "baseball bat"},
    {"supercategory": "sports",      "id": 40, "name": "baseball glove"},
    {"supercategory": "sports",      "id": 41, "name": "skateboard"},
    {"supercategory": "sports",      "id": 42, "name": "surfboard"},
    {"supercategory": "sports",      "id": 43, "name": "tennis racket"},
    {"supercategory": "kitchen",     "id": 44, "name": "bottle"},
    {"supercategory": "kitchen",     "id": 46, "name": "wine glass"},
    {"supercategory": "kitchen",     "id": 47, "name": "cup"},
    {"supercategory": "kitchen",     "id": 48, "name": "fork"},
    {"supercategory": "kitchen",     "id": 49, "name": "knife"},
    {"supercategory": "kitchen",     "id": 50, "name": "spoon"},
    {"supercategory": "kitchen",     "id": 51, "name": "bowl"},
    {"supercategory": "food",        "id": 52, "name": "banana"},
    {"supercategory": "food",        "id": 53, "name": "apple"},
    {"supercategory": "food",        "id": 54, "name": "sandwich"},
    {"supercategory": "food",        "id": 55, "name": "orange"},
    {"supercategory": "food",        "id": 56, "name": "broccoli"},
    {"supercategory": "food",        "id": 57, "name": "carrot"},
    {"supercategory": "food",        "id": 58, "name": "hot dog"},
    {"supercategory": "food",        "id": 59, "name": "pizza"},
    {"supercategory": "food",        "id": 60, "name": "donut"},
    {"supercategory": "food",        "id": 61, "name": "cake"},
    {"supercategory": "furniture",   "id": 62, "name": "chair"},
    {"supercategory": "furniture",   "id": 63, "name": "couch"},
    {"supercategory": "furniture",   "id": 64, "name": "potted plant"},
    {"supercategory": "furniture",   "id": 65, "name": "bed"},
    {"supercategory": "furniture",   "id": 67, "name": "dining table"},
    {"supercategory": "furniture",   "id": 70, "name": "toilet"},
    {"supercategory": "electronic",  "id": 72, "name": "tv"},
    {"supercategory": "electronic",  "id": 73, "name": "laptop"},
    {"supercategory": "electronic",  "id": 74, "name": "mouse"},
    {"supercategory": "electronic",  "id": 75, "name": "remote"},
    {"supercategory": "electronic",  "id": 76, "name": "keyboard"},
    {"supercategory": "electronic",  "id": 77, "name": "cell phone"},
    {"supercategory": "appliance",   "id": 78, "name": "microwave"},
    {"supercategory": "appliance",   "id": 79, "name": "oven"},
    {"supercategory": "appliance",   "id": 80, "name": "toaster"},
    {"supercategory": "appliance",   "id": 81, "name": "sink"},
    {"supercategory": "appliance",   "id": 82, "name": "refrigerator"},
    {"supercategory": "indoor",      "id": 84, "name": "book"},
    {"supercategory": "indoor",      "id": 85, "name": "clock"},
    {"supercategory": "indoor",      "id": 86, "name": "vase"},
    {"supercategory": "indoor",      "id": 87, "name": "scissors"},
    {"supercategory": "indoor",      "id": 88, "name": "teddy bear"},
    {"supercategory": "indoor",      "id": 89, "name": "hair drier"},
    {"supercategory": "indoor",      "id": 90, "name": "toothbrush"},
]
# fmt: on


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 80
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.multiscale_range = 0
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (192, 192)
        self.mosaic_prob = 0.5

        self.data_num_workers = 2

        self.no_aug_epochs = 20
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.seed = None

        self.bgr = True
        self.normalize = False
        self.bgr = True
        self.base_value = 114
        self.xyxy = False

        self.resize_as_tensor = False
        self.backbone_name = 'efficientvit_b0.r224_in1k'
        self.pretrained_backbone = True
        self.COCO = 'COCO_V1'

        self.min_hits = 2
        self.lenght_track = 5
        self.test_conf = 0.55
        self.minimum_threshold = 0.1
        self.iou_threshold = 0.3
        self.rescoring = True

        self.nmsthre = 0.45
        self.low_nmsthre = 0.45

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

            backbone = YOLOPAFPN_EFVT(self.backbone_name, self.pretrained_backbone, depth=self.depth, depthwise=True, act=self.act)

            input_feature = [int(round(i / self.width)) for i in backbone.in_channels]
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=input_feature, strides=backbone.strides,
                act=self.act, depthwise=True,
            )

            head.use_l1 = True

            self.model = YOLOX(self.multisize, self, backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()

        return self.model
