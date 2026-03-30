#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from .yolo_base import Exp as MyExp
import torch 

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.normalize=False
        self.test_conf = 0.001
        self.input_size = (576, 576)
        self.test_size = (352, 352)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.swap = (2,0,1)
        self.bgr = True
        self.data_num_workers = 4
        self.nmsthre = 0.5
        self.resize_as_tensor=False
        self.xyxy=False
        self.base_value=114
    
    




