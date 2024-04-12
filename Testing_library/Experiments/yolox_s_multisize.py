#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2021-2022 Megvii Inc. and its affiliates. 
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

import os

from .yolo_base_multisize import Exp as MyExp
import torch 

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.normalize=False
        self.nmsthre = 0.5
        self.test_conf = 0.45
        self.resize_frequency=7
        self.test_size = (576, 576)
        self.reduced_size = (346, 346)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.Add_Background=False
        self.normalize=False
        self.resize_as_tensor=False
        self.xyxy=False
        self.base_value=114




