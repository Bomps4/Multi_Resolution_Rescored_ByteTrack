from .yolo_base import Exp as MyExp
from torch import nn
import torch
import os 
from loguru import logger
from NN_Train_test.Models.Heads.yolo_head import   YOLOXHead
from NN_Train_test.Models.Heads.yolo_pafpn import YOLOPAFPN
from NN_Train_test.Models.yolox import YOLOX



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 30
        self.depth = 0.33
        self.width = 0.25
        self.data_num_workers=8
        self.input_size = (320, 256)
        self.max_epoch = 20
        # minimum learning rate during warmup
        self.warmup_lr = 0.0008
        
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.0007 / 64
        #self.random_size = (10, 20)
        #self.mosaic_scale = (0.5, 1.5)
        self.test_size = (320, 256)
        self.normalize=False
        self.resize_as_tensor=False
        self.test_conf = 0.3
        self.pretrained=False
        #self.mosaic_prob = 0.5
        #self.enable_mixup = False
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,conf_thre=self.test_conf,nms_thre=self.nmsthre, 
                act=self.act, depthwise=True
            )
            
            if self.pretrained:
                backbone = YOLOPAFPN(
                    self.depth, self.width, in_channels=in_channels,
                    act=self.act, depthwise=True,
                )
                head2 = YOLOXHead(
                    80, self.width, in_channels=in_channels,conf_thre=self.test_conf,nms_thre=self.nmsthre, 
                    act=self.act, depthwise=True
                )
                cut_model=YOLOX(backbone,head2)
                
                weight_dict=torch.load(self.pretrained_file)
                cut_model.load_state_dict(weight_dict['model'])
                backbone=cut_model.backbone
            self.model = YOLOX(backbone, head)
            
            if self.pretrained:
                self.model.head.apply(init_yolo)
            else:
                self.model.apply(init_yolo)
            self.model.head.initialize_biases(1e-2)

        return self.model
    
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size
            '''
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
            questi vanno dopo aver creato l'optimizer
             optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer
            '''
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr,amsgrad=True
            )
           

        return self.optimizer
    
    