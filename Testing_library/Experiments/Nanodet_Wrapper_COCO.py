        
from .yolo_base import Exp as MyExp
from .yolo_base import worker_init_reset_seed
from math import ceil
import torch 
import os


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.archi_name = 'NanoDet'
        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 80
        self.data_num_workers = 4
        self.warmup_lr = 0.00007
        
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.000115 / 128
        # factor of model depth
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.input_size = (320,320)
        self.test_size = (320,320)
        self.normalize=True
        self.base_value=114
        self.nmsthre = 0.65
        self.config='/home/bomps/Scrivania/workfile/CNN_TRAINING_REFACTORING/new_version/New_Trainer/NN_Train_test/Models/nanodet/config/nanodet-plus-m_320.yml'
        #self.mean = torch.tensor([0.5, 0.5, 0.5])
        #self.std = torch.tensor([0.5, 0.5, 0.5])
        self.resize_as_tensor=False #difference for using the PIL or the pytorch resize 
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255][::-1])
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255][::-1])
        self.swap =(2,0,1)
        self.test_conf = 0.01
        self.pretrained=False
        self.bgr = True
        
        # name of annotation file for evaluation
        # self.val_dat_dir = "/media/bomps/UB_WIN_SHARED/COCO/val2017"
        # # name of annotation file for testing
        # self.val_ann_dir= "/media/bomps/UB_WIN_SHARED/ILSVRC2015/ILSVRC2015/Annotations/VID/"
        # self.val_annot_file = '/media/bomps/UB_WIN_SHARED/COCO/annotations/instances_val2017.json'
        # self.gt_path = '/media/bomps/UB_WIN_SHARED/COCO/annotations/instances_val2017.json'

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.COCO='COCO_V1'

    def get_model(self):
        from ..Models.nanodet.nanodet.util import  cfg, load_config
        from ..Models.Nanodet_Wrapper import NanoDetPlusWrapper
        if(not hasattr(self,'model')):

            load_config(cfg, self.config)
            print(cfg)
            self.model=NanoDetPlusWrapper(self.multisize,self,cfg=cfg.model)

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
             optimizer.add_param_group(+
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer
            '''
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr
            )
           

        return self.optimizer

    # def get_evaluator(self,saving_dir):
    #     from ..Evaluators.Vid_Coco_Evaluator import COCO_EVAL
    #     from ..Dataset.Imagenet import data_dict,DATASET_NAMES
    #     import torch


    #     self.selected_classes=torch.tensor(list(range(80)))

        

    #     categories = [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]



    #     self.from_coco_to_imgvid = {i:categories[i]['id'] for i in range(len(categories))}


    #     numeral_classes =  [i['id']for i in categories]
        
        
    #     classes=DATASET_NAMES[self.COCO][1:]

        

        

    #     if 'evaluator' not in self.__dict__:

            

    #         self.evaluator = COCO_EVAL(classes,list(self.from_coco_to_imgvid.values()),saving_dir,self.Add_Background,self.gt_path)
        
    #     return self.evaluator


    