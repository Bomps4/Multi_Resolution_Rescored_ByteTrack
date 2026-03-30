from .yolo_nano_COCO_multisize import Exp as My_Exp
from ..Dataset.Imagenet import data_dict 
import os 
class Exp(My_Exp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'BYTES_YOLOX'
        self.input_size = (320, 320)
        self.test_size = (320, 320)
        self.reduced_size = (192, 192)
        self.resize_frequency=11
        self.rescoring=True
        self.bgr=False
        self.class_agnostic=True
        self.nmsthre=0.3
        self.test_conf=0.25
        self.minimum_threshold=0.05
        self.iou_threshold=0.3 #intersection between detection in multiple frames to 
        self.lenght_track=5  #number of frames for which a track remain active
        self.min_hits=2 #minimum number of associations before a track is followed.
        self.seq_lenght=-1 #meaning take the video full lenght
        self.data_num_workers = 0
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    def get_model(self,selected_classes=None):
        from ..Models.Postprocess.Bytes_postprocess import BYTES
        from ..Models.NN_Bytes import NN_Augmented
        NN_model=super().get_model()
        if (selected_classes is None):
            selected_classes=[ data_dict[self.COCO][i] - int(not self.Add_Background) for i in data_dict[self.COCO]]
        
        PP_method=BYTES(self.lenght_track,self.min_hits,iou_threshold=self.iou_threshold,min_score=self.test_conf,rescoring=self.rescoring)
        return NN_Augmented(NN_model,PP_method,self.minimum_threshold,self.nmsthre,self.num_classes,selected_classes,self.test_size)
    
    
    
        
