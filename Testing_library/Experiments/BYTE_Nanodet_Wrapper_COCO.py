from .Nanodet_Wrapper_COCO import Exp as My_Exp
import os
from ..Dataset.Imagenet import data_dict 
class Exp(My_Exp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'BYTES_NANODET'
        self.minimum_threshold=0.3
        self.input_size = (320, 256)
        self.test_size = (320, 256)
        self.test_conf=0.45
        self.lownmsthre = 0.45
        self.nmsthre = 0.45
        self.iou_threshold=0.3 #intersection between detection in multiple frames to 
        self.lenght_track=5  #number of frames for which a track remain active
        self.min_hits=2 #minimum number of associations before a track is followed.
        self.seq_lenght=-1 #meaning take the video full lenght
        self.data_num_workers = 0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    def get_model(self,selected_classes=None):
        from ..Models.Postprocess.Bytes_postprocess import BYTES
        from ..Models.NN_Bytes import NN_Augmented
        if selected_classes is None:
            selected_classes=[ data_dict[self.COCO][i] - int(not self.Add_Background) for i in data_dict[self.COCO]]
        NN_model=super().get_model()
        
        PP_method=BYTES(self.lenght_track,self.min_hits,iou_threshold=self.iou_threshold,min_score=self.test_conf)
        return NN_Augmented(NN_model,PP_method,self.minimum_threshold,self.nmsthre,self.num_classes,selected_classes)
    
    
    
        
