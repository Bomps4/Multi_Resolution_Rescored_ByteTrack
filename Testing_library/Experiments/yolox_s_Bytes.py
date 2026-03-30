import os

from .yolo_base import Exp as MyExp
import torch 
from torch import distributed as dist
from ..Dataset.Imagenet import data_dict 


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        #self.high_threshold=0.3
        self.max_age=3
        self.min_hits=2
        self.iou_threshold=0.3
        self.min_score=0.1
        self.normalize=False
        self.test_conf = 0.25
        self.test_size = (320, 320)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.rescoring=True
        self.minimum_threshold=0.1
    def get_model(self):
        from ..Models.Postprocess.Bytes_postprocess import BYTES
        from ..Models.NN_Bytes import NN_Augmented
        model=super(Exp,self).get_model()

        if selected_classes is None:
            selected_classes=[ data_dict[self.COCO][i] - int(not self.Add_Background) for i in data_dict[self.COCO]]

        postProcessing=BYTES(self.lenght_track,self.min_hits,iou_threshold=self.iou_threshold,min_score=self.test_conf,rescoring=self.rescoring)
        self.model=NN_Augmented(model,postProcessing,self.test_conf,self.nmsthre,self.num_classes,selected_classes,self.test_size)
        return self.model

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from ..Dataset.Imagenet import Imagenet_VID_Dataset,NAMES
        from ..My_transforms.Transforms import T_Resize_as_YOLO,T_To_tensor
        transforms=[T_Resize_as_YOLO(self.test_size),T_To_tensor(False,self.mean,self.std)]
	
	#print(labels)
	#for i in images:
	#	plt.imshow(np.transpose(i,axes=[1,2,0]))
	#	plt.show()
	
        def collate(seq_of_seq):
            return seq_of_seq[0][0],seq_of_seq[0][1]
        
        
        valdataset = Imagenet_VID_Dataset(self.val_ann_dir,self.val_dat_dir,val=True,transform=transforms,seq_lenght=-1)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
       
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": sampler,
        }
        
        dataloader_kwargs["batch_size"] = 1
        val_loader = torch.utils.data.DataLoader(valdataset,collate_fn=collate,**dataloader_kwargs)

        return val_loader