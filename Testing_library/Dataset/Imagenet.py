#Copyright (C)  Bomps4 (luca.bompani5@unibo.it) 2023-2024 University of Bologna, Italy.
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
#from detectron2.structures import BoxMode
from PIL import Image
import numpy as np
import torch
import random
from random import choice,sample
import gc
import glob 
from torch import nn,Tensor
from typing import Union,List,Tuple,Dict
import pandas as pd
import PIL
from ..My_transforms.Transforms import Selective_Compose
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
from ..My_transforms.functional import *
from ..My_transforms.Transforms import pil_to_tensor

from loguru import logger
from copy import deepcopy

COCO_V1_NAMES=['__background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
COCO_v2_NAMES=["__background__","person","bicycle","car","motorcycle","airplane",
    "bus","train","truck","boat","traffic light","fire hydrant","N/A","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","N/A","backpack","umbrella","N/A","N/A","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle",
    "N/A","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","N/A","dining table","N/A","N/A",
    "toilet","N/A","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","N/A","book","clock",
    "vase","scissors","teddy bear","hair drier","toothbrush"]


key_to_name={'n02691156': 'airplane',
'n02419796': 'antelope',
'n02131653': 'bear',
'n02834778': 'bicycle',
'n01503061': 'bird',
'n02924116': 'bus',
'n02958343': 'car',
'n02402425': 'cattle',
'n02084071': 'dog',
'n02121808': 'domestic cat',
'n02503517': 'elephant',
'n02118333': 'fox',
'n02510455': 'giant panda',
'n02342885': 'hamster',
'n02374451': 'horse',
'n02129165': 'lion',
'n01674464': 'lizard',
'n02484322': 'monkey',
'n03790512': 'motorcycle',
'n02324045': 'rabbit',
'n02509815': 'red panda',
'n02411705': 'sheep',
'n01726692': 'snake',
'n02355227': 'squirrel',
'n02129604': 'tiger',
'n04468005': 'train',
'n01662784': 'turtle',
'n04530566': 'watercraft',
'n02062744': 'whale',
'n02391049': 'zebra'}

keys=sorted(list(key_to_name.keys()),key=lambda x:key_to_name[x])
NAMES=['__background__']+[key_to_name[i] for i in keys]#



imagenet_to_coco={j:i+1 for i,j in enumerate(keys)}

data_dict={}

in_coco_v1={'n02691156':5,'n02834778':2,'n01503061':15,'n02924116':6,'n02402425':20,'n02958343':3,'n02084071':17,
	'n02121808':16,'n02503517':21,'n02374451':18,'n02131653':22,'n03790512':4,
	'n02411705':19,'n04468005':7,'n04530566':9,'n02391049':23}

in_coco_v2={'n02691156':5,'n02834778':2,'n01503061':16,'n02924116':6,'n02402425':21,
	    'n02958343':3,'n02084071':18,'n02121808':17,'n02503517':22,'n02374451':19,'n02131653':23
		,'n03790512':4,'n02411705':20,'n04468005':7,'n04530566':9,'n02391049':24}

DATASET_NAMES={'COCO_V1':COCO_V1_NAMES,'COCO_V2':COCO_v2_NAMES,'Imgvid':NAMES}
data_dict={'COCO_V1':in_coco_v1,'COCO_V2':in_coco_v2,'Imgvid':imagenet_to_coco}
def read_image(image_path):
	return Image.open(image_path)


def set_sub_captioning(IMG_NET_IMG,IMG_NET_ANN,sub=1,COCO='Imgvid',Add_Background=True):
	def ds_image_net_val():
		import xml.etree.ElementTree as ET
		import os
		from detectron2.structures import BoxMode
		local=	IMG_NET_ANN+'val/'
		videos=sorted(os.listdir(local))
		index=0
		to_return=[]
		in_dataset=data_dict[COCO]
		for j in videos:
			frames=sorted(os.listdir(local+j)) #annotations
			frames=frames[::sub]
			#frames.insert(1,frames_2)
			images=[k.split('.')[0]+'.JPEG' for k in frames] #actual images
			for f,img in zip(frames,images):
				record={}
				tree = ET.parse(local+j+'/'+f)
				root = tree.getroot()
				record["width"]=int(root.find('size').find('width').text)
				record["height"]=int(root.find('size').find('height').text)
				record["image_id"] = index
				index+=1
				record['file_name']=IMG_NET_IMG+'val/'+j+'/'+img
				changed_once=False
				annotations=[]
				for to_dect in root.findall('object'):
					obj={}
					species=to_dect.find('name').text
					if(species not in in_dataset):
						continue	
					bbox=[int(child.text) for child in to_dect.find('bndbox')]

					species = int(in_dataset[species]) 
					species = species if (Add_Background) else species -1
					changed_once=True
					obj["category_id"]=species				
					obj["bbox"]=[bbox[1],bbox[3],bbox[0],bbox[2]]
					obj["bbox_mode"]= BoxMode.XYXY_ABS
					annotations.append(obj)

				record["annotations"] = annotations
				to_return.append(record)
		return to_return
	return ds_image_net_val



class Mosaic_Augment(nn.Module):
	def __init__(self,Image_names,output_size:Union[List[int],Tuple[int],int],scale_range:List[float],p:float=0.2):
		super(Mosaic_Augment,self).__init__()
		self.other_images=Image_names
		self.p=0.3
		if(isinstance(output_size,int)):
			output_size=[output_size,output_size]
		
		self.output_size=output_size
		self.scale_range=scale_range

	def forward(self,image:Union[Tensor,PIL.Image.Image],target:Dict[str,Tensor]):
		r=torch.rand(1)
		if(r>self.p):
			
			return image,target
		other_images_inf=[i[0] for i in sample(self.other_images,3)] #other images informations (path,height annotations ecc.)
		
		
		scale_x = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
		scale_y = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
		divid_point_x = int(scale_x * self.output_size[0])
		divid_point_y = int(scale_y * self.output_size[1])

		is_tensor=isinstance(image,torch.Tensor)
		if(is_tensor):
			image=to_pil_image(image)
		read_images=[image]+[read_image(i['file_name']) for i in other_images_inf]#other images read
		
		

		
		
		
		new_image=Image.new('RGB',self.output_size)
		new_categories=torch.cat([target["labels"]]+[i['annotations']["labels"] for i in other_images_inf])
		
		old_size=[get_size(i) for i in read_images]
		new_bboxes=[target["boxes"]]+[i['annotations']["boxes"] for i in other_images_inf]
		#resizing top lef 0,0
		new_size=(divid_point_y,divid_point_x) #resizing top lef 0,0
		new_image.paste(F.resize(read_images[0],new_size),(0,0))
		new_bboxes[0]=resize_bounding_box(new_bboxes[0],old_size[0],new_size)
		
		#resizing top lef 0,0
		new_size=(divid_point_y,self.output_size[0]-divid_point_x)
		new_image.paste(F.resize(read_images[1],new_size),(divid_point_x,0))
		new_bboxes[1]=resize_bounding_box(new_bboxes[1],old_size[1],new_size)+torch.tensor([divid_point_x,0,divid_point_x,0]).unsqueeze(0)
		
		#resizing top lef 0,0
		new_size=(self.output_size[1]-divid_point_y,divid_point_x)
		new_image.paste(F.resize(read_images[2],new_size),(0,divid_point_y))
		new_bboxes[2]=resize_bounding_box(new_bboxes[2],old_size[2],new_size)+torch.tensor([0,divid_point_y,0,divid_point_y]).unsqueeze(0)

		#resizing top lef 0,0
		new_size=(self.output_size[1]-divid_point_y,self.output_size[0]-divid_point_x)
		new_image.paste(F.resize(read_images[3],new_size),(divid_point_x,divid_point_y))
		new_bboxes[3]=resize_bounding_box(new_bboxes[3],old_size[3],new_size)+torch.tensor([divid_point_x,divid_point_y,divid_point_x,divid_point_y]).unsqueeze(0)
		
		return new_image,{"boxes":torch.cat(new_bboxes),"labels":new_categories,"width":new_size[0],"height":new_size[1]}

			
		
		
		



class Imagenet_VID_Dataset(Dataset):
	def __init__(self,imagenet_annotation:str,imagenet_images:str,transform:Union[List[nn.Module],nn.Module],val=False,seq_lenght=1,sub_camp=1,max_labels=50,COCO='Imgvid',Add_Background=True):
		self.seq_lenght=seq_lenght
		self.max_labels=max_labels
		self.COCO=COCO
		
		self.sub_camp=sub_camp
		
		local=imagenet_annotation
		self.val=val
		if(not val):
			dirs=sorted(os.listdir(local))
		else:
			dirs=['val']
		index=0
		vid_index=0
		to_return=[]
		in_dataset=data_dict[COCO]
		species_total=set()
		for indice,i in enumerate(dirs):
			cur_dir=local+i+'/'
			videos=sorted(os.listdir(cur_dir))
			
			for j in videos:
				to_return.append([])
				frames=sorted(os.listdir(cur_dir+j))
				frames=frames[::self.sub_camp]
				#frames.insert(1,frames_2)
				vid_index+=1
				images=[k.split('.')[0]+'.JPEG' for k in frames]
				for f,img in zip(frames,images):
					record={}
					tree = ET.parse(cur_dir+j+'/'+f)
					root = tree.getroot()
					record['vid_index']=vid_index

					record["width"]=int(root.find('size').find('width').text)
					record["height"]=int(root.find('size').find('height').text)
					record["image_id"] = index
					index+=1
					
					record['file_name']=imagenet_images+i+'/'+j+'/'+img
					annotations=[]
					species=0
					changed_once=False
					obj={"labels":[],"boxes":[]}
					for to_dect in root.findall('object'):
						species=to_dect.find('name').text
						if(species not in in_dataset):
							continue

						bbox=[int(child.text) for child in to_dect.find('bndbox')]
						species= int(in_dataset[species])
						species = species if (Add_Background) else species -1
						species_total.add(species)
						changed_once=True
						
						obj["labels"].append(species)
						obj["boxes"].append(torch.tensor([bbox[1],bbox[3],bbox[0],bbox[2]]).unsqueeze(0))
						#obj["bbox_mode"]= BoxMode.XYXY_ABS

					if(not val and not changed_once ):
						continue
					if(len(obj["labels"])==0):
						obj["labels"]=torch.empty((0,))
						obj["boxes"]=torch.empty((0,4))
					else:
						obj["labels"]=torch.tensor(obj["labels"])
						obj["boxes"]=torch.cat(obj["boxes"],dim=0)
					record['annotations']=obj
					to_return[-1].append(record)
				
				is_there_an_object=[i['annotations']["labels"].shape[0]==0 for i in to_return[-1]] #a list of booleans all true if there isn't an object that can be detected in the video
				if(all(is_there_an_object)):
					to_return.pop()
		
		sequences=[] 
		
		for  i in to_return: #ogniuno di questi è un video
			for j in range(0,len(i),seq_lenght):
				sequences.append([]) 
				for k in range(0,min(seq_lenght,len(i)-j)): #ogni sequenza è lunga al seq_lenght
					sequences[-1].append(i[j+k])
						
		
			# random.shuffle(sequences)
		# else:
		# sequences=to_return
			#if(self.sub_camp>1):
			#	sequences=[i for i in sequences]
		self.data=sequences
		
		self.transform=Selective_Compose(transform)
		
	def __len__(self):
		if(self.seq_lenght==-1):
			return len(self.data)
		if( not hasattr(self,'length')):
			lenght=0
			for i in self.data:
				lenght+=len(i)
			self.length=lenght
		return self.length
	
	def add_transform(self,transform):
		self.transform.add_trasform(transform)

	def remove_transform(self,idx):
		self.transform.remove_transform(idx)


		
	def __getitem__(self, idx:int)->Tuple[List[Tensor],Dict[str,Union[Tensor,int,float]]]:
		sequence=deepcopy(self.data[idx])
		
		images=[]
		gts=[]
		for i in sequence:
			
			annotations=deepcopy(i['annotations'])
			image,gt=self.transform(read_image(i['file_name']),{**annotations,'vid_index':i['vid_index']})
			images.append(image)
			if 'resize_factor' in gt and 'mask' in gt:
				r=gt['resize_factor']
				bool_mask=gt['mask']
			
			gt=torch.cat((gt['labels'].unsqueeze(1),gt["boxes"]),dim=-1)
			
			_padded_gt=torch.zeros((self.max_labels,5))
			_padded_gt[:gt.shape[0],:5]=gt
			#if(not self.val):
			
			gt={"annotations":_padded_gt,"width":i["width"],"height":i["height"],"image_id":i["image_id"],'file_name':i['file_name'],'vid_index':i['vid_index'],'resize_factor':r,'mask':bool_mask}
			gts.append(gt)
		
		# if(self.seq_lenght!= 1):
		# 	images=torch.stack(images)
		# 	return images,gts
				
			
		'''
	
		
		sequence=[{**gt, **{"image_id":i["image_id"],"width":i["width"],"height":i["height"]}} for i,gt in zip(sequence,gts)]
		#i["annotations"]|{"image_id":i["image_id"],"width":i["width"],"height":i["height"]} for i in sequence]
		
		if(len(sequence)<self.seq_lenght):
			images+=[images[-1] for i in range(len(sequence),self.seq_lenght,1)]
			sequence+=[sequence[-1] for i in range(len(sequence),self.seq_lenght,1)]
		'''
		images=images[0]
		
		gts=gts[0]

		return images,gts

if __name__ == '__main__':
	print(NAMES)
	print(VID_classes)
	print(imagenet_to_coco)
	for i in imagenet_to_coco:
		print(key_to_name[i],' ',i,' ',imagenet_to_coco[i])
	for i in in_coco_v1:
		print(key_to_name[i])
	print(len(COCO_V1_NAMES))
	print(len(COCO_v2_NAMES))
