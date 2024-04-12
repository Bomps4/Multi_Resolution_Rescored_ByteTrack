#Copyright (C)  Bomps4 (luca.bompani5@unibo.it) 2023-2024 University of Bologna, Italy.

import xml.etree.ElementTree as ET
import os
from detectron2.structures import BoxMode
from Imagenet import data_dict,NAMES
import json
IMG_NET_ANN='/scratch/ILSVRC2015/Annotations/VID/'
local=	IMG_NET_ANN+'val/'
COCO='Imgvid'
videos=sorted(os.listdir(local))
index=0
sub=1
to_return=[]
in_dataset=data_dict[COCO]
c_id=0
gt=[]
gt_images=[]
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
        record['file_name']=j+'/'+img
        gt_images.append({"id": record["image_id"],"width":record["width"],"height": record["height"],"file_name": record['file_name'],"license": 3})
        changed_once=False
        annotations=[]
        for to_dect in root.findall('object'):
            obj={}
            species=to_dect.find('name').text
            if(species not in in_dataset):
                continue	
            bbox=[int(child.text) for child in to_dect.find('bndbox')]

            species = int(in_dataset[species]) 
            species = species -1
            changed_once=True
            obj["category_id"]=species				
            obj["bbox"]=[bbox[1],bbox[3],bbox[0]-bbox[1],bbox[2]-bbox[3]]
            obj['area']=bbox[0]*bbox[2]
            obj['id']=c_id
            obj['image_id']=record["image_id"]
            c_id+=1
            annotations.append(obj)

            gt.append({'id':obj['id'],"image_id":record["image_id"],"category_id":obj["category_id"],"area": obj['area'],"iscrowd": 0,"bbox":obj["bbox"]})

        record["annotations"] = annotations
        to_return.append(record)
categories=[{'id':int(i),'name':str(j),'supercategory':''}for i,j in enumerate(NAMES[1:])]
print(categories)
print(gt_images[0])
info={
        "year": 2023,
        "version": 1,
        "description": 'boh',
        "contributor": 'io',
        "url": '',
        "date_created": '',
        }
license={"id": 3,"name": 'Fittizia',"url": '',}


ground_truths={
        "info": info,
        "images": gt_images,
        "annotations": gt,
        "licenses": [license],
        "categories":categories
        }

fil=open('./ground_truth.json','w')
json_out=json.dumps(ground_truths)
fil.write(json_out)
