class COCO_EVAL(object):
    def __init__(self,classes,ids,saving_directory,add_background):
        self.saving_directory=saving_directory
        self.categories=[{'id':0,'name':'__background__','supercategory':''}]+[{'id':int(i),'name':str(classes[i-int(add_background)]),'supercategory':''}for i in ids]
        self.license={"id": 3,"name": 'Fittizia',"url": '',}
        self.image_ids=set()
        self.image_ids_gt=set()
        self.images=[]
        self.gts=[]
        self.detections=[]
        self.current_id=0
    def add_image_informations(self,id,width,height,file_name):
        self.image_ids.add(id)
        im={"id": id,"width": width,"height": height,"file_name": file_name,"license": 3}
        self.images.append(im)

    def add_gt_annotations(self,id,classe,bbox):
        #assert id in self.image_ids
        self.image_ids_gt.add(id)
        gt={'id':self.current_id,"image_id": id,"category_id":classe,"area": int(bbox[-1]*bbox[-2]),"iscrowd": 0,"bbox":[int(i) for i in bbox]}
        self.gts.append(gt)
        self.current_id+=1
    def add_detection(self,id,classe,score,bbox):
        #assert id in self.image_ids

        record={}
        record['bbox'] = [int(i) for i in bbox]
        record['category_id']= classe
        record["image_id"] = id
        record['score']=score

        self.detections.append(record)

    def evaluation(self,classe=None):
        import json
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        info={
        "year": 2023,
        "version": 1,
        "description": 'boh',
        "contributor": 'io',
        "url": '',
        "date_created": '',
        }
        ground_truths={
        "info": info,
        "images": self.images,
        "annotations": self.gts,
        "licenses": [self.license],
        "categories":self.categories
        }
        
        json_gts=json.dumps(ground_truths)
        
        json_dts=json.dumps(self.detections)
        
        if(classe is None):
            fil=open(self.saving_directory+'ground_truth.json','w')
        else:
            fil=open(self.saving_directory+f'ground_truth_{classe}.json','w')
        fil.write(json_gts)
        
        if(classe is None):
            fil_b=open(self.saving_directory+'detections.json','w')
        else:
            fil_b=open(self.saving_directory+f'detections_{classe}.json','w')
        fil_b.write(json_dts)
        fil.close()
        fil_b.close()
        cocoGt=COCO(self.saving_directory+'ground_truth.json')
        
        cocoDt=cocoGt.loadRes(self.saving_directory+'detections.json')
        annType  = 'bbox'
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        #cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        cocoEval.summarize()
        
        A_P=[f" \n Average Precision  (AP) @[ IoU={j} | area=   {k} | maxDets=100 ] = {i} \n" for j,k,i in zip(['0.5:0.95','0.5','0.75','0.5:0.95','0.5:0.95','0.5:0.95'],['all','all','all','small','medium','large'],cocoEval.stats[:6])]
        A_R=[f" \n Average Recall     (AR) @[ IoU={j} | area=   {k} | maxDets=  1 ] = {i} \n" for j,k,i in zip(['0.5:0.95','0.5','0.75','0.5:0.95','0.5:0.95','0.5:0.95'],['all','all','all','small','medium','large'],cocoEval.stats[6:])]
      
        return A_P+A_R