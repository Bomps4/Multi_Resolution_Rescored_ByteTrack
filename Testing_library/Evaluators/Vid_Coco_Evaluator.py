import json
import os
import numpy as np
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pprint import pformat
import copy
from ..utils.env import synchronize, get_rank, is_main_process


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs.
    Adapted from pycocotools — prints removed, Python 3 compatibility fixes applied.
    :return: None
    '''
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs


def merge_json_files(input_files, output_file):
    """
    Merge multiple JSON files (each containing a list) into a single unified JSON file,
    flattening all lists into one, and remove the input files after processing.

    Args:
        input_files (list of str): Paths to the input JSON files.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as outfile:
        outfile.write("[")

        first_element = True
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                data = json.load(infile)

                if not isinstance(data, list):
                    raise ValueError(f"File {input_file} does not contain a JSON list.")

                for element in data:
                    if not first_element:
                        outfile.write(",")
                    else:
                        first_element = False

                    json.dump(element, outfile)

            os.remove(input_file)

        outfile.write("]")


def unite_and_remove(directory):
    list_of_files = glob.glob(f'{directory}detections_*')

    with open(list_of_files[0]) as json_file:
        data = json.load(json_file)

    if len(list_of_files) != 1:
        for file in list_of_files[1:]:
            with open(file) as json_file:
                partial_data = json.load(json_file)
                for key in partial_data:
                    data[key] += partial_data[key]

    for file in list_of_files:
        os.remove(file)

    with open(f'{directory}united_detections.json', 'w') as united_json:
        json.dump(data, united_json)


class COCO_EVAL(object):
    def __init__(self, classes, ids, saving_directory, add_background, gt_file):
        self.saving_directory = saving_directory
        self.categories = (
            [{'id': 0, 'name': '__background__', 'supercategory': ''}]
            + [{'id': int(i), 'name': str(classes[i-1]), 'supercategory': ''} for i in ids]
        )
        self.license = {"id": 3, "name": 'Fittizia', "url": ''}
        self.classes_ids = ids

        self.image_ids = []
        self.image_ids_gt = set()
        self.images = []
        self.gt_file = gt_file

        self.evaluator_idx = get_rank()

        if os.path.exists(gt_file):
            self.cocoGt = COCO(gt_file)
            self.cocoEval = COCOeval(self.cocoGt, iouType='bbox')
            self.cocoEval.params.catIds = self.classes_ids

        self.batch_execute = 640
        self.processed = 0
        self.current_block = 0
        self.gts = []
        self.evalImgs = []
        self.total_detections = []
        self.detections = []
        self.current_id = 0

    def add_image_informations(self, _id, width, height, file_name):
        self.image_ids.append(_id)
        im = {"id": _id, "width": width, "height": height, "file_name": file_name, "license": 3}
        self.images.append(im)

    def add_gt_annotations(self, _id, classe, bbox, track_id=None):
        self.image_ids_gt.add(_id)
        gt = {
            'id': self.current_id,
            "image_id": _id,
            "category_id": classe,
            "area": int(bbox[-1] * bbox[-2]),
            "iscrowd": 0,
            "bbox": [int(i) for i in bbox],
        }
        if track_id is not None:
            gt['track_id'] = track_id
        self.gts.append(gt)
        self.current_id += 1

    def add_detections(self, _id, classes, scores, bboxes):
        if len(classes) == 0 or len(scores) == 0 or len(bboxes) == 0:
            results = []
        else:
            results = [
                {"image_id": _id, 'bbox': bbox, 'category_id': classe, 'score': score}
                for bbox, classe, score in zip(bboxes, classes, scores)
            ]

        self.detections.extend(results)

    def process_detections(self, results):
        if isinstance(results, str):
            cocoDt = self.cocoGt.loadRes(results)
        else:
            if results == []:
                return

            img_ids = list(set([i['image_id'] for i in results]))
            self.image_ids.extend(img_ids)
            cocoDt = self.cocoGt.loadRes(results)
            self.cocoEval.params.imgIds = img_ids

        self.cocoEval.cocoDt = cocoDt
        self.cocoEval.evaluate()

    def accumulate(self):
        self.cocoEval.accumulate()

    def summarize(self):
        self.cocoEval.summarize()

    def ground_truth_writing(self):
        info = {
            "year": 2023,
            "version": 1,
            "description": 'boh',
            "contributor": 'io',
            "url": '',
            "date_created": '',
        }

        ground_truths = {
            "info": info,
            "images": self.images,
            "annotations": self.gts,
            "licenses": [self.license],
            "categories": self.categories,
        }

        self.gt_file = self.gt_file if self.gt_file.endswith('.json') else self.gt_file + '.json'

        with open(self.gt_file, 'w') as gt_onnx:
            json.dump(ground_truths, gt_onnx, indent=4)

    def evaluation(self, rank, results_file=None):
        results_file = f'{self.saving_directory}united_detections.json'
        with open(results_file, 'w') as json_det:
            json.dump(self.detections, json_det, indent=4)
        self.detections = []
        self.process_detections(results_file)

        synchronize()

        if rank == 0:
            self.accumulate()
            self.summarize()

            precision = self.cocoEval.eval['absolute_precision']
            recall = self.cocoEval.eval['recall']
            mAP = self.cocoEval.stats[0]

            A_P = [
                f" \n Average Precision  (AP) @[ IoU={j} | area=   {k} | maxDets=100 ] = {i} \n"
                for j, k, i in zip(
                    ['0.5:0.95', '0.5', '0.75', '0.5:0.95', '0.5:0.95', '0.5:0.95'],
                    ['all', 'all', 'all', 'small', 'medium', 'large'],
                    self.cocoEval.stats[:6],
                )
            ]
            A_R = [
                f" \n Average Recall     (AR) @[ IoU={j} | area=   {k} | maxDets=  {dets} ] = {i} \n"
                for dets, j, k, i in zip(
                    [1, 10, 100, 100, 100, 100],
                    ['0.5:0.95', '0.5:0.95', '0.5:0.95', '0.5:0.95', '0.5:0.95', '0.5:0.95'],
                    ['all', 'all', 'all', 'small', 'medium', 'large'],
                    self.cocoEval.stats[6:],
                )
            ]

            out_pre = [f"output precision {np.mean(precision[0,:,0,2])}"]
            out_recall = [f"output recall {np.mean(recall[0,:,0,2])}"]
            f1_score = [
                f"output f1_score {2*np.mean(precision[0,:,0,2])*np.mean(recall[0,:,0,2])/(np.mean(precision[0,:,0,2])+np.mean(recall[0,:,0,2]))}"
            ]
            printable = pformat(A_P + A_R + out_pre + out_recall + f1_score)

            self.cocoEval = COCOeval(self.cocoGt, iouType='bbox')
            self.image_ids = []
            self.total_detections = []
            self.evalImgs = []
            self.detections = []

            return mAP, printable
