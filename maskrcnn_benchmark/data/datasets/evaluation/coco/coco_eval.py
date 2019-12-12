import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.modeling.roi_heads.depth_head.inference import Masker as Masker2
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import pycocotools.mask as mask_util
import numpy as np

import json
from pycocotools.cocoeval import COCOeval
from maskrcnn_benchmark.data.datasets.torch2.pycocotools2.DepthEval import DEPTHeval
from pycocotools.coco import COCO
from maskrcnn_benchmark.data.datasets.torch2.pycocotools2.coco2 import COCO2

import time
from collections import defaultdict
import copy
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format*****")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset)
    if "depth" in iou_types:
        # predictions: list of BoxList
        # dataset: ScanNetDataset
        logger.info('Preparing depth results')

        coco_results["depth"] = prepare_for_depth(predictions, dataset)

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        # print('££££££££££', iou_type)
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if iou_type == 'depth':
                res = evaluate_depth_predictions(dataset.coco, coco_results[iou_type], file_path, iou_type)
            else:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def prepare_for_coco_segmentation(predictions, dataset):
    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask") # masks [num<100, 1, 28, 28]
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.

        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis],  dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results

def prepare_for_depth(predictions, dataset):
    masker = Masker2()
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # print('%%%%%%%%%%%', len(predictions)) #number of validation images
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        depths = prediction.get_field("depth")
        # depths.shape [100, 1, 28, 28]

        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.

        depths = masker(depths.expand(1, -1, -1, -1, -1), prediction)
        depths = depths[0]
        # print('WWWWWWWWWWWWWWWWWWWWWW', depths)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        # rles = [
        #     mask_util.encode(np.array(depth[0, :, :, np.newaxis],  dtype=np.uint8, order="F"))[0]
        #     for depth in depths
        # ]
        # for rle in rles:
        #     rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "depth": np.array(depth).tolist(),
                    "score": scores[k],
                }
                for k, depth in enumerate(depths)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'keypoints': keypoint,
            'score': scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):


    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)



    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
#####################################################################
def evaluate_depth_predictions(
    coco_gt, coco_results, json_result_file, iou_type="depth"
):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO2()


    coco_eval = DEPTHeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()



#############################################################################
class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        # "depth": ["AP", "AP50", "AP75", "APm", "APl"],
        "depth": ["abs_rel", "sq_rel", "rmse_log", "log10_mean", "a1", "a2", "a3"]
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints", "depth")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        assert isinstance(coco_eval, COCOeval) or isinstance(coco_eval, DEPTHeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType

        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]

        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)

###########################################################################################################################


###########################################################################################################################
# class DEPTHeval:
#     def __init__(self, cocoGt=None, cocoDt=None, iouType='depth'):
#         self.cocoGt = cocoGt  # ground truth COCO API
#         self.cocoDt = cocoDt  # detections COCO API
#
#         self.params = Params(iouType=iouType)  # parameters
#         self.stats = []  # result summarization
#         self.ious = {}  # ious between all gts and dts
#
#     def _prepare(self):
#         '''
#         Prepare ._gts and ._dts for evaluation based on params
#         :return: None
#         '''
#         def _toMask(anns, coco):
#             # modify ann['segmentation'] by reference
#             for ann in anns:
#                 rle = coco.annToRLE(ann)
#                 ann['segmentation'] = rle
#         p = self.params
#         if p.useCats:
#             gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
#             dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
#         else:
#             gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
#             dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))
#
#         # convert ground truth to mask if iouType == 'segm'
#         # if p.iouType == 'segm':
#         #     _toMask(gts, self.cocoGt)
#         #     _toMask(dts, self.cocoDt)
#         # set ignore flag
#         for gt in gts:
#             gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
#             gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
#
#         self._gts = defaultdict(list)       # gt for evaluation
#         self._dts = defaultdict(list)       # dt for evaluation
#         for gt in gts:
#             self._gts[gt['image_id'], gt['category_id']].append(gt)
#         for dt in dts:
#             self._dts[dt['image_id'], dt['category_id']].append(dt)
#         self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
#         self.eval = {}                  # accumulated evaluation results
#
#
#     def evaluate(self):
#         '''
#         Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
#         :return: None
#         '''
#         tic = time.time()
#         print('Running per image evaluation...')
#         p = self.params
#         # add backward compatibility if useSegm is specified in params
#
#         print('Evaluate annotation type *{}*'.format(p.iouType))
#         p.imgIds = list(np.unique(p.imgIds))
#         if p.useCats:
#             p.catIds = list(np.unique(p.catIds))
#         p.maxDets = sorted(p.maxDets)
#         self.params=p
#
#         self._prepare()
#         # loop through images, area range, max detection number
#         catIds = p.catIds if p.useCats else [-1]
#
#         if p.iouType == 'depth':
#             computeIoU = self.computeIoU
#
#         self.ious = {(imgId, catId): computeIoU(imgId, catId) \
#                         for imgId in p.imgIds
#                         for catId in catIds}
#
#         evaluateImg = self.evaluateImg
#         maxDet = p.maxDets[-1]
#         self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
#                  for catId in catIds
#                  for areaRng in p.areaRng
#                  for imgId in p.imgIds
#              ]
#         self._paramsEval = copy.deepcopy(self.params)
#         toc = time.time()
#         print('DONE (t={:0.2f}s).'.format(toc-tic))
#
#     def computeIoU(self, imgId, catId):
#         p = self.params
#         if p.useCats:
#             gt = self._gts[imgId,catId]
#             dt = self._dts[imgId,catId]
#         else:
#             gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
#             dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
#         if len(gt) == 0 and len(dt) ==0:
#             return []
#         inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
#         dt = [dt[i] for i in inds]
#         if len(dt) > p.maxDets[-1]:
#             dt=dt[0:p.maxDets[-1]]
#
#         # print('eeeeeeeeeeeeeeeeee', dt, gt)
#
#         if p.iouType == 'depth':
#             g = [g['depth'] for g in gt]
#             d = [d['depth'] for d in dt]
#         else:
#             raise Exception('unknown iouType for iou computation')
#
#         # compute iou between each dt and gt region
#         iscrowd = [int(o['iscrowd']) for o in gt]
#         # ious = maskUtils.iou(d,g,iscrowd)
#         ious = 0
#         return ious
#
# class Params:
#     def __init__(self, iouType='depth'):
#         if iouType == 'depth':
#             self.setDepthParams()
#
#     def setDepthParams(self):
#         self.imgIds = []
#         self.catIds = []
#         self.maxDets = [1, 10, 100]
#         self.useCats = 1

##########################################################################################################################


############################################################################################################################
# class COCO2:
#     def __init__(self, annotation_file=None):
#         """
#         Constructor of Microsoft COCO helper class for reading and visualizing annotations.
#         :param annotation_file (str): location of annotation file
#         :param image_folder (str): location to the folder that hosts images.
#         :return:
#         """
#         # load dataset
#         self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
#         self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
#         if not annotation_file == None:
#             print('loading annotations into memory...')
#             tic = time.time()
#             dataset = json.load(open(annotation_file, 'r'))
#             assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
#             print('Done (t={:0.2f}s)'.format(time.time()- tic))
#             self.dataset = dataset
#             self.createIndex()
#
#     def createIndex(self):
#         # create index
#         print('creating index...')
#         anns, cats, imgs = {}, {}, {}
#         imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
#         if 'annotations' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 imgToAnns[ann['image_id']].append(ann)
#                 anns[ann['id']] = ann
#
#         if 'images' in self.dataset:
#             for img in self.dataset['images']:
#                 imgs[img['id']] = img
#
#         if 'categories' in self.dataset:
#             for cat in self.dataset['categories']:
#                 cats[cat['id']] = cat
#
#         if 'annotations' in self.dataset and 'categories' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 catToImgs[ann['category_id']].append(ann['image_id'])
#
#         print('index created!')
#
#         # create class members
#         self.anns = anns
#         self.imgToAnns = imgToAnns
#         self.catToImgs = catToImgs
#         self.imgs = imgs
#         self.cats = cats
#
#     def loadRes(self, resFile):
#         """
#         Load result file and return a result api object.
#         :param   resFile (str)     : file name of result file
#         :return: res (obj)         : result api object
#         """
#         res = COCO2()
#         res.dataset['images'] = [img for img in self.dataset['images']]
#
#         print('Loading and preparing results...')
#         tic = time.time()
#         if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
#             anns = json.load(open(resFile))
#         elif type(resFile) == np.ndarray:
#             anns = self.loadNumpyAnnotations(resFile)
#         else:
#             anns = resFile
#         assert type(anns) == list, 'results in not an array of objects'
#         annsImgIds = [ann['image_id'] for ann in anns]
#         assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
#             'Results do not correspond to current coco set'
#         if 'caption' in anns[0]:
#             imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
#             res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
#             for id, ann in enumerate(anns):
#                 ann['id'] = id + 1
#         elif 'depth' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 # ann['area'] =
#                 ann['id'] = id + 1
#                 ann['iscrowd'] = 0
#         print('DONE (t={:0.2f}s)'.format(time.time() - tic))
#
#         res.dataset['annotations'] = anns
#         res.createIndex()
#         return res
