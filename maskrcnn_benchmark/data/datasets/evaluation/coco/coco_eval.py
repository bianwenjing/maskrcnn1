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

from PIL import Image
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.utils.miscellaneous import mkdir

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
    if "whole_depth" in iou_types:
        logger.info('Preparing whole depth result')
        coco_results["whole_depth"] = prepare_for_whole_depth(predictions, dataset)
    if "depth" in iou_types:
        # predictions: list of BoxList
        # dataset: ScanNetDataset
        logger.info('Preparing depth results')
        coco_results["depth"] = prepare_for_depth(predictions, dataset)


    results = COCOResults(*iou_types)
    ###############################################
    result_one_category = COCOResults(*iou_types)
    ###############################################
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if iou_type == 'whole_depth':
                res = evaluate_whole_depth_prediction(dataset.coco, coco_results[iou_type], file_path, iou_type)
            elif iou_type == 'depth':
                res = evaluate_depth_predictions(dataset.coco, coco_results[iou_type], file_path, iou_type)
            else:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
            )

            results.update(res)
############################################################################
    ##############  inference for each category
    eval_cate = False
    # care_cate = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    care_cate = [39]
    cate = {}
    # cate['otherfur'] = [25, 43, 51, 56, 57, 67, 68, 70, 81, 85, 87, 90, 91, 96, 97, 99, 104, 110, 116, 122, 126, 129, 143, 153,
    #             155, 177, 213, 233, 234, 282, 289, 307, 326, 354, 363, 389, 410, 411, 484, 525, 566,
    #             581, 765, 822, 851, 1117, 1169, 1177, 1180, 1202, 1209, 1227, 1242, 1244, 1256, 1270, 1290, 1341, 1351]
    # cate['sofa']=[6,1313]
    # cate['bed'] = [11, 1191, 494, 786, 1349]
    # cate['bathtub'] = [42]
    # cate['bookshelf']=[18]
    # cate['carbinet'] = [7, 29, 75, 1164,1173, 385, 1322, 815]
    # cate['chair'] = [2, 10, 23, 74, 1184, 1291, 1338, 885]
    # cate['counter'] = [159, 35, 1156]
    # cate['curtain'] = [21]
    # cate['desk'] = [9]
    # cate['door'] = [5, 161, 1167, 276, 188, 1208, 385, 649, 569, 1345]
    # cate['picture'] = [15, 1188, 1218, 15]
    # cate['refrigerator'] = [27, 165, 1062]
    # cate['showercurtain'] = [55]
    # cate['sink'] = [14]
    # cate['table'] = [4, 24, 44, 45, 1193, 108, 222, 1355]
    # cate['toilet'] = [17, 1257]
    # cate['window'] = [16]

    mode = 'segm'
    if eval_cate==True:
        if mode == 'depth':
            res = evaluate_depth_predictions(dataset.coco, coco_results[mode], file_path, mode)
        else:
            res = evaluate_predictions_on_coco(dataset.coco, coco_results[mode], file_path, mode)
        for catId in dataset.coco.getCatIds():
            if catId in care_cate:
                print('##############', catId)
                res.params.catIds = [catId]
                res.evaluate(each_category = True)
                if mode =='segm':
                    res.accumulate()
                res.summarize()
                # result_one_category.update(res)
                # if output_folder:
                #     name = str(catId) + ".pth"
                #     torch.save(result_one_category, os.path.join(output_folder, name))
        # for i in cate.keys():
        #     print('%%%%%%%%%%%%', i)
        #     res.params.catIds = cate[i]
        #     res.evaluate()
        #     res.accumulate()
        #     res.summarize()
#################################################################################

    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        ###################################################################################
        prediction = prediction[0]  # prediction[1] is for whole depth
        #######################################################################################
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
        ###################################################################################
        prediction = prediction[0]  # prediction[1] is for whole depth
        #######################################################################################
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
            # masks.shape(num, 1, 968, 1296)
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
    masker = Masker(threshold=0.5, padding=1)
    masker2 = Masker2()
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # print('%%%%%%%%%%%', len(predictions)) #number of validation images
    for image_id, prediction in tqdm(enumerate(predictions)):
        ###################################################################################
        prediction = prediction[0]  # prediction[1] is for whole depth
        #######################################################################################
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        depths = prediction.get_field("depth")
        masks = prediction.get_field("mask")
        # print('@@@@@@@@@@@@@@@@@', depths[0][0])
        # depths.shape [48, 1, 28, 28]

        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.

        depths = masker2(depths.expand(1, -1, -1, -1, -1), prediction)
        depths = depths[0]
        ###########################
        masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
        masks = masks[0].int()
        depths = depths*masks
        #################################
        output_dir = cfg.OUTPUT_DIR + '/depth/' + str(image_id)
        dep_dir = []
        for i in range(int(depths.shape[0])):
            dd = output_dir + "_" + str(i) + ".png"
            # dd = output_dir + "_" + str(i) + ".tiff"
            mkdir(cfg.OUTPUT_DIR + '/depth')
            depths = np.array(depths)
            Image.fromarray(depths[i][0]).save(dd)
            dep_dir.append(dd)
        # print('@@@@@@@@@@@@@@',dep_dir)
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

        # coco_results.extend(
        #     [
        #         {
        #             "image_id": original_id,
        #             "category_id": mapped_labels[k],
        #             "depth": np.array(depth).tolist(),
        #             "score": scores[k],
        #         }
        #         for k, depth in enumerate(depths)
        #     ]
        # )
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "depth": depth,
                    "score": scores[k],
                }
                for k, depth in enumerate(dep_dir)
            ]
        )
    return coco_results

def prepare_for_whole_depth(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # print('%%%%%%%%%%%', len(predictions)) #number of validation images
    for image_id, prediction in tqdm(enumerate(predictions)):
        prediction = prediction[1]
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        # image_width = img_info["width"]
        # image_height = img_info["height"]
        # prediction = prediction.resize((image_width, image_height))


        output_dir = cfg.OUTPUT_DIR + '/whole_depth/' + str(image_id) + ".png"
        # dd = output_dir + "_" + str(i) + ".tiff"
        mkdir(cfg.OUTPUT_DIR + '/whole_depth')
        prediction = np.array(prediction.cpu().to(torch.int32))   #float -> int32, in order to save as image
        Image.fromarray(prediction).save(output_dir)

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "whole_depth": output_dir,
                    "category_id": 1
                }
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



    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO2()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = DEPTHeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
##############################################################################################
# # for each category
#     for catId in coco_gt.getCatIds():
#         print('£££££££££££££££££££££££',catId) #input category id
#         coco_eval.params.catIds = [catId]
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()

    return coco_eval
#####################################################################
def evaluate_depth_predictions(
    coco_gt, coco_results, json_result_file, iou_type="depth"
):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    print('************json file finished**************')
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO2()
    coco_eval = DEPTHeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    # coco_eval.accumulate()
    coco_eval.summarize()
    ##############################################################################################
    # for each category
    # for catId in coco_gt.getCatIds():
    #     print('£££££££££££££££££££££££', catId)  # input category id
    #     coco_eval.params.catIds = [catId]
    #     coco_eval.evaluate()
    #     # coco_eval.accumulate()
    #     coco_eval.summarize()


    return coco_eval

def evaluate_whole_depth_prediction(
        coco_gt, coco_results, json_result_file, iou_type
):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    print('************json file finished**************')
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO2()
    coco_eval = DEPTHeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.summarize()
    return coco_eval

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
        # "depth": ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10_mean", "a1", "a2", "a3"],
        # "whole_depth": ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10_mean", "a1", "a2", "a3"]
        "depth": ["abs_rel", "imae", "irmse", "log_mae", "rmse_log", "mae", "rmse", "scale_invar", "sq_rel"],
        "whole_depth": ["abs_rel", "imae", "irmse", "log_mae", "rmse_log", "mae", "rmse", "scale_invar", "sq_rel"]
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints", "depth", "whole_depth")
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

