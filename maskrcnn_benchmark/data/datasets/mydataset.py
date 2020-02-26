import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.structures.depth_map import DepthMap
from PIL import Image
import os
import numpy as np

from .torch2.coco2 import CocoDetection2

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


# class ScanNetDataset(torchvision.datasets.coco.CocoDetection):
class ScanNetDataset(CocoDetection2):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(ScanNetDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        self.PATH_DIR = '/home/wenjing/storage/ScanNetv2/'
        self.img_size = (320, 240)

    def __getitem__(self, idx):
        img, anno = super(ScanNetDataset, self).__getitem__(idx)
        ############################resize image################################################
        img = img.resize(self.img_size)
        ###########################################################################
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, self.img_size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, self.img_size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)
##############################################################################
        if anno and "depth" in anno[0]:
            depth_dir = [obj["depth"] for obj in anno]
            num_obj = len(depth_dir)
            depth_dir = depth_dir[0]
            depth_dir = os.path.join(self.PATH_DIR, depth_dir)
            depth_i = Image.open(depth_dir).resize(self.img_size)   # (1296,968)
            depth_i = torch.from_numpy(np.array(depth_i))
            self.l = []
            depth = self.l
            for i in range(num_obj):
                depth.append(depth_i)
            depth = DepthMap(depth, self.img_size, mode='mask')
            target.add_field("depth", depth)
        # if anno and "depth" in anno[0]:
        #     depth = [obj["depth"] for obj in anno]
        #     depth = SegmentationMask(depth, img.size, mode='poly')
        #     target.add_field("depth", depth)

###################################################################################
###################################################################################
        if anno and "intrinsic" in anno[0]:
            intrinsic = [obj["intrinsic"] for obj in anno]
            intrinsic = torch.tensor(intrinsic)
            target.add_field("intrinsic", intrinsic)
####################################################################################
        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
