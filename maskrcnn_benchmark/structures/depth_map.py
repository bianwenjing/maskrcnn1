import cv2
import copy
import torch
import numpy as np
from maskrcnn_benchmark.layers.misc import interpolate
from maskrcnn_benchmark.utils import cv2_util
import pycocotools.mask as mask_utils

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


""" ABSTRACT
Segmentations come in either:
1) Binary masks
2) Polygons

Binary masks can be represented in a contiguous array
and operations can be carried out more efficiently,
therefore BinaryMaskList handles them together.

Polygons are handled separately for each instance,
by PolygonInstance and instances are handled by
PolygonList.

SegmentationList is supposed to represent both,
therefore it wraps the functions of BinaryMaskList
and PolygonList to make it transparent.
"""


class BinaryMaskList(object):
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, masks, size):
        """
            Arguments:
                masks: Either torch.tensor of [num_instances, H, W]
                    or list of torch.tensors of [H, W] with num_instances elems,
                    or RLE (Run Length Encoding) - interpreted as list of dicts,
                    or BinaryMaskList.
                size: absolute image size, width first

            After initialization, a hard copy will be made, to leave the
            initializing source data intact.
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2

        if isinstance(masks, torch.Tensor):
            # The raw data representation is passed as argument
            masks = masks.clone()
        elif isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                masks = torch.empty([0, size[1], size[0]])  # num_instances = 0!
            elif isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).clone()
            # elif isinstance(masks[0], dict) and "counts" in masks[0]:
            #     if(isinstance(masks[0]["counts"], (list, tuple))):
            #         masks = mask_utils.frPyObjects(masks, size[1], size[0])
            #     # RLE interpretation
            #     rle_sizes = [tuple(inst["size"]) for inst in masks]
            #
            #     masks = mask_utils.decode(masks)  # [h, w, n]
            #     masks = torch.tensor(masks).permute(2, 0, 1)  # [n, h, w]
            #
            #     assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), (
            #         "All the sizes must be the same size: %s" % rle_sizes
            #     )
            #
            #     # in RLE, height come first in "size"
            #     rle_height, rle_width = rle_sizes[0]
            #     assert masks.shape[1] == rle_height
            #     assert masks.shape[2] == rle_width
            #
            #     width, height = size
            #     if width != rle_width or height != rle_height:
            #         masks = interpolate(
            #             input=masks[None].float(),
            #             size=(height, width),
            #             mode="bilinear",
            #             align_corners=False,
            #         )[0].type_as(masks)
            else:
                RuntimeError(
                    "Type of `masks[0]` could not be interpreted: %s"
                    % type(masks)
                )
        elif isinstance(masks, BinaryMaskList):
            # just hard copy the BinaryMaskList instance's underlying data
            masks = masks.masks.clone()
        else:
            RuntimeError(
                "Type of `masks` argument could not be interpreted:%s"
                % type(masks)
            )

        if len(masks.shape) == 2:
            # if only a single instance mask is passed
            masks = masks[None]

        assert len(masks.shape) == 3
        assert masks.shape[1] == size[1], "%s != %s" % (masks.shape[1], size[1])
        assert masks.shape[2] == size[0], "%s != %s" % (masks.shape[2], size[0])

        self.masks = masks
        # print(masks.shape,'£££££££££££££££££££££££')
        self.size = tuple(size)

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.masks.flip(dim)
        return BinaryMaskList(flipped_masks, self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return BinaryMaskList(cropped_masks, cropped_size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_masks = interpolate(
            input=self.masks[None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].type_as(self.masks)
        resized_size = width, height
        return BinaryMaskList(resized_masks, resized_size)

    def qconvert_to_polygon(self):
        if self.masks.numel() == 0:
            return PolygonList([], self.size)

        contours = self._findContours()
        return PolygonList(contours, self.size)

    def to(self, *args, **kwargs):
        return self

    def _findContours(self):
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, hierarchy = cv2_util.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )

            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert (
                    entity.shape[1] == 1
                ), "Hierarchical contours are not allowed"
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        if self.masks.numel() == 0:
            raise RuntimeError("Indexing empty BinaryMaskList")
        return BinaryMaskList(self.masks[index], self.size)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class DepthMap(object):

    """
    This class stores the segmentations for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode="mask"):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(size[0], torch.Tensor):
            assert isinstance(size[1], torch.Tensor)
            size = size[0].item(), size[1].item()

        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))


        if mode == "mask":
            self.instances = BinaryMaskList(instances, size)
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        self.mode = mode
        self.size = tuple(size)

    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return DepthMap(flipped_instances, self.size, self.mode)

    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return DepthMap(cropped_instances, cropped_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return DepthMap(resized_instances, resized_size, self.mode)

    def to(self, *args, **kwargs):
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self

        if mode == "poly":
            converted_instances = self.instances.convert_to_polygon()
        elif mode == "mask":
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        return DepthMap(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == "poly":
            instances = instances.convert_to_binarymask()
        # If there is only 1 instance
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return DepthMap(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_segmentation = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_segmentation
        raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
