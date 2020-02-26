# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
from ..whole_depth_head.whole_depth_head import whole_depth




class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.whole_depth_on = cfg.MODEL.WHOLE_DEPTH_ON
        self.FPN_RES = cfg.MODEL.BACKBONE.CONV_BODY
        ###################################################################################
        if self.whole_depth_on:
            self.whole_depth = whole_depth(cfg, 2048)
        #####################################################################################
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.FPN_RES == "R-50-FPN":
            features, resnet_output = self.backbone(images.tensors)
        else:
            features = self.backbone(images.tensors)
            resnet_output = features
        # print('!!!!!!!!!!!!!!!!!', resnet_output.shape) (2,2048,25,34)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        if self.whole_depth_on:
            x_depth, whole_depth_loss = self.whole_depth(resnet_output, targets)
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.whole_depth_on:
                losses.update(whole_depth_loss)

            return losses

        if self.whole_depth_on:
            return result, x_depth
        return result, 0

    # def flatten(self, x):
    #     N = x.shape[0]  # read in N, C, H, W
    #     return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

