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
from torch.utils.tensorboard import SummaryWriter
import numpy as np




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
        # self.depth_estimate = torch.nn.Linear(self.backbone.out_channels * 13 * 17, 480*640)
        # self.depth_estimate = torch.nn.Linear(self.backbone.out_channels * 13 * 17, 13 * 17)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, target_depth = None):
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
        features = self.backbone(images.tensors)
        # print(len(features), list(features[4].shape))
        # features_flatten = self.flatten(features[4])
        # depth_map = self.depth_estimate(features_flatten)
        # loss_depth_model = nn.MSELoss()
        # loss_depth = loss_depth_model(depth_map, target_depth)
        # depth_loss = {'loss_depth' : loss_depth}
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses.update(depth_loss)
            # print(losses)
            # writer = SummaryWriter()
            #
            # writer.add_scalar('classifier loss', losses['loss_classifier'])
            # writer.add_scalar('box reg loss', losses['loss_box_reg'])
            # writer.add_scalar('mask loss', losses['loss_mask'])
            # writer.add_scalar('objectness loss', losses['loss_objectness'])
            # writer.add_scalar('rpn box reg loss', losses['loss_rpn_box_reg'])

            return losses
 ##############################################################################
        if self.training and target_depth is not None:
            with torch.no_grad():
                images = to_image_list(images)
                features = self.backbone(images.tensors)
                proposals, proposal_losses = self.rpn(images, features, targets)
                x, result, detector_losses = self.roi_heads(features, proposals, targets)

        return result

    def flatten(self, x):
        N = x.shape[0]  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

