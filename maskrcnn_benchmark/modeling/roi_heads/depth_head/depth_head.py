import torch
from torch import nn

from .make_roi_depth_feature_extractor import make_roi_depth_feature_extractor

from .roi_depth_predictors import make_roi_depth_predictor
from .inference import make_roi_depth_post_processor
from .loss import make_roi_depth_loss_evaluator
from maskrcnn_benchmark.structures.bounding_box import BoxList

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIDepthHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIDepthHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_depth_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_depth_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_depth_post_processor(cfg)
        self.loss_evaluator = make_roi_depth_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        depth_logits = self.predictor(x)
        # depth_logits shape (# of proposals, 20 classes, 28, 28)

        if not self.training:
            result = self.post_processor(depth_logits, proposals)
            return x, result, {}

        loss_depth = self.loss_evaluator(proposals, depth_logits, targets)

        return x, all_proposals, dict(loss_depth=loss_depth)


def build_roi_depth_head(cfg, in_channels):
    return ROIDepthHead(cfg, in_channels)
