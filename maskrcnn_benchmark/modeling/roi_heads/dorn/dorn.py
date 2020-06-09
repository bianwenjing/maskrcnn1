import torch
from torch import nn
from .Ordinal import OrdinalRegressionLayer
from .SceneUnderstanding import SceneUnderstandingModule
from .loss import make_dorn_loss_evaluator
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.depth_head.inference import make_roi_depth_post_processor

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels") # positive category
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0   #remove label = 0 (background)
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class DORN(nn.Module):
    def __init__(self, cfg, in_channels, ord_num=90, gamma=1.0, beta=80.0,
                 input_size=(385, 513), kernel_size=16, pyramid=[4, 8, 12],
                 batch_norm=False,
                 discretization="SID", pretrained=True):
        super(DORN, self).__init__()
        self.cfg = cfg.clone()
        self.ord_num = ord_num
        out_channels = 256
        self.aspp_module = SceneUnderstandingModule(cfg, in_channels,out_channels, ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid,
                                                                 batch_norm=batch_norm)
        self.orl = OrdinalRegressionLayer()
        self.loss_evaluator = make_whole_depth_loss_evaluator(cfg)
        # self.feature_extractor = make_roi_depth_feature_extractor(cfg, in_channels)
        # self.predictor = make_roi_depth_predictor(
        #     cfg, self.feature_extractor.out_channels)
        # self.post_processor = make_roi_depth_post_processor(cfg)
        # self.loss_evaluator = make_roi_depth_loss_evaluator(cfg)

        #####################################################
        # for name, param in self.named_parameters():
        #     param.requires_grad = False
        #####################################################

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
            ##############################
            if self.decouple == 'independent':
                proposals = targets
            ##############################
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        x = self.aspp_module(features, proposals)
        depth_labels, ord_labels = self.orl(x)

        if not self.training:
            depth_logits = depth_logits*10000
            result = self.post_processor(depth_logits, proposals)
            return x, result, {}

        loss_depth = self.loss_evaluator(proposals, depth_logits, targets)
        # loss_depth = self.loss_evaluator(targets, depth_logits, targets)
        return x, all_proposals, dict(loss_depth=loss_depth)


def build_dorn_head(cfg, in_channels):
    return ROIDepthHead(cfg, in_channels)
