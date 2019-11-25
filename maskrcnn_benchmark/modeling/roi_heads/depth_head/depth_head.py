import torch
from torch import nn
from .make_roi_depth_feature_extractor import make_roi_depth_feature_extractor
class ROIDepthHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIDepthHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_depth_feature_extractor(cfg, in_channels)

def build_roi_depth_head(cfg, in_channels):
    return ROIDepthHead(cfg, in_channels)
