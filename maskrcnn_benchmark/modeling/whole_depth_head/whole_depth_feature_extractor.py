from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3

registry.WHOLE_DEPTH_FEATURE_EXTRACTORS.register(
    "RES", ResNet50Conv5ROIFeatureExtractor
)


@registry.WHOLE_DEPTH_FEATURE_EXTRACTORS.register("FPN")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()
        input_size = in_channels

        use_gn = cfg.MODEL.WHOLE_DEPTH.USE_GN
        layers = cfg.MODEL.WHOLE_DEPTH.CONV_LAYERS
        dilation = cfg.MODEL.WHOLE_DEPTH.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "depth_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn, use_relu=True
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features
##############################################################################################

    def forward(self, x):

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x

def make_whole_depth_feature_extractor(cfg, in_channels):
    func = registry.WHOLE_DEPTH_FEATURE_EXTRACTORS[
        cfg.MODEL.WHOLE_DEPTH.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)