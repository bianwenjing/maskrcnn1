from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry

from maskrcnn_benchmark.layers import UpProjModule


@registry.ROI_DEPTH_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_DEPTH_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        # self.conv5_depth = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.conv5_depth = UpProjModule(num_inputs, dim_reduced)
        self.depth_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            # elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

            elif "batchnorm" in name:
                nn.init.constant_(param, 1)
            else:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
###################################################################
# from FCRN
#     def weights_init(m):
#         # Initialize filters with Gaussian random weights
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.ConvTranspose2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#######################################################################
    def forward(self, x):
        # x = F.relu(self.conv5_depth(x))
        x = self.conv5_depth(x)
        return self.depth_fcn_logits(x)

def make_roi_depth_predictor(cfg, in_channels):
    func = registry.ROI_DEPTH_PREDICTOR[cfg.MODEL.ROI_DEPTH_HEAD.PREDICTOR]
    return func(cfg, in_channels)
