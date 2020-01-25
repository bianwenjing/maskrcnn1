import torch
import torch.nn as nn
import math
from .loss import make_whole_depth_loss_evaluator
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from torch.nn import functional as F
from maskrcnn_benchmark.layers import UpProjModule
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers import Conv2d
import numpy as np

def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()

class FullImageEncoder(nn.Module):
    def __init__(self):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(8, stride=8, padding=(4, 2))  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 6 * 5, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        self.upsample = nn.UpsamplingBilinear2d(size=(33, 45))  # KITTI 49X65 NYU 33X45

        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        print('!!!!!!!!!!!!!!!!!!!!', x.shape)
        x1 = self.global_pooling(x)
        print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 4 * 4)
        print('!!!!!!!!!!!!!!!!!!!!!!!!', x3.shape)
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out

class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        self.encoder = FullImageEncoder()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 136, 1),  # KITTI 142 NYU 136 In paper, K = 80 is best, so use 160 is good!
            # nn.UpsamplingBilinear2d(scale_factor=8)
            nn.UpsamplingBilinear2d(size=(257, 353))
        )

        weights_init(self.modules(), type='xavier')

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out

class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        print('ord > 0.5 size:', (ord_c1 > 0.5).size())
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1


class DORN(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DORN, self).__init__()

        self.aspp_module = SceneUnderstandingModule()
        self.orl = OrdinalRegressionLayer()
        self.loss_evaluator = make_whole_depth_loss_evaluator()

    def forward(self, features, targets = None):
        x = self.aspp_module(features)
        depth_labels, ord_labels = self.orl(x)
        targets_c = self.get_labels_sid(targets)
        if not self.training:
            return depth_labels, {}
        loss = make_whole_depth_loss_evaluator(ord_labels, targets_c)
        return depth_labels, dict(whole_depth_loss=loss)

    def get_labels_sid(self, targets):
        min = 0.02
        max = 80.0
        K = 68.0

        if torch.cuda.is_available():
            alpha_ = torch.tensor(min).cuda()
            beta_ = torch.tensor(max).cuda()
            K_ = torch.tensor(K).cuda()
        else:
            alpha_ = torch.tensor(min)
            beta_ = torch.tensor(max)
            K_ = torch.tensor(K)
        depth = alpha_ * (beta_ / alpha_) ** (targets / K_)
        # print(depth.size())
        return depth.float()

class ORIG(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ORIG, self).__init__()

        # resolution = 28
        # scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        # sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        input_size = in_channels
        # self.pooler = pooler

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
        features_out_channels = layer_features
##############################################################################################################################
##predictor
########################################################################################################################
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes = 1
        dim_reduced = cfg.MODEL.WHOLE_DEPTH.CONV_LAYERS[-1]
        num_inputs = features_out_channels

        self.conv5_depth = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        # self.conv5_depth = UpProjModule(num_inputs, dim_reduced)
        # self.conv6_depth = UpProjModule(dim_reduced, dim_reduced)
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
        ###############################################################################
        self.loss_evaluator = make_whole_depth_loss_evaluator(cfg)


    def forward(self, x, targets = None):
        # x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        ###########################################
        x = self.conv5_depth(x)
        self.depth = self.conv5_depth(x)
        x = self.depth
        x = self.depth_fcn_logits(x)
        # x=self.conv6_depth(x)
        if x.shape[0]==1:  # batch size = 1 sometimes
            x = x[0]
        else:
            x = torch.squeeze(x)
        if not self.training:
            return x, {}
        loss = self.loss_evaluator(x, targets)

        return x, dict(whole_depth_loss=loss)



def whole_depth(cfg, in_channels):
    model_option = cfg.MODEL.WHOLE_DEPTH.MODEL_OPTION
    if model_option == 'DORN':
        return DORN(cfg,in_channels)
    elif model_option == 'ORIG':
        return ORIG(cfg, in_channels)
