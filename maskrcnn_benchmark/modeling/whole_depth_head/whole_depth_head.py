import torch
import torch.nn as nn
import math
from .loss import make_whole_depth_loss_evaluator
from .whole_depth_feature_extractor import make_whole_depth_feature_extractor
from .whole_depth_predictor import make_whole_depth_predictor
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
        self.global_fc = nn.Linear(2048 * 4 * 4, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        self.upsample = nn.UpsamplingBilinear2d(size=(25, 34))  # KITTI 49X65 NYU 33X45

        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 4 * 4)
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
            nn.UpsamplingBilinear2d(size=(385, 513))
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
        # print('ord > 0.5 size:', (ord_c1 > 0.5).size())
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1


class DORN(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DORN, self).__init__()

        self.aspp_module = SceneUnderstandingModule()
        self.orl = OrdinalRegressionLayer()
        self.loss_evaluator = make_whole_depth_loss_evaluator(cfg)

        #####################################################
        # for name, param in self.named_parameters():
        #     param.requires_grad = False
        #####################################################

    def forward(self, features, targets = None):
        x = self.aspp_module(features) # 2,1, 385, 513
        depth_labels, ord_labels = self.orl(x)
        # depth_label (2, 1, 385, 513)
        # ord_labels (2, 68, 385, 513)
        if not self.training:
            return depth_labels, {}
        ###########################################################
        depth_targets = []
        w = ord_labels.shape[2]
        h = ord_labels.shape[3]

        for box in targets:
            box = box.get_field("depth").resize((h, w)).get_mask_tensor()
            if len(box.shape) == 3:
                depth_targets.append(box[0])
                # box = torch.squeeze(box[0])
            else:
                depth_targets.append(box)

        depth_targets_tensor = depth_targets[0]
        depth_targets.pop(0)
        for depth_target in depth_targets:
            depth_targets_tensor = torch.stack((depth_targets_tensor, depth_target))
        depth_targets_tensor = depth_targets_tensor.cuda().float()
        if len(depth_targets_tensor.shape)==3:
            depth_targets_tensor = depth_targets_tensor[:, None, :, :]
        else:
            depth_targets_tensor = depth_targets_tensor[None, None, :, :]
        #####################################################
        targets_c = self.get_labels_sid(depth_targets_tensor)

        loss = self.loss_evaluator(ord_labels, targets_c)
        return depth_labels, dict(whole_depth_loss=loss)

    def get_labels_sid(args, depth):
        alpha = 0.02
        beta = 10.0
        K = 68.0

        alpha = torch.tensor(alpha)
        beta = torch.tensor(beta)
        K = torch.tensor(K)

        if torch.cuda.is_available():
            alpha = alpha.cuda()
            beta = beta.cuda()
            K = K.cuda()

        labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
        if torch.cuda.is_available():
            labels = labels.cuda()
        return labels.int()

class ORIG(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ORIG, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_whole_depth_feature_extractor(cfg, in_channels)
        self.predictor = make_whole_depth_predictor(
            cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = make_whole_depth_loss_evaluator(cfg)


    def forward(self, x, targets = None, images = None):
        # print('$$$$$$$$$$$$$', len(x), x[0].shape)  #(2, 2048, 25, 34) fpn
        if isinstance(x, list):  #for res50 x[0] is (2, 1024, 50, 67)
            x = x[0]
        x = self.feature_extractor(x)
        x = self.predictor(x)
        if x.shape[0]==1:  # batch size = 1 sometimes
            x = x[0]
        else:
            x = torch.squeeze(x)
        if not self.training:
            return x, {}
        loss = self.loss_evaluator(x, targets, images)

        return x, dict(whole_depth_loss=loss)


def whole_depth(cfg, in_channels):
    model_option = cfg.MODEL.WHOLE_DEPTH.MODEL_OPTION
    if model_option == 'DORN':
        return DORN(cfg,in_channels)
    elif model_option == 'ORIG':
        return ORIG(cfg, in_channels)

