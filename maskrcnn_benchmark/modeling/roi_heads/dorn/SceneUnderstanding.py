import torch.nn as nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.poolers import Pooler

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
    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.h = h // kernel_size + 1
        self.w = w // kernel_size + 1
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        # self.upsample = nn.UpsamplingBilinear2d(size=(25, 34))  # KITTI 49X65 NYU 33X45

        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * self.h * self.w)
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        # out = self.upsample(x5)
        return x5

class SceneUnderstandingModule(nn.Module):
    def __init__(self, cfg, in_channels=2048, out_channels=512, ord_num=10, size=(14,14), kernel_size=16, pyramid=[4, 8, 12], batch_norm=False):
        super(SceneUnderstandingModule, self).__init__()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.size = size
        h, w = self.size
        # input_size = in_channels
        self.pooler = pooler

        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size)
        self.aspp1 = nn.Sequential(
            self.conv_bn_relu(batch_norm, in_channels, out_channels, kernel_size=1, padding=0),
            self.conv_bn_relu(batch_norm, out_channels, out_channels, kernel_size=1, padding=0)
        )
        self.aspp2 = nn.Sequential(
            self.conv_bn_relu(batch_norm, in_channels, out_channels, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]),
            self.conv_bn_relu(batch_norm, out_channels, out_channels, kernel_size=1, padding=0)
        )
        self.aspp3 = nn.Sequential(
            self.conv_bn_relu(batch_norm, in_channels, out_channels, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]),
            self.conv_bn_relu(batch_norm, out_channels, out_channels, kernel_size=1, padding=0)
        )
        self.aspp4 = nn.Sequential(
            self.conv_bn_relu(batch_norm, in_channels, out_channels, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]),
            self.conv_bn_relu(batch_norm, out_channels, out_channels, kernel_size=1, padding=0)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            self.conv_bn_relu(batch_norm, out_channels*5, in_channels, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels, ord_num * 2, 1)
        )

        # weights_init(self.modules(), type='xavier')

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        N, C, H, W = x.shape

        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        out = F.interpolate(out, size=self.size, mode="bilinear", align_corners=True)
        return out

    def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
                nn.ReLU(inplace=True),
            )

