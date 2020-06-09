import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as trans

class DORN_LOSS(object):
    def __call__(self, ord_labels, target):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """
        N, C, H, W = ord_labels.size()
        ord_num = C
        # print('ord_num = ', ord_num)

        self.loss = 0.0

        # for k in range(ord_num):
        #     '''
        #     p^k_(w, h) = e^y(w, h, 2k+1) / [e^(w, h, 2k) + e^(w, h, 2k+1)]
        #     '''
        #     p_k = ord_labels[:, k, :, :]
        #     p_k = p_k.view(N, 1, H, W)
        #
        #     '''
        #     对每个像素而言，
        #     如果k小于l(w, h), log(p_k)
        #     如果k大于l(w, h), log(1-p_k)
        #     希望分类正确的p_k越大越好
        #     '''
        #     mask_0 = (target >= k).detach()   # 分类正确
        #     mask_1 = (target < k).detach()  # 分类错误
        #
        #     one = torch.ones(p_k[mask_1].size())
        #     if torch.cuda.is_available():
        #         one = one.cuda()
        #     self.loss += torch.sum(torch.log(torch.clamp(p_k[mask_0], min = 1e-7, max = 1e7))) \
        #                  + torch.sum(torch.log(torch.clamp(one - p_k[mask_1], min = 1e-7, max = 1e7)))

        # faster version
        if torch.cuda.is_available():
            K = torch.zeros((N, C, H, W), dtype=torch.int).cuda()
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).cuda()
        else:
            K = torch.zeros((N, C, H, W), dtype=torch.int)
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size())
        if torch.cuda.is_available():
            one = one.cuda()

        self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

        # del K
        # del one
        # del mask_0
        # del mask_1

        N = N * H * W
        self.loss /= (-N)  # negative
        return self.loss

def make_dorn_loss_evaluator(cfg):
    loss_evaluator = DORN_LOSS()
    return loss_evaluator