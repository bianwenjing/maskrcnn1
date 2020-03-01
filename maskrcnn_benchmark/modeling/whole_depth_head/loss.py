import torch
import torch.nn.functional as F
import numpy as np
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

class ORIG_LOSS(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, pred, targets, images):
        depth_targets = []
        w = pred.shape[1]
        h = pred.shape[2]
        for box in targets:
            box = box.get_field("depth").resize((h, w)).get_mask_tensor()
            if len(box.shape)==3:
                box = torch.squeeze(box[0])
            # print('$$$$$$$$$$$$$$$$', box.shape)
            depth_targets.append(box)

        depth_targets_tensor = depth_targets[0]

        depth_targets.pop(0)
        for depth_target in depth_targets:
            depth_targets_tensor = torch.stack((depth_targets_tensor, depth_target))
        depth_targets_tensor = depth_targets_tensor.cuda().float()
        #################only positive depth as target#################
        valid_mask = (depth_targets_tensor > 0).detach()
        depth_targets_tensor = depth_targets_tensor[valid_mask]
        pred = pred[valid_mask]
        ##################################
        if self.model_name == 'berhu':
            loss = self.berhu(pred, depth_targets_tensor)
        elif self.model_name == 'adaptive':
            loss = self.adaptive_loss((pred, depth_targets_tensor, images))
        else:
            loss = F.mse_loss(pred, depth_targets_tensor)
        return loss

    def berhu(self, pred, target):
        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        diff = (target - pred).abs()

        huber_mask = (diff > huber_c).detach()
        huber_mask_inverse = (diff <= huber_c).detach()
        diff = diff[huber_mask_inverse]
        diff2 = (diff[huber_mask]**2 + huber_c)/ (2*huber_c)

        loss = torch.cat((diff, diff2)).mean()

        return loss

    def adaptive_loss(self, pred, target, image):
        p1 = 1
        p2 = 0.1

        loss = p1 * self.berhu(pred, target) + p2 * self.grad_loss(pred, image)

    def grad_loss(self, pred, image):
        img_grad = self.gradient(image)
        pred_gradient = self.gradient(pred)
        return torch.from_numpy(pred_gradient * np.exp(img_grad))

    def gradient(self, img_gray):
        gx = np.gradient(img_gray, axis=0)
        gy = np.gradient(img_gray, axis=1)
        g = gx * gx + gy * gy
        return np.sqrt(g)

class MONO_LOSS(object):
    def __call__(self, pred, targets, image):
        depth_targets = []
        w = pred.shape[1]
        h = pred.shape[2]
        for box in targets:
            box = box.get_field("depth").resize((h, w)).get_mask_tensor()
            if len(box.shape)==3:
                box = torch.squeeze(box[0])
            # print('$$$$$$$$$$$$$$$$', box.shape)
            depth_targets.append(box)

        depth_targets_tensor = depth_targets[0]

        depth_targets.pop(0)
        for depth_target in depth_targets:
            depth_targets_tensor = torch.stack((depth_targets_tensor, depth_target))
        depth_targets_tensor = depth_targets_tensor.cuda().float()
        #################only positive depth as target#################
        valid_mask = (depth_targets_tensor > 0).detach()
        depth_targets_tensor = depth_targets_tensor[valid_mask]
        pred = pred[valid_mask]
        ###################################
        loss = self.adaptive_loss(pred, depth_targets_tensor, image)
        return loss

    def adaptive_loss(self, pred, target, image):
        p1 = 1
        p2 = 0.1

        loss = p1 * self.berhu(pred, target) + p2 * self.grad_loss(pred, image)

    def grad_loss(self, pred, image):
        img_grad = self.gradient(image)
        pred_gradient = self.gradient(pred)
        return torch.from_numpy(pred_gradient * np.exp(img_grad))

    def gradient(self, img_gray):
        gx = np.gradient(img_gray, axis=0)
        gy = np.gradient(img_gray, axis=1)
        g = gx * gx + gy * gy
        return np.sqrt(g)

    def berhu(self, pred, target):
        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        diff = (target - pred).abs()

        huber_mask = (diff > huber_c).detach()
        huber_mask_inverse = (diff <= huber_c).detach()
        diff = diff[huber_mask_inverse]
        diff2 = (diff[huber_mask]**2 + huber_c)/ (2*huber_c)

        loss = torch.cat((diff, diff2)).mean()

        return loss


def make_whole_depth_loss_evaluator(cfg):
    # loss_type = cfg.MODEL.WHOLE_DEPTH.LOSS
    model_option = cfg.MODEL.WHOLE_DEPTH.MODEL_OPTION
    if model_option == 'ORIG':
        loss_evaluator = ORIG_LOSS(cfg.MODEL.WHOLE_DEPTH.LOSS)
    elif model_option == 'DORN':
        loss_evaluator = DORN_LOSS()
    return loss_evaluator