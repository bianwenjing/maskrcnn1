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

class ORIG_LOSS(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, pred, targets, images=None):
        depth_targets = []
        w = pred.shape[1]
        h = pred.shape[2]
        for box in targets:
            box = box.get_field("depth").resize((h, w)).get_mask_tensor()
            if len(box.shape)==3:
                box = torch.squeeze(box[0])
            depth_targets.append(box)

        depth_targets_tensor = depth_targets[0]

        depth_targets.pop(0)
        for depth_target in depth_targets:
            depth_targets_tensor = torch.stack((depth_targets_tensor, depth_target))
        depth_targets_tensor = depth_targets_tensor.cuda().float()
        #################only positive depth as target#################
        valid_mask = (depth_targets_tensor > 0).detach()
        depth_targets_tensor_vector = depth_targets_tensor[valid_mask]
        pred_vector = pred[valid_mask]
        ##################################
        # pred_vector = pred_vector + 0.2
        depth_targets_tensor_vector = self.depth_normalise(depth_targets_tensor_vector)
        if self.model_name == 'berhu':
            loss = self.berhu(pred_vector, depth_targets_tensor_vector)
        elif self.model_name == 'adaptive':
            ####################resize image and change to grayscale#############################
            images = images.cpu().detach()
            pred = pred.cpu().detach()
            tensor_to_pil = trans.ToPILImage()
            pil_to_tensor = trans.ToTensor()
            pred_size = (pred.shape[2], pred.shape[1])
            images_resized = tensor_to_pil(images[0]).resize(pred_size).convert('L')
            images_resized = pil_to_tensor(images_resized)[0]  # (1, 50, 68) -> (50, 68)
            batch = images.shape[0]
            for i in range(1, batch):
                image = tensor_to_pil(images[i]).resize(pred_size).convert('L')
                image = pil_to_tensor(image)[0]
                images = torch.stack((images_resized, image))
            ###########################################################################
            loss = self.adaptive_loss(pred_vector, depth_targets_tensor_vector, pred, images)
        elif self.model_name == 'scale_invariant':
            loss = self.sale_invariant(pred_vector, depth_targets_tensor_vector)
        else:
            loss = F.mse_loss(pred_vector, depth_targets_tensor_vector)
        return loss

    def compute_valid_depth_mask(d1, d2=None):
        """Computes the mask of valid values for one or two depth maps

        Returns a valid mask that only selects values that are valid depth value
        in both depth maps (if d2 is given).
        Valid depth values are >0 and finite.
        """
        if d2 is None:
            valid_mask = np.isfinite(d1)
            valid_mask[valid_mask] = (d1[valid_mask] > 0)
        else:
            valid_mask = np.isfinite(d1) & np.isfinite(d2)
            valid_mask[valid_mask] = (d1[valid_mask] > 0) & (d2[valid_mask] > 0)
        return valid_mask


    def berhu(self, pred, target):
        huber_c = torch.max(pred - target).abs()
        huber_c = 0.2 * huber_c

        diff = (target - pred).abs()

        huber_mask = (diff > huber_c).detach()
        huber_mask_inverse = (diff <= huber_c).detach()
        diff1 = diff[huber_mask_inverse]
        diff2 = (diff[huber_mask]**2 + huber_c)/ (2*huber_c)

        loss = torch.cat((diff1, diff2)).mean()
        return loss

    def adaptive_loss(self, pred_vector, target_vector, pred, images):
        p1 = 1
        p2 = 0.1
        loss = p1 * self.berhu(pred_vector, target_vector) + p2 * self.grad_loss(pred, images)
        return loss

    def grad_loss(self, pred, image):
        img_grad = self.gradient(image)
        pred_gradient = self.gradient(pred)
        loss = np.sum(np.absolute(pred_gradient * np.exp(-np.square(img_grad))))
        return loss

    def gradient(self, img_gray):
        gx = np.gradient(img_gray, axis=1)
        gy = np.gradient(img_gray, axis=2)
        g = gx * gx + gy * gy
        return np.sqrt(g)

    def sale_invariant(self, pred, target):
        assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
        log_diff = np.log(depth1) - np.log(depth2)
        num_pixels = float(log_diff.size)

        loss = np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))
        return loss
    def depth_normalise(self, target):
        # max = torch.max(torch.max(pred), torch.max(target))
        max = 10000
        target = target/max
        return target





def make_whole_depth_loss_evaluator(cfg):
    # loss_type = cfg.MODEL.WHOLE_DEPTH.LOSS
    model_option = cfg.MODEL.WHOLE_DEPTH.MODEL_OPTION
    if model_option == 'ORIG':
        loss_evaluator = ORIG_LOSS(cfg.MODEL.WHOLE_DEPTH.LOSS)
    elif model_option == 'DORN':
        loss_evaluator = DORN_LOSS()
    return loss_evaluator