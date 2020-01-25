import torch
import torch.nn.functional as F
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
    def __call__(self, pred, targets):
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
        loss = F.mse_loss(pred, depth_targets_tensor)
        return loss



def make_whole_depth_loss_evaluator(cfg):
    # loss_type = cfg.MODEL.WHOLE_DEPTH.LOSS
    model_option = cfg.MODEL.WHOLE_DEPTH.MODEL_OPTION
    if model_option == 'ORIG':
        loss_evaluator = ORIG_LOSS()
    elif model_option == 'DORN':
        loss_evaluator = DORN_LOSS()
    return loss_evaluator