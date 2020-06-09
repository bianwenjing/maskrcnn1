import torch
from torch.nn import functional as F


from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np
import torchvision.transforms as trans
from maskrcnn_benchmark.structures.depth_map import DepthMap

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, model_name, adaptive):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.model_name = model_name
        self.adaptive = adaptive

    def match_targets_to_proposals2(self, proposal, target, mask_result):
        match_quality_matrix = boxlist_iou(mask_result, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        mask_result = mask_result.copy_with_fields(["labels", "mask"])
        # print('&&&&&&&&&&&&&',mask_result.get_field("mask").shape)  #(68,1,14,14)
        # mask_num = len(mask_result.get_field("mask"))
        depth = target.get_field("depth")
        # depth_ls = []
        # for i in range(mask_num):
        #     if len(depth.shape) == 3:
        #         depth_ls.append(depth[0])
        #     else:
        #         depth_ls.append(depth)
        # depth = DepthMap(depth, self.img_size, mode='mask')
        mask_result.add_field("depth", depth)
        if self.model_name == 'adaptive':
            mask_result.add_field("gray_img", target.get_field("gray_img"))
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = mask_result[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets2(self, proposals, targets, mask_result):
        labels = []
        maps = []
        masks=[]
        masked_images = []
        for proposals_per_image, targets_per_image, mask_result_per_image in zip(proposals, targets, mask_result):
            matched_targets = self.match_targets_to_proposals2(
                proposals_per_image, targets_per_image, mask_result_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            depth_maps = matched_targets.get_field("depth")
            depth_maps = depth_maps[positive_inds]

            segmentation_masks = matched_targets.get_field("mask")
            segmentation_masks = segmentation_masks[positive_inds]

            # focal_i = matched_targets.get_field("intrinsic")

            positive_proposals = proposals_per_image[positive_inds]
            maps_per_image = project_masks_on_boxes(
                depth_maps, positive_proposals, self.discretization_size
            )
            # masks_per_image = project_masks_on_boxes(
            #     segmentation_masks, positive_proposals, self.discretization_size
            # )
            masks_per_image = torch.zeros((segmentation_masks.shape))
            masks_per_image[segmentation_masks > 0.5] = 1
            # print('%%%%%%%%%%%%%%', masks_per_image.shape)
            masks_per_image = torch.squeeze(masks_per_image).cuda()
            if len(masks_per_image.shape)==2:
                masks_per_image = masks_per_image.unsqueeze(0)
            # maps_per_image=maps_per_image * masks_per_image # apply segmentation mask to extract object from background


            labels.append(labels_per_image)
            maps.append(maps_per_image)
            masks.append(masks_per_image)

            if self.model_name == 'adaptive':
                imgs = matched_targets.get_field("gray_img")
                imgs = imgs[positive_inds]
                masked_images_per_image = project_masks_on_boxes(imgs, positive_proposals, self.discretization_size)
                masked_images.append(masked_images_per_image)

        # return labels, maps, masks, focal_i[0]
        return labels, maps, masks, masked_images

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        # target = target.copy_with_fields(["labels", "depth", "masks", "intrinsic"])

        if self.model_name == 'adaptive':
            target = target.copy_with_fields(["labels", "depth", "masks", "gray_img"])
        else:
            target = target.copy_with_fields(["labels", "depth", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        maps = []
        masks=[]
        images = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            depth_maps = matched_targets.get_field("depth")
            depth_maps = depth_maps[positive_inds]

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            # focal_i = matched_targets.get_field("intrinsic")

            positive_proposals = proposals_per_image[positive_inds]

            maps_per_image = project_masks_on_boxes(
                depth_maps, positive_proposals, self.discretization_size
            )
            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            # maps_per_image=maps_per_image * masks_per_image # apply segmentation mask to extract object from background


            labels.append(labels_per_image)
            maps.append(maps_per_image)
            masks.append(masks_per_image)

            if self.model_name == 'adaptive':
                imgs = matched_targets.get_field("gray_img")
                imgs = imgs[positive_inds]
                masked_images_per_image = project_masks_on_boxes(imgs, positive_proposals, self.discretization_size)
                images.append(masked_images_per_image)

        # return labels, maps, masks, focal_i[0]
        return labels, maps, masks, images

    def __call__(self, proposals, depth_logits, targets, mask_result, decouple):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        if decouple == 'independent' or decouple == 'coupled':
            labels, depth_targets, mask_targets, images = self.prepare_targets(proposals, targets)
        else:
            labels, depth_targets, mask_targets, masked_images = self.prepare_targets2(proposals, targets, mask_result)
         # len(depth_targets) batch size

        labels = cat(labels, dim=0).detach()
        depth_targets = cat(depth_targets, dim=0).detach()
        mask_targets = cat(mask_targets, dim=0).detach()

        positive_inds = torch.nonzero(labels > 0).squeeze(1).detach()
        labels_pos = labels[positive_inds].detach()
        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if depth_targets.numel() == 0:
            return depth_logits.sum() * 0

        ######################################################

        depth_pred = depth_logits[positive_inds, labels_pos] * mask_targets.detach()
        depth_targets = depth_targets * mask_targets.detach()
        valid_mask = (depth_targets > 0).detach()
        depth_targets_vector = depth_targets[valid_mask].detach()
        depth_pred_vector = depth_pred[valid_mask]
        depth_targets_vector = self.depth_normalise(depth_targets_vector)

        if self.model_name == 'MSE':
            depth_loss = F.mse_loss(depth_pred_vector, depth_targets_vector)
        elif self.model_name == 'berhu':
            depth_loss = self.berhu(depth_pred_vector, depth_targets_vector)
        elif self.model_name == 'scale':
            depth_loss = self.scale_invariant(depth_pred_vector, depth_targets_vector)
        elif self.model_name == 'adaptive':
            ####################resize image and change to grayscale#############################
            depth_pred_nomask = depth_logits[positive_inds, labels_pos].detach()
            images = cat(images, dim=0)
            # masked_images = images * mask_targets
            # masked_images = masked_images.cpu().detach()
            # depth_pred = depth_pred.cpu().detach()
            ###########################################################################
            if self.adaptive == 'L1':
                depth_loss = self.adaptive_loss(depth_pred_vector, depth_targets_vector, depth_pred_nomask, images, mask_targets)

            elif self.adaptive == 'L2':
                depth_loss = self.adaptive_loss2(depth_pred_vector, depth_targets_vector, depth_pred_nomask, images,
                                                mask_targets)
        return depth_loss

    def berhu(self, pred, target):
        huber_c = torch.max(pred - target).abs()
        huber_c = 0.2 * huber_c

        diff = (target - pred).abs()

        huber_mask = (diff > huber_c).detach()
        huber_mask_inverse = (diff <= huber_c).detach()
        diff1 = diff[huber_mask_inverse]
        diff2 = (diff[huber_mask] ** 2 + huber_c**2) / (2 * huber_c)

        loss = torch.cat((diff1, diff2)).mean()
        # print('^^^^^^^^^^', loss)
        return loss

    def adaptive_loss(self, pred_vector, target_vector, pred, images, masks):
        p1 = 1
        p2 = 0.001
        loss = p1 * self.berhu(pred_vector, target_vector) + p2 * self.grad_loss(pred, images, masks)
        return loss
    def adaptive_loss2(self, pred_vector, target_vector, pred, images, masks):
        p1 = 10
        p2 = 0.00005
        # print('%%%%%%%', p1*F.mse_loss(pred_vector, target_vector))
        loss = p1 * F.mse_loss(pred_vector, target_vector) + p2 * self.grad_loss(pred, images, masks)
        return loss

    def grad_loss(self, pred, image, masks):
        img_grad = self.gradient(image) * masks
        pred_gradient = self.gradient(pred) * masks
        # loss = np.sum(np.absolute(pred_gradient * np.exp(-np.square(img_grad))))
        loss = torch.sum(torch.abs(pred_gradient*torch.exp(-torch.pow(img_grad, 2))))
        # print('$$$', loss)
        return loss

    def gradient(self, img_gray):
        # gx = np.gradient(img_gray.cpu().detach().numpy(), axis=1)
        # gx = torch.from_numpy(gx*masks)
        # gy = np.gradient(img_gray.cpu().detach().numpy(), axis=2) * masks
        # gy = torch.from_numpy(gy)
        gx = torch.from_numpy(np.gradient(img_gray.cpu().detach().numpy(), axis=1)).to(torch.device("cuda"))
        gy = torch.from_numpy(np.gradient(img_gray.cpu().detach().numpy(), axis=2)).to(torch.device("cuda"))
        g = gx * gx + gy * gy
        return torch.sqrt(g)

    def depth_normalise(self,target):
        # max = torch.max(torch.max(pred), torch.max(target))
        max = 10000
        target = target/max
        return target

    def scale_invariant(self, pred, target):
        # assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
        log_diff = torch.log(pred) - torch.log(target)
        num_pixels = len(log_diff)

        loss = torch.sqrt(torch.sum(torch.square(log_diff)) / num_pixels - torch.square(torch.sum(log_diff)) / torch.square(num_pixels))
        return loss



def make_roi_depth_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_DEPTH_HEAD.RESOLUTION, cfg.MODEL.ROI_DEPTH_HEAD.LOSS, cfg.MODEL.ROI_DEPTH_HEAD.ADA
    )

    return loss_evaluator

